"""Bounded proposer-prior planning over learned operator dynamics.

The planner searches only neural latent states.  It contains no repository,
permutation, graph, or other domain transition rules, and it never executes a
returned operator sequence.  Forward and backward effects come from the model;
the caller remains responsible for committing and independently verifying any
proposed procedure in a real environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import itertools
import math
import struct
from typing import Any, Protocol, Sequence, runtime_checkable

import torch
from torch import nn
from torch.nn import functional as F


@runtime_checkable
class OperatorPlanningModel(Protocol):
    """Tensor boundary required by :class:`BidirectionalOperatorPlanner`."""

    width: int

    def initiation_logits(
        self,
        states: torch.Tensor,
        operators: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor: ...

    def predict_effects(
        self,
        states: torch.Tensor,
        operators: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor: ...

    def termination_logits(
        self,
        candidate_states: torch.Tensor,
        goals: torch.Tensor,
    ) -> torch.Tensor: ...

    def proposer_logits(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        operators: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class PrimitiveChoice:
    """One decoded but unexecuted primitive action/argument proposal."""

    action_identity: str
    arguments: tuple[str, ...]
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_identity, str) or not self.action_identity:
            raise ValueError("action_identity must be a non-empty string")
        if not isinstance(self.arguments, tuple) or any(
            not isinstance(argument, str) for argument in self.arguments
        ):
            raise TypeError("primitive arguments must be an immutable string tuple")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("primitive score must be finite when present")


@runtime_checkable
class PrimitiveChoiceDecoder(Protocol):
    """Optional binding boundary from an operator to executable primitives.

    The callback may use ``PrimitiveDecoder`` and public action schemas.  It
    returns proposals only; neither it nor this planner invokes an action.
    Returning an empty sequence rejects that neural expansion as unbound.
    """

    def decode(
        self,
        *,
        operator_index: int,
        operator_identity: str,
        source_state: torch.Tensor,
        predicted_state: torch.Tensor,
        goal: torch.Tensor,
        reverse_search: bool,
    ) -> Sequence[PrimitiveChoice]: ...


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Hard, predeclared limits and neural decision thresholds."""

    maximum_expansions: int
    maximum_depth: int
    proposals_per_state: int
    join_tolerance: float = 1e-4
    deduplication_tolerance: float = 1e-6
    minimum_initiation_probability: float = 0.0
    termination_probability: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "maximum_expansions",
            "maximum_depth",
            "proposals_per_state",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("join_tolerance", "deduplication_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "minimum_initiation_probability",
            "termination_probability",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class SearchAccounting:
    """Auditable neural work consumed by one plan attempt."""

    forward_expansions: int
    backward_expansions: int
    proposer_calls: int
    operator_scores: int
    initiation_rejections: int
    binding_rejections: int
    duplicate_states: int

    @property
    def total_expansions(self) -> int:
        return self.forward_expansions + self.backward_expansions


@dataclass(frozen=True, slots=True)
class OperatorPlan:
    """One unexecuted neural proposal plus complete search accounting."""

    found: bool
    operator_indices: tuple[int, ...]
    operator_identities: tuple[str, ...]
    primitive_choices: tuple[tuple[PrimitiveChoice, ...], ...] | None
    forward_depth: int
    backward_depth: int
    meeting_distance: float | None
    predicted_terminal: bool
    reason: str
    accounting: SearchAccounting

    @property
    def is_primitive_bound(self) -> bool:
        return self.found and self.primitive_choices is not None


@dataclass(frozen=True, slots=True)
class _Step:
    operator_index: int
    operator_identity: str
    primitive_choices: tuple[PrimitiveChoice, ...] | None


@dataclass(slots=True)
class _Node:
    state: torch.Tensor
    path: tuple[_Step, ...]
    score: float


class BidirectionalOperatorPlanner:
    """Generic best-first bidirectional search prioritized by a neural proposer."""

    def __init__(self, model: OperatorPlanningModel) -> None:
        self.model = model

    def plan(
        self,
        origin: Any,
        goal: Any,
        operators: Sequence[Any],
        budget: SearchBudget,
        *,
        primitive_decoder: PrimitiveChoiceDecoder | None = None,
    ) -> OperatorPlan:
        """Encode public contracts and return a proposal without execution."""

        if not operators:
            raise ValueError("at least one operator is required")
        if getattr(goal, "exact", None) is not True:
            raise ValueError(
                "bidirectional planning requires an exact Goal or a learned "
                "goal-completion model"
            )
        for method in (
            "encode_states",
            "encode_goals",
            "encode_goal_states",
            "encode_operators",
        ):
            if not hasattr(self.model, method):
                raise TypeError(
                    "high-level planning requires model state/goal/operator encoders"
                )
        state = self.model.encode_states((origin,))  # type: ignore[attr-defined]
        goal_condition = self.model.encode_goals((goal,))  # type: ignore[attr-defined]
        goal_state = self.model.encode_goal_states((goal,))  # type: ignore[attr-defined]
        operator_embeddings = self.model.encode_operators(operators)  # type: ignore[attr-defined]
        identities = tuple(_operator_identity(operator) for operator in operators)
        return self.plan_embeddings(
            state[0],
            goal_state[0],
            operator_embeddings,
            budget,
            goal_condition=goal_condition[0],
            operator_identities=identities,
            primitive_decoder=primitive_decoder,
        )

    @torch.no_grad()
    def plan_embeddings(
        self,
        origin: torch.Tensor,
        goal_state: torch.Tensor,
        operator_embeddings: torch.Tensor,
        budget: SearchBudget,
        *,
        goal_condition: torch.Tensor | None = None,
        operator_identities: Sequence[str] | None = None,
        primitive_decoder: PrimitiveChoiceDecoder | None = None,
    ) -> OperatorPlan:
        """Search learned state geometry under an exact expansion ceiling."""

        if goal_condition is None:
            goal_condition = goal_state
        if operator_identities is None:
            operator_identities = tuple(
                _embedding_identity(row) for row in operator_embeddings
            )
        identities = tuple(operator_identities)
        self._validate_inputs(
            origin,
            goal_state,
            goal_condition,
            operator_embeddings,
            identities,
            budget,
        )
        was_training: bool | None = None
        if isinstance(self.model, nn.Module):
            was_training = self.model.training
            self.model.eval()
        try:
            return self._search(
                origin,
                goal_state,
                goal_condition,
                operator_embeddings,
                identities,
                budget,
                primitive_decoder,
            )
        finally:
            if isinstance(self.model, nn.Module) and was_training is not None:
                self.model.train(was_training)

    def _search(
        self,
        origin: torch.Tensor,
        goal_state: torch.Tensor,
        goal_condition: torch.Tensor,
        operators: torch.Tensor,
        operator_identities: tuple[str, ...],
        budget: SearchBudget,
        primitive_decoder: PrimitiveChoiceDecoder | None,
    ) -> OperatorPlan:
        initial_distance = _distance(origin, goal_state)
        if initial_distance <= budget.join_tolerance:
            return OperatorPlan(
                found=True,
                operator_indices=(),
                operator_identities=(),
                primitive_choices=() if primitive_decoder is not None else None,
                forward_depth=0,
                backward_depth=0,
                meeting_distance=initial_distance,
                predicted_terminal=False,
                reason="origin_matches_goal_embedding",
                accounting=SearchAccounting(0, 0, 0, 0, 0, 0, 0),
            )

        counter = itertools.count()
        forward_heap: list[tuple[float, int, _Node]] = []
        backward_heap: list[tuple[float, int, _Node]] = []
        forward_root = _Node(origin.detach().clone(), (), 0.0)
        backward_root = _Node(goal_state.detach().clone(), (), 0.0)
        heapq.heappush(forward_heap, (0.0, next(counter), forward_root))
        heapq.heappush(backward_heap, (0.0, next(counter), backward_root))
        forward_seen = [forward_root]
        backward_seen = [backward_root]

        forward_expansions = 0
        backward_expansions = 0
        proposer_calls = 0
        operator_scores = 0
        initiation_rejections = 0
        binding_rejections = 0
        duplicate_states = 0
        reverse_on_next_tie = False

        def accounting() -> SearchAccounting:
            return SearchAccounting(
                forward_expansions=forward_expansions,
                backward_expansions=backward_expansions,
                proposer_calls=proposer_calls,
                operator_scores=operator_scores,
                initiation_rejections=initiation_rejections,
                binding_rejections=binding_rejections,
                duplicate_states=duplicate_states,
            )

        while (
            forward_heap or backward_heap
        ) and forward_expansions + backward_expansions < budget.maximum_expansions:
            if not backward_heap:
                reverse = False
            elif not forward_heap:
                reverse = True
            elif len(backward_heap) == len(forward_heap):
                reverse = reverse_on_next_tie
                reverse_on_next_tie = not reverse_on_next_tie
            else:
                # Expanding the smaller live frontier is structural search
                # control, independent of any domain's action semantics.
                reverse = len(backward_heap) < len(forward_heap)
            heap = backward_heap if reverse else forward_heap
            _, _, node = heapq.heappop(heap)
            if len(node.path) >= budget.maximum_depth:
                continue

            state_batch = node.state.unsqueeze(0)
            target_batch = (origin if reverse else goal_condition).unsqueeze(0)
            proposal = self.model.proposer_logits(
                state_batch,
                target_batch,
                operators,
            )
            initiation = self.model.initiation_logits(
                state_batch,
                operators,
                reverse=reverse,
            )
            expected_shape = (1, operators.shape[0])
            if proposal.shape != expected_shape or initiation.shape != expected_shape:
                raise RuntimeError(
                    "planning model returned invalid proposal/initiation shape"
                )
            if not bool(torch.isfinite(proposal).all().item()) or not bool(
                torch.isfinite(initiation).all().item()
            ):
                raise RuntimeError("planning model returned non-finite scores")
            proposer_calls += 1
            operator_scores += operators.shape[0]

            proposal_log_probability = F.log_softmax(proposal[0], dim=-1)
            initiation_log_probability = F.logsigmoid(initiation[0])
            ranking = proposal_log_probability + initiation_log_probability
            initiation_probability = initiation[0].sigmoid()
            permitted = initiation_probability >= budget.minimum_initiation_probability
            initiation_rejections += int((~permitted).sum().item())
            ranking = ranking.masked_fill(~permitted, -torch.inf)

            remaining = budget.maximum_expansions - (
                forward_expansions + backward_expansions
            )
            take = min(budget.proposals_per_state, remaining, operators.shape[0])
            if take <= 0:
                break
            ranked_values, ranked_indices = torch.topk(ranking, k=take)
            finite = torch.isfinite(ranked_values)
            ranked_values = ranked_values[finite]
            ranked_indices = ranked_indices[finite]
            if not ranked_indices.numel():
                continue

            selected_operators = operators[ranked_indices]
            predicted = self.model.predict_effects(
                state_batch,
                selected_operators,
                reverse=reverse,
            )
            expected_effect_shape = (1, ranked_indices.numel(), origin.numel())
            if predicted.shape != expected_effect_shape:
                raise RuntimeError("planning model returned invalid effect shape")
            if not bool(torch.isfinite(predicted).all().item()):
                raise RuntimeError("planning model returned non-finite effects")
            termination = self.model.termination_logits(
                predicted,
                target_batch,
            )
            if termination.shape != (1, ranked_indices.numel()):
                raise RuntimeError("planning model returned invalid termination shape")
            if not bool(torch.isfinite(termination).all().item()):
                raise RuntimeError("planning model returned non-finite termination scores")

            for offset, operator_index_tensor in enumerate(ranked_indices):
                operator_index = int(operator_index_tensor.item())
                operator_identity = operator_identities[operator_index]
                candidate = predicted[0, offset].detach().clone()
                choices: tuple[PrimitiveChoice, ...] | None = None
                if primitive_decoder is not None:
                    source_state = candidate if reverse else node.state
                    predicted_state = node.state if reverse else candidate
                    decoded = tuple(
                        primitive_decoder.decode(
                            operator_index=operator_index,
                            operator_identity=operator_identity,
                            source_state=source_state.detach().clone(),
                            predicted_state=predicted_state.detach().clone(),
                            goal=goal_condition.detach().clone(),
                            reverse_search=reverse,
                        )
                    )
                    if not decoded or any(
                        not isinstance(choice, PrimitiveChoice) for choice in decoded
                    ):
                        binding_rejections += 1
                        if reverse:
                            backward_expansions += 1
                        else:
                            forward_expansions += 1
                        continue
                    choices = decoded
                step = _Step(operator_index, operator_identity, choices)
                if reverse:
                    backward_expansions += 1
                    path = (step,) + node.path
                    same_seen = backward_seen
                    opposite_seen = forward_seen
                else:
                    forward_expansions += 1
                    path = node.path + (step,)
                    same_seen = forward_seen
                    opposite_seen = backward_seen

                # Forward termination is a learned proposal, not an external
                # success claim.  The result records that distinction.
                terminal_probability = float(termination[0, offset].sigmoid().item())
                if not reverse and terminal_probability >= budget.termination_probability:
                    return _successful_plan(
                        path,
                        forward_depth=len(path),
                        backward_depth=0,
                        meeting_distance=_distance(candidate, goal_state),
                        predicted_terminal=True,
                        reason="learned_termination",
                        accounting=accounting(),
                    )

                nearest, meeting_distance = _nearest(candidate, opposite_seen)
                if nearest is not None and meeting_distance <= budget.join_tolerance:
                    if reverse:
                        prefix = nearest.path
                        suffix = path
                    else:
                        prefix = path
                        suffix = nearest.path
                    if len(prefix) + len(suffix) <= budget.maximum_depth:
                        return _successful_plan(
                            prefix + suffix,
                            forward_depth=len(prefix),
                            backward_depth=len(suffix),
                            meeting_distance=meeting_distance,
                            predicted_terminal=False,
                            reason="latent_frontier_join",
                            accounting=accounting(),
                        )

                if _within(candidate, same_seen, budget.deduplication_tolerance):
                    duplicate_states += 1
                    continue
                new_score = node.score + float(ranked_values[offset].item())
                # Termination is a priority hint only unless it crosses the
                # explicit threshold above.
                new_score += 0.1 * float(F.logsigmoid(termination[0, offset]).item())
                child = _Node(candidate, path, new_score)
                same_seen.append(child)
                heapq.heappush(heap, (-new_score, next(counter), child))

        reason = (
            "expansion_budget_exhausted"
            if forward_expansions + backward_expansions >= budget.maximum_expansions
            else "frontiers_exhausted"
        )
        return OperatorPlan(
            found=False,
            operator_indices=(),
            operator_identities=(),
            primitive_choices=() if primitive_decoder is not None else None,
            forward_depth=0,
            backward_depth=0,
            meeting_distance=None,
            predicted_terminal=False,
            reason=reason,
            accounting=accounting(),
        )

    def _validate_inputs(
        self,
        origin: torch.Tensor,
        goal_state: torch.Tensor,
        goal_condition: torch.Tensor,
        operators: torch.Tensor,
        operator_identities: tuple[str, ...],
        budget: SearchBudget,
    ) -> None:
        if not isinstance(budget, SearchBudget):
            raise TypeError("budget must be a SearchBudget")
        if (
            origin.ndim != 1
            or goal_state.shape != origin.shape
            or goal_condition.shape != origin.shape
        ):
            raise ValueError(
                "origin, goal_state, and goal_condition must share shape [width]"
            )
        if origin.numel() != self.model.width:
            raise ValueError("state width does not match the planning model")
        if operators.ndim != 2 or operators.shape[1] != origin.numel():
            raise ValueError("operators must have shape [count, width]")
        if operators.shape[0] <= 0:
            raise ValueError("at least one operator embedding is required")
        if len(operator_identities) != operators.shape[0] or any(
            not isinstance(identity, str) or not identity
            for identity in operator_identities
        ):
            raise ValueError(
                "operator_identities must provide one non-empty identity per candidate"
            )
        if len(set(operator_identities)) != len(operator_identities):
            raise ValueError("operator candidate identities must be unique")
        if budget.proposals_per_state > operators.shape[0]:
            raise ValueError("proposals_per_state exceeds the operator count")
        reference = origin
        for tensor in (goal_state, goal_condition, operators):
            if tensor.device != reference.device or tensor.dtype != reference.dtype:
                raise ValueError("all planner inputs must share device and dtype")
        if not origin.is_floating_point():
            raise ValueError("planner embeddings must be floating point")
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (origin, goal_state, goal_condition, operators)
        ):
            raise ValueError("planner embeddings must be finite")


def _successful_plan(
    path: tuple[_Step, ...],
    *,
    forward_depth: int,
    backward_depth: int,
    meeting_distance: float,
    predicted_terminal: bool,
    reason: str,
    accounting: SearchAccounting,
) -> OperatorPlan:
    choices = tuple(step.primitive_choices for step in path)
    primitive_choices = (
        None if any(item is None for item in choices) else tuple(choices)  # type: ignore[arg-type]
    )
    return OperatorPlan(
        found=True,
        operator_indices=tuple(step.operator_index for step in path),
        operator_identities=tuple(step.operator_identity for step in path),
        primitive_choices=primitive_choices,
        forward_depth=forward_depth,
        backward_depth=backward_depth,
        meeting_distance=meeting_distance,
        predicted_terminal=predicted_terminal,
        reason=reason,
        accounting=accounting,
    )


def _operator_identity(operator: Any) -> str:
    digest = getattr(operator, "digest", None)
    if isinstance(digest, str) and digest:
        return digest
    from angler.procedures.trunk import canonical_schema_text

    material = canonical_schema_text(operator).encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _embedding_identity(embedding: torch.Tensor) -> str:
    value = embedding.detach().cpu().contiguous()
    digest = hashlib.sha256(b"project-angler.operator-candidate.v1\x00")
    dtype = str(value.dtype).encode("ascii")
    digest.update(struct.pack("<I", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack("<I", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(left - right).item())


def _nearest(
    state: torch.Tensor,
    nodes: Sequence[_Node],
) -> tuple[_Node | None, float]:
    if not nodes:
        return None, math.inf
    nearest = min(nodes, key=lambda node: _distance(state, node.state))
    return nearest, _distance(state, nearest.state)


def _within(state: torch.Tensor, nodes: Sequence[_Node], tolerance: float) -> bool:
    return any(_distance(state, node.state) <= tolerance for node in nodes)


__all__ = [
    "BidirectionalOperatorPlanner",
    "OperatorPlan",
    "OperatorPlanningModel",
    "PrimitiveChoice",
    "PrimitiveChoiceDecoder",
    "SearchAccounting",
    "SearchBudget",
]
