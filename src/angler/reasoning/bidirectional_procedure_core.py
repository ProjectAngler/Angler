"""Learned forward/backward dynamics and exact-frontier procedure construction.

The constructor is deliberately task-blind.  It receives only an origin
permutation, a proposed goal permutation, and a fixed primitive action set.
Neural dynamics define the search graph; exact decoded states join the two
frontiers.  Environment execution and success judgment remain outside this
module.

The design is a clean-room synthesis of reversed-transition training from
Backward Learning (MIT), future-state relabeling from HER, and the shared
subgoal idea from SGT-PG (MIT).  No donor source code is copied here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


Permutation = tuple[int, ...]

_DIGEST_DOMAIN = b"project-angler.bidirectional-procedure-core.v1\x00"


@dataclass(frozen=True, slots=True)
class BidirectionalProcedureConfig:
    """Topology and hard planning limits for one generic procedure core."""

    item_count: int = 7
    hidden_width: int = 256
    action_width: int = 64
    maximum_horizon: int = 12

    def __post_init__(self) -> None:
        if self.item_count < 3:
            raise ValueError("item_count must be at least three")
        if self.hidden_width <= 0 or self.action_width <= 0:
            raise ValueError("model widths must be positive")
        if self.maximum_horizon <= 0:
            raise ValueError("maximum_horizon must be positive")

    @property
    def action_count(self) -> int:
        """One adjacent-exchange primitive for each neighboring position."""

        return self.item_count - 1


@dataclass(frozen=True, slots=True)
class ProcedureLearningBatch:
    """Observed transitions plus future-state relabels from real trajectories."""

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    origins: torch.Tensor
    goals: torch.Tensor
    horizons: torch.Tensor
    first_actions: torch.Tensor
    last_actions: torch.Tensor
    midpoints: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProcedurePlan:
    """One committed model-derived candidate; it is not evidence of success."""

    found: bool
    origin: Permutation
    goal: Permutation
    actions: tuple[int, ...]
    meeting_state: Permutation | None
    forward_depth: int
    backward_depth: int
    forward_expansions: int
    backward_expansions: int
    exact_frontier_join: bool
    reason: str

    @property
    def total_expansions(self) -> int:
        return self.forward_expansions + self.backward_expansions


class BidirectionalProcedureCore(nn.Module):
    """Learn transition geometry, then compose it with bounded generic search."""

    def __init__(self, config: BidirectionalProcedureConfig) -> None:
        super().__init__()
        self.config = config
        n = config.item_count
        hidden = config.hidden_width
        action_width = config.action_width

        self.state_encoder = nn.Sequential(
            nn.Linear(n * n, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.action_embedding = nn.Embedding(config.action_count, action_width)
        self.horizon_embedding = nn.Embedding(
            config.maximum_horizon + 1,
            action_width,
        )

        transition_width = hidden + action_width
        self.forward_dynamics = _head(transition_width, hidden, n * n)
        self.backward_dynamics = _head(transition_width, hidden, n * n)

        pair_width = hidden * 3
        goal_width = hidden * 3 + action_width
        self.inverse_action = _head(pair_width, hidden, config.action_count)
        self.forward_policy = _head(goal_width, hidden, config.action_count)
        self.backward_policy = _head(goal_width, hidden, config.action_count)
        self.distance = _head(pair_width, hidden, config.maximum_horizon + 1)
        self.midpoint = _head(goal_width, hidden, n * n)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_state_logits(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        encoded, actions = self._transition_inputs(states, actions)
        return self.forward_dynamics(
            torch.cat((encoded, self.action_embedding(actions)), dim=-1)
        ).view(-1, self.config.item_count, self.config.item_count)

    def backward_state_logits(
        self,
        next_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        encoded, actions = self._transition_inputs(next_states, actions)
        return self.backward_dynamics(
            torch.cat((encoded, self.action_embedding(actions)), dim=-1)
        ).view(-1, self.config.item_count, self.config.item_count)

    def inverse_action_logits(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        left, right = self._paired_states(states, next_states)
        return self.inverse_action(
            torch.cat((left, right, right - left), dim=-1)
        )

    def forward_policy_logits(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        horizons: torch.Tensor,
    ) -> torch.Tensor:
        return self._policy_logits(
            self.forward_policy,
            states,
            goals,
            horizons,
        )

    def backward_policy_logits(
        self,
        goals: torch.Tensor,
        origins: torch.Tensor,
        horizons: torch.Tensor,
    ) -> torch.Tensor:
        return self._policy_logits(
            self.backward_policy,
            goals,
            origins,
            horizons,
        )

    def distance_logits(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
    ) -> torch.Tensor:
        left, right = self._paired_states(states, goals)
        return self.distance(torch.cat((left, right, right - left), dim=-1))

    def midpoint_logits(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        horizons: torch.Tensor,
    ) -> torch.Tensor:
        left, right, horizon_features = self._goal_inputs(
            states,
            goals,
            horizons,
        )
        return self.midpoint(
            torch.cat((left, right, right - left, horizon_features), dim=-1)
        ).view(-1, self.config.item_count, self.config.item_count)

    def learning_losses(
        self,
        batch: ProcedureLearningBatch,
        *,
        cycle_weight: float = 0.25,
        auxiliary_weight: float = 0.2,
    ) -> Mapping[str, torch.Tensor]:
        """Compute dense losses from transitions and trajectory subsegments."""

        if not math.isfinite(cycle_weight) or cycle_weight < 0.0:
            raise ValueError("cycle_weight must be finite and nonnegative")
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0.0:
            raise ValueError("auxiliary_weight must be finite and nonnegative")
        self._validate_batch(batch)
        forward_logits = self.forward_state_logits(batch.states, batch.actions)
        backward_logits = self.backward_state_logits(
            batch.next_states,
            batch.actions,
        )
        forward_loss = _state_cross_entropy(forward_logits, batch.next_states)
        backward_loss = _state_cross_entropy(backward_logits, batch.states)
        inverse_loss = F.cross_entropy(
            self.inverse_action_logits(batch.states, batch.next_states),
            batch.actions,
        )

        forward_probabilities = forward_logits.softmax(dim=-1)
        backward_probabilities = backward_logits.softmax(dim=-1)
        forward_cycle = self.forward_state_logits(
            backward_probabilities,
            batch.actions,
        )
        backward_cycle = self.backward_state_logits(
            forward_probabilities,
            batch.actions,
        )
        cycle_loss = 0.5 * (
            _state_cross_entropy(forward_cycle, batch.next_states)
            + _state_cross_entropy(backward_cycle, batch.states)
        )

        forward_policy_loss = F.cross_entropy(
            self.forward_policy_logits(
                batch.origins,
                batch.goals,
                batch.horizons,
            ),
            batch.first_actions,
        )
        backward_policy_loss = F.cross_entropy(
            self.backward_policy_logits(
                batch.goals,
                batch.origins,
                batch.horizons,
            ),
            batch.last_actions,
        )
        distance_loss = F.cross_entropy(
            self.distance_logits(batch.origins, batch.goals),
            batch.horizons,
        )
        midpoint_loss = _state_cross_entropy(
            self.midpoint_logits(
                batch.origins,
                batch.goals,
                batch.horizons,
            ),
            batch.midpoints,
        )
        auxiliary = (
            forward_policy_loss
            + backward_policy_loss
            + distance_loss
            + midpoint_loss
        )
        total = (
            forward_loss
            + backward_loss
            + inverse_loss
            + cycle_weight * cycle_loss
            + auxiliary_weight * auxiliary
        )
        return {
            "total": total,
            "forward": forward_loss,
            "backward": backward_loss,
            "inverse_action": inverse_loss,
            "cycle": cycle_loss,
            "forward_policy": forward_policy_loss,
            "backward_policy": backward_policy_loss,
            "distance": distance_loss,
            "midpoint": midpoint_loss,
        }

    @torch.no_grad()
    def construct_procedure(
        self,
        origin: Sequence[int],
        goal: Sequence[int],
        *,
        maximum_steps: int,
        maximum_expansions: int,
        use_backward: bool = True,
        actions_per_state: int | None = None,
        policy_action_permutation: Sequence[int] | None = None,
        backward_action_permutation: Sequence[int] | None = None,
    ) -> ProcedurePlan:
        """Construct one candidate without calling an environment or verifier.

        `policy_action_permutation` is an experimental causal ablation.  It
        redirects each learned policy suggestion to another valid primitive
        while leaving the learned dynamics and search budget unchanged.

        `backward_action_permutation` is an experimental causal ablation.  It
        changes which action reaches the backward model while preserving the
        recorded primitive and amount of model computation.
        """

        origin_state = _validate_permutation(origin, self.config.item_count)
        goal_state = _validate_permutation(goal, self.config.item_count)
        if maximum_steps <= 0 or maximum_steps > self.config.maximum_horizon:
            raise ValueError("maximum_steps is outside the configured horizon")
        if maximum_expansions <= 0:
            raise ValueError("maximum_expansions must be positive")
        action_count = self.config.action_count
        if actions_per_state is None:
            actions_per_state = action_count
        if not 1 <= actions_per_state <= action_count:
            raise ValueError("actions_per_state is outside the action space")
        policy_permutation = tuple(range(action_count))
        if policy_action_permutation is not None:
            policy_permutation = tuple(policy_action_permutation)
            if sorted(policy_permutation) != list(range(action_count)):
                raise ValueError("policy_action_permutation must be a permutation")
        backward_permutation = tuple(range(action_count))
        if backward_action_permutation is not None:
            backward_permutation = tuple(backward_action_permutation)
            if sorted(backward_permutation) != list(range(action_count)):
                raise ValueError("backward_action_permutation must be a permutation")

        if origin_state == goal_state:
            return ProcedurePlan(
                True,
                origin_state,
                goal_state,
                (),
                origin_state,
                0,
                0,
                0,
                0,
                True,
                "origin_is_goal",
            )

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            forward_seen: dict[Permutation, tuple[int, ...]] = {origin_state: ()}
            backward_seen: dict[Permutation, tuple[int, ...]] = {goal_state: ()}
            forward_frontier: dict[Permutation, tuple[int, ...]] = {
                origin_state: ()
            }
            backward_frontier: dict[Permutation, tuple[int, ...]] = {
                goal_state: ()
            }
            forward_depth = 0
            backward_depth = 0
            forward_expansions = 0
            backward_expansions = 0

            while forward_frontier or (use_backward and backward_frontier):
                can_forward = bool(forward_frontier) and forward_depth < maximum_steps
                can_backward = (
                    use_backward
                    and bool(backward_frontier)
                    and backward_depth < maximum_steps
                )
                if not can_forward and not can_backward:
                    break

                # Expanding the smaller live frontier is the generic
                # bidirectional-search policy; learned heads order its actions.
                expand_forward = can_forward and (
                    not can_backward
                    or len(forward_frontier) <= len(backward_frontier)
                )
                remaining_budget = maximum_expansions - (
                    forward_expansions + backward_expansions
                )
                if remaining_budget <= 0:
                    break

                if expand_forward:
                    expanded, attempted = self._expand_forward_frontier(
                        forward_frontier,
                        goal_state,
                        maximum_steps - forward_depth,
                        actions_per_state,
                        remaining_budget,
                        device,
                        policy_permutation,
                    )
                    forward_expansions += attempted
                    forward_depth += 1
                    next_frontier: dict[Permutation, tuple[int, ...]] = {}
                    for state, path in expanded:
                        if state in forward_seen:
                            continue
                        forward_seen[state] = path
                        if state in backward_seen:
                            suffix = backward_seen[state]
                            if len(path) + len(suffix) <= maximum_steps:
                                return ProcedurePlan(
                                    True,
                                    origin_state,
                                    goal_state,
                                    path + suffix,
                                    state,
                                    len(path),
                                    len(suffix),
                                    forward_expansions,
                                    backward_expansions,
                                    True,
                                    "exact_frontier_join",
                                )
                        next_frontier[state] = path
                    forward_frontier = next_frontier
                else:
                    expanded, attempted = self._expand_backward_frontier(
                        backward_frontier,
                        origin_state,
                        maximum_steps - backward_depth,
                        actions_per_state,
                        remaining_budget,
                        device,
                        policy_permutation,
                        backward_permutation,
                    )
                    backward_expansions += attempted
                    backward_depth += 1
                    next_frontier = {}
                    for state, suffix in expanded:
                        if state in backward_seen:
                            continue
                        backward_seen[state] = suffix
                        if state in forward_seen:
                            prefix = forward_seen[state]
                            if len(prefix) + len(suffix) <= maximum_steps:
                                return ProcedurePlan(
                                    True,
                                    origin_state,
                                    goal_state,
                                    prefix + suffix,
                                    state,
                                    len(prefix),
                                    len(suffix),
                                    forward_expansions,
                                    backward_expansions,
                                    True,
                                    "exact_frontier_join",
                                )
                        next_frontier[state] = suffix
                    backward_frontier = next_frontier

                if not use_backward and goal_state in forward_seen:
                    path = forward_seen[goal_state]
                    return ProcedurePlan(
                        True,
                        origin_state,
                        goal_state,
                        path,
                        goal_state,
                        len(path),
                        0,
                        forward_expansions,
                        0,
                        True,
                        "forward_frontier_reached_goal",
                    )

            reason = (
                "expansion_budget_exhausted"
                if forward_expansions + backward_expansions >= maximum_expansions
                else "no_exact_join_within_horizon"
            )
            return ProcedurePlan(
                False,
                origin_state,
                goal_state,
                (),
                None,
                forward_depth,
                backward_depth,
                forward_expansions,
                backward_expansions,
                False,
                reason,
            )
        finally:
            self.train(was_training)

    def _expand_forward_frontier(
        self,
        frontier: Mapping[Permutation, tuple[int, ...]],
        goal: Permutation,
        horizon: int,
        actions_per_state: int,
        budget: int,
        device: torch.device,
        policy_action_permutation: Sequence[int],
    ) -> tuple[list[tuple[Permutation, tuple[int, ...]]], int]:
        records: list[tuple[Permutation, tuple[int, ...], int]] = []
        states = tuple(frontier)
        state_tensor = permutations_to_tensor(states, device=device)
        goal_tensor = permutations_to_tensor((goal,) * len(states), device=device)
        horizons = torch.full(
            (len(states),),
            min(horizon, self.config.maximum_horizon),
            device=device,
            dtype=torch.long,
        )
        order = self.forward_policy_logits(
            state_tensor,
            goal_tensor,
            horizons,
        ).argsort(dim=-1, descending=True)
        for row, state in enumerate(states):
            for ranked_action in order[row, :actions_per_state].tolist():
                if len(records) >= budget:
                    break
                action = policy_action_permutation[int(ranked_action)]
                records.append((state, frontier[state], int(action)))
            if len(records) >= budget:
                break
        if not records:
            return [], 0
        source = permutations_to_tensor(
            tuple(record[0] for record in records),
            device=device,
        )
        actions = torch.tensor(
            [record[2] for record in records],
            device=device,
            dtype=torch.long,
        )
        predicted = _decode_permutation_logits(self.forward_state_logits(source, actions))
        result: list[tuple[Permutation, tuple[int, ...]]] = []
        for record, state in zip(records, predicted, strict=True):
            if state is not None:
                result.append((state, record[1] + (record[2],)))
        return result, len(records)

    def _expand_backward_frontier(
        self,
        frontier: Mapping[Permutation, tuple[int, ...]],
        origin: Permutation,
        horizon: int,
        actions_per_state: int,
        budget: int,
        device: torch.device,
        policy_action_permutation: Sequence[int],
        action_permutation: Sequence[int],
    ) -> tuple[list[tuple[Permutation, tuple[int, ...]]], int]:
        records: list[tuple[Permutation, tuple[int, ...], int]] = []
        states = tuple(frontier)
        state_tensor = permutations_to_tensor(states, device=device)
        origin_tensor = permutations_to_tensor((origin,) * len(states), device=device)
        horizons = torch.full(
            (len(states),),
            min(horizon, self.config.maximum_horizon),
            device=device,
            dtype=torch.long,
        )
        order = self.backward_policy_logits(
            state_tensor,
            origin_tensor,
            horizons,
        ).argsort(dim=-1, descending=True)
        for row, state in enumerate(states):
            for ranked_action in order[row, :actions_per_state].tolist():
                if len(records) >= budget:
                    break
                action = policy_action_permutation[int(ranked_action)]
                records.append((state, frontier[state], int(action)))
            if len(records) >= budget:
                break
        if not records:
            return [], 0
        target = permutations_to_tensor(
            tuple(record[0] for record in records),
            device=device,
        )
        model_actions = torch.tensor(
            [action_permutation[record[2]] for record in records],
            device=device,
            dtype=torch.long,
        )
        predicted = _decode_permutation_logits(
            self.backward_state_logits(target, model_actions)
        )
        result: list[tuple[Permutation, tuple[int, ...]]] = []
        for record, state in zip(records, predicted, strict=True):
            if state is not None:
                result.append((state, (record[2],) + record[1]))
        return result, len(records)

    def _transition_inputs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_states(states)
        if (
            actions.ndim != 1
            or actions.shape[0] != states.shape[0]
            or actions.dtype != torch.long
            or actions.device != states.device
        ):
            raise ValueError("actions must be torch.long with shape [batch]")
        if bool(((actions < 0) | (actions >= self.config.action_count)).any().item()):
            raise ValueError("actions contain an index outside the action space")
        return self.state_encoder(states.flatten(1)), actions

    def _paired_states(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_states(left)
        self._validate_states(right)
        if left.shape != right.shape or left.device != right.device:
            raise ValueError("paired states must share shape and device")
        return self.state_encoder(left.flatten(1)), self.state_encoder(right.flatten(1))

    def _goal_inputs(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        horizons: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_left, encoded_right = self._paired_states(left, right)
        if (
            horizons.ndim != 1
            or horizons.shape[0] != left.shape[0]
            or horizons.dtype != torch.long
            or horizons.device != left.device
        ):
            raise ValueError("horizons must be torch.long with shape [batch]")
        if bool(
            ((horizons < 0) | (horizons > self.config.maximum_horizon)).any().item()
        ):
            raise ValueError("horizons are outside the configured range")
        return encoded_left, encoded_right, self.horizon_embedding(horizons)

    def _policy_logits(
        self,
        head: nn.Module,
        left: torch.Tensor,
        right: torch.Tensor,
        horizons: torch.Tensor,
    ) -> torch.Tensor:
        encoded_left, encoded_right, horizon_features = self._goal_inputs(
            left,
            right,
            horizons,
        )
        return head(
            torch.cat(
                (
                    encoded_left,
                    encoded_right,
                    encoded_right - encoded_left,
                    horizon_features,
                ),
                dim=-1,
            )
        )

    def _validate_states(self, states: torch.Tensor) -> None:
        expected = (self.config.item_count, self.config.item_count)
        if states.ndim != 3 or states.shape[1:] != expected:
            raise ValueError(f"states must have shape [batch, {expected[0]}, {expected[1]}]")
        if not torch.is_floating_point(states):
            raise ValueError("state tensors must be floating point")
        if states.device != next(self.parameters()).device:
            raise ValueError("state tensors must share the model device")
        if not bool(torch.isfinite(states).all().item()):
            raise ValueError("state tensors must be finite")

    def _validate_batch(self, batch: ProcedureLearningBatch) -> None:
        state_fields = (
            batch.states,
            batch.next_states,
            batch.origins,
            batch.goals,
            batch.midpoints,
        )
        for states in state_fields:
            self._validate_states(states)
        batch_size = batch.states.shape[0]
        if any(states.shape[0] != batch_size for states in state_fields):
            raise ValueError("all learning-batch states must share a batch")
        index_fields = (
            (batch.actions, self.config.action_count, "actions"),
            (batch.first_actions, self.config.action_count, "first_actions"),
            (batch.last_actions, self.config.action_count, "last_actions"),
            (batch.horizons, self.config.maximum_horizon + 1, "horizons"),
        )
        for values, ceiling, name in index_fields:
            if (
                values.shape != (batch_size,)
                or values.dtype != torch.long
                or values.device != batch.states.device
            ):
                raise ValueError(f"{name} must be torch.long with shape [batch]")
            if bool(((values < 0) | (values >= ceiling)).any().item()):
                raise ValueError(f"{name} contains an out-of-range value")
        if bool((batch.horizons == 0).any().item()):
            raise ValueError("trajectory relabels require a positive horizon")


def permutations_to_tensor(
    permutations: Sequence[Sequence[int]],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode public symbolic states without adding task or evaluator features."""

    if not permutations:
        raise ValueError("at least one permutation is required")
    item_count = len(permutations[0])
    validated = tuple(
        _validate_permutation(permutation, item_count)
        for permutation in permutations
    )
    indices = torch.tensor(validated, device=device, dtype=torch.long)
    return F.one_hot(indices, num_classes=item_count).to(dtype=dtype)


def procedure_core_digest(module: nn.Module) -> str:
    """Hash all persistent learned state for causal reset/swap comparisons."""

    digest = hashlib.sha256(_DIGEST_DOMAIN)
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _head(input_width: int, hidden_width: int, output_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_width),
        nn.Linear(input_width, hidden_width),
        nn.SiLU(),
        nn.Linear(hidden_width, output_width),
    )


def _state_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("state logits and targets must share shape")
    target_indices = targets.argmax(dim=-1)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_indices.reshape(-1),
    )


def _decode_permutation_logits(logits: torch.Tensor) -> tuple[Permutation | None, ...]:
    decoded = logits.argmax(dim=-1).detach().cpu().tolist()
    result: list[Permutation | None] = []
    for row in decoded:
        candidate = tuple(int(value) for value in row)
        result.append(
            candidate if sorted(candidate) == list(range(len(candidate))) else None
        )
    return tuple(result)


def _validate_permutation(
    permutation: Sequence[int],
    item_count: int,
) -> Permutation:
    if isinstance(permutation, (str, bytes)):
        raise TypeError("a permutation must be an integer sequence")
    result = tuple(permutation)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError("permutation values must be integers")
    if len(result) != item_count or sorted(result) != list(range(item_count)):
        raise ValueError("state is not a complete permutation")
    return result


__all__ = [
    "BidirectionalProcedureConfig",
    "BidirectionalProcedureCore",
    "Permutation",
    "ProcedureLearningBatch",
    "ProcedurePlan",
    "permutations_to_tensor",
    "procedure_core_digest",
]
