"""Supervised neural losses over externally verified operator examples.

This module turns immutable observations into optimization targets.  It does
not execute a primitive, import a domain, or infer that a prediction is
correct.  Applicability labels, successor states, and primitive sequences are
accepted only as externally observed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from angler.procedures.execution import (
    BindingConditionedOperatorHeads,
    OperatorBinding,
    SharedPrimitiveSequenceDecoder,
    TypedEntityCandidate,
    canonicalize_binding_context,
)
from angler.procedures.records import ActionSchema, Goal, GroundAction, State
from angler.procedures.trunk import NeuralOperatorCore


def _meet_in_middle_scores(
    forward_states: torch.Tensor,
    backward_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if (
        forward_states.ndim != 2
        or backward_states.shape != forward_states.shape
        or mask.shape != (forward_states.shape[0],)
        or mask.dtype != torch.bool
    ):
        raise ValueError("meet-in-the-middle tensors are misaligned")
    pairwise = -(
        (forward_states[:, None, :] - backward_states[None, :, :]) ** 2
    ).mean(dim=-1)
    allowed = mask[:, None] & mask[None, :]
    allowed = allowed & ~torch.eye(
        len(mask),
        dtype=torch.bool,
        device=mask.device,
    )
    pairwise = pairwise.masked_fill(~allowed, -torch.inf)
    return pairwise.max(dim=1).values


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class CandidateEvidenceFusion(nn.Module):
    """Learn a stable, candidate-local calibration of neural evidence.

    Initiation, proposal, and latent-join evidence are all learned signals.
    This head only learns how reliable their relative scales are; it has no
    state, goal, binding, action, or world input from which it could encode a
    solution rule.  Unlike candidate-set standardization, adding another
    candidate cannot change an existing candidate's score.
    """

    channel_count = 3

    def __init__(self) -> None:
        super().__init__()
        self.centers = nn.Parameter(
            torch.tensor((-math.log(2.0), 0.0, 0.0))
        )
        # scale = 0.25 + softplus(raw); initialize the effective scale to 1.
        self.raw_scales = nn.Parameter(
            torch.full((self.channel_count,), _inverse_softplus(0.75))
        )
        self.weight_logits = nn.Parameter(torch.zeros(self.channel_count))
        # temperature = 0.5 + 5.5 * sigmoid(raw); initialize it to 3.
        self.raw_temperature = nn.Parameter(
            torch.tensor(math.log((2.5 / 5.5) / (1.0 - 2.5 / 5.5)))
        )

    def forward(
        self,
        initiation_logits: torch.Tensor,
        proposer_logits: torch.Tensor,
        join_scores: torch.Tensor,
        mask: torch.Tensor,
        *,
        include_proposer: bool = True,
    ) -> torch.Tensor:
        if not isinstance(include_proposer, bool):
            raise TypeError("include_proposer must be boolean")
        if (
            initiation_logits.shape != proposer_logits.shape
            or join_scores.shape != initiation_logits.shape
            or mask.shape != initiation_logits.shape
            or initiation_logits.ndim < 1
            or mask.dtype != torch.bool
        ):
            raise ValueError(
                "candidate evidence and its boolean mask must share one shape"
            )
        if (
            proposer_logits.device != initiation_logits.device
            or join_scores.device != initiation_logits.device
            or mask.device != initiation_logits.device
            or proposer_logits.dtype != initiation_logits.dtype
            or join_scores.dtype != initiation_logits.dtype
        ):
            raise ValueError("candidate evidence must share device and dtype")
        if not bool(mask.any().item()):
            raise ValueError("candidate evidence requires one active candidate")

        sources = torch.stack(
            (F.logsigmoid(initiation_logits), proposer_logits, join_scores),
            dim=-1,
        )
        if not bool(torch.isfinite(sources[mask]).all().item()):
            raise ValueError("active candidate evidence must be finite")

        centers = self.centers.to(
            device=initiation_logits.device,
            dtype=initiation_logits.dtype,
        )
        safe_sources = torch.where(
            mask.unsqueeze(-1),
            sources,
            centers.expand_as(sources),
        )
        scales = 0.25 + F.softplus(self.raw_scales).to(
            device=initiation_logits.device,
            dtype=initiation_logits.dtype,
        )
        calibrated = F.softsign((safe_sources - centers) / scales)
        weights = torch.softmax(
            self.weight_logits.to(
                device=initiation_logits.device,
                dtype=initiation_logits.dtype,
            ),
            dim=0,
        )
        if not include_proposer:
            weights = weights * weights.new_tensor((1.0, 0.0, 1.0))
        temperature = 0.5 + 5.5 * torch.sigmoid(
            self.raw_temperature.to(
                device=initiation_logits.device,
                dtype=initiation_logits.dtype,
            )
        )
        logits = temperature * (calibrated * weights).sum(dim=-1)
        return logits.masked_fill(~mask, -torch.inf)


@dataclass(frozen=True, slots=True)
class VerifiedOperatorExample:
    """One externally observed training example with explicit candidates."""

    before: State
    after: State
    goal: Goal
    positive_binding: OperatorBinding
    candidate_bindings: tuple[OperatorBinding, ...]
    applicability_labels: tuple[bool, ...]
    verified_primitives: tuple[GroundAction, ...]
    allowed_schemas: tuple[ActionSchema, ...]
    entity_candidates: tuple[TypedEntityCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.before, State) or not isinstance(self.after, State):
            raise TypeError("before and after must be State values")
        if not isinstance(self.goal, Goal):
            raise TypeError("goal must be a Goal")
        if self.before.namespace != self.after.namespace or (
            self.goal.namespace != self.before.namespace
        ):
            raise ValueError("states and goal must share one namespace")
        if self.goal.exact is not True:
            raise ValueError("verified operator examples require an exact goal")
        if not isinstance(self.positive_binding, OperatorBinding):
            raise TypeError("positive_binding must be an OperatorBinding")

        self._require_tuple(
            self.candidate_bindings,
            OperatorBinding,
            "candidate_bindings",
        )
        if not self.candidate_bindings:
            raise ValueError("candidate_bindings must not be empty")
        binding_digests = tuple(item.digest for item in self.candidate_bindings)
        if len(set(binding_digests)) != len(binding_digests):
            raise ValueError("candidate_bindings must have unique digests")
        if binding_digests.count(self.positive_binding.digest) != 1:
            raise ValueError("positive_binding must occur exactly once in candidates")
        if any(
            binding.operator.namespace != self.before.namespace
            for binding in self.candidate_bindings
        ):
            raise ValueError("candidate bindings must belong to the state namespace")

        if type(self.applicability_labels) is not tuple or any(
            type(label) is not bool for label in self.applicability_labels
        ):
            raise TypeError("applicability_labels must be an immutable bool tuple")
        if len(self.applicability_labels) != len(self.candidate_bindings):
            raise ValueError("one applicability label is required per binding")
        if not self.applicability_labels[self.positive_index]:
            raise ValueError("the positive binding must be externally applicable")

        self._require_tuple(
            self.verified_primitives,
            GroundAction,
            "verified_primitives",
        )
        self._require_tuple(self.allowed_schemas, ActionSchema, "allowed_schemas")
        if not self.allowed_schemas:
            raise ValueError("allowed_schemas must not be empty")
        schema_digests = tuple(schema.digest for schema in self.allowed_schemas)
        if len(set(schema_digests)) != len(schema_digests):
            raise ValueError("allowed_schemas must have unique digests")
        if any(schema.namespace != self.before.namespace for schema in self.allowed_schemas):
            raise ValueError("allowed schemas must belong to the state namespace")

        self._require_tuple(
            self.entity_candidates,
            TypedEntityCandidate,
            "entity_candidates",
        )
        if len(set(self.entity_candidates)) != len(self.entity_candidates):
            raise ValueError("entity_candidates must be unique")
        available_entities = {
            (candidate.type_name, candidate.value)
            for candidate in self.entity_candidates
        }
        for binding in self.candidate_bindings:
            for assignment in binding.assignments:
                key = (assignment.entity.type_name, assignment.entity.value)
                if key not in available_entities:
                    raise ValueError(
                        "every binding assignment must occur in entity_candidates"
                    )

        allowed = set(schema_digests)
        for primitive in self.verified_primitives:
            if primitive.namespace != self.before.namespace:
                raise ValueError("verified primitives must share the state namespace")
            if primitive.schema.digest not in allowed:
                raise ValueError("a verified primitive is not in allowed_schemas")
            for parameter, argument in zip(
                primitive.schema.parameters,
                primitive.arguments,
                strict=True,
            ):
                if (parameter.type_name, argument) not in available_entities:
                    raise ValueError(
                        "verified primitive arguments need typed entity candidates"
                    )

    @staticmethod
    def _require_tuple(values: object, item_type: type, name: str) -> None:
        if type(values) is not tuple:
            raise TypeError(f"{name} must be an immutable tuple")
        if any(not isinstance(item, item_type) for item in values):
            raise TypeError(f"{name} contains an invalid value")

    @property
    def positive_index(self) -> int:
        """Index used for binding-proposal imitation in this candidate set."""

        digest = self.positive_binding.digest
        for index, binding in enumerate(self.candidate_bindings):
            if binding.digest == digest:
                return index
        raise RuntimeError("validated positive binding is missing")


@dataclass(frozen=True, slots=True)
class VerifiedOperatorTrajectory:
    """A contiguous multi-operator chain backed only by observed outcomes.

    The record deliberately contains no search node, route score, benchmark
    identity, or solver metadata.  It preserves only the verified learning
    examples needed to train latent dynamics through their own predictions.
    """

    steps: tuple[VerifiedOperatorExample, ...]

    def __post_init__(self) -> None:
        if type(self.steps) is not tuple or any(
            not isinstance(step, VerifiedOperatorExample) for step in self.steps
        ):
            raise TypeError("trajectory steps must be an immutable example tuple")
        if len(self.steps) < 2:
            raise ValueError("a verified trajectory requires at least two steps")

        goal = self.steps[0].goal
        namespace = self.steps[0].before.namespace
        candidate_universe = tuple(
            binding.digest for binding in self.steps[0].candidate_bindings
        )
        for index, step in enumerate(self.steps):
            if step.goal != goal:
                raise ValueError("trajectory steps must share one exact goal")
            if step.before.namespace != namespace or step.after.namespace != namespace:
                raise ValueError("trajectory steps must share one namespace")
            if index and self.steps[index - 1].after != step.before:
                raise ValueError("trajectory steps must form one contiguous chain")
            if tuple(
                binding.digest for binding in step.candidate_bindings
            ) != candidate_universe:
                raise ValueError(
                    "trajectory steps must share the deployment candidate universe"
                )

        positive_digests = tuple(step.positive_binding.digest for step in self.steps)
        if len(set(positive_digests)) != len(positive_digests):
            raise ValueError("trajectory steps cannot reuse a consumed binding")

        def satisfies(state: State) -> bool:
            return state.namespace == goal.namespace and state.records == goal.required

        if any(satisfies(step.before) for step in self.steps):
            raise ValueError("only the final trajectory endpoint may satisfy its goal")
        if any(satisfies(step.after) for step in self.steps[:-1]):
            raise ValueError("an intermediate trajectory endpoint satisfies its goal")
        if not satisfies(self.steps[-1].after):
            raise ValueError("the final trajectory endpoint must satisfy its exact goal")

    @property
    def goal(self) -> Goal:
        return self.steps[0].goal


@dataclass(frozen=True, slots=True)
class TrajectoryLossWeights:
    """Relative weights inside the multi-step latent-dynamics objective."""

    forward: float = 1.0
    reverse: float = 1.0
    bridge: float = 1.0
    # Control and calibrated termination initially have much larger gradient
    # norms than representation/dynamics learning.  Keep them auxiliary so
    # they shape an acquired latent space instead of collapsing it.
    termination: float = 0.1
    control: float = 0.1
    fusion: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.forward,
            self.reverse,
            self.bridge,
            self.termination,
            self.control,
            self.fusion,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in values
        ):
            raise ValueError("trajectory loss weights must be finite and non-negative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one trajectory loss weight must be positive")


@dataclass(frozen=True, slots=True)
class TrajectoryLearningLosses:
    """Named losses for autonomous forward and reverse latent rollouts."""

    forward: torch.Tensor
    reverse: torch.Tensor
    bridge: torch.Tensor
    termination: torch.Tensor
    control: torch.Tensor
    fusion: torch.Tensor
    total: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "forward": self.forward,
            "reverse": self.reverse,
            "bridge": self.bridge,
            "termination": self.termination,
            "control": self.control,
            "fusion": self.fusion,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class OperatorLossWeights:
    """Explicit relative weights for independently inspectable losses."""

    effect: float = 1.0
    predecessor: float = 1.0
    initiation: float = 1.0
    termination: float = 1.0
    proposer: float = 1.0
    primitive_action: float = 1.0
    primitive_argument: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.effect,
            self.predecessor,
            self.initiation,
            self.termination,
            self.proposer,
            self.primitive_action,
            self.primitive_argument,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in values
        ):
            raise ValueError("loss weights must be finite non-negative numbers")
        if not any(value > 0 for value in values):
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True, slots=True)
class OperatorLearningLosses:
    """Named raw losses and their configured weighted total."""

    effect: torch.Tensor
    predecessor: torch.Tensor
    initiation: torch.Tensor
    termination: torch.Tensor
    proposer: torch.Tensor
    primitive_action: torch.Tensor
    primitive_argument: torch.Tensor
    total: torch.Tensor
    positive_binding_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "effect": self.effect,
            "predecessor": self.predecessor,
            "initiation": self.initiation,
            "termination": self.termination,
            "proposer": self.proposer,
            "primitive_action": self.primitive_action,
            "primitive_argument": self.primitive_argument,
            "total": self.total,
        }


class CompositeOperatorLearner(nn.Module):
    """Compute all neural operator losses without invoking an executor."""

    _SELECTION_PLASTIC_PREFIXES = (
        "heads.core.initiation_head.",
        "heads.binding_proposer.query.",
        "heads.binding_proposer.keys.",
        "heads.binding_proposer.pair_bias.",
        "candidate_fusion.",
    )

    def __init__(
        self,
        core: NeuralOperatorCore,
        decoder: SharedPrimitiveSequenceDecoder,
        *,
        heads: BindingConditionedOperatorHeads | None = None,
        weights: OperatorLossWeights | None = None,
        trajectory_weights: TrajectoryLossWeights | None = None,
        binding_hash_width: int = 192,
        hidden_width: int = 192,
    ) -> None:
        super().__init__()
        if not isinstance(core, NeuralOperatorCore):
            raise TypeError("core must be a NeuralOperatorCore")
        if not isinstance(decoder, SharedPrimitiveSequenceDecoder):
            raise TypeError("decoder must be a SharedPrimitiveSequenceDecoder")
        if decoder.width != core.width:
            raise ValueError("core and primitive decoder widths must match")
        if heads is None:
            heads = BindingConditionedOperatorHeads(
                core,
                binding_hash_width=binding_hash_width,
                hidden_width=hidden_width,
            )
        elif not isinstance(heads, BindingConditionedOperatorHeads):
            raise TypeError("heads must be BindingConditionedOperatorHeads")
        elif heads.core is not core:
            raise ValueError("binding-conditioned heads must share the supplied core")
        self.heads = heads
        self.decoder = decoder
        self.candidate_fusion = CandidateEvidenceFusion()
        self.weights = weights or OperatorLossWeights()
        self.trajectory_weights = trajectory_weights or TrajectoryLossWeights()

    @property
    def core(self) -> NeuralOperatorCore:
        return self.heads.core

    def candidate_selection_logits(
        self,
        initiation_logits: torch.Tensor,
        proposer_logits: torch.Tensor,
        join_scores: torch.Tensor,
        mask: torch.Tensor,
        *,
        include_proposer: bool = True,
        memory_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use the learned evidence calibration shared with inference.

        ``memory_bias`` is an optional candidate-local residual produced and
        bounded by a procedural-memory module.  Omitting it preserves the
        original scoring path exactly.
        """

        logits = self.candidate_fusion(
            initiation_logits,
            proposer_logits,
            join_scores,
            mask,
            include_proposer=include_proposer,
        )
        if memory_bias is None:
            return logits
        if not isinstance(memory_bias, torch.Tensor):
            raise TypeError("memory_bias must be a tensor or None")
        if memory_bias.shape != logits.shape:
            raise ValueError("memory_bias must match the candidate-logit shape")
        if memory_bias.device != logits.device or memory_bias.dtype != logits.dtype:
            raise ValueError("memory_bias must share candidate-logit device and dtype")
        if not bool(torch.isfinite(memory_bias).all().item()):
            raise ValueError("memory_bias must contain only finite values")
        return (logits + memory_bias).masked_fill(~mask, -torch.inf)

    def horizon_agnostic_join_scores(
        self,
        forward_states: torch.Tensor,
        backward_states: torch.Tensor,
        goal_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Blend direct and bridge evidence using learned termination.

        A challenge's action budget is only a safety ceiling, never an oracle
        for the number of procedure applications required.  Each candidate is
        therefore scored as a possible final step and as a possible bridge to
        another reverse-predicted step.  The learned termination head decides
        which interpretation to trust for that candidate.
        """

        if (
            forward_states.ndim != 2
            or backward_states.shape != forward_states.shape
            or goal_state.shape not in {
                (forward_states.shape[1],),
                (1, forward_states.shape[1]),
            }
            or mask.shape != (forward_states.shape[0],)
            or mask.dtype != torch.bool
        ):
            raise ValueError("horizon-agnostic join tensors are misaligned")
        if (
            backward_states.device != forward_states.device
            or goal_state.device != forward_states.device
            or mask.device != forward_states.device
            or backward_states.dtype != forward_states.dtype
            or goal_state.dtype != forward_states.dtype
        ):
            raise ValueError("horizon-agnostic join tensors must share device and dtype")
        if not bool(mask.any().item()):
            raise ValueError("horizon-agnostic join requires one active candidate")
        if not (
            bool(torch.isfinite(forward_states).all().item())
            and bool(torch.isfinite(backward_states).all().item())
            and bool(torch.isfinite(goal_state).all().item())
        ):
            raise ValueError("horizon-agnostic join inputs must be finite")

        goal = goal_state.reshape(1, -1)
        direct = -((forward_states - goal) ** 2).mean(dim=-1)
        if int(mask.sum().item()) == 1:
            # A lone remaining candidate has no distinct bridge partner.
            bridge = direct
        else:
            bridge = _meet_in_middle_scores(forward_states, backward_states, mask)
            if not bool(torch.isfinite(bridge[mask]).all().item()):
                raise ValueError("active reverse-frontier evidence must be finite")
        termination = self.core.termination_logits(
            forward_states,
            goal.expand(forward_states.shape[0], -1),
        )
        if not bool(torch.isfinite(termination).all().item()):
            raise ValueError("learned termination evidence must be finite")
        # Stop calibration is grounded only in externally verified stop labels;
        # candidate imitation may consume it but cannot repurpose it.
        final_probability = torch.sigmoid(termination).detach()
        return (
            final_probability * direct + (1.0 - final_probability) * bridge
        ).masked_fill(~mask, -torch.inf)

    def configure_plasticity(self, scope: str) -> tuple[str, ...]:
        """Select a learned update surface without changing inference.

        ``full`` exposes the complete learner for initial acquisition.
        ``selection`` keeps the acquired latent dynamics, termination, and
        primitive decoder fixed while allowing only neural applicability,
        proposal, and evidence calibration to adapt to a changed candidate
        distribution.  Neither scope encodes an action or solution route.
        """

        if scope not in {"full", "selection"}:
            raise ValueError("plasticity scope must be 'full' or 'selection'")
        enabled_names: list[str] = []
        for name, parameter in self.named_parameters():
            enabled = scope == "full" or name.startswith(
                self._SELECTION_PLASTIC_PREFIXES
            )
            parameter.requires_grad_(enabled)
            if enabled:
                enabled_names.append(name)
        if not enabled_names:
            raise RuntimeError("plasticity scope selected no parameters")
        return tuple(enabled_names)

    def forward(
        self,
        examples: Sequence[VerifiedOperatorExample],
    ) -> OperatorLearningLosses:
        if not examples:
            raise ValueError("at least one verified example is required")
        if any(not isinstance(item, VerifiedOperatorExample) for item in examples):
            raise TypeError("examples must contain VerifiedOperatorExample values")

        effect_losses: list[torch.Tensor] = []
        predecessor_losses: list[torch.Tensor] = []
        initiation_losses: list[torch.Tensor] = []
        termination_losses: list[torch.Tensor] = []
        proposer_losses: list[torch.Tensor] = []
        primitive_action_losses: list[torch.Tensor] = []
        primitive_argument_losses: list[torch.Tensor] = []
        positive_indices: list[int] = []

        for example in examples:
            (
                effect,
                predecessor,
                initiation,
                termination,
                proposer,
                primitive_action,
                primitive_argument,
            ) = self._example_losses(example)
            effect_losses.append(effect)
            predecessor_losses.append(predecessor)
            initiation_losses.append(initiation)
            termination_losses.append(termination)
            proposer_losses.append(proposer)
            primitive_action_losses.append(primitive_action)
            primitive_argument_losses.append(primitive_argument)
            positive_indices.append(example.positive_index)

        effect = torch.stack(effect_losses).mean()
        predecessor = torch.stack(predecessor_losses).mean()
        initiation = torch.stack(initiation_losses).mean()
        termination = torch.stack(termination_losses).mean()
        proposer = torch.stack(proposer_losses).mean()
        primitive_action = torch.stack(primitive_action_losses).mean()
        primitive_argument = torch.stack(primitive_argument_losses).mean()
        total = (
            self.weights.effect * effect
            + self.weights.predecessor * predecessor
            + self.weights.initiation * initiation
            + self.weights.termination * termination
            + self.weights.proposer * proposer
            + self.weights.primitive_action * primitive_action
            + self.weights.primitive_argument * primitive_argument
        )
        return OperatorLearningLosses(
            effect=effect,
            predecessor=predecessor,
            initiation=initiation,
            termination=termination,
            proposer=proposer,
            primitive_action=primitive_action,
            primitive_argument=primitive_argument,
            total=total,
            positive_binding_indices=tuple(positive_indices),
        )

    def trajectory_losses(
        self,
        trajectories: Sequence[VerifiedOperatorTrajectory],
        *,
        teacher_forcing_ratio: float = 0.0,
    ) -> TrajectoryLearningLosses:
        """Train dynamics on their own chained predictions in both directions.

        ``teacher_forcing_ratio`` blends the next rollout input toward the
        observed latent.  Zero is a fully autonomous rollout; one is ordinary
        teacher forcing.  Predictions and targets remain connected to their
        shared encoder so forward and reverse consistency train one geometry
        instead of chasing detached copies of the same moving representation.
        This method never executes an action.
        """

        if not trajectories:
            raise ValueError("at least one verified trajectory is required")
        if any(not isinstance(item, VerifiedOperatorTrajectory) for item in trajectories):
            raise TypeError("trajectories must contain VerifiedOperatorTrajectory values")
        if (
            isinstance(teacher_forcing_ratio, bool)
            or not isinstance(teacher_forcing_ratio, (int, float))
            or not math.isfinite(float(teacher_forcing_ratio))
            or not 0.0 <= float(teacher_forcing_ratio) <= 1.0
        ):
            raise ValueError("teacher_forcing_ratio must be finite and in [0, 1]")
        ratio = float(teacher_forcing_ratio)

        forward_losses: list[torch.Tensor] = []
        reverse_losses: list[torch.Tensor] = []
        bridge_losses: list[torch.Tensor] = []
        termination_losses: list[torch.Tensor] = []
        control_losses: list[torch.Tensor] = []
        fusion_losses: list[torch.Tensor] = []
        for trajectory in trajectories:
            forward, reverse, bridge, termination, control, fusion = self._trajectory_losses(
                trajectory,
                teacher_forcing_ratio=ratio,
            )
            forward_losses.append(forward)
            reverse_losses.append(reverse)
            bridge_losses.append(bridge)
            termination_losses.append(termination)
            control_losses.append(control)
            fusion_losses.append(fusion)

        forward = torch.stack(forward_losses).mean()
        reverse = torch.stack(reverse_losses).mean()
        bridge = torch.stack(bridge_losses).mean()
        termination = torch.stack(termination_losses).mean()
        control = torch.stack(control_losses).mean()
        fusion = torch.stack(fusion_losses).mean()
        weights = self.trajectory_weights
        weight_total = (
            weights.forward
            + weights.reverse
            + weights.bridge
            + weights.termination
            + weights.control
            + weights.fusion
        )
        total = (
            weights.forward * forward
            + weights.reverse * reverse
            + weights.bridge * bridge
            + weights.termination * termination
            + weights.control * control
            + weights.fusion * fusion
        ) / weight_total
        return TrajectoryLearningLosses(
            forward=forward,
            reverse=reverse,
            bridge=bridge,
            termination=termination,
            control=control,
            fusion=fusion,
            total=total,
        )

    def _trajectory_losses(
        self,
        trajectory: VerifiedOperatorTrajectory,
        *,
        teacher_forcing_ratio: float,
    ) -> tuple[torch.Tensor, ...]:
        steps = trajectory.steps
        observed = self.core.encode_states(
            (steps[0].before, *(step.after for step in steps))
        )
        goal_state = self.core.encode_goal_states((trajectory.goal,))
        candidate_sets = tuple(
            self.heads.encode_candidates(step.candidate_bindings)
            for step in steps
        )
        candidates = tuple(
            candidate_set[
                step.positive_index : step.positive_index + 1
            ]
            for step, candidate_set in zip(steps, candidate_sets, strict=True)
        )
        goal = self.core.encode_goals((trajectory.goal,))

        forward_state = observed[:1]
        forward_predictions: list[torch.Tensor] = []
        forward_losses: list[torch.Tensor] = []
        control_losses: list[torch.Tensor] = []
        fusion_losses: list[torch.Tensor] = []
        used = torch.zeros(
            len(steps[0].candidate_bindings),
            dtype=torch.bool,
            device=forward_state.device,
        )
        for index, candidate in enumerate(candidates):
            step = steps[index]
            candidate_set = candidate_sets[index]
            mask = ~used
            labels = torch.tensor(
                step.applicability_labels,
                device=forward_state.device,
                dtype=forward_state.dtype,
            )
            if index == 0:
                relative_contexts = tuple(
                    canonicalize_binding_context(
                        step.before,
                        step.goal,
                        binding,
                    )
                    for binding in step.candidate_bindings
                )
                relative_states = self.core.encode_states(
                    tuple(item[0] for item in relative_contexts)
                )
                relative_goals = self.core.encode_goals(
                    tuple(item[1] for item in relative_contexts)
                )
                initiation_logits = torch.diagonal(
                    self.core.initiation_logits(relative_states, candidate_set)
                )
                proposer_logits = torch.diagonal(
                    self.heads.binding_proposer.score_candidates(
                        relative_states,
                        relative_goals,
                        candidate_set,
                    )
                )
            else:
                initiation_logits = self.core.initiation_logits(
                    forward_state,
                    candidate_set,
                )[0]
                proposer_logits = self.heads.binding_proposer.score_candidates(
                    forward_state,
                    goal,
                    candidate_set,
                )[0]

            forward_candidates = self.core.predict_effects(
                forward_state,
                candidate_set,
            )[0]
            backward_candidates = self.core.predict_effects(
                goal_state,
                candidate_set,
                reverse=True,
            )[0]
            join = self.horizon_agnostic_join_scores(
                forward_candidates,
                backward_candidates,
                goal_state,
                mask,
            )
            combined_logits = self.candidate_selection_logits(
                initiation_logits,
                proposer_logits,
                join,
                mask,
            )
            target_index = torch.tensor(
                (step.positive_index,),
                device=forward_state.device,
                dtype=torch.long,
            )
            current_control = [
                F.binary_cross_entropy_with_logits(
                    initiation_logits[mask],
                    labels[mask],
                ),
                F.cross_entropy(
                    proposer_logits.masked_fill(~mask, -torch.inf).unsqueeze(0),
                    target_index,
                ),
            ]
            fusion_losses.append(
                F.cross_entropy(combined_logits.unsqueeze(0), target_index)
            )
            if index:
                primitive_action, primitive_argument = self._primitive_losses(
                    step,
                    forward_state,
                    goal,
                )
                current_control.extend((primitive_action, primitive_argument))
            control_losses.append(torch.stack(current_control).mean())

            predicted = forward_candidates[
                step.positive_index : step.positive_index + 1
            ]
            target = observed[index + 1 : index + 2]
            forward_predictions.append(predicted)
            forward_losses.append(F.mse_loss(predicted, target))
            forward_state = torch.lerp(predicted, target, teacher_forcing_ratio)
            used[step.positive_index] = True

        reverse_state = goal_state
        reverse_predictions: list[torch.Tensor | None] = [None] * len(steps)
        reverse_losses: list[torch.Tensor] = []
        for index in range(len(steps) - 1, -1, -1):
            predicted = self.core.predict_effects(
                reverse_state,
                candidates[index],
                reverse=True,
            )[:, 0, :]
            target = observed[index : index + 1]
            reverse_predictions[index] = predicted
            reverse_losses.append(F.mse_loss(predicted, target))
            reverse_state = torch.lerp(predicted, target, teacher_forcing_ratio)

        bridges = [
            F.mse_loss(forward_predictions[index - 1], reverse_predictions[index])
            for index in range(1, len(steps))
        ]
        rolled_successors = torch.cat(forward_predictions, dim=0)
        termination_targets = torch.zeros(
            len(steps),
            device=rolled_successors.device,
            dtype=rolled_successors.dtype,
        )
        termination_targets[-1] = 1.0
        termination_logits = self.core.termination_logits(
            rolled_successors,
            goal_state.expand(len(steps), -1),
        )
        termination_bce = F.binary_cross_entropy_with_logits(
            termination_logits,
            termination_targets,
        )
        # BCE alone repeatedly learned an all-positive or all-negative bias on
        # small streams.  These generic margins calibrate the public zero
        # threshold and order the verified final above every intermediate;
        # they do not reveal a route or force a fixed horizon at runtime.
        margin = termination_logits.new_tensor(1.0)
        final_margin = F.relu(margin - termination_logits[-1]).square()
        intermediate_margin = F.relu(
            margin + termination_logits[:-1]
        ).square().mean()
        ordering_margin = F.relu(
            margin - (termination_logits[-1] - termination_logits[:-1])
        ).square().mean()
        termination = (
            termination_bce
            + final_margin
            + intermediate_margin
            + ordering_margin
        )
        return (
            torch.stack(forward_losses).mean(),
            torch.stack(reverse_losses).mean(),
            torch.stack(bridges).mean(),
            termination,
            torch.stack(control_losses).mean(),
            torch.stack(fusion_losses).mean(),
        )

    def _example_losses(
        self,
        example: VerifiedOperatorExample,
    ) -> tuple[torch.Tensor, ...]:
        before = self.core.encode_states((example.before,))
        after = self.core.encode_states((example.after,))
        goal = self.core.encode_goals((example.goal,))
        goal_state = self.core.encode_goal_states((example.goal,))
        candidates = self.heads.encode_candidates(example.candidate_bindings)
        positive = example.positive_index

        relative_contexts = tuple(
            canonicalize_binding_context(
                example.before,
                example.goal,
                binding,
            )
            for binding in example.candidate_bindings
        )
        relative_before = self.core.encode_states(
            tuple(item[0] for item in relative_contexts)
        )
        relative_goal = self.core.encode_goals(
            tuple(item[1] for item in relative_contexts)
        )

        initiation_logits = torch.diagonal(
            self.core.initiation_logits(relative_before, candidates)
        )
        applicability = torch.tensor(
            example.applicability_labels,
            device=before.device,
            dtype=before.dtype,
        )
        initiation_loss = F.binary_cross_entropy_with_logits(
            initiation_logits,
            applicability,
        )

        predicted_after = self.core.predict_effects(before, candidates)[:, positive, :]
        effect_loss = F.mse_loss(predicted_after, after)
        predicted_before = self.core.predict_effects(
            after,
            candidates,
            reverse=True,
        )[:, positive, :]
        predecessor_loss = F.mse_loss(predicted_before, before)

        # Show termination both the externally observed successor and its own
        # predicted successor.  The former teaches the convergence relation
        # directly; the latter keeps the decision calibrated to the latent
        # state it will actually receive during autonomous rollout.
        termination_states = torch.cat((before, after, predicted_after), dim=0)
        # Exact goals have a concrete state-space anchor.  Termination learns
        # convergence in that shared representation; the descriptive goal
        # encoder remains available to proposal and primitive decoding.
        termination_goals = goal_state.expand(3, -1)
        after_satisfies = self._satisfies_exact(example.after, example.goal)
        termination_targets = torch.tensor(
            (
                self._satisfies_exact(example.before, example.goal),
                after_satisfies,
                after_satisfies,
            ),
            device=before.device,
            dtype=before.dtype,
        )
        termination_loss = F.binary_cross_entropy_with_logits(
            self.core.termination_logits(termination_states, termination_goals),
            termination_targets,
        )

        proposer_logits = torch.diagonal(
            self.heads.binding_proposer.score_candidates(
                relative_before,
                relative_goal,
                candidates,
            )
        )
        target = torch.tensor((positive,), device=before.device, dtype=torch.long)
        proposer_loss = F.cross_entropy(proposer_logits.unsqueeze(0), target)

        primitive_action_loss, primitive_argument_loss = self._primitive_losses(
            example,
            before,
            goal,
        )
        return (
            effect_loss,
            predecessor_loss,
            initiation_loss,
            termination_loss,
            proposer_loss,
            primitive_action_loss,
            primitive_argument_loss,
        )

    def _primitive_losses(
        self,
        example: VerifiedOperatorExample,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(example.verified_primitives) >= self.decoder.maximum_steps:
            raise ValueError(
                "verified primitive sequence leaves no decoder step for STOP"
            )
        schema_indices = {
            schema.digest: index
            for index, schema in enumerate(example.allowed_schemas)
        }
        entity_indices = {
            (candidate.type_name, candidate.value): index
            for index, candidate in enumerate(example.entity_candidates)
        }
        history = self.decoder.initial_history(
            1,
            device=state.device,
            dtype=state.dtype,
        )
        action_losses: list[torch.Tensor] = []
        argument_losses: list[torch.Tensor] = []
        sequence: tuple[GroundAction | None, ...] = (
            *example.verified_primitives,
            None,
        )
        for step, primitive in enumerate(sequence):
            scores = self.decoder(
                state,
                goal,
                (example.positive_binding,),
                example.allowed_schemas,
                example.entity_candidates,
                torch.tensor((step,), device=state.device, dtype=torch.long),
                history=history,
            )
            if primitive is None:
                target_action = scores.stop_index
                argument_indices: tuple[int, ...] = ()
            else:
                target_action = schema_indices[primitive.schema.digest]
                argument_indices = tuple(
                    entity_indices[(parameter.type_name, argument)]
                    for parameter, argument in zip(
                        primitive.schema.parameters,
                        primitive.arguments,
                        strict=True,
                    )
                )
            action_losses.append(
                F.cross_entropy(
                    scores.action_logits,
                    torch.tensor(
                        (target_action,),
                        device=state.device,
                        dtype=torch.long,
                    ),
                )
            )
            if primitive is not None:
                for position, entity_index in enumerate(argument_indices):
                    logits = scores.argument_logits[
                        0,
                        target_action,
                        position,
                    ].unsqueeze(0)
                    argument_losses.append(
                        F.cross_entropy(
                            logits,
                            torch.tensor(
                                (entity_index,),
                                device=state.device,
                                dtype=torch.long,
                            ),
                        )
                    )
                history = self.decoder.advance_history(
                    history,
                    target_action,
                    argument_indices,
                    example.allowed_schemas,
                    example.entity_candidates,
                )

        action_loss = torch.stack(action_losses).mean()
        if argument_losses:
            argument_loss = torch.stack(argument_losses).mean()
        else:
            argument_loss = action_loss * 0.0
        return action_loss, argument_loss

    @staticmethod
    def _satisfies_exact(state: State, goal: Goal) -> float:
        return float(state.namespace == goal.namespace and state.records == goal.required)


__all__ = [
    "CandidateEvidenceFusion",
    "CompositeOperatorLearner",
    "OperatorLearningLosses",
    "OperatorLossWeights",
    "TrajectoryLearningLosses",
    "TrajectoryLossWeights",
    "VerifiedOperatorExample",
    "VerifiedOperatorTrajectory",
]
