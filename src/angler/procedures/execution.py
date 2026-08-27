"""Typed binding and neural primitive-sequence proposal without execution.

This module is the explicit boundary between a learned symbolic operator and a
candidate primitive procedure.  It can enumerate complete typed bindings and
decode :class:`~angler.procedures.records.GroundAction` values, but it imports
no world and calls no executor.  Evaluation must commit and execute returned
actions through an external domain adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import re
from typing import Mapping, Sequence

import torch
from torch import nn

from angler.procedures.operators import LearnedOperator, TypedVariable
from angler.procedures.records import ActionSchema, Goal, GroundAction, Record, State
from angler.procedures.trunk import (
    FrozenHashTextEncoder,
    NeuralOperatorCore,
    SchemaEncoder,
)


_QUALIFIED_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")
_DIGEST_DOMAIN = b"project-angler.operator-binding.v1\x00"


@dataclass(frozen=True, slots=True, order=True)
class TypedEntityCandidate:
    """One externally supplied possible binding value and its declared type."""

    value: str
    type_name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or not self.value
            or self.value != self.value.strip()
        ):
            raise ValueError("entity value must be a non-empty stripped string")
        if not isinstance(self.type_name, str) or not _QUALIFIED_NAME.fullmatch(
            self.type_name
        ):
            raise ValueError("entity type_name must be a canonical qualified name")


@dataclass(frozen=True, slots=True)
class BindingAssignment:
    """One typed variable-to-entity assignment."""

    variable: TypedVariable
    entity: TypedEntityCandidate

    def __post_init__(self) -> None:
        if not isinstance(self.variable, TypedVariable):
            raise TypeError("binding variable must be a TypedVariable")
        if not isinstance(self.entity, TypedEntityCandidate):
            raise TypeError("binding entity must be a TypedEntityCandidate")
        if self.variable.type_name != self.entity.type_name:
            raise ValueError("binding entity type does not match its variable")


@dataclass(frozen=True, slots=True)
class OperatorBinding:
    """A complete, canonical typed binding for one learned operator."""

    operator: LearnedOperator
    assignments: tuple[BindingAssignment, ...]
    _digest_cache: str = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.operator, LearnedOperator):
            raise TypeError("binding operator must be a LearnedOperator")
        if not isinstance(self.assignments, tuple):
            raise TypeError("binding assignments must be an immutable tuple")
        if any(
            not isinstance(assignment, BindingAssignment)
            for assignment in self.assignments
        ):
            raise TypeError("assignments must contain only BindingAssignment values")
        by_name: dict[str, BindingAssignment] = {}
        for assignment in self.assignments:
            name = assignment.variable.name
            if name in by_name:
                raise ValueError("a binding cannot assign one variable twice")
            by_name[name] = assignment
        declared = {variable.name: variable for variable in self.operator.variables}
        if set(by_name) != set(declared):
            missing = sorted(set(declared) - set(by_name))
            extra = sorted(set(by_name) - set(declared))
            raise ValueError(
                f"binding must cover every operator variable; missing={missing}, extra={extra}"
            )
        for name, assignment in by_name.items():
            if assignment.variable != declared[name]:
                raise ValueError("binding variable declaration differs from the operator")
        canonical = tuple(by_name[variable.name] for variable in self.operator.variables)
        object.__setattr__(self, "assignments", canonical)
        payload = {
            "operator": self.operator.digest,
            "assignments": [
                {
                    "variable": assignment.variable.name,
                    "type_name": assignment.variable.type_name,
                    "value": assignment.entity.value,
                }
                for assignment in canonical
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(
            self,
            "_digest_cache",
            "sha256:" + hashlib.sha256(_DIGEST_DOMAIN + encoded).hexdigest(),
        )

    def value_for(self, variable_name: str) -> str:
        for assignment in self.assignments:
            if assignment.variable.name == variable_name:
                return assignment.entity.value
        raise KeyError(variable_name)

    @property
    def digest(self) -> str:
        return self._digest_cache


def canonicalize_binding_context(
    state: State,
    goal: Goal,
    binding: OperatorBinding,
) -> tuple[State, Goal]:
    """Express one task in binding-relative role coordinates.

    The transformation is name-independent and contains no domain rule.  All
    concrete values selected by a binding become their learned variable roles;
    every remaining value receives a deterministic local placeholder shared by
    the state and goal.  Neural scorers can therefore learn relational facts
    such as occupancy or co-reference without memorizing entity spellings.
    """

    if not isinstance(state, State) or not isinstance(goal, Goal):
        raise TypeError("binding context requires a State and Goal")
    if not isinstance(binding, OperatorBinding):
        raise TypeError("binding context requires an OperatorBinding")
    if state.namespace != goal.namespace or binding.operator.namespace != state.namespace:
        raise ValueError("binding context must remain in one namespace")

    roles_by_value: dict[str, list[str]] = {}
    for assignment in binding.assignments:
        roles_by_value.setdefault(assignment.entity.value, []).append(
            assignment.variable.name
        )
    values = sorted(
        {
            argument
            for record in state.records + goal.required + goal.forbidden
            for argument in record.arguments
        }
    )
    unbound = {
        value: f"unbound:{index}"
        for index, value in enumerate(
            item for item in values if item not in roles_by_value
        )
    }

    def replace(value: str) -> str:
        roles = roles_by_value.get(value)
        if roles:
            return "roles:" + "+".join(sorted(roles))
        return unbound[value]

    def rewrite(record: Record) -> Record:
        return Record(
            record.predicate,
            tuple(replace(argument) for argument in record.arguments),
        )

    return (
        State.from_records(state.namespace, (rewrite(item) for item in state.records)),
        Goal.from_records(
            goal.namespace,
            (rewrite(item) for item in goal.required),
            forbidden=(rewrite(item) for item in goal.forbidden),
            exact=goal.exact,
        ),
    )


def enumerate_operator_bindings(
    operator: LearnedOperator,
    candidates: Sequence[TypedEntityCandidate],
    *,
    maximum_bindings: int = 1024,
    allow_entity_reuse: bool = True,
) -> tuple[OperatorBinding, ...]:
    """Enumerate complete bindings from typed candidates, with a hard ceiling.

    The function knows only declared variable types.  It applies no domain
    predicates, preconditions, or preferred entity ordering.
    """

    if not isinstance(operator, LearnedOperator):
        raise TypeError("operator must be a LearnedOperator")
    if (
        isinstance(maximum_bindings, bool)
        or not isinstance(maximum_bindings, int)
        or maximum_bindings <= 0
    ):
        raise ValueError("maximum_bindings must be a positive integer")
    if not isinstance(allow_entity_reuse, bool):
        raise TypeError("allow_entity_reuse must be bool")
    if any(not isinstance(candidate, TypedEntityCandidate) for candidate in candidates):
        raise TypeError("candidates must contain only TypedEntityCandidate values")
    unique = tuple(sorted(set(candidates)))
    by_type: dict[str, tuple[TypedEntityCandidate, ...]] = {}
    for type_name in {candidate.type_name for candidate in unique}:
        by_type[type_name] = tuple(
            candidate for candidate in unique if candidate.type_name == type_name
        )
    pools = tuple(by_type.get(variable.type_name, ()) for variable in operator.variables)
    if any(not pool for pool in pools):
        return ()
    if not pools:
        return (OperatorBinding(operator, ()),)

    result: list[OperatorBinding] = []
    for values in itertools.product(*pools):
        if not allow_entity_reuse and len(set(values)) != len(values):
            continue
        assignments = tuple(
            BindingAssignment(variable, entity)
            for variable, entity in zip(operator.variables, values, strict=True)
        )
        result.append(OperatorBinding(operator, assignments))
        if len(result) > maximum_bindings:
            raise ValueError("typed binding enumeration exceeds maximum_bindings")
    return tuple(result)


class BindingEncoder(nn.Module):
    """Encode novel bindings structurally rather than through a lookup ID."""

    def __init__(
        self,
        *,
        hash_width: int,
        hidden_width: int,
        output_width: int,
    ) -> None:
        super().__init__()
        self.features = FrozenHashTextEncoder(hash_width)
        self.output_width = output_width
        self.projection = nn.Sequential(
            nn.LayerNorm(hash_width),
            nn.Linear(hash_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, bindings: Sequence[OperatorBinding]) -> torch.Tensor:
        if not bindings:
            raise ValueError("at least one binding is required")
        if any(not isinstance(binding, OperatorBinding) for binding in bindings):
            raise TypeError("bindings must contain only OperatorBinding values")
        payloads = []
        for binding in bindings:
            # Operator identity is deliberately absent.  The operator schema
            # has its own structural encoder; this channel represents roles,
            # types, and values so unseen assignments remain representable.
            payloads.append(
                json.dumps(
                    {
                        "assignments": [
                            {
                                "role": item.variable.name,
                                "type": item.variable.type_name,
                                "value": item.entity.value,
                            }
                            for item in binding.assignments
                        ]
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        reference = next(self.projection.parameters())
        raw = self.features.encode_texts(
            payloads,
            device=reference.device,
            dtype=reference.dtype,
        )
        return self.projection(raw)


class BindingCandidateComposer(nn.Module):
    """Compose schema and binding features into one reusable candidate tensor."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self.network = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        operator_embeddings: torch.Tensor,
        binding_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if operator_embeddings.shape != binding_embeddings.shape:
            raise ValueError("operator and binding embeddings must share shape")
        if (
            operator_embeddings.ndim not in (2, 3)
            or operator_embeddings.shape[-1] != self.width
        ):
            raise ValueError("candidate embeddings must have rank two or three")
        if (
            operator_embeddings.device != binding_embeddings.device
            or operator_embeddings.dtype != binding_embeddings.dtype
        ):
            raise ValueError("candidate embeddings must share device and dtype")
        return self.network(
            torch.cat(
                (
                    operator_embeddings,
                    binding_embeddings,
                    operator_embeddings * binding_embeddings,
                ),
                dim=-1,
            )
        )


class GoalConditionedBindingProposer(nn.Module):
    """Score structurally composed operator-binding candidates for a goal."""

    def __init__(
        self,
        width: int,
        *,
        composer: BindingCandidateComposer | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.composer = composer or BindingCandidateComposer(width)
        self.query = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        self.keys = nn.Linear(width, width, bias=False)
        self.pair_bias = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def compose_candidates(
        self,
        operator_embeddings: torch.Tensor,
        binding_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.composer(operator_embeddings, binding_embeddings)

    def forward(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        operator_embeddings: torch.Tensor,
        binding_embeddings: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            states.ndim != 2
            or goals.shape != states.shape
            or states.shape[-1] != self.width
        ):
            raise ValueError("states and goals must share shape [batch, width]")
        candidates = self.compose_candidates(
            operator_embeddings,
            binding_embeddings,
        )
        return self.score_candidates(states, goals, candidates, mask=mask)

    def score_candidates(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        candidates: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score an already composed structural candidate tensor."""

        if (
            states.ndim != 2
            or goals.shape != states.shape
            or states.shape[-1] != self.width
        ):
            raise ValueError("states and goals must share shape [batch, width]")
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0).expand(states.shape[0], -1, -1)
        if (
            candidates.ndim != 3
            or candidates.shape[0] != states.shape[0]
            or candidates.shape[-1] != self.width
            or candidates.shape[1] <= 0
        ):
            raise ValueError(
                "candidates must have shape [count, width] or "
                "[batch, count, width]"
            )
        if (
            candidates.device != states.device
            or candidates.dtype != states.dtype
            or goals.device != states.device
            or goals.dtype != states.dtype
        ):
            raise ValueError("proposal tensors must share device and dtype")
        context = torch.cat((states, goals, goals - states), dim=-1)
        logits = (
            self.query(context).unsqueeze(1) * self.keys(candidates)
        ).sum(dim=-1) / math.sqrt(self.width)
        pair = torch.cat(
            (
                context.unsqueeze(1).expand(-1, candidates.shape[1], -1),
                candidates,
            ),
            dim=-1,
        )
        logits = logits + self.pair_bias(pair).squeeze(-1)
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("binding candidate mask must be boolean")
            if mask.shape == (candidates.shape[1],):
                mask = mask.unsqueeze(0).expand_as(logits)
            if mask.shape != logits.shape or mask.device != logits.device:
                raise ValueError("binding candidate mask does not match logits")
            logits = logits.masked_fill(~mask, -torch.inf)
        return logits


@dataclass(frozen=True, slots=True)
class BindingConditionedPredictions:
    candidate_embeddings: torch.Tensor
    initiation_logits: torch.Tensor
    effect_embeddings: torch.Tensor
    predecessor_embeddings: torch.Tensor
    termination_logits: torch.Tensor
    proposer_logits: torch.Tensor


class BindingConditionedOperatorHeads(nn.Module):
    """Reuse one NeuralOperatorCore while distinguishing unseen bindings."""

    def __init__(
        self,
        core: NeuralOperatorCore,
        *,
        binding_hash_width: int = 192,
        hidden_width: int = 192,
    ) -> None:
        super().__init__()
        if not isinstance(core, NeuralOperatorCore):
            raise TypeError("core must be a NeuralOperatorCore")
        self.core = core
        self.binding_encoder = BindingEncoder(
            hash_width=binding_hash_width,
            hidden_width=hidden_width,
            output_width=core.width,
        )
        composer = BindingCandidateComposer(core.width)
        self.binding_proposer = GoalConditionedBindingProposer(
            core.width,
            composer=composer,
        )

    @property
    def composer(self) -> BindingCandidateComposer:
        """The one composer shared by prediction and proposal paths."""

        return self.binding_proposer.composer

    def encode_candidates(
        self,
        bindings: Sequence[OperatorBinding],
    ) -> torch.Tensor:
        if not bindings:
            raise ValueError("at least one operator binding is required")
        operators = self.core.encode_operators(
            tuple(binding.operator for binding in bindings)
        )
        binding_features = self.binding_encoder(bindings)
        return self.binding_proposer.compose_candidates(
            operators,
            binding_features,
        )

    def forward(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        bindings: Sequence[OperatorBinding],
    ) -> BindingConditionedPredictions:
        candidates = self.encode_candidates(bindings)
        initiation = self.core.initiation_logits(states, candidates)
        effects = self.core.predict_effects(states, candidates)
        predecessors = self.core.predict_effects(states, candidates, reverse=True)
        termination = self.core.termination_logits(effects, goals)
        proposal = self.binding_proposer.score_candidates(
            states,
            goals,
            candidates,
        )
        return BindingConditionedPredictions(
            candidate_embeddings=candidates,
            initiation_logits=initiation,
            effect_embeddings=effects,
            predecessor_embeddings=predecessors,
            termination_logits=termination,
            proposer_logits=proposal,
        )


@dataclass(frozen=True, slots=True)
class PrimitiveStepScores:
    """Action/STOP logits and typed per-argument entity pointer logits."""

    action_logits: torch.Tensor
    argument_logits: torch.Tensor
    argument_mask: torch.Tensor
    stop_index: int


@dataclass(frozen=True, slots=True)
class DecodedPrimitiveSequence:
    """Ground actions proposed for external commit and execution."""

    actions: tuple[GroundAction, ...]
    stopped: bool
    steps_scored: int


class SharedPrimitiveSequenceDecoder(nn.Module):
    """Shared neural decoder over structural actions, bindings, and entities."""

    def __init__(
        self,
        *,
        width: int = 128,
        hidden_width: int = 192,
        hash_width: int = 192,
        maximum_steps: int = 16,
    ) -> None:
        super().__init__()
        if width <= 0 or hidden_width <= 0 or maximum_steps <= 0:
            raise ValueError("decoder widths and maximum_steps must be positive")
        self.width = width
        self.maximum_steps = maximum_steps
        self.schema_encoder = SchemaEncoder(
            hash_width=hash_width,
            hidden_width=hidden_width,
            output_width=width,
        )
        self.binding_encoder = BindingEncoder(
            hash_width=hash_width,
            hidden_width=hidden_width,
            output_width=width,
        )
        self.step_projection = nn.Sequential(
            nn.Linear(4, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(width * 6),
            nn.Linear(width * 6, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.action_query = nn.Linear(width, width, bias=False)
        self.action_key = nn.Linear(width, width, bias=False)
        self.stop_key = nn.Parameter(torch.empty(width))
        self.argument_query = nn.Linear(width, width, bias=False)
        self.action_to_argument = nn.Linear(width, width, bias=False)
        self.parameter_to_argument = nn.Linear(width, width, bias=False)
        self.entity_key = nn.Linear(width, width, bias=False)
        self.choice_projection = nn.Linear(width * 2, width)
        self.history_cell = nn.GRUCell(width, width)
        nn.init.normal_(self.stop_key, mean=0.0, std=1.0 / math.sqrt(width))

    def initial_history(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = next(self.context.parameters())
        return torch.zeros(
            (batch_size, self.width),
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def forward(
        self,
        state_embeddings: torch.Tensor,
        goal_embeddings: torch.Tensor,
        bindings: Sequence[OperatorBinding],
        action_schemas: Sequence[ActionSchema],
        entity_candidates: Sequence[TypedEntityCandidate],
        steps: torch.Tensor,
        *,
        history: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> PrimitiveStepScores:
        batch = len(bindings)
        if batch <= 0 or not action_schemas:
            raise ValueError("bindings and action_schemas must not be empty")
        if state_embeddings.shape != (batch, self.width) or goal_embeddings.shape != (
            batch,
            self.width,
        ):
            raise ValueError("state and goal embeddings must have shape [batch, width]")
        if steps.shape != (batch,) or steps.dtype != torch.long:
            raise ValueError("steps must be torch.long with shape [batch]")
        if bool(((steps < 0) | (steps >= self.maximum_steps)).any().item()):
            raise ValueError("steps are outside the decoder horizon")
        if any(not isinstance(schema, ActionSchema) for schema in action_schemas):
            raise TypeError("action_schemas must contain only ActionSchema values")
        if any(
            not isinstance(candidate, TypedEntityCandidate)
            for candidate in entity_candidates
        ):
            raise TypeError(
                "entity_candidates must contain only TypedEntityCandidate values"
            )
        if history is None:
            history = self.initial_history(
                batch,
                device=state_embeddings.device,
                dtype=state_embeddings.dtype,
            )
        if history.shape != state_embeddings.shape:
            raise ValueError("history must share the state embedding shape")
        for tensor in (goal_embeddings, history, steps):
            if tensor.device != state_embeddings.device:
                raise ValueError("decoder tensors must share one device")
        if goal_embeddings.dtype != state_embeddings.dtype or history.dtype != state_embeddings.dtype:
            raise ValueError("decoder floating tensors must share one dtype")

        binding_features = self.binding_encoder(bindings)
        operator_features = self.schema_encoder(
            tuple(binding.operator for binding in bindings)
        )
        action_features = self.schema_encoder(action_schemas)
        step_features = self.step_projection(self._step_features(steps, state_embeddings))
        context = self.context(
            torch.cat(
                (
                    state_embeddings,
                    goal_embeddings,
                    operator_features,
                    binding_features,
                    step_features,
                    history,
                ),
                dim=-1,
            )
        )

        action_keys = torch.cat(
            (action_features, self.stop_key.unsqueeze(0)),
            dim=0,
        )
        action_logits = (
            self.action_query(context).unsqueeze(1)
            * self.action_key(action_keys).unsqueeze(0)
        ).sum(dim=-1) / math.sqrt(self.width)

        argument_mask = self._argument_mask(action_schemas, entity_candidates, context.device)
        arities = torch.tensor(
            [len(schema.parameters) for schema in action_schemas],
            device=context.device,
        )
        feasible = torch.ones(len(action_schemas), dtype=torch.bool, device=context.device)
        for action_index, arity in enumerate(arities.tolist()):
            if arity:
                feasible[action_index] = bool(
                    argument_mask[action_index, :arity].any(dim=-1).all().item()
                )
        if action_mask is not None:
            if action_mask.shape != (len(action_schemas),) or action_mask.dtype != torch.bool:
                raise ValueError("action_mask must be boolean with one row per action")
            if action_mask.device != context.device:
                raise ValueError("action_mask must share the decoder device")
            feasible = feasible & action_mask
        complete_action_mask = torch.cat(
            (feasible, torch.ones(1, dtype=torch.bool, device=context.device))
        )
        action_logits = action_logits.masked_fill(
            ~complete_action_mask.unsqueeze(0),
            -torch.inf,
        )

        max_parameters = argument_mask.shape[1]
        entity_count = len(entity_candidates)
        if max_parameters == 0 or entity_count == 0:
            argument_logits = torch.empty(
                (batch, len(action_schemas), max_parameters, entity_count),
                device=context.device,
                dtype=context.dtype,
            )
        else:
            parameter_features = self._parameter_features(action_schemas)
            entity_features = self._binding_entity_features(
                bindings,
                entity_candidates,
            )
            query = self.argument_query(context)[:, None, None, :]
            query = query + self.action_to_argument(action_features)[None, :, None, :]
            query = query + self.parameter_to_argument(parameter_features)[None, :, :, :]
            keys = self.entity_key(entity_features)[:, None, None, :, :]
            argument_logits = (query.unsqueeze(3) * keys).sum(dim=-1)
            argument_logits = argument_logits / math.sqrt(self.width)
            argument_logits = argument_logits.masked_fill(
                ~argument_mask.unsqueeze(0),
                -torch.inf,
            )
        return PrimitiveStepScores(
            action_logits=action_logits,
            argument_logits=argument_logits,
            argument_mask=argument_mask,
            stop_index=len(action_schemas),
        )

    def select_greedy(
        self,
        scores: PrimitiveStepScores,
        action_schemas: Sequence[ActionSchema],
        entity_candidates: Sequence[TypedEntityCandidate],
        *,
        batch_index: int = 0,
    ) -> GroundAction | None:
        """Turn pointer scores into a typed proposal; never execute it."""

        action_index, argument_indices = self._select_indices(
            scores,
            action_schemas,
            entity_candidates,
            batch_index=batch_index,
        )
        if action_index == scores.stop_index:
            return None
        arguments = tuple(
            entity_candidates[index].value for index in argument_indices
        )
        return action_schemas[action_index].ground(*arguments)

    def decode_sequence_greedy(
        self,
        state_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
        binding: OperatorBinding,
        action_schemas: Sequence[ActionSchema],
        entity_candidates: Sequence[TypedEntityCandidate],
        *,
        maximum_steps: int | None = None,
    ) -> DecodedPrimitiveSequence:
        """Decode one bounded sequence of GroundActions for external commit."""

        if state_embedding.shape != (self.width,) or goal_embedding.shape != (
            self.width,
        ):
            raise ValueError("single-example embeddings must have shape [width]")
        limit = self.maximum_steps if maximum_steps is None else maximum_steps
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.maximum_steps:
            raise ValueError("maximum_steps is outside the decoder horizon")
        history = self.initial_history(
            1,
            device=state_embedding.device,
            dtype=state_embedding.dtype,
        )
        actions: list[GroundAction] = []
        for step in range(limit):
            scores = self(
                state_embedding.unsqueeze(0),
                goal_embedding.unsqueeze(0),
                (binding,),
                action_schemas,
                entity_candidates,
                torch.tensor([step], device=state_embedding.device, dtype=torch.long),
                history=history,
            )
            action_index, argument_indices = self._select_indices(
                scores,
                action_schemas,
                entity_candidates,
                batch_index=0,
            )
            if action_index == scores.stop_index:
                return DecodedPrimitiveSequence(tuple(actions), True, step + 1)
            action = action_schemas[action_index].ground(
                *(entity_candidates[index].value for index in argument_indices)
            )
            actions.append(action)
            history = self.advance_history(
                history,
                action_index,
                argument_indices,
                action_schemas,
                entity_candidates,
            )
        return DecodedPrimitiveSequence(tuple(actions), False, limit)

    def advance_history(
        self,
        history: torch.Tensor,
        action_index: int,
        argument_indices: tuple[int, ...],
        action_schemas: Sequence[ActionSchema],
        entity_candidates: Sequence[TypedEntityCandidate],
    ) -> torch.Tensor:
        """Advance teacher-forced decoder state without executing an action."""

        if history.shape != (1, self.width):
            raise ValueError("teacher-forced history must have shape [1, width]")
        if isinstance(action_index, bool) or not isinstance(action_index, int):
            raise TypeError("action_index must be an integer")
        if not 0 <= action_index < len(action_schemas):
            raise IndexError("action_index is outside the action schema set")
        if not isinstance(argument_indices, tuple):
            raise TypeError("argument_indices must be an immutable tuple")
        schema = action_schemas[action_index]
        if len(argument_indices) != len(schema.parameters):
            raise ValueError("argument indices must match the selected schema arity")
        for position, candidate_index in enumerate(argument_indices):
            if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
                raise TypeError("argument indices must contain integers")
            if not 0 <= candidate_index < len(entity_candidates):
                raise IndexError("argument candidate index is outside the entity set")
            if (
                entity_candidates[candidate_index].type_name
                != schema.parameters[position].type_name
            ):
                raise ValueError("teacher-forced argument has the wrong declared type")
        return self._advance_history(
            history,
            action_index,
            argument_indices,
            action_schemas,
            entity_candidates,
        )

    def _advance_history(
        self,
        history: torch.Tensor,
        action_index: int,
        argument_indices: tuple[int, ...],
        action_schemas: Sequence[ActionSchema],
        entity_candidates: Sequence[TypedEntityCandidate],
    ) -> torch.Tensor:
        action = self.schema_encoder(action_schemas)[action_index]
        if argument_indices:
            entities = self._entity_features(entity_candidates)
            argument = entities[list(argument_indices)].mean(dim=0)
        else:
            argument = torch.zeros_like(action)
        choice = self.choice_projection(torch.cat((action, argument))).unsqueeze(0)
        return self.history_cell(choice, history)

    @staticmethod
    def _select_indices(
        scores: PrimitiveStepScores,
        action_schemas: Sequence[ActionSchema],
        entity_candidates: Sequence[TypedEntityCandidate],
        *,
        batch_index: int,
    ) -> tuple[int, tuple[int, ...]]:
        if not 0 <= batch_index < scores.action_logits.shape[0]:
            raise IndexError("batch_index is outside the score batch")
        if scores.stop_index != len(action_schemas):
            raise ValueError("score STOP index does not match action schemas")
        action_index = int(scores.action_logits[batch_index].argmax().item())
        if action_index == scores.stop_index:
            return action_index, ()
        schema = action_schemas[action_index]
        indices: list[int] = []
        for position in range(len(schema.parameters)):
            logits = scores.argument_logits[batch_index, action_index, position]
            if not bool(torch.isfinite(logits).any().item()):
                raise RuntimeError("selected action has no typed argument candidate")
            index = int(logits.argmax().item())
            if index >= len(entity_candidates):
                raise RuntimeError("argument pointer selected an invalid candidate")
            indices.append(index)
        return action_index, tuple(indices)

    def _parameter_features(
        self,
        schemas: Sequence[ActionSchema],
    ) -> torch.Tensor:
        maximum = max((len(schema.parameters) for schema in schemas), default=0)
        if maximum == 0:
            reference = next(self.parameters())
            return torch.empty(
                (len(schemas), 0, self.width),
                device=reference.device,
                dtype=reference.dtype,
            )
        texts = []
        for schema in schemas:
            for position in range(maximum):
                if position < len(schema.parameters):
                    parameter = schema.parameters[position]
                    texts.append(
                        json.dumps(
                            {
                                "position": position,
                                "name": parameter.name,
                                "type_name": parameter.type_name,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                else:
                    texts.append("padding-parameter")
        return self.schema_encoder.encode_values(texts).view(
            len(schemas),
            maximum,
            self.width,
        )

    def _entity_features(
        self,
        candidates: Sequence[TypedEntityCandidate],
    ) -> torch.Tensor:
        return self.schema_encoder.encode_values(
            [
                json.dumps(
                    {"type_name": item.type_name, "value": item.value},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for item in candidates
            ]
        )

    def _binding_entity_features(
        self,
        bindings: Sequence[OperatorBinding],
        candidates: Sequence[TypedEntityCandidate],
    ) -> torch.Tensor:
        """Encode copy candidates together with their selected binding roles.

        Concrete entity spellings can be entirely novel at inference.  The
        decoder therefore receives the abstract roles occupied by each value
        in the selected binding and learns a pointer over those roles.  Values
        that are not assigned remain available for constants or future
        observations, but carry an empty role set.
        """

        payloads: list[str] = []
        for binding in bindings:
            roles_by_entity: dict[tuple[str, str], list[str]] = {}
            for assignment in binding.assignments:
                key = (
                    assignment.entity.type_name,
                    assignment.entity.value,
                )
                roles_by_entity.setdefault(key, []).append(
                    assignment.variable.name
                )
            for candidate in candidates:
                roles = tuple(
                    sorted(
                        roles_by_entity.get(
                            (candidate.type_name, candidate.value),
                            (),
                        )
                    )
                )
                payloads.append(
                    json.dumps(
                        {
                            "binding_roles": roles,
                            "type_name": candidate.type_name,
                            "value": candidate.value,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        return self.schema_encoder.encode_values(payloads).view(
            len(bindings),
            len(candidates),
            self.width,
        )

    @staticmethod
    def _argument_mask(
        schemas: Sequence[ActionSchema],
        candidates: Sequence[TypedEntityCandidate],
        device: torch.device,
    ) -> torch.Tensor:
        maximum = max((len(schema.parameters) for schema in schemas), default=0)
        mask = torch.zeros(
            (len(schemas), maximum, len(candidates)),
            dtype=torch.bool,
            device=device,
        )
        for action_index, schema in enumerate(schemas):
            for position, parameter in enumerate(schema.parameters):
                for candidate_index, candidate in enumerate(candidates):
                    mask[action_index, position, candidate_index] = (
                        parameter.type_name == candidate.type_name
                    )
        return mask

    def _step_features(
        self,
        steps: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        scaled = steps.to(dtype=reference.dtype) / max(1, self.maximum_steps - 1)
        return torch.stack(
            (
                scaled,
                scaled.square(),
                torch.sin(math.pi * scaled),
                torch.cos(math.pi * scaled),
            ),
            dim=-1,
        )


__all__ = [
    "BindingAssignment",
    "BindingCandidateComposer",
    "BindingConditionedOperatorHeads",
    "BindingConditionedPredictions",
    "BindingEncoder",
    "DecodedPrimitiveSequence",
    "GoalConditionedBindingProposer",
    "OperatorBinding",
    "PrimitiveStepScores",
    "SharedPrimitiveSequenceDecoder",
    "TypedEntityCandidate",
    "canonicalize_binding_context",
    "enumerate_operator_bindings",
]
