"""Typed public-relation encoding for procedural candidates.

The boundary is deliberately narrow: deterministic code parses declared
entities and relations, while a shared neural message-passing core learns how
those relations matter.  It contains no path search, procedure construction,
candidate rule, evaluator identity, or answer lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import torch
from torch import nn

from .recurrent_core import ReasoningCoreConfig, RecurrentReasoningCore
from .action_trace_matching import ActionTraceMatcherDecoder, ActionTraceProceduralCore
from .candidate_procedure_decoder import CandidateProcedureDecode
from .scalable_procedural_core import PlasticProcedureState, ProceduralCoreOutput


_QUERY = re.compile(
    r"^origin=(?P<origin>[^;]*); goal=(?P<goal>[^;]*); "
    r"forbidden=(?P<forbidden>[^;]*); budget=(?P<budget>[1-9][0-9]*)$"
)
_ACTION = re.compile(
    r"^(?P<label>[A-Z]) in=(?P<input>\S+) out=(?P<output>\S+) "
    r"reads=(?P<reads>\S*) writes=(?P<writes>\S*) graph=(?P<graph>\S+)$"
)
_ATOM = re.compile(r"^[ST][0-9]+$")
_NODE = re.compile(r"^[0-9]+$")

_ENTITY_KINDS = ("candidate", "state", "type", "topology")
_RELATION_KINDS = (
    "origin",
    "goal",
    "forbidden",
    "input_type",
    "output_type",
    "reads",
    "writes",
    "topology_member",
    "topology_edge",
)
_MENTION_ROLES = ("unary", "source", "target")
FEATURE_WIDTH = 16


@dataclass(frozen=True, slots=True)
class PublicRelationComponent:
    """One task-local candidate expressed only through public relations."""

    label: str
    input_type: str
    output_type: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    topology_edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PublicProcedureRelations:
    """Cognee-inspired typed graph interchange owned by Angler."""

    origin: tuple[str, ...]
    goal: tuple[str, ...]
    forbidden: tuple[str, ...]
    budget: int
    components: tuple[PublicRelationComponent, ...]


@dataclass(frozen=True, slots=True)
class RelationalIncidenceTensors:
    """One unpadded typed incidence graph and its candidate entity indices."""

    fact_features: torch.Tensor
    fact_mask: torch.Tensor
    entity_features: torch.Tensor
    entity_mask: torch.Tensor
    mention_features: torch.Tensor
    mention_mask: torch.Tensor
    mention_fact_indices: torch.Tensor
    mention_entity_indices: torch.Tensor
    candidate_entity_indices: torch.Tensor


def _csv(value: str, *, atom: re.Pattern[str]) -> tuple[str, ...]:
    if value in ("", "none"):
        return ()
    values = tuple(value.split(","))
    if any(not atom.fullmatch(item) for item in values) or len(set(values)) != len(values):
        raise ValueError("public relation list contains an invalid or duplicate symbol")
    return values


def parse_public_relations(
    query_text: str,
    action_texts: Iterable[str],
) -> PublicProcedureRelations:
    """Parse the frozen public text grammar without interpreting a solution."""

    first = query_text.splitlines()[0]
    match = _QUERY.fullmatch(first)
    if match is None:
        raise ValueError("query text does not match the public relation grammar")
    components = []
    labels = set()
    for action_text in action_texts:
        action = _ACTION.fullmatch(action_text)
        if action is None:
            raise ValueError("action text does not match the public relation grammar")
        label = action.group("label")
        if label in labels:
            raise ValueError("candidate labels must be unique")
        labels.add(label)
        input_type = action.group("input")
        output_type = action.group("output")
        if not _ATOM.fullmatch(input_type) or not _ATOM.fullmatch(output_type):
            raise ValueError("public type symbols are invalid")
        edges = []
        for edge in action.group("graph").split(","):
            parts = edge.split(">")
            if len(parts) != 2 or any(not _NODE.fullmatch(item) for item in parts):
                raise ValueError("public topology edge is invalid")
            edges.append((parts[0], parts[1]))
        if len(set(edges)) != len(edges):
            raise ValueError("public topology edges must be unique")
        components.append(
            PublicRelationComponent(
                label=label,
                input_type=input_type,
                output_type=output_type,
                reads=_csv(action.group("reads"), atom=_ATOM),
                writes=_csv(action.group("writes"), atom=_ATOM),
                topology_edges=tuple(edges),
            )
        )
    if not components:
        raise ValueError("a public procedure graph requires candidates")
    return PublicProcedureRelations(
        origin=_csv(match.group("origin"), atom=_ATOM),
        goal=_csv(match.group("goal"), atom=_ATOM),
        forbidden=_csv(match.group("forbidden"), atom=_ATOM),
        budget=int(match.group("budget")),
        components=tuple(components),
    )


def _one_hot(index: int) -> list[float]:
    row = [0.0] * FEATURE_WIDTH
    row[index] = 1.0
    return row


def build_relational_incidence(
    graph: PublicProcedureRelations,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    include_topology_edges: bool = True,
) -> RelationalIncidenceTensors:
    """Turn typed equality/incidence into tensors; never infer a procedure."""

    if not isinstance(graph, PublicProcedureRelations):
        raise TypeError("graph must be PublicProcedureRelations")
    entity_rows: list[list[float]] = []
    entity_indices: dict[tuple[str, ...], int] = {}

    def entity(key: tuple[str, ...], kind: str) -> int:
        existing = entity_indices.get(key)
        if existing is not None:
            return existing
        index = len(entity_rows)
        entity_indices[key] = index
        entity_rows.append(_one_hot(_ENTITY_KINDS.index(kind)))
        return index

    candidate_indices = [
        entity(("candidate", component.label), "candidate")
        for component in graph.components
    ]
    states = sorted(
        set(graph.origin)
        | set(graph.goal)
        | set(graph.forbidden)
        | {item for component in graph.components for item in (*component.reads, *component.writes)}
    )
    types = sorted(
        {item for component in graph.components for item in (component.input_type, component.output_type)}
    )
    state_indices = {name: entity(("state", name), "state") for name in states}
    type_indices = {name: entity(("type", name), "type") for name in types}

    fact_rows: list[list[float]] = []
    mention_rows: list[list[float]] = []
    mention_facts: list[int] = []
    mention_entities: list[int] = []

    def fact(kind: str, endpoints: tuple[tuple[str, int], ...]) -> None:
        fact_index = len(fact_rows)
        fact_rows.append(_one_hot(_RELATION_KINDS.index(kind)))
        for role, entity_index in endpoints:
            mention_rows.append(_one_hot(_MENTION_ROLES.index(role)))
            mention_facts.append(fact_index)
            mention_entities.append(entity_index)

    for name in graph.origin:
        fact("origin", (("unary", state_indices[name]),))
    for name in graph.goal:
        fact("goal", (("unary", state_indices[name]),))
    for name in graph.forbidden:
        fact("forbidden", (("unary", state_indices[name]),))

    for component, candidate_index in zip(graph.components, candidate_indices, strict=True):
        fact(
            "input_type",
            (("source", type_indices[component.input_type]), ("target", candidate_index)),
        )
        fact(
            "output_type",
            (("source", candidate_index), ("target", type_indices[component.output_type])),
        )
        for name in component.reads:
            fact("reads", (("source", state_indices[name]), ("target", candidate_index)))
        for name in component.writes:
            fact("writes", (("source", candidate_index), ("target", state_indices[name])))
        local_nodes = sorted({item for edge in component.topology_edges for item in edge})
        local_indices = {
            name: entity(("topology", component.label, name), "topology")
            for name in local_nodes
        }
        for node_index in local_indices.values():
            fact("topology_member", (("source", candidate_index), ("target", node_index)))
        if include_topology_edges:
            for source, target in component.topology_edges:
                fact(
                    "topology_edge",
                    (("source", local_indices[source]), ("target", local_indices[target])),
                )

    if not fact_rows or not mention_rows:
        raise ValueError("public procedure graph produced no relational incidence")
    target = torch.device(device)
    return RelationalIncidenceTensors(
        fact_features=torch.tensor([fact_rows], device=target, dtype=dtype),
        fact_mask=torch.ones((1, len(fact_rows)), device=target, dtype=torch.bool),
        entity_features=torch.tensor([entity_rows], device=target, dtype=dtype),
        entity_mask=torch.ones((1, len(entity_rows)), device=target, dtype=torch.bool),
        mention_features=torch.tensor([mention_rows], device=target, dtype=dtype),
        mention_mask=torch.ones((1, len(mention_rows)), device=target, dtype=torch.bool),
        mention_fact_indices=torch.tensor([mention_facts], device=target, dtype=torch.long),
        mention_entity_indices=torch.tensor([mention_entities], device=target, dtype=torch.long),
        candidate_entity_indices=torch.tensor(candidate_indices, device=target, dtype=torch.long),
    )


class StructuredRelationalEncoder(nn.Module):
    """Learn candidate representations from a typed, rename-free graph."""

    def __init__(
        self,
        *,
        width: int = 128,
        reasoning_steps: int = 6,
        workspace_slots: int = 8,
        maximum_entities: int = 256,
    ) -> None:
        super().__init__()
        self.core = RecurrentReasoningCore(
            ReasoningCoreConfig(
                knowledge_width=FEATURE_WIDTH,
                core_width=width,
                workspace_slots=workspace_slots,
                attention_heads=4,
                feedforward_width=4 * width,
                reasoning_steps=reasoning_steps,
                maximum_reasoning_steps=reasoning_steps,
                maximum_entities=maximum_entities,
            )
        )
        self.width = width

    def forward(
        self,
        tensors: RelationalIncidenceTensors,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entities, workspace = self.core.encode(
            tensors.fact_features,
            tensors.fact_mask,
            tensors.entity_features,
            tensors.entity_mask,
            tensors.mention_features,
            tensors.mention_mask,
            tensors.mention_fact_indices,
            tensors.mention_entity_indices,
        )
        candidates = entities.index_select(1, tensors.candidate_entity_indices)
        return candidates, workspace


class SemanticRelationalFusion(nn.Module):
    """Add a learned relational residual to detached foundation features."""

    def __init__(self, *, semantic_width: int, relational_width: int, hidden_width: int) -> None:
        super().__init__()
        self.semantic_width = semantic_width
        self.relational_width = relational_width
        self.relational_residual = nn.Sequential(
            nn.LayerNorm(relational_width),
            nn.Linear(relational_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, semantic_width),
        )
        nn.init.zeros_(self.relational_residual[-1].weight)
        nn.init.zeros_(self.relational_residual[-1].bias)

    def forward(
        self,
        semantic: torch.Tensor,
        relational: torch.Tensor,
        *,
        include_semantic: bool = True,
        include_relational: bool = True,
    ) -> torch.Tensor:
        if semantic.ndim != 3 or semantic.shape[-1] != self.semantic_width:
            raise ValueError("semantic features have the wrong shape")
        if relational.shape != (*semantic.shape[:-1], self.relational_width):
            raise ValueError("relational features must align with semantic items")
        if semantic.device != relational.device or semantic.dtype != relational.dtype:
            raise ValueError("semantic and relational features must share device and dtype")
        base = semantic if include_semantic else torch.zeros_like(semantic)
        residual = self.relational_residual(relational)
        if not include_relational:
            residual = torch.zeros_like(residual)
        return base + residual


class StructuredActionTraceProceduralCore(ActionTraceProceduralCore):
    """V6-compatible writer that permits gradient credit into graph features."""

    def apply_structured_action_trace_feedback(
        self,
        state: PlasticProcedureState,
        output: ProceduralCoreOutput,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        outcome: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> PlasticProcedureState:
        self._validate_state(state)
        reference = next(self.parameters())
        expected = (1, self.maximum_trace_steps, self.config.content_width)
        if action_features.shape != expected:
            raise ValueError("action_features must be [1, maximum_trace_steps, content_width]")
        if action_mask.dtype is not torch.bool or action_mask.shape != expected[:2]:
            raise ValueError("action_mask must be boolean [1, maximum_trace_steps]")
        if not bool(action_mask.any().item()):
            raise ValueError("the procedure trace must contain at least one action")
        if bool((action_mask & ((~action_mask).cumsum(dim=1) > 0)).any().item()):
            raise ValueError("action_mask must expose one contiguous action prefix")
        if action_features.device != reference.device or action_features.dtype != reference.dtype:
            raise ValueError("action_features must match the core device and dtype")
        if outcome.shape != (1,) or outcome.device != reference.device or outcome.dtype != reference.dtype:
            raise ValueError("outcome must match the core device and dtype")
        if not bool(((outcome == 1) | (outcome == -1)).all().item()):
            raise ValueError("outcome must be +1 success or -1 failure")

        actions = self.action_trace_projection(action_features)
        actions = actions + self.action_trace_positions.unsqueeze(0)
        current = state
        for step in range(int(action_mask.sum().item())):
            write_input = torch.cat(
                (
                    output.query_state,
                    output.procedure_summary,
                    actions[:, step],
                    outcome.unsqueeze(1),
                ),
                dim=-1,
            )
            key = self.action_write_key(write_input).squeeze(0)
            value = self.action_write_value(write_input).squeeze(0)
            gate = self.action_write_gate(write_input).squeeze()
            similarity = torch.mv(current.keys, key) / math.sqrt(self.config.model_width)
            weights = torch.softmax(similarity - 4.0 * current.strengths, dim=0)
            rates = (gate * weights).clamp(0.0, 1.0)
            current = PlasticProcedureState(
                keys=(1.0 - rates.unsqueeze(1)) * current.keys + rates.unsqueeze(1) * key,
                values=(1.0 - rates.unsqueeze(1)) * current.values + rates.unsqueeze(1) * value,
                strengths=(current.strengths + rates * (1.0 - current.strengths)).clamp(0.0, 1.0),
                step=state.step,
            )
        updated = PlasticProcedureState(
            keys=current.keys,
            values=current.values,
            strengths=current.strengths,
            step=state.step + 1,
        )
        return updated.detached_clone() if detach_state else updated


class StructuredActionTraceMatcherDecoder(ActionTraceMatcherDecoder):
    """V6 matcher that propagates procedure loss into relational features."""

    def forward(
        self,
        procedure_slots: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        memory_values: torch.Tensor,
        memory_mask: torch.Tensor,
        teacher_actions: torch.Tensor | None = None,
        correspondence: bool = True,
    ) -> CandidateProcedureDecode:
        self._validate(procedure_slots, action_features, action_mask, teacher_actions)
        reference = next(self.parameters())
        if memory_values.ndim != 2 or memory_values.shape[1] != self.procedure_width:
            raise ValueError("memory_values must be [memory_slots, procedure_width]")
        if memory_mask.dtype is not torch.bool or memory_mask.shape != (memory_values.shape[0],):
            raise ValueError("memory_mask must be boolean [memory_slots]")
        if memory_values.device != reference.device or memory_values.dtype != reference.dtype:
            raise ValueError("memory_values must match the decoder device and dtype")
        if torch.is_inference(procedure_slots):
            procedure_slots = procedure_slots.clone()
        slots = self.procedure_projection(procedure_slots)
        actions = self.action_projection(action_features)
        if correspondence and bool(memory_mask.any().item()):
            memory = self.matcher_memory_projection(self.matcher_memory_norm(memory_values))
            scores = torch.einsum("baw,sw->bas", actions, memory) / math.sqrt(self.hidden_width)
            scores = scores.masked_fill(~memory_mask.view(1, 1, -1), -torch.inf)
            aligned = torch.softmax(scores, dim=-1) @ memory.unsqueeze(0)
            actions = actions + self.matcher_update(
                torch.cat((actions, actions - aligned), dim=-1)
            )

        batch = slots.shape[0]
        hidden = torch.tanh(self.initial_hidden(slots.mean(dim=1)))
        previous = self.start_token.unsqueeze(0).expand(batch, -1)
        logits_rows, selected_rows = [], []
        for step in range(self.maximum_steps):
            query = (hidden + self.position_tokens[step]).unsqueeze(1)
            context, _ = self.slot_attention(query, slots, slots, need_weights=False)
            hidden = self.recurrent(torch.cat((previous, context.squeeze(1)), dim=-1), hidden)
            pointer = self.output_query(self.output_norm(hidden))
            action_logits = torch.einsum("bw,baw->ba", pointer, actions) / math.sqrt(self.hidden_width)
            action_logits = action_logits.masked_fill(~action_mask, -torch.inf)
            logits = torch.cat((action_logits, self.stop_score(pointer)), dim=-1)
            logits_rows.append(logits)
            selected = logits.argmax(dim=-1)
            selected_rows.append(selected)
            previous_indices = selected if teacher_actions is None else teacher_actions[:, step]
            previous = self._selected_embedding(previous_indices, actions)
        return CandidateProcedureDecode(
            logits=torch.stack(logits_rows, dim=1),
            selected_indices=torch.stack(selected_rows, dim=1),
        )


__all__ = [
    "FEATURE_WIDTH",
    "PublicProcedureRelations",
    "PublicRelationComponent",
    "RelationalIncidenceTensors",
    "SemanticRelationalFusion",
    "StructuredRelationalEncoder",
    "StructuredActionTraceMatcherDecoder",
    "StructuredActionTraceProceduralCore",
    "build_relational_incidence",
    "parse_public_relations",
]
