"""Dynamic-action neural controller for the GlyphMachine trace precursor.

The controller receives only the public typed task projection.  It learns a
causal successor representation from visible traces, retains trace and scalar
attempt evidence in one fixed-capacity typed state, and emits one bounded
autoregressive procedure over the actions declared by the current task.

There is deliberately no complete-plan candidate set.  At each step the same
network builds a soft transition-belief lattice from public evidence, sends a
learned goal signal backward through that lattice for the remaining public
budget, scores the variable action set plus STOP, makes one choice, and
advances one public state belief.  The evaluator is used only to validate the
committed public procedure; outcome judging remains outside this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import torch
from torch import nn
from torch.nn import functional as F

from angler.procedures.records import ActionSchema, GroundAction, State, Transition
from angler.procedures.trunk import FrozenHashTextEncoder
from experiments.evaluators.glyph_machine_trace_suite import (
    CommittedGlyphProcedure,
    GeneratedGlyphMachineTask,
    GlyphMachineTraceStream,
    PublicGlyphMachineTask,
    commit_glyph_procedure,
    glyph_machine_mechanism_partition,
    judge_glyph_procedure_attempt,
    make_glyph_machine_control_stream,
    make_glyph_machine_trace_stream,
)
from experiments.runners.phase5_skill_memory_stream import (
    ConditionalReversibleTransition,
)


_CHECKPOINT_VERSION = "angler.phase5-glyph-machine-trace.v3.2"
_RESULT_VERSION = "angler.phase5-glyph-machine-experiment.v3.2"
_TASK_DIGEST_DOMAIN = b"project-angler.glyph-machine.runner-task.v1\x00"
_STATE_DIGEST_DOMAIN = b"project-angler.glyph-machine.associative-state.v2\x00"
_PUBLIC_EVENT_KEY_DOMAIN = b"project-angler.glyph-machine.public-event-key.v1\x00"
_PUBLIC_SUCCESSOR_DOMAIN = b"project-angler.glyph-machine.public-successor.v1\x00"
_TRAINING_ROLLOUTS_PER_TASK = 2
_PUBLIC_ID_WORDS = 4
_MAX_REASONING_STEPS = 4
_ANCHOR_RESIDUAL_LIMIT = 0.25


@dataclass(frozen=True, slots=True)
class GlyphMachineRunProfile:
    """One architecture expressed at two resource scales."""

    name: str
    width: int
    hidden_width: int
    hash_width: int
    graph_layers: int
    graph_heads: int
    transition_rank: int
    memory_slots: int
    memory_heads: int
    memory_read_top_k: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.width,
            self.hidden_width,
            self.hash_width,
            self.graph_layers,
            self.graph_heads,
            self.transition_rank,
            self.memory_slots,
            self.memory_heads,
            self.memory_read_top_k,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("glyph-machine profile dimensions must be integers")
        if any(value <= 0 for value in integer_fields):
            raise ValueError("glyph-machine profile dimensions must be positive")
        if self.width % 2 or self.width % self.graph_heads or self.width % self.memory_heads:
            raise ValueError("profile width must be even and divisible by all heads")
        if self.transition_rank >= self.width:
            raise ValueError("transition rank must be smaller than profile width")
        if self.memory_read_top_k > self.memory_slots:
            raise ValueError("memory read top-k cannot exceed memory slots")


GLYPH_MACHINE_PROFILES: Mapping[str, GlyphMachineRunProfile] = {
    "smoke": GlyphMachineRunProfile(
        name="smoke",
        width=32,
        hidden_width=64,
        hash_width=64,
        graph_layers=1,
        graph_heads=4,
        transition_rank=8,
        memory_slots=8,
        memory_heads=4,
        memory_read_top_k=2,
    ),
    "resource_graph": GlyphMachineRunProfile(
        name="resource_graph",
        width=512,
        hidden_width=1024,
        hash_width=256,
        graph_layers=5,
        graph_heads=8,
        transition_rank=64,
        memory_slots=32,
        memory_heads=8,
        memory_read_top_k=4,
    ),
}


@dataclass(frozen=True, slots=True)
class GlyphMachineExperimentConfig:
    """Bounded train/development/final experiment dimensions."""

    profile: str
    seed: int
    train_mechanisms: int
    development_mechanisms: int
    final_mechanisms: int
    training_epochs: int
    supports_per_mechanism: int
    queries_per_mechanism: int
    observations_per_support: int
    learning_rate: float
    gradient_clip: float
    rollout_temperature: float

    def __post_init__(self) -> None:
        if self.profile not in GLYPH_MACHINE_PROFILES:
            raise ValueError("experiment profile is not registered")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("experiment seed must be a nonnegative integer")
        bounded_counts = (
            (self.train_mechanisms, 64, "train_mechanisms"),
            (self.development_mechanisms, 16, "development_mechanisms"),
            (self.final_mechanisms, 16, "final_mechanisms"),
        )
        for value, maximum, label in bounded_counts:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{label} must be between one and {maximum}")
        for value, label in (
            (self.training_epochs, "training_epochs"),
            (self.supports_per_mechanism, "supports_per_mechanism"),
            (self.queries_per_mechanism, "queries_per_mechanism"),
            (self.observations_per_support, "observations_per_support"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        for value, label in (
            (self.learning_rate, "learning_rate"),
            (self.gradient_clip, "gradient_clip"),
            (self.rollout_temperature, "rollout_temperature"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")


def default_glyph_machine_experiment_config(
    profile: str,
    *,
    seed: int = 109_001,
) -> GlyphMachineExperimentConfig:
    """Return tiny smoke or full-partition resource-graph dimensions."""

    if profile == "smoke":
        return GlyphMachineExperimentConfig(
            profile=profile,
            seed=seed,
            train_mechanisms=8,
            development_mechanisms=4,
            final_mechanisms=4,
            training_epochs=1,
            supports_per_mechanism=1,
            queries_per_mechanism=1,
            observations_per_support=1,
            learning_rate=2.0e-3,
            gradient_clip=5.0,
            rollout_temperature=1.0,
        )
    if profile == "resource_graph":
        return GlyphMachineExperimentConfig(
            profile=profile,
            seed=seed,
            train_mechanisms=64,
            development_mechanisms=16,
            final_mechanisms=16,
            training_epochs=1,
            supports_per_mechanism=2,
            queries_per_mechanism=2,
            observations_per_support=2,
            learning_rate=5.0e-4,
            gradient_clip=5.0,
            rollout_temperature=1.0,
        )
    raise ValueError(f"unknown glyph-machine profile: {profile}")


@dataclass(frozen=True, slots=True)
class GlyphAssociativeState:
    """One fixed-capacity, device-resident set of typed causal events.

    The slot ranges are statically partitioned by ``GlyphAssociativeMemory``.
    Trace and scalar-outcome writes therefore share one competence state but
    cannot evict one another.
    """

    keys: torch.Tensor
    values: torch.Tensor
    occupied: torch.Tensor
    write_counts: torch.Tensor
    public_source_action_ids: torch.Tensor
    public_successor_ids: torch.Tensor
    trace_cursor: torch.Tensor
    outcome_cursor: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.keys, torch.Tensor) or self.keys.ndim != 3:
            raise ValueError("associative keys must have shape [batch, slots, width]")
        if self.values.shape != self.keys.shape:
            raise ValueError("associative values must match key shape")
        if not self.keys.is_floating_point() or self.values.dtype != self.keys.dtype:
            raise ValueError("associative keys and values must share floating dtype")
        if self.values.device != self.keys.device:
            raise ValueError("associative keys and values must share device")
        if (
            self.occupied.shape != self.keys.shape[:2]
            or self.occupied.dtype != torch.bool
            or self.occupied.device != self.keys.device
        ):
            raise ValueError("occupied must be boolean with shape [batch, slots]")
        if (
            self.write_counts.shape != self.keys.shape[:2]
            or self.write_counts.dtype != torch.long
            or self.write_counts.device != self.keys.device
        ):
            raise ValueError("write_counts must be long with shape [batch, slots]")
        metadata_shape = (*self.keys.shape[:2], _PUBLIC_ID_WORDS)
        for label, metadata in (
            ("public_source_action_ids", self.public_source_action_ids),
            ("public_successor_ids", self.public_successor_ids),
        ):
            if (
                metadata.shape != metadata_shape
                or metadata.dtype != torch.long
                or metadata.device != self.keys.device
            ):
                raise ValueError(
                    f"{label} must be long with shape [batch, slots, {_PUBLIC_ID_WORDS}]"
                )
        for label, cursor in (
            ("trace_cursor", self.trace_cursor),
            ("outcome_cursor", self.outcome_cursor),
        ):
            if (
                cursor.shape != (self.keys.shape[0],)
                or cursor.dtype != torch.long
                or cursor.device != self.keys.device
            ):
                raise ValueError(f"{label} must be long with shape [batch]")
        if min(self.keys.shape) <= 0:
            raise ValueError("associative state dimensions must be positive")
        if not bool(torch.isfinite(self.keys).all().item()) or not bool(
            torch.isfinite(self.values).all().item()
        ):
            raise ValueError("associative state must be finite")
        if bool((self.write_counts < 0).any().item()):
            raise ValueError("associative write counts cannot be negative")
        if bool((self.trace_cursor < 0).any().item()) or bool(
            (self.outcome_cursor < 0).any().item()
        ):
            raise ValueError("associative cursors cannot be negative")
        if not torch.equal(self.occupied, self.write_counts > 0):
            raise ValueError("associative occupancy must match positive write counts")

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    @property
    def slot_count(self) -> int:
        return int(self.keys.shape[1])

    @property
    def width(self) -> int:
        return int(self.keys.shape[2])

    def numel(self) -> int:
        return sum(
            value.numel()
            for value in (
                self.keys,
                self.values,
                self.occupied,
                self.write_counts,
                self.public_source_action_ids,
                self.public_successor_ids,
                self.trace_cursor,
                self.outcome_cursor,
            )
        )


@dataclass(frozen=True, slots=True)
class GlyphAssociativeRead:
    contexts: torch.Tensor
    attention_weights: torch.Tensor
    evidence_counts: torch.Tensor
    available: torch.Tensor


@dataclass(frozen=True, slots=True)
class GlyphAssociativeWrite:
    state: GlyphAssociativeState
    accepted: bool
    write_slots: tuple[int, ...]
    delta_norm: float


class GlyphAssociativeMemory(nn.Module):
    """Soft content-addressed event memory with atomic bounded writes.

    Keys and values are supplied by learned Glyph encoders.  Retrieval is a
    soft attention over occupied slots and never performs an exact transition
    table lookup.  Empty memory contributes an exact all-zero context.
    """

    def __init__(
        self,
        width: int,
        *,
        slots: int,
        read_top_k: int,
    ) -> None:
        super().__init__()
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 1
            or isinstance(slots, bool)
            or not isinstance(slots, int)
            or slots <= 1
            or isinstance(read_top_k, bool)
            or not isinstance(read_top_k, int)
            or not 1 <= read_top_k <= slots
        ):
            raise ValueError("associative dimensions are invalid")
        self.width = width
        self.slots = slots
        self.read_top_k = read_top_k
        self.trace_slot_count = slots // 2
        self.outcome_slot_count = slots - self.trace_slot_count
        # Memory addresses arrive as final, identity-preserving keys.  This
        # nonpersistent zero-sized buffer gives state validation a device and
        # dtype reference without inserting a learned map between equal public
        # pairs at write and read time.
        self.register_buffer(
            "_device_dtype_anchor",
            torch.empty(0),
            persistent=False,
        )

    def state_numel(self, batch_size: int = 1) -> int:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be positive")
        return batch_size * (
            self.slots * (2 * self.width + 2 + 2 * _PUBLIC_ID_WORDS) + 2
        )

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> GlyphAssociativeState:
        reference = self._device_dtype_anchor
        target_device = reference.device if device is None else torch.device(device)
        target_dtype = reference.dtype if dtype is None else dtype
        if target_device != reference.device or target_dtype != reference.dtype:
            raise ValueError("associative state must match memory parameters")
        shape = (batch_size, self.slots, self.width)
        return GlyphAssociativeState(
            keys=torch.zeros(shape, device=target_device, dtype=target_dtype),
            values=torch.zeros(shape, device=target_device, dtype=target_dtype),
            occupied=torch.zeros(
                shape[:2], device=target_device, dtype=torch.bool
            ),
            write_counts=torch.zeros(
                shape[:2], device=target_device, dtype=torch.long
            ),
            public_source_action_ids=torch.zeros(
                (*shape[:2], _PUBLIC_ID_WORDS),
                device=target_device,
                dtype=torch.long,
            ),
            public_successor_ids=torch.zeros(
                (*shape[:2], _PUBLIC_ID_WORDS),
                device=target_device,
                dtype=torch.long,
            ),
            trace_cursor=torch.zeros(
                (batch_size,), device=target_device, dtype=torch.long
            ),
            outcome_cursor=torch.zeros(
                (batch_size,), device=target_device, dtype=torch.long
            ),
        )

    def read(
        self,
        query_keys: torch.Tensor,
        state: GlyphAssociativeState,
        *,
        lane: str = "all",
    ) -> GlyphAssociativeRead:
        """Read one differentiable context per candidate query key."""

        self._validate_state(state)
        if query_keys.ndim == 2:
            query_keys = query_keys.unsqueeze(0)
        if (
            query_keys.ndim != 3
            or query_keys.shape[0] != state.batch_size
            or query_keys.shape[1] <= 0
            or query_keys.shape[2] != self.width
            or query_keys.device != state.keys.device
            or query_keys.dtype != state.keys.dtype
            or not bool(torch.isfinite(query_keys).all().item())
        ):
            raise ValueError("query keys must be finite [batch, count, width]")
        if lane not in ("all", "trace", "outcome"):
            raise ValueError("associative read lane is invalid")
        normalized_queries = F.normalize(query_keys, dim=-1, eps=1.0e-8)
        normalized_keys = F.normalize(state.keys, dim=-1, eps=1.0e-8)
        logits = torch.einsum(
            "bcw,bsw->bcs", normalized_queries, normalized_keys
        ) * math.sqrt(self.width)
        lane_mask = torch.ones_like(state.occupied)
        if lane == "trace":
            lane_mask[:, self.trace_slot_count :] = False
        elif lane == "outcome":
            lane_mask[:, : self.trace_slot_count] = False
        occupied_slots = state.occupied & lane_mask
        occupied = occupied_slots.unsqueeze(1).expand_as(logits)
        has_evidence = occupied.any(dim=-1)
        masked = logits.masked_fill(~occupied, -torch.inf)
        # Hard top-k selects only the declared number of candidates; the
        # selected values remain a differentiable soft attention.
        top_indices = masked.topk(self.read_top_k, dim=-1).indices
        selected = torch.zeros_like(occupied)
        selected.scatter_(-1, top_indices, True)
        retrieval_mask = occupied & selected
        sparse = logits.masked_fill(~retrieval_mask, -torch.inf)
        safe = torch.where(
            has_evidence.unsqueeze(-1), sparse, torch.zeros_like(sparse)
        )
        weights = torch.softmax(safe, dim=-1)
        weights = weights * retrieval_mask.to(dtype=weights.dtype)
        totals = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(
            totals > 0.0,
            weights / totals.clamp_min(torch.finfo(weights.dtype).tiny),
            torch.zeros_like(weights),
        )
        raw_contexts = torch.einsum("bcs,bsw->bcw", weights, state.values)
        counts = torch.einsum(
            "bcs,bs->bc",
            weights,
            state.write_counts.to(dtype=weights.dtype),
        )
        evidence_confidence = 1.0 - torch.exp(-counts)
        # Match confidence keeps even a one-slot read differentiable with
        # respect to its learned key; softmax alone is constant in that case.
        match_logits = torch.einsum("bcs,bcs->bc", weights, logits)
        match_confidence = torch.sigmoid(match_logits)
        contexts = raw_contexts * (
            evidence_confidence * match_confidence
        ).unsqueeze(-1)
        contexts = torch.where(
            has_evidence.unsqueeze(-1), contexts, torch.zeros_like(contexts)
        )
        if not bool(torch.isfinite(contexts).all().item()):
            raise RuntimeError("associative read produced non-finite context")
        return GlyphAssociativeRead(
            contexts=contexts,
            attention_weights=weights,
            evidence_counts=counts,
            available=has_evidence,
        )

    def write_events(
        self,
        event_keys: torch.Tensor,
        event_values: torch.Tensor,
        state: GlyphAssociativeState,
        *,
        lane: str,
        public_source_action_ids: torch.Tensor | None = None,
        public_successor_ids: torch.Tensor | None = None,
        minimum_effect: float = 0.0,
    ) -> GlyphAssociativeWrite:
        """Atomically stage one or more learned events, or restore exactly."""

        self._validate_state(state)
        if state.batch_size != 1:
            raise ValueError("Glyph event writes currently require batch size one")
        if lane not in ("trace", "outcome"):
            raise ValueError("associative write lane is invalid")
        if event_keys.ndim == 1:
            event_keys = event_keys.unsqueeze(0)
        if event_values.ndim == 1:
            event_values = event_values.unsqueeze(0)
        if (
            event_keys.ndim != 2
            or event_keys.shape != event_values.shape
            or event_keys.shape[0] <= 0
            or event_keys.shape[1] != self.width
            or event_keys.device != state.keys.device
            or event_values.device != state.keys.device
            or event_keys.dtype != state.keys.dtype
            or event_values.dtype != state.keys.dtype
            or not bool(torch.isfinite(event_keys).all().item())
            or not bool(torch.isfinite(event_values).all().item())
        ):
            raise ValueError("event keys and values must be finite [count, width]")
        if not math.isfinite(minimum_effect) or minimum_effect < 0.0:
            raise ValueError("minimum_effect must be finite and nonnegative")
        event_count = event_keys.shape[0]
        if lane == "trace":
            for label, metadata in (
                ("public_source_action_ids", public_source_action_ids),
                ("public_successor_ids", public_successor_ids),
            ):
                if (
                    not isinstance(metadata, torch.Tensor)
                    or metadata.shape != (event_count, _PUBLIC_ID_WORDS)
                    or metadata.dtype != torch.long
                    or metadata.device != state.keys.device
                ):
                    raise ValueError(
                        f"trace {label} must be long [count, {_PUBLIC_ID_WORDS}]"
                    )
        elif public_source_action_ids is not None or public_successor_ids is not None:
            raise ValueError("outcome events cannot carry public trace identities")

        keys = state.keys
        values = state.values
        occupied = state.occupied
        counts = state.write_counts
        source_action_ids = state.public_source_action_ids
        successor_ids = state.public_successor_ids
        trace_cursor = state.trace_cursor
        outcome_cursor = state.outcome_cursor
        write_slots: list[int] = []
        for event_index, (event_key, event_value) in enumerate(
            zip(event_keys, event_values, strict=True)
        ):
            if lane == "trace":
                local_slot = int(trace_cursor[0].item())
                slot = local_slot
                lane_slots = self.trace_slot_count
            else:
                local_slot = int(outcome_cursor[0].item())
                slot = self.trace_slot_count + local_slot
                lane_slots = self.outcome_slot_count
            selector = F.one_hot(
                torch.tensor(slot, device=keys.device), self.slots
            ).to(dtype=keys.dtype).reshape(1, self.slots, 1)
            keys = keys + selector * (event_key.reshape(1, 1, -1) - keys)
            values = values + selector * (
                event_value.reshape(1, 1, -1) - values
            )
            occupied = occupied | selector.squeeze(-1).to(dtype=torch.bool)
            count_selector = selector.squeeze(-1).to(dtype=torch.bool)
            counts = torch.where(count_selector, counts.new_tensor(1), counts)
            metadata_selector = count_selector.unsqueeze(-1)
            if lane == "trace":
                assert public_source_action_ids is not None
                assert public_successor_ids is not None
                source_action_ids = torch.where(
                    metadata_selector,
                    public_source_action_ids[event_index].reshape(1, 1, -1),
                    source_action_ids,
                )
                successor_ids = torch.where(
                    metadata_selector,
                    public_successor_ids[event_index].reshape(1, 1, -1),
                    successor_ids,
                )
                trace_cursor = trace_cursor.new_tensor(
                    ((local_slot + 1) % lane_slots,)
                )
            else:
                zero_metadata = torch.zeros(
                    (1, 1, _PUBLIC_ID_WORDS),
                    device=keys.device,
                    dtype=torch.long,
                )
                source_action_ids = torch.where(
                    metadata_selector, zero_metadata, source_action_ids
                )
                successor_ids = torch.where(
                    metadata_selector, zero_metadata, successor_ids
                )
                outcome_cursor = outcome_cursor.new_tensor(
                    ((local_slot + 1) % lane_slots,)
                )
            write_slots.append(slot)

        candidate = GlyphAssociativeState(
            keys,
            values,
            occupied,
            counts,
            source_action_ids,
            successor_ids,
            trace_cursor,
            outcome_cursor,
        )
        delta = torch.sqrt(
            (candidate.keys - state.keys).square().sum()
            + (candidate.values - state.values).square().sum()
            + (candidate.write_counts - state.write_counts)
            .to(dtype=state.keys.dtype)
            .square()
            .sum()
            + (candidate.public_source_action_ids != state.public_source_action_ids)
            .to(dtype=state.keys.dtype)
            .sum()
            + (candidate.public_successor_ids != state.public_successor_ids)
            .to(dtype=state.keys.dtype)
            .sum()
            + (candidate.trace_cursor - state.trace_cursor)
            .to(dtype=state.keys.dtype)
            .square()
            .sum()
            + (candidate.outcome_cursor - state.outcome_cursor)
            .to(dtype=state.keys.dtype)
            .square()
            .sum()
        )
        accepted = bool(torch.isfinite(delta).item()) and float(delta.item()) > minimum_effect
        return GlyphAssociativeWrite(
            state=candidate if accepted else state,
            accepted=accepted,
            write_slots=tuple(write_slots),
            delta_norm=float(delta.detach().item()) if accepted else 0.0,
        )

    def public_trace_is_identifiable(
        self,
        state: GlyphAssociativeState,
        source_action_id: torch.Tensor,
        successor_id: torch.Tensor,
    ) -> bool:
        """Use only retained public bindings to qualify a held-out event."""

        self._validate_state(state)
        for label, value in (
            ("source_action_id", source_action_id),
            ("successor_id", successor_id),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (_PUBLIC_ID_WORDS,)
                or value.dtype != torch.long
                or value.device != state.keys.device
            ):
                raise ValueError(f"{label} must be a device-resident public id")
        trace_range = slice(0, self.trace_slot_count)
        occupied = state.occupied[0, trace_range]
        pairs = state.public_source_action_ids[0, trace_range]
        successors = state.public_successor_ids[0, trace_range]
        matches = occupied & (pairs == source_action_id).all(dim=-1)
        if not bool(matches.any().item()):
            return False
        # Conflicting public successors make the source/action unidentifiable.
        return bool((successors[matches] == successor_id).all().item())

    def _validate_state(self, state: GlyphAssociativeState) -> None:
        if not isinstance(state, GlyphAssociativeState):
            raise TypeError("state must be GlyphAssociativeState")
        if (
            state.slot_count != self.slots
            or state.width != self.width
            or state.keys.device != self._device_dtype_anchor.device
            or state.keys.dtype != self._device_dtype_anchor.dtype
        ):
            raise ValueError("associative state topology does not match memory")
        if bool((state.trace_cursor >= self.trace_slot_count).any().item()) or bool(
            (state.outcome_cursor >= self.outcome_slot_count).any().item()
        ):
            raise ValueError("associative lane cursor is outside reserved capacity")


@dataclass(frozen=True, slots=True)
class GlyphTaskEncoding:
    """Neural projection of one public task; ordering follows public tuples."""

    state_embeddings: torch.Tensor
    action_embeddings: torch.Tensor
    state_address_anchors: torch.Tensor
    action_address_anchors: torch.Tensor
    pair_key_anchors: torch.Tensor
    stop_key_anchors: torch.Tensor
    origin_embedding: torch.Tensor
    goal_embedding: torch.Tensor


@dataclass(frozen=True, slots=True)
class GlyphStepScores:
    """Dynamic action/STOP scores and learned multi-step causal predictions."""

    logits: torch.Tensor
    action_logits: torch.Tensor
    stop_logit: torch.Tensor
    successor_state_logits: torch.Tensor
    associative_recall_logits: torch.Tensor
    predicted_successors: torch.Tensor
    raw_reversible_successors: torch.Tensor
    plastic_context: torch.Tensor
    transition_lattice_logits: torch.Tensor
    reasoning_node_codes: torch.Tensor
    reasoning_action_logits: torch.Tensor
    current_state_belief: torch.Tensor
    reasoning_steps: int


@dataclass(frozen=True, slots=True)
class GlyphTransitionLattice:
    """Learned public-evidence beliefs for every declared source/action pair."""

    successor_state_logits: torch.Tensor
    successor_probabilities: torch.Tensor
    associative_recall_logits: torch.Tensor
    predicted_successors: torch.Tensor
    raw_reversible_successors: torch.Tensor
    trace_contexts: torch.Tensor
    outcome_contexts: torch.Tensor


@dataclass(frozen=True, slots=True)
class GlyphNeuralRollout:
    """One sampled or greedy neural trajectory, never a plan enumeration."""

    procedure: CommittedGlyphProcedure
    step_logits: tuple[torch.Tensor, ...]
    selected_indices: tuple[int, ...]
    step_query_keys: tuple[torch.Tensor, ...]
    step_current_embeddings: tuple[torch.Tensor, ...]
    step_state_beliefs: tuple[torch.Tensor, ...]
    action_count: int
    task_digest: str
    incoming_state_digest: str

    def __post_init__(self) -> None:
        lengths = {
            len(self.step_logits),
            len(self.selected_indices),
            len(self.step_query_keys),
            len(self.step_current_embeddings),
            len(self.step_state_beliefs),
        }
        if len(lengths) != 1:
            raise ValueError("rollout decision records must have equal length")
        if not self.step_logits:
            raise ValueError("a rollout must make at least one action-or-STOP decision")
        if not 1 <= self.action_count <= 3:
            raise ValueError("rollout action count must be one through three")
        for logits, selected, query_key, current, belief in zip(
            self.step_logits,
            self.selected_indices,
            self.step_query_keys,
            self.step_current_embeddings,
            self.step_state_beliefs,
            strict=True,
        ):
            if logits.shape != (self.action_count + 1,):
                raise ValueError("rollout step logits have the wrong dynamic width")
            if not 0 <= selected <= self.action_count:
                raise ValueError("rollout selection is outside actions plus STOP")
            if query_key.ndim != 1 or current.shape != query_key.shape:
                raise ValueError("rollout key/current records must share one width")
            if belief.ndim != 1 or belief.shape[0] < 2:
                raise ValueError("rollout state beliefs must be nontrivial vectors")
            if not bool(torch.isfinite(belief).all().item()) or not bool(
                torch.isclose(
                    belief.sum(),
                    belief.new_tensor(1.0),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                ).item()
            ):
                raise ValueError("rollout state beliefs must be finite probabilities")


@dataclass(frozen=True, slots=True)
class GlyphTraceAcquisition:
    state: GlyphAssociativeState
    public_transitions: int
    accepted_writes: int


@dataclass(frozen=True, slots=True)
class GlyphScalarFeedback:
    state: GlyphAssociativeState
    accepted: bool
    scalar_observations: int
    write_slots: tuple[int, ...]
    delta_norm: float


class TypedGlyphGraphEncoder(nn.Module):
    """Shared typed graph encoder over a variable public state/action set."""

    def __init__(self, profile: GlyphMachineRunProfile) -> None:
        super().__init__()
        self.width = profile.width
        self.hash_features = FrozenHashTextEncoder(profile.hash_width)
        # Opaque public symbols must retain their identity while the contextual
        # graph remains plastic.  This second encoder is parameter-free and
        # emits directly at controller width, so no trainable projection can
        # erase the identity substrate.
        self.public_identity_features = FrozenHashTextEncoder(profile.width)
        self.context_residual_fraction = 0.25
        self.state_projection = nn.Sequential(
            nn.LayerNorm(profile.hash_width),
            nn.Linear(profile.hash_width, profile.width),
            nn.SiLU(),
            nn.Linear(profile.width, profile.width),
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(profile.hash_width),
            nn.Linear(profile.hash_width, profile.width),
            nn.SiLU(),
            nn.Linear(profile.width, profile.width),
        )
        self.type_embeddings = nn.Parameter(torch.empty(2, profile.width))
        layer = nn.TransformerEncoderLayer(
            d_model=profile.width,
            nhead=profile.graph_heads,
            dim_feedforward=profile.hidden_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph = nn.TransformerEncoder(
            layer,
            num_layers=profile.graph_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(profile.width)
        nn.init.normal_(self.type_embeddings, mean=0.0, std=1.0 / math.sqrt(profile.width))

    def forward(
        self,
        states: Sequence[State],
        actions: Sequence[ActionSchema],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 2 <= len(states) <= 4 or not 1 <= len(actions) <= 3:
            raise ValueError("glyph graph requires 2-4 states and 1-3 actions")
        reference = self.type_embeddings
        state_hashes = self.hash_features.encode_texts(
            [_state_text(value) for value in states],
            device=reference.device,
            dtype=reference.dtype,
        )
        action_hashes = self.hash_features.encode_texts(
            [_action_text(value) for value in actions],
            device=reference.device,
            dtype=reference.dtype,
        )
        state_nodes = self.state_projection(state_hashes) + self.type_embeddings[0]
        action_nodes = self.action_projection(action_hashes) + self.type_embeddings[1]
        joined = torch.cat((state_nodes, action_nodes), dim=0).unsqueeze(0)
        contextual = self.output_norm(self.graph(joined)).squeeze(0)
        state_context = contextual[: len(states)]
        action_context = contextual[len(states) :]
        state_anchors = self.public_identity_features.encode_texts(
            [f"state\x00{_state_text(value)}" for value in states],
            device=reference.device,
            dtype=reference.dtype,
        )
        action_anchors = self.public_identity_features.encode_texts(
            [f"action\x00{_action_text(value)}" for value in actions],
            device=reference.device,
            dtype=reference.dtype,
        )
        return (
            self._anchored_context(state_anchors, state_context),
            self._anchored_context(action_anchors, action_context),
        )

    def _anchored_context(
        self,
        anchors: torch.Tensor,
        contextual: torch.Tensor,
    ) -> torch.Tensor:
        """Add a separately centered, norm-bounded learned residual.

        The bound is per public symbol and relative to that symbol's frozen
        anchor norm.  A shared or arbitrarily large contextual vector therefore
        cannot collapse distinct public identities.
        """

        if anchors.shape != contextual.shape or anchors.ndim != 2:
            raise ValueError("identity anchors and contexts must share [items, width]")
        centered = contextual - contextual.mean(dim=0, keepdim=True)
        anchor_scale = math.sqrt(self.width)
        scaled_anchors = anchors * anchor_scale
        residual_limits = (
            self.context_residual_fraction
            * scaled_anchors.norm(dim=-1, keepdim=True)
        )
        residual_norms = centered.norm(dim=-1, keepdim=True)
        residual_scales = torch.clamp(
            residual_limits
            / residual_norms.clamp_min(torch.finfo(contextual.dtype).eps),
            max=1.0,
        )
        bounded = centered * residual_scales
        return F.layer_norm(scaled_anchors + bounded, (self.width,))


class GlyphBackwardProcedureReasoner(nn.Module):
    """Learned fixed-depth goal propagation over soft public transition beliefs.

    This module never constructs action sequences.  It performs at most three
    shared neural message-passing updates over ``[state, action, successor]``
    beliefs, then returns one score per currently declared action.  Forward
    transition beliefs are sufficient: goal-conditioned node codes flow from
    successor nodes back to their source/action edges without a reverse store
    or an assumed inverse transition.
    """

    def __init__(self, profile: GlyphMachineRunProfile) -> None:
        super().__init__()
        width = profile.width
        rank = profile.transition_rank
        self.width = width
        self.goal_tokens = nn.Parameter(torch.empty(2, width))
        self.depth_tokens = nn.Parameter(torch.empty(_MAX_REASONING_STEPS, width))
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(7 * width),
            nn.Linear(7 * width, rank),
            nn.SiLU(),
            nn.Linear(rank, width, bias=False),
        )
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )
        self.state_cell = nn.GRUCell(width, width)
        nn.init.normal_(self.goal_tokens, mean=0.0, std=1.0 / math.sqrt(width))
        nn.init.normal_(self.depth_tokens, mean=0.0, std=1.0 / math.sqrt(width))

    def forward(
        self,
        state_embeddings: torch.Tensor,
        goal_state_index: int,
        lattice: GlyphTransitionLattice,
        current_state_belief: torch.Tensor,
        *,
        steps_remaining: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state_embeddings.ndim != 2 or state_embeddings.shape[-1] != self.width:
            raise ValueError("reasoner state embeddings must be [states, width]")
        state_count = state_embeddings.shape[0]
        if (
            isinstance(goal_state_index, bool)
            or not isinstance(goal_state_index, int)
            or not 0 <= goal_state_index < state_count
        ):
            raise ValueError("reasoner goal index is outside the public state set")
        if (
            isinstance(steps_remaining, bool)
            or not isinstance(steps_remaining, int)
            or not 1 <= steps_remaining <= _MAX_REASONING_STEPS
        ):
            raise ValueError("reasoner steps must be an integer from one through four")
        action_count = lattice.successor_probabilities.shape[1]
        expected_lattice_shape = (state_count, action_count, state_count)
        if (
            lattice.successor_probabilities.shape != expected_lattice_shape
            or lattice.successor_state_logits.shape != expected_lattice_shape
            or lattice.predicted_successors.shape
            != (state_count, action_count, self.width)
            or lattice.trace_contexts.shape
            != (state_count, action_count, self.width)
            or lattice.outcome_contexts.shape
            != (state_count, action_count, self.width)
        ):
            raise ValueError("reasoner transition lattice has incompatible dimensions")
        if (
            current_state_belief.shape != (state_count,)
            or current_state_belief.device != state_embeddings.device
            or current_state_belief.dtype != state_embeddings.dtype
            or not bool(torch.isfinite(current_state_belief).all().item())
            or bool((current_state_belief < 0.0).any().item())
            or not bool(
                torch.isclose(
                    current_state_belief.sum(),
                    current_state_belief.new_tensor(1.0),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                ).item()
            )
        ):
            raise ValueError("reasoner current state belief must be a probability vector")

        goal_flags = torch.zeros(
            state_count,
            device=state_embeddings.device,
            dtype=torch.long,
        )
        goal_flags[goal_state_index] = 1
        node_codes = self.goal_tokens[goal_flags]
        goal = state_embeddings[goal_state_index]
        goal_mask = goal_flags.to(dtype=state_embeddings.dtype).unsqueeze(-1)

        for depth_index in range(steps_remaining - 1):
            successor_codes = torch.einsum(
                "sat,tw->saw",
                lattice.successor_probabilities,
                node_codes,
            )
            edges = self._edge_messages(
                lattice.predicted_successors,
                goal,
                successor_codes,
                lattice.trace_contexts,
                lattice.outcome_contexts,
                self.depth_tokens[depth_index],
            )
            edge_logits = self.edge_gate(edges).squeeze(-1)
            edge_weights = torch.softmax(edge_logits, dim=1)
            pooled = torch.einsum("sa,saw->sw", edge_weights, edges)
            node_codes = self.state_cell(pooled, node_codes)
            # The public destination remains a source signal at every shared
            # update; what reaches other nodes is still determined by learned
            # edge messages over learned transition probabilities.
            node_codes = node_codes + goal_mask * self.goal_tokens[1]

        current_probabilities = torch.einsum(
            "s,sat->at",
            current_state_belief,
            lattice.successor_probabilities,
        )
        current_successors = current_probabilities @ state_embeddings
        current_successor_codes = current_probabilities @ node_codes
        current_trace_contexts = torch.einsum(
            "s,saw->aw",
            current_state_belief,
            lattice.trace_contexts,
        )
        current_outcome_contexts = torch.einsum(
            "s,saw->aw",
            current_state_belief,
            lattice.outcome_contexts,
        )
        current_edges = self._edge_messages(
            current_successors,
            goal,
            current_successor_codes,
            current_trace_contexts,
            current_outcome_contexts,
            self.depth_tokens[steps_remaining - 1],
        )
        action_logits = self.edge_gate(current_edges).squeeze(-1)
        if not bool(torch.isfinite(action_logits).all().item()) or not bool(
            torch.isfinite(node_codes).all().item()
        ):
            raise RuntimeError("glyph procedure reasoner produced non-finite values")
        return action_logits, node_codes

    def _edge_messages(
        self,
        predicted_successors: torch.Tensor,
        goal: torch.Tensor,
        successor_codes: torch.Tensor,
        trace_contexts: torch.Tensor,
        outcome_contexts: torch.Tensor,
        depth_token: torch.Tensor,
    ) -> torch.Tensor:
        if predicted_successors.shape != successor_codes.shape or (
            predicted_successors.shape != trace_contexts.shape
            or predicted_successors.shape != outcome_contexts.shape
        ):
            raise ValueError("reasoner edge tensors must share one shape")
        if predicted_successors.shape[-1] != self.width:
            raise ValueError("reasoner edge tensors have the wrong width")
        leading = predicted_successors.shape[:-1]
        flat_successors = predicted_successors.reshape(-1, self.width)
        flat_goals = goal.reshape(1, -1).expand(flat_successors.shape[0], -1)
        flat_codes = successor_codes.reshape(-1, self.width)
        flat_trace = trace_contexts.reshape(-1, self.width)
        flat_outcome = outcome_contexts.reshape(-1, self.width)
        depth_rows = depth_token.reshape(1, -1).expand(flat_successors.shape[0], -1)
        features = torch.cat(
            (
                _comparison_features(flat_successors, flat_goals),
                flat_codes + depth_rows,
                flat_trace,
                flat_outcome,
            ),
            dim=-1,
        )
        return self.edge_encoder(features).reshape(*leading, self.width)


class GlyphMachineController(nn.Module):
    """One scalable dynamic controller with learned causal action effects."""

    def __init__(self, profile: GlyphMachineRunProfile) -> None:
        super().__init__()
        if not isinstance(profile, GlyphMachineRunProfile):
            raise TypeError("profile must be a GlyphMachineRunProfile")
        self.profile = profile
        width = profile.width
        hidden_width = profile.hidden_width
        self.graph_encoder = TypedGlyphGraphEncoder(profile)
        self.public_address_features = FrozenHashTextEncoder(width)
        self.memory = GlyphAssociativeMemory(
            width,
            slots=profile.memory_slots,
            read_top_k=profile.memory_read_top_k,
        )
        self.event_key_encoder = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.trace_value_encoder = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.outcome_content_encoder = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.outcome_direction_encoder = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.causal_transition = ConditionalReversibleTransition(
            width,
            rank=profile.transition_rank,
        )
        # Identity initialization starved the conditioning path of its first
        # gradient.  A tiny nonzero coupling remains algebraically invertible
        # while an empty associative state still contributes exactly zero.
        nn.init.normal_(
            self.causal_transition.first_up.weight,
            mean=0.0,
            std=1.0e-3 / math.sqrt(profile.transition_rank),
        )
        nn.init.normal_(
            self.causal_transition.second_up.weight,
            mean=0.0,
            std=1.0e-3 / math.sqrt(profile.transition_rank),
        )
        self.successor_query = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        comparison_width = 4 * width
        self.procedure_reasoner = GlyphBackwardProcedureReasoner(profile)
        self.stop_head = nn.Sequential(
            nn.LayerNorm(comparison_width + width),
            nn.Linear(comparison_width + width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        self.procedure_cell = nn.GRUCell(width, width)
        self.procedure_start = nn.Parameter(torch.zeros(width))

    def initial_state(self, batch_size: int = 1) -> GlyphAssociativeState:
        return self.memory.initial_state(batch_size)

    def transition_event_keys(
        self,
        encoding: GlyphTaskEncoding,
    ) -> torch.Tensor:
        """Return one final learned key for every public state/action pair.

        The frozen pair anchor contains the complete public structures.  The
        learned event encoder may adjust an address only through an orthogonal,
        smoothly norm-bounded residual, so a large shared residual cannot erase
        pair identity.  Nothing in this path depends on a task goal, trace
        successor, memory slot, or the other pairs' contextual graph codes.
        """

        width = self.profile.width
        state_anchors = encoding.state_address_anchors
        action_anchors = encoding.action_address_anchors
        pair_anchors = encoding.pair_key_anchors
        if (
            state_anchors.ndim != 2
            or action_anchors.ndim != 2
            or state_anchors.shape[-1] != width
            or action_anchors.shape[-1] != width
            or pair_anchors.shape
            != (state_anchors.shape[0], action_anchors.shape[0], width)
        ):
            raise ValueError("public transition anchors have incompatible shapes")
        source_rows = state_anchors[:, None, :].expand_as(pair_anchors)
        action_rows = action_anchors[None, :, :].expand_as(pair_anchors)
        learned = self.event_key_encoder(
            torch.cat((source_rows, action_rows), dim=-1)
        )
        return _orthogonal_bounded_anchor(pair_anchors, learned)

    def event_query_keys(
        self,
        encoding: GlyphTaskEncoding,
        current_state_belief: torch.Tensor,
        goal_state_index: int,
    ) -> torch.Tensor:
        """Return current action keys followed by a learned anchored STOP key."""

        state_count = encoding.state_address_anchors.shape[0]
        if (
            current_state_belief.shape != (state_count,)
            or current_state_belief.device
            != encoding.state_address_anchors.device
            or current_state_belief.dtype != encoding.state_address_anchors.dtype
            or not bool(torch.isfinite(current_state_belief).all().item())
            or bool((current_state_belief < 0.0).any().item())
            or not bool(
                torch.isclose(
                    current_state_belief.sum(),
                    current_state_belief.new_tensor(1.0),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                ).item()
            )
        ):
            raise ValueError("current event address belief must be a probability vector")
        if (
            isinstance(goal_state_index, bool)
            or not isinstance(goal_state_index, int)
            or not 0 <= goal_state_index < state_count
        ):
            raise ValueError("event address goal is outside the public state set")

        pair_keys = self.transition_event_keys(encoding)
        action_keys = torch.einsum("s,saw->aw", current_state_belief, pair_keys)
        stop_anchor = torch.einsum(
            "s,sw->w",
            current_state_belief,
            encoding.stop_key_anchors[:, goal_state_index],
        )
        current_anchor = current_state_belief @ encoding.state_address_anchors
        goal_anchor = encoding.state_address_anchors[goal_state_index]
        stop_residual = self.event_key_encoder(
            torch.cat((current_anchor, goal_anchor), dim=-1).unsqueeze(0)
        )[0]
        stop_key = _orthogonal_bounded_anchor(stop_anchor, stop_residual)
        return torch.cat((action_keys, stop_key.unsqueeze(0)), dim=0)

    def trace_event_value(
        self,
        before: torch.Tensor,
        action: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        if before.shape != action.shape or before.shape != after.shape:
            raise ValueError("trace value inputs must share one width")
        learned = self.trace_value_encoder(
            torch.cat((before, action, after), dim=-1)
        )
        return _orthogonal_bounded_anchor(after, learned)

    def outcome_event_values(
        self,
        currents: Sequence[torch.Tensor],
        query_keys: Sequence[torch.Tensor],
        procedure_code: torch.Tensor,
        reward: float,
    ) -> torch.Tensor:
        if len(currents) != len(query_keys) or not currents:
            raise ValueError("outcome events require aligned nonempty decision records")
        numeric = _validate_reward(reward)
        signed = procedure_code.new_tensor(2.0 * numeric - 1.0)
        rows = torch.stack(
            [
                torch.cat((current, key, procedure_code), dim=-1)
                for current, key in zip(currents, query_keys, strict=True)
            ]
        )
        content = torch.tanh(self.outcome_content_encoder(rows))
        direction = torch.tanh(self.outcome_direction_encoder(rows))
        return 0.5 * (content + signed * direction)

    def encode_task(self, task: PublicGlyphMachineTask) -> GlyphTaskEncoding:
        if not isinstance(task, PublicGlyphMachineTask):
            raise TypeError("task must be a PublicGlyphMachineTask")
        state_embeddings, action_embeddings = self.graph_encoder(
            task.states,
            task.actions,
        )
        reference = state_embeddings
        state_address_anchors = self.public_address_features.encode_texts(
            [_state_address_text(value) for value in task.states],
            device=reference.device,
            dtype=reference.dtype,
        )
        action_address_anchors = self.public_address_features.encode_texts(
            [_action_address_text(value) for value in task.actions],
            device=reference.device,
            dtype=reference.dtype,
        )
        pair_key_anchors = self.public_address_features.encode_texts(
            [
                _pair_key_text(state, action)
                for state in task.states
                for action in task.actions
            ],
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(len(task.states), len(task.actions), self.profile.width)
        stop_key_anchors = self.public_address_features.encode_texts(
            [
                _stop_key_text(current, goal)
                for current in task.states
                for goal in task.states
            ],
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(len(task.states), len(task.states), self.profile.width)
        state_index = {state.digest: index for index, state in enumerate(task.states)}
        goal_state = _public_goal_state(task)
        return GlyphTaskEncoding(
            state_embeddings=state_embeddings,
            action_embeddings=action_embeddings,
            state_address_anchors=state_address_anchors,
            action_address_anchors=action_address_anchors,
            pair_key_anchors=pair_key_anchors,
            stop_key_anchors=stop_key_anchors,
            origin_embedding=state_embeddings[state_index[task.origin.digest]],
            goal_embedding=state_embeddings[state_index[goal_state.digest]],
        )

    def transition_lattice(
        self,
        encoding: GlyphTaskEncoding,
        state: GlyphAssociativeState,
        *,
        include_reversible_transition: bool = True,
    ) -> GlyphTransitionLattice:
        """Build all public source/action beliefs with one vectorized neural pass."""

        if state.batch_size != 1:
            raise ValueError("glyph-machine lattice currently requires batch size one")
        if type(include_reversible_transition) is not bool:
            raise TypeError("include_reversible_transition must be bool")
        width = self.profile.width
        states = encoding.state_embeddings
        actions = encoding.action_embeddings
        if (
            states.ndim != 2
            or actions.ndim != 2
            or states.shape[-1] != width
            or actions.shape[-1] != width
        ):
            raise ValueError("glyph-machine encoding has incompatible dimensions")
        state_count = states.shape[0]
        action_count = actions.shape[0]
        source_grid = states[:, None, :].expand(state_count, action_count, width)
        action_grid = actions[None, :, :].expand(state_count, action_count, width)
        flat_sources = source_grid.reshape(-1, width)
        flat_actions = action_grid.reshape(-1, width)
        query_keys = self.transition_event_keys(encoding).reshape(-1, width)
        trace_contexts = self.memory.read(
            query_keys,
            state,
            lane="trace",
        ).contexts[0].reshape(state_count, action_count, width)
        outcome_contexts = self.memory.read(
            query_keys,
            state,
            lane="outcome",
        ).contexts[0].reshape(state_count, action_count, width)
        if not include_reversible_transition:
            trace_contexts = torch.zeros_like(trace_contexts)
            outcome_contexts = torch.zeros_like(outcome_contexts)
        flat_trace = trace_contexts.reshape(-1, width)
        condition = torch.cat((flat_actions, flat_trace), dim=-1)
        raw_successors = (
            self.causal_transition(flat_sources, condition)
            if include_reversible_transition
            else flat_sources
        ).reshape(state_count, action_count, width)
        causal_queries = self.successor_query(raw_successors.reshape(-1, width))
        causal_logits = (
            causal_queries @ states.transpose(0, 1)
        ).reshape(state_count, action_count, state_count) / math.sqrt(width)
        # Trace values are already anchored in the same successor state space.
        # Compare them directly: no learned decoder may rotate away the public
        # successor identity retained by the value anchor.
        associative_recall_logits = (
            flat_trace @ states.transpose(0, 1)
        ).reshape(state_count, action_count, state_count) / math.sqrt(width)
        if not include_reversible_transition:
            associative_recall_logits = torch.zeros_like(associative_recall_logits)
        successor_state_logits = causal_logits + associative_recall_logits
        successor_probabilities = torch.softmax(successor_state_logits, dim=-1)
        predicted_successors = torch.einsum(
            "sat,tw->saw",
            successor_probabilities,
            states,
        )
        return GlyphTransitionLattice(
            successor_state_logits=successor_state_logits,
            successor_probabilities=successor_probabilities,
            associative_recall_logits=associative_recall_logits,
            predicted_successors=predicted_successors,
            raw_reversible_successors=raw_successors,
            trace_contexts=trace_contexts,
            outcome_contexts=outcome_contexts,
        )

    def score_actions(
        self,
        task: PublicGlyphMachineTask,
        state: GlyphAssociativeState,
        *,
        current_state_belief: torch.Tensor | None = None,
        goal_state_index: int | None = None,
        steps_remaining: int | None = None,
        encoding: GlyphTaskEncoding | None = None,
        include_reversible_transition: bool = True,
    ) -> GlyphStepScores:
        _validate_controller_task_state(self, task, state)
        encoded = self.encode_task(task) if encoding is None else encoding
        states = encoded.state_embeddings
        actions = encoded.action_embeddings
        state_count = states.shape[0]
        if goal_state_index is None:
            goal_digest = _public_goal_state(task).digest
            goal_state_index = next(
                index
                for index, value in enumerate(task.states)
                if value.digest == goal_digest
            )
        if steps_remaining is None:
            steps_remaining = task.max_steps
        if current_state_belief is None:
            origin_index = next(
                index
                for index, value in enumerate(task.states)
                if value.digest == task.origin.digest
            )
            current_state_belief = F.one_hot(
                torch.tensor(origin_index, device=states.device),
                state_count,
            ).to(dtype=states.dtype)
        lattice = self.transition_lattice(
            encoded,
            state,
            include_reversible_transition=include_reversible_transition,
        )
        return self._score_actions_from_lattice(
            task,
            state,
            encoded,
            lattice,
            current_state_belief=current_state_belief,
            goal_state_index=goal_state_index,
            steps_remaining=steps_remaining,
            include_reversible_transition=include_reversible_transition,
        )

    def _score_actions_from_lattice(
        self,
        task: PublicGlyphMachineTask,
        state: GlyphAssociativeState,
        encoded: GlyphTaskEncoding,
        lattice: GlyphTransitionLattice,
        *,
        current_state_belief: torch.Tensor,
        goal_state_index: int,
        steps_remaining: int,
        include_reversible_transition: bool,
    ) -> GlyphStepScores:
        """Score the production action-plus-STOP surface from one lattice.

        Inference and public suffix supervision both call this exact path.  The
        caller may reuse a lattice across visible suffixes, but it cannot train
        a separate answer or STOP head that inference does not consume.
        """

        states = encoded.state_embeddings
        actions = encoded.action_embeddings
        current_probabilities = torch.einsum(
            "s,sat->at",
            current_state_belief,
            lattice.successor_probabilities,
        )
        successor_state_logits = torch.log(
            current_probabilities.clamp_min(torch.finfo(current_probabilities.dtype).tiny)
        )
        associative_recall_logits = torch.einsum(
            "s,sat->at",
            current_state_belief,
            lattice.associative_recall_logits,
        )
        predicted_successors = current_probabilities @ states
        raw_successors = torch.einsum(
            "s,saw->aw",
            current_state_belief,
            lattice.raw_reversible_successors,
        )
        action_contexts = torch.einsum(
            "s,saw->aw",
            current_state_belief,
            lattice.trace_contexts,
        )
        action_logits, reasoning_node_codes = self.procedure_reasoner(
            states,
            goal_state_index,
            lattice,
            current_state_belief,
            steps_remaining=steps_remaining,
        )

        current = current_state_belief @ states
        goal = states[goal_state_index]
        stop_query_key = self.event_query_keys(
            encoded,
            current_state_belief,
            goal_state_index,
        )[-1:]
        stop_context = self.memory.read(
            stop_query_key,
            state,
            lane="outcome",
        ).contexts[0, 0]
        if not include_reversible_transition:
            stop_context = torch.zeros_like(stop_context)
        stop_features = _comparison_features(
            current.unsqueeze(0),
            goal.unsqueeze(0),
        )
        stop_logit = self.stop_head(
            torch.cat((stop_features, stop_context.unsqueeze(0)), dim=-1)
        ).reshape(())
        logits = torch.cat((action_logits, stop_logit.unsqueeze(0)), dim=0)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("glyph-machine controller produced non-finite logits")
        return GlyphStepScores(
            logits=logits,
            action_logits=action_logits,
            stop_logit=stop_logit,
            successor_state_logits=successor_state_logits,
            associative_recall_logits=associative_recall_logits,
            predicted_successors=predicted_successors,
            raw_reversible_successors=raw_successors,
            plastic_context=action_contexts,
            transition_lattice_logits=lattice.successor_state_logits,
            reasoning_node_codes=reasoning_node_codes,
            reasoning_action_logits=action_logits,
            current_state_belief=current_state_belief,
            reasoning_steps=steps_remaining,
        )

    def summarize_procedure(
        self,
        task: PublicGlyphMachineTask,
        procedure: CommittedGlyphProcedure,
        *,
        encoding: GlyphTaskEncoding | None = None,
    ) -> torch.Tensor:
        encoded = self.encode_task(task) if encoding is None else encoding
        by_digest = {
            action.digest: encoded.action_embeddings[index]
            for index, action in enumerate(task.actions)
        }
        procedure_code = self.procedure_start
        for action in procedure.actions:
            procedure_code = self.procedure_cell(
                by_digest[action.schema.digest],
                procedure_code,
            )
        return procedure_code

    def public_trace_losses(
        self,
        task: PublicGlyphMachineTask,
        state: GlyphAssociativeState,
        *,
        transitions: Sequence[Transition] | None = None,
    ) -> torch.Tensor:
        """Return one visible-successor objective per public event."""

        transitions = (
            _public_transitions(task)
            if transitions is None
            else tuple(transitions)
        )
        if not transitions:
            raise ValueError("public trace loss requires at least one visible transition")
        encoded = self.encode_task(task)
        state_indices = {value.digest: index for index, value in enumerate(task.states)}
        action_indices = {
            value.digest: index for index, value in enumerate(task.actions)
        }
        lattice = self.transition_lattice(encoded, state)
        losses: list[torch.Tensor] = []
        for transition in transitions:
            before_index = state_indices[transition.before.digest]
            after_index = state_indices[transition.after.digest]
            action_index = action_indices[transition.action.schema.digest]
            successor_logits = lattice.successor_state_logits[
                before_index,
                action_index,
            ].unsqueeze(0)
            target = torch.tensor(
                (after_index,),
                device=successor_logits.device,
                dtype=torch.long,
            )
            classification = F.cross_entropy(successor_logits, target)
            direct_recall_logits = lattice.associative_recall_logits[
                before_index,
                action_index,
            ].unsqueeze(0)
            direct_recall = F.cross_entropy(direct_recall_logits, target)
            raw_successor = lattice.raw_reversible_successors[
                before_index,
                action_index,
            ]
            target_embedding = encoded.state_embeddings[after_index]
            alignment = 1.0 - F.cosine_similarity(
                raw_successor.unsqueeze(0),
                target_embedding.unsqueeze(0),
                dim=-1,
                eps=1.0e-8,
            ).mean()
            losses.append(classification + direct_recall + 0.25 * alignment)
        return torch.stack(losses)

    def public_backward_reasoning_losses(
        self,
        task: PublicGlyphMachineTask,
        state: GlyphAssociativeState,
    ) -> torch.Tensor:
        """Teach production action/STOP decisions from public trace suffixes.

        A suffix is already-visible experience, not a generated candidate plan:
        its final public state supplies the destination, its length supplies the
        public horizon, and its first observed action supplies one dynamic
        action target.  The visible endpoint supplies STOP.  Prefix and endpoint
        groups are averaged separately so their relative weight cannot depend
        on trace length.  Observation-free queries never call this objective.
        """

        _validate_controller_task_state(self, task, state)
        encoded = self.encode_task(task)
        state_indices = {value.digest: index for index, value in enumerate(task.states)}
        action_indices = {
            value.digest: index for index, value in enumerate(task.actions)
        }
        lattice = self.transition_lattice(encoded, state)
        prefix_losses: list[torch.Tensor] = []
        endpoint_losses: list[torch.Tensor] = []
        state_count = len(task.states)
        stop_index = len(task.actions)
        for observation in task.observations:
            transitions = observation.transitions
            suffix_start = max(0, len(transitions) - _MAX_REASONING_STEPS)
            goal_index = state_indices[transitions[-1].after.digest]
            for start_index in range(suffix_start, len(transitions)):
                transition = transitions[start_index]
                before_index = state_indices[transition.before.digest]
                if before_index == goal_index:
                    continue
                steps_remaining = len(transitions) - start_index
                current_belief = F.one_hot(
                    torch.tensor(before_index, device=encoded.state_embeddings.device),
                    state_count,
                ).to(dtype=encoded.state_embeddings.dtype)
                scores = self._score_actions_from_lattice(
                    task,
                    state,
                    encoded,
                    lattice,
                    current_state_belief=current_belief,
                    goal_state_index=goal_index,
                    steps_remaining=steps_remaining,
                    include_reversible_transition=True,
                )
                target = torch.tensor(
                    (action_indices[transition.action.schema.digest],),
                    device=scores.logits.device,
                    dtype=torch.long,
                )
                prefix_losses.append(
                    F.cross_entropy(scores.logits.unsqueeze(0), target)
                )

            endpoint_belief = F.one_hot(
                torch.tensor(goal_index, device=encoded.state_embeddings.device),
                state_count,
            ).to(dtype=encoded.state_embeddings.dtype)
            endpoint_scores = self._score_actions_from_lattice(
                task,
                state,
                encoded,
                lattice,
                current_state_belief=endpoint_belief,
                goal_state_index=goal_index,
                steps_remaining=1,
                include_reversible_transition=True,
            )
            endpoint_target = torch.tensor(
                (stop_index,),
                device=endpoint_scores.logits.device,
                dtype=torch.long,
            )
            endpoint_losses.append(
                F.cross_entropy(
                    endpoint_scores.logits.unsqueeze(0),
                    endpoint_target,
                )
            )

        group_means = []
        if prefix_losses:
            group_means.append(torch.stack(prefix_losses).mean())
        if endpoint_losses:
            group_means.append(torch.stack(endpoint_losses).mean())
        if not group_means:
            return encoded.state_embeddings.new_empty((0,))
        return torch.stack(group_means)

    def public_backward_reasoning_loss(
        self,
        task: PublicGlyphMachineTask,
        state: GlyphAssociativeState,
    ) -> torch.Tensor:
        losses = self.public_backward_reasoning_losses(task, state)
        if not losses.numel():
            raise ValueError("public backward reasoning requires an eligible suffix")
        return losses.mean()

    def public_trace_loss(
        self,
        task: PublicGlyphMachineTask,
        state: GlyphAssociativeState,
        *,
        transitions: Sequence[Transition] | None = None,
    ) -> torch.Tensor:
        """Mean compatibility wrapper over event-weighted trace losses."""

        return self.public_trace_losses(
            task,
            state,
            transitions=transitions,
        ).mean()


def acquire_public_traces(
    controller: GlyphMachineController,
    task: PublicGlyphMachineTask,
    state: GlyphAssociativeState,
) -> GlyphTraceAcquisition:
    """Transactionally retain visible transition events at fixed capacity."""

    _validate_controller_task_state(controller, task, state)
    return _acquire_transition_sequence(
        controller,
        task,
        state,
        _public_transitions(task),
    )


def _acquire_transition_sequence(
    controller: GlyphMachineController,
    task: PublicGlyphMachineTask,
    state: GlyphAssociativeState,
    transitions: Sequence[Transition],
    *,
    encoding: GlyphTaskEncoding | None = None,
) -> GlyphTraceAcquisition:
    encoded = controller.encode_task(task) if encoding is None else encoding
    state_indices = {value.digest: index for index, value in enumerate(task.states)}
    action_indices = {value.digest: index for index, value in enumerate(task.actions)}
    pair_event_keys = controller.transition_event_keys(encoded)
    current_state = state
    observed = 0
    accepted = 0
    for transition in transitions:
        if not isinstance(transition, Transition):
            raise TypeError("public transition sequence contains a non-transition")
        before_index = state_indices[transition.before.digest]
        after_index = state_indices[transition.after.digest]
        action_index = action_indices[transition.action.schema.digest]
        before = encoded.state_embeddings[before_index]
        action = encoded.action_embeddings[action_index]
        after = encoded.state_embeddings[after_index]
        event_key = pair_event_keys[before_index, action_index]
        event_value = controller.trace_event_value(before, action, after)
        source_action_id, successor_id = _public_transition_ids(
            transition,
            device=current_state.keys.device,
        )
        write = controller.memory.write_events(
            event_key,
            event_value,
            current_state,
            lane="trace",
            public_source_action_ids=source_action_id.unsqueeze(0),
            public_successor_ids=successor_id.unsqueeze(0),
        )
        current_state = write.state
        observed += 1
        accepted += int(write.accepted)
    return GlyphTraceAcquisition(current_state, observed, accepted)


def acquire_and_score_public_traces(
    controller: GlyphMachineController,
    task: PublicGlyphMachineTask,
    state: GlyphAssociativeState,
) -> tuple[GlyphTraceAcquisition, torch.Tensor, int, int]:
    """Acquire evidence and weight every identifiable public event equally.

    ``U`` is the number of unique post-write events. ``I`` counts only held-out
    events whose exact public source/action was already retained with one
    consistent public successor.  No unobserved reversibility is assumed.
    """

    _validate_controller_task_state(controller, task, state)
    transitions = _unique_public_transitions(task)
    if not transitions:
        raise ValueError("trace learning requires visible transitions")
    encoding = controller.encode_task(task)
    acquisition = _acquire_transition_sequence(
        controller,
        task,
        state,
        transitions,
        encoding=encoding,
    )
    post_write_losses = controller.public_trace_losses(
        task,
        acquisition.state,
        transitions=transitions,
    )
    unique_events = len(transitions)
    identifiable_losses: list[torch.Tensor] = []
    if len(transitions) > 1:
        for held_out_index, held_out in enumerate(transitions):
            retained = tuple(
                transition
                for index, transition in enumerate(transitions)
                if index != held_out_index
            )
            held_out_state = _clone_glyph_state(state)
            held_out_state = _acquire_transition_sequence(
                controller,
                task,
                held_out_state,
                retained,
                encoding=encoding,
            ).state
            source_action_id, successor_id = _public_transition_ids(
                held_out,
                device=held_out_state.keys.device,
            )
            if controller.memory.public_trace_is_identifiable(
                held_out_state,
                source_action_id,
                successor_id,
            ):
                identifiable_losses.append(
                    controller.public_trace_losses(
                        task,
                        held_out_state,
                        transitions=(held_out,),
                    )[0]
                )
    identifiable_events = len(identifiable_losses)
    numerator = post_write_losses.sum()
    if identifiable_losses:
        numerator = numerator + torch.stack(identifiable_losses).sum()
    objective = numerator / (unique_events + identifiable_events)
    return acquisition, objective, unique_events, identifiable_events


def rollout_glyph_procedure(
    controller: GlyphMachineController,
    task: PublicGlyphMachineTask,
    state: GlyphAssociativeState,
    *,
    greedy: bool = True,
    temperature: float = 1.0,
    include_reversible_transition: bool = True,
) -> GlyphNeuralRollout:
    """Run the dynamic controller once, choosing only declared actions or STOP."""

    _validate_controller_task_state(controller, task, state)
    if type(greedy) is not bool:
        raise TypeError("greedy must be bool")
    if type(include_reversible_transition) is not bool:
        raise TypeError("include_reversible_transition must be bool")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    encoded = controller.encode_task(task)
    state_indices = {value.digest: index for index, value in enumerate(task.states)}
    origin_index = state_indices[task.origin.digest]
    goal_index = state_indices[_public_goal_state(task).digest]
    current_belief = F.one_hot(
        torch.tensor(origin_index, device=encoded.state_embeddings.device),
        len(task.states),
    ).to(dtype=encoded.state_embeddings.dtype)
    current = current_belief @ encoded.state_embeddings
    chosen_actions: list[GroundAction] = []
    step_logits: list[torch.Tensor] = []
    selected_indices: list[int] = []
    step_query_keys: list[torch.Tensor] = []
    step_current_embeddings: list[torch.Tensor] = []
    step_state_beliefs: list[torch.Tensor] = []
    stopped = False
    stop_index = len(task.actions)
    for step_index in range(task.max_steps):
        scores = controller.score_actions(
            task,
            state,
            current_state_belief=current_belief,
            steps_remaining=task.max_steps - step_index,
            encoding=encoded,
            include_reversible_transition=include_reversible_transition,
        )
        decision_logits = scores.logits / temperature
        probabilities = torch.softmax(decision_logits, dim=-1)
        if greedy:
            selected = int(decision_logits.argmax(dim=-1).item())
        else:
            selected = int(torch.multinomial(probabilities, 1).item())
        step_logits.append(decision_logits)
        selected_indices.append(selected)
        query_keys = controller.event_query_keys(
            encoded,
            current_belief,
            goal_index,
        )
        step_query_keys.append(query_keys[selected])
        step_current_embeddings.append(current)
        step_state_beliefs.append(current_belief)
        if selected == stop_index:
            stopped = True
            break
        chosen_actions.append(task.actions[selected].ground())
        current_belief = torch.softmax(
            scores.successor_state_logits[selected],
            dim=-1,
        )
        current = current_belief @ encoded.state_embeddings
    procedure = commit_glyph_procedure(task, chosen_actions, stopped=stopped)
    return GlyphNeuralRollout(
        procedure=procedure,
        step_logits=tuple(step_logits),
        selected_indices=tuple(selected_indices),
        step_query_keys=tuple(step_query_keys),
        step_current_embeddings=tuple(step_current_embeddings),
        step_state_beliefs=tuple(step_state_beliefs),
        action_count=len(task.actions),
        task_digest=_public_task_digest(task),
        incoming_state_digest=glyph_associative_state_digest(state),
    )


def scalar_outcome_loss(
    rollout: GlyphNeuralRollout,
    reward: float,
) -> torch.Tensor:
    """Centered whole-trajectory fallback used only by narrow API tests.

    Meta-training uses ``centered_trajectory_preference_loss`` over multiple
    independently judged attempts.  This compatibility helper never labels
    each action in a failed procedure as independently incorrect.
    """

    advantage = _validate_reward(reward) - 0.5
    return -rollout.step_logits[0].new_tensor(advantage) * (
        _trajectory_log_probability(rollout)
    )


def centered_trajectory_preference_loss(
    rollouts: Sequence[GlyphNeuralRollout],
    rewards: Sequence[float],
) -> torch.Tensor:
    """Apply one centered terminal advantage to each complete trajectory."""

    if len(rollouts) != len(rewards) or len(rollouts) < 2:
        raise ValueError("trajectory preference requires at least two aligned attempts")
    numeric = tuple(_validate_reward(value) for value in rewards)
    reference = rollouts[0].step_logits[0]
    mean_reward = sum(numeric) / len(numeric)
    terms = [
        -reference.new_tensor(reward - mean_reward)
        * _trajectory_log_probability(rollout)
        for rollout, reward in zip(rollouts, numeric, strict=True)
    ]
    return torch.stack(terms).mean()


def _trajectory_log_probability(rollout: GlyphNeuralRollout) -> torch.Tensor:
    selected = [
        F.log_softmax(logits, dim=-1)[index]
        for logits, index in zip(
            rollout.step_logits,
            rollout.selected_indices,
            strict=True,
        )
    ]
    return torch.stack(selected).mean()


def apply_scalar_procedure_feedback(
    controller: GlyphMachineController,
    task: PublicGlyphMachineTask,
    rollout: GlyphNeuralRollout,
    reward: float,
    state: GlyphAssociativeState,
    *,
    binding_state: GlyphAssociativeState | None = None,
    minimum_effect: float = 0.0,
) -> GlyphScalarFeedback:
    """Apply exactly one scalar outcome or restore the exact incoming state."""

    _validate_controller_task_state(controller, task, state)
    numeric = _validate_reward(reward)
    if rollout.task_digest != _public_task_digest(task):
        raise ValueError("rollout is bound to a different public task")
    bound_state = state if binding_state is None else binding_state
    _validate_controller_task_state(controller, task, bound_state)
    if rollout.incoming_state_digest != glyph_associative_state_digest(bound_state):
        raise ValueError("rollout is stale for its declared competence state")
    if binding_state is not None and not _trace_lane_equal(state, binding_state):
        raise ValueError("rebound scalar feedback changed retained public traces")
    if not math.isfinite(minimum_effect) or minimum_effect < 0.0:
        raise ValueError("minimum_effect must be finite and nonnegative")
    incoming_digest = glyph_associative_state_digest(state)
    encoded = controller.encode_task(task)
    procedure_embedding = controller.summarize_procedure(
        task,
        rollout.procedure,
        encoding=encoded,
    )
    event_keys = torch.stack(rollout.step_query_keys)
    event_values = controller.outcome_event_values(
        rollout.step_current_embeddings,
        rollout.step_query_keys,
        procedure_embedding,
        numeric,
    )
    write = controller.memory.write_events(
        event_keys,
        event_values,
        state,
        lane="outcome",
        minimum_effect=minimum_effect,
    )
    accepted = write.accepted
    candidate_state = write.state if accepted else state
    if accepted:
        try:
            rescored = controller.score_actions(task, candidate_state)
            finite = bool(torch.isfinite(rescored.logits).all().item())
        except (RuntimeError, ValueError):
            finite = False
        if not finite:
            candidate_state = state
            accepted = False
    if not accepted and glyph_associative_state_digest(candidate_state) != incoming_digest:
        raise RuntimeError("rejected scalar transaction failed exact restoration")
    return GlyphScalarFeedback(
        state=candidate_state,
        accepted=accepted,
        scalar_observations=1,
        write_slots=write.write_slots,
        delta_norm=write.delta_norm if accepted else 0.0,
    )


def snapshot_glyph_state(state: GlyphAssociativeState) -> dict[str, torch.Tensor]:
    if not isinstance(state, GlyphAssociativeState):
        raise TypeError("state must be GlyphAssociativeState")
    return {
        "keys": state.keys.detach().clone(),
        "values": state.values.detach().clone(),
        "occupied": state.occupied.detach().clone(),
        "write_counts": state.write_counts.detach().clone(),
        "public_source_action_ids": state.public_source_action_ids.detach().clone(),
        "public_successor_ids": state.public_successor_ids.detach().clone(),
        "trace_cursor": state.trace_cursor.detach().clone(),
        "outcome_cursor": state.outcome_cursor.detach().clone(),
    }


def restore_glyph_state(
    snapshot: Mapping[str, torch.Tensor],
) -> GlyphAssociativeState:
    expected = {
        "keys",
        "values",
        "occupied",
        "write_counts",
        "public_source_action_ids",
        "public_successor_ids",
        "trace_cursor",
        "outcome_cursor",
    }
    if set(snapshot) != expected:
        raise ValueError("Glyph associative snapshot keys differ")
    if any(not isinstance(value, torch.Tensor) for value in snapshot.values()):
        raise TypeError("Glyph associative snapshot values must be tensors")
    return GlyphAssociativeState(
        keys=snapshot["keys"].detach().clone(),
        values=snapshot["values"].detach().clone(),
        occupied=snapshot["occupied"].detach().clone(),
        write_counts=snapshot["write_counts"].detach().clone(),
        public_source_action_ids=snapshot["public_source_action_ids"].detach().clone(),
        public_successor_ids=snapshot["public_successor_ids"].detach().clone(),
        trace_cursor=snapshot["trace_cursor"].detach().clone(),
        outcome_cursor=snapshot["outcome_cursor"].detach().clone(),
    )


def glyph_associative_state_digest(state: GlyphAssociativeState) -> str:
    if not isinstance(state, GlyphAssociativeState):
        raise TypeError("state must be GlyphAssociativeState")
    digest = hashlib.sha256(_STATE_DIGEST_DOMAIN)
    for name, value in snapshot_glyph_state(state).items():
        tensor = value.cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "big"))
        digest.update(dtype_bytes)
        digest.update(tensor.ndim.to_bytes(4, "big"))
        for size in tensor.shape:
            digest.update(int(size).to_bytes(8, "big"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def save_glyph_checkpoint(
    path: str | Path,
    controller: GlyphMachineController,
    state: GlyphAssociativeState,
) -> None:
    """Freeze slow weights and the exact fixed-capacity competence state."""

    target = Path(path)
    payload = {
        "version": _CHECKPOINT_VERSION,
        "profile": asdict(controller.profile),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in controller.state_dict().items()
        },
        "competence_state": {
            name: value.detach().cpu().clone()
            for name, value in snapshot_glyph_state(state).items()
        },
        "competence_digest": glyph_associative_state_digest(state),
    }
    torch.save(payload, target)


def load_glyph_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[GlyphMachineController, GlyphAssociativeState]:
    """Reload a frozen controller/state pair before any held-out evaluation."""

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != _CHECKPOINT_VERSION:
        raise RuntimeError("glyph-machine checkpoint identity is invalid")
    raw_profile = payload.get("profile")
    if not isinstance(raw_profile, dict):
        raise RuntimeError("glyph-machine checkpoint profile is missing")
    profile = GlyphMachineRunProfile(**raw_profile)
    if GLYPH_MACHINE_PROFILES.get(profile.name) != profile:
        raise RuntimeError("glyph-machine checkpoint profile is not registered")
    controller = GlyphMachineController(profile).to(device)
    controller.load_state_dict(payload["model_state"], strict=True)
    state = restore_glyph_state(payload["competence_state"])
    if glyph_associative_state_digest(state) != payload.get("competence_digest"):
        raise RuntimeError("glyph-machine checkpoint competence state changed")
    controller.eval()
    return controller, state


def glyph_machine_parameter_report(
    controller: GlyphMachineController,
) -> dict[str, int | str]:
    """Report measured slow-weight and constant online-state capacity."""

    total = sum(parameter.numel() for parameter in controller.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in controller.parameters()
        if parameter.requires_grad
    )
    return {
        "profile": controller.profile.name,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "fixed_competence_state_elements": controller.memory.state_numel(1),
        "graph_layers": controller.profile.graph_layers,
        "width": controller.profile.width,
        "procedure_reasoner": "recurrent_backward_goal_messages",
        "maximum_reasoning_steps": _MAX_REASONING_STEPS,
    }


def build_glyph_machine_controller(
    profile: str,
    *,
    device: torch.device | str = "cpu",
) -> GlyphMachineController:
    try:
        selected = GLYPH_MACHINE_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown glyph-machine profile: {profile}") from error
    return GlyphMachineController(selected).to(device)


@dataclass(slots=True)
class _ScalarJudgeLedger:
    judge: Callable[[GeneratedGlyphMachineTask, CommittedGlyphProcedure], float]
    calls: int = 0

    def __call__(
        self,
        pair: GeneratedGlyphMachineTask,
        procedure: CommittedGlyphProcedure,
    ) -> float:
        value = self.judge(pair, procedure)
        numeric = _validate_reward(value)
        self.calls += 1
        return numeric


def build_glyph_machine_evaluation_arms(
    stream: GlyphMachineTraceStream,
) -> dict[str, tuple[GlyphMachineTraceStream, bool]]:
    """Build four matched arms before any controller execution."""

    if not isinstance(stream, GlyphMachineTraceStream):
        raise TypeError("stream must be a GlyphMachineTraceStream")
    correct = make_glyph_machine_control_stream(stream, "correct")
    return {
        "correct": (correct, True),
        "no_trace": (
            make_glyph_machine_control_stream(stream, "no_trace"),
            True,
        ),
        "wrong_trace": (
            make_glyph_machine_control_stream(stream, "wrong_trace"),
            True,
        ),
        "reversible_removed": (correct, False),
    }


def _sample_training_preferences(
    controller: GlyphMachineController,
    pair: GeneratedGlyphMachineTask,
    state: GlyphAssociativeState,
    ledger: _ScalarJudgeLedger,
    config: GlyphMachineExperimentConfig,
) -> tuple[GlyphAssociativeState, torch.Tensor, int]:
    rollouts: list[GlyphNeuralRollout] = []
    rewards: list[float] = []
    accepted = 0
    incoming_digest = glyph_associative_state_digest(state)
    for _ in range(_TRAINING_ROLLOUTS_PER_TASK):
        rollout = rollout_glyph_procedure(
            controller,
            pair.learner,
            _clone_glyph_state(state),
            greedy=False,
            temperature=config.rollout_temperature,
        )
        if rollout.incoming_state_digest != incoming_digest:
            raise RuntimeError("preference rollouts did not share one incoming state")
        reward = ledger(pair, rollout.procedure)
        rollouts.append(rollout)
        rewards.append(reward)
    preference = centered_trajectory_preference_loss(rollouts, rewards)
    # Both attempts are compared before either can change competence.  Their
    # already-bound scalar events are then committed sequentially to the
    # reserved outcome lane while the public trace lane stays byte-exact.
    current_state = state
    for rollout, reward in zip(rollouts, rewards, strict=True):
        feedback = apply_scalar_procedure_feedback(
            controller,
            pair.learner,
            rollout,
            reward,
            current_state,
            binding_state=state,
        )
        current_state = feedback.state
        accepted += int(feedback.accepted)
    return (
        current_state,
        preference,
        accepted,
    )


def train_glyph_machine_controller(
    controller: GlyphMachineController,
    config: GlyphMachineExperimentConfig,
    *,
    judge: Callable[
        [GeneratedGlyphMachineTask, CommittedGlyphProcedure],
        float,
    ] = judge_glyph_procedure_attempt,
) -> dict[str, object]:
    """Meta-train one slow lineage from public traces and terminal scalars."""

    if controller.profile.name != config.profile:
        raise ValueError("controller and experiment profiles differ")
    ledger = _ScalarJudgeLedger(judge)
    trainable = tuple(
        parameter for parameter in controller.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("glyph-machine training selected no slow parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    commitments = glyph_machine_mechanism_partition("train")[
        : config.train_mechanisms
    ]
    losses: list[float] = []
    gradient_norms: list[float] = []
    trace_terms = 0
    public_trace_events = 0
    identifiable_leave_one_out_events = 0
    public_reasoning_terms = 0
    policy_terms = 0
    policy_attempts = 0
    accepted_online_writes = 0
    controller.train()
    started = time.perf_counter()
    for epoch in range(config.training_epochs):
        for mechanism_index, commitment in enumerate(commitments):
            episode_seed = _experiment_seed(
                config.seed,
                "train",
                epoch,
                mechanism_index,
            )
            _seed_torch(episode_seed, next(controller.parameters()).device)
            stream = _experiment_stream(
                config,
                partition="train",
                mechanism_index=mechanism_index,
                commitment=commitment,
                epoch=epoch,
            )
            state = controller.initial_state()
            objectives: list[torch.Tensor] = []
            for pair in stream.supports:
                acquisition, trace_objective, unique_events, identifiable_events = (
                    acquire_and_score_public_traces(
                        controller,
                        pair.learner,
                        state,
                    )
                )
                objectives.append(trace_objective)
                trace_terms += 1
                public_trace_events += unique_events
                identifiable_leave_one_out_events += identifiable_events
                state = acquisition.state
                accepted_online_writes += acquisition.accepted_writes
                reasoning_losses = controller.public_backward_reasoning_losses(
                    pair.learner,
                    state,
                )
                if reasoning_losses.numel():
                    objectives.append(reasoning_losses.mean())
                    public_reasoning_terms += int(reasoning_losses.numel())
                state, preference, accepted = _sample_training_preferences(
                    controller,
                    pair,
                    state,
                    ledger,
                    config,
                )
                objectives.append(preference)
                policy_terms += 1
                policy_attempts += _TRAINING_ROLLOUTS_PER_TASK
                accepted_online_writes += accepted
            for pair in stream.queries:
                state, preference, accepted = _sample_training_preferences(
                    controller,
                    pair,
                    state,
                    ledger,
                    config,
                )
                objectives.append(preference)
                policy_terms += 1
                policy_attempts += _TRAINING_ROLLOUTS_PER_TASK
                accepted_online_writes += accepted
            loss = torch.stack(objectives).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("glyph-machine meta-training loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                config.gradient_clip,
                error_if_nonfinite=True,
            )
            if not any(
                parameter.grad is not None
                and bool(parameter.grad.detach().count_nonzero().item())
                for parameter in trainable
            ):
                raise RuntimeError("glyph-machine objectives produced no slow gradient")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
    elapsed = time.perf_counter() - started
    expected_calls = (
        config.training_epochs
        * config.train_mechanisms
        * (config.supports_per_mechanism + config.queries_per_mechanism)
        * _TRAINING_ROLLOUTS_PER_TASK
    )
    if ledger.calls != expected_calls or ledger.calls != policy_attempts:
        raise RuntimeError("training violated exact scalar-per-attempt accounting")
    return {
        "epochs": config.training_epochs,
        "mechanisms": config.train_mechanisms,
        "optimizer_steps": len(losses),
        "trace_loss_terms": trace_terms,
        "public_trace_events": public_trace_events,
        "public_backward_reasoning_terms": public_reasoning_terms,
        "identifiable_leave_one_out_trace_terms": (
            identifiable_leave_one_out_events
        ),
        # Compatibility name now reports only eligible public-history folds.
        "leave_one_out_trace_terms": identifiable_leave_one_out_events,
        "policy_loss_terms": policy_terms,
        "policy_attempts": policy_attempts,
        "rollouts_per_training_task": _TRAINING_ROLLOUTS_PER_TASK,
        "scalar_judge_calls": ledger.calls,
        "expected_scalar_judge_calls": expected_calls,
        "accepted_online_writes": accepted_online_writes,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "elapsed_seconds": elapsed,
        "one_persistent_slow_lineage": True,
        "fresh_fast_state_per_mechanism": True,
        "complete_plan_candidates": 0,
    }


def evaluate_glyph_machine_partition(
    controller: GlyphMachineController,
    config: GlyphMachineExperimentConfig,
    *,
    partition: str,
    mechanism_count: int,
    judge: Callable[
        [GeneratedGlyphMachineTask, CommittedGlyphProcedure],
        float,
    ] = judge_glyph_procedure_attempt,
) -> dict[str, object]:
    """Evaluate matched causal arms and return every mechanism-level score."""

    if partition not in ("development", "final"):
        raise ValueError("evaluation partition must be development or final")
    maximum = 16
    if (
        isinstance(mechanism_count, bool)
        or not isinstance(mechanism_count, int)
        or not 1 <= mechanism_count <= maximum
    ):
        raise ValueError("evaluation mechanism_count must be between one and sixteen")
    if any(parameter.requires_grad for parameter in controller.parameters()):
        raise RuntimeError("evaluation requires frozen slow weights")
    commitments = glyph_machine_mechanism_partition(partition)[:mechanism_count]
    ledger = _ScalarJudgeLedger(judge)
    trace_only_rows: list[dict[str, float | str]] = []
    sequential_rows: list[dict[str, float | str]] = []
    started = time.perf_counter()
    controller.eval()
    with torch.no_grad():
        for mechanism_index, commitment in enumerate(commitments):
            base_stream = _experiment_stream(
                config,
                partition=partition,
                mechanism_index=mechanism_index,
                commitment=commitment,
                epoch=0,
            )
            arms = build_glyph_machine_evaluation_arms(base_stream)
            trace_only_rows.append(
                _evaluate_glyph_arm_row(
                    controller,
                    commitment,
                    arms,
                    ledger,
                    mode="trace_only",
                )
            )
            sequential_rows.append(
                _evaluate_glyph_arm_row(
                    controller,
                    commitment,
                    arms,
                    ledger,
                    mode="sequential_adaptation",
                )
            )
    elapsed = time.perf_counter() - started
    expected_calls = (
        mechanism_count
        * 4
        * (config.supports_per_mechanism + 2 * config.queries_per_mechanism)
    )
    if ledger.calls != expected_calls:
        raise RuntimeError("evaluation violated exact scalar call accounting")
    return {
        "partition": partition,
        "mechanisms": mechanism_count,
        # Acceptance claims are intentionally trace-only.  Sequential scalar
        # adaptation remains a separate diagnostic and cannot inflate them.
        "rows": trace_only_rows,
        "summary": _evaluation_summary(trace_only_rows),
        "trace_only": {
            "rows": trace_only_rows,
            "summary": _evaluation_summary(trace_only_rows),
            "scalar_judge_calls": (
                mechanism_count * 4 * config.queries_per_mechanism
            ),
        },
        "sequential_adaptation": {
            "rows": sequential_rows,
            "summary": _evaluation_summary(sequential_rows),
            "scalar_judge_calls": (
                mechanism_count
                * 4
                * (config.supports_per_mechanism + config.queries_per_mechanism)
            ),
        },
        "scalar_judge_calls": ledger.calls,
        "expected_scalar_judge_calls": expected_calls,
        "elapsed_seconds": elapsed,
    }


def _evaluate_glyph_arm_row(
    controller: GlyphMachineController,
    commitment: str,
    arms: Mapping[str, tuple[GlyphMachineTraceStream, bool]],
    ledger: _ScalarJudgeLedger,
    *,
    mode: str,
) -> dict[str, float | str]:
    if mode not in ("trace_only", "sequential_adaptation"):
        raise ValueError("Glyph evaluation mode is invalid")
    arm_scores: dict[str, float] = {}
    for arm, (stream, include_transition) in arms.items():
        state = controller.initial_state()
        if mode == "trace_only":
            for pair in stream.supports:
                state = acquire_public_traces(
                    controller,
                    pair.learner,
                    state,
                ).state
            support_snapshot = snapshot_glyph_state(state)
            query_rewards = []
            for pair in stream.queries:
                query_state = restore_glyph_state(support_snapshot)
                rollout = rollout_glyph_procedure(
                    controller,
                    pair.learner,
                    query_state,
                    include_reversible_transition=include_transition,
                )
                query_rewards.append(ledger(pair, rollout.procedure))
        else:
            for pair in stream.supports:
                state = acquire_public_traces(
                    controller,
                    pair.learner,
                    state,
                ).state
                rollout = rollout_glyph_procedure(
                    controller,
                    pair.learner,
                    state,
                    include_reversible_transition=include_transition,
                )
                reward = ledger(pair, rollout.procedure)
                state = apply_scalar_procedure_feedback(
                    controller,
                    pair.learner,
                    rollout,
                    reward,
                    state,
                ).state
            query_rewards = []
            for pair in stream.queries:
                rollout = rollout_glyph_procedure(
                    controller,
                    pair.learner,
                    state,
                    include_reversible_transition=include_transition,
                )
                reward = ledger(pair, rollout.procedure)
                query_rewards.append(reward)
                state = apply_scalar_procedure_feedback(
                    controller,
                    pair.learner,
                    rollout,
                    reward,
                    state,
                ).state
        arm_scores[arm] = sum(query_rewards) / len(query_rewards)
    return {
        "mechanism_commitment": commitment,
        **arm_scores,
        "correct_over_no_trace": arm_scores["correct"] - arm_scores["no_trace"],
        "correct_over_wrong_trace": (
            arm_scores["correct"] - arm_scores["wrong_trace"]
        ),
        "reversible_contribution": (
            arm_scores["correct"] - arm_scores["reversible_removed"]
        ),
    }


def run_glyph_machine_experiment(
    config: GlyphMachineExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    result_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    judge: Callable[
        [GeneratedGlyphMachineTask, CommittedGlyphProcedure],
        float,
    ] = judge_glyph_procedure_attempt,
) -> dict[str, object]:
    """Train, reload/freeze, and only then open the final partition."""

    controller = build_glyph_machine_controller(config.profile, device=device)
    target_device = next(controller.parameters()).device
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    total_started = time.perf_counter()
    architecture_parameters = glyph_machine_parameter_report(controller)
    training = train_glyph_machine_controller(
        controller,
        config,
        judge=judge,
    )
    controller.requires_grad_(False)
    development = evaluate_glyph_machine_partition(
        controller,
        config,
        partition="development",
        mechanism_count=config.development_mechanisms,
        judge=judge,
    )

    def reload_and_open_final(path: Path) -> tuple[
        GlyphMachineController,
        dict[str, object],
    ]:
        save_glyph_checkpoint(path, controller, controller.initial_state())
        reloaded, reloaded_state = load_glyph_checkpoint(path, device=target_device)
        reloaded.requires_grad_(False)
        if any(parameter.requires_grad for parameter in reloaded.parameters()):
            raise RuntimeError("reloaded final controller is not frozen")
        if bool(reloaded_state.occupied.any().item()):
            raise RuntimeError("final reload carried mechanism-local fast state")
        final_result = evaluate_glyph_machine_partition(
            reloaded,
            config,
            partition="final",
            mechanism_count=config.final_mechanisms,
            judge=judge,
        )
        return reloaded, final_result

    if checkpoint_path is None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "glyph-machine-train-only.pt"
            final_controller, final = reload_and_open_final(checkpoint)
    else:
        checkpoint = Path(checkpoint_path)
        final_controller, final = reload_and_open_final(checkpoint)

    total_scalar_calls = (
        int(training["scalar_judge_calls"])
        + int(development["scalar_judge_calls"])
        + int(final["scalar_judge_calls"])
    )
    expected_scalar_calls = (
        int(training["expected_scalar_judge_calls"])
        + int(development["expected_scalar_judge_calls"])
        + int(final["expected_scalar_judge_calls"])
    )
    if total_scalar_calls != expected_scalar_calls:
        raise RuntimeError("experiment scalar accounting is inconsistent")
    result: dict[str, object] = {
        "result_version": _RESULT_VERSION,
        "config": asdict(config),
        "parameters": architecture_parameters,
        "training": training,
        "development": development,
        "final": final,
        "checkpoint_reload_before_final": True,
        "final_slow_weights_frozen": not any(
            parameter.requires_grad for parameter in final_controller.parameters()
        ),
        "total_scalar_judge_calls": total_scalar_calls,
        "expected_scalar_judge_calls": expected_scalar_calls,
        "elapsed_seconds": time.perf_counter() - total_started,
        "device": str(target_device),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(target_device))
            if target_device.type == "cuda"
            else 0
        ),
    }
    if result_path is not None:
        write_glyph_machine_result(result_path, result)
    return result


def write_glyph_machine_result(
    path: str | Path,
    result: Mapping[str, object],
) -> None:
    target = Path(path)
    encoded = json.dumps(
        dict(result),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    target.write_text(encoded + "\n", encoding="utf-8")


def _experiment_stream(
    config: GlyphMachineExperimentConfig,
    *,
    partition: str,
    mechanism_index: int,
    commitment: str,
    epoch: int,
) -> GlyphMachineTraceStream:
    stream_seed = _experiment_seed(
        config.seed,
        f"{partition}-stream",
        epoch,
        mechanism_index,
    )
    surface_seed = _experiment_seed(
        config.seed,
        f"{partition}-surface",
        epoch,
        mechanism_index,
    )
    return make_glyph_machine_trace_stream(
        stream_seed,
        surface_seed=surface_seed,
        supports=config.supports_per_mechanism,
        queries=config.queries_per_mechanism,
        observations_per_support=config.observations_per_support,
        mechanism_commitment=commitment,
        mechanism_partition=partition,
    )


def _experiment_seed(
    seed: int,
    scope: str,
    epoch: int,
    mechanism_index: int,
) -> int:
    material = (
        f"project-angler.glyph-machine.experiment.v1\x00{seed}\x00{scope}\x00"
        f"{epoch}\x00{mechanism_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _evaluation_summary(
    rows: Sequence[Mapping[str, float | str]],
) -> dict[str, object]:
    metric_names = (
        "correct",
        "no_trace",
        "wrong_trace",
        "reversible_removed",
        "correct_over_no_trace",
        "correct_over_wrong_trace",
        "reversible_contribution",
    )
    summary: dict[str, object] = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        summary[name] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
    summary["positive_correct_over_no_trace_mechanisms"] = sum(
        float(row["correct_over_no_trace"]) > 0.0 for row in rows
    )
    return summary


def _public_transitions(task: PublicGlyphMachineTask) -> tuple[Transition, ...]:
    return tuple(
        transition
        for trace in task.observations
        for transition in trace.transitions
    )


def _unique_public_transitions(
    task: PublicGlyphMachineTask,
) -> tuple[Transition, ...]:
    selected: list[Transition] = []
    seen: set[tuple[str, str, str]] = set()
    for transition in _public_transitions(task):
        identity = (
            transition.before.digest,
            transition.action.schema.digest,
            transition.after.digest,
        )
        if identity not in seen:
            seen.add(identity)
            selected.append(transition)
    return tuple(selected)


def _public_transition_ids(
    transition: Transition,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode only public source/action/successor identities for eligibility."""

    if not isinstance(transition, Transition):
        raise TypeError("public transition identity requires a Transition")
    source_action = _public_id_words(
        _PUBLIC_EVENT_KEY_DOMAIN,
        (transition.before.digest, transition.action.schema.digest),
        device=device,
    )
    successor = _public_id_words(
        _PUBLIC_SUCCESSOR_DOMAIN,
        (transition.after.digest,),
        device=device,
    )
    return source_action, successor


def _public_id_words(
    domain: bytes,
    values: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    encoded = json.dumps(
        tuple(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = hashlib.sha256(domain + encoded).digest()
    words = tuple(
        int.from_bytes(raw[index : index + 8], "big", signed=True)
        for index in range(0, len(raw), 8)
    )
    return torch.tensor(words, device=device, dtype=torch.long)


def _clone_glyph_state(state: GlyphAssociativeState) -> GlyphAssociativeState:
    """Clone a live state without detaching its differentiable event values."""

    return GlyphAssociativeState(
        keys=state.keys.clone(),
        values=state.values.clone(),
        occupied=state.occupied.clone(),
        write_counts=state.write_counts.clone(),
        public_source_action_ids=state.public_source_action_ids.clone(),
        public_successor_ids=state.public_successor_ids.clone(),
        trace_cursor=state.trace_cursor.clone(),
        outcome_cursor=state.outcome_cursor.clone(),
    )


def _trace_lane_equal(
    left: GlyphAssociativeState,
    right: GlyphAssociativeState,
) -> bool:
    if left.keys.shape != right.keys.shape:
        return False
    trace_slots = left.slot_count // 2
    trace_range = slice(0, trace_slots)
    return all(
        torch.equal(a, b)
        for a, b in (
            (left.keys[:, trace_range], right.keys[:, trace_range]),
            (left.values[:, trace_range], right.values[:, trace_range]),
            (left.occupied[:, trace_range], right.occupied[:, trace_range]),
            (
                left.write_counts[:, trace_range],
                right.write_counts[:, trace_range],
            ),
            (
                left.public_source_action_ids[:, trace_range],
                right.public_source_action_ids[:, trace_range],
            ),
            (
                left.public_successor_ids[:, trace_range],
                right.public_successor_ids[:, trace_range],
            ),
            (left.trace_cursor, right.trace_cursor),
        )
    )


def _orthogonal_bounded_anchor(
    anchor: torch.Tensor,
    learned_residual: torch.Tensor,
) -> torch.Tensor:
    """Preserve ``anchor`` plus a smooth orthogonal residual below 0.25 norm."""

    if (
        not isinstance(anchor, torch.Tensor)
        or not isinstance(learned_residual, torch.Tensor)
        or anchor.shape != learned_residual.shape
        or anchor.ndim < 1
        or not anchor.is_floating_point()
        or learned_residual.dtype != anchor.dtype
        or learned_residual.device != anchor.device
    ):
        raise ValueError("anchor and learned residual must share one floating shape")
    epsilon = torch.finfo(anchor.dtype).eps
    anchor_norm = anchor.norm(dim=-1, keepdim=True)
    unit_anchor = anchor / anchor_norm.clamp_min(epsilon)
    orthogonal = learned_residual - (
        learned_residual * unit_anchor
    ).sum(dim=-1, keepdim=True) * unit_anchor
    residual_norm = orthogonal.norm(dim=-1, keepdim=True)
    limit = residual_norm.new_tensor(_ANCHOR_RESIDUAL_LIMIT)
    smooth_norm = limit * torch.tanh(residual_norm / limit)
    bounded = orthogonal * (
        smooth_norm / residual_norm.clamp_min(epsilon)
    )
    return anchor + bounded


def _comparison_features(
    source: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    if source.shape != goal.shape or source.ndim != 2:
        raise ValueError("comparison endpoints must share rank-two shape")
    return torch.cat((source, goal, source * goal, goal - source), dim=-1)


def _public_goal_state(task: PublicGlyphMachineTask) -> State:
    matches = tuple(
        state for state in task.states if state.records == task.goal.required
    )
    if len(matches) != 1:
        raise ValueError("public goal must identify one declared state")
    return matches[0]


def _state_text(state: State) -> str:
    rows = [state.namespace]
    rows.extend(
        f"{record.predicate}({','.join(record.arguments)})"
        for record in state.records
    )
    return "|".join(rows)


def _action_text(action: ActionSchema) -> str:
    parameters = ",".join(
        f"{parameter.name}:{parameter.type_name}"
        for parameter in action.parameters
    )
    description = "<none>" if action.description is None else action.description
    return f"{action.name}({parameters})|description:{description}"


def _state_address_text(state: State) -> str:
    return f"source_state|{_state_text(state)}"


def _action_address_text(action: ActionSchema) -> str:
    return f"action_schema|{_action_text(action)}"


def _pair_key_text(state: State, action: ActionSchema) -> str:
    state_text = _state_text(state)
    action_text = _action_text(action)
    # The reversible text encoding forms a single lexical cross-feature for
    # the full pair.  Repetition gives this complete pair feature enough mass
    # that shared state or action vocabulary cannot dominate the frozen hash
    # row; it is not a digest or a learned/equality-based route.
    joint = f"{state_text}\x00{action_text}".encode("utf-8").hex()
    cross_features = " ".join(f"joint:{joint}" for _ in range(8))
    return (
        f"transition_pair|source|{state_text}|action|{action_text}|"
        f"{cross_features}"
    )


def _stop_key_text(current: State, goal: State) -> str:
    current_text = _state_text(current)
    goal_text = _state_text(goal)
    joint = f"{current_text}\x00{goal_text}".encode("utf-8").hex()
    cross_features = " ".join(f"stop_joint:{joint}" for _ in range(8))
    return (
        f"stop_event|current|{current_text}|goal|{goal_text}|{cross_features}"
    )


def _public_task_digest(task: PublicGlyphMachineTask) -> str:
    encoded = json.dumps(
        task.to_canonical(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(_TASK_DIGEST_DOMAIN + encoded).hexdigest()


def _validate_reward(reward: float) -> float:
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("scalar reward must be numeric")
    numeric = float(reward)
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError("scalar reward must be terminal zero or one")
    return numeric


def _validate_controller_task_state(
    controller: GlyphMachineController,
    task: PublicGlyphMachineTask,
    state: GlyphAssociativeState,
) -> None:
    if not isinstance(controller, GlyphMachineController):
        raise TypeError("controller must be a GlyphMachineController")
    if not isinstance(task, PublicGlyphMachineTask):
        raise TypeError("task must be a PublicGlyphMachineTask")
    if not isinstance(state, GlyphAssociativeState):
        raise TypeError("state must be a GlyphAssociativeState")
    if state.batch_size != 1 or state.width != controller.profile.width:
        raise ValueError("competence state topology does not match controller")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(GLYPH_MACHINE_PROFILES),
        default="smoke",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    controller = build_glyph_machine_controller(args.profile, device=args.device)
    print(json.dumps(glyph_machine_parameter_report(controller), sort_keys=True))


__all__ = [
    "GLYPH_MACHINE_PROFILES",
    "GlyphAssociativeMemory",
    "GlyphAssociativeRead",
    "GlyphAssociativeState",
    "GlyphAssociativeWrite",
    "GlyphBackwardProcedureReasoner",
    "GlyphMachineController",
    "GlyphMachineExperimentConfig",
    "GlyphMachineRunProfile",
    "GlyphNeuralRollout",
    "GlyphScalarFeedback",
    "GlyphStepScores",
    "GlyphTaskEncoding",
    "GlyphTraceAcquisition",
    "GlyphTransitionLattice",
    "acquire_and_score_public_traces",
    "acquire_public_traces",
    "apply_scalar_procedure_feedback",
    "build_glyph_machine_controller",
    "build_glyph_machine_evaluation_arms",
    "centered_trajectory_preference_loss",
    "default_glyph_machine_experiment_config",
    "evaluate_glyph_machine_partition",
    "glyph_associative_state_digest",
    "glyph_machine_parameter_report",
    "load_glyph_checkpoint",
    "rollout_glyph_procedure",
    "run_glyph_machine_experiment",
    "save_glyph_checkpoint",
    "scalar_outcome_loss",
    "snapshot_glyph_state",
    "restore_glyph_state",
    "train_glyph_machine_controller",
    "write_glyph_machine_result",
]


if __name__ == "__main__":
    main()
