"""Learned paired public-relation addressing for persistent credit memory.

V19 preserves the strict factorized V17 memory and adds one outcome-blind
public relation row per slot.  A shared symmetric comparator supplies a
bounded residual to the semantic content logit.  Outcomes still affect only
stored value content; deterministic code only allocates, snapshots, replays,
and applies declared measurement lesions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .shared_semantic_metric_credit_memory import SharedSemanticMetricCreditMemoryCore
from .strict_key_value_credit_memory import (
    MEMORY_RANK,
    MEMORY_SLOTS,
    RELATION_WIDTH,
    _require_bool,
)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True, slots=True)
class PairedPublicRelationCreditState:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    acquisition: torch.Tensor
    public_relations: torch.Tensor
    step: int

    @property
    def bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.keys,
                self.values,
                self.usage,
                self.acquisition,
                self.public_relations,
            )
        )

    def detached_clone(self) -> "PairedPublicRelationCreditState":
        return PairedPublicRelationCreditState(
            keys=self.keys.detach().clone(),
            values=self.values.detach().clone(),
            usage=self.usage.detach().clone(),
            acquisition=self.acquisition.detach().clone(),
            public_relations=self.public_relations.detach().clone(),
            step=self.step,
        )


@dataclass(frozen=True, slots=True)
class PairedPublicRelationCreditSnapshot:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    acquisition: torch.Tensor
    public_relations: torch.Tensor
    step: int


@dataclass(frozen=True, slots=True)
class PairedPublicRelationCreditOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    residuals: torch.Tensor
    read_queries: torch.Tensor
    read_weights: torch.Tensor
    read_values: torch.Tensor
    temporal_hidden: torch.Tensor
    write_keys: torch.Tensor
    read_strengths: torch.Tensor
    write_strengths: torch.Tensor
    erase_gates: torch.Tensor
    observed_relations: torch.Tensor
    observed_temporal: torch.Tensor
    pair_query_relations: torch.Tensor
    semantic_content_logits: torch.Tensor
    pair_raw_scores: torch.Tensor
    pair_address_residuals: torch.Tensor
    address_logits: torch.Tensor
    state_step: int
    read_enabled: bool
    pair_residual_enabled: bool

    def row(self, index: int) -> "PairedPublicRelationCreditOutput":
        if type(index) is not int or not 0 <= index < self.logits.shape[0]:
            raise IndexError("paired credit-output row is outside the batch")
        fields = {
            name: getattr(self, name)[index : index + 1]
            for name in (
                "logits",
                "base_logits",
                "residuals",
                "read_queries",
                "read_weights",
                "read_values",
                "temporal_hidden",
                "write_keys",
                "read_strengths",
                "write_strengths",
                "erase_gates",
                "observed_relations",
                "observed_temporal",
                "pair_query_relations",
                "semantic_content_logits",
                "pair_raw_scores",
                "pair_address_residuals",
                "address_logits",
            )
        }
        return PairedPublicRelationCreditOutput(
            **fields,
            state_step=self.state_step,
            read_enabled=self.read_enabled,
            pair_residual_enabled=self.pair_residual_enabled,
        )


@dataclass(frozen=True, slots=True)
class PairedPublicRelationCreditEvent:
    relation_features: torch.Tensor
    temporal_features: torch.Tensor
    base_logits: torch.Tensor
    pair_query_features: torch.Tensor
    outcomes: torch.Tensor
    evidence_refs: tuple[str, ...]
    allocation_index: int
    read_strength: float
    write_strength: float
    erase_gate: float
    write_key: torch.Tensor
    write_value: torch.Tensor
    public_relation: torch.Tensor
    pair_residual_enabled: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "angler.paired-public-relation-credit-event.v1",
            "relation_features": self.relation_features.tolist(),
            "temporal_features": self.temporal_features.tolist(),
            "base_logits": self.base_logits.tolist(),
            "pair_query_features": self.pair_query_features.tolist(),
            "outcomes": self.outcomes.tolist(),
            "evidence_refs": list(self.evidence_refs),
            "allocation_index": self.allocation_index,
            "read_strength": self.read_strength,
            "write_strength": self.write_strength,
            "erase_gate": self.erase_gate,
            "write_key": self.write_key.tolist(),
            "write_value": self.write_value.tolist(),
            "public_relation": self.public_relation.tolist(),
            "pair_residual_enabled": self.pair_residual_enabled,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PairedPublicRelationCreditEvent":
        if record.get("schema") != "angler.paired-public-relation-credit-event.v1":
            raise ValueError("paired credit-event schema is unsupported")
        try:
            event = cls(
                relation_features=torch.tensor(record["relation_features"], dtype=torch.float32),
                temporal_features=torch.tensor(record["temporal_features"], dtype=torch.float32),
                base_logits=torch.tensor(record["base_logits"], dtype=torch.float32),
                pair_query_features=torch.tensor(record["pair_query_features"], dtype=torch.float32),
                outcomes=torch.tensor(record["outcomes"], dtype=torch.float32),
                evidence_refs=tuple(record["evidence_refs"]),
                allocation_index=int(record["allocation_index"]),
                read_strength=float(record["read_strength"]),
                write_strength=float(record["write_strength"]),
                erase_gate=float(record["erase_gate"]),
                write_key=torch.tensor(record["write_key"], dtype=torch.float32),
                write_value=torch.tensor(record["write_value"], dtype=torch.float32),
                public_relation=torch.tensor(record["public_relation"], dtype=torch.float32),
                pair_residual_enabled=record["pair_residual_enabled"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("paired credit-event record is malformed") from error
        event._validate_record_fields()
        return event

    def _validate_record_fields(self) -> None:
        if (
            self.relation_features.shape != (1, RELATION_WIDTH)
            or self.temporal_features.ndim != 2
            or self.temporal_features.shape[0] != 1
            or self.base_logits.shape != (1,)
            or self.pair_query_features.shape != (1, RELATION_WIDTH)
            or self.outcomes.shape != (1,)
            or len(self.evidence_refs) != 1
            or not self.evidence_refs[0]
            or type(self.allocation_index) is not int
            or not 0 <= self.allocation_index < MEMORY_SLOTS
            or self.write_key.shape != (1, MEMORY_RANK)
            or self.write_value.shape != (1, MEMORY_RANK)
            or self.public_relation.shape != (1, RELATION_WIDTH)
        ):
            raise ValueError("paired credit-event fields are invalid")
        _require_bool("event.pair_residual_enabled", self.pair_residual_enabled)
        scalars = (self.read_strength, self.write_strength, self.erase_gate)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("paired credit-event gates are non-finite")
        if self.read_strength < 1.0 or not (
            0.0 <= self.write_strength <= 1.0 and 0.0 <= self.erase_gate <= 1.0
        ):
            raise ValueError("paired credit-event gates are invalid")
        tensors = (
            self.relation_features,
            self.temporal_features,
            self.base_logits,
            self.pair_query_features,
            self.outcomes,
            self.write_key,
            self.write_value,
            self.public_relation,
        )
        if not all(value.is_floating_point() for value in tensors) or not all(
            bool(torch.isfinite(value).all().item()) for value in tensors
        ):
            raise ValueError("paired credit-event tensors must be finite floats")
        if not bool(((self.outcomes == -1) | (self.outcomes == 1)).all().item()):
            raise ValueError("paired credit-event outcomes must be +/-1")
        if not torch.equal(self.public_relation, self.relation_features):
            raise ValueError("paired credit-event sidecar must equal the observed relation")


class PairedPublicRelationCreditMemoryCore(SharedSemanticMetricCreditMemoryCore):
    """V17 memory with a learned symmetric public-relation comparator."""

    def __init__(
        self,
        *,
        temporal_width: int,
        relation_width: int = RELATION_WIDTH,
        rank: int = MEMORY_RANK,
        memory_slots: int = MEMORY_SLOTS,
        maximum_residual: float = 4.0,
        maximum_pair_residual: float = 2.0,
        pair_residual_enabled_by_default: bool = True,
    ) -> None:
        super().__init__(
            temporal_width=temporal_width,
            relation_width=relation_width,
            rank=rank,
            memory_slots=memory_slots,
            maximum_residual=maximum_residual,
        )
        if maximum_pair_residual != 2.0:
            raise ValueError("V19 requires an exact pair-residual bound of 2.0")
        _require_bool(
            "pair_residual_enabled_by_default", pair_residual_enabled_by_default
        )
        self.maximum_pair_residual = 2.0
        # Arm selection is not a parameter or buffer, so full and unary cores
        # retain byte-identical model state while inherited evaluators can call
        # predict without V19-specific kwargs.
        self.pair_residual_enabled_by_default = pair_residual_enabled_by_default
        self.paired_relation_scorer = nn.Sequential(
            nn.LayerNorm(relation_width * 3),
            nn.Linear(relation_width * 3, relation_width),
            nn.SiLU(),
            nn.Linear(relation_width, 1, bias=False),
        )
        nn.init.zeros_(self.paired_relation_scorer[-1].weight)

    def initial_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PairedPublicRelationCreditState:
        reference = next(self.parameters())
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        if resolved_device != reference.device or resolved_dtype != reference.dtype:
            raise ValueError("initial state must match core device/dtype")
        return PairedPublicRelationCreditState(
            keys=torch.zeros(self.memory_slots, self.rank, device=resolved_device, dtype=resolved_dtype),
            values=torch.zeros(self.memory_slots, self.rank, device=resolved_device, dtype=resolved_dtype),
            usage=torch.zeros(self.memory_slots, device=resolved_device, dtype=resolved_dtype),
            acquisition=torch.zeros(self.memory_slots, device=resolved_device, dtype=resolved_dtype),
            public_relations=torch.zeros(
                self.memory_slots,
                self.relation_width,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            step=0,
        )

    initial_ability_state = initial_state

    def paired_address_residuals(
        self,
        pair_query_features: torch.Tensor,
        state: PairedPublicRelationCreditState,
        *,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw and bounded per-slot pair scores, masked to occupied slots."""

        _require_bool("enabled", enabled)
        self.validate_state(state)
        pair_query = self._validate_pair_queries(pair_query_features)
        shape = (pair_query.shape[0], self.memory_slots)
        if not enabled:
            zero = pair_query.new_zeros(shape)
            return zero, zero
        query = pair_query.unsqueeze(1)
        stored = state.public_relations.unsqueeze(0)
        features = torch.cat(
            (0.5 * (query + stored), (query - stored).abs(), query * stored),
            dim=-1,
        )
        raw = self.paired_relation_scorer(features).squeeze(-1)
        residual = self.maximum_pair_residual * torch.tanh(raw)
        occupied = state.usage.gt(0.0).unsqueeze(0)
        raw = torch.where(occupied, raw, torch.zeros_like(raw))
        residual = torch.where(occupied, residual, torch.zeros_like(residual))
        return raw, residual

    def predict(
        self,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: PairedPublicRelationCreditState | None = None,
        read_enabled: bool = True,
        pair_residual_enabled: bool | None = None,
        pair_query_features: torch.Tensor | None = None,
    ) -> PairedPublicRelationCreditOutput:
        """Predict before feedback, optionally disabling only the V19 residual."""

        _require_bool("read_enabled", read_enabled)
        enabled = (
            self.pair_residual_enabled_by_default
            if pair_residual_enabled is None
            else pair_residual_enabled
        )
        _require_bool("pair_residual_enabled", enabled)
        self._validate_observations(relation_features, temporal_features, base_logits)
        current = self.initial_state() if state is None else state
        self.validate_state(current)

        relations = relation_features.detach()
        temporal = temporal_features.detach()
        base = base_logits.detach()
        pair_query = relations if pair_query_features is None else self._validate_pair_queries(
            pair_query_features
        )
        if pair_query.shape != relations.shape:
            raise ValueError("pair query features must align with relation features")

        if not enabled:
            inherited = super().predict(
                relations,
                temporal,
                base,
                state=current,
                read_enabled=read_enabled,
            )
            zero = inherited.read_weights.new_zeros(inherited.read_weights.shape)
            output = PairedPublicRelationCreditOutput(
                logits=inherited.logits,
                base_logits=inherited.base_logits,
                residuals=inherited.residuals,
                read_queries=inherited.read_queries,
                read_weights=inherited.read_weights,
                read_values=inherited.read_values,
                temporal_hidden=inherited.temporal_hidden,
                write_keys=inherited.write_keys,
                read_strengths=inherited.read_strengths,
                write_strengths=inherited.write_strengths,
                erase_gates=inherited.erase_gates,
                observed_relations=inherited.observed_relations,
                observed_temporal=inherited.observed_temporal,
                pair_query_relations=pair_query,
                semantic_content_logits=self._semantic_content_logits(
                    inherited.read_queries,
                    inherited.read_strengths,
                    current,
                ),
                pair_raw_scores=zero,
                pair_address_residuals=zero,
                address_logits=self._address_logits_without_pair(
                    inherited.read_queries,
                    inherited.read_strengths,
                    current,
                ),
                state_step=inherited.state_step,
                read_enabled=inherited.read_enabled,
                pair_residual_enabled=False,
            )
            self._validate_paired_output(output)
            return output

        semantic = self.semantic_codes(relations)
        temporal_hidden = self.temporal_network(temporal)
        strength_input = torch.cat((relations, temporal_hidden), dim=-1)
        read_strengths = F.softplus(
            self.read_strength_network(strength_input).squeeze(-1)
        ) + 1.0
        write_strengths = torch.sigmoid(
            self.write_strength_network(strength_input).squeeze(-1)
        )
        erase_gates = torch.sigmoid(self.erase_network(strength_input).squeeze(-1))

        normalized_queries = F.normalize(semantic, dim=-1, eps=1.0e-6)
        normalized_keys = F.normalize(current.keys, dim=-1, eps=1.0e-6)
        semantic_content = torch.einsum(
            "br,sr->bs", normalized_queries, normalized_keys
        ) * read_strengths.unsqueeze(-1)
        pair_raw, pair_residuals = self.paired_address_residuals(
            pair_query,
            current,
            enabled=enabled,
        )
        coordinate = current.usage.new_tensor(current.step / self.memory_slots)
        age = (coordinate - current.acquisition).clamp(0.0, 1.0)
        slot_context = torch.stack((current.usage, current.acquisition, age), dim=-1)
        temporal_bias = self.slot_temporal_network(slot_context).squeeze(-1)
        address_logits = (
            semantic_content
            + pair_residuals
            + temporal_bias
            + torch.log(current.usage.clamp_min(1.0e-6))
        )
        read_weights = torch.softmax(address_logits, dim=-1)
        if current.step == 0 or not read_enabled:
            read_weights = torch.zeros_like(read_weights)
        read_values = read_weights @ current.values
        residuals = self.decode_retrieved_values(read_values)
        if not read_enabled:
            residuals = torch.zeros_like(residuals)
        output = PairedPublicRelationCreditOutput(
            logits=base + residuals,
            base_logits=base,
            residuals=residuals,
            read_queries=semantic,
            read_weights=read_weights,
            read_values=read_values,
            temporal_hidden=temporal_hidden,
            write_keys=semantic,
            read_strengths=read_strengths,
            write_strengths=write_strengths,
            erase_gates=erase_gates,
            observed_relations=relations,
            observed_temporal=temporal,
            pair_query_relations=pair_query,
            semantic_content_logits=semantic_content,
            pair_raw_scores=pair_raw,
            pair_address_residuals=pair_residuals,
            address_logits=address_logits,
            state_step=current.step,
            read_enabled=read_enabled,
            pair_residual_enabled=enabled,
        )
        self._validate_paired_output(output)
        return output

    forward = predict

    def apply_feedback(
        self,
        state: PairedPublicRelationCreditState,
        output: PairedPublicRelationCreditOutput,
        outcomes: torch.Tensor,
        *,
        evidence_refs: str | Sequence[str],
        detach_state: bool = True,
    ) -> tuple[PairedPublicRelationCreditState, PairedPublicRelationCreditEvent]:
        """Write strict credit and exact-overwrite the outcome-blind sidecar."""

        _require_bool("detach_state", detach_state)
        self.validate_state(state)
        self._validate_paired_output(output)
        self._validate_outcomes(outcomes, output.logits)
        if output.logits.shape[0] != 1:
            raise ValueError("one persistent state accepts one event at a time")
        if output.state_step != state.step:
            raise ValueError("paired credit output is stale for the supplied state")
        if state.step >= self.memory_slots:
            raise OverflowError("paired credit memory capacity is exhausted")
        refs = self._normalize_evidence_refs(evidence_refs, batch=1)

        write_value = self.outcome_values(outcomes)
        allocation = self._least_unused_allocation(state.usage)
        slot_weights = output.write_strengths[0] * allocation
        slot = slot_weights.unsqueeze(-1)
        write_key = output.write_keys[0].unsqueeze(0)
        value = write_value[0].unsqueeze(0)
        keys = (1.0 - slot) * state.keys + slot * write_key
        erased = state.values * (1.0 - slot * output.erase_gates[0])
        values = (1.0 - slot) * erased + slot * value
        usage = state.usage + slot_weights * (1.0 - state.usage)
        coordinate = state.usage.new_tensor((state.step + 1) / self.memory_slots)
        acquisition = (1.0 - slot_weights) * state.acquisition + slot_weights * coordinate
        winner = int(allocation.argmax().item())
        public_relations = state.public_relations.detach().clone()
        public_relations[winner] = output.observed_relations[0].detach()
        updated = PairedPublicRelationCreditState(
            keys,
            values,
            usage,
            acquisition,
            public_relations,
            state.step + 1,
        )
        self.validate_state(updated)
        result = updated.detached_clone() if detach_state else updated
        event = PairedPublicRelationCreditEvent(
            relation_features=output.observed_relations.detach().cpu().clone(),
            temporal_features=output.observed_temporal.detach().cpu().clone(),
            base_logits=output.base_logits.detach().cpu().clone(),
            pair_query_features=output.pair_query_relations.detach().cpu().clone(),
            outcomes=outcomes.detach().cpu().clone(),
            evidence_refs=refs,
            allocation_index=winner,
            read_strength=float(output.read_strengths[0].detach().item()),
            write_strength=float(output.write_strengths[0].detach().item()),
            erase_gate=float(output.erase_gates[0].detach().item()),
            write_key=write_key.detach().cpu().clone(),
            write_value=value.detach().cpu().clone(),
            public_relation=output.observed_relations.detach().cpu().clone(),
            pair_residual_enabled=output.pair_residual_enabled,
        )
        event._validate_record_fields()
        return result, event

    def capture_state(
        self, state: PairedPublicRelationCreditState
    ) -> PairedPublicRelationCreditSnapshot:
        self.validate_state(state)
        return PairedPublicRelationCreditSnapshot(
            state.keys.detach().cpu().clone(),
            state.values.detach().cpu().clone(),
            state.usage.detach().cpu().clone(),
            state.acquisition.detach().cpu().clone(),
            state.public_relations.detach().cpu().clone(),
            state.step,
        )

    def restore_state(
        self, snapshot: PairedPublicRelationCreditSnapshot
    ) -> PairedPublicRelationCreditState:
        if not isinstance(snapshot, PairedPublicRelationCreditSnapshot):
            raise TypeError("snapshot must be PairedPublicRelationCreditSnapshot")
        reference = next(self.parameters())
        state = PairedPublicRelationCreditState(
            snapshot.keys.detach().to(reference).clone(),
            snapshot.values.detach().to(reference).clone(),
            snapshot.usage.detach().to(reference).clone(),
            snapshot.acquisition.detach().to(reference).clone(),
            snapshot.public_relations.detach().to(reference).clone(),
            snapshot.step,
        )
        self.validate_state(state)
        return state

    def state_digest(self, state: PairedPublicRelationCreditState) -> str:
        self.validate_state(state)
        return self._digest_state(state, include_public_relations=True)

    def state_digest_without_public_relations(
        self, state: PairedPublicRelationCreditState
    ) -> str:
        self.validate_state(state)
        return self._digest_state(state, include_public_relations=False)

    def zero_state_like(
        self, state: PairedPublicRelationCreditState
    ) -> PairedPublicRelationCreditState:
        self.validate_state(state)
        return PairedPublicRelationCreditState(
            torch.zeros_like(state.keys),
            torch.zeros_like(state.values),
            torch.zeros_like(state.usage),
            torch.zeros_like(state.acquisition),
            torch.zeros_like(state.public_relations),
            0,
        )

    def deranged_public_sidecar_state(
        self, state: PairedPublicRelationCreditState
    ) -> PairedPublicRelationCreditState:
        """Apply the frozen ascending occupied-slot rotate-by-one lesion."""

        self.validate_state(state)
        occupied = torch.nonzero(state.usage.gt(0.0), as_tuple=False).flatten()
        if occupied.numel() < 2:
            raise ValueError("sidecar derangement requires at least two occupied slots")
        sidecar = state.public_relations.detach().clone()
        sidecar[occupied] = torch.roll(sidecar[occupied], shifts=-1, dims=0)
        result = PairedPublicRelationCreditState(
            state.keys.detach().clone(),
            state.values.detach().clone(),
            state.usage.detach().clone(),
            state.acquisition.detach().clone(),
            sidecar,
            state.step,
        )
        self.validate_state(result)
        return result

    def replay(
        self,
        events: Sequence[PairedPublicRelationCreditEvent],
        *,
        state: PairedPublicRelationCreditState | None = None,
        detach_state: bool = True,
    ) -> tuple[PairedPublicRelationCreditState, tuple[PairedPublicRelationCreditOutput, ...]]:
        _require_bool("detach_state", detach_state)
        current = self.initial_state() if state is None else state
        self.validate_state(current)
        reference = next(self.parameters())
        outputs = []
        for event in events:
            if not isinstance(event, PairedPublicRelationCreditEvent):
                raise TypeError("replay accepts PairedPublicRelationCreditEvent objects")
            event._validate_record_fields()
            output = self.predict(
                event.relation_features.to(reference),
                event.temporal_features.to(reference),
                event.base_logits.to(reference),
                state=current,
            )
            current, reconstructed = self.apply_feedback(
                current,
                output,
                event.outcomes.to(reference),
                evidence_refs=event.evidence_refs,
                detach_state=detach_state,
            )
            if (
                not torch.equal(reconstructed.relation_features, event.relation_features)
                or not torch.equal(reconstructed.temporal_features, event.temporal_features)
                or not torch.equal(reconstructed.base_logits, event.base_logits)
                or not torch.equal(reconstructed.pair_query_features, event.pair_query_features)
                or not torch.equal(reconstructed.outcomes, event.outcomes)
                or reconstructed.evidence_refs != event.evidence_refs
                or reconstructed.pair_residual_enabled != event.pair_residual_enabled
                or reconstructed.allocation_index != event.allocation_index
                or abs(reconstructed.read_strength - event.read_strength) > 1.0e-6
                or abs(reconstructed.write_strength - event.write_strength) > 1.0e-6
                or abs(reconstructed.erase_gate - event.erase_gate) > 1.0e-6
                or not torch.allclose(reconstructed.write_key, event.write_key, rtol=0.0, atol=1.0e-6)
                or not torch.allclose(reconstructed.write_value, event.write_value, rtol=0.0, atol=1.0e-6)
                or not torch.equal(reconstructed.public_relation, event.public_relation)
            ):
                raise RuntimeError("paired credit-event replay diverged from evidence")
            outputs.append(output)
        return current, tuple(outputs)

    def validate_state(self, state: PairedPublicRelationCreditState) -> None:
        if not isinstance(state, PairedPublicRelationCreditState):
            raise TypeError("state must be PairedPublicRelationCreditState")
        reference = next(self.parameters())
        shapes = {
            "keys": (self.memory_slots, self.rank),
            "values": (self.memory_slots, self.rank),
            "usage": (self.memory_slots,),
            "acquisition": (self.memory_slots,),
            "public_relations": (self.memory_slots, self.relation_width),
        }
        if type(state.step) is not int or not 0 <= state.step <= self.memory_slots:
            raise ValueError("paired credit-memory step is outside capacity")
        for name, shape in shapes.items():
            value = getattr(state, name)
            if value.shape != shape:
                raise ValueError(f"paired credit-memory {name} has the wrong shape")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("paired credit-memory state must match core device/dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("paired credit-memory state must be finite")
            if name in {"keys", "values"} and not bool(
                (value.abs() <= 1.0 + 1.0e-6).all().item()
            ):
                raise ValueError(f"paired credit-memory {name} exceeded [-1,1]")
            if name in {"usage", "acquisition"} and not bool(
                ((value >= 0.0) & (value <= 1.0 + 1.0e-6)).all().item()
            ):
                raise ValueError(f"paired credit-memory {name} exceeded [0,1]")
        if state.public_relations.requires_grad or state.public_relations.grad_fn is not None:
            raise ValueError("paired credit-memory public sidecar must be detached")
        occupied_count = int(state.usage.gt(0.0).count_nonzero().item())
        if occupied_count != state.step:
            raise ValueError("paired credit-memory occupied slot count must equal step")
        unused = state.usage.eq(0.0)
        if not torch.equal(
            state.public_relations[unused], torch.zeros_like(state.public_relations[unused])
        ):
            raise ValueError("unused paired credit-memory sidecar rows must be exact zero")

    def parameter_groups(self) -> dict[str, tuple[str, ...]]:
        prefixes = {
            "semantic_metric": ("semantic_metric_network.",),
            "paired_public_relation": ("paired_relation_scorer.",),
            "address_strength_and_recency": (
                "temporal_network.",
                "read_strength_network.",
                "write_strength_network.",
                "erase_network.",
                "slot_temporal_network.",
            ),
            "outcome_value": ("outcome_embedding.", "outcome_value_network."),
            "value_readout": ("residual_network.",),
        }
        names = tuple(name for name, _ in self.named_parameters())
        groups = {
            group: tuple(
                name
                for name in names
                if any(name.startswith(prefix) for prefix in group_prefixes)
            )
            for group, group_prefixes in prefixes.items()
        }
        flattened = tuple(name for group in groups.values() for name in group)
        if len(flattened) != len(names) or set(flattened) != set(names):
            raise RuntimeError("V19 parameter ownership is incomplete")
        return groups

    def architecture_report(self) -> dict[str, Any]:
        return {
            "parameter_groups": self.parameter_groups(),
            "parameter_count": self.parameter_count,
            "memory_slots": self.memory_slots,
            "rank": self.rank,
            "relation_width": self.relation_width,
            "semantic_metric_inputs": ("detached_relation",),
            "semantic_metric_shared_for": ("read_query", "write_key"),
            "semantic_metric_normalized": True,
            "public_temporal_roles": (
                "read_strength",
                "write_strength",
                "inactive_fresh_slot_erase",
            ),
            "state_coordinate_roles": ("slot_recency",),
            "public_sidecar_shape": (self.memory_slots, self.relation_width),
            "public_sidecar_write": "exact_detached_winner_overwrite",
            "pair_features": ("half_sum", "absolute_difference", "product"),
            "pair_scorer_inputs": ("detached_query_relation", "detached_stored_relation"),
            "pair_residual_bound": self.maximum_pair_residual,
            "unused_pair_residual_exact_zero": True,
            "independent_query_or_key_networks": False,
            "temporal_affects_semantic_code_or_decoder": False,
            "outcome_affects_semantic_code_or_gates": False,
            "outcome_affects_address_or_sidecar": False,
            "metadata_inputs": False,
            "stored_value_inputs": ("transformed_outcome_embedding",),
            "residual_decoder_inputs": ("retrieved_value",),
            "generic_allocation": "stable_least_unused",
            "erase_gate_role": "fresh_slot_noop_under_stable_least_unused",
            "erase_gate_active_before_capacity": False,
            "learned_retention_evidence_claimed": False,
            "v13_parameters_owned": False,
        }

    parameter_report = architecture_report

    def _validate_pair_queries(self, pair_query_features: torch.Tensor) -> torch.Tensor:
        reference = next(self.parameters())
        if (
            pair_query_features.ndim != 2
            or pair_query_features.shape[-1] != self.relation_width
            or pair_query_features.shape[0] == 0
            or pair_query_features.device != reference.device
            or pair_query_features.dtype != reference.dtype
            or not pair_query_features.is_floating_point()
            or not bool(torch.isfinite(pair_query_features).all().item())
        ):
            raise ValueError("pair query features must be finite [batch,64] core tensors")
        return pair_query_features.detach()

    def _semantic_content_logits(
        self,
        semantic_queries: torch.Tensor,
        read_strengths: torch.Tensor,
        state: PairedPublicRelationCreditState,
    ) -> torch.Tensor:
        queries = F.normalize(semantic_queries, dim=-1, eps=1.0e-6)
        keys = F.normalize(state.keys, dim=-1, eps=1.0e-6)
        return torch.einsum("br,sr->bs", queries, keys) * read_strengths.unsqueeze(-1)

    def _address_logits_without_pair(
        self,
        semantic_queries: torch.Tensor,
        read_strengths: torch.Tensor,
        state: PairedPublicRelationCreditState,
    ) -> torch.Tensor:
        content = self._semantic_content_logits(semantic_queries, read_strengths, state)
        coordinate = state.usage.new_tensor(state.step / self.memory_slots)
        age = (coordinate - state.acquisition).clamp(0.0, 1.0)
        slot_context = torch.stack((state.usage, state.acquisition, age), dim=-1)
        temporal_bias = self.slot_temporal_network(slot_context).squeeze(-1)
        return content + temporal_bias + torch.log(state.usage.clamp_min(1.0e-6))

    def _validate_paired_output(self, output: PairedPublicRelationCreditOutput) -> None:
        if not isinstance(output, PairedPublicRelationCreditOutput):
            raise TypeError("output must be PairedPublicRelationCreditOutput")
        batch = output.logits.shape[0]
        if batch == 0 or output.logits.ndim != 1:
            raise ValueError("paired credit output logits must be a nonempty vector")
        vector = (batch, self.rank)
        slots = (batch, self.memory_slots)
        scalars = (
            "logits",
            "base_logits",
            "residuals",
            "read_strengths",
            "write_strengths",
            "erase_gates",
        )
        for name in scalars:
            if getattr(output, name).shape != (batch,):
                raise ValueError(f"paired credit output {name} has the wrong shape")
        for name in ("read_queries", "read_values", "temporal_hidden", "write_keys"):
            if getattr(output, name).shape != vector:
                raise ValueError(f"paired credit output {name} has the wrong shape")
        for name in (
            "read_weights",
            "semantic_content_logits",
            "pair_raw_scores",
            "pair_address_residuals",
            "address_logits",
        ):
            if getattr(output, name).shape != slots:
                raise ValueError(f"paired credit output {name} has the wrong shape")
        if output.observed_relations.shape != (batch, self.relation_width):
            raise ValueError("paired credit output relation observation has the wrong shape")
        if output.pair_query_relations.shape != (batch, self.relation_width):
            raise ValueError("paired credit output pair query has the wrong shape")
        if output.observed_temporal.shape != (batch, self.temporal_width):
            raise ValueError("paired credit output temporal observation has the wrong shape")
        if type(output.state_step) is not int or not 0 <= output.state_step <= self.memory_slots:
            raise ValueError("paired credit output state step is invalid")
        _require_bool("output.read_enabled", output.read_enabled)
        _require_bool("output.pair_residual_enabled", output.pair_residual_enabled)
        reference = next(self.parameters())
        tensors = tuple(
            getattr(output, name)
            for name in (
                *scalars,
                "read_queries",
                "read_weights",
                "read_values",
                "temporal_hidden",
                "write_keys",
                "observed_relations",
                "observed_temporal",
                "pair_query_relations",
                "semantic_content_logits",
                "pair_raw_scores",
                "pair_address_residuals",
                "address_logits",
            )
        )
        if any(value.device != reference.device or value.dtype != reference.dtype for value in tensors):
            raise ValueError("paired credit output must match core device/dtype")
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ValueError("paired credit output must be finite")
        if not bool((output.read_strengths >= 1.0).all().item()):
            raise ValueError("paired credit read strengths are invalid")
        for name in ("write_strengths", "erase_gates"):
            value = getattr(output, name)
            if not bool(((value >= 0.0) & (value <= 1.0)).all().item()):
                raise ValueError(f"paired credit {name} exceeded [0,1]")
        if not bool((output.pair_address_residuals.abs() <= 2.0 + 1.0e-6).all().item()):
            raise ValueError("paired address residual exceeded [-2,2]")

    def _digest_state(
        self,
        state: PairedPublicRelationCreditState,
        *,
        include_public_relations: bool,
    ) -> str:
        digest = hashlib.sha256()
        schema = (
            b"angler.paired-public-relation-credit-state.v1\0"
            if include_public_relations
            else b"angler.paired-public-relation-credit-nontarget-state.v1\0"
        )
        digest.update(schema)
        digest.update(str(state.step).encode("ascii") + b"\0")
        values = [
            ("keys", state.keys),
            ("values", state.values),
            ("usage", state.usage),
            ("acquisition", state.acquisition),
        ]
        if include_public_relations:
            values.append(("public_relations", state.public_relations))
        for name, value in values:
            digest.update(name.encode("ascii") + b"\0")
            digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(_tensor_bytes(value))
        return digest.hexdigest()


PairedPublicRelationCreditMemoryState = PairedPublicRelationCreditState


__all__ = [
    "MEMORY_RANK",
    "MEMORY_SLOTS",
    "RELATION_WIDTH",
    "PairedPublicRelationCreditEvent",
    "PairedPublicRelationCreditMemoryCore",
    "PairedPublicRelationCreditMemoryState",
    "PairedPublicRelationCreditOutput",
    "PairedPublicRelationCreditSnapshot",
    "PairedPublicRelationCreditState",
]
