"""Bounded persistent credit memory for frozen procedural relations.

The core consumes detached 64-D relation features produced by the frozen V13
representation.  It predicts from prior state before an outcome is available,
then records a learned key/value update only after objective feedback arrives.
Slot allocation is generic least-unused bookkeeping; all content, outcome
interpretation, reading, writing, temporal modulation, and score residuals are
learned and shared across tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


RELATION_WIDTH = 64
MEMORY_RANK = 32
MEMORY_SLOTS = 512


def _tensor_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _require_bool(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


@dataclass(frozen=True, slots=True)
class StructureKeyedCreditState:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    acquisition: torch.Tensor
    step: int

    @property
    def bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.keys, self.values, self.usage, self.acquisition)
        )

    def detached_clone(self) -> "StructureKeyedCreditState":
        return StructureKeyedCreditState(
            keys=self.keys.detach().clone(),
            values=self.values.detach().clone(),
            usage=self.usage.detach().clone(),
            acquisition=self.acquisition.detach().clone(),
            step=self.step,
        )


@dataclass(frozen=True, slots=True)
class StructureKeyedCreditSnapshot:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    acquisition: torch.Tensor
    step: int


@dataclass(frozen=True, slots=True)
class StructureKeyedCreditOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    residuals: torch.Tensor
    read_queries: torch.Tensor
    read_weights: torch.Tensor
    read_values: torch.Tensor
    temporal_hidden: torch.Tensor
    write_keys: torch.Tensor
    write_strengths: torch.Tensor
    observed_relations: torch.Tensor
    observed_temporal: torch.Tensor
    state_step: int
    read_enabled: bool

    def row(self, index: int) -> "StructureKeyedCreditOutput":
        if type(index) is not int or not 0 <= index < self.logits.shape[0]:
            raise IndexError("credit-output row is outside the batch")
        return StructureKeyedCreditOutput(
            logits=self.logits[index : index + 1],
            base_logits=self.base_logits[index : index + 1],
            residuals=self.residuals[index : index + 1],
            read_queries=self.read_queries[index : index + 1],
            read_weights=self.read_weights[index : index + 1],
            read_values=self.read_values[index : index + 1],
            temporal_hidden=self.temporal_hidden[index : index + 1],
            write_keys=self.write_keys[index : index + 1],
            write_strengths=self.write_strengths[index : index + 1],
            observed_relations=self.observed_relations[index : index + 1],
            observed_temporal=self.observed_temporal[index : index + 1],
            state_step=self.state_step,
            read_enabled=self.read_enabled,
        )


@dataclass(frozen=True, slots=True)
class StructureKeyedCreditEvent:
    relation_features: torch.Tensor
    temporal_features: torch.Tensor
    base_logits: torch.Tensor
    outcomes: torch.Tensor
    evidence_refs: tuple[str, ...]
    allocation_index: int
    write_strength: float
    erase_gate: float
    write_key: torch.Tensor
    write_value: torch.Tensor

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "angler.structure-keyed-credit-event.v1",
            "relation_features": self.relation_features.tolist(),
            "temporal_features": self.temporal_features.tolist(),
            "base_logits": self.base_logits.tolist(),
            "outcomes": self.outcomes.tolist(),
            "evidence_refs": list(self.evidence_refs),
            "allocation_index": self.allocation_index,
            "write_strength": self.write_strength,
            "erase_gate": self.erase_gate,
            "write_key": self.write_key.tolist(),
            "write_value": self.write_value.tolist(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "StructureKeyedCreditEvent":
        if record.get("schema") != "angler.structure-keyed-credit-event.v1":
            raise ValueError("credit-event schema is unsupported")
        try:
            event = cls(
                relation_features=torch.tensor(
                    record["relation_features"], dtype=torch.float32
                ),
                temporal_features=torch.tensor(
                    record["temporal_features"], dtype=torch.float32
                ),
                base_logits=torch.tensor(record["base_logits"], dtype=torch.float32),
                outcomes=torch.tensor(record["outcomes"], dtype=torch.float32),
                evidence_refs=tuple(record["evidence_refs"]),
                allocation_index=int(record["allocation_index"]),
                write_strength=float(record["write_strength"]),
                erase_gate=float(record["erase_gate"]),
                write_key=torch.tensor(record["write_key"], dtype=torch.float32),
                write_value=torch.tensor(record["write_value"], dtype=torch.float32),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("credit-event record is malformed") from error
        event._validate_record_fields()
        return event

    def _validate_record_fields(self) -> None:
        if (
            self.relation_features.ndim != 2
            or self.relation_features.shape[0] != 1
            or self.temporal_features.ndim != 2
            or self.temporal_features.shape[0] != 1
            or self.base_logits.shape != (1,)
            or self.outcomes.shape != (1,)
            or len(self.evidence_refs) != 1
            or not self.evidence_refs[0]
            or type(self.allocation_index) is not int
            or not 0 <= self.allocation_index < MEMORY_SLOTS
            or self.write_key.shape != (1, MEMORY_RANK)
            or self.write_value.shape != (1, MEMORY_RANK)
            or not math.isfinite(self.write_strength)
            or not math.isfinite(self.erase_gate)
        ):
            raise ValueError("credit-event fields are invalid")
        tensors = (
            self.relation_features,
            self.temporal_features,
            self.base_logits,
            self.outcomes,
            self.write_key,
            self.write_value,
        )
        if not all(value.is_floating_point() for value in tensors) or not all(
            bool(torch.isfinite(value).all().item()) for value in tensors
        ):
            raise ValueError("credit-event tensors must be finite floating values")
        if not bool(((self.outcomes == -1) | (self.outcomes == 1)).all().item()):
            raise ValueError("credit-event outcomes must be +/-1")
        if not (0.0 <= self.write_strength <= 1.0 and 0.0 <= self.erase_gate <= 1.0):
            raise ValueError("credit-event gates are outside [0,1]")


class StructureKeyedCreditMemoryCore(nn.Module):
    """A learned 512-slot ability state keyed by frozen V13 relations."""

    def __init__(
        self,
        *,
        temporal_width: int,
        relation_width: int = RELATION_WIDTH,
        rank: int = MEMORY_RANK,
        memory_slots: int = MEMORY_SLOTS,
        maximum_residual: float = 4.0,
    ) -> None:
        super().__init__()
        if type(temporal_width) is not int or temporal_width <= 0:
            raise ValueError("temporal_width must be a positive integer")
        if relation_width != RELATION_WIDTH:
            raise ValueError("V14 requires frozen 64-D V13 relation features")
        if rank != MEMORY_RANK or memory_slots != MEMORY_SLOTS:
            raise ValueError("V14 requires exactly 512 slots of width 32")
        if (
            isinstance(maximum_residual, bool)
            or not isinstance(maximum_residual, (int, float))
            or not math.isfinite(float(maximum_residual))
            or maximum_residual <= 0
        ):
            raise ValueError("maximum_residual must be finite and positive")

        self.temporal_width = temporal_width
        self.relation_width = relation_width
        self.rank = rank
        self.memory_slots = memory_slots
        self.maximum_residual = float(maximum_residual)

        self.temporal_network = nn.Sequential(
            nn.LayerNorm(temporal_width),
            nn.Linear(temporal_width, rank),
            nn.SiLU(),
        )
        relation_temporal_width = relation_width + rank
        self.query_network = nn.Sequential(
            nn.LayerNorm(relation_temporal_width),
            nn.Linear(relation_temporal_width, rank),
            nn.Tanh(),
        )
        self.key_network = nn.Sequential(
            nn.LayerNorm(relation_temporal_width),
            nn.Linear(relation_temporal_width, rank),
            nn.Tanh(),
        )
        self.read_strength_network = nn.Linear(relation_temporal_width, 1)
        self.write_strength_network = nn.Linear(relation_temporal_width, 1)
        self.slot_temporal_network = nn.Sequential(
            nn.Linear(3, rank),
            nn.SiLU(),
            nn.Linear(rank, 1, bias=False),
        )
        self.outcome_embedding = nn.Embedding(2, rank)
        writer_width = rank * 5
        self.value_network = nn.Sequential(
            nn.LayerNorm(writer_width),
            nn.Linear(writer_width, rank * 2),
            nn.SiLU(),
            nn.Linear(rank * 2, rank),
            nn.Tanh(),
        )
        self.erase_network = nn.Sequential(
            nn.LayerNorm(writer_width),
            nn.Linear(writer_width, 1),
        )
        self.residual_gate_network = nn.Sequential(
            nn.LayerNorm(rank * 3),
            nn.Linear(rank * 3, rank),
            nn.Sigmoid(),
        )
        # Bias-free read-value path makes an empty/zero state an exact zero
        # intervention rather than a learned relation-only fallback.
        self.residual_network = nn.Sequential(
            nn.Linear(rank * 2, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, 1, bias=False),
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def initial_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> StructureKeyedCreditState:
        reference = next(self.parameters())
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        return StructureKeyedCreditState(
            keys=torch.zeros(
                self.memory_slots,
                self.rank,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            values=torch.zeros(
                self.memory_slots,
                self.rank,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            usage=torch.zeros(
                self.memory_slots,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            acquisition=torch.zeros(
                self.memory_slots,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            step=0,
        )

    initial_ability_state = initial_state

    def predict(
        self,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: StructureKeyedCreditState | None = None,
        read_enabled: bool = True,
    ) -> StructureKeyedCreditOutput:
        """Predict from prior state without accepting the current outcome."""

        _require_bool("read_enabled", read_enabled)
        self._validate_observations(relation_features, temporal_features, base_logits)
        current = self.initial_state() if state is None else state
        self.validate_state(current)

        relations = relation_features.detach()
        temporal_observed = temporal_features.detach()
        base = base_logits.detach()
        temporal_hidden = self.temporal_network(temporal_observed)
        relation_temporal = torch.cat((relations, temporal_hidden), dim=-1)
        queries = self.query_network(relation_temporal)
        write_keys = self.key_network(relation_temporal)
        read_strengths = F.softplus(
            self.read_strength_network(relation_temporal).squeeze(-1)
        ) + 1.0
        write_strengths = torch.sigmoid(
            self.write_strength_network(relation_temporal).squeeze(-1)
        )

        read_weights = self._content_read_weights(
            queries,
            read_strengths,
            current,
        )
        if not read_enabled:
            read_weights = torch.zeros_like(read_weights)
        read_values = read_weights @ current.values
        residual_gate = self.residual_gate_network(
            torch.cat((queries, temporal_hidden, read_values), dim=-1)
        )
        gated_read = read_values * residual_gate
        raw_residual = self.residual_network(
            torch.cat((gated_read, queries * gated_read), dim=-1)
        ).squeeze(-1)
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        if not read_enabled:
            residuals = torch.zeros_like(residuals)
        logits = base + residuals
        output = StructureKeyedCreditOutput(
            logits=logits,
            base_logits=base,
            residuals=residuals,
            read_queries=queries,
            read_weights=read_weights,
            read_values=read_values,
            temporal_hidden=temporal_hidden,
            write_keys=write_keys,
            write_strengths=write_strengths,
            observed_relations=relations,
            observed_temporal=temporal_observed,
            state_step=current.step,
            read_enabled=read_enabled,
        )
        self._validate_output(output)
        return output

    forward = predict

    def outcome_loss(
        self,
        output: StructureKeyedCreditOutput,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_outcomes(outcomes, output.logits)
        loss = F.softplus(-outcomes * output.logits).mean()
        if loss.shape != () or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("persistent-credit loss is invalid")
        return loss

    def apply_feedback(
        self,
        state: StructureKeyedCreditState,
        output: StructureKeyedCreditOutput,
        outcomes: torch.Tensor,
        *,
        evidence_refs: str | Sequence[str],
        detach_state: bool = True,
    ) -> tuple[StructureKeyedCreditState, StructureKeyedCreditEvent]:
        """Apply one chronological learned write after objective feedback."""

        _require_bool("detach_state", detach_state)
        self.validate_state(state)
        self._validate_output(output)
        self._validate_outcomes(outcomes, output.logits)
        if output.logits.shape[0] != 1:
            raise ValueError("one persistent state accepts one event at a time")
        if output.state_step != state.step:
            raise ValueError("credit output is stale for the supplied state")
        if state.step >= self.memory_slots:
            raise OverflowError("persistent-credit memory capacity is exhausted")
        refs = self._normalize_evidence_refs(evidence_refs, batch=1)

        token_indices = (outcomes.detach() > 0).to(dtype=torch.long)
        outcome_hidden = self.outcome_embedding(token_indices)
        writer_input = torch.cat(
            (
                output.write_keys,
                outcome_hidden,
                output.write_keys * outcome_hidden,
                output.read_values,
                output.temporal_hidden,
            ),
            dim=-1,
        )
        write_value = self.value_network(writer_input)
        erase_gate = torch.sigmoid(self.erase_network(writer_input).squeeze(-1))
        allocation = self._least_unused_allocation(state.usage)
        slot_weights = output.write_strengths[0] * allocation
        slot = slot_weights.unsqueeze(-1)

        write_key = output.write_keys[0].unsqueeze(0)
        value = write_value[0].unsqueeze(0)
        keys = (1.0 - slot) * state.keys + slot * write_key
        erased = state.values * (1.0 - slot * erase_gate[0])
        values = (1.0 - slot) * erased + slot * value
        usage = state.usage + slot_weights * (1.0 - state.usage)
        coordinate = state.usage.new_tensor((state.step + 1) / self.memory_slots)
        acquisition = (1.0 - slot_weights) * state.acquisition + slot_weights * coordinate
        updated = StructureKeyedCreditState(
            keys=keys,
            values=values,
            usage=usage,
            acquisition=acquisition,
            step=state.step + 1,
        )
        self.validate_state(updated)
        result = updated.detached_clone() if detach_state else updated
        winner = int(allocation.argmax().item())
        event = StructureKeyedCreditEvent(
            relation_features=output.observed_relations.detach().cpu().clone(),
            temporal_features=output.observed_temporal.detach().cpu().clone(),
            base_logits=output.base_logits.detach().cpu().clone(),
            outcomes=outcomes.detach().cpu().clone(),
            evidence_refs=refs,
            allocation_index=winner,
            write_strength=float(output.write_strengths[0].detach().item()),
            erase_gate=float(erase_gate[0].detach().item()),
            write_key=write_key.detach().cpu().clone(),
            write_value=value.detach().cpu().clone(),
        )
        event._validate_record_fields()
        return result, event

    def capture_state(
        self,
        state: StructureKeyedCreditState,
    ) -> StructureKeyedCreditSnapshot:
        self.validate_state(state)
        return StructureKeyedCreditSnapshot(
            keys=state.keys.detach().cpu().clone(),
            values=state.values.detach().cpu().clone(),
            usage=state.usage.detach().cpu().clone(),
            acquisition=state.acquisition.detach().cpu().clone(),
            step=state.step,
        )

    def restore_state(
        self,
        snapshot: StructureKeyedCreditSnapshot,
    ) -> StructureKeyedCreditState:
        if not isinstance(snapshot, StructureKeyedCreditSnapshot):
            raise TypeError("snapshot must be StructureKeyedCreditSnapshot")
        reference = next(self.parameters())
        state = StructureKeyedCreditState(
            keys=snapshot.keys.detach().to(reference).clone(),
            values=snapshot.values.detach().to(reference).clone(),
            usage=snapshot.usage.detach().to(reference).clone(),
            acquisition=snapshot.acquisition.detach().to(reference).clone(),
            step=snapshot.step,
        )
        self.validate_state(state)
        return state

    def state_digest(self, state: StructureKeyedCreditState) -> str:
        self.validate_state(state)
        digest = hashlib.sha256()
        digest.update(b"angler.structure-keyed-credit-state.v1\0")
        digest.update(str(state.step).encode("ascii") + b"\0")
        for name, value in (
            ("keys", state.keys),
            ("values", state.values),
            ("usage", state.usage),
            ("acquisition", state.acquisition),
        ):
            digest.update(name.encode("ascii") + b"\0")
            digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(_tensor_bytes(value))
        return digest.hexdigest()

    def zero_state_like(
        self,
        state: StructureKeyedCreditState,
    ) -> StructureKeyedCreditState:
        self.validate_state(state)
        return StructureKeyedCreditState(
            keys=torch.zeros_like(state.keys),
            values=torch.zeros_like(state.values),
            usage=torch.zeros_like(state.usage),
            acquisition=torch.zeros_like(state.acquisition),
            step=0,
        )

    def replay(
        self,
        events: Sequence[StructureKeyedCreditEvent],
        *,
        state: StructureKeyedCreditState | None = None,
        detach_state: bool = True,
    ) -> tuple[StructureKeyedCreditState, tuple[StructureKeyedCreditOutput, ...]]:
        _require_bool("detach_state", detach_state)
        current = self.initial_state() if state is None else state
        self.validate_state(current)
        reference = next(self.parameters())
        outputs = []
        for event in events:
            if not isinstance(event, StructureKeyedCreditEvent):
                raise TypeError("replay accepts StructureKeyedCreditEvent objects")
            event._validate_record_fields()
            relation = event.relation_features.to(reference)
            temporal = event.temporal_features.to(reference)
            base = event.base_logits.to(reference)
            outcomes = event.outcomes.to(reference)
            output = self.predict(relation, temporal, base, state=current)
            current, reconstructed = self.apply_feedback(
                current,
                output,
                outcomes,
                evidence_refs=event.evidence_refs,
                detach_state=detach_state,
            )
            if (
                reconstructed.allocation_index != event.allocation_index
                or abs(reconstructed.write_strength - event.write_strength) > 1.0e-6
                or abs(reconstructed.erase_gate - event.erase_gate) > 1.0e-6
                or not torch.allclose(
                    reconstructed.write_key,
                    event.write_key,
                    rtol=0.0,
                    atol=1.0e-6,
                )
                or not torch.allclose(
                    reconstructed.write_value,
                    event.write_value,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ):
                raise RuntimeError("credit-event replay diverged from its evidence")
            outputs.append(output)
        return current, tuple(outputs)

    def validate_state(self, state: StructureKeyedCreditState) -> None:
        if not isinstance(state, StructureKeyedCreditState):
            raise TypeError("state must be StructureKeyedCreditState")
        reference = next(self.parameters())
        if state.keys.shape != (self.memory_slots, self.rank) or state.values.shape != (
            self.memory_slots,
            self.rank,
        ):
            raise ValueError("credit-memory keys/values have the wrong shape")
        if state.usage.shape != (self.memory_slots,) or state.acquisition.shape != (
            self.memory_slots,
        ):
            raise ValueError("credit-memory coordinates have the wrong shape")
        if type(state.step) is not int or not 0 <= state.step <= self.memory_slots:
            raise ValueError("credit-memory step is outside its capacity")
        for name, value in (
            ("keys", state.keys),
            ("values", state.values),
            ("usage", state.usage),
            ("acquisition", state.acquisition),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("credit-memory state must match core device/dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("credit-memory state must be finite")
            if name in {"keys", "values"} and not bool(
                (value.abs() <= 1.0 + 1.0e-6).all().item()
            ):
                raise ValueError(f"credit-memory {name} exceeded [-1,1]")
            if name in {"usage", "acquisition"} and not bool(
                ((value >= 0.0) & (value <= 1.0 + 1.0e-6)).all().item()
            ):
                raise ValueError(f"credit-memory {name} exceeded [0,1]")

    def parameter_report(self) -> dict[str, Any]:
        names = tuple(name for name, _ in self.named_parameters())
        required = (
            "temporal_network.",
            "query_network.",
            "key_network.",
            "value_network.",
            "residual_network.",
        )
        if not all(any(name.startswith(prefix) for name in names) for prefix in required):
            raise RuntimeError("V14 learned-network ownership is incomplete")
        if tuple(self.named_buffers()):
            raise RuntimeError("V14 must not own frozen V13 tensors")
        return {
            "parameter_names": names,
            "parameter_count": self.parameter_count,
            "memory_slots": self.memory_slots,
            "rank": self.rank,
            "relation_width": self.relation_width,
            "v13_parameters_owned": False,
            "metadata_inputs": False,
            "outcome_prediction_inputs": False,
            "generic_allocation": "stable_least_unused",
        }

    def _content_read_weights(
        self,
        queries: torch.Tensor,
        strengths: torch.Tensor,
        state: StructureKeyedCreditState,
    ) -> torch.Tensor:
        query = F.normalize(queries, dim=-1, eps=1.0e-6)
        keys = F.normalize(state.keys, dim=-1, eps=1.0e-6)
        content = torch.einsum("br,sr->bs", query, keys) * strengths.unsqueeze(-1)
        coordinate = state.usage.new_tensor(state.step / self.memory_slots)
        age = (coordinate - state.acquisition).clamp(0.0, 1.0)
        slot_context = torch.stack((state.usage, state.acquisition, age), dim=-1)
        temporal_bias = self.slot_temporal_network(slot_context).squeeze(-1)
        logits = content + temporal_bias + torch.log(state.usage.clamp_min(1.0e-6))
        weights = torch.softmax(logits, dim=-1)
        if state.step == 0:
            weights = torch.zeros_like(weights)
        return weights

    def _least_unused_allocation(self, usage: torch.Tensor) -> torch.Tensor:
        if usage.shape != (self.memory_slots,):
            raise ValueError("allocation usage has the wrong shape")
        _, order = torch.sort(usage, stable=True)
        winner = order[0]
        return F.one_hot(winner, num_classes=self.memory_slots).to(dtype=usage.dtype)

    def _validate_observations(
        self,
        relations: torch.Tensor,
        temporal: torch.Tensor,
        base: torch.Tensor,
    ) -> None:
        reference = next(self.parameters())
        if relations.ndim != 2 or relations.shape[-1] != self.relation_width:
            raise ValueError("relation_features must be [batch,64]")
        if temporal.shape != (relations.shape[0], self.temporal_width):
            raise ValueError("temporal_features must be [batch,temporal_width]")
        if base.shape != (relations.shape[0],):
            raise ValueError("base_logits must be [batch]")
        if relations.shape[0] == 0:
            raise ValueError("credit prediction requires a nonempty batch")
        for name, value in (
            ("relation_features", relations),
            ("temporal_features", temporal),
            ("base_logits", base),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match core device/dtype")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite floating values")

    def _validate_outcomes(
        self,
        outcomes: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> None:
        if (
            outcomes.shape != reference_logits.shape
            or outcomes.device != reference_logits.device
            or outcomes.dtype != reference_logits.dtype
            or not outcomes.is_floating_point()
            or not bool(torch.isfinite(outcomes).all().item())
            or not bool(((outcomes == -1) | (outcomes == 1)).all().item())
        ):
            raise ValueError("outcomes must be aligned finite +/-1 tensors")

    def _validate_output(self, output: StructureKeyedCreditOutput) -> None:
        if not isinstance(output, StructureKeyedCreditOutput):
            raise TypeError("output must be StructureKeyedCreditOutput")
        batch = output.logits.shape[0]
        expected_vector = (batch, self.rank)
        expected_slots = (batch, self.memory_slots)
        expected_relation = (batch, self.relation_width)
        expected_temporal = (batch, self.temporal_width)
        for name in ("logits", "base_logits", "residuals", "write_strengths"):
            if getattr(output, name).shape != (batch,):
                raise ValueError(f"credit output {name} has the wrong shape")
        for name in (
            "read_queries",
            "read_values",
            "temporal_hidden",
            "write_keys",
        ):
            if getattr(output, name).shape != expected_vector:
                raise ValueError(f"credit output {name} has the wrong shape")
        if output.read_weights.shape != expected_slots:
            raise ValueError("credit output read_weights has the wrong shape")
        if output.observed_relations.shape != expected_relation:
            raise ValueError("credit output observed relations have the wrong shape")
        if output.observed_temporal.shape != expected_temporal:
            raise ValueError("credit output observed temporal has the wrong shape")
        if type(output.state_step) is not int or not 0 <= output.state_step <= self.memory_slots:
            raise ValueError("credit output state_step is invalid")
        _require_bool("output.read_enabled", output.read_enabled)
        reference = next(self.parameters())
        for value in (
            output.logits,
            output.base_logits,
            output.residuals,
            output.read_queries,
            output.read_weights,
            output.read_values,
            output.temporal_hidden,
            output.write_keys,
            output.write_strengths,
            output.observed_relations,
            output.observed_temporal,
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("credit output must match core device/dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("credit output must be finite")
        if not bool(
            ((output.write_strengths >= 0.0) & (output.write_strengths <= 1.0)).all().item()
        ):
            raise ValueError("credit output write strengths exceeded [0,1]")

    @staticmethod
    def _normalize_evidence_refs(
        refs: str | Sequence[str],
        *,
        batch: int,
    ) -> tuple[str, ...]:
        values = (refs,) if isinstance(refs, str) else tuple(refs)
        if len(values) != batch or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("one nonempty evidence ref is required per event")
        return values


__all__ = [
    "MEMORY_RANK",
    "MEMORY_SLOTS",
    "RELATION_WIDTH",
    "StructureKeyedCreditEvent",
    "StructureKeyedCreditMemoryCore",
    "StructureKeyedCreditOutput",
    "StructureKeyedCreditSnapshot",
    "StructureKeyedCreditState",
]
