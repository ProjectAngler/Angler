"""Strict learned key/value credit memory for frozen procedural relations.

The module enforces an information bottleneck rather than relying on a
convention in an experiment runner.  Detached relation and temporal features
can form addresses and gates only.  Objective feedback can form stored value
content only.  The score residual is decoded from the retrieved value only.
No task identity, outcome-conditioned address, or deterministic procedure
rule is represented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
class StrictKeyValueCreditState:
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

    def detached_clone(self) -> "StrictKeyValueCreditState":
        return StrictKeyValueCreditState(
            keys=self.keys.detach().clone(),
            values=self.values.detach().clone(),
            usage=self.usage.detach().clone(),
            acquisition=self.acquisition.detach().clone(),
            step=self.step,
        )


@dataclass(frozen=True, slots=True)
class StrictKeyValueCreditSnapshot:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    acquisition: torch.Tensor
    step: int


@dataclass(frozen=True, slots=True)
class StrictKeyValueCreditOutput:
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
    state_step: int
    read_enabled: bool

    def row(self, index: int) -> "StrictKeyValueCreditOutput":
        if type(index) is not int or not 0 <= index < self.logits.shape[0]:
            raise IndexError("credit-output row is outside the batch")
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
            )
        }
        return StrictKeyValueCreditOutput(
            **fields,
            state_step=self.state_step,
            read_enabled=self.read_enabled,
        )


@dataclass(frozen=True, slots=True)
class StrictKeyValueCreditEvent:
    relation_features: torch.Tensor
    temporal_features: torch.Tensor
    base_logits: torch.Tensor
    outcomes: torch.Tensor
    evidence_refs: tuple[str, ...]
    allocation_index: int
    read_strength: float
    write_strength: float
    erase_gate: float
    write_key: torch.Tensor
    write_value: torch.Tensor

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "angler.strict-key-value-credit-event.v1",
            "relation_features": self.relation_features.tolist(),
            "temporal_features": self.temporal_features.tolist(),
            "base_logits": self.base_logits.tolist(),
            "outcomes": self.outcomes.tolist(),
            "evidence_refs": list(self.evidence_refs),
            "allocation_index": self.allocation_index,
            "read_strength": self.read_strength,
            "write_strength": self.write_strength,
            "erase_gate": self.erase_gate,
            "write_key": self.write_key.tolist(),
            "write_value": self.write_value.tolist(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "StrictKeyValueCreditEvent":
        if record.get("schema") != "angler.strict-key-value-credit-event.v1":
            raise ValueError("strict credit-event schema is unsupported")
        try:
            event = cls(
                relation_features=torch.tensor(record["relation_features"], dtype=torch.float32),
                temporal_features=torch.tensor(record["temporal_features"], dtype=torch.float32),
                base_logits=torch.tensor(record["base_logits"], dtype=torch.float32),
                outcomes=torch.tensor(record["outcomes"], dtype=torch.float32),
                evidence_refs=tuple(record["evidence_refs"]),
                allocation_index=int(record["allocation_index"]),
                read_strength=float(record["read_strength"]),
                write_strength=float(record["write_strength"]),
                erase_gate=float(record["erase_gate"]),
                write_key=torch.tensor(record["write_key"], dtype=torch.float32),
                write_value=torch.tensor(record["write_value"], dtype=torch.float32),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("strict credit-event record is malformed") from error
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
        ):
            raise ValueError("strict credit-event fields are invalid")
        scalars = (self.read_strength, self.write_strength, self.erase_gate)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("strict credit-event gates are non-finite")
        if not (self.read_strength >= 1.0):
            raise ValueError("strict credit-event read strength is invalid")
        if not (0.0 <= self.write_strength <= 1.0 and 0.0 <= self.erase_gate <= 1.0):
            raise ValueError("strict credit-event gates are outside [0,1]")
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
            raise ValueError("strict credit-event tensors must be finite floats")
        if not bool(((self.outcomes == -1) | (self.outcomes == 1)).all().item()):
            raise ValueError("strict credit-event outcomes must be +/-1")


class StrictKeyValueCreditMemoryCore(nn.Module):
    """A strict 512-slot learned association between addresses and feedback."""

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
            raise ValueError("V16 requires frozen 64-D V13 relation features")
        if rank != MEMORY_RANK or memory_slots != MEMORY_SLOTS:
            raise ValueError("V16 requires exactly 512 slots of width 32")
        if (
            isinstance(maximum_residual, bool)
            or not isinstance(maximum_residual, (int, float))
            or not math.isfinite(float(maximum_residual))
            or float(maximum_residual) <= 0.0
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
        address_width = relation_width + rank
        self.query_network = nn.Sequential(
            nn.LayerNorm(address_width),
            nn.Linear(address_width, rank),
            nn.Tanh(),
        )
        self.key_network = nn.Sequential(
            nn.LayerNorm(address_width),
            nn.Linear(address_width, rank),
            nn.Tanh(),
        )
        self.read_strength_network = nn.Linear(address_width, 1)
        self.write_strength_network = nn.Linear(address_width, 1)
        self.erase_network = nn.Sequential(nn.LayerNorm(address_width), nn.Linear(address_width, 1))
        self.slot_temporal_network = nn.Sequential(
            nn.Linear(3, rank),
            nn.SiLU(),
            nn.Linear(rank, 1, bias=False),
        )

        self.outcome_embedding = nn.Embedding(2, rank)
        self.outcome_value_network = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, rank * 2),
            nn.SiLU(),
            nn.Linear(rank * 2, rank),
            nn.Tanh(),
        )
        self.residual_network = nn.Sequential(
            nn.Linear(rank, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, 1, bias=False),
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def parameter_groups(self) -> dict[str, tuple[str, ...]]:
        prefixes = {
            "address": (
                "temporal_network.",
                "query_network.",
                "key_network.",
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
            group: tuple(name for name in names if any(name.startswith(prefix) for prefix in group_prefixes))
            for group, group_prefixes in prefixes.items()
        }
        flattened = tuple(name for group in groups.values() for name in group)
        if len(flattened) != len(names) or set(flattened) != set(names):
            raise RuntimeError("strict architecture parameter ownership is incomplete")
        return groups

    def architecture_report(self) -> dict[str, Any]:
        groups = self.parameter_groups()
        if tuple(self.named_buffers()):
            raise RuntimeError("V16 must not own frozen representation tensors")
        return {
            "parameter_groups": groups,
            "parameter_count": self.parameter_count,
            "memory_slots": self.memory_slots,
            "rank": self.rank,
            "relation_width": self.relation_width,
            "address_inputs": ("detached_relation", "detached_temporal", "state_coordinates"),
            "outcome_inputs": ("two_token_embedding",),
            "stored_value_inputs": ("transformed_outcome_embedding",),
            "residual_decoder_inputs": ("retrieved_value",),
            "outcome_affects_address_or_gates": False,
            "query_affects_value_content_or_decoder": False,
            "metadata_inputs": False,
            "generic_allocation": "stable_least_unused",
            "erase_gate_role": "fresh_slot_noop_under_stable_least_unused",
            "erase_gate_active_before_capacity": False,
            "learned_retention_evidence_claimed": False,
            "v13_parameters_owned": False,
        }

    parameter_report = architecture_report

    def initial_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> StrictKeyValueCreditState:
        reference = next(self.parameters())
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        if resolved_device != reference.device or resolved_dtype != reference.dtype:
            raise ValueError("initial state must match core device/dtype")
        return StrictKeyValueCreditState(
            keys=torch.zeros(self.memory_slots, self.rank, device=resolved_device, dtype=resolved_dtype),
            values=torch.zeros(self.memory_slots, self.rank, device=resolved_device, dtype=resolved_dtype),
            usage=torch.zeros(self.memory_slots, device=resolved_device, dtype=resolved_dtype),
            acquisition=torch.zeros(self.memory_slots, device=resolved_device, dtype=resolved_dtype),
            step=0,
        )

    initial_ability_state = initial_state

    def predict(
        self,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: StrictKeyValueCreditState | None = None,
        read_enabled: bool = True,
    ) -> StrictKeyValueCreditOutput:
        """Predict from prior state without accepting the current outcome."""

        _require_bool("read_enabled", read_enabled)
        self._validate_observations(relation_features, temporal_features, base_logits)
        current = self.initial_state() if state is None else state
        self.validate_state(current)

        relations = relation_features.detach()
        temporal = temporal_features.detach()
        base = base_logits.detach()
        temporal_hidden = self.temporal_network(temporal)
        address = torch.cat((relations, temporal_hidden), dim=-1)
        queries = self.query_network(address)
        write_keys = self.key_network(address)
        read_strengths = F.softplus(self.read_strength_network(address).squeeze(-1)) + 1.0
        write_strengths = torch.sigmoid(self.write_strength_network(address).squeeze(-1))
        erase_gates = torch.sigmoid(self.erase_network(address).squeeze(-1))

        read_weights = self._content_read_weights(queries, read_strengths, current)
        if not read_enabled:
            read_weights = torch.zeros_like(read_weights)
        read_values = read_weights @ current.values
        residuals = self.decode_retrieved_values(read_values)
        if not read_enabled:
            residuals = torch.zeros_like(residuals)
        logits = base + residuals
        output = StrictKeyValueCreditOutput(
            logits=logits,
            base_logits=base,
            residuals=residuals,
            read_queries=queries,
            read_weights=read_weights,
            read_values=read_values,
            temporal_hidden=temporal_hidden,
            write_keys=write_keys,
            read_strengths=read_strengths,
            write_strengths=write_strengths,
            erase_gates=erase_gates,
            observed_relations=relations,
            observed_temporal=temporal,
            state_step=current.step,
            read_enabled=read_enabled,
        )
        self._validate_output(output)
        return output

    forward = predict

    def decode_retrieved_values(self, read_values: torch.Tensor) -> torch.Tensor:
        reference = next(self.parameters())
        if (
            read_values.ndim != 2
            or read_values.shape[-1] != self.rank
            or read_values.device != reference.device
            or read_values.dtype != reference.dtype
            or not bool(torch.isfinite(read_values).all().item())
        ):
            raise ValueError("retrieved values must be finite [batch,32] core tensors")
        raw = self.residual_network(read_values).squeeze(-1)
        return self.maximum_residual * torch.tanh(raw)

    def outcome_values(self, outcomes: torch.Tensor) -> torch.Tensor:
        reference = next(self.parameters())
        if (
            outcomes.ndim != 1
            or outcomes.numel() == 0
            or outcomes.device != reference.device
            or outcomes.dtype != reference.dtype
            or not bool(torch.isfinite(outcomes).all().item())
            or not bool(((outcomes == -1) | (outcomes == 1)).all().item())
        ):
            raise ValueError("outcomes must be finite core +/-1 vectors")
        indices = (outcomes.detach() > 0).to(dtype=torch.long)
        return self.outcome_value_network(self.outcome_embedding(indices))

    def outcome_loss(
        self,
        output: StrictKeyValueCreditOutput,
        outcomes: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_outcomes(outcomes, output.logits)
        loss = F.softplus(-outcomes * output.logits).mean()
        if loss.shape != () or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("strict credit loss is invalid")
        return loss

    def apply_feedback(
        self,
        state: StrictKeyValueCreditState,
        output: StrictKeyValueCreditOutput,
        outcomes: torch.Tensor,
        *,
        evidence_refs: str | Sequence[str],
        detach_state: bool = True,
    ) -> tuple[StrictKeyValueCreditState, StrictKeyValueCreditEvent]:
        """Store one outcome-derived value after its prediction."""

        _require_bool("detach_state", detach_state)
        self.validate_state(state)
        self._validate_output(output)
        self._validate_outcomes(outcomes, output.logits)
        if output.logits.shape[0] != 1:
            raise ValueError("one persistent state accepts one event at a time")
        if output.state_step != state.step:
            raise ValueError("strict credit output is stale for the supplied state")
        if state.step >= self.memory_slots:
            raise OverflowError("strict credit memory capacity is exhausted")
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
        updated = StrictKeyValueCreditState(keys, values, usage, acquisition, state.step + 1)
        self.validate_state(updated)
        result = updated.detached_clone() if detach_state else updated
        winner = int(allocation.argmax().item())
        event = StrictKeyValueCreditEvent(
            relation_features=output.observed_relations.detach().cpu().clone(),
            temporal_features=output.observed_temporal.detach().cpu().clone(),
            base_logits=output.base_logits.detach().cpu().clone(),
            outcomes=outcomes.detach().cpu().clone(),
            evidence_refs=refs,
            allocation_index=winner,
            read_strength=float(output.read_strengths[0].detach().item()),
            write_strength=float(output.write_strengths[0].detach().item()),
            erase_gate=float(output.erase_gates[0].detach().item()),
            write_key=write_key.detach().cpu().clone(),
            write_value=value.detach().cpu().clone(),
        )
        event._validate_record_fields()
        return result, event

    def capture_state(self, state: StrictKeyValueCreditState) -> StrictKeyValueCreditSnapshot:
        self.validate_state(state)
        return StrictKeyValueCreditSnapshot(
            state.keys.detach().cpu().clone(),
            state.values.detach().cpu().clone(),
            state.usage.detach().cpu().clone(),
            state.acquisition.detach().cpu().clone(),
            state.step,
        )

    def restore_state(self, snapshot: StrictKeyValueCreditSnapshot) -> StrictKeyValueCreditState:
        if not isinstance(snapshot, StrictKeyValueCreditSnapshot):
            raise TypeError("snapshot must be StrictKeyValueCreditSnapshot")
        reference = next(self.parameters())
        state = StrictKeyValueCreditState(
            snapshot.keys.detach().to(reference).clone(),
            snapshot.values.detach().to(reference).clone(),
            snapshot.usage.detach().to(reference).clone(),
            snapshot.acquisition.detach().to(reference).clone(),
            snapshot.step,
        )
        self.validate_state(state)
        return state

    def state_digest(self, state: StrictKeyValueCreditState) -> str:
        self.validate_state(state)
        digest = hashlib.sha256()
        digest.update(b"angler.strict-key-value-credit-state.v1\0")
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

    def zero_state_like(self, state: StrictKeyValueCreditState) -> StrictKeyValueCreditState:
        self.validate_state(state)
        return StrictKeyValueCreditState(
            torch.zeros_like(state.keys),
            torch.zeros_like(state.values),
            torch.zeros_like(state.usage),
            torch.zeros_like(state.acquisition),
            0,
        )

    def replay(
        self,
        events: Sequence[StrictKeyValueCreditEvent],
        *,
        state: StrictKeyValueCreditState | None = None,
        detach_state: bool = True,
    ) -> tuple[StrictKeyValueCreditState, tuple[StrictKeyValueCreditOutput, ...]]:
        _require_bool("detach_state", detach_state)
        current = self.initial_state() if state is None else state
        self.validate_state(current)
        reference = next(self.parameters())
        outputs = []
        for event in events:
            if not isinstance(event, StrictKeyValueCreditEvent):
                raise TypeError("replay accepts StrictKeyValueCreditEvent objects")
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
                reconstructed.allocation_index != event.allocation_index
                or abs(reconstructed.read_strength - event.read_strength) > 1.0e-6
                or abs(reconstructed.write_strength - event.write_strength) > 1.0e-6
                or abs(reconstructed.erase_gate - event.erase_gate) > 1.0e-6
                or not torch.allclose(reconstructed.write_key, event.write_key, rtol=0.0, atol=1.0e-6)
                or not torch.allclose(reconstructed.write_value, event.write_value, rtol=0.0, atol=1.0e-6)
            ):
                raise RuntimeError("strict credit-event replay diverged from evidence")
            outputs.append(output)
        return current, tuple(outputs)

    def validate_state(self, state: StrictKeyValueCreditState) -> None:
        if not isinstance(state, StrictKeyValueCreditState):
            raise TypeError("state must be StrictKeyValueCreditState")
        reference = next(self.parameters())
        if state.keys.shape != (self.memory_slots, self.rank) or state.values.shape != (
            self.memory_slots,
            self.rank,
        ):
            raise ValueError("strict credit-memory keys/values have the wrong shape")
        if state.usage.shape != (self.memory_slots,) or state.acquisition.shape != (
            self.memory_slots,
        ):
            raise ValueError("strict credit-memory coordinates have the wrong shape")
        if type(state.step) is not int or not 0 <= state.step <= self.memory_slots:
            raise ValueError("strict credit-memory step is outside capacity")
        for name, value in (
            ("keys", state.keys),
            ("values", state.values),
            ("usage", state.usage),
            ("acquisition", state.acquisition),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("strict credit-memory state must match core device/dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("strict credit-memory state must be finite")
            if name in {"keys", "values"} and not bool(
                (value.abs() <= 1.0 + 1.0e-6).all().item()
            ):
                raise ValueError(f"strict credit-memory {name} exceeded [-1,1]")
            if name in {"usage", "acquisition"} and not bool(
                ((value >= 0.0) & (value <= 1.0 + 1.0e-6)).all().item()
            ):
                raise ValueError(f"strict credit-memory {name} exceeded [0,1]")

    def _content_read_weights(
        self,
        queries: torch.Tensor,
        strengths: torch.Tensor,
        state: StrictKeyValueCreditState,
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
            raise ValueError("strict allocation usage has the wrong shape")
        _, order = torch.sort(usage, stable=True)
        return F.one_hot(order[0], num_classes=self.memory_slots).to(dtype=usage.dtype)

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
        if base.shape != (relations.shape[0],) or relations.shape[0] == 0:
            raise ValueError("base_logits must be a nonempty aligned vector")
        for name, value in (
            ("relation_features", relations),
            ("temporal_features", temporal),
            ("base_logits", base),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match core device/dtype")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite floating values")

    def _validate_outcomes(self, outcomes: torch.Tensor, logits: torch.Tensor) -> None:
        if (
            outcomes.shape != logits.shape
            or outcomes.device != logits.device
            or outcomes.dtype != logits.dtype
            or not outcomes.is_floating_point()
            or not bool(torch.isfinite(outcomes).all().item())
            or not bool(((outcomes == -1) | (outcomes == 1)).all().item())
        ):
            raise ValueError("outcomes must be aligned finite +/-1 tensors")

    def _validate_output(self, output: StrictKeyValueCreditOutput) -> None:
        if not isinstance(output, StrictKeyValueCreditOutput):
            raise TypeError("output must be StrictKeyValueCreditOutput")
        if output.logits.ndim != 1:
            raise ValueError("strict credit output logits must be a vector")
        batch = output.logits.shape[0]
        vector = (batch, self.rank)
        slots = (batch, self.memory_slots)
        for name in (
            "logits",
            "base_logits",
            "residuals",
            "read_strengths",
            "write_strengths",
            "erase_gates",
        ):
            if getattr(output, name).shape != (batch,):
                raise ValueError(f"strict credit output {name} has the wrong shape")
        for name in ("read_queries", "read_values", "temporal_hidden", "write_keys"):
            if getattr(output, name).shape != vector:
                raise ValueError(f"strict credit output {name} has the wrong shape")
        if output.read_weights.shape != slots:
            raise ValueError("strict credit output read weights have the wrong shape")
        if output.observed_relations.shape != (batch, self.relation_width):
            raise ValueError("strict credit output relation observation has the wrong shape")
        if output.observed_temporal.shape != (batch, self.temporal_width):
            raise ValueError("strict credit output temporal observation has the wrong shape")
        if type(output.state_step) is not int or not 0 <= output.state_step <= self.memory_slots:
            raise ValueError("strict credit output state step is invalid")
        _require_bool("output.read_enabled", output.read_enabled)
        reference = next(self.parameters())
        tensors = tuple(
            getattr(output, name)
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
            )
        )
        if any(value.device != reference.device or value.dtype != reference.dtype for value in tensors):
            raise ValueError("strict credit output must match core device/dtype")
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ValueError("strict credit output must be finite")
        if not bool((output.read_strengths >= 1.0).all().item()):
            raise ValueError("strict credit read strengths are invalid")
        for name in ("write_strengths", "erase_gates"):
            value = getattr(output, name)
            if not bool(((value >= 0.0) & (value <= 1.0)).all().item()):
                raise ValueError(f"strict credit {name} exceeded [0,1]")

    @staticmethod
    def _normalize_evidence_refs(refs: str | Sequence[str], *, batch: int) -> tuple[str, ...]:
        values = (refs,) if isinstance(refs, str) else tuple(refs)
        if len(values) != batch or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("one nonempty evidence ref is required per event")
        return values


__all__ = [
    "MEMORY_RANK",
    "MEMORY_SLOTS",
    "RELATION_WIDTH",
    "StrictKeyValueCreditEvent",
    "StrictKeyValueCreditMemoryCore",
    "StrictKeyValueCreditOutput",
    "StrictKeyValueCreditSnapshot",
    "StrictKeyValueCreditState",
]
