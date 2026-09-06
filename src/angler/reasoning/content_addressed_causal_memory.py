"""Bounded content-addressed causal memory for procedural apprenticeship.

This is a compact NTM/DNC-style learned memory: reads use content similarity,
writes blend content addressing with learned allocation, and erase/add updates
remain bounded by convex construction.  Outcome values are only learned token
indices; no outcome sign, task identity, or procedure rule is encoded in code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .situated_feedback import _normalize_masked_scores


@dataclass(frozen=True, slots=True)
class ContentAddressedCausalMemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    step: int

    @property
    def bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.keys, self.values, self.usage)
        )

    def detached_clone(self) -> "ContentAddressedCausalMemoryState":
        return ContentAddressedCausalMemoryState(
            keys=self.keys.detach().clone(),
            values=self.values.detach().clone(),
            usage=self.usage.detach().clone(),
            step=self.step,
        )


@dataclass(frozen=True, slots=True)
class ContentAddressedCausalMemoryOutput:
    weights: torch.Tensor
    scores: torch.Tensor
    normalized_base_scores: torch.Tensor
    residuals: torch.Tensor
    pair_features: torch.Tensor
    temporal_hidden: torch.Tensor
    read_keys: torch.Tensor
    read_weights: torch.Tensor
    read_values: torch.Tensor
    feature_gates: torch.Tensor
    write_gates: torch.Tensor
    write_keys: torch.Tensor
    write_values: torch.Tensor
    erase_gates: torch.Tensor
    write_strengths: torch.Tensor
    allocation_mix: torch.Tensor
    content_strengths: torch.Tensor
    write_weights: torch.Tensor


class ContentAddressedCausalMemoryCore(nn.Module):
    """Exactly sixteen bounded 32-wide slots with learned reads and writes."""

    def __init__(
        self,
        *,
        content_width: int,
        temporal_width: int,
        rank: int = 32,
        memory_slots: int = 16,
        maximum_residual: float = 4.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("content_width", content_width),
            ("temporal_width", temporal_width),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if rank != 32 or memory_slots != 16:
            raise ValueError("V7 memory identity requires exactly 16 slots of width 32")
        if not isinstance(maximum_residual, (int, float)) or maximum_residual <= 0:
            raise ValueError("maximum_residual must be positive")
        self.content_width = content_width
        self.temporal_width = temporal_width
        self.rank = rank
        self.memory_slots = memory_slots
        self.maximum_residual = float(maximum_residual)

        self.query_projection = nn.Linear(content_width, rank, bias=False)
        self.candidate_projection = nn.Linear(content_width, rank, bias=False)
        self.temporal_projection = nn.Linear(temporal_width, rank, bias=False)
        self.semantic_network = nn.Sequential(
            nn.LayerNorm(rank * 4),
            nn.Linear(rank * 4, rank),
            nn.SiLU(),
        )
        self.outcome_embedding = nn.Embedding(3, rank)
        self.outcome_fusion = nn.Sequential(
            nn.LayerNorm(rank * 3),
            nn.Linear(rank * 3, rank),
            nn.SiLU(),
        )
        self.read_key_network = nn.Sequential(
            nn.LayerNorm(rank * 3),
            nn.Linear(rank * 3, rank),
            nn.Tanh(),
        )
        self.read_strength = nn.Linear(rank * 3, 1)
        self.neuromodulator = nn.Sequential(
            nn.LayerNorm(rank * 4),
            nn.Linear(rank * 4, rank),
            nn.SiLU(),
            nn.Linear(rank, rank * 2),
        )
        self.score_network = nn.Sequential(
            nn.LayerNorm(rank * 3),
            nn.Linear(rank * 3, rank),
            nn.SiLU(),
            nn.Linear(rank, 1),
        )
        # key, value, erase, write strength, allocation mix, content strength
        self.writer = nn.Sequential(
            nn.LayerNorm(rank * 4),
            nn.Linear(rank * 4, rank * 2),
            nn.SiLU(),
            nn.Linear(rank * 2, rank * 3 + 3),
        )
        self.allocation_logits = nn.Parameter(torch.zeros(memory_slots))
        self.allocation_usage_pressure = nn.Parameter(torch.zeros(()))
        # Task-neutral learned addresses break the otherwise permanent
        # zero-key/uniform-write symmetry.  They are shared across every task
        # and contain no family, event, or procedure identity.
        self.slot_address_anchors = nn.Parameter(torch.empty(memory_slots, rank))
        nn.init.normal_(self.slot_address_anchors, std=1.0 / rank**0.5)
        nn.init.normal_(self.allocation_logits, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_memory_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ContentAddressedCausalMemoryState:
        reference = next(self.parameters())
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        return ContentAddressedCausalMemoryState(
            keys=torch.tanh(self.slot_address_anchors).to(
                device=resolved_device, dtype=resolved_dtype
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
            step=0,
        )

    # Alias keeps the V5/V6 runner structure easy to adapt.
    initial_plastic_state = initial_memory_state

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_outcomes: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        *,
        memory_state: ContentAddressedCausalMemoryState | None = None,
    ) -> ContentAddressedCausalMemoryOutput:
        self._validate_inputs(
            query_features,
            candidate_features,
            temporal_features,
            candidate_outcomes,
            candidate_mask,
            base_scores,
        )
        state = memory_state or self.initial_memory_state()
        self._validate_state(state)

        query = self.query_projection(query_features.detach())
        candidates = self.candidate_projection(candidate_features.detach())
        temporal_hidden = self.temporal_projection(temporal_features.detach())
        expanded_query = query.unsqueeze(1).expand_as(candidates)
        semantic = self.semantic_network(
            torch.cat(
                (candidates, expanded_query, candidates * expanded_query, temporal_hidden),
                dim=-1,
            )
        )
        outcome = self._outcome_features(candidate_outcomes)
        pairs = self.outcome_fusion(
            torch.cat((semantic, outcome, semantic * outcome), dim=-1)
        )

        read_input = torch.cat((pairs, outcome, temporal_hidden), dim=-1)
        read_keys = self.read_key_network(read_input)
        read_strengths = F.softplus(self.read_strength(read_input).squeeze(-1)) + 1.0
        read_weights = self._content_weights(
            read_keys,
            state.keys,
            read_strengths,
            usage=state.usage,
        )
        read_weights = read_weights.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        read_values = torch.einsum("bns,sr->bnr", read_weights, state.values)

        gate_logits = self.neuromodulator(
            torch.cat((pairs, outcome, temporal_hidden, read_values), dim=-1)
        )
        feature_gates, write_gates = gate_logits.chunk(2, dim=-1)
        feature_gates = torch.sigmoid(feature_gates).masked_fill(
            ~candidate_mask.unsqueeze(-1), 0.0
        )
        write_gates = torch.sigmoid(write_gates).masked_fill(
            ~candidate_mask.unsqueeze(-1), 0.0
        )
        gated_pairs = pairs * feature_gates
        raw_residual = self.score_network(
            torch.cat((gated_pairs, read_values, gated_pairs * read_values), dim=-1)
        ).squeeze(-1)
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        residuals = residuals.masked_fill(~candidate_mask, 0.0)
        normalized_base = _normalize_masked_scores(base_scores.detach(), candidate_mask)
        scores = (normalized_base + residuals).masked_fill(~candidate_mask, -torch.inf)

        write_input = torch.cat((pairs, outcome, temporal_hidden, read_values), dim=-1)
        proposed = self.writer(write_input)
        write_keys = torch.tanh(proposed[..., : self.rank]) * write_gates
        write_values = (
            torch.tanh(proposed[..., self.rank : self.rank * 2]) * write_gates
        )
        erase_gates = (
            torch.sigmoid(proposed[..., self.rank * 2 : self.rank * 3]) * write_gates
        )
        write_strengths = torch.sigmoid(proposed[..., self.rank * 3])
        allocation_mix = torch.sigmoid(proposed[..., self.rank * 3 + 1])
        content_strengths = (
            F.softplus(proposed[..., self.rank * 3 + 2]) + 1.0
        )
        visible = candidate_mask.to(dtype=write_strengths.dtype)
        write_strengths = write_strengths * visible
        allocation_mix = allocation_mix * visible
        content_strengths = content_strengths * visible
        content_write = self._content_weights(
            write_keys,
            state.keys,
            content_strengths.clamp_min(1.0),
        )
        allocation = self._allocation_weights(state.usage).view(1, 1, -1)
        write_weights = write_strengths.unsqueeze(-1) * (
            allocation_mix.unsqueeze(-1) * content_write
            + (1.0 - allocation_mix.unsqueeze(-1)) * allocation
        )
        write_weights = write_weights.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)

        return ContentAddressedCausalMemoryOutput(
            weights=torch.softmax(scores, dim=-1),
            scores=scores,
            normalized_base_scores=normalized_base,
            residuals=residuals,
            pair_features=pairs,
            temporal_hidden=temporal_hidden,
            read_keys=read_keys,
            read_weights=read_weights,
            read_values=read_values,
            feature_gates=feature_gates,
            write_gates=write_gates,
            write_keys=write_keys,
            write_values=write_values,
            erase_gates=erase_gates,
            write_strengths=write_strengths,
            allocation_mix=allocation_mix,
            content_strengths=content_strengths,
            write_weights=write_weights,
        )

    def apply_mixed_outcome_update(
        self,
        state: ContentAddressedCausalMemoryState,
        output: ContentAddressedCausalMemoryOutput,
        candidate_outcomes: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> ContentAddressedCausalMemoryState:
        """Apply one event in supplied order through shared erase/add writes."""

        self._validate_state(state)
        self._validate_outcomes(candidate_outcomes, candidate_mask)
        if candidate_mask.shape[0] != 1:
            raise ValueError("one persistent memory accepts one chronological event at a time")
        self._validate_output(output, candidate_mask)

        keys, values, usage = state.keys, state.values, state.usage
        for candidate in range(candidate_mask.shape[1]):
            strength = output.write_strengths[0, candidate]
            content = self._content_weights(
                output.write_keys[:, candidate : candidate + 1],
                keys,
                output.content_strengths[:, candidate : candidate + 1].clamp_min(1.0),
            ).squeeze(0).squeeze(0)
            allocation = self._allocation_weights(usage)
            mix = output.allocation_mix[0, candidate]
            slot_weights = strength * (mix * content + (1.0 - mix) * allocation)

            slot = slot_weights.unsqueeze(-1)
            write_key = output.write_keys[0, candidate].unsqueeze(0)
            write_value = output.write_values[0, candidate].unsqueeze(0)
            erase = output.erase_gates[0, candidate].unsqueeze(0)
            keys = (1.0 - slot) * keys + slot * write_key
            erased_values = values * (1.0 - slot * erase)
            values = (1.0 - slot) * erased_values + slot * write_value
            usage = usage + slot_weights * (1.0 - usage)

        updated = ContentAddressedCausalMemoryState(
            keys=keys,
            values=values,
            usage=usage,
            step=state.step + 1,
        )
        self._validate_state(updated)
        return updated.detached_clone() if detach_state else updated

    def _outcome_features(self, outcomes: torch.Tensor) -> torch.Tensor:
        return self.outcome_embedding(outcomes.detach().to(dtype=torch.long) + 1)

    def _content_weights(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        strengths: torch.Tensor,
        *,
        usage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_norm = F.normalize(queries, dim=-1, eps=1.0e-6)
        key_norm = F.normalize(keys, dim=-1, eps=1.0e-6)
        logits = torch.einsum("...r,sr->...s", query_norm, key_norm)
        logits = logits * strengths.unsqueeze(-1)
        if usage is not None:
            logits = logits + torch.log(usage.clamp_min(1.0e-4))
        return torch.softmax(logits, dim=-1)

    def _allocation_weights(self, usage: torch.Tensor) -> torch.Tensor:
        pressure = F.softplus(self.allocation_usage_pressure) + 1.0
        return torch.softmax(self.allocation_logits - pressure * usage, dim=-1)

    def _validate_inputs(self, query, candidates, temporal, outcomes, mask, base) -> None:
        reference = next(self.parameters())
        if query.ndim != 2 or query.shape[-1] != self.content_width:
            raise ValueError("query_features must be [batch, content_width]")
        if candidates.ndim != 3 or candidates.shape != (
            query.shape[0], candidates.shape[1], self.content_width
        ):
            raise ValueError("candidate_features must be [batch, candidates, content_width]")
        if temporal.shape != (*candidates.shape[:2], self.temporal_width):
            raise ValueError("temporal_features must be [batch, candidates, temporal_width]")
        self._validate_outcomes(outcomes, mask)
        if base.shape != mask.shape:
            raise ValueError("base_scores must be [batch, candidates]")
        for name, value in (
            ("query_features", query),
            ("candidate_features", candidates),
            ("temporal_features", temporal),
            ("base_scores", base),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match the core device and dtype")
            visible = value if name != "base_scores" else value[mask]
            if not bool(torch.isfinite(visible).all().item()):
                raise ValueError(f"{name} must be finite")

    def _validate_outcomes(self, outcomes, mask) -> None:
        reference = next(self.parameters())
        if mask.dtype is not torch.bool or outcomes.shape != mask.shape:
            raise ValueError("candidate_outcomes and boolean candidate_mask must be [batch, candidates]")
        if outcomes.device != reference.device or outcomes.dtype != reference.dtype:
            raise ValueError("candidate_outcomes must match the core device and dtype")
        if mask.device != reference.device or not bool(mask.any(dim=1).all().item()):
            raise ValueError("candidate_mask must match the core and expose one candidate per row")
        if not bool(torch.isfinite(outcomes).all().item()):
            raise ValueError("candidate_outcomes must be finite")
        if not bool(((outcomes == -1) | (outcomes == 0) | (outcomes == 1)).all().item()):
            raise ValueError("candidate_outcomes must contain only -1, 0, or +1")

    def _validate_state(self, state: ContentAddressedCausalMemoryState) -> None:
        if not isinstance(state, ContentAddressedCausalMemoryState):
            raise TypeError("memory_state must be ContentAddressedCausalMemoryState")
        reference = next(self.parameters())
        expected = (self.memory_slots, self.rank)
        if state.keys.shape != expected or state.values.shape != expected:
            raise ValueError("memory keys/values have the wrong shape")
        if state.usage.shape != (self.memory_slots,):
            raise ValueError("memory usage has the wrong shape")
        if type(state.step) is not int or state.step < 0:
            raise ValueError("memory step must be a non-negative integer")
        for name, value in (
            ("keys", state.keys),
            ("values", state.values),
            ("usage", state.usage),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("memory state must match the core device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("memory state must be finite")
            if name in {"keys", "values"} and not bool((value.abs() <= 1.0 + 1.0e-6).all().item()):
                raise ValueError(f"memory {name} exceeded its [-1, 1] bound")
        if not bool(((state.usage >= 0.0) & (state.usage <= 1.0 + 1.0e-6)).all().item()):
            raise ValueError("memory usage exceeded its [0, 1] bound")

    def _validate_output(
        self,
        output: ContentAddressedCausalMemoryOutput,
        mask: torch.Tensor,
    ) -> None:
        reference = next(self.parameters())
        expected_vector = (*mask.shape, self.rank)
        expected_slots = (*mask.shape, self.memory_slots)
        for name in ("pair_features", "temporal_hidden", "write_keys", "write_values", "erase_gates"):
            value = getattr(output, name)
            if value.shape != expected_vector:
                raise ValueError(f"output {name} has the wrong shape")
        if output.write_weights.shape != expected_slots:
            raise ValueError("output write_weights has the wrong shape")
        for name in ("write_strengths", "allocation_mix", "content_strengths"):
            if getattr(output, name).shape != mask.shape:
                raise ValueError(f"output {name} has the wrong shape")
        for name in ("pair_features", "temporal_hidden", "write_keys", "write_values", "erase_gates", "write_strengths", "allocation_mix", "content_strengths", "write_weights"):
            value = getattr(output, name)
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"output {name} must match the core device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"output {name} must be finite")


__all__ = [
    "ContentAddressedCausalMemoryCore",
    "ContentAddressedCausalMemoryOutput",
    "ContentAddressedCausalMemoryState",
]
