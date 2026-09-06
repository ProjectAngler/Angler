"""Shared relation-only semantic addressing for strict credit memory.

V17 changes exactly one mechanism from the consumed V16 core: independent
temporal-conditioned query and key projections are replaced by one shared
Siamese metric over detached frozen relation features.  Outcome value content,
value-only decoding, state, allocation, replay, and bounds remain inherited.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .strict_key_value_credit_memory import (
    MEMORY_RANK,
    MEMORY_SLOTS,
    RELATION_WIDTH,
    StrictKeyValueCreditEvent,
    StrictKeyValueCreditMemoryCore,
    StrictKeyValueCreditOutput,
    StrictKeyValueCreditSnapshot,
    StrictKeyValueCreditState,
    _require_bool,
)


SharedSemanticMetricCreditEvent = StrictKeyValueCreditEvent
SharedSemanticMetricCreditOutput = StrictKeyValueCreditOutput
SharedSemanticMetricCreditSnapshot = StrictKeyValueCreditSnapshot
SharedSemanticMetricCreditState = StrictKeyValueCreditState


class SharedSemanticMetricCreditMemoryCore(StrictKeyValueCreditMemoryCore):
    """V16 strict memory with one shared normalized relation metric."""

    def __init__(
        self,
        *,
        temporal_width: int,
        relation_width: int = RELATION_WIDTH,
        rank: int = MEMORY_RANK,
        memory_slots: int = MEMORY_SLOTS,
        maximum_residual: float = 4.0,
    ) -> None:
        super().__init__(
            temporal_width=temporal_width,
            relation_width=relation_width,
            rank=rank,
            memory_slots=memory_slots,
            maximum_residual=maximum_residual,
        )
        # Remove the consumed V16 address projections completely so they
        # cannot remain as unused trainable parameters or an accidental path.
        del self.query_network
        del self.key_network
        self.semantic_metric_network = nn.Sequential(
            nn.LayerNorm(relation_width),
            nn.Linear(relation_width, relation_width),
            nn.SiLU(),
            nn.Linear(relation_width, rank),
        )

    def semantic_codes(self, relation_features: torch.Tensor) -> torch.Tensor:
        """Encode detached relations in the one shared normalized metric."""

        reference = next(self.parameters())
        if (
            relation_features.ndim != 2
            or relation_features.shape[-1] != self.relation_width
            or relation_features.shape[0] == 0
            or relation_features.device != reference.device
            or relation_features.dtype != reference.dtype
            or not relation_features.is_floating_point()
            or not bool(torch.isfinite(relation_features).all().item())
        ):
            raise ValueError("relation features must be finite [batch,64] core tensors")
        raw = self.semantic_metric_network(relation_features.detach())
        return F.normalize(raw, dim=-1, eps=1.0e-6)

    def predict(
        self,
        relation_features: torch.Tensor,
        temporal_features: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        state: StrictKeyValueCreditState | None = None,
        read_enabled: bool = True,
    ) -> StrictKeyValueCreditOutput:
        """Predict with shared semantic content and temporal-only gates."""

        _require_bool("read_enabled", read_enabled)
        self._validate_observations(relation_features, temporal_features, base_logits)
        current = self.initial_state() if state is None else state
        self.validate_state(current)

        relations = relation_features.detach()
        temporal = temporal_features.detach()
        base = base_logits.detach()
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

        read_weights = self._content_read_weights(semantic, read_strengths, current)
        if not read_enabled:
            read_weights = torch.zeros_like(read_weights)
        read_values = read_weights @ current.values
        residuals = self.decode_retrieved_values(read_values)
        if not read_enabled:
            residuals = torch.zeros_like(residuals)
        output = StrictKeyValueCreditOutput(
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
            state_step=current.step,
            read_enabled=read_enabled,
        )
        self._validate_output(output)
        return output

    forward = predict

    def parameter_groups(self) -> dict[str, tuple[str, ...]]:
        prefixes = {
            "semantic_metric": ("semantic_metric_network.",),
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
            raise RuntimeError("V17 parameter ownership is incomplete")
        if any(name.startswith(("query_network.", "key_network.")) for name in names):
            raise RuntimeError("V17 retained a prohibited independent metric")
        return groups

    def architecture_report(self) -> dict[str, Any]:
        groups = self.parameter_groups()
        if tuple(self.named_buffers()):
            raise RuntimeError("V17 must not own frozen representation tensors")
        return {
            "parameter_groups": groups,
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
            "stored_value_inputs": ("transformed_outcome_embedding",),
            "residual_decoder_inputs": ("retrieved_value",),
            "independent_query_or_key_networks": False,
            "temporal_affects_semantic_code_or_decoder": False,
            "outcome_affects_semantic_code_or_gates": False,
            "metadata_inputs": False,
            "generic_allocation": "stable_least_unused",
            "erase_gate_role": "fresh_slot_noop_under_stable_least_unused",
            "erase_gate_active_before_capacity": False,
            "learned_retention_evidence_claimed": False,
            "v13_parameters_owned": False,
        }

    parameter_report = architecture_report


__all__ = [
    "MEMORY_RANK",
    "MEMORY_SLOTS",
    "RELATION_WIDTH",
    "SharedSemanticMetricCreditEvent",
    "SharedSemanticMetricCreditMemoryCore",
    "SharedSemanticMetricCreditOutput",
    "SharedSemanticMetricCreditSnapshot",
    "SharedSemanticMetricCreditState",
    "StrictKeyValueCreditEvent",
    "StrictKeyValueCreditOutput",
    "StrictKeyValueCreditSnapshot",
    "StrictKeyValueCreditState",
]
