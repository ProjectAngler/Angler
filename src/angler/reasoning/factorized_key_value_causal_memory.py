"""Outcome-invariant addresses with learned outcome-bound memory values.

The core borrows the key/value separation of Key-Value Memory Networks.  A
procedure's semantic and temporal representation determines its address;
feedback affects only the stored value.  Success and failure remain unrelated
learned tokens.  The neutral token is padding and therefore contributes an
exactly zero value without encoding either token's desirability in code.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .content_addressed_causal_memory import (
    ContentAddressedCausalMemoryOutput,
    ContentAddressedCausalMemoryState,
)
from .situated_feedback import _normalize_masked_scores
from .sparse_memory_causal_routing import SparseMemoryCausalRoutingCore


class FactorizedKeyValueCausalMemoryCore(SparseMemoryCausalRoutingCore):
    """A bounded top-2 memory with outcome-free keys and bound values."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        rank = self.rank

        # Index 1 corresponds to the neutral outcome (outcome + 1). PyTorch
        # keeps padding_idx exactly zero and excludes it from embedding updates.
        self.outcome_embedding = nn.Embedding(3, rank, padding_idx=1)
        self.read_key_network = nn.Sequential(
            nn.LayerNorm(rank * 2),
            nn.Linear(rank * 2, rank),
            nn.Tanh(),
        )
        self.read_strength = nn.Linear(rank * 2, 1)
        # Address controls are outcome-free by construction: per-feature write
        # gate, scalar strength, content/allocation mix, and content strength.
        self.address_controls = nn.Sequential(
            nn.LayerNorm(rank * 2),
            nn.Linear(rank * 2, rank * 2),
            nn.SiLU(),
            nn.Linear(rank * 2, rank + 3),
        )
        self.semantic_payload_projection = nn.Linear(rank, rank, bias=False)
        self.outcome_payload_projection = nn.Linear(rank, rank, bias=False)
        # Bias-free writer preserves exact zero for the neutral payload.
        self.writer = nn.Sequential(
            nn.Linear(rank, rank * 2, bias=False),
            nn.SiLU(),
            nn.Linear(rank * 2, rank, bias=False),
        )
        # Modulation acts only after retrieval, so it cannot move addresses.
        self.neuromodulator = nn.Sequential(
            nn.Linear(rank * 3, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, rank, bias=False),
        )

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

        # This entire address side is independent of candidate_outcomes.
        address_input = torch.cat((semantic, temporal_hidden), dim=-1)
        read_keys = self.read_key_network(address_input)
        read_strengths = F.softplus(self.read_strength(address_input).squeeze(-1)) + 1.0
        read_weights = self._content_weights(
            read_keys,
            state.keys,
            read_strengths,
            usage=state.usage,
        )
        read_weights = read_weights.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        read_values = torch.einsum("bns,sr->bnr", read_weights, state.values)

        score_features = torch.cat(
            (semantic, read_values, semantic * read_values), dim=-1
        )
        feature_gates = torch.sigmoid(self.neuromodulator(score_features))
        feature_gates = feature_gates.masked_fill(
            ~candidate_mask.unsqueeze(-1), 0.0
        )
        gated_semantic = semantic * feature_gates
        raw_residual = self.score_network(
            torch.cat(
                (gated_semantic, read_values, gated_semantic * read_values), dim=-1
            )
        ).squeeze(-1)
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        residuals = residuals.masked_fill(~candidate_mask, 0.0)
        normalized_base = _normalize_masked_scores(base_scores.detach(), candidate_mask)
        scores = (normalized_base + residuals).masked_fill(~candidate_mask, -torch.inf)

        controls = self.address_controls(address_input)
        write_gates = torch.sigmoid(controls[..., : self.rank])
        visible = candidate_mask.to(dtype=write_gates.dtype)
        write_gates = write_gates * visible.unsqueeze(-1)
        write_strengths = torch.sigmoid(controls[..., self.rank]) * visible
        allocation_mix = torch.sigmoid(controls[..., self.rank + 1]) * visible
        content_strengths = (
            F.softplus(controls[..., self.rank + 2]) + 1.0
        ) * visible
        write_keys = read_keys * write_gates

        outcome = self._outcome_features(candidate_outcomes)
        semantic_payload = self.semantic_payload_projection(semantic)
        outcome_payload = self.outcome_payload_projection(outcome)
        bound_payload = semantic_payload * outcome_payload
        write_values = torch.tanh(self.writer(bound_payload)) * write_gates
        # Erasure is address-side and consequently invariant under outcome
        # permutation; all polarity information remains in write_values.
        erase_gates = write_gates

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
        write_weights = write_weights.masked_fill(
            ~candidate_mask.unsqueeze(-1), 0.0
        )
        write_weights = self._top_k_preserve_mass(write_weights, self.routing_k)

        return ContentAddressedCausalMemoryOutput(
            weights=torch.softmax(scores, dim=-1),
            scores=scores,
            normalized_base_scores=normalized_base,
            residuals=residuals,
            pair_features=semantic,
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


__all__ = ["FactorizedKeyValueCausalMemoryCore"]
