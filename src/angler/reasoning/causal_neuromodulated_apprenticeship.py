"""Outcome-causal ANML-style modulation for apprenticeship state updates.

The V5 OML representation and functional state remain intact.  This successor
adds one shared learned modulator which observes semantic pair features,
learned outcome tokens, Moving-Origin features, and current fast-state context.
It emits bounded feature and write gates; no branch assigns a meaning or sign
to any outcome token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .outcome_aware_apprenticeship import (
    OutcomeAwareApprenticeshipCore,
    OutcomeAwarePlasticState,
)
from .situated_feedback import OMLSituatedFeedbackPolicy, _normalize_masked_scores


@dataclass(frozen=True, slots=True)
class CausalNeuromodulatedOutput:
    weights: torch.Tensor
    scores: torch.Tensor
    normalized_base_scores: torch.Tensor
    residuals: torch.Tensor
    pair_features: torch.Tensor
    feature_gates: torch.Tensor
    write_gates: torch.Tensor


class CausalNeuromodulatedApprenticeshipCore(OutcomeAwareApprenticeshipCore):
    """V5 apprenticeship with one outcome-causal learned neuromodulator."""

    def __init__(
        self,
        *,
        content_width: int,
        temporal_width: int,
        rank: int = 32,
        maximum_residual: float = 4.0,
        maximum_update: float = 0.25,
    ) -> None:
        super().__init__(
            content_width=content_width,
            temporal_width=temporal_width,
            rank=rank,
            maximum_residual=maximum_residual,
            maximum_update=maximum_update,
        )
        self.neuromodulator = nn.Sequential(
            nn.LayerNorm(rank * 4),
            nn.Linear(rank * 4, rank),
            nn.SiLU(),
            nn.Linear(rank, rank * 2),
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
        plastic_state: OutcomeAwarePlasticState | None = None,
    ) -> CausalNeuromodulatedOutput:
        self._validate_outcomes(candidate_outcomes, candidate_mask)
        self._validate_base_scores(base_scores, candidate_mask)
        state = plastic_state or self.initial_plastic_state()
        self._validate_plastic_state(state)

        # Call the inherited OML representation directly, before V5 outcome
        # fusion, so the modulator receives a distinct semantic channel.
        semantic_pairs = OMLSituatedFeedbackPolicy.pair_features(
            self,
            query_features,
            candidate_features,
            temporal_features,
            candidate_mask,
        )
        outcome_features = self._outcome_features(candidate_outcomes)
        temporal_hidden = self.temporal_projection(temporal_features.detach())
        state_context = torch.tanh(state.fast_weight).view(1, 1, -1).expand_as(
            semantic_pairs
        )
        gate_logits = self.neuromodulator(
            torch.cat(
                (
                    semantic_pairs,
                    outcome_features,
                    temporal_hidden,
                    state_context,
                ),
                dim=-1,
            )
        )
        feature_gates, write_gates = gate_logits.chunk(2, dim=-1)
        feature_gates = torch.sigmoid(feature_gates)
        write_gates = torch.sigmoid(write_gates)
        feature_gates = feature_gates.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        write_gates = write_gates.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)

        v5_pairs = self.outcome_fusion(
            torch.cat(
                (
                    semantic_pairs,
                    outcome_features,
                    semantic_pairs * outcome_features,
                ),
                dim=-1,
            )
        )
        pair_features = v5_pairs * feature_gates
        raw_residual = (
            torch.einsum("bnr,r->bn", pair_features, state.fast_weight)
            + state.fast_bias
        )
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        residuals = residuals.masked_fill(~candidate_mask, 0.0)
        normalized_base = _normalize_masked_scores(base_scores.detach(), candidate_mask)
        scores = (normalized_base + residuals).masked_fill(~candidate_mask, -torch.inf)
        return CausalNeuromodulatedOutput(
            weights=torch.softmax(scores, dim=-1),
            scores=scores,
            normalized_base_scores=normalized_base,
            residuals=residuals,
            pair_features=pair_features,
            feature_gates=feature_gates,
            write_gates=write_gates,
        )

    def apply_mixed_outcome_update(
        self,
        state: OutcomeAwarePlasticState,
        output: CausalNeuromodulatedOutput,
        candidate_outcomes: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> OutcomeAwarePlasticState:
        """Apply one bounded, vector-gated functional state update."""

        self._validate_plastic_state(state)
        self._validate_outcomes(candidate_outcomes, candidate_mask)
        expected_features = (*candidate_mask.shape, self.rank)
        if output.pair_features.shape != expected_features:
            raise ValueError("output pair_features do not align with the candidate batch")
        if output.feature_gates.shape != expected_features or output.write_gates.shape != expected_features:
            raise ValueError("neuromodulator gates do not align with the candidate batch")
        if output.weights.shape != candidate_mask.shape:
            raise ValueError("output weights do not align with the candidate batch")
        if candidate_mask.shape[0] != 1:
            raise ValueError("one persistent state accepts one chronological event at a time")
        reference = next(self.parameters())
        for name, value in (
            ("output pair_features", output.pair_features),
            ("output weights", output.weights),
            ("output feature_gates", output.feature_gates),
            ("output write_gates", output.write_gates),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match the core device and dtype")
            if not bool(torch.isfinite(value[candidate_mask]).all().item()):
                raise ValueError(f"{name} must be finite on visible candidates")
        if not bool(
            (
                (output.feature_gates[candidate_mask] >= 0.0)
                & (output.feature_gates[candidate_mask] <= 1.0)
                & (output.write_gates[candidate_mask] >= 0.0)
                & (output.write_gates[candidate_mask] <= 1.0)
            ).all().item()
        ):
            raise ValueError("neuromodulator gates must remain in [0, 1]")

        outcome_features = self._outcome_features(candidate_outcomes)
        writer_input = torch.cat(
            (
                output.pair_features,
                outcome_features,
                output.pair_features * outcome_features,
                output.weights.unsqueeze(-1),
            ),
            dim=-1,
        )
        proposed = self.state_writer(writer_input)
        proposed_weight = torch.tanh(proposed[..., : self.rank])
        proposed_bias = torch.tanh(proposed[..., self.rank])
        shared_gates = torch.sigmoid(proposed[..., self.rank + 1])
        visible = candidate_mask.to(dtype=proposed.dtype)
        denominator = visible.sum().clamp_min(1.0)
        vector_gate = shared_gates.unsqueeze(-1) * output.write_gates
        weight_update = (
            proposed_weight * vector_gate * visible.unsqueeze(-1)
        ).sum(dim=(0, 1)) / denominator
        bias_gate = output.write_gates.mean(dim=-1) * shared_gates
        bias_update = (proposed_bias * bias_gate * visible).sum() / denominator
        updated = OutcomeAwarePlasticState(
            fast_weight=state.fast_weight + self.maximum_update * weight_update,
            fast_bias=state.fast_bias + self.maximum_update * bias_update,
            step=state.step + 1,
        )
        self._validate_plastic_state(updated)
        return updated.detached_clone() if detach_state else updated


__all__ = [
    "CausalNeuromodulatedApprenticeshipCore",
    "CausalNeuromodulatedOutput",
]
