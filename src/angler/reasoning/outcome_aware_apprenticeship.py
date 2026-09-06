"""Outcome-conditioned functional OML state for procedural apprenticeship.

This module extends the existing situated OML representation with a learned
three-token outcome channel and a learned, constant-size state writer.  It
does not interpret success or failure in code: every visible candidate,
including candidates whose outcome is unknown, passes through the same
learned scorer and update rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .situated_feedback import OMLSituatedFeedbackPolicy, _normalize_masked_scores


@dataclass(frozen=True, slots=True)
class OutcomeAwarePlasticState:
    """One constant-size fast state shared by the chronological stream."""

    fast_weight: torch.Tensor
    fast_bias: torch.Tensor
    step: int

    def detached_clone(self) -> "OutcomeAwarePlasticState":
        return OutcomeAwarePlasticState(
            fast_weight=self.fast_weight.detach().clone(),
            fast_bias=self.fast_bias.detach().clone(),
            step=self.step,
        )


@dataclass(frozen=True, slots=True)
class OutcomeAwareApprenticeshipOutput:
    """Candidate selection plus the learned features used by the writer."""

    weights: torch.Tensor
    scores: torch.Tensor
    normalized_base_scores: torch.Tensor
    residuals: torch.Tensor
    pair_features: torch.Tensor


class OutcomeAwareApprenticeshipCore(OMLSituatedFeedbackPolicy):
    """A slow learned representation with one functional, persistent OML head.

    ``-1``, ``0``, and ``+1`` are merely indices into a learned embedding.
    There is no outcome-based masking, sign multiplication, sorting, or
    hand-written preference.  Calling :meth:`apply_mixed_outcome_update` with
    ``detach_state=False`` preserves the inner-update graph so a later outer
    loss can train the representation, outcome channel, and writer.
    """

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
        )
        if not isinstance(maximum_update, (int, float)) or maximum_update <= 0:
            raise ValueError("maximum_update must be positive")
        self.maximum_update = float(maximum_update)
        self.outcome_embedding = nn.Embedding(3, rank)
        self.outcome_fusion = nn.Sequential(
            nn.LayerNorm(rank * 3),
            nn.Linear(rank * 3, rank),
            nn.SiLU(),
        )
        # The writer emits a proposed fast weight, bias, and learned gate for
        # every visible candidate.  All outcome classes use these same layers.
        self.state_writer = nn.Sequential(
            nn.LayerNorm(rank * 3 + 1),
            nn.Linear(rank * 3 + 1, rank),
            nn.SiLU(),
            nn.Linear(rank, rank + 2),
        )

    def initial_plastic_state(self, *, detach: bool = False) -> OutcomeAwarePlasticState:
        """Return the learned OML initialization as a functional state."""

        state = OutcomeAwarePlasticState(
            fast_weight=self.fast_weight,
            fast_bias=self.fast_bias,
            step=0,
        )
        return state.detached_clone() if detach else state

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
    ) -> OutcomeAwareApprenticeshipOutput:
        self._validate_outcomes(candidate_outcomes, candidate_mask)
        self._validate_base_scores(base_scores, candidate_mask)
        state = plastic_state or self.initial_plastic_state()
        self._validate_plastic_state(state)

        # ``pair_features`` detaches foundation semantics and Moving-Origin
        # observations before applying the trainable slow representation.
        semantic_pairs = super().pair_features(
            query_features,
            candidate_features,
            temporal_features,
            candidate_mask,
        )
        outcome_features = self._outcome_features(candidate_outcomes)
        pair_features = self.outcome_fusion(
            torch.cat(
                (
                    semantic_pairs,
                    outcome_features,
                    semantic_pairs * outcome_features,
                ),
                dim=-1,
            )
        )
        raw_residual = (
            torch.einsum("bnr,r->bn", pair_features, state.fast_weight)
            + state.fast_bias
        )
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        residuals = residuals.masked_fill(~candidate_mask, 0.0)
        normalized_base = _normalize_masked_scores(base_scores.detach(), candidate_mask)
        scores = (normalized_base + residuals).masked_fill(~candidate_mask, -torch.inf)
        return OutcomeAwareApprenticeshipOutput(
            weights=torch.softmax(scores, dim=-1),
            scores=scores,
            normalized_base_scores=normalized_base,
            residuals=residuals,
            pair_features=pair_features,
        )

    def apply_mixed_outcome_update(
        self,
        state: OutcomeAwarePlasticState,
        output: OutcomeAwareApprenticeshipOutput,
        candidate_outcomes: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        detach_state: bool = True,
    ) -> OutcomeAwarePlasticState:
        """Apply one chronological learned update from a mixed-outcome event."""

        self._validate_plastic_state(state)
        self._validate_outcomes(candidate_outcomes, candidate_mask)
        if output.pair_features.ndim != 3 or output.pair_features.shape != (
            candidate_mask.shape[0],
            candidate_mask.shape[1],
            self.rank,
        ):
            raise ValueError("output pair_features do not align with the candidate batch")
        if output.weights.shape != candidate_mask.shape:
            raise ValueError("output weights do not align with the candidate batch")
        if candidate_mask.shape[0] != 1:
            raise ValueError("one persistent state accepts one chronological event at a time")
        reference = next(self.parameters())
        for name, value in (
            ("output pair_features", output.pair_features),
            ("output weights", output.weights),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match the core device and dtype")
            visible = value[candidate_mask]
            if not bool(torch.isfinite(visible).all().item()):
                raise ValueError(f"{name} must be finite on visible candidates")

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
        gates = torch.sigmoid(proposed[..., self.rank + 1])
        visible = candidate_mask.to(dtype=proposed.dtype)
        denominator = visible.sum().clamp_min(1.0)
        weight_update = (
            proposed_weight * gates.unsqueeze(-1) * visible.unsqueeze(-1)
        ).sum(dim=(0, 1)) / denominator
        bias_update = (proposed_bias * gates * visible).sum() / denominator
        updated = OutcomeAwarePlasticState(
            fast_weight=state.fast_weight + self.maximum_update * weight_update,
            fast_bias=state.fast_bias + self.maximum_update * bias_update,
            step=state.step + 1,
        )
        self._validate_plastic_state(updated)
        return updated.detached_clone() if detach_state else updated

    def _outcome_features(self, outcomes: torch.Tensor) -> torch.Tensor:
        # Tokenization is representation plumbing, not an outcome preference.
        token_indices = outcomes.detach().to(dtype=torch.long) + 1
        return self.outcome_embedding(token_indices)

    def _validate_outcomes(self, outcomes: torch.Tensor, mask: torch.Tensor) -> None:
        reference = next(self.parameters())
        if mask.dtype is not torch.bool or outcomes.shape != mask.shape:
            raise ValueError("candidate_outcomes and boolean candidate_mask must be [batch, candidates]")
        if outcomes.device != reference.device or outcomes.dtype != reference.dtype:
            raise ValueError("candidate_outcomes must match the core device and dtype")
        if mask.device != reference.device:
            raise ValueError("candidate_mask must match the core device")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("every row must expose at least one candidate")
        if not bool(torch.isfinite(outcomes).all().item()):
            raise ValueError("candidate_outcomes must be finite")
        if not bool(((outcomes == -1) | (outcomes == 0) | (outcomes == 1)).all().item()):
            raise ValueError("candidate_outcomes must contain only -1, 0, or +1")

    def _validate_base_scores(self, scores: torch.Tensor, mask: torch.Tensor) -> None:
        reference = next(self.parameters())
        if scores.shape != mask.shape:
            raise ValueError("base_scores must be [batch, candidates]")
        if scores.device != reference.device or scores.dtype != reference.dtype:
            raise ValueError("base_scores must match the core device and dtype")
        if not bool(torch.isfinite(scores[mask]).all().item()):
            raise ValueError("base_scores must be finite on visible candidates")

    def _validate_plastic_state(self, state: OutcomeAwarePlasticState) -> None:
        if not isinstance(state, OutcomeAwarePlasticState):
            raise TypeError("plastic_state must be OutcomeAwarePlasticState")
        reference = next(self.parameters())
        if state.fast_weight.shape != (self.rank,) or state.fast_bias.shape != ():
            raise ValueError("plastic state has the wrong shape")
        if type(state.step) is not int or state.step < 0:
            raise ValueError("plastic state step must be a non-negative integer")
        for value in (state.fast_weight, state.fast_bias):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("plastic state must match the core device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("plastic state must be finite")


__all__ = [
    "OutcomeAwareApprenticeshipCore",
    "OutcomeAwareApprenticeshipOutput",
    "OutcomeAwarePlasticState",
]
