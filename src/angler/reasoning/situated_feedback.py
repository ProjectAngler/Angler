"""Replay-free outcome adaptation for situated evidence selection.

The slow situated reader and foundation representations remain frozen.  This
module owns only a small, bounded fast residual over candidate-selection
scores.  It receives the consequence of the selected evidence, never the
correct answer or a hand-written procedure rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SituatedFeedbackOutput:
    weights: torch.Tensor
    scores: torch.Tensor
    normalized_base_scores: torch.Tensor
    residuals: torch.Tensor


class SituatedFeedbackPolicy(nn.Module):
    """Constant-size fast policy over a frozen situated reader."""

    def __init__(
        self,
        *,
        content_width: int,
        temporal_width: int,
        rank: int = 32,
        maximum_residual: float = 4.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("content_width", content_width),
            ("temporal_width", temporal_width),
            ("rank", rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(maximum_residual, (int, float)) or maximum_residual <= 0:
            raise ValueError("maximum_residual must be positive")
        self.content_width = content_width
        self.temporal_width = temporal_width
        self.rank = rank
        self.maximum_residual = float(maximum_residual)
        self.query_projection = nn.Linear(content_width, rank, bias=False)
        self.candidate_projection = nn.Linear(content_width, rank, bias=False)
        self.temporal_projection = nn.Linear(temporal_width, rank, bias=False)
        self.residual_network = nn.Sequential(
            nn.LayerNorm(rank * 4),
            nn.Linear(rank * 4, rank),
            nn.SiLU(),
            nn.Linear(rank, 1),
        )
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    @property
    def plastic_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> SituatedFeedbackOutput:
        self._validate_inputs(
            query_features,
            candidate_features,
            temporal_features,
            candidate_mask,
            base_scores,
        )
        # Foundation and slow-reader values are observations, never update
        # targets of the fast outcome learner.
        query = query_features.detach()
        candidates = candidate_features.detach()
        temporal = temporal_features.detach()
        base = base_scores.detach()

        query_hidden = self.query_projection(query)
        candidate_hidden = self.candidate_projection(candidates)
        temporal_hidden = self.temporal_projection(temporal)
        expanded_query = query_hidden.unsqueeze(1).expand_as(candidate_hidden)
        raw_residual = self.residual_network(
            torch.cat(
                (
                    candidate_hidden,
                    expanded_query,
                    candidate_hidden * expanded_query,
                    temporal_hidden,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        residuals = residuals.masked_fill(~candidate_mask, 0.0)
        normalized_base = _normalize_masked_scores(base, candidate_mask)
        scores = (normalized_base + residuals).masked_fill(~candidate_mask, -torch.inf)
        return SituatedFeedbackOutput(
            weights=torch.softmax(scores, dim=-1),
            scores=scores,
            normalized_base_scores=normalized_base,
            residuals=residuals,
        )

    def capture_plastic_state(self) -> dict[str, torch.Tensor]:
        """Return an independent constant-size snapshot for rollback/reset."""

        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }

    def restore_plastic_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("plastic-state keys do not match this policy")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
                raise ValueError(f"plastic-state tensor {name!r} has the wrong shape")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"plastic-state tensor {name!r} is not finite")
        self.load_state_dict(dict(state), strict=True)

    def plastic_state_digest(self) -> str:
        digest = hashlib.sha256(b"project-angler.situated-feedback-policy.v1\x00")
        for name, value in sorted(self.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(str(tensor.dtype).encode("ascii") + b"\x00")
            for size in tensor.shape:
                digest.update(int(size).to_bytes(8, "big"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return "sha256:" + digest.hexdigest()

    def _validate_inputs(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        temporal: torch.Tensor,
        mask: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> None:
        if query.ndim != 2 or query.shape[-1] != self.content_width:
            raise ValueError("query_features must be [batch, content_width]")
        expected_candidates = (query.shape[0], candidates.shape[1], self.content_width)
        if candidates.ndim != 3 or candidates.shape != expected_candidates:
            raise ValueError("candidate_features must be [batch, candidates, content_width]")
        expected_temporal = (query.shape[0], candidates.shape[1], self.temporal_width)
        if temporal.shape != expected_temporal:
            raise ValueError("temporal_features must be [batch, candidates, temporal_width]")
        if mask.dtype is not torch.bool or mask.shape != candidates.shape[:2]:
            raise ValueError("candidate_mask must be boolean [batch, candidates]")
        if base_scores.shape != mask.shape:
            raise ValueError("base_scores must be [batch, candidates]")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("every row must expose at least one candidate")
        reference = next(self.parameters())
        for name, value in (
            ("query_features", query),
            ("candidate_features", candidates),
            ("temporal_features", temporal),
            ("base_scores", base_scores),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match policy device and dtype")
            visible = value if name != "base_scores" else value[mask]
            if not bool(torch.isfinite(visible).all().item()):
                raise ValueError(f"{name} must be finite on visible values")
        if mask.device != reference.device:
            raise ValueError("candidate_mask must match policy device")


class OMLSituatedFeedbackPolicy(nn.Module):
    """Meta-learned slow pair representation with a tiny online fast head."""

    def __init__(
        self,
        *,
        content_width: int,
        temporal_width: int,
        rank: int = 32,
        maximum_residual: float = 4.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("content_width", content_width),
            ("temporal_width", temporal_width),
            ("rank", rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(maximum_residual, (int, float)) or maximum_residual <= 0:
            raise ValueError("maximum_residual must be positive")
        self.content_width = content_width
        self.temporal_width = temporal_width
        self.rank = rank
        self.maximum_residual = float(maximum_residual)
        self.query_projection = nn.Linear(content_width, rank, bias=False)
        self.candidate_projection = nn.Linear(content_width, rank, bias=False)
        self.temporal_projection = nn.Linear(temporal_width, rank, bias=False)
        self.feature_network = nn.Sequential(
            nn.LayerNorm(rank * 4),
            nn.Linear(rank * 4, rank),
            nn.SiLU(),
        )
        # Only these 33 values change during deployment when rank=32.
        self.fast_weight = nn.Parameter(torch.zeros(rank))
        self.fast_bias = nn.Parameter(torch.zeros(()))

    @property
    def plastic_parameter_count(self) -> int:
        return self.fast_weight.numel() + self.fast_bias.numel()

    @property
    def slow_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.slow_parameters())

    def slow_parameters(self):
        for name, parameter in self.named_parameters():
            if name not in {"fast_weight", "fast_bias"}:
                yield parameter

    def freeze_slow(self) -> None:
        for parameter in self.slow_parameters():
            parameter.requires_grad_(False)
        self.fast_weight.requires_grad_(True)
        self.fast_bias.requires_grad_(True)

    def reset_fast(self) -> None:
        with torch.no_grad():
            self.fast_weight.zero_()
            self.fast_bias.zero_()

    def pair_features(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        reference = next(self.parameters())
        if query_features.ndim != 2 or query_features.shape[-1] != self.content_width:
            raise ValueError("query_features must be [batch, content_width]")
        if candidate_features.shape != (
            query_features.shape[0],
            candidate_features.shape[1],
            self.content_width,
        ):
            raise ValueError("candidate_features must be [batch, candidates, content_width]")
        if temporal_features.shape != (
            query_features.shape[0],
            candidate_features.shape[1],
            self.temporal_width,
        ):
            raise ValueError("temporal_features must be [batch, candidates, temporal_width]")
        if candidate_mask.dtype is not torch.bool or candidate_mask.shape != candidate_features.shape[:2]:
            raise ValueError("candidate_mask must be boolean [batch, candidates]")
        for name, value in (
            ("query_features", query_features),
            ("candidate_features", candidate_features),
            ("temporal_features", temporal_features),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match policy device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")
        query = query_features.detach()
        candidates = candidate_features.detach()
        temporal = temporal_features.detach()
        query_hidden = self.query_projection(query)
        candidate_hidden = self.candidate_projection(candidates)
        temporal_hidden = self.temporal_projection(temporal)
        expanded_query = query_hidden.unsqueeze(1).expand_as(candidate_hidden)
        return self.feature_network(
            torch.cat(
                (
                    candidate_hidden,
                    expanded_query,
                    candidate_hidden * expanded_query,
                    temporal_hidden,
                ),
                dim=-1,
            )
        )

    def forward_with_fast(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        *,
        fast_weight: torch.Tensor,
        fast_bias: torch.Tensor,
    ) -> SituatedFeedbackOutput:
        features = self.pair_features(
            query_features,
            candidate_features,
            temporal_features,
            candidate_mask,
        )
        if fast_weight.shape != (self.rank,) or fast_bias.shape != ():
            raise ValueError("functional fast state has the wrong shape")
        raw_residual = torch.einsum("bcr,r->bc", features, fast_weight) + fast_bias
        residuals = self.maximum_residual * torch.tanh(raw_residual)
        residuals = residuals.masked_fill(~candidate_mask, 0.0)
        normalized_base = _normalize_masked_scores(base_scores.detach(), candidate_mask)
        scores = (normalized_base + residuals).masked_fill(~candidate_mask, -torch.inf)
        return SituatedFeedbackOutput(
            weights=torch.softmax(scores, dim=-1),
            scores=scores,
            normalized_base_scores=normalized_base,
            residuals=residuals,
        )

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> SituatedFeedbackOutput:
        return self.forward_with_fast(
            query_features,
            candidate_features,
            temporal_features,
            candidate_mask,
            base_scores,
            fast_weight=self.fast_weight,
            fast_bias=self.fast_bias,
        )

    def capture_plastic_state(self) -> dict[str, torch.Tensor]:
        return {
            "fast_weight": self.fast_weight.detach().cpu().clone(),
            "fast_bias": self.fast_bias.detach().cpu().clone(),
        }

    def restore_plastic_state(self, state: Mapping[str, torch.Tensor]) -> None:
        if set(state) != {"fast_weight", "fast_bias"}:
            raise ValueError("plastic state must contain only fast_weight and fast_bias")
        if state["fast_weight"].shape != self.fast_weight.shape or state["fast_bias"].shape != self.fast_bias.shape:
            raise ValueError("plastic-state tensor shape does not match")
        if not all(torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("plastic state is not finite")
        with torch.no_grad():
            self.fast_weight.copy_(state["fast_weight"].to(self.fast_weight))
            self.fast_bias.copy_(state["fast_bias"].to(self.fast_bias))

    def plastic_state_digest(self) -> str:
        digest = hashlib.sha256(b"project-angler.oml-situated-fast-state.v1\x00")
        for name, value in sorted(self.capture_plastic_state().items()):
            digest.update(name.encode("utf-8") + b"\x00")
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return "sha256:" + digest.hexdigest()


def situated_outcome_loss(
    output: SituatedFeedbackOutput,
    selected_indices: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    residual_penalty: float = 1.0e-4,
) -> torch.Tensor:
    """Reinforce successes and suppress failures for actually selected evidence."""

    if selected_indices.dtype is not torch.long or selected_indices.shape != (output.weights.shape[0],):
        raise ValueError("selected_indices must be int64 [batch]")
    if outcomes.shape != selected_indices.shape:
        raise ValueError("outcomes must align with selected_indices")
    if outcomes.dtype not in (torch.float32, torch.float64):
        raise ValueError("outcomes must be floating point")
    if not bool(((outcomes == 1.0) | (outcomes == -1.0)).all().item()):
        raise ValueError("outcomes must contain only +1 success or -1 failure")
    if selected_indices.device != output.weights.device or outcomes.device != output.weights.device:
        raise ValueError("feedback tensors must match the policy device")
    selected = output.weights.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
    policy_loss = -(outcomes * torch.log(selected.clamp_min(1.0e-8))).mean()
    return policy_loss + float(residual_penalty) * output.residuals.square().mean()


def _normalize_masked_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    visible = scores.masked_fill(~mask, 0.0)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean = visible.sum(dim=1, keepdim=True) / count
    centered = (scores - mean).masked_fill(~mask, 0.0)
    variance = centered.square().sum(dim=1, keepdim=True) / count
    scale = variance.sqrt().clamp_min(1.0e-6)
    return (centered / scale).masked_fill(~mask, -torch.inf)


__all__ = [
    "OMLSituatedFeedbackPolicy",
    "SituatedFeedbackOutput",
    "SituatedFeedbackPolicy",
    "situated_outcome_loss",
]
