"""OML-ready natural-language procedure representations.

The slow representation learner (RLN) is a fresh
``NaturalLanguageTraceGraphEncoder`` followed by one small outcome trunk.  A
single bias-free prediction weight is retained only as the fixed starting
point for functional inner-loop adaptation; it is a buffer, never an
outer-owned parameter.  Outcome labels enter the balanced loss only and are
not inputs to the trace encoder.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .natural_trace_graph_causal_memory import NaturalLanguageTraceGraphEncoder


RELATIONAL_WIDTH = 32
FAST_HIDDEN_WIDTH = 64
EPISODES_PER_MECHANISM = 6


def _require_bool(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


def representation_metrics(codes: torch.Tensor) -> dict[str, Any]:
    """Return finite, detached collapse diagnostics for 32-D trace codes."""

    if (
        not isinstance(codes, torch.Tensor)
        or codes.ndim < 2
        or codes.shape[-1] != RELATIONAL_WIDTH
        or codes.numel() == 0
        or not codes.is_floating_point()
        or not bool(torch.isfinite(codes).all().item())
    ):
        raise ValueError("codes must be finite [...,32] floating tensors")
    flat = codes.detach().reshape(-1, RELATIONAL_WIDTH).to(dtype=torch.float64)
    if flat.shape[0] < 2:
        raise ValueError("representation metrics require at least two codes")

    centered = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    spectrum = singular.square()
    total = spectrum.sum()
    if float(total.item()) > 0.0:
        probabilities = spectrum / total
        positive = probabilities > 0
        effective_rank = torch.exp(
            -(probabilities[positive] * probabilities[positive].log()).sum()
        )
    else:
        effective_rank = total.new_zeros(())

    normalized = F.normalize(flat, dim=-1, eps=1.0e-12)
    similarities = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(
        flat.shape[0], device=flat.device, dtype=torch.bool
    )
    pairwise = similarities.masked_select(off_diagonal)
    variance = centered.square().mean(dim=0)
    return {
        "code_count": int(flat.shape[0]),
        "width": RELATIONAL_WIDTH,
        "effective_rank": float(effective_rank.item()),
        "mean_off_diagonal_cosine": float(pairwise.mean().item()),
        "fraction_distinct_pairs_above_0_999": float(
            (pairwise > 0.999).to(torch.float64).mean().item()
        ),
        "finite_nonzero_variance_dimensions": int((variance > 0).sum().item()),
        "minimum_dimension_variance": float(variance.min().item()),
        "maximum_dimension_variance": float(variance.max().item()),
        "all_finite": True,
    }


class OMLNaturalTraceRepresentation(nn.Module):
    """Fresh slow trace RLN plus one fixed functional-head initialization."""

    def __init__(
        self,
        *,
        step_width: int,
        relational_width: int = RELATIONAL_WIDTH,
        fast_hidden_width: int = FAST_HIDDEN_WIDTH,
    ) -> None:
        super().__init__()
        if type(step_width) is not int or step_width <= 0:
            raise ValueError("step_width must be a positive integer")
        if relational_width != RELATIONAL_WIDTH:
            raise ValueError("the V12 RLN identity requires relational_width=32")
        if fast_hidden_width != FAST_HIDDEN_WIDTH:
            raise ValueError("the V12 PLN identity requires fast_hidden_width=64")

        self.step_width = step_width
        self.relational_width = relational_width
        self.fast_hidden_width = fast_hidden_width
        self.trace_graph_encoder = NaturalLanguageTraceGraphEncoder(
            step_width=step_width,
            relational_width=relational_width,
        )
        self.outcome_trunk = nn.Sequential(
            nn.LayerNorm(relational_width),
            nn.Linear(relational_width, fast_hidden_width),
            nn.SiLU(),
        )

        # Match nn.Linear(H,1,bias=False) initialization without retaining a
        # trainable head module.  This is one fixed buffer and is cloned into
        # each functional inner trajectory.
        fast_initial_weight = torch.empty(1, fast_hidden_width)
        nn.init.kaiming_uniform_(fast_initial_weight, a=math.sqrt(5))
        self.register_buffer(
            "fast_initial_weight",
            fast_initial_weight,
            persistent=True,
        )

    def _validate_episode_tensors(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> None:
        if (
            step_features.ndim != 4
            or step_features.shape[1] != EPISODES_PER_MECHANISM
            or step_features.shape[-1] != self.step_width
        ):
            raise ValueError(
                "step_features must be [mechanisms,6,steps,step_width]"
            )
        if (
            step_mask.dtype is not torch.bool
            or step_mask.shape != step_features.shape[:-1]
            or step_mask.device != step_features.device
        ):
            raise ValueError("step_mask must be aligned boolean episode masks")

    @staticmethod
    def _validate_outcomes(outcomes: torch.Tensor, logits: torch.Tensor) -> None:
        if (
            outcomes.shape != logits.shape
            or outcomes.ndim != 2
            or outcomes.shape[1] != EPISODES_PER_MECHANISM
            or outcomes.device != logits.device
            or outcomes.dtype != logits.dtype
            or not bool(torch.isfinite(outcomes).all().item())
            or not bool(((outcomes == -1) | (outcomes == 1)).all().item())
        ):
            raise ValueError("outcomes must be aligned finite +/-1 tensors")
        positive = (outcomes == 1).sum(dim=1)
        negative = (outcomes == -1).sum(dim=1)
        if not bool(
            (
                (positive == EPISODES_PER_MECHANISM // 2)
                & (negative == EPISODES_PER_MECHANISM // 2)
            ).all().item()
        ):
            raise ValueError("every mechanism must contain three +/- outcomes")

    def encode(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> torch.Tensor:
        """Encode public traces without accepting outcomes or metadata."""

        _require_bool("include_direction", include_direction)
        _require_bool("include_step_semantics", include_step_semantics)
        self._validate_episode_tensors(step_features, step_mask)
        observed = (
            step_features
            if include_step_semantics
            else torch.zeros_like(step_features)
        )
        codes = self.trace_graph_encoder(
            observed,
            step_mask,
            include_direction=include_direction,
        )
        expected = (*step_features.shape[:2], self.relational_width)
        if codes.shape != expected or not bool(torch.isfinite(codes).all().item()):
            raise RuntimeError("the natural-trace RLN produced invalid codes")
        return codes

    def rln_features(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> torch.Tensor:
        codes = self.encode(
            step_features,
            step_mask,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        hidden = self.outcome_trunk(codes)
        if hidden.shape != (*codes.shape[:-1], self.fast_hidden_width):
            raise RuntimeError("the OML outcome trunk produced invalid features")
        if not bool(torch.isfinite(hidden).all().item()):
            raise RuntimeError("the OML outcome trunk produced non-finite features")
        return hidden

    def _validate_fast_weight(
        self,
        fast_weight: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if (
            fast_weight.shape != (1, self.fast_hidden_width)
            or fast_weight.device != reference.device
            or fast_weight.dtype != reference.dtype
            or not fast_weight.is_floating_point()
            or not bool(torch.isfinite(fast_weight).all().item())
        ):
            raise ValueError("fast_weight must be finite aligned [1,64]")

    def functional_logits(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        fast_weight: torch.Tensor | None = None,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> torch.Tensor:
        hidden = self.rln_features(
            step_features,
            step_mask,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        weight = self.fast_initial_weight if fast_weight is None else fast_weight
        self._validate_fast_weight(weight, hidden)
        logits = F.linear(hidden, weight).squeeze(-1)
        if logits.shape != step_features.shape[:2] or not bool(
            torch.isfinite(logits).all().item()
        ):
            raise RuntimeError("functional PLN logits are invalid")
        return logits

    def functional_loss(
        self,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        outcomes: torch.Tensor,
        fast_weight: torch.Tensor | None = None,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> torch.Tensor:
        logits = self.functional_logits(
            step_features,
            step_mask,
            fast_weight,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        self._validate_outcomes(outcomes, logits)
        loss = F.softplus(-outcomes * logits).mean()
        if loss.shape != () or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("balanced OML outcome loss is invalid")
        return loss

    def fresh_fast_weight(self) -> torch.Tensor:
        """Return one differentiable functional PLN start for an inner loop."""

        return self.fast_initial_weight.detach().clone().requires_grad_(True)

    def rln_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return the exhaustive outer-owned parameter partition."""

        return tuple(self.named_parameters())

    def parameter_partition_report(self) -> dict[str, Any]:
        named = self.rln_named_parameters()
        names = tuple(name for name, _ in named)
        if not names or any(
            not name.startswith(("trace_graph_encoder.", "outcome_trunk."))
            for name in names
        ):
            raise RuntimeError("RLN parameter ownership is not exhaustive")
        buffers = dict(self.named_buffers())
        if set(buffers) != {"fast_initial_weight"}:
            raise RuntimeError("the fixed fast initialization buffer changed")
        return {
            "rln_parameter_names": names,
            "rln_tensor_count": len(named),
            "rln_parameter_count": sum(value.numel() for _, value in named),
            "fast_initial_buffer_name": "fast_initial_weight",
            "fast_initial_shape": tuple(self.fast_initial_weight.shape),
            "fast_initial_count": self.fast_initial_weight.numel(),
            "fast_initial_outer_owned": False,
            "outcome_encoder_inputs": False,
        }

    @staticmethod
    def representation_metrics(codes: torch.Tensor) -> dict[str, Any]:
        return representation_metrics(codes)


__all__ = [
    "EPISODES_PER_MECHANISM",
    "FAST_HIDDEN_WIDTH",
    "OMLNaturalTraceRepresentation",
    "RELATIONAL_WIDTH",
    "representation_metrics",
]
