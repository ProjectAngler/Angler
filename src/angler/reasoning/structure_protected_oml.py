"""Structure-protected OML over paired public procedure traces.

One shared natural-trace encoder maps both the reference procedure and an
attempt into the same relational space.  A learned, role-sensitive trunk then
forms a generic relation representation for a functional fast outcome head.
Outcome labels are used only by the balanced loss; they never enter either
trace encoder call.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .natural_trace_graph_causal_memory import NaturalLanguageTraceGraphEncoder


RELATIONAL_WIDTH = 32
RELATION_INPUT_WIDTH = RELATIONAL_WIDTH * 4
FAST_HIDDEN_WIDTH = 64
EPISODES_PER_MECHANISM = 6


def _require_bool(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


class StructureProtectedOMLCore(nn.Module):
    """Shared trace encoder, learned pair trunk, and fixed fast-head start."""

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
            raise ValueError("the V13 encoder identity requires relational_width=32")
        if fast_hidden_width != FAST_HIDDEN_WIDTH:
            raise ValueError("the V13 pair trunk requires fast_hidden_width=64")

        self.step_width = step_width
        self.relational_width = relational_width
        self.fast_hidden_width = fast_hidden_width
        self.trace_graph_encoder = NaturalLanguageTraceGraphEncoder(
            step_width=step_width,
            relational_width=relational_width,
        )
        self.pair_trunk = nn.Sequential(
            nn.LayerNorm(RELATION_INPUT_WIDTH),
            nn.Linear(RELATION_INPUT_WIDTH, fast_hidden_width),
            nn.SiLU(),
        )

        fast_initial_weight = torch.empty(1, fast_hidden_width)
        nn.init.kaiming_uniform_(fast_initial_weight, a=math.sqrt(5))
        self.register_buffer(
            "fast_initial_weight",
            fast_initial_weight,
            persistent=True,
        )

    def _validate_trace(
        self,
        name: str,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> None:
        if (
            not isinstance(step_features, torch.Tensor)
            or step_features.ndim != 4
            or step_features.shape[1] != EPISODES_PER_MECHANISM
            or step_features.shape[-1] != self.step_width
        ):
            raise ValueError(
                f"{name}_step_features must be [mechanisms,6,steps,step_width]"
            )
        if (
            not isinstance(step_mask, torch.Tensor)
            or step_mask.dtype is not torch.bool
            or step_mask.shape != step_features.shape[:-1]
            or step_mask.device != step_features.device
        ):
            raise ValueError(f"{name}_step_mask must be an aligned boolean mask")

    def _encode_trace(
        self,
        name: str,
        step_features: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        include_direction: bool,
        include_step_semantics: bool,
    ) -> torch.Tensor:
        self._validate_trace(name, step_features, step_mask)
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
            raise RuntimeError(f"the shared encoder produced invalid {name} codes")
        return codes

    def encode_views(
        self,
        reference_step_features: torch.Tensor,
        reference_step_mask: torch.Tensor,
        attempt_step_features: torch.Tensor,
        attempt_step_mask: torch.Tensor,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw shared-backbone codes for direct normalized InfoNCE."""

        _require_bool("include_direction", include_direction)
        _require_bool("include_step_semantics", include_step_semantics)
        if reference_step_features.shape[:2] != attempt_step_features.shape[:2]:
            raise ValueError("reference and attempt mechanism/episode axes must align")
        reference_codes = self._encode_trace(
            "reference",
            reference_step_features,
            reference_step_mask,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        attempt_codes = self._encode_trace(
            "attempt",
            attempt_step_features,
            attempt_step_mask,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        return reference_codes, attempt_codes

    def relation_features(
        self,
        reference_step_features: torch.Tensor,
        reference_step_mask: torch.Tensor,
        attempt_step_features: torch.Tensor,
        attempt_step_mask: torch.Tensor,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> torch.Tensor:
        reference, attempt = self.encode_views(
            reference_step_features,
            reference_step_mask,
            attempt_step_features,
            attempt_step_mask,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        relation = torch.cat(
            (reference, attempt, (reference - attempt).abs(), reference * attempt),
            dim=-1,
        )
        hidden = self.pair_trunk(relation)
        expected = (*reference.shape[:-1], self.fast_hidden_width)
        if hidden.shape != expected or not bool(torch.isfinite(hidden).all().item()):
            raise RuntimeError("the learned pair trunk produced invalid features")
        return hidden

    def _validate_fast_weight(
        self,
        fast_weight: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if (
            not isinstance(fast_weight, torch.Tensor)
            or fast_weight.shape != (1, self.fast_hidden_width)
            or fast_weight.device != reference.device
            or fast_weight.dtype != reference.dtype
            or not fast_weight.is_floating_point()
            or not bool(torch.isfinite(fast_weight).all().item())
        ):
            raise ValueError("fast_weight must be finite aligned [1,64]")

    @staticmethod
    def _validate_outcomes(outcomes: torch.Tensor, logits: torch.Tensor) -> None:
        if (
            not isinstance(outcomes, torch.Tensor)
            or outcomes.shape != logits.shape
            or outcomes.ndim != 2
            or outcomes.shape[1] != EPISODES_PER_MECHANISM
            or outcomes.device != logits.device
            or outcomes.dtype != logits.dtype
            or not bool(torch.isfinite(outcomes).all().item())
            or not bool(((outcomes == -1) | (outcomes == 1)).all().item())
        ):
            raise ValueError("outcomes must be aligned finite +/-1 tensors")
        positives = (outcomes == 1).sum(dim=1)
        negatives = (outcomes == -1).sum(dim=1)
        expected = EPISODES_PER_MECHANISM // 2
        if not bool(((positives == expected) & (negatives == expected)).all().item()):
            raise ValueError("every mechanism must contain three +/- outcomes")

    def functional_logits(
        self,
        reference_step_features: torch.Tensor,
        reference_step_mask: torch.Tensor,
        attempt_step_features: torch.Tensor,
        attempt_step_mask: torch.Tensor,
        fast_weight: torch.Tensor | None = None,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> torch.Tensor:
        hidden = self.relation_features(
            reference_step_features,
            reference_step_mask,
            attempt_step_features,
            attempt_step_mask,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        weight = self.fast_initial_weight if fast_weight is None else fast_weight
        self._validate_fast_weight(weight, hidden)
        logits = F.linear(hidden, weight).squeeze(-1)
        if logits.shape != reference_step_features.shape[:2] or not bool(
            torch.isfinite(logits).all().item()
        ):
            raise RuntimeError("functional PLN logits are invalid")
        return logits

    def functional_loss(
        self,
        reference_step_features: torch.Tensor,
        reference_step_mask: torch.Tensor,
        attempt_step_features: torch.Tensor,
        attempt_step_mask: torch.Tensor,
        outcomes: torch.Tensor,
        fast_weight: torch.Tensor | None = None,
        *,
        include_direction: bool = True,
        include_step_semantics: bool = True,
    ) -> torch.Tensor:
        logits = self.functional_logits(
            reference_step_features,
            reference_step_mask,
            attempt_step_features,
            attempt_step_mask,
            fast_weight,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
        self._validate_outcomes(outcomes, logits)
        loss = F.softplus(-outcomes * logits).mean()
        if loss.shape != () or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("balanced paired OML outcome loss is invalid")
        return loss

    def fresh_fast_weight(self) -> torch.Tensor:
        """Return a differentiable clone for one functional trajectory."""

        return self.fast_initial_weight.detach().clone().requires_grad_(True)

    def encoder_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (f"trace_graph_encoder.{name}", parameter)
            for name, parameter in self.trace_graph_encoder.named_parameters()
        )

    def trunk_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (f"pair_trunk.{name}", parameter)
            for name, parameter in self.pair_trunk.named_parameters()
        )

    def parameter_partition_report(self) -> dict[str, Any]:
        encoder = self.encoder_named_parameters()
        trunk = self.trunk_named_parameters()
        all_named = tuple(self.named_parameters())
        owned_names = tuple(name for name, _ in (*encoder, *trunk))
        if (
            not encoder
            or not trunk
            or owned_names != tuple(name for name, _ in all_named)
            or len({id(value) for _, value in (*encoder, *trunk)})
            != len((*encoder, *trunk))
        ):
            raise RuntimeError("V13 encoder/trunk ownership is not exhaustive")
        if set(dict(self.named_buffers())) != {"fast_initial_weight"}:
            raise RuntimeError("the fixed V13 fast initialization buffer changed")
        return {
            "encoder_parameter_names": tuple(name for name, _ in encoder),
            "encoder_tensor_count": len(encoder),
            "encoder_parameter_count": sum(value.numel() for _, value in encoder),
            "trunk_parameter_names": tuple(name for name, _ in trunk),
            "trunk_tensor_count": len(trunk),
            "trunk_parameter_count": sum(value.numel() for _, value in trunk),
            "partitions_disjoint": True,
            "all_trainable_parameters_owned": True,
            "fast_initial_buffer_name": "fast_initial_weight",
            "fast_initial_shape": tuple(self.fast_initial_weight.shape),
            "fast_initial_outer_owned": False,
            "outcome_encoder_inputs": False,
            "metadata_encoder_inputs": False,
        }


__all__ = [
    "EPISODES_PER_MECHANISM",
    "FAST_HIDDEN_WIDTH",
    "RELATIONAL_WIDTH",
    "RELATION_INPUT_WIDTH",
    "StructureProtectedOMLCore",
]
