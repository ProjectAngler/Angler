"""Learned selection for a bounded, uniformly active plastic-state snapshot.

Semantic relevance is predicted from numeric representations and observed
contribution evidence.  Deterministic code is deliberately limited to tensor
validation, capacity accounting, and reproducible placement.  A decision is
held across requests until an explicit consolidation boundary; this module is
not a per-prompt adapter router.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_ACTIVE_BUDGET_BYTES = 1_073_741_824
_SHA256_REF = re.compile(r"sha256:[0-9a-f]{64}")


def _finite_tuple(value: Sequence[float], *, width: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _ref(value: str, label: str) -> str:
    if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 reference")
    return value


@dataclass(frozen=True, slots=True)
class PlasticExpert:
    """One immutable learned increment available to a working-set decision."""

    expert_ref: str
    adapter_ref: str
    resident_bytes: int
    rank: int
    representation: tuple[float, ...]
    mandatory: bool = False

    def validate(self, *, representation_width: int) -> "PlasticExpert":
        _ref(self.expert_ref, "expert_ref")
        _ref(self.adapter_ref, "adapter_ref")
        if type(self.resident_bytes) is not int or self.resident_bytes <= 0:
            raise ValueError("resident_bytes must be a positive integer")
        if type(self.rank) is not int or not 1 <= self.rank <= 4096:
            raise ValueError("rank must be 1 through 4096")
        _finite_tuple(
            self.representation,
            width=representation_width,
            label="expert representation",
        )
        if type(self.mandatory) is not bool:
            raise TypeError("mandatory must be bool")
        return self


@dataclass(frozen=True, slots=True)
class PlasticWorkingSetDecision:
    """One learned proposal with mechanically enforced resource placement."""

    decision_ref: str
    selected_expert_refs: tuple[str, ...]
    predicted_contributions: tuple[tuple[str, float], ...]
    active_bytes: int
    adapter_capacity_bytes: int
    reserved_bytes: int
    total_accounted_bytes: int
    total_budget_bytes: int
    novelty_probability: float


@dataclass(frozen=True, slots=True)
class PlasticityVramBudget:
    """Measured end-to-end VRAM envelope across all participating GPU ranks."""

    total_bytes: int = DEFAULT_ACTIVE_BUDGET_BYTES
    backend_pool_bytes: int = 0
    router_bytes: int = 0
    communication_bytes: int = 0
    staging_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total_bytes",
            "backend_pool_bytes",
            "router_bytes",
            "communication_bytes",
            "staging_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if self.reserved_bytes >= self.total_bytes:
            raise ValueError("plasticity VRAM reserves leave no adapter capacity")

    @property
    def reserved_bytes(self) -> int:
        return (
            self.backend_pool_bytes
            + self.router_bytes
            + self.communication_bytes
            + self.staging_bytes
        )

    @property
    def adapter_capacity_bytes(self) -> int:
        return self.total_bytes - self.reserved_bytes


class LearnedPlasticWorkingSetCoordinator(nn.Module):
    """Predict increment contribution without text labels or semantic rules."""

    def __init__(
        self,
        *,
        context_width: int,
        expert_width: int,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        for name, value in (
            ("context_width", context_width),
            ("expert_width", expert_width),
            ("hidden_width", hidden_width),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.context_width = context_width
        self.expert_width = expert_width
        self.hidden_width = hidden_width
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, hidden_width),
            nn.SiLU(),
        )
        self.expert_encoder = nn.Sequential(
            nn.LayerNorm(expert_width),
            nn.Linear(expert_width, hidden_width),
            nn.SiLU(),
        )
        self.contribution_head = nn.Sequential(
            nn.Linear(hidden_width * 3, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        self.novelty_head = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )

    def _validate_inputs(
        self, context: torch.Tensor, expert_representations: torch.Tensor
    ) -> None:
        if (
            not isinstance(context, torch.Tensor)
            or context.ndim != 2
            or context.shape[-1] != self.context_width
            or not context.is_floating_point()
            or not bool(torch.isfinite(context).all().item())
        ):
            raise ValueError("context must be finite floating [batch,context_width]")
        if (
            not isinstance(expert_representations, torch.Tensor)
            or expert_representations.ndim != 3
            or expert_representations.shape[0] != context.shape[0]
            or expert_representations.shape[-1] != self.expert_width
            or expert_representations.shape[1] < 1
            or expert_representations.device != context.device
            or expert_representations.dtype != context.dtype
            or not bool(torch.isfinite(expert_representations).all().item())
        ):
            raise ValueError(
                "expert representations must be finite aligned "
                "[batch,experts,expert_width]"
            )

    def forward(
        self, context: torch.Tensor, expert_representations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(context, expert_representations)
        context_hidden = self.context_encoder(context)
        expert_hidden = self.expert_encoder(expert_representations)
        expanded = context_hidden.unsqueeze(1).expand_as(expert_hidden)
        joint = torch.cat(
            (expanded, expert_hidden, expanded * expert_hidden), dim=-1
        )
        contributions = self.contribution_head(joint).squeeze(-1)
        novelty_logits = self.novelty_head(context_hidden).squeeze(-1)
        if not bool(torch.isfinite(contributions).all().item()) or not bool(
            torch.isfinite(novelty_logits).all().item()
        ):
            raise RuntimeError("plastic coordinator produced non-finite outputs")
        return contributions, novelty_logits

    def learning_loss(
        self,
        context: torch.Tensor,
        expert_representations: torch.Tensor,
        observed_contributions: torch.Tensor,
        novelty_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Learn from measured counterfactual contribution and novelty targets."""

        predicted, novelty_logits = self(context, expert_representations)
        if (
            observed_contributions.shape != predicted.shape
            or observed_contributions.device != predicted.device
            or observed_contributions.dtype != predicted.dtype
            or not bool(torch.isfinite(observed_contributions).all().item())
        ):
            raise ValueError("observed contributions must align with predictions")
        if (
            novelty_targets.shape != novelty_logits.shape
            or novelty_targets.device != novelty_logits.device
            or novelty_targets.dtype != novelty_logits.dtype
            or not bool(torch.isfinite(novelty_targets).all().item())
            or not bool(((novelty_targets >= 0) & (novelty_targets <= 1)).all().item())
        ):
            raise ValueError("novelty targets must be aligned probabilities")
        contribution_loss = F.smooth_l1_loss(predicted, observed_contributions)
        novelty_loss = F.binary_cross_entropy_with_logits(
            novelty_logits, novelty_targets
        )
        result = contribution_loss + novelty_loss
        if result.shape != () or not bool(torch.isfinite(result).item()):
            raise RuntimeError("plastic coordinator loss is invalid")
        return result

    @torch.inference_mode()
    def choose_working_set(
        self,
        *,
        context: Sequence[float],
        experts: Sequence[PlasticExpert],
        vram_budget: PlasticityVramBudget | None = None,
    ) -> PlasticWorkingSetDecision:
        """Score semantically opaque experts, then enforce an exact byte ceiling."""

        budget = PlasticityVramBudget() if vram_budget is None else vram_budget
        if not isinstance(budget, PlasticityVramBudget):
            raise TypeError("vram_budget must be PlasticityVramBudget")
        budget_bytes = budget.adapter_capacity_bytes
        if isinstance(experts, (str, bytes)) or not experts:
            raise ValueError("at least one expert is required")
        checked = tuple(
            expert.validate(representation_width=self.expert_width)
            for expert in experts
        )
        refs = [expert.expert_ref for expert in checked]
        if len(set(refs)) != len(refs):
            raise ValueError("expert references must be unique")
        adapters = [expert.adapter_ref for expert in checked]
        if len(set(adapters)) != len(adapters):
            raise ValueError("adapter references must be unique")
        context_values = _finite_tuple(
            context, width=self.context_width, label="context representation"
        )
        parameter = next(self.parameters())
        context_tensor = torch.tensor(
            [context_values], dtype=parameter.dtype, device=parameter.device
        )
        expert_tensor = torch.tensor(
            [[expert.representation for expert in checked]],
            dtype=parameter.dtype,
            device=parameter.device,
        )
        contributions, novelty_logits = self(context_tensor, expert_tensor)
        scores = [float(value) for value in contributions[0].cpu()]
        mandatory = [index for index, expert in enumerate(checked) if expert.mandatory]
        mandatory_bytes = sum(checked[index].resident_bytes for index in mandatory)
        if mandatory_bytes > budget_bytes:
            raise ValueError("mandatory plastic state exceeds the active budget")

        selected = set(mandatory)
        active_bytes = mandatory_bytes
        # Scores carry learned semantic judgment.  This ordering performs only
        # deterministic physical placement and never sees text or task labels.
        optional = sorted(
            (index for index in range(len(checked)) if index not in selected),
            key=lambda index: (-scores[index], checked[index].expert_ref),
        )
        for index in optional:
            expert = checked[index]
            if scores[index] <= 0.0:
                continue
            if active_bytes + expert.resident_bytes <= budget_bytes:
                selected.add(index)
                active_bytes += expert.resident_bytes

        selected_refs = tuple(
            expert.expert_ref
            for index, expert in enumerate(checked)
            if index in selected
        )
        predicted = tuple(
            (expert.expert_ref, scores[index])
            for index, expert in enumerate(checked)
        )
        unsigned = {
            "schema": "jenny2.plastic-working-set-decision.v1",
            "selected_expert_refs": list(selected_refs),
            "predicted_contributions": [
                [ref, value.hex()] for ref, value in predicted
            ],
            "active_bytes": active_bytes,
            "adapter_capacity_bytes": budget_bytes,
            "reserved_bytes": budget.reserved_bytes,
            "total_accounted_bytes": active_bytes + budget.reserved_bytes,
            "total_budget_bytes": budget.total_bytes,
            "novelty_probability": float(torch.sigmoid(novelty_logits[0]).cpu()).hex(),
        }
        decision_ref = "sha256:" + hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return PlasticWorkingSetDecision(
            decision_ref=decision_ref,
            selected_expert_refs=selected_refs,
            predicted_contributions=predicted,
            active_bytes=active_bytes,
            adapter_capacity_bytes=budget_bytes,
            reserved_bytes=budget.reserved_bytes,
            total_accounted_bytes=active_bytes + budget.reserved_bytes,
            total_budget_bytes=budget.total_bytes,
            novelty_probability=float(torch.sigmoid(novelty_logits[0]).cpu()),
        )


__all__ = [
    "DEFAULT_ACTIVE_BUDGET_BYTES",
    "LearnedPlasticWorkingSetCoordinator",
    "PlasticExpert",
    "PlasticityVramBudget",
    "PlasticWorkingSetDecision",
]
