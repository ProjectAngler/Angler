"""Learned projection of frozen donor correspondence into Angler actions."""

from __future__ import annotations

import torch
from torch import nn


class DonorCandidateFusion(nn.Module):
    """Fuse public candidate correspondence without selecting a procedure.

    The donor tensor is an ordinary learned feature boundary.  This module has
    no candidate labels, task identities, thresholds, search, or ordering
    rule; the downstream procedural decoder remains responsible for behavior.
    """

    def __init__(
        self,
        *,
        semantic_width: int,
        donor_width: int,
        hidden_width: int = 512,
    ) -> None:
        super().__init__()
        for value, name in (
            (semantic_width, "semantic_width"),
            (donor_width, "donor_width"),
            (hidden_width, "hidden_width"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.semantic_width = semantic_width
        self.donor_width = donor_width
        self.donor_projection = nn.Sequential(
            nn.LayerNorm(donor_width),
            nn.Linear(donor_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, semantic_width),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(2 * semantic_width),
            nn.Linear(2 * semantic_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, semantic_width, bias=False),
        )
        self.logit_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        semantic: torch.Tensor,
        donor: torch.Tensor,
        *,
        include_donor: bool = True,
    ) -> torch.Tensor:
        if (
            semantic.ndim != 3
            or donor.ndim != 3
            or semantic.shape[:2] != donor.shape[:2]
            or semantic.shape[-1] != self.semantic_width
            or donor.shape[-1] != self.donor_width
            or semantic.device != donor.device
            or semantic.dtype != donor.dtype
            or not semantic.is_floating_point()
            or not bool(torch.isfinite(semantic).all().item())
            or not bool(torch.isfinite(donor).all().item())
        ):
            raise ValueError("semantic and donor candidate tensors are invalid")
        if type(include_donor) is not bool:
            raise TypeError("include_donor must be bool")
        if not include_donor:
            return semantic
        projected = self.donor_projection(donor)
        update = self.residual(torch.cat((semantic, projected), dim=-1))
        return semantic + torch.sigmoid(self.logit_gate) * update

