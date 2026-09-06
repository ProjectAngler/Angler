"""Outcome-blind geometry for grouped public relation representations.

The learned boundary is intentionally narrow: it accepts only already encoded
public ``(reference, attempt)`` relation vectors.  Relation names and all task,
outcome, temporal, memory, and evaluator metadata stay outside the module.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RELATION_WIDTH = 64
HIDDEN_WIDTH = 128
TEMPERATURE = 0.10
NORMALIZATION_EPSILON = 1.0e-8


def _validate_relation_batch(name: str, values: torch.Tensor) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.dtype is not torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if values.ndim != 2 or values.shape[1] != RELATION_WIDTH:
        raise ValueError(f"{name} must have shape [N,{RELATION_WIDTH}]")
    if values.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one row")
    if not values.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_normalized_codes(name: str, values: torch.Tensor) -> None:
    _validate_relation_batch(name, values)
    norms = torch.linalg.vector_norm(values, dim=-1)
    nonzero = norms > NORMALIZATION_EPSILON
    if bool(nonzero.any().item()):
        expected = torch.ones_like(norms[nonzero])
        if not bool(torch.allclose(norms[nonzero], expected, rtol=1.0e-5, atol=1.0e-6)):
            raise ValueError(f"{name} nonzero rows must be L2 normalized")


class GroupedPublicRelationGeometry(nn.Module):
    """Map frozen public relation rows into a shared normalized geometry."""

    def __init__(self) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(RELATION_WIDTH)
        self.hidden_projection = nn.Linear(
            RELATION_WIDTH,
            HIDDEN_WIDTH,
            bias=False,
        )
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(
            HIDDEN_WIDTH,
            RELATION_WIDTH,
            bias=False,
        )

    def forward(self, relation_features: torch.Tensor) -> torch.Tensor:
        """Return one outcome-blind normalized code per public relation row."""

        _validate_relation_batch("relation_features", relation_features)
        parameter = next(self.parameters())
        if relation_features.device != parameter.device:
            raise ValueError("relation_features must be on the module device")
        hidden = self.input_normalization(relation_features)
        hidden = self.hidden_projection(hidden)
        hidden = self.activation(hidden)
        raw_codes = self.output_projection(hidden)
        codes = F.normalize(
            raw_codes,
            p=2.0,
            dim=-1,
            eps=NORMALIZATION_EPSILON,
        )
        if codes.shape != relation_features.shape or not bool(
            torch.isfinite(codes).all().item()
        ):
            raise RuntimeError("public relation geometry produced invalid codes")
        return codes


def supervised_contrastive_loss(
    normalized_codes: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen V20 supervised-contrastive objective.

    ``positive_mask`` is a precomputed equivalence mask, not a relation-label
    vector.  It is consumed only by this pure loss helper and never by the
    learned representation.
    """

    _validate_normalized_codes("normalized_codes", normalized_codes)
    row_count = normalized_codes.shape[0]
    if not isinstance(positive_mask, torch.Tensor):
        raise TypeError("positive_mask must be a torch.Tensor")
    if positive_mask.dtype is not torch.bool:
        raise TypeError("positive_mask must have dtype torch.bool")
    if positive_mask.shape != (row_count, row_count):
        raise ValueError("positive_mask must have shape [N,N]")
    if positive_mask.device != normalized_codes.device:
        raise ValueError("positive_mask must be on the code device")
    if bool(torch.diagonal(positive_mask).any().item()):
        raise ValueError("positive_mask diagonal must be false")
    if not torch.equal(positive_mask, positive_mask.T):
        raise ValueError("positive_mask must be symmetric")
    positive_counts = positive_mask.sum(dim=1)
    if bool((positive_counts <= 0).any().item()):
        raise ValueError("every anchor must have at least one positive")

    logits = torch.matmul(normalized_codes, normalized_codes.T) / TEMPERATURE
    # Sort each mathematically unordered set before reducing it.  This keeps
    # the float32 objective invariant to a joint row/mask permutation without
    # changing the supervised-contrastive formula.
    per_anchor_values = []
    for anchor in range(row_count):
        denominator = torch.cat(
            (logits[anchor, :anchor], logits[anchor, anchor + 1 :])
        ).sort().values
        positives = logits[anchor, positive_mask[anchor]].sort().values
        per_anchor_values.append(
            torch.logsumexp(denominator, dim=0) - positives.mean()
        )
    per_anchor = torch.stack(per_anchor_values).sort().values
    loss = per_anchor.mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("supervised contrastive loss is invalid")
    return loss


def cosine_similarity_logits(
    normalized_queries: torch.Tensor,
    normalized_gallery: torch.Tensor,
) -> torch.Tensor:
    """Return fixed-temperature cosine logits for metadata-free evaluation."""

    _validate_normalized_codes("normalized_queries", normalized_queries)
    _validate_normalized_codes("normalized_gallery", normalized_gallery)
    if normalized_queries.device != normalized_gallery.device:
        raise ValueError("queries and gallery must be on the same device")
    logits = torch.matmul(normalized_queries, normalized_gallery.T) / TEMPERATURE
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("cosine similarity logits are non-finite")
    return logits


__all__ = [
    "GroupedPublicRelationGeometry",
    "HIDDEN_WIDTH",
    "NORMALIZATION_EPSILON",
    "RELATION_WIDTH",
    "TEMPERATURE",
    "cosine_similarity_logits",
    "supervised_contrastive_loss",
]
