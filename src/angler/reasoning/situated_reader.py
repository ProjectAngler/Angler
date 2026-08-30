"""Learned reading of semantically retrieved, temporally situated evidence.

The reader receives detached representations and trusted Moving Origin
coordinates.  It contains no recency rule, procedure table, task identifier,
or answer lookup: outcome training must teach it which evidence matters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
from torch import nn

from angler.memory import RecallBatch, SituatedRecall


@dataclass(frozen=True, slots=True)
class SituatedFeatureSpec:
    """Stable tensor layout for Moving Origin and retrieval coordinates."""

    landmarks: tuple[str, ...] = ()
    include_acquired_ordinal: bool = True

    def __post_init__(self) -> None:
        if type(self.landmarks) is not tuple:
            raise TypeError("landmarks must be a tuple")
        if any(type(item) is not str or not item for item in self.landmarks):
            raise ValueError("landmark names must be non-empty strings")
        if len(set(self.landmarks)) != len(self.landmarks):
            raise ValueError("landmark names must be unique")
        if type(self.include_acquired_ordinal) is not bool:
            raise TypeError("include_acquired_ordinal must be boolean")

    @property
    def width(self) -> int:
        # age, acquired position, log age, backend score/missing, world-valid
        # tri-state, then BEFORE/AT/AFTER/UNKNOWN for each declared landmark.
        return 7 + int(self.include_acquired_ordinal) + 4 * len(self.landmarks)


@dataclass(frozen=True, slots=True)
class SituatedReaderOutput:
    """Action evidence plus an inspectable attribution over candidates."""

    logits: torch.Tensor
    context: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor


def encode_situated_features(
    recalls: Sequence[RecallBatch],
    *,
    now: Sequence[int],
    spec: SituatedFeatureSpec,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorize coordinates without prescribing how the learner uses them."""

    if not recalls:
        raise ValueError("at least one recall batch is required")
    if len(recalls) != len(now):
        raise ValueError("one current origin is required per recall batch")
    if any(type(value) is not int or value < 0 for value in now):
        raise ValueError("current origins must be non-negative integers")
    candidate_count = max(len(batch.items) for batch in recalls)
    if candidate_count == 0:
        raise ValueError("each reader batch collectively needs a candidate")

    features = torch.zeros(
        (len(recalls), candidate_count, spec.width),
        device=device,
        dtype=dtype,
    )
    mask = torch.zeros(
        (len(recalls), candidate_count),
        device=device,
        dtype=torch.bool,
    )
    for batch_index, (batch, current) in enumerate(zip(recalls, now, strict=True)):
        horizon = max(current + 1, 1)
        log_horizon = math.log1p(horizon)
        for candidate_index, item in enumerate(batch.items):
            values = _feature_row(item, horizon=horizon, log_horizon=log_horizon, spec=spec)
            features[batch_index, candidate_index] = torch.tensor(
                values,
                device=features.device,
                dtype=dtype,
            )
            mask[batch_index, candidate_index] = True
    return features, mask


def _feature_row(
    item: SituatedRecall,
    *,
    horizon: int,
    log_horizon: float,
    spec: SituatedFeatureSpec,
) -> list[float]:
    score_missing = item.backend_score is None
    score = 0.0 if score_missing else math.tanh(float(item.backend_score))
    world_true = float(item.world_valid_at_query is True)
    world_false = float(item.world_valid_at_query is False)
    world_unknown = float(item.world_valid_at_query is None)
    values = [
        item.age / horizon,
        math.log1p(item.age) / log_horizon,
        score,
        float(score_missing),
        world_true,
        world_false,
        world_unknown,
    ]
    if spec.include_acquired_ordinal:
        values.insert(1, item.acquired_ordinal / horizon)
    relations = dict(item.landmark_relations)
    for name in spec.landmarks:
        relation = relations.get(name)
        values.extend(
            (
                float(relation == "BEFORE"),
                float(relation == "AT"),
                float(relation == "AFTER"),
                float(relation not in {"BEFORE", "AT", "AFTER"}),
            )
        )
    return values


class LearnedSituatedMemoryReader(nn.Module):
    """Outcome-trained attention over external evidence candidates."""

    def __init__(
        self,
        *,
        content_width: int,
        temporal_width: int,
        hidden_width: int,
        action_count: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("content_width", content_width),
            ("temporal_width", temporal_width),
            ("hidden_width", hidden_width),
            ("action_count", action_count),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.content_width = content_width
        self.temporal_width = temporal_width
        self.hidden_width = hidden_width
        self.action_count = action_count

        self.content_projection = nn.Sequential(
            nn.LayerNorm(content_width),
            nn.Linear(content_width, hidden_width),
            nn.SiLU(),
        )
        self.query_projection = nn.Sequential(
            nn.LayerNorm(content_width),
            nn.Linear(content_width, hidden_width),
            nn.SiLU(),
        )
        self.temporal_projection = nn.Sequential(
            nn.LayerNorm(temporal_width),
            nn.Linear(temporal_width, hidden_width),
            nn.SiLU(),
        )
        self.score_network = nn.Sequential(
            nn.Linear(hidden_width * 4, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        self.value_network = nn.Sequential(
            nn.Linear(hidden_width * 2, hidden_width),
            nn.SiLU(),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_width * 3),
            nn.Linear(hidden_width * 3, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, action_count),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        temporal_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> SituatedReaderOutput:
        self._validate_inputs(
            query_features,
            candidate_features,
            temporal_features,
            candidate_mask,
        )
        content = self.content_projection(candidate_features)
        query = self.query_projection(query_features)
        temporal = self.temporal_projection(temporal_features)
        expanded_query = query.unsqueeze(1).expand_as(content)
        score_inputs = torch.cat(
            (content, expanded_query, content * expanded_query, temporal),
            dim=-1,
        )
        scores = self.score_network(score_inputs).squeeze(-1)
        scores = scores.masked_fill(~candidate_mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        values = self.value_network(torch.cat((content, temporal), dim=-1))
        context = torch.sum(weights.unsqueeze(-1) * values, dim=1)
        logits = self.action_head(
            torch.cat((query, context, query * context), dim=-1)
        )
        return SituatedReaderOutput(
            logits=logits,
            context=context,
            weights=weights,
            scores=scores,
        )

    def _validate_inputs(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        temporal: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if query.ndim != 2 or query.shape[-1] != self.content_width:
            raise ValueError("query_features must have shape [batch, content_width]")
        expected_candidates = (query.shape[0], candidates.shape[1], self.content_width)
        if candidates.ndim != 3 or candidates.shape != expected_candidates:
            raise ValueError(
                "candidate_features must have shape [batch, candidates, content_width]"
            )
        expected_temporal = (query.shape[0], candidates.shape[1], self.temporal_width)
        if temporal.shape != expected_temporal:
            raise ValueError(
                "temporal_features must have shape [batch, candidates, temporal_width]"
            )
        if mask.dtype is not torch.bool or mask.shape != candidates.shape[:2]:
            raise ValueError("candidate_mask must be boolean [batch, candidates]")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("every batch row needs at least one visible candidate")
        reference = next(self.parameters())
        for name, value in (
            ("query_features", query),
            ("candidate_features", candidates),
            ("temporal_features", temporal),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match the reader device and dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")
        if mask.device != reference.device:
            raise ValueError("candidate_mask must match the reader device")


__all__ = [
    "LearnedSituatedMemoryReader",
    "SituatedFeatureSpec",
    "SituatedReaderOutput",
    "encode_situated_features",
]
