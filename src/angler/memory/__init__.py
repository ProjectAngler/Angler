"""Temporally situated external-memory interfaces for Project Angler."""

from .cognee_adapter import CogneeConfigurationError, CogneeProjectionBackend
from .contracts import (
    MemoryHit,
    MemoryProjection,
    ProjectionBackend,
    RecallBatch,
    SituatedRecall,
)
from .moving_origin import (
    AFTER,
    AT,
    BEFORE,
    FairNaiveTemporalView,
    Landmark,
    MovingOriginIndex,
    RecentResult,
    TemporalAnchor,
    TemporalPosition,
)
from .situated import SituatedMemory, decode_projection, encode_projection, projection_id

__all__ = [
    "AFTER",
    "AT",
    "BEFORE",
    "CogneeConfigurationError",
    "CogneeProjectionBackend",
    "FairNaiveTemporalView",
    "Landmark",
    "MemoryHit",
    "MemoryProjection",
    "MovingOriginIndex",
    "ProjectionBackend",
    "RecallBatch",
    "RecentResult",
    "SituatedMemory",
    "SituatedRecall",
    "TemporalAnchor",
    "TemporalPosition",
    "decode_projection",
    "encode_projection",
    "projection_id",
]
