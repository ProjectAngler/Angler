"""Temporally situated external-memory interfaces for Project Angler."""

from .cognee_adapter import CogneeConfigurationError, CogneeProjectionBackend
from .cognitive_graph import (
    CanonicalEpisodeSource,
    CognitiveGraphProjection,
    CognitiveReferenceBackend,
    CognitiveReferenceHit,
    TypedCognitiveRecall,
    TypedCognitiveRecallBatch,
    TypedProjectionReceipt,
    TypedSituatedMemory,
    episode_record,
)
from .cognee_cognitive_adapter import (
    CogneeCognitiveAdapter,
    CogneeCognitiveConfigurationError,
    CogneeStructuredBindings,
)
from .cognitive_acquisition import CognitiveAcquisition, CognitiveGraphProjectionV2
from .cognitive_acquisition_graph import (
    AcquisitionProjectionReceipt,
    AcquisitionRecall,
    AcquisitionRecallBatch,
    AcquisitionReferenceBackend,
    AcquisitionReferenceHit,
    AcquisitionSituatedMemory,
    CanonicalAcquisitionSource,
    prospective_batch_record,
    prospective_resolution_record,
)
from .cognee_acquisition_adapter import (
    CogneeAcquisitionAdapter,
    CogneeAcquisitionBindings,
    CogneeAcquisitionConfigurationError,
)
from .cognee_subprocess_bindings import (
    CogneeSubprocessBindings,
    cognee_worker_launch_argv,
    cognee_worker_launch_cwd,
)
from .cognee_worker_protocol import (
    CogneeWorkerScope,
    DEFAULT_COGNEE_WORKER_SCOPE,
)
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
    "CogneeCognitiveAdapter",
    "CogneeCognitiveConfigurationError",
    "CogneeStructuredBindings",
    "CanonicalEpisodeSource",
    "CanonicalAcquisitionSource",
    "AcquisitionProjectionReceipt",
    "AcquisitionRecall",
    "AcquisitionRecallBatch",
    "AcquisitionReferenceBackend",
    "AcquisitionReferenceHit",
    "AcquisitionSituatedMemory",
    "CogneeAcquisitionAdapter",
    "CogneeAcquisitionBindings",
    "CogneeAcquisitionConfigurationError",
    "CogneeSubprocessBindings",
    "CogneeWorkerScope",
    "CognitiveAcquisition",
    "CognitiveGraphProjection",
    "CognitiveGraphProjectionV2",
    "CognitiveReferenceBackend",
    "CognitiveReferenceHit",
    "DEFAULT_COGNEE_WORKER_SCOPE",
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
    "TypedCognitiveRecall",
    "TypedCognitiveRecallBatch",
    "TypedProjectionReceipt",
    "TypedSituatedMemory",
    "decode_projection",
    "encode_projection",
    "projection_id",
    "episode_record",
    "prospective_batch_record",
    "prospective_resolution_record",
    "cognee_worker_launch_argv",
    "cognee_worker_launch_cwd",
]
