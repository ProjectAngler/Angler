"""Trainable processing cores kept separate from foundation-model knowledge."""

from .recurrent_core import (
    ReasoningCoreConfig,
    ReasoningTrajectory,
    RecurrentReasoningCore,
    reasoning_state_digest,
    restore_reasoning_state,
    snapshot_reasoning_state,
)
from .self_referential_memory import (
    SelfReferentialMemory,
    SelfReferentialState,
    detach_self_referential_state,
    restore_self_referential_state,
    self_referential_state_digest,
    snapshot_self_referential_state,
)

__all__ = [
    "ReasoningCoreConfig",
    "ReasoningTrajectory",
    "RecurrentReasoningCore",
    "reasoning_state_digest",
    "restore_reasoning_state",
    "snapshot_reasoning_state",
    "SelfReferentialMemory",
    "SelfReferentialState",
    "detach_self_referential_state",
    "restore_self_referential_state",
    "self_referential_state_digest",
    "snapshot_self_referential_state",
]
