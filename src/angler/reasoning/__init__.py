"""Trainable processing cores kept separate from foundation-model knowledge."""

from .recurrent_core import (
    ReasoningCoreConfig,
    ReasoningTrajectory,
    RecurrentReasoningCore,
    reasoning_state_digest,
    restore_reasoning_state,
    snapshot_reasoning_state,
)

__all__ = [
    "ReasoningCoreConfig",
    "ReasoningTrajectory",
    "RecurrentReasoningCore",
    "reasoning_state_digest",
    "restore_reasoning_state",
    "snapshot_reasoning_state",
]
