"""Deterministic evidence primitives for Project Angler's CR0 scaffold.

This package is deliberately small and standard-library only.  It validates
construction-time evidence structures; it does not persist evidence, authorize
work, run a learner, or make safety/scientific decisions.
"""

from .canonical import (
    CANONICALIZATION_PROFILE,
    CanonicalizationError,
    canonical_bytes,
    compute_artifact_id,
    compute_content_id,
    parse_json,
    verify_artifact_id,
    verify_content_id,
)
from .schema_validation import (
    EvidenceValidationError,
    validate_envelope,
    validate_episode,
    validate_manifest,
    validate_manifest_immutability,
)
from .visibility import (
    VisibilityClass,
    VisibilityDenied,
    authorize_projection,
    combine_policies,
    may_mutate,
    validate_visibility_policy,
)

__all__ = [
    "CANONICALIZATION_PROFILE",
    "CanonicalizationError",
    "EvidenceValidationError",
    "VisibilityClass",
    "VisibilityDenied",
    "authorize_projection",
    "canonical_bytes",
    "combine_policies",
    "compute_artifact_id",
    "compute_content_id",
    "may_mutate",
    "parse_json",
    "validate_envelope",
    "validate_episode",
    "validate_manifest",
    "validate_manifest_immutability",
    "validate_visibility_policy",
    "verify_artifact_id",
    "verify_content_id",
]
