"""Structural validators for the three CR0 EVIDENCE contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .canonical import (
    compute_artifact_id,
    is_digest_id,
    is_protected_mode,
    supported_major,
    verify_content_id,
)
from .visibility import validate_visibility_policy

ENVELOPE_CONTRACT = "ANG-CTR-EVIDENCE-ENVELOPE-001"
EPISODE_CONTRACT = "ANG-CTR-EPISODE-001"
MANIFEST_CONTRACT = "ANG-CTR-EXPERIMENT-MANIFEST-001"

_SEMANTIC_BINDINGS = {
    "model_ref",
    "tokenizer_ref",
    "plastic_state_ref",
    "updater_optimizer_ref",
    "task_partition_ref",
    "tool_registry_ref",
    "seed_set_ref",
    "resource_inventory_ref",
    "execution_plan_ref",
    "experiment_manifest_ref",
    "evaluation_gate_ref",
}
_EPISODE_PARTITIONS = {
    "TRAIN",
    "DEVELOPMENT",
    "HELD_OUT",
    "SEALED_TRANSFER",
    "SAFETY_CONTROL",
}
_LEARNER_FORBIDDEN_PARTITIONS = {
    "HELD_OUT",
    "SEALED_TRANSFER",
    "SAFETY_CONTROL",
}


class EvidenceValidationError(ValueError):
    """Stable typed error without embedded protected values."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise EvidenceValidationError(code)


def _require(mapping: Mapping[str, Any], fields: set[str], code: str) -> None:
    if not isinstance(mapping, Mapping) or not fields.issubset(mapping):
        _fail(code)


def _contract(value: Any, expected_id: str) -> None:
    _require(value, {"id", "version"}, "EVIDENCE_REQUIRED_FIELD_MISSING")
    if value["id"] != expected_id:
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")
    if not supported_major(value["version"]):
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail("EVIDENCE_TIMESTAMP_INVALID")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail("EVIDENCE_TIMESTAMP_INVALID")
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        _fail("EVIDENCE_TIMESTAMP_INVALID")
    return parsed


def _explicit_ref_or_na(value: Any) -> None:
    if isinstance(value, str) and value:
        return
    if (
        isinstance(value, Mapping)
        and value.get("not_applicable") is True
        and isinstance(value.get("reason_code"), str)
        and value["reason_code"]
    ):
        return
    _fail("EVIDENCE_REQUIRED_FIELD_MISSING")


def _typed_ref(value: Any) -> None:
    required = {
        "artifact_id",
        "content_id",
        "artifact_type",
        "payload_contract",
        "relationship",
        "expected_visibility_class",
    }
    _require(value, required, "EVIDENCE_PARENT_INVALID")
    if not is_digest_id(value["artifact_id"]) or not is_digest_id(value["content_id"]):
        _fail("EVIDENCE_PARENT_INVALID")
    _require(value["payload_contract"], {"id", "version"}, "EVIDENCE_PARENT_INVALID")
    if not supported_major(value["payload_contract"]["version"]):
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")


def validate_envelope(
    envelope: Mapping[str, Any],
    *,
    payload: Any | None = None,
    commitment_key: bytes | None = None,
    require_authorization_complete: bool = False,
) -> dict[str, Any]:
    required = {
        "envelope_contract",
        "payload_contract",
        "artifact_type",
        "artifact_id",
        "payload_binding",
        "producer",
        "created_at_utc",
        "reproducibility",
        "parents",
        "visibility",
        "semantic_bindings",
        "human_impact",
        "integrity",
        "attestations",
        "extensions",
    }
    _require(envelope, required, "EVIDENCE_REQUIRED_FIELD_MISSING")
    _contract(envelope["envelope_contract"], ENVELOPE_CONTRACT)
    _require(envelope["payload_contract"], {"id", "version"}, "EVIDENCE_REQUIRED_FIELD_MISSING")
    if not supported_major(envelope["payload_contract"]["version"]):
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")
    if not is_digest_id(envelope["artifact_id"]):
        _fail("EVIDENCE_IDENTITY_MISMATCH")

    binding = envelope["payload_binding"]
    _require(
        binding,
        {"mode", "content_id", "canonicalization_profile", "canonical_size_or_protected_size_class"},
        "EVIDENCE_REQUIRED_FIELD_MISSING",
    )
    if binding["canonicalization_profile"] != "ANG-CANON-JSON-001@1":
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")
    mode = binding["mode"]
    if mode not in {"SHA256", "HMAC_SHA256", "AUTHENTICATED_CIPHERTEXT_SHA256"}:
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")
    if not is_digest_id(binding["content_id"]):
        _fail("EVIDENCE_CONTENT_MISMATCH")
    if is_protected_mode(mode) and not binding.get("commitment_key_or_cipher_profile_ref"):
        _fail("EVIDENCE_REQUIRED_FIELD_MISSING")

    try:
        checked_visibility = validate_visibility_policy(envelope["visibility"])
    except ValueError as exc:
        _fail(str(exc))
    low_entropy = bool(envelope["integrity"].get("low_entropy_or_enumerable"))
    if (
        checked_visibility["class"] in {"SEALED_EVALUATION", "HUMAN_AUTHORITY"}
        and low_entropy
        and mode == "SHA256"
    ):
        _fail("EVIDENCE_PROTECTED_COMMITMENT_REQUIRED")

    _timestamp(envelope["created_at_utc"])
    _require(
        envelope["producer"],
        {"component_id", "code_identity", "dependency_snapshot_ref", "run_or_transaction_ref", "authority_or_delegation_ref"},
        "EVIDENCE_REQUIRED_FIELD_MISSING",
    )
    _require(envelope["reproducibility"], {"level", "tolerance_profile_ref"}, "EVIDENCE_REQUIRED_FIELD_MISSING")
    if envelope["reproducibility"]["level"] not in {
        "DETERMINISTIC",
        "SEEDED_TOLERANT",
        "OBSERVATIONAL",
    }:
        _fail("EVIDENCE_REPRODUCIBILITY_INVALID")

    if not isinstance(envelope["parents"], list):
        _fail("EVIDENCE_PARENT_INVALID")
    parent_ids: set[str] = set()
    for parent in envelope["parents"]:
        _typed_ref(parent)
        if parent["artifact_id"] == envelope["artifact_id"] or parent["artifact_id"] in parent_ids:
            _fail("EVIDENCE_PARENT_INVALID")
        parent_ids.add(parent["artifact_id"])

    semantic = envelope["semantic_bindings"]
    _require(semantic, _SEMANTIC_BINDINGS, "EVIDENCE_REQUIRED_FIELD_MISSING")
    for key in _SEMANTIC_BINDINGS:
        _explicit_ref_or_na(semantic[key])

    impact = envelope["human_impact"]
    _require(impact, {"requirement", "basis_ref", "assessment_ref"}, "EVIDENCE_REQUIRED_FIELD_MISSING")
    if impact["requirement"] not in {"REQUIRED", "NOT_REQUIRED_WITH_BASIS"}:
        _fail("EVIDENCE_AUTHORIZATION_INCOMPLETE")
    authorization_complete = not (
        impact["requirement"] == "REQUIRED" and impact["assessment_ref"] is None
    )
    if require_authorization_complete and not authorization_complete:
        _fail("EVIDENCE_AUTHORIZATION_INCOMPLETE")

    if payload is not None:
        if not verify_content_id(
            payload,
            binding["content_id"],
            mode=mode,
            key=commitment_key,
        ):
            _fail("EVIDENCE_CONTENT_MISMATCH")
        if compute_artifact_id(envelope) != envelope["artifact_id"]:
            _fail("EVIDENCE_IDENTITY_MISMATCH")

    return {
        "valid": True,
        "authorization_complete": authorization_complete,
        "candidate_only": not authorization_complete,
    }


def _episode_ref(value: Any) -> None:
    _typed_ref(value)
    _require(value, {"run_ref", "attempt_ref"}, "EPISODE_REFERENCE_MISMATCH")


def validate_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "episode_id",
        "task_spec_ref",
        "trajectory_ref",
        "feedback_ref",
        "model_ref",
        "tokenizer_ref",
        "plastic_state_ref",
        "execution_plan_ref",
        "resource_inventory_ref",
        "tool_registry_ref",
        "experiment_manifest_ref",
        "environment_and_verifier_refs",
        "partition",
        "learner_eligibility",
        "eligibility_policy_ref",
        "feedback_exposure",
        "attempt_index",
        "started_at_utc",
        "ended_at_utc",
        "termination",
        "resource_usage_ref",
        "declared_failure_refs",
    }
    _require(episode, required, "EPISODE_REQUIRED_FIELD_MISSING")
    if not supported_major(episode["schema_version"]):
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")
    if not is_digest_id(episode["episode_id"]):
        _fail("EPISODE_REFERENCE_MISMATCH")
    refs = [episode["task_spec_ref"], episode["trajectory_ref"], episode["feedback_ref"]]
    for ref in refs:
        _episode_ref(ref)
    run_attempts = {(ref["run_ref"], ref["attempt_ref"]) for ref in refs}
    if len(run_attempts) != 1:
        _fail("EPISODE_REFERENCE_MISMATCH")
    if episode["partition"] not in _EPISODE_PARTITIONS:
        _fail("EPISODE_PARTITION_INVALID")
    if episode["learner_eligibility"] not in {"ELIGIBLE", "INELIGIBLE"}:
        _fail("EPISODE_LEARNING_INELIGIBLE")
    if (
        episode["partition"] in _LEARNER_FORBIDDEN_PARTITIONS
        and episode["learner_eligibility"] != "INELIGIBLE"
    ):
        _fail("EPISODE_LEARNING_INELIGIBLE")
    if episode["feedback_exposure"] not in {
        "NONE",
        "SCORE_ONLY",
        "STRUCTURED_OUTCOME",
        "FULL_DECLARED_FEEDBACK",
    }:
        _fail("EPISODE_FEEDBACK_EXPOSURE_INVALID")
    if not isinstance(episode["attempt_index"], int) or episode["attempt_index"] < 0:
        _fail("EPISODE_REFERENCE_MISMATCH")
    if _timestamp(episode["ended_at_utc"]) < _timestamp(episode["started_at_utc"]):
        _fail("EPISODE_TIME_INVALID")
    if episode["termination"] not in {"COMPLETED", "FAILED", "TIMEOUT", "ABORTED", "CONTAINED"}:
        _fail("EPISODE_TERMINATION_INVALID")
    return {"valid": True, "learner_eligible": episode["learner_eligibility"] == "ELIGIBLE"}


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "manifest_id",
        "purpose_and_claims",
        "hypotheses_and_falsifiers",
        "prohibited_claims",
        "code_and_dependency_refs",
        "model_tokenizer_state_updater_optimizer_refs",
        "tool_registry_and_permission_refs",
        "resource_inventory_plan_headroom_ceiling_refs",
        "task_environment_generator_verifier_refs",
        "partition_and_visibility_refs",
        "evaluation_suite_and_intervention_refs",
        "baseline_and_fair_budget_refs",
        "metrics_statistics_thresholds_and_negative_controls",
        "seed_set_and_tolerance_profile",
        "human_impact_requirement_and_ref",
        "required_gate_refs",
        "expected_output_contracts",
        "stop_incident_and_rollback_refs",
        "responsible_component_and_authority_refs",
        "planned_start_and_expiry",
        "executable",
        "pending_refs",
    }
    _require(manifest, required, "MANIFEST_INCOMPLETE")
    if not supported_major(manifest["schema_version"]):
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")
    if not is_digest_id(manifest["manifest_id"]):
        _fail("MANIFEST_INCOMPLETE")
    if not isinstance(manifest["pending_refs"], list):
        _fail("MANIFEST_INCOMPLETE")
    if manifest["executable"] and manifest["pending_refs"]:
        _fail("MANIFEST_INCOMPLETE")
    reproduction = manifest["seed_set_and_tolerance_profile"]
    _require(reproduction, {"seed_set_ref", "reproduction_level", "tolerance_profile_ref"}, "MANIFEST_INCOMPLETE")
    level = reproduction["reproduction_level"]
    if level not in {"DETERMINISTIC", "SEEDED_TOLERANT", "OBSERVATIONAL"}:
        _fail("MANIFEST_INCOMPLETE")
    if level == "SEEDED_TOLERANT" and reproduction["tolerance_profile_ref"] is None:
        _fail("MANIFEST_INCOMPLETE")
    impact = manifest["human_impact_requirement_and_ref"]
    _require(impact, {"requirement", "assessment_ref"}, "MANIFEST_INCOMPLETE")
    if impact["requirement"] == "REQUIRED" and not impact["assessment_ref"]:
        _fail("MANIFEST_INCOMPLETE")
    times = manifest["planned_start_and_expiry"]
    _require(times, {"planned_start_utc", "expires_at_utc"}, "MANIFEST_INCOMPLETE")
    if _timestamp(times["expires_at_utc"]) <= _timestamp(times["planned_start_utc"]):
        _fail("MANIFEST_INCOMPLETE")
    controls = manifest["metrics_statistics_thresholds_and_negative_controls"]
    _require(controls, {"metrics", "statistics_policy_ref", "thresholds", "negative_controls"}, "MANIFEST_INCOMPLETE")
    if not controls["thresholds"] or not controls["negative_controls"]:
        _fail("MANIFEST_INCOMPLETE")
    return {"valid": True, "executable": bool(manifest["executable"])}


def validate_manifest_immutability(
    original: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    """Reject material mutation that reuses a manifest identity."""

    if original == candidate:
        return True
    if original.get("manifest_id") != candidate.get("manifest_id"):
        return True
    control_key = "metrics_statistics_thresholds_and_negative_controls"
    original_controls = original.get(control_key, {})
    candidate_controls = candidate.get(control_key, {})
    if original_controls.get("thresholds") != candidate_controls.get("thresholds"):
        _fail("MANIFEST_THRESHOLD_REUSE")
    _fail("MANIFEST_MUTATION")
