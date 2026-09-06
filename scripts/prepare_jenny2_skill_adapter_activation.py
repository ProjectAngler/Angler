#!/usr/bin/env python3
"""Stage a binding or emit a qualification-bound, inert activation plan."""

from __future__ import annotations

import argparse
import json

from angler.runtime.skill_adapter_activation import (
    ExpectedSkillComponent,
    prepare_skill_adapter_activation,
)


_COMPONENT_FIELDS = {
    "name",
    "adapter_model_sha256",
    "adapter_config_sha256",
    "weight",
}
_COMPARATIVE_QUALIFICATION_BASIS = "comparative-qualified-v2"
_CANDIDATE_ONLY_EXPERIMENTAL_BASIS = "candidate-only-experimental-v1"


def _component(value: str) -> ExpectedSkillComponent:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "expected component must be a JSON object"
        ) from exc
    if type(payload) is not dict or set(payload) != _COMPONENT_FIELDS:
        raise argparse.ArgumentTypeError(
            "expected component fields must be exactly "
            + ", ".join(sorted(_COMPONENT_FIELDS))
        )
    if any(type(payload[field]) is not str for field in _COMPONENT_FIELDS):
        raise argparse.ArgumentTypeError("every expected component field must be a string")
    return ExpectedSkillComponent(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed rank-16 skill adapter composition. --staging-only "
            "writes only the immutable candidate binding needed by the isolated "
            "verifier. Final mode defaults to the comparative v2 qualification; the "
            "explicit candidate-only experimental basis accepts only its compact "
            "PASS verification. Both emit inert operator-run activation/rollback "
            "argv. This script never executes commands, polls a service, or changes "
            "runtime state."
        )
    )
    parser.add_argument("--staging-only", action="store_true")
    parser.add_argument(
        "--activation-basis",
        choices=(
            _COMPARATIVE_QUALIFICATION_BASIS,
            _CANDIDATE_ONLY_EXPERIMENTAL_BASIS,
        ),
        default=_COMPARATIVE_QUALIFICATION_BASIS,
    )
    parser.add_argument("--composite-run", required=True)
    parser.add_argument("--expected-training-result-sha256", required=True)
    parser.add_argument("--expected-adapter-model-sha256", required=True)
    parser.add_argument("--expected-adapter-config-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-ref", required=True)
    parser.add_argument("--expected-source-training-result-sha256", required=True)
    parser.add_argument(
        "--expected-component",
        action="append",
        required=True,
        type=_component,
        help=(
            "repeat JSON: {name,adapter_model_sha256,adapter_config_sha256,weight}"
        ),
    )
    parser.add_argument("--expected-rank", required=True, type=int)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--candidate-qualification", required=True)
    parser.add_argument(
        "--expected-candidate-qualification-sha256",
        required=True,
    )
    parser.add_argument("--expected-candidate-qualification-ref", required=True)
    parser.add_argument(
        "--expected-calibration-protocol-manifest-ref",
        required=True,
    )
    parser.add_argument("--expected-pre-evaluation-manifest-ref", required=True)
    parser.add_argument("--expected-selected-weight", required=True)
    parser.add_argument("--served-qualification")
    parser.add_argument("--expected-served-qualification-sha256")
    parser.add_argument("--expected-served-qualification-ref")
    parser.add_argument("--candidate-only-verification")
    parser.add_argument("--expected-candidate-only-verification-sha256")
    parser.add_argument("--expected-candidate-only-verification-ref")
    parser.add_argument("--rollback-binding")
    parser.add_argument("--expected-rollback-binding-ref")
    parser.add_argument("--active-service-override")
    parser.add_argument("--expected-active-service-override-sha256")
    parser.add_argument("--candidate-container-name")
    parser.add_argument("--rollback-container-name")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cache-path", default="/opt/angler/runtime-cache/sglang-qwen38"
    )
    args = parser.parse_args()

    comparative_values = {
        "--served-qualification": args.served_qualification,
        "--expected-served-qualification-sha256": (
            args.expected_served_qualification_sha256
        ),
        "--expected-served-qualification-ref": (
            args.expected_served_qualification_ref
        ),
    }
    candidate_only_values = {
        "--candidate-only-verification": args.candidate_only_verification,
        "--expected-candidate-only-verification-sha256": (
            args.expected_candidate_only_verification_sha256
        ),
        "--expected-candidate-only-verification-ref": (
            args.expected_candidate_only_verification_ref
        ),
    }
    rollback_values = {
        "--rollback-binding": args.rollback_binding,
        "--expected-rollback-binding-ref": args.expected_rollback_binding_ref,
        "--active-service-override": args.active_service_override,
        "--expected-active-service-override-sha256": (
            args.expected_active_service_override_sha256
        ),
        "--candidate-container-name": args.candidate_container_name,
        "--rollback-container-name": args.rollback_container_name,
    }
    if args.staging_only:
        supplied = [
            name
            for name, value in (
                comparative_values | candidate_only_values | rollback_values
            ).items()
            if value is not None
        ]
        if args.activation_basis != _COMPARATIVE_QUALIFICATION_BASIS:
            supplied.append("--activation-basis")
        if supplied:
            parser.error(
                "--staging-only cannot be combined with final activation inputs: "
                + ", ".join(supplied)
            )
    else:
        if args.activation_basis == _COMPARATIVE_QUALIFICATION_BASIS:
            required_evidence = comparative_values
            forbidden_evidence = candidate_only_values
        else:
            required_evidence = candidate_only_values
            forbidden_evidence = comparative_values
        supplied_forbidden = [
            name for name, value in forbidden_evidence.items() if value is not None
        ]
        if supplied_forbidden:
            parser.error(
                f"{args.activation_basis} cannot be combined with: "
                + ", ".join(supplied_forbidden)
            )
        missing = [
            name
            for name, value in (required_evidence | rollback_values).items()
            if value is None
        ]
        if missing:
            parser.error(
                "final activation requires: " + ", ".join(missing)
            )

    plan = prepare_skill_adapter_activation(
        composite_run=args.composite_run,
        expected_training_result_sha256=args.expected_training_result_sha256,
        expected_adapter_model_sha256=args.expected_adapter_model_sha256,
        expected_adapter_config_sha256=args.expected_adapter_config_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_manifest_ref=args.expected_manifest_ref,
        expected_source_training_result_sha256=(
            args.expected_source_training_result_sha256
        ),
        expected_components=args.expected_component,
        expected_rank=args.expected_rank,
        expected_profile=args.expected_profile,
        candidate_qualification_path=args.candidate_qualification,
        expected_candidate_qualification_sha256=(
            args.expected_candidate_qualification_sha256
        ),
        expected_candidate_qualification_ref=(
            args.expected_candidate_qualification_ref
        ),
        expected_calibration_protocol_manifest_ref=(
            args.expected_calibration_protocol_manifest_ref
        ),
        expected_pre_evaluation_manifest_ref=(
            args.expected_pre_evaluation_manifest_ref
        ),
        expected_selected_weight=args.expected_selected_weight,
        activation_basis=args.activation_basis,
        served_qualification_path=args.served_qualification,
        expected_served_qualification_sha256=(
            args.expected_served_qualification_sha256
        ),
        expected_served_qualification_ref=(
            args.expected_served_qualification_ref
        ),
        candidate_only_verification_path=args.candidate_only_verification,
        expected_candidate_only_verification_sha256=(
            args.expected_candidate_only_verification_sha256
        ),
        expected_candidate_only_verification_ref=(
            args.expected_candidate_only_verification_ref
        ),
        rollback_binding_path=args.rollback_binding,
        expected_rollback_binding_ref=args.expected_rollback_binding_ref,
        active_service_override_path=args.active_service_override,
        expected_active_service_override_sha256=(
            args.expected_active_service_override_sha256
        ),
        candidate_container_name=args.candidate_container_name,
        rollback_container_name=args.rollback_container_name,
        output_root=args.output,
        cache_path=args.cache_path,
        staging_only=args.staging_only,
    )
    print(
        "JENNY2_SKILL_ADAPTER_ACTIVATION_PLAN="
        + json.dumps(
            {
                "binding_path": str(plan.binding_path),
                "binding_ref": plan.binding_ref,
                "commands_executed": False,
                "output_root": str(plan.output_root),
                "plan_path": (
                    None if plan.plan_path is None else str(plan.plan_path)
                ),
                "plan_sha256": plan.plan_sha256,
                "polling_performed": False,
                "rollback_binding_ref": plan.rollback_binding_ref,
                "runtime_changed": False,
                "served_qualification_ref": plan.served_qualification_ref,
                "status": plan.status,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
