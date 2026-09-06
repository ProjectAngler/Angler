#!/usr/bin/env python3
"""Capture isolated Jenny binding arms, then finalize their comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from angler.runtime.jenny2_qualification import (
    capture_cumulative_binding_arm,
    capture_resolution_state_conditioning_probe,
    finalize_cumulative_binding_qualification,
    verify_candidate_only_reading,
)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binding-path", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--parent-binding-path", required=True)
    parser.add_argument("--parent-served-model", required=True)
    parser.add_argument("--source-state-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument(
        "--library-root", default="/home/angler/Desktop/Jenny Library"
    )
    parser.add_argument("--qualification-id", required=True)


def _summary(prefix: str, artifact, **extra: object) -> None:
    print(
        prefix
        + json.dumps(
            {
                "artifact_path": str(artifact.path),
                "artifact_sha256": artifact.sha256,
                **extra,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture parent and candidate whole-system arms in separate invocations "
            "so one TP2 endpoint can be restarted between them, then finalize their "
            "content-addressed receipts. Source state and Library stay read-only; "
            "PASS is not activation."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "capture-arm",
        help="run one exact arm against the model currently served on loopback",
    )
    _common_arguments(capture)
    capture.add_argument(
        "--arm",
        required=True,
        choices=("parent_control", "weighted_composite"),
    )
    capture.add_argument("--clone-root", required=True)

    probe = commands.add_parser(
        "probe-resolution-state",
        help=(
            "run two no-commit deliberations against a novel state relevance "
            "contrast without replaying the Library protocol"
        ),
    )
    probe.add_argument("--binding-path", required=True)
    probe.add_argument("--served-model", required=True)
    probe.add_argument("--source-state-root", required=True)
    probe.add_argument("--clone-root", required=True)
    probe.add_argument("--result-root", required=True)
    probe.add_argument(
        "--library-root", default="/home/angler/Desktop/Jenny Library"
    )
    probe.add_argument("--probe-id", required=True)
    probe.add_argument("--source-target-ref", required=True)
    probe.add_argument("--expected-source-state-ref", required=True)
    probe.add_argument("--expected-source-ordinal", required=True, type=int)
    probe.add_argument("--expected-source-last-event-ref", required=True)

    compact = commands.add_parser(
        "verify-candidate-reading",
        help=(
            "run the four-turn candidate-only source/answer-lesion whole-system "
            "verification without invoking the parent model"
        ),
    )
    compact.add_argument("--binding-path", required=True)
    compact.add_argument("--calibration-path", required=True)
    compact.add_argument("--served-model", required=True)
    compact.add_argument("--source-state-root", required=True)
    compact.add_argument("--source-exposed-clone-root", required=True)
    compact.add_argument("--answer-lesioned-control-clone-root", required=True)
    compact.add_argument("--result-root", required=True)
    compact.add_argument(
        "--source-exposed-library-root",
        default=(
            "/opt/angler/src/angler/experiments/corpora/"
            "jenny2-candidate-reading-verification-v1/source-library"
        ),
    )
    compact.add_argument(
        "--answer-lesioned-control-library-root",
        default=(
            "/opt/angler/src/angler/experiments/corpora/"
            "jenny2-candidate-reading-verification-v1/control-library"
        ),
    )
    compact.add_argument("--verification-id", required=True)

    finalize = commands.add_parser(
        "finalize",
        help="validate and compare two completed content-addressed arm captures",
    )
    _common_arguments(finalize)
    finalize.add_argument("--clone-root", required=True)
    finalize.add_argument("--parent-clone-root", required=True)
    finalize.add_argument("--parent-arm-artifact", required=True)
    finalize.add_argument("--weighted-arm-artifact", required=True)
    args = parser.parse_args()

    if args.command == "verify-candidate-reading":
        artifact = verify_candidate_only_reading(
            binding_path=Path(args.binding_path),
            calibration_path=Path(args.calibration_path),
            served_model=args.served_model,
            source_state_root=Path(args.source_state_root),
            source_exposed_clone_root=Path(args.source_exposed_clone_root),
            answer_lesioned_control_clone_root=Path(
                args.answer_lesioned_control_clone_root
            ),
            result_root=Path(args.result_root),
            source_exposed_library_root=Path(args.source_exposed_library_root),
            answer_lesioned_control_library_root=Path(
                args.answer_lesioned_control_library_root
            ),
            verification_id=args.verification_id,
        )
        _summary(
            "JENNY2_CANDIDATE_ONLY_READING_VERIFICATION=",
            artifact,
            passed=artifact.passed,
        )
        return 0 if artifact.passed else 2

    if args.command == "probe-resolution-state":
        artifact = capture_resolution_state_conditioning_probe(
            binding_path=Path(args.binding_path),
            served_model=args.served_model,
            source_state_root=Path(args.source_state_root),
            clone_root=Path(args.clone_root),
            result_root=Path(args.result_root),
            library_root=Path(args.library_root),
            probe_id=args.probe_id,
            source_target_ref=args.source_target_ref,
            expected_source_state_ref=args.expected_source_state_ref,
            expected_source_ordinal=args.expected_source_ordinal,
            expected_source_last_event_ref=args.expected_source_last_event_ref,
        )
        _summary(
            "JENNY2_RESOLUTION_STATE_PROBE=",
            artifact,
            probe_complete=artifact.passed,
        )
        return 0 if artifact.passed else 2

    common = {
        "binding_path": Path(args.binding_path),
        "served_model": args.served_model,
        "parent_binding_path": Path(args.parent_binding_path),
        "parent_served_model": args.parent_served_model,
        "source_state_root": Path(args.source_state_root),
        "result_root": Path(args.result_root),
        "library_root": Path(args.library_root),
        "qualification_id": args.qualification_id,
    }
    if args.command == "capture-arm":
        artifact = capture_cumulative_binding_arm(
            **common,
            arm_name=args.arm,
            clone_root=Path(args.clone_root),
        )
        _summary(
            "JENNY2_CUMULATIVE_BINDING_ARM_CAPTURE=",
            artifact,
            arm=args.arm,
            capture_complete=artifact.passed,
        )
        return 0 if artifact.passed else 2

    artifact = finalize_cumulative_binding_qualification(
        **common,
        clone_root=Path(args.clone_root),
        parent_clone_root=Path(args.parent_clone_root),
        parent_arm_artifact=Path(args.parent_arm_artifact),
        weighted_arm_artifact=Path(args.weighted_arm_artifact),
    )
    _summary(
        "JENNY2_CUMULATIVE_BINDING_QUALIFICATION=",
        artifact,
        passed=artifact.passed,
    )
    return 0 if artifact.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
