"""One-shot atomic harness for the V23-D1 development diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.runners import phase6_anml_fidelity_v23_d1 as d1
from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


ROOT = Path("/opt/angler/project")
PREFIX = Path("/opt/angler/results/phase6-anml-fidelity-v23-d1")
CLAIM = PREFIX.with_suffix(".claim.json")
RESULT = PREFIX.with_suffix(".json")
FAILURE = PREFIX.with_suffix(".failure.json")

SOURCES = {
    "leaf": ROOT / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-ANML-FIDELITY-V23-D1-001.md",
    "runner": ROOT / "experiments/runners/phase6_anml_fidelity_v23_d1.py",
    "test": ROOT / "tests/unit/experiments/test_phase6_anml_fidelity_v23_d1.py",
    "harness": ROOT / ".angler_v23_d1_once.py",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def expected_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "leaf": args.expected_leaf_sha256.upper(),
        "runner": args.expected_runner_sha256.upper(),
        "test": args.expected_test_sha256.upper(),
        "harness": args.expected_harness_sha256.upper(),
    }


def verify_sources(args: argparse.Namespace) -> dict[str, str]:
    expected = expected_hashes(args)
    actual = {name: sha256_file(path) for name, path in SOURCES.items()}
    if actual != expected:
        raise RuntimeError(f"V23-D1 source identity mismatch: {actual}")
    if sha256_file(d1.V22_CHECKPOINT) != d1.V22_CHECKPOINT_SHA256:
        raise RuntimeError("V23-D1 sealed V22 checkpoint identity changed")
    return actual


def verify_outputs_absent() -> None:
    occupied = [str(path) for path in (CLAIM, RESULT, FAILURE) if path.exists()]
    if occupied:
        raise RuntimeError(f"V23-D1 output identity is already occupied: {occupied}")


def write_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def run_once(args: argparse.Namespace) -> dict[str, object]:
    hashes = verify_sources(args)
    verify_outputs_absent()
    claim = {
        "artifact_schema": "angler.anml-fidelity-v23-d1.claim.v1",
        "protocol_id": d1.PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "sealed_v22_checkpoint_sha256": d1.V22_CHECKPOINT_SHA256,
        "first_result_without_tuning": True,
        "development_only": True,
    }
    write_exclusive(CLAIM, claim)
    claim_sha = sha256_file(CLAIM)
    try:
        result = d1.run("cuda")
        result["claim"] = {"path": str(CLAIM), "sha256": claim_sha}
        result["source_sha256"] = hashes
        result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        v22.atomic_write_json(RESULT, result)
        if FAILURE.exists():
            raise RuntimeError("V23-D1 success collided with a failure artifact")
        return {
            "status": "complete",
            "classification": result["classification"],
            "selected_configuration": result["decision"]["selected_configuration"],
            "claim_sha256": claim_sha,
            "result_sha256": sha256_file(RESULT),
            "result_path": str(RESULT),
        }
    except BaseException as error:
        failure = {
            "artifact_schema": "angler.anml-fidelity-v23-d1.failure.v1",
            "protocol_id": d1.PROTOCOL_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "claim_sha256": claim_sha,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "result_exists": RESULT.exists(),
        }
        if not FAILURE.exists():
            v22.atomic_write_json(FAILURE, failure)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("mode", choices=("preflight", "run"))
    result.add_argument("--expected-leaf-sha256", required=True)
    result.add_argument("--expected-runner-sha256", required=True)
    result.add_argument("--expected-test-sha256", required=True)
    result.add_argument("--expected-harness-sha256", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    hashes = verify_sources(args)
    if args.mode == "preflight":
        verify_outputs_absent()
        payload = {
            "status": "preflight_pass",
            "source_sha256": hashes,
            "mechanics": d1.synthetic_preflight("cuda"),
            "outputs_absent": True,
        }
    else:
        payload = run_once(args)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True))
        raise

