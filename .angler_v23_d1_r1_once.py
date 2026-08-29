"""One-shot harness for V23-D1 terminal-publication recovery R1."""

from __future__ import annotations

import argparse
from importlib import metadata
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.runners import phase6_anml_fidelity_v23_d1_r1 as recovery
from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


ROOT = Path("/opt/angler/project")
PREFIX = Path("/opt/angler/results/phase6-anml-fidelity-v23-d1-r1")
CLAIM = PREFIX.with_suffix(".claim.json")
RESULT = PREFIX.with_suffix(".json")
FAILURE = PREFIX.with_suffix(".failure.json")

ORIGINAL_SOURCES = {
    "source_leaf": (
        ROOT / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-ANML-FIDELITY-V23-D1-001.md",
        "474A7A75EC6D5C1864C573DBF3D8CD61AB568121AE61F97A857F30B7D8AA4901",
    ),
    "source_runner": (
        ROOT / "experiments/runners/phase6_anml_fidelity_v23_d1.py",
        recovery.SOURCE_RUNNER_SHA256,
    ),
    "source_test": (
        ROOT / "tests/unit/experiments/test_phase6_anml_fidelity_v23_d1.py",
        "48FA2665B324E937046F3C010A6307ABBD16AC985482ECBF3416C8CEB2D18D1B",
    ),
    "source_harness": (
        ROOT / ".angler_v23_d1_once.py",
        "A4B0F59C71D054DB84C0C344385BCF6107E711DD40DA34FC510DF3446F1AA5C8",
    ),
}
RECOVERY_SOURCES = {
    "leaf": ROOT / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-ANML-FIDELITY-V23-D1-R1-001.md",
    "runner": ROOT / "experiments/runners/phase6_anml_fidelity_v23_d1_r1.py",
    "test": ROOT / "tests/unit/experiments/test_phase6_anml_fidelity_v23_d1_r1.py",
    "harness": ROOT / ".angler_v23_d1_r1_once.py",
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


def verify_inputs(args: argparse.Namespace) -> dict[str, object]:
    original = {name: sha256_file(path) for name, (path, _expected) in ORIGINAL_SOURCES.items()}
    expected_original = {name: expected for name, (_path, expected) in ORIGINAL_SOURCES.items()}
    if original != expected_original:
        raise RuntimeError(f"V23-D1-R1 frozen source identity mismatch: {original}")
    recovered = {name: sha256_file(path) for name, path in RECOVERY_SOURCES.items()}
    if recovered != expected_hashes(args):
        raise RuntimeError(f"V23-D1-R1 recovery source identity mismatch: {recovered}")
    evidence = {
        "claim": sha256_file(recovery.SOURCE_CLAIM),
        "failure": sha256_file(recovery.SOURCE_FAILURE),
    }
    if evidence != {
        "claim": recovery.SOURCE_CLAIM_SHA256,
        "failure": recovery.SOURCE_FAILURE_SHA256,
    }:
        raise RuntimeError(f"V23-D1-R1 failed-evidence identity mismatch: {evidence}")
    if recovery.SOURCE_RESULT.exists():
        raise RuntimeError("V23-D1-R1 source result unexpectedly exists")
    return {"original_source_sha256": original, "recovery_source_sha256": recovered, "failed_evidence_sha256": evidence}


def verify_outputs_absent() -> None:
    occupied = [str(path) for path in (CLAIM, RESULT, FAILURE) if path.exists()]
    if occupied:
        raise RuntimeError(f"V23-D1-R1 output identity is already occupied: {occupied}")


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


def environment_record() -> dict[str, object]:
    libraries: dict[str, str] = {}
    for name in ("numpy", "scipy", "pytest"):
        try:
            libraries[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libraries[name] = "not-installed"
    return {
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "libraries": libraries,
    }


def run_once(args: argparse.Namespace) -> dict[str, object]:
    inputs = verify_inputs(args)
    verify_outputs_absent()
    claim = {
        "artifact_schema": "angler.anml-fidelity-v23-d1-r1.claim.v1",
        "protocol_id": recovery.PROTOCOL_ID,
        "source_protocol_id": recovery.SOURCE_PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **inputs,
        "first_recovered_result_without_tuning": True,
        "scientific_configuration_changed": False,
        "source_outcome": "HARNESS_ERROR_PRESERVED",
        "development_only": True,
    }
    write_exclusive(CLAIM, claim)
    claim_sha = sha256_file(CLAIM)
    try:
        result = recovery.run("cuda")
        result["claim"] = {"path": str(CLAIM), "sha256": claim_sha}
        result["source_sha256"] = inputs
        result["execution_environment"] = environment_record()
        result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        v22._validate_json_value(result)
        v22.atomic_write_json(RESULT, result)
        if FAILURE.exists():
            raise RuntimeError("V23-D1-R1 success collided with a failure artifact")
        return {
            "status": "complete",
            "source_outcome": "HARNESS_ERROR_PRESERVED",
            "classification": result["classification"],
            "selected_configuration": result["decision"]["selected_configuration"],
            "claim_sha256": claim_sha,
            "result_sha256": sha256_file(RESULT),
            "result_path": str(RESULT),
        }
    except BaseException as error:
        failure = {
            "artifact_schema": "angler.anml-fidelity-v23-d1-r1.failure.v1",
            "protocol_id": recovery.PROTOCOL_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "claim_sha256": claim_sha,
            "source_outcome": "HARNESS_ERROR_PRESERVED",
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
    inputs = verify_inputs(args)
    if args.mode == "preflight":
        verify_outputs_absent()
        payload = {
            "status": "preflight_pass",
            "inputs": inputs,
            "mechanics": recovery.synthetic_preflight("cuda"),
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

