"""One-shot equivalence and speed benchmark for parallel Angler episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from experiments.runners import parallel_episode_executor as executor
from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


ROOT = Path("/opt/angler/project")
PREFIX = Path("/opt/angler/results/parallel-episode-executor-v1-r1")
CLAIM = PREFIX.with_suffix(".claim.json")
RESULT = PREFIX.with_suffix(".json")
FAILURE = PREFIX.with_suffix(".failure.json")
SEQUENTIAL_PATHS = {
    panel: Path(f"/opt/angler/results/parallel-episode-executor-v1-r1.sequential-panel-{panel}.json")
    for panel in executor.PANELS
}
PARALLEL_PATHS = {
    panel: Path(f"/opt/angler/results/parallel-episode-executor-v1-r1.parallel-panel-{panel}.json")
    for panel in executor.PANELS
}
SOURCES = {
    "leaf": ROOT / "docs/blueprints/branches/learning/work/ANG-WORK-LEARNING-PARALLEL-EPISODE-EXECUTOR-001.md",
    "runner": ROOT / "experiments/runners/parallel_episode_executor.py",
    "test": ROOT / "tests/unit/experiments/test_parallel_episode_executor.py",
    "harness": ROOT / ".angler_parallel_episode_benchmark_once.py",
}
WALL_CEILING_SECONDS = 30.0 * 60.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "leaf": args.expected_leaf_sha256.upper(),
        "runner": args.expected_runner_sha256.upper(),
        "test": args.expected_test_sha256.upper(),
        "harness": args.expected_harness_sha256.upper(),
    }


def verify_sources(args: argparse.Namespace) -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path in SOURCES.items()}
    if actual != expected_hashes(args):
        raise RuntimeError(f"parallel episode source identity mismatch: {actual}")
    if sha256_file(executor.V22_CHECKPOINT) != executor.V22_CHECKPOINT_SHA256:
        raise RuntimeError("parallel episode sealed V22 checkpoint changed")
    return actual


def all_outputs() -> tuple[Path, ...]:
    return (CLAIM, RESULT, FAILURE, *SEQUENTIAL_PATHS.values(), *PARALLEL_PATHS.values())


def verify_outputs_absent() -> None:
    occupied = tuple(str(path) for path in all_outputs() if path.exists())
    if occupied:
        raise RuntimeError(f"parallel episode output identity is occupied: {occupied}")


def write_exclusive(path: Path, payload: dict[str, object]) -> None:
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


def preflight() -> dict[str, object]:
    record = executor.run_v22_panel_segment(0, 1)
    semantic = record["semantic"]
    if any(arm["terminal_fast_step"] != 1 for arm in semantic["arms"].values()):
        raise RuntimeError("parallel episode preflight did not execute one exact update")
    maximum = int(record["execution"]["maximum_allocated_bytes"])
    if maximum > executor.AGGREGATE_CUDA_CEILING_BYTES:
        raise RuntimeError("parallel episode preflight exceeded memory ceiling")
    del record
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "panel": 0,
        "updates": 1,
        "arms": tuple(executor.ARM_NAMES),
        "maximum_allocated_bytes": maximum,
        "checkpoint_sha256": executor.V22_CHECKPOINT_SHA256,
        "semantic_updates_persisted": False,
    }


def run_once(args: argparse.Namespace) -> dict[str, object]:
    sources = verify_sources(args)
    verify_outputs_absent()
    mechanics = preflight()
    verify_outputs_absent()
    claim = {
        "artifact_schema": "angler.parallel-episode-executor-v1-r1.claim.v1",
        "protocol_id": executor.PROTOCOL_ID,
        "created_at_utc": utc_now(),
        "source_sha256": sources,
        "checkpoint_sha256": executor.V22_CHECKPOINT_SHA256,
        "panels": executor.PANELS,
        "updates_per_panel": executor.BENCHMARK_STEPS,
        "arms": executor.ARM_NAMES,
        "preflight": mechanics,
        "no_scientific_claim": True,
    }
    write_exclusive(CLAIM, claim)
    claim_sha = sha256_file(CLAIM)
    started = time.monotonic()
    try:
        sequential_run = executor.launch_panel_workers(
            SEQUENTIAL_PATHS,
            steps=executor.BENCHMARK_STEPS,
            parallel=False,
        )
        parallel_run = executor.launch_panel_workers(
            PARALLEL_PATHS,
            steps=executor.BENCHMARK_STEPS,
            parallel=True,
        )
        sequential_receipts = executor.load_worker_receipts(SEQUENTIAL_PATHS)
        parallel_receipts = executor.load_worker_receipts(PARALLEL_PATHS)
        comparison = executor.compare_semantics(sequential_receipts, parallel_receipts)
        sequential_seconds = float(sequential_run["elapsed_seconds"])
        parallel_seconds = float(parallel_run["elapsed_seconds"])
        speedup = sequential_seconds / parallel_seconds
        parallel_peak_sum = sum(
            int(receipt["execution"]["maximum_allocated_bytes"])
            for receipt in parallel_receipts
        )
        if parallel_peak_sum > executor.AGGREGATE_CUDA_CEILING_BYTES:
            raise RuntimeError("parallel episode aggregate CUDA allocation exceeded 2 GiB")
        elapsed = time.monotonic() - started
        if elapsed > WALL_CEILING_SECONDS:
            raise RuntimeError("parallel episode benchmark exceeded wall ceiling")
        result = {
            "artifact_schema": "angler.parallel-episode-executor-v1-r1.result.v1",
            "protocol_id": executor.PROTOCOL_ID,
            "classification": (
                "PARALLEL_ACCELERATOR_SUPPORTED"
                if parallel_seconds < sequential_seconds
                else "MECHANICALLY_VALID_NO_SPEEDUP"
            ),
            "created_at_utc": utc_now(),
            "claim_sha256": claim_sha,
            "source_sha256": sources,
            "checkpoint_sha256": executor.V22_CHECKPOINT_SHA256,
            "comparison": comparison,
            "timing": {
                "sequential_seconds": sequential_seconds,
                "parallel_seconds": parallel_seconds,
                "speedup": speedup,
                "total_benchmark_seconds": elapsed,
            },
            "resources": {
                "parallel_worker_peak_sum_bytes": parallel_peak_sum,
                "aggregate_cuda_ceiling_bytes": executor.AGGREGATE_CUDA_CEILING_BYTES,
            },
            "worker_receipts": {
                "sequential": {
                    str(panel): {"path": str(SEQUENTIAL_PATHS[panel]), "sha256": sha256_file(SEQUENTIAL_PATHS[panel])}
                    for panel in executor.PANELS
                },
                "parallel": {
                    str(panel): {"path": str(PARALLEL_PATHS[panel]), "sha256": sha256_file(PARALLEL_PATHS[panel])}
                    for panel in executor.PANELS
                },
            },
            "bounded_interpretation": "testing/training episode scheduling only; sequential learner mathematics unchanged",
        }
        v22.atomic_write_json(RESULT, result)
        return {
            "status": "complete",
            "classification": result["classification"],
            "speedup": speedup,
            "claim_sha256": claim_sha,
            "result_sha256": sha256_file(RESULT),
            "result_path": str(RESULT),
        }
    except BaseException as error:
        failure = {
            "artifact_schema": "angler.parallel-episode-executor-v1-r1.failure.v1",
            "protocol_id": executor.PROTOCOL_ID,
            "created_at_utc": utc_now(),
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
    sources = verify_sources(args)
    if args.mode == "preflight":
        verify_outputs_absent()
        payload = {"status": "preflight_pass", "source_sha256": sources, "mechanics": preflight(), "outputs_absent": True}
    else:
        payload = run_once(args)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True))
        raise
