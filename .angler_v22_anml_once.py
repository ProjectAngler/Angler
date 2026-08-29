"""One-shot, resumable launcher for Project Angler's frozen V22 ANML trial.

This file owns only execution integrity: immutable claim publication, exact
source binding, checkpoint replacement, cumulative time enforcement, and
terminal evidence.  All learning and evaluation semantics live in the V22
runner.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


CLAIM_SCHEMA = "angler.phase6-anml-selective-plasticity-v22.claim.v1"
HARNESS_STATE_SCHEMA = "angler.phase6-anml-selective-plasticity-v22.harness-state.v1"
REPORT_SCHEMA = "angler.phase6-anml-selective-plasticity-v22.report.v1"
FAILURE_SCHEMA = "angler.phase6-anml-selective-plasticity-v22.failure.v1"

PREFIX = Path("/opt/angler/results/phase6-anml-selective-plasticity-v22")
CLAIM_PATH = PREFIX.with_suffix(".claim.json")
PROGRESS_PATH = PREFIX.with_suffix(".progress.pt")
CHECKPOINT_PATH = PREFIX.with_suffix(".pt")
REPORT_PATH = PREFIX.with_suffix(".json")
FAILURE_PATH = PREFIX.with_suffix(".failure.json")
CLAIM_TEMP = CLAIM_PATH.with_name(CLAIM_PATH.name + ".tmp")
PROGRESS_NEXT = PROGRESS_PATH.with_name(PROGRESS_PATH.name + ".next")
CHECKPOINT_NEXT = CHECKPOINT_PATH.with_name(CHECKPOINT_PATH.name + ".next")
REPORT_TEMP = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
FAILURE_TEMP = FAILURE_PATH.with_name(FAILURE_PATH.name + ".tmp")

RUNNER_PATH = ROOT / "experiments/runners/phase6_anml_selective_plasticity_v22.py"
TEST_PATH = ROOT / "tests/unit/experiments/test_phase6_anml_selective_plasticity_v22.py"
LEAF_PATH = ROOT / v22.ACTIVE_LEAF
HARNESS_PATH = Path(__file__).resolve()

EXPECTED_SOURCE_HASHES = {
    str(RUNNER_PATH): "9A36AD5BAFF77A17890D04B535666A54DDBDD179EE02BA979DE9A864756DF9DA",
    str(TEST_PATH): "F90A342E025F0037DFD85B9C7144ADB74AED22687983947FC1DC9EC90D7D5D50",
    str(LEAF_PATH): v22.ACTIVE_LEAF_SHA256,
}

WALL_CEILING_SECONDS = v22.SEMANTIC_WALL_TIME_CEILING_SECONDS
PUBLICATION_RESERVE_SECONDS = 30.0
CHECKPOINT_CEILING_BYTES = 16 * 1024**2
JSON_CEILING_BYTES = 32 * 1024**2
DEVICE = torch.device("cuda:0")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("V22 JSON evidence contains a non-finite float")
        return value
    raise TypeError(f"V22 JSON evidence contains unsupported {type(value).__name__}")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    ready = _json_ready(value)
    return (json.dumps(ready, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _publish_json(target: Path, temporary: Path, value: Mapping[str, object]) -> dict[str, object]:
    if _lexists(target) or _lexists(temporary):
        raise RuntimeError(f"V22 immutable JSON target is occupied: {target}")
    payload = _json_bytes(value)
    if not 0 < len(payload) <= JSON_CEILING_BYTES:
        raise RuntimeError("V22 JSON evidence exceeds its byte ceiling")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"path": str(target), "bytes": len(payload), "sha256": sha256_file(target)}


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"V22 JSON identity is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"V22 JSON identity is not an object: {path}")
    return value


def _output_paths() -> tuple[Path, ...]:
    return (
        CLAIM_PATH,
        PROGRESS_PATH,
        CHECKPOINT_PATH,
        REPORT_PATH,
        FAILURE_PATH,
        CLAIM_TEMP,
        PROGRESS_NEXT,
        CHECKPOINT_NEXT,
        REPORT_TEMP,
        FAILURE_TEMP,
    )


def _source_record(expected_harness_sha256: str) -> dict[str, object]:
    expected_harness = expected_harness_sha256.upper()
    if len(expected_harness) != 64 or any(c not in "0123456789ABCDEF" for c in expected_harness):
        raise ValueError("V22 expected harness SHA-256 is invalid")
    required = dict(EXPECTED_SOURCE_HASHES)
    required[str(HARNESS_PATH)] = expected_harness
    records: dict[str, object] = {}
    for raw_path, expected in required.items():
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"V22 source is not a regular file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"V22 source hash changed: {path}")
        records[str(path)] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return records


def _cuda_record() -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V22 requires exactly one visible CUDA device")
    v22.configure_anml_numerics(DEVICE)
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": str(DEVICE),
        "device_count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(properties.total_memory),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "dtype": "torch.float32",
        "tf32": False,
        "autocast": False,
    }


def _verify_fresh_outputs() -> None:
    occupied = [str(path) for path in _output_paths() if _lexists(path)]
    if occupied:
        raise RuntimeError(f"V22 output identity is already occupied: {occupied}")


def _validate_claim(expected_harness_sha256: str) -> tuple[dict[str, object], str]:
    claim = _read_json(CLAIM_PATH)
    if claim.get("artifact_schema") != CLAIM_SCHEMA:
        raise RuntimeError("V22 claim schema changed")
    if claim.get("protocol_id") != v22.PROTOCOL_ID:
        raise RuntimeError("V22 claim protocol changed")
    if claim.get("plan_digest") != v22.anml_plan_digest():
        raise RuntimeError("V22 claim plan digest changed")
    sources = claim.get("sources")
    current = _source_record(expected_harness_sha256)
    if sources != current:
        raise RuntimeError("V22 claim source binding changed")
    if (
        claim.get("wall_ceiling_seconds") != WALL_CEILING_SECONDS
        or claim.get("publication_reserve_seconds") != PUBLICATION_RESERVE_SECONDS
    ):
        raise RuntimeError("V22 claim wall ceiling changed")
    return claim, sha256_file(CLAIM_PATH)


def verify_launch(*, resume: bool, expected_harness_sha256: str) -> dict[str, object]:
    PREFIX.parent.mkdir(parents=True, exist_ok=True)
    if PREFIX.parent.is_symlink() or not PREFIX.parent.is_dir():
        raise RuntimeError("V22 output parent is invalid")
    sources = _source_record(expected_harness_sha256)
    dependencies = v22.verify_anml_dependencies()
    cuda = _cuda_record()
    if resume:
        if not CLAIM_PATH.is_file() or REPORT_PATH.exists() or FAILURE_PATH.exists():
            raise RuntimeError("V22 resume state is not active")
        if not (PROGRESS_PATH.is_file() or CHECKPOINT_PATH.is_file()):
            raise RuntimeError("V22 resume has no checkpoint")
        for path in (CLAIM_TEMP, PROGRESS_NEXT, CHECKPOINT_NEXT, REPORT_TEMP, FAILURE_TEMP):
            if _lexists(path):
                raise RuntimeError(f"V22 resume temporary is occupied: {path}")
        claim, claim_sha256 = _validate_claim(expected_harness_sha256)
    else:
        _verify_fresh_outputs()
        claim = None
        claim_sha256 = None
    return {
        "status": "PASS",
        "resume": resume,
        "protocol_id": v22.PROTOCOL_ID,
        "plan_digest": v22.anml_plan_digest(),
        "sources": sources,
        "dependencies": dependencies,
        "cuda": cuda,
        "outputs": {str(path): _lexists(path) for path in _output_paths()},
        "claim": claim,
        "claim_sha256": claim_sha256,
    }


def _feature_parity(system: v22.ANMLSystem) -> tuple[dict[str, object], dict[str, object]]:
    stream = v22.v12._relation_credit_panel_streams(
        v22.v19.v12_champion_paired_graph_context_plan()["commitments"],
        v22.v19.v12_champion_paired_graph_context_plan()["panel_seed_pairs"][0],
    )[0]
    ordinary = v22.exact_equivalent_feature_parity(
        system.controller, system.second_order_anml.gate, stream
    )
    duplicate = v22.exact_equivalent_feature_parity(
        system.controller,
        system.second_order_anml.gate,
        stream,
        duplicate_same_contract=True,
    )
    v22.mark_feature_equivalence_verified(system, (ordinary, duplicate))
    return ordinary, duplicate


def run_preflight(expected_harness_sha256: str) -> tuple[dict[str, object], v22.ANMLSystem]:
    launch = verify_launch(resume=False, expected_harness_sha256=expected_harness_sha256)
    synthetic = v22.synthetic_cuda_preflight(DEVICE)
    system = v22.build_anml_system(device=DEVICE)
    ordinary, duplicate = _feature_parity(system)
    _verify_fresh_outputs()
    return (
        {
            **launch,
            "synthetic_cuda": synthetic,
            "feature_parity": {"ordinary": ordinary, "duplicate_same_contract": duplicate},
            "system_digest": v22.anml_system_digest(system),
            "semantic_updates_performed": False,
        },
        system,
    )


def _create_claim(preflight: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    claim = {
        "artifact_schema": CLAIM_SCHEMA,
        "protocol_id": v22.PROTOCOL_ID,
        "plan_digest": v22.anml_plan_digest(),
        "created_at_utc": utc_now(),
        "sources": copy.deepcopy(preflight["sources"]),
        "dependencies": copy.deepcopy(preflight["dependencies"]),
        "environment": copy.deepcopy(preflight["cuda"]),
        "synthetic_cuda_preflight": copy.deepcopy(preflight["synthetic_cuda"]),
        "feature_parity": copy.deepcopy(preflight["feature_parity"]),
        "initial_system_digest": preflight["system_digest"],
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "publication_reserve_seconds": PUBLICATION_RESERVE_SECONDS,
        "memory_ceiling_bytes": v22.ALLOCATED_MEMORY_CEILING_BYTES,
        "include_random_control": False,
        "first_result_without_tuning": True,
        "semantic_identity_consumed": True,
    }
    publication = _publish_json(CLAIM_PATH, CLAIM_TEMP, claim)
    return claim, publication


def _checkpoint_state(
    *,
    phase: str,
    claim_sha256: str,
    cumulative_elapsed_seconds: float,
    evaluation_state: Mapping[str, object] | None,
    fit_report: Mapping[str, object] | None,
    evaluation_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if cumulative_elapsed_seconds < 0.0 or cumulative_elapsed_seconds > WALL_CEILING_SECONDS:
        raise RuntimeError("V22 cumulative time is invalid")
    return {
        "artifact_schema": HARNESS_STATE_SCHEMA,
        "phase": phase,
        "claim_sha256": claim_sha256,
        "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
        "evaluation_state": copy.deepcopy(evaluation_state),
        "fit_report": copy.deepcopy(fit_report),
        "evaluation_result": copy.deepcopy(evaluation_result),
    }


def _publish_progress(system: v22.ANMLSystem, harness_state: Mapping[str, object]) -> dict[str, object]:
    if _lexists(PROGRESS_NEXT):
        raise RuntimeError("V22 progress candidate is occupied")
    record = v22.save_anml_checkpoint(PROGRESS_NEXT, system, harness_state)
    if not 0 < PROGRESS_NEXT.stat().st_size <= CHECKPOINT_CEILING_BYTES:
        raise RuntimeError("V22 progress checkpoint exceeds its byte ceiling")
    os.replace(PROGRESS_NEXT, PROGRESS_PATH)
    return {**record, "path": str(PROGRESS_PATH), "sha256": sha256_file(PROGRESS_PATH)}


def _publish_final_checkpoint(
    system: v22.ANMLSystem, harness_state: Mapping[str, object]
) -> dict[str, object]:
    if _lexists(CHECKPOINT_PATH) or _lexists(CHECKPOINT_NEXT):
        raise RuntimeError("V22 final checkpoint identity is occupied")
    record = v22.save_anml_checkpoint(CHECKPOINT_NEXT, system, harness_state)
    if not 0 < CHECKPOINT_NEXT.stat().st_size <= CHECKPOINT_CEILING_BYTES:
        raise RuntimeError("V22 final checkpoint exceeds its byte ceiling")
    os.replace(CHECKPOINT_NEXT, CHECKPOINT_PATH)
    public_record = dict(record)
    stored_harness = public_record.pop("harness_state")
    public_record["harness_state_digest"] = v22._object_digest(
        v22._OBJECT_DIGEST_DOMAIN, stored_harness
    )
    return {
        **public_record,
        "path": str(CHECKPOINT_PATH),
        "sha256": sha256_file(CHECKPOINT_PATH),
    }


def _acquire_claim_lock():
    import fcntl

    handle = CLAIM_PATH.open("rb")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise RuntimeError("V22 claim is already executing")
    return handle


def _release_claim_lock(handle: object) -> None:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _public_evaluation(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    state = result.pop("terminal_evaluation_state")
    result["terminal_evaluation_state_digest"] = v22._object_digest(
        v22._OBJECT_DIGEST_DOMAIN, state
    )
    return result


def _terminal_report(
    *,
    claim: Mapping[str, object],
    claim_sha256: str,
    system: v22.ANMLSystem,
    harness_state: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    fit_report = harness_state.get("fit_report")
    evaluation = harness_state.get("evaluation_result")
    if not isinstance(fit_report, Mapping) or not isinstance(evaluation, Mapping):
        raise RuntimeError("V22 terminal checkpoint lacks scientific results")
    terminal_system = v22.anml_checkpoint_summary(system)
    stored_harness = terminal_system.pop("harness_state")
    if terminal_system["harness_state_digest"] != v22._object_digest(
        v22._OBJECT_DIGEST_DOMAIN, stored_harness
    ):
        raise RuntimeError("V22 terminal harness-state digest changed")
    return {
        "artifact_schema": REPORT_SCHEMA,
        "protocol_id": v22.PROTOCOL_ID,
        "classification": evaluation.get("classification"),
        "created_at_utc": utc_now(),
        "claim": {
            "path": str(CLAIM_PATH),
            "sha256": claim_sha256,
            "record": copy.deepcopy(claim),
        },
        "fit": copy.deepcopy(fit_report),
        "evaluation": _public_evaluation(evaluation),
        "checkpoint": copy.deepcopy(checkpoint),
        "terminal_system": terminal_system,
        "cumulative_elapsed_seconds": harness_state["cumulative_elapsed_seconds"],
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "publication_reserve_seconds": PUBLICATION_RESERVE_SECONDS,
        "first_result_accepted_without_tuning": True,
        "bounded_interpretation": (
            "fresh instances of the frozen public mechanisms at the 64-feature cut; "
            "not unrestricted domain transfer, consciousness, or AGI"
        ),
    }


def _execute(
    *,
    system: v22.ANMLSystem,
    claim: Mapping[str, object],
    claim_sha256: str,
    elapsed_before: float,
) -> dict[str, object]:
    started = time.monotonic()
    prior = copy.deepcopy(system.harness_state)
    fit_report = prior.get("fit_report") if isinstance(prior, Mapping) else None
    resume_evaluation = prior.get("evaluation_state") if isinstance(prior, Mapping) else None

    def cumulative() -> float:
        return elapsed_before + time.monotonic() - started

    def deadline() -> None:
        if cumulative() >= WALL_CEILING_SECONDS - PUBLICATION_RESERVE_SECONDS:
            raise RuntimeError("V22 reached its cumulative wall-time ceiling")
        if torch.cuda.max_memory_allocated(0) > v22.ALLOCATED_MEMORY_CEILING_BYTES:
            raise RuntimeError("V22 exceeded its CUDA allocation ceiling")

    def progress_callback(active: v22.ANMLSystem, info: dict[str, object]) -> None:
        deadline()
        phase = str(info["phase"])
        # The terminal fit callback precedes fit_anml's return value.  Retain the
        # prior 200-update checkpoint until the complete fit report can be bound.
        if phase == "meta_fit" and active.completed_meta_updates == v22.OUTER_UPDATES:
            return
        evaluation_state = info.get("evaluation_state") if phase == "lifetime" else None
        state = _checkpoint_state(
            phase=phase,
            claim_sha256=claim_sha256,
            cumulative_elapsed_seconds=cumulative(),
            evaluation_state=evaluation_state if isinstance(evaluation_state, Mapping) else None,
            fit_report=fit_report if isinstance(fit_report, Mapping) else None,
        )
        _publish_progress(active, state)

    if system.completed_meta_updates < v22.OUTER_UPDATES:
        fit_report = v22.fit_anml(
            system,
            progress_callback=progress_callback,
            deadline_callback=deadline,
            wall_time_limit_seconds=WALL_CEILING_SECONDS,
        )
        resume_evaluation = None
        transition = _checkpoint_state(
            phase="lifetime",
            claim_sha256=claim_sha256,
            cumulative_elapsed_seconds=cumulative(),
            evaluation_state=None,
            fit_report=fit_report,
        )
        _publish_progress(system, transition)
    elif not isinstance(fit_report, Mapping):
        raise RuntimeError("V22 completed fit checkpoint lacks its fit report")

    evaluation = v22.evaluate_anml(
        system,
        include_random_control=False,
        progress_callback=progress_callback,
        deadline_callback=deadline,
        resume_state=resume_evaluation if isinstance(resume_evaluation, Mapping) else None,
        elapsed_before_seconds=cumulative(),
        wall_time_limit_seconds=WALL_CEILING_SECONDS,
    )
    terminal_state = _checkpoint_state(
        phase="terminal",
        claim_sha256=claim_sha256,
        cumulative_elapsed_seconds=cumulative(),
        evaluation_state=evaluation["terminal_evaluation_state"],
        fit_report=fit_report,
        evaluation_result=evaluation,
    )
    checkpoint = _publish_final_checkpoint(system, terminal_state)
    deadline()
    report = _terminal_report(
        claim=claim,
        claim_sha256=claim_sha256,
        system=system,
        harness_state=terminal_state,
        checkpoint=checkpoint,
    )
    publication = _publish_json(REPORT_PATH, REPORT_TEMP, report)
    return {"report": publication, "classification": report["classification"]}


def _preserve_failure(error: BaseException, claim_sha256: str | None) -> None:
    if not CLAIM_PATH.is_file() or REPORT_PATH.exists() or FAILURE_PATH.exists():
        return
    failure = {
        "artifact_schema": FAILURE_SCHEMA,
        "protocol_id": v22.PROTOCOL_ID,
        "classification": "INVALID_NO_CLAIM",
        "created_at_utc": utc_now(),
        "claim_sha256": claim_sha256,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(limit=20),
        "outputs": {
            str(path): {
                "exists": _lexists(path),
                "sha256": sha256_file(path) if path.is_file() and not path.is_symlink() else None,
                "bytes": path.stat().st_size if path.is_file() and not path.is_symlink() else None,
            }
            for path in _output_paths()
        },
    }
    _publish_json(FAILURE_PATH, FAILURE_TEMP, failure)


def _run_fresh(expected_harness_sha256: str) -> dict[str, object]:
    preflight, system = run_preflight(expected_harness_sha256)
    identity_started = time.monotonic()
    claim, publication = _create_claim(preflight)
    claim_sha256 = str(publication["sha256"])
    initial = _checkpoint_state(
        phase="meta_fit",
        claim_sha256=claim_sha256,
        cumulative_elapsed_seconds=0.0,
        evaluation_state=None,
        fit_report=None,
    )
    _publish_progress(system, initial)
    lock = _acquire_claim_lock()
    try:
        return _execute(
            system=system,
            claim=claim,
            claim_sha256=claim_sha256,
            elapsed_before=time.monotonic() - identity_started,
        )
    finally:
        _release_claim_lock(lock)


def _run_resume(expected_harness_sha256: str) -> dict[str, object]:
    launch = verify_launch(resume=True, expected_harness_sha256=expected_harness_sha256)
    claim = launch["claim"]
    claim_sha256 = str(launch["claim_sha256"])
    if not isinstance(claim, Mapping):
        raise RuntimeError("V22 resume claim is invalid")
    lock = _acquire_claim_lock()
    started = time.monotonic()
    try:
        source = CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else PROGRESS_PATH
        system = v22.load_anml_checkpoint(source, device=DEVICE)
        state = system.harness_state
        if state.get("artifact_schema") != HARNESS_STATE_SCHEMA:
            raise RuntimeError("V22 resume harness-state schema changed")
        if state.get("claim_sha256") != claim_sha256:
            raise RuntimeError("V22 resume claim binding changed")
        elapsed = float(state.get("cumulative_elapsed_seconds", -1.0))
        elapsed += time.monotonic() - started
        if CHECKPOINT_PATH.is_file():
            if elapsed >= WALL_CEILING_SECONDS - PUBLICATION_RESERVE_SECONDS:
                raise RuntimeError("V22 resume exhausted the terminal publication reserve")
            report_state = copy.deepcopy(state)
            report_state["cumulative_elapsed_seconds"] = elapsed
            checkpoint = {
                "path": str(CHECKPOINT_PATH),
                "bytes": CHECKPOINT_PATH.stat().st_size,
                "sha256": sha256_file(CHECKPOINT_PATH),
                "completed_meta_updates": system.completed_meta_updates,
                "system_digest": v22.anml_system_digest(system),
            }
            report = _terminal_report(
                claim=claim,
                claim_sha256=claim_sha256,
                system=system,
                harness_state=report_state,
                checkpoint=checkpoint,
            )
            publication = _publish_json(REPORT_PATH, REPORT_TEMP, report)
            return {"report": publication, "classification": report["classification"]}
        return _execute(
            system=system,
            claim=claim,
            claim_sha256=claim_sha256,
            elapsed_before=elapsed,
        )
    finally:
        _release_claim_lock(lock)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "preflight", "run", "resume"))
    parser.add_argument("--expected-harness-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    claim_sha256: str | None = None
    try:
        if args.mode == "verify":
            result = verify_launch(
                resume=False, expected_harness_sha256=args.expected_harness_sha256
            )
        elif args.mode == "preflight":
            result, _ = run_preflight(args.expected_harness_sha256)
        elif args.mode == "run":
            result = _run_fresh(args.expected_harness_sha256)
        else:
            if CLAIM_PATH.is_file():
                claim_sha256 = sha256_file(CLAIM_PATH)
            result = _run_resume(args.expected_harness_sha256)
        print(json.dumps(_json_ready(result), sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        if CLAIM_PATH.is_file() and claim_sha256 is None:
            claim_sha256 = sha256_file(CLAIM_PATH)
        try:
            _preserve_failure(error, claim_sha256)
        except BaseException as preservation_error:
            print(f"V22 failure preservation also failed: {preservation_error}", file=sys.stderr)
        print(f"V22 {args.mode} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
