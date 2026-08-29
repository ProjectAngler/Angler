from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from experiments.runners import phase6_oml_relation_representation as oml


PROTOCOL_ID = "phase6.public-oml-relation-representation.v20"
ARTIFACT_SCHEMA = "angler.phase6-v20-oml-report.v1"
CLAIM_SCHEMA = "angler.phase6-v20-oml-run-claim.v1"

SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
V19_FAILURE_REPORT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.failure.json"
)
V19_RECOVERY_REPORT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context-"
    "eval-recovery-r1.json"
)
D2_REPORT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.json"
)

PREFIX = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml"
)
CLAIM_PATH = PREFIX.with_suffix(".claim.json")
PROGRESS_PATH = PREFIX.with_suffix(".progress.pt")
CHECKPOINT_PATH = PREFIX.with_suffix(".pt")
REPORT_PATH = PREFIX.with_suffix(".json")
FAILURE_PATH = PREFIX.with_suffix(".failure.json")
PROGRESS_TEMP = PROGRESS_PATH.with_suffix(PROGRESS_PATH.suffix + ".tmp")
CHECKPOINT_TEMP = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
# The runner uses its own one-level-deeper atomic temporary around the path it
# receives from this harness.  Track those exact names so crash continuation
# cannot silently leave an unaccounted checkpoint artifact.
PROGRESS_RUNNER_TEMP = PROGRESS_TEMP.with_name(PROGRESS_TEMP.name + ".tmp")
CHECKPOINT_RUNNER_TEMP = CHECKPOINT_TEMP.with_name(CHECKPOINT_TEMP.name + ".tmp")
REPORT_TEMP = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
FAILURE_TEMP = FAILURE_PATH.with_suffix(FAILURE_PATH.suffix + ".tmp")

EXPECTED_ARTIFACT_HASHES = {
    str(SOURCE_CHECKPOINT): (
        "10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F"
    ),
    str(V19_FAILURE_REPORT): (
        "C297B861A26FF53EA489E70E537F6EECA7C20B54769394C85C042147838116EE"
    ),
    str(V19_RECOVERY_REPORT): (
        "55592E9861EC16301603D0CD7BB2A104E596BAAA97BDC65D50DCC517951A0800"
    ),
    str(D2_REPORT): (
        "69D56232A4E70720AFD8428208A0F5ED4B4C2C75AED3D71DFF678F5BA10E6C9F"
    ),
}

# The V20 runner and focused test values are filled only after their independent
# implementation review.  verify_launch deliberately refuses a semantic claim
# while either value remains pending.
EXPECTED_REPOSITORY_HASHES = {
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V20-OML-001.md": (
        "BB2CCDDB80B25ACD5B79CB4DDC52F347ED77C8F843D972B7D4049C2B4546F257"
    ),
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_cross_variation_plasticity_v16.py": (
        "EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB"
    ),
    "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
        "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
    ),
    "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py": (
        "E9656044749805E626C2DD443EBB5C34E95656CE11128AB5B4D6A3425C927517"
    ),
    "experiments/runners/phase6_oml_relation_representation.py": (
        "6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3"
    ),
    "tests/unit/experiments/test_phase6_oml_relation_representation.py": (
        "5605100352C092C901AF154BEED2B522211EFD0B29748C69BA5CFAC6BE2445DF"
    ),
}

EXPECTED_PLAN_DIGEST = (
    "sha256:c9a60a2d19ad0e728b8cdb980eddb16779b32f3af45fbd071003969f5f8cee1d"
)
EXPECTED_OUTER_UPDATES = 240
PROGRESS_INTERVAL = 40
MANUAL_SEED = 2_026_082_901
MAX_ALLOCATED_BYTES = 12 * 1024**3
MAX_CHECKPOINT_BYTES = 16 * 1024**2
MAX_REPORT_BYTES = 4 * 1024**2
MAX_SEMANTIC_SECONDS = 150 * 60.0
EQUIVALENCE_TOLERANCE = 1.0e-6

VALID_CLASSIFICATIONS = {
    "OML_V19_HARMONIZED_ADVANCEMENT",
    "OML_COMPONENT_SUPPORTED_NOT_INTEGRATED",
    "OML_CROSS_MECHANISM_ADVANCEMENT",
    "SECOND_ORDER_OML_CREDIT_SUPPORTED",
    "FAST_ADAPTATION_SUPPORTED_OML_ATTRIBUTION_NOT_ESTABLISHED",
    "OML_NOT_SUPPORTED",
}

_CUDA_CONFIGURED = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_claim_created_utc(claim_record: Mapping[str, object]) -> datetime:
    raw = claim_record.get("created_utc")
    if not isinstance(raw, str):
        raise RuntimeError("V20 claim created_utc is missing or invalid")
    try:
        created = datetime.fromisoformat(raw)
    except ValueError as error:
        raise RuntimeError("V20 claim created_utc is not ISO-8601") from error
    if created.tzinfo is None or created.utcoffset() is None:
        raise RuntimeError("V20 claim created_utc is not timezone-aware")
    return created.astimezone(timezone.utc)


def _identity_wall_snapshot(
    claim_record: Mapping[str, object],
    *,
    invocation_started: float | None = None,
    invocation_start_identity_age: float | None = None,
    enforce_remaining: bool = True,
) -> dict[str, object]:
    """Return the immutable-claim wall budget without granting resume time."""

    created = _parse_claim_created_utc(claim_record)
    now = datetime.now(timezone.utc)
    observed_age = (now - created).total_seconds()
    if not math.isfinite(observed_age) or observed_age < 0.0:
        raise RuntimeError("V20 identity clock is before immutable claim creation")
    invocation_elapsed = (
        0.0 if invocation_started is None else time.perf_counter() - invocation_started
    )
    if not math.isfinite(invocation_elapsed) or invocation_elapsed < 0.0:
        raise RuntimeError("V20 invocation monotonic clock is invalid")
    effective_age = observed_age
    if invocation_start_identity_age is not None:
        if (
            not math.isfinite(invocation_start_identity_age)
            or invocation_start_identity_age < 0.0
        ):
            raise RuntimeError("V20 invocation identity age is invalid")
        effective_age = max(
            effective_age,
            invocation_start_identity_age + invocation_elapsed,
        )
    remaining = MAX_SEMANTIC_SECONDS - effective_age
    if enforce_remaining and remaining <= 0.0:
        raise TimeoutError("V20 immutable claim exhausted its 150-minute wall budget")
    deadline = created.timestamp() + MAX_SEMANTIC_SECONDS
    return {
        "claim_created_utc": created.isoformat(),
        "identity_deadline_utc": datetime.fromtimestamp(
            deadline,
            tz=timezone.utc,
        ).isoformat(),
        "observed_wall_age_seconds": observed_age,
        "identity_wall_elapsed_seconds": effective_age,
        "invocation_elapsed_seconds": invocation_elapsed,
        "remaining_identity_wall_seconds": max(0.0, remaining),
    }


def json_ready(value: object) -> object:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("V20 report contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"V20 report value is not JSON-safe: {type(value).__name__}")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            json_ready(value),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json_atomic(
    path: Path,
    temporary: Path,
    value: Mapping[str, object],
    *,
    byte_ceiling: int,
) -> dict[str, object]:
    payload = _json_bytes(value)
    if len(payload) > byte_ceiling:
        raise RuntimeError(
            f"V20 JSON exceeds its byte ceiling: {len(payload)} > {byte_ceiling}"
        )
    if path.exists() or temporary.exists():
        raise FileExistsError(f"V20 JSON identity is occupied: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"V20 expected a JSON object: {path}")
    return value


def _assert_finalized_hash_constants() -> None:
    pending = tuple(
        name
        for name, expected in EXPECTED_REPOSITORY_HASHES.items()
        if expected == "PENDING"
    )
    if pending or EXPECTED_PLAN_DIGEST == "PENDING":
        raise RuntimeError(
            "V20 launch constants are not finalized: "
            f"paths={pending}, plan_digest={EXPECTED_PLAN_DIGEST}"
        )


def _validate_source_reports() -> dict[str, object]:
    failure = _read_json(V19_FAILURE_REPORT)
    recovery = _read_json(V19_RECOVERY_REPORT)
    d2 = _read_json(D2_REPORT)
    if failure.get("classification") != "HARNESS_ERROR_PRESERVED":
        raise RuntimeError("V20 source V19 failure classification changed")
    if (
        recovery.get("classification") != "PAIRED_GRAPH_COMPONENT_SUPPORTED"
        or recovery.get("component_supported") is not True
    ):
        raise RuntimeError("V20 accepted V19 recovery classification changed")
    if d2.get("classification") != "REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED":
        raise RuntimeError("V20 immutable D2 classification changed")
    return {
        "v19_failure_classification": failure.get("classification"),
        "v19_recovery_classification": recovery.get("classification"),
        "v19_component_supported": recovery.get("component_supported"),
        "d2_classification": d2.get("classification"),
    }


def _output_state() -> dict[str, bool]:
    return {
        str(path): path.exists()
        for path in (
            CLAIM_PATH,
            PROGRESS_PATH,
            CHECKPOINT_PATH,
            REPORT_PATH,
            FAILURE_PATH,
            PROGRESS_TEMP,
            CHECKPOINT_TEMP,
            PROGRESS_RUNNER_TEMP,
            CHECKPOINT_RUNNER_TEMP,
            REPORT_TEMP,
            FAILURE_TEMP,
        )
    }


def verify_launch(*, resume: bool = False) -> dict[str, object]:
    """Verify frozen bytes and output state without creating a run claim."""

    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    _assert_finalized_hash_constants()
    observed_repository = {
        relative: sha256_file(ROOT / relative)
        for relative in EXPECTED_REPOSITORY_HASHES
    }
    if observed_repository != EXPECTED_REPOSITORY_HASHES:
        raise RuntimeError("V20 frozen repository launch bytes changed")
    observed_artifacts = {
        path: sha256_file(Path(path)) for path in EXPECTED_ARTIFACT_HASHES
    }
    if observed_artifacts != EXPECTED_ARTIFACT_HASHES:
        raise RuntimeError("V20 frozen source artifact bytes changed")
    source_report_state = _validate_source_reports()
    dependency_report = oml.verify_oml_dependencies(
        SOURCE_CHECKPOINT,
        d2_path=D2_REPORT,
    )
    plan = oml.oml_fit_plan()
    plan_digest = oml.oml_plan_digest()
    if (
        plan.get("protocol_id") != PROTOCOL_ID
        or plan_digest != EXPECTED_PLAN_DIGEST
        or plan.get("plan_digest", plan_digest) != plan_digest
        or plan.get("outer_updates", EXPECTED_OUTER_UPDATES)
        != EXPECTED_OUTER_UPDATES
    ):
        raise RuntimeError("V20 frozen plan identity changed")

    state = _output_state()
    terminal = state[str(REPORT_PATH)] or state[str(FAILURE_PATH)]
    if resume:
        if not state[str(CLAIM_PATH)] or terminal:
            raise RuntimeError("V20 resume requires a claim and no terminal artifact")
        # Stale temporary files are handled only by the explicit resume path,
        # after the immutable claim has been validated.
    elif any(state.values()):
        occupied = tuple(path for path, exists in state.items() if exists)
        raise RuntimeError(f"V20 one-shot identity is occupied: {occupied}")
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": plan_digest,
        "repository_hashes": observed_repository,
        "artifact_hashes": observed_artifacts,
        "dependency_report": dependency_report,
        "source_report_state": source_report_state,
        "output_state": state,
        "resume": resume,
    }


def _configure_cuda() -> dict[str, object]:
    global _CUDA_CONFIGURED
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V20 requires exactly one visible CUDA device")
    if not _CUDA_CONFIGURED:
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(MANUAL_SEED)
        torch.cuda.manual_seed_all(MANUAL_SEED)
        _CUDA_CONFIGURED = True
    torch.cuda.set_device(0)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.is_autocast_enabled()
    ):
        raise RuntimeError("V20 CUDA numerical mode is not frozen FP32")
    device_properties = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "device_capability": tuple(torch.cuda.get_device_capability(0)),
        "device_total_memory_bytes": int(device_properties.total_memory),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "autocast_enabled": torch.is_autocast_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "manual_seed": MANUAL_SEED,
    }


def _validate_preflight_mode_metrics(
    result: Mapping[str, object],
    *,
    observed_peak_allocated_bytes: int,
) -> dict[str, object]:
    try:
        raw_memory = (
            result["full_mode_maximum_allocated_bytes"],
            result["split_mode_maximum_allocated_bytes"],
            result["selected_mode_maximum_allocated_bytes"],
            result["maximum_allocated_bytes"],
            result["allocated_memory_ceiling_bytes"],
        )
    except KeyError as error:
        raise RuntimeError("V20 preflight per-mode memory metrics are invalid") from error
    if any(type(value) is not int for value in raw_memory):
        raise RuntimeError("V20 preflight per-mode memory metrics are not integers")
    (
        full_allocated,
        split_allocated,
        selected_allocated,
        aggregate_allocated,
        reported_ceiling,
    ) = raw_memory
    if (
        min(
            full_allocated,
            split_allocated,
            selected_allocated,
            aggregate_allocated,
            observed_peak_allocated_bytes,
        )
        < 0
        or reported_ceiling != MAX_ALLOCATED_BYTES
        or aggregate_allocated != max(full_allocated, split_allocated)
        or observed_peak_allocated_bytes != split_allocated
    ):
        raise RuntimeError("V20 preflight memory accounting changed")
    if full_allocated <= MAX_ALLOCATED_BYTES:
        expected_mode = "full"
        expected_selected_allocated = full_allocated
    elif split_allocated <= MAX_ALLOCATED_BYTES:
        expected_mode = "split_4_plus_4"
        expected_selected_allocated = split_allocated
    else:
        raise RuntimeError("V20 both predeclared outer modes exceed the memory ceiling")
    selected_mode = result.get("selected_outer_mode")
    if (
        selected_mode != expected_mode
        or selected_allocated != expected_selected_allocated
        or selected_allocated > MAX_ALLOCATED_BYTES
    ):
        raise RuntimeError("V20 preflight outer-mode selection changed")
    return {
        "selected_outer_mode": selected_mode,
        "full_mode_maximum_allocated_bytes": full_allocated,
        "split_mode_maximum_allocated_bytes": split_allocated,
        "selected_mode_maximum_allocated_bytes": selected_allocated,
        "maximum_allocated_bytes": aggregate_allocated,
    }


def run_cuda_preflight() -> dict[str, object]:
    """Run the synthetic full/split check without writing or claiming outputs."""

    _assert_finalized_hash_constants()
    if oml.oml_plan_digest() != EXPECTED_PLAN_DIGEST:
        raise RuntimeError("V20 synthetic CUDA preflight plan identity changed")
    before = _output_state()
    cuda = _configure_cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    result = oml.synthetic_cuda_preflight(device="cuda:0")
    torch.cuda.synchronize(0)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    after = _output_state()
    if after != before:
        raise RuntimeError("V20 synthetic CUDA preflight created an output artifact")
    if not isinstance(result, Mapping):
        raise RuntimeError("V20 synthetic CUDA preflight returned no mapping")
    objective_delta = float(result.get("full_split_objective_abs_delta", math.inf))
    gradient_delta = float(
        result.get("full_split_max_gradient_abs_delta", math.inf)
    )
    memory = _validate_preflight_mode_metrics(
        result,
        observed_peak_allocated_bytes=peak_allocated,
    )
    selected = memory["selected_outer_mode"]
    if (
        not math.isfinite(objective_delta)
        or not math.isfinite(gradient_delta)
        or objective_delta > EQUIVALENCE_TOLERANCE
        or gradient_delta > EQUIVALENCE_TOLERANCE
    ):
        raise RuntimeError("V20 synthetic CUDA preflight failed its frozen envelope")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "plan_digest": oml.oml_plan_digest(),
        "cuda": cuda,
        "result": dict(result),
        "selected_outer_mode": selected,
        "full_split_objective_abs_delta": objective_delta,
        "full_split_max_gradient_abs_delta": gradient_delta,
        "mode_memory": memory,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "elapsed_seconds": time.perf_counter() - started,
        "semantic_streams_generated": 0,
        "semantic_fit_performed": False,
        "claim_created": False,
    }


def create_run_claim(
    launch: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    if CLAIM_PATH.exists():
        raise FileExistsError("V20 run claim already exists")
    claim = {
        "artifact_schema": CLAIM_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "process_id": os.getpid(),
        "one_shot": True,
        "resume_same_identity_only": True,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "fit_identity": {
            "outer_updates_per_arm": EXPECTED_OUTER_UPDATES,
            "progress_interval": PROGRESS_INTERVAL,
            "outer_mode": preflight["selected_outer_mode"],
            "device": "cuda:0",
            "numerical_mode": "fp32_no_tf32_no_autocast_exact_second_order",
        },
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "repository_hashes": dict(launch["repository_hashes"]),
        "artifact_hashes": dict(launch["artifact_hashes"]),
        "synthetic_cuda_preflight": dict(preflight),
        "ceilings": {
            "allocated_memory_bytes": MAX_ALLOCATED_BYTES,
            "semantic_wall_seconds": MAX_SEMANTIC_SECONDS,
            "checkpoint_bytes": MAX_CHECKPOINT_BYTES,
            "terminal_json_bytes": MAX_REPORT_BYTES,
        },
    }
    CLAIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(claim)
    with CLAIM_PATH.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(CLAIM_PATH.parent)
    return {
        "path": str(CLAIM_PATH),
        "sha256": sha256_file(CLAIM_PATH),
        "bytes": CLAIM_PATH.stat().st_size,
        "record": claim,
    }


def _validate_existing_claim(launch: Mapping[str, object]) -> dict[str, object]:
    claim = _read_json(CLAIM_PATH)
    if (
        claim.get("artifact_schema") != CLAIM_SCHEMA
        or claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or claim.get("one_shot") is not True
        or claim.get("resume_same_identity_only") is not True
        or claim.get("harness_sha256") != sha256_file(Path(__file__).resolve())
        or claim.get("repository_hashes") != launch["repository_hashes"]
        or claim.get("artifact_hashes") != launch["artifact_hashes"]
    ):
        raise RuntimeError("V20 existing claim does not bind this exact identity")
    fit_identity = claim.get("fit_identity")
    preflight = claim.get("synthetic_cuda_preflight")
    ceilings = claim.get("ceilings")
    if (
        not isinstance(fit_identity, Mapping)
        or fit_identity.get("outer_updates_per_arm") != EXPECTED_OUTER_UPDATES
        or fit_identity.get("progress_interval") != PROGRESS_INTERVAL
        or fit_identity.get("outer_mode") not in {"full", "split_4_plus_4"}
        or not isinstance(preflight, Mapping)
        or preflight.get("status") != "PASS"
        or preflight.get("semantic_fit_performed") is not False
        or preflight.get("claim_created") is not False
        or not isinstance(ceilings, Mapping)
        or ceilings.get("semantic_wall_seconds") != MAX_SEMANTIC_SECONDS
        or ceilings.get("allocated_memory_bytes") != MAX_ALLOCATED_BYTES
        or ceilings.get("checkpoint_bytes") != MAX_CHECKPOINT_BYTES
        or ceilings.get("terminal_json_bytes") != MAX_REPORT_BYTES
    ):
        raise RuntimeError("V20 existing claim lost its frozen preflight or fit identity")
    preflight_result = preflight.get("result")
    recorded_peak = preflight.get("peak_allocated_bytes")
    if not isinstance(preflight_result, Mapping) or type(recorded_peak) is not int:
        raise RuntimeError("V20 existing claim preflight metrics are invalid")
    memory = _validate_preflight_mode_metrics(
        preflight_result,
        observed_peak_allocated_bytes=recorded_peak,
    )
    if (
        preflight.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or preflight.get("selected_outer_mode") != memory["selected_outer_mode"]
        or preflight.get("mode_memory") != memory
        or fit_identity.get("outer_mode") != memory["selected_outer_mode"]
    ):
        raise RuntimeError("V20 existing claim preflight binding changed")
    _identity_wall_snapshot(claim, enforce_remaining=True)
    return {
        "path": str(CLAIM_PATH),
        "sha256": sha256_file(CLAIM_PATH),
        "bytes": CLAIM_PATH.stat().st_size,
        "record": claim,
    }


def _clear_stale_resume_temporaries() -> tuple[str, ...]:
    removed = []
    expected_parent = PREFIX.parent.resolve()
    for path in (
        PROGRESS_TEMP,
        CHECKPOINT_TEMP,
        PROGRESS_RUNNER_TEMP,
        CHECKPOINT_RUNNER_TEMP,
        REPORT_TEMP,
        FAILURE_TEMP,
    ):
        if not path.exists():
            continue
        if path.resolve().parent != expected_parent or not path.is_file():
            raise RuntimeError(f"V20 refuses an unsafe stale temporary: {path}")
        path.unlink()
        removed.append(str(path))
    return tuple(removed)


def _acquire_claim_execution_lock():
    """Hold the claim inode so two resume processes cannot run one identity."""

    import fcntl

    handle = CLAIM_PATH.open("rb")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def _release_claim_execution_lock(handle: object) -> None:
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _checkpoint_record(path: Path, system: object) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"V20 checkpoint is absent: {path}")
    size = path.stat().st_size
    if not 0 < size <= MAX_CHECKPOINT_BYTES:
        raise RuntimeError(f"V20 checkpoint exceeds its byte ceiling: {size}")
    outer_mode = getattr(system, "outer_mode", None)
    if outer_mode not in {"full", "split_4_plus_4"}:
        raise RuntimeError("V20 checkpoint lacks its bound outer mode")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": size,
        "completed_updates": int(system.completed_updates),
        "outer_mode": outer_mode,
        "system_digest": oml.oml_system_digest(system),
    }


def _publish_progress(system: object, event: Mapping[str, object]) -> dict[str, object]:
    completed = int(system.completed_updates)
    if (
        completed <= 0
        or completed > EXPECTED_OUTER_UPDATES
        or completed % PROGRESS_INTERVAL != 0
        or int(event.get("completed_updates", completed)) != completed
    ):
        raise RuntimeError("V20 progress callback escaped a frozen boundary")
    if PROGRESS_TEMP.exists():
        raise RuntimeError("V20 progress temporary already exists")
    oml.save_oml_checkpoint(PROGRESS_TEMP, system)
    _fsync_file(PROGRESS_TEMP)
    record = _checkpoint_record(PROGRESS_TEMP, system)
    os.replace(PROGRESS_TEMP, PROGRESS_PATH)
    _fsync_directory(PROGRESS_PATH.parent)
    record["path"] = str(PROGRESS_PATH)
    peak = int(torch.cuda.max_memory_allocated(0))
    if peak > MAX_ALLOCATED_BYTES:
        raise RuntimeError("V20 allocated-memory ceiling exceeded at progress boundary")
    print(
        json.dumps(
            {
                "event": "V20_PROGRESS_CHECKPOINT",
                "protocol_id": PROTOCOL_ID,
                "completed_updates": completed,
                "system_digest": record["system_digest"],
                "checkpoint_sha256": record["sha256"],
                "peak_allocated_bytes": peak,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def _validate_fit_report(
    report: object,
    *,
    start_update: int,
    outer_mode: str,
    system: object,
) -> dict[str, object]:
    if not isinstance(report, dict):
        raise RuntimeError("V20 fit returned no report mapping")
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or report.get("start_update") != start_update
        or report.get("terminal_update") != EXPECTED_OUTER_UPDATES
        or report.get("outer_mode") != outer_mode
        or report.get("paired_arm_updates")
        != EXPECTED_OUTER_UPDATES - start_update
        or report.get("unique_stream_uses")
        != 16 * (EXPECTED_OUTER_UPDATES - start_update)
        or report.get("inner_loss_uses_per_arm")
        != 8 * (EXPECTED_OUTER_UPDATES - start_update)
        or int(system.completed_updates) != EXPECTED_OUTER_UPDATES
        or getattr(system, "outer_mode", None) != outer_mode
        or report.get("system_digest") != oml.oml_system_digest(system)
    ):
        raise RuntimeError("V20 fit report changed the frozen semantic identity")
    return report


def _validate_evaluation(report: object, system: object) -> dict[str, object]:
    if not isinstance(report, dict):
        raise RuntimeError("V20 evaluation returned no report mapping")
    classification = report.get("classification")
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or report.get("source_checkpoint_sha256")
        != EXPECTED_ARTIFACT_HASHES[str(SOURCE_CHECKPOINT)]
        or report.get("accepted_v19_recovery_report_sha256")
        != EXPECTED_ARTIFACT_HASHES[str(V19_RECOVERY_REPORT)]
        or report.get("first_result_accepted_without_tuning") is not True
        or classification not in VALID_CLASSIFICATIONS
        or report.get("system_digest") != oml.oml_system_digest(system)
    ):
        raise RuntimeError("V20 evaluation identity or classification changed")
    for key in ("families", "comparisons", "gates", "d2_binding"):
        if key not in report:
            raise RuntimeError(f"V20 evaluation omitted {key}")
    if classification == "OML_COMPONENT_SUPPORTED_NOT_INTEGRATED":
        # The immutable D2 result fixes this pilot's ANML trigger false.
        raise RuntimeError("V20 evaluation contradicted the immutable false D2 trigger")
    d2_binding = report["d2_binding"]
    gates = report["gates"]
    if (
        not isinstance(d2_binding, Mapping)
        or d2_binding.get("sha256") != EXPECTED_ARTIFACT_HASHES[str(D2_REPORT)]
        or d2_binding.get("classification")
        != "REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED"
        or d2_binding.get("same_module_overlap") is not False
        or not isinstance(gates, Mapping)
        or gates.get("d2_same_module_overlap") is not False
        or gates.get("anml_trigger") is not False
    ):
        raise RuntimeError("V20 evaluation contradicted the immutable D2 denial")
    return report


def _load_resume_system(outer_mode: str) -> tuple[object, str]:
    if CHECKPOINT_PATH.is_file():
        system = oml.load_oml_checkpoint(
            CHECKPOINT_PATH,
            SOURCE_CHECKPOINT,
            device="cuda:0",
        )
        oml.bind_oml_outer_mode(system, outer_mode)
        if int(system.completed_updates) != EXPECTED_OUTER_UPDATES:
            raise RuntimeError("V20 published final checkpoint is not terminal")
        if getattr(system, "outer_mode", None) != outer_mode:
            raise RuntimeError("V20 final checkpoint outer mode changed")
        return system, "final"
    if PROGRESS_PATH.is_file():
        system = oml.load_oml_checkpoint(
            PROGRESS_PATH,
            SOURCE_CHECKPOINT,
            device="cuda:0",
        )
        oml.bind_oml_outer_mode(system, outer_mode)
        completed = int(system.completed_updates)
        if (
            completed <= 0
            or completed > EXPECTED_OUTER_UPDATES
            or completed % PROGRESS_INTERVAL != 0
        ):
            raise RuntimeError("V20 progress checkpoint is outside a frozen boundary")
        if getattr(system, "outer_mode", None) != outer_mode:
            raise RuntimeError("V20 progress checkpoint outer mode changed")
        return system, "progress"
    system = oml.build_oml_system(SOURCE_CHECKPOINT, device="cuda:0")
    oml.bind_oml_outer_mode(system, outer_mode)
    return system, "source"


def _save_reload_and_publish_final(
    system: object,
    outer_mode: str,
) -> tuple[object, dict[str, object]]:
    if CHECKPOINT_PATH.exists() or CHECKPOINT_TEMP.exists():
        raise RuntimeError("V20 final checkpoint identity is occupied")
    expected_digest = oml.oml_system_digest(system)
    oml.save_oml_checkpoint(CHECKPOINT_TEMP, system)
    _fsync_file(CHECKPOINT_TEMP)
    preliminary = _checkpoint_record(CHECKPOINT_TEMP, system)
    del system
    gc.collect()
    torch.cuda.empty_cache()
    restored = oml.load_oml_checkpoint(
        CHECKPOINT_TEMP,
        SOURCE_CHECKPOINT,
        device="cuda:0",
    )
    oml.bind_oml_outer_mode(restored, outer_mode)
    if (
        int(restored.completed_updates) != EXPECTED_OUTER_UPDATES
        or getattr(restored, "outer_mode", None) != outer_mode
        or oml.oml_system_digest(restored) != expected_digest
    ):
        raise RuntimeError("V20 strict final checkpoint reload changed the system")
    os.replace(CHECKPOINT_TEMP, CHECKPOINT_PATH)
    _fsync_directory(CHECKPOINT_PATH.parent)
    preliminary.update(
        {
            "path": str(CHECKPOINT_PATH),
            "strict_reload_verified": True,
            "system_digest": expected_digest,
        }
    )
    return restored, preliminary


def _environment_record(cuda: Mapping[str, object]) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "cuda": dict(cuda),
    }


def _run_semantic(
    launch: Mapping[str, object],
    claim: Mapping[str, object],
    *,
    resume: bool,
    stale_temporaries_removed: Sequence[str],
    invocation_started: float,
) -> int:
    claim_record = claim.get("record")
    if not isinstance(claim_record, Mapping):
        raise RuntimeError("V20 semantic run has no immutable claim record")
    identity_start = _identity_wall_snapshot(
        claim_record,
        invocation_started=invocation_started,
        enforce_remaining=True,
    )
    invocation_start_identity_age = float(
        identity_start["identity_wall_elapsed_seconds"]
    )

    def identity_wall() -> dict[str, object]:
        return _identity_wall_snapshot(
            claim_record,
            invocation_started=invocation_started,
            invocation_start_identity_age=invocation_start_identity_age,
            enforce_remaining=True,
        )

    cuda = _configure_cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started_utc = utc_now()
    fit_identity = claim["record"]["fit_identity"]
    assert isinstance(fit_identity, Mapping)
    outer_mode = str(fit_identity["outer_mode"])
    progress_records: list[dict[str, object]] = []

    if resume:
        system, resume_source = _load_resume_system(outer_mode)
    else:
        system = oml.build_oml_system(SOURCE_CHECKPOINT, device="cuda:0")
        oml.bind_oml_outer_mode(system, outer_mode)
        resume_source = "source"
    start_update = int(system.completed_updates)
    if resume_source == "final":
        fit_report: dict[str, object] = {
            "protocol_id": PROTOCOL_ID,
            "plan_digest": EXPECTED_PLAN_DIGEST,
            "start_update": EXPECTED_OUTER_UPDATES,
            "terminal_update": EXPECTED_OUTER_UPDATES,
            "system_digest": oml.oml_system_digest(system),
            "resume_skipped_completed_fit": True,
        }
        checkpoint = _checkpoint_record(CHECKPOINT_PATH, system)
        checkpoint["strict_reload_verified"] = True
    else:
        def progress_callback(
            callback_system: object,
            event: Mapping[str, object],
        ) -> None:
            identity_wall()
            progress_records.append(_publish_progress(callback_system, event))

        remaining = float(identity_wall()["remaining_identity_wall_seconds"])
        if remaining <= 0.0:
            raise TimeoutError("V20 immutable claim has no wall-time budget remaining")
        raw_fit = oml.fit_oml(
            system,
            outer_mode=outer_mode,
            progress_callback=progress_callback,
            wall_time_limit_seconds=remaining,
        )
        fit_report = _validate_fit_report(
            raw_fit,
            start_update=start_update,
            outer_mode=outer_mode,
            system=system,
        )
        system, checkpoint = _save_reload_and_publish_final(system, outer_mode)

    identity_wall()
    evaluation_started = time.perf_counter()
    evaluation = _validate_evaluation(
        oml.evaluate_oml(system, d2_path=D2_REPORT),
        system,
    )
    torch.cuda.synchronize(0)
    evaluation_seconds = time.perf_counter() - evaluation_started
    identity_end = identity_wall()
    invocation_seconds = time.perf_counter() - invocation_started
    identity_wall_seconds = float(identity_end["identity_wall_elapsed_seconds"])
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if peak_allocated > MAX_ALLOCATED_BYTES:
        raise RuntimeError("V20 semantic run exceeded the allocated-memory ceiling")
    terminal_digest = oml.oml_system_digest(system)
    if evaluation["system_digest"] != terminal_digest:
        raise RuntimeError("V20 evaluation changed or misreported the terminal system")

    classification = str(evaluation["classification"])
    report = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "classification": classification,
        "passed": classification == "OML_V19_HARMONIZED_ADVANCEMENT",
        "run_identity": {
            "one_shot": True,
            "resume_used": resume,
            "resume_source": resume_source,
            "start_update": start_update,
            "terminal_update": EXPECTED_OUTER_UPDATES,
            "outer_mode": outer_mode,
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "identity_wall": identity_end,
            "claim": {
                "path": claim["path"],
                "sha256": claim["sha256"],
                "bytes": claim["bytes"],
            },
            "stale_temporaries_removed_on_resume": tuple(stale_temporaries_removed),
        },
        "source_integrity": {
            "repository_hashes": dict(launch["repository_hashes"]),
            "artifact_hashes": dict(launch["artifact_hashes"]),
            "dependency_report": launch["dependency_report"],
            "source_report_state": launch["source_report_state"],
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "plan_digest": EXPECTED_PLAN_DIGEST,
            "terminal_system_digest": terminal_digest,
        },
        "environment": _environment_record(cuda),
        "resources": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_memory_ceiling_bytes": MAX_ALLOCATED_BYTES,
            "semantic_wall_ceiling_seconds": MAX_SEMANTIC_SECONDS,
        },
        "timings_seconds": {
            "evaluation": evaluation_seconds,
            "invocation_elapsed": invocation_seconds,
            "identity_wall_elapsed": identity_wall_seconds,
        },
        "fit_report": fit_report,
        "progress_checkpoints": tuple(progress_records),
        "evaluation": evaluation,
        "checkpoint": checkpoint,
        "scope_and_effects": {
            "gpu_used": True,
            "network_used": False,
            "package_install_used": False,
            "model_or_llm_used": False,
            "public_synthetic_train_partition_only": True,
            "replay_used": False,
            "deterministic_solver_used": False,
            "identity_inputs_used": False,
            "external_effects": False,
            "deployment_used": False,
            "promoted_state_changed": False,
        },
    }
    if FAILURE_PATH.exists():
        raise RuntimeError("V20 failure terminal already exists")
    publication = _write_json_atomic(
        REPORT_PATH,
        REPORT_TEMP,
        report,
        byte_ceiling=MAX_REPORT_BYTES,
    )
    print(
        json.dumps(
            {
                "event": "V20_OML_COMPLETE",
                "protocol_id": PROTOCOL_ID,
                "classification": classification,
                "report": publication,
                "checkpoint": checkpoint,
                "invocation_elapsed_seconds": invocation_seconds,
                "identity_wall_elapsed_seconds": identity_wall_seconds,
                "peak_allocated_bytes": peak_allocated,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _preserve_failure(
    error: BaseException,
    *,
    launch: Mapping[str, object],
    claim: Mapping[str, object],
    resume: bool,
    invocation_started: float,
) -> dict[str, object] | None:
    if REPORT_PATH.exists() or FAILURE_PATH.exists():
        return None
    for temporary in (REPORT_TEMP, FAILURE_TEMP):
        if temporary.exists() and temporary.is_file():
            temporary.unlink()
    claim_record = claim.get("record")
    identity_wall = None
    if isinstance(claim_record, Mapping):
        try:
            identity_wall = _identity_wall_snapshot(
                claim_record,
                invocation_started=invocation_started,
                enforce_remaining=False,
            )
        except BaseException as clock_error:
            identity_wall = {
                "status": "INVALID",
                "error_type": type(clock_error).__name__,
                "error": str(clock_error)[:16_384],
            }
    invocation_elapsed = time.perf_counter() - invocation_started
    failure = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "classification": "HARNESS_ERROR_PRESERVED",
        "passed": False,
        "failed_utc": utc_now(),
        "resume_used": resume,
        "claim": {
            "path": claim["path"],
            "sha256": claim["sha256"],
            "bytes": claim["bytes"],
        },
        "source_integrity": {
            "repository_hashes": dict(launch["repository_hashes"]),
            "artifact_hashes": dict(launch["artifact_hashes"]),
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "plan_digest": EXPECTED_PLAN_DIGEST,
        },
        "failure": {
            "type": type(error).__name__,
            "message": str(error)[:16_384],
            "traceback": traceback.format_exc()[-262_144:],
        },
        "preserved_state": {
            "progress_exists": PROGRESS_PATH.is_file(),
            "progress_sha256": (
                sha256_file(PROGRESS_PATH) if PROGRESS_PATH.is_file() else None
            ),
            "checkpoint_exists": CHECKPOINT_PATH.is_file(),
            "checkpoint_sha256": (
                sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
            ),
        },
        "timings_seconds": {
            "invocation_elapsed": invocation_elapsed,
            "identity_wall_elapsed": (
                identity_wall.get("identity_wall_elapsed_seconds")
                if isinstance(identity_wall, Mapping)
                else None
            ),
        },
        "identity_wall": identity_wall,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_initialized": torch.cuda.is_initialized(),
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(0))
                if torch.cuda.is_initialized()
                else 0
            ),
        },
    }
    return _write_json_atomic(
        FAILURE_PATH,
        FAILURE_TEMP,
        failure,
        byte_ceiling=MAX_REPORT_BYTES,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Angler V20 OML one-shot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify:
        result = verify_launch(resume=False)
        print(json.dumps(json_ready(result), sort_keys=True), flush=True)
        return 0
    if args.preflight:
        launch = verify_launch(resume=False)
        result = run_cuda_preflight()
        print(
            json.dumps(
                json_ready({"launch": launch, "preflight": result}),
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    resume = bool(args.resume)
    launch = verify_launch(resume=resume)
    stale_temporaries_removed: tuple[str, ...] = ()
    if resume:
        claim = _validate_existing_claim(launch)
    else:
        preflight = run_cuda_preflight()
        claim = create_run_claim(launch, preflight)
    try:
        claim_lock = _acquire_claim_execution_lock()
    except BlockingIOError:
        print(
            json.dumps(
                {
                    "event": "V20_OML_ALREADY_ACTIVE",
                    "protocol_id": PROTOCOL_ID,
                    "claim": claim["path"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    invocation_started = time.perf_counter()
    try:
        try:
            if resume:
                stale_temporaries_removed = _clear_stale_resume_temporaries()
            return _run_semantic(
                launch,
                claim,
                resume=resume,
                stale_temporaries_removed=stale_temporaries_removed,
                invocation_started=invocation_started,
            )
        except BaseException as error:
            failure_record = _preserve_failure(
                error,
                launch=launch,
                claim=claim,
                resume=resume,
                invocation_started=invocation_started,
            )
            print(
                json.dumps(
                    {
                        "event": "V20_OML_FAILED",
                        "protocol_id": PROTOCOL_ID,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failure": failure_record,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 1
    finally:
        _release_claim_execution_lock(claim_lock)


if __name__ == "__main__":
    raise SystemExit(main())
