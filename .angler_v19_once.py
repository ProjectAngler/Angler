from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# V19's semantic identity is CPU-only. Hide CUDA before importing torch so the
# one-shot harness cannot initialize or select a GPU device accidentally.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.runners import phase6_v12_champion_paired_graph_context as v19


PROTOCOL_ID = "phase6.public-v12-champion-paired-graph-context.v19"
EXPECTED_PLAN_DIGEST = (
    "sha256:e66d9e4e90e4c3b2ccb704144c7a591009cde57b6367c3e1cc0b9dd64b8d40d5"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C"
)
SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v12-conflict.pt"
)
REPORT_PATH = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.json"
)
CHECKPOINT_PATH = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
REPORT_TEMP = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
CHECKPOINT_TEMP = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
RUN_CLAIM_PATH = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.claim.json"
)
FAILURE_REPORT_PATH = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.failure.json"
)
FAILURE_REPORT_TEMP = FAILURE_REPORT_PATH.with_suffix(
    FAILURE_REPORT_PATH.suffix + ".tmp"
)

EXPECTED_SOURCE_HASHES = {
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
        "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
    ),
    "tests/unit/experiments/test_phase6_v12_champion_paired_graph_context.py": (
        "C0D1DBBDE81B628D8D9CCFA751DCB9CFE951B3809860BE5298494C103D1E12BD"
    ),
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001.md": (
        "B819DA5F6D10151E7613ADECBBA076DF7642559D35BEA2EA74551FD791C6668D"
    ),
}

VALID_CLASSIFICATIONS = {
    "FULL_V12_REPLACEMENT",
    "PAIRED_GRAPH_COMPONENT_SUPPORTED",
    "PAIRED_GRAPH_CONTEXT_NOT_SUPPORTED",
}
TERMINAL_CONTEXT_UPDATES = 512
TERMINAL_STREAMS = 4_096
TERMINAL_ROWS = 16_384
MANUAL_SEED = 2_026_083_901


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise ValueError("V19 report contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"V19 report value is not JSON-safe: {type(value).__name__}")


def write_json_atomic(path: Path, temporary: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(
        json_ready(value),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_launch() -> dict[str, str]:
    occupied = tuple(
        str(path)
        for path in (
            REPORT_PATH,
            CHECKPOINT_PATH,
            REPORT_TEMP,
            CHECKPOINT_TEMP,
            RUN_CLAIM_PATH,
            FAILURE_REPORT_PATH,
            FAILURE_REPORT_TEMP,
        )
        if path.exists()
    )
    if occupied:
        raise RuntimeError(f"V19 one-shot output identity is already occupied: {occupied}")
    observed = {
        relative: sha256_file(ROOT / relative)
        for relative in EXPECTED_SOURCE_HASHES
    }
    if observed != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("V19 frozen launch source changed")
    if not SOURCE_CHECKPOINT.is_file():
        raise RuntimeError("V19 terminal V12 checkpoint is absent")
    if sha256_file(SOURCE_CHECKPOINT) != EXPECTED_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("V19 terminal V12 checkpoint hash changed")
    if v19.V12_CHECKPOINT_SHA256 != EXPECTED_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("V19 runner source-checkpoint binding changed")
    plan = v19.v12_champion_paired_graph_context_plan()
    if plan.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("V19 protocol identity changed")
    if plan.get("plan_digest") != EXPECTED_PLAN_DIGEST:
        raise RuntimeError("V19 plan digest changed")
    if (
        plan.get("context_updates") != TERMINAL_CONTEXT_UPDATES
        or plan.get("streams_per_update") != 8
        or plan.get("rows_per_stream") != 4
    ):
        raise RuntimeError("V19 fixed semantic-fit identity changed")
    return observed


def create_run_claim(launch_hashes: Mapping[str, str]) -> dict[str, object]:
    claim = {
        "artifact_schema": "angler.phase6-v19-run-claim.v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "process_id": os.getpid(),
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "launch_hashes": dict(launch_hashes),
        "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
        "fit_identity": {
            "optimizer_steps": TERMINAL_CONTEXT_UPDATES,
            "streams": TERMINAL_STREAMS,
            "rows": TERMINAL_ROWS,
        },
        "fit_calls": 1,
        "evaluation_calls": 1,
        "one_shot": True,
    }
    payload = json.dumps(
        json_ready(claim),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    with RUN_CLAIM_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(RUN_CLAIM_PATH),
        "sha256": sha256_file(RUN_CLAIM_PATH),
        "bytes": RUN_CLAIM_PATH.stat().st_size,
        "preserve_permanently": True,
    }


def save_and_verify_checkpoint(
    system: v19.V12ChampionPairedGraphContextSystem,
    *,
    require_terminal: bool = True,
) -> dict[str, object]:
    if CHECKPOINT_TEMP.exists() or CHECKPOINT_PATH.exists():
        raise RuntimeError("V19 checkpoint identity became occupied during execution")
    v19.save_v12_champion_paired_graph_context_checkpoint(CHECKPOINT_TEMP, system)
    restored = v19.load_v12_champion_paired_graph_context_checkpoint(
        CHECKPOINT_TEMP,
        device="cpu",
    )
    expected_digest = v19.paired_graph_system_digest(system)
    restored_digest = v19.paired_graph_system_digest(restored)
    if restored_digest != expected_digest:
        raise RuntimeError("V19 strict checkpoint reload changed the learned system")
    if restored.context_updates != system.context_updates:
        raise RuntimeError("V19 strict checkpoint reload changed the update count")
    if require_terminal and restored.context_updates != TERMINAL_CONTEXT_UPDATES:
        raise RuntimeError("V19 strict checkpoint reload lost terminal updates")
    if restored.optimizer_state is None:
        raise RuntimeError("V19 strict checkpoint reload lost optimizer state")
    expected_optimizer_digest = v19.paired_graph_optimizer_digest(
        system.optimizer_state
    )
    restored_optimizer_digest = v19.paired_graph_optimizer_digest(
        restored.optimizer_state
    )
    if restored_optimizer_digest != expected_optimizer_digest:
        raise RuntimeError("V19 strict checkpoint reload changed optimizer state")
    expected_mutable_digest = v19.paired_graph_mutable_digest(system.controller)
    restored_mutable_digest = v19.paired_graph_mutable_digest(restored.controller)
    if restored_mutable_digest != expected_mutable_digest:
        raise RuntimeError("V19 strict checkpoint reload changed mutable weights")
    checkpoint_hash = sha256_file(CHECKPOINT_TEMP)
    checkpoint_bytes = CHECKPOINT_TEMP.stat().st_size
    CHECKPOINT_TEMP.replace(CHECKPOINT_PATH)
    return {
        "path": str(CHECKPOINT_PATH),
        "sha256": checkpoint_hash,
        "bytes": checkpoint_bytes,
        "strict_reload_verified": True,
        "system_digest": restored_digest,
        "mutable_digest": restored_mutable_digest,
        "context_updates": restored.context_updates,
        "optimizer_present": True,
        "optimizer_digest": restored_optimizer_digest,
    }


def validate_fit_report(
    system: v19.V12ChampionPairedGraphContextSystem,
    fit_report: Mapping[str, object],
) -> None:
    expected_identity = {
        "optimizer_steps": TERMINAL_CONTEXT_UPDATES,
        "streams": TERMINAL_STREAMS,
        "rows": TERMINAL_ROWS,
    }
    if (
        system.context_updates != TERMINAL_CONTEXT_UPDATES
        or system.optimizer_state is None
        or fit_report.get("protocol_id") != PROTOCOL_ID
        or fit_report.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or fit_report.get("stage") != "paired_graph_context"
        or fit_report.get("start_update") != 0
        or fit_report.get("terminal_update") != TERMINAL_CONTEXT_UPDATES
        or any(fit_report.get(key) != value for key, value in expected_identity.items())
    ):
        raise RuntimeError("V19 returned an invalid fixed semantic-fit result")


def validate_evaluation_report(evaluation: Mapping[str, object]) -> str:
    classification = evaluation.get("classification")
    component_supported = evaluation.get("component_supported")
    full_replacement = evaluation.get("full_v12_replacement")
    if (
        evaluation.get("protocol_id") != PROTOCOL_ID
        or evaluation.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or classification not in VALID_CLASSIFICATIONS
        or type(component_supported) is not bool
        or type(full_replacement) is not bool
        or (classification == "FULL_V12_REPLACEMENT") is not full_replacement
        or (
            classification == "PAIRED_GRAPH_COMPONENT_SUPPORTED"
        )
        is not (component_supported and not full_replacement)
        or (
            classification == "PAIRED_GRAPH_CONTEXT_NOT_SUPPORTED"
        )
        is not (not component_supported)
    ):
        raise RuntimeError("V19 returned an invalid causal evaluation result")
    required = (
        "aggregate",
        "causal_delta",
        "attribution",
        "relation_exact_under_primary_lesion",
        "component_supported",
        "attribution_supported",
        "full_v12_replacement",
        "terminal_system_digest",
    )
    if any(key not in evaluation for key in required):
        raise RuntimeError("V19 causal evaluation omitted a required result")
    return str(classification)


def main() -> int:
    launch_hashes = verify_launch()
    started_wall = utc_now()
    started = time.perf_counter()
    run_claim = create_run_claim(launch_hashes)
    system: v19.V12ChampionPairedGraphContextSystem | None = None
    checkpoint: dict[str, object] | None = None
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(MANUAL_SEED)
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "event": "semantic_fit_started",
                    "device": "cpu",
                    "started_utc": started_wall,
                    "run_claim": run_claim,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        load_started = time.perf_counter()
        system = v19.load_v12_champion_paired_graph_context_source(
            SOURCE_CHECKPOINT,
            device="cpu",
        )
        load_seconds = time.perf_counter() - load_started
        initial_system_digest = v19.paired_graph_system_digest(system)

        fit_started = time.perf_counter()
        fit_report = v19.fit_v12_champion_paired_graph_context(system)
        fit_seconds = time.perf_counter() - fit_started
        validate_fit_report(system, fit_report)

        evaluation_started = time.perf_counter()
        evaluation = v19.evaluate_v12_champion_paired_graph_context(system)
        evaluation_seconds = time.perf_counter() - evaluation_started
        classification = validate_evaluation_report(evaluation)
        terminal_digest = v19.paired_graph_system_digest(system)
        if evaluation["terminal_system_digest"] != terminal_digest:
            raise RuntimeError("V19 evaluation changed or misreported the learned system")
        if torch.cuda.is_initialized():
            raise RuntimeError("V19 CPU-only semantic identity initialized CUDA")

        checkpoint_started = time.perf_counter()
        checkpoint = save_and_verify_checkpoint(system)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        causal_result = {
            "classification": classification,
            "component_supported": evaluation["component_supported"],
            "attribution_supported": evaluation["attribution_supported"],
            "full_v12_replacement": evaluation["full_v12_replacement"],
            "relation_exact_under_primary_lesion": (
                evaluation["relation_exact_under_primary_lesion"]
            ),
            "causal_delta": evaluation["causal_delta"],
            "aggregate": evaluation["aggregate"],
            "attribution": evaluation["attribution"],
        }
        report = {
            "artifact_schema": "angler.phase6-v19-paired-graph-context-report.v1",
            "protocol_id": PROTOCOL_ID,
            "classification": classification,
            "passed": evaluation["full_v12_replacement"],
            "component_supported": evaluation["component_supported"],
            "run_identity": {
                "one_shot": True,
                "fit_calls": 1,
                "evaluation_calls": 1,
                "device": "cpu",
                "torch_threads": 1,
                "torch_interop_threads": 1,
                "deterministic_algorithms": True,
                "manual_seed": MANUAL_SEED,
                "plan_digest": EXPECTED_PLAN_DIGEST,
                "fit_identity": {
                    "optimizer_steps": TERMINAL_CONTEXT_UPDATES,
                    "streams": TERMINAL_STREAMS,
                    "rows": TERMINAL_ROWS,
                },
                "started_utc": started_wall,
                "completed_utc": utc_now(),
                "run_claim": run_claim,
            },
            "source_integrity": {
                "launch_hashes": launch_hashes,
                "harness_sha256": sha256_file(Path(__file__).resolve()),
                "v12_checkpoint_path": str(SOURCE_CHECKPOINT),
                "v12_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
                "source_binding": asdict(system.source),
                "initial_system_digest": initial_system_digest,
                "terminal_system_digest": terminal_digest,
            },
            "timings_seconds": {
                "source_load": load_seconds,
                "semantic_fit": fit_seconds,
                "causal_evaluation": evaluation_seconds,
                "checkpoint_save_and_reload": checkpoint_seconds,
                "harness_total": time.perf_counter() - started,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_initialized": torch.cuda.is_initialized(),
            },
            "fit_report": fit_report,
            "evaluation_report": evaluation,
            "causal_result": causal_result,
            "checkpoint": checkpoint,
            "scope_and_effects": {
                "gpu_used": False,
                "network_used": False,
                "package_install_used": False,
                "model_or_llm_used": False,
                "replay_used": False,
                "srwm_used": False,
                "router_used": False,
                "deterministic_solver_used": False,
                "identity_inputs_used": False,
                "joint_training_performed": False,
                "external_effects": False,
            },
        }
        write_json_atomic(REPORT_PATH, REPORT_TEMP, report)
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "event": "semantic_fit_completed",
                    "classification": classification,
                    "passed": evaluation["full_v12_replacement"],
                    "component_supported": evaluation["component_supported"],
                    "report": {
                        "path": str(REPORT_PATH),
                        "sha256": sha256_file(REPORT_PATH),
                        "bytes": REPORT_PATH.stat().st_size,
                    },
                    "checkpoint": checkpoint,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        preservation_error: str | None = None
        if system is not None and system.context_updates > 0 and checkpoint is None:
            try:
                if CHECKPOINT_TEMP.exists() or CHECKPOINT_PATH.exists():
                    preservation_error = (
                        "checkpoint identity already exists and was preserved"
                    )
                else:
                    checkpoint = save_and_verify_checkpoint(
                        system,
                        require_terminal=False,
                    )
            except BaseException as checkpoint_error:
                preservation_error = (
                    f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                )
        failure = {
            "artifact_schema": "angler.phase6-v19-paired-graph-context-report.v1",
            "protocol_id": PROTOCOL_ID,
            "classification": "HARNESS_ERROR_PRESERVED",
            "passed": False,
            "run_identity": {
                "one_shot": True,
                "device": "cpu",
                "started_utc": started_wall,
                "failed_utc": utc_now(),
                "plan_digest": EXPECTED_PLAN_DIGEST,
                "run_claim": run_claim,
            },
            "source_integrity": {
                "launch_hashes": launch_hashes,
                "harness_sha256": sha256_file(Path(__file__).resolve()),
            },
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "context_updates_preserved": (
                    system.context_updates if system is not None else 0
                ),
                "checkpoint_preservation_error": preservation_error,
                "checkpoint_temporary_preserved": CHECKPOINT_TEMP.exists(),
                "checkpoint_final_preserved": CHECKPOINT_PATH.exists(),
            },
            "checkpoint": checkpoint,
            "timings_seconds": {"harness_total": time.perf_counter() - started},
        }
        if not FAILURE_REPORT_PATH.exists() and not FAILURE_REPORT_TEMP.exists():
            write_json_atomic(
                FAILURE_REPORT_PATH,
                FAILURE_REPORT_TEMP,
                failure,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
