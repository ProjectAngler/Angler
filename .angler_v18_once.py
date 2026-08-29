from __future__ import annotations

from collections.abc import Mapping, Sequence
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

# The semantic identity is CPU-only.  Hide CUDA before importing torch so the
# one-shot harness cannot accidentally initialize or select a GPU device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.runners import phase6_v12_champion_context_incidence as v18


PROTOCOL_ID = "phase6.public-v12-champion-context-incidence.v18"
EXPECTED_PLAN_DIGEST = (
    "sha256:aa262085903f93239a61f366334f976bd925be8df6c9c32d9b9122bacac6e09d"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C"
)
SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v12-conflict.pt"
)
REPORT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v18-context-incidence.json"
)
CHECKPOINT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v18-context-incidence.pt"
)
REPORT_TEMP = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
CHECKPOINT_TEMP = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
RUN_CLAIM_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v18-context-incidence.claim.json"
)
FAILURE_REPORT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v18-context-incidence.failure.json"
)
FAILURE_REPORT_TEMP = FAILURE_REPORT_PATH.with_suffix(
    FAILURE_REPORT_PATH.suffix + ".tmp"
)

EXPECTED_SOURCE_HASHES = {
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_v12_champion_context_residual.py": (
        "3B5B05CA4122F08133213AA811D5C5EDCA6B9869EF56B132273A90CE42724333"
    ),
    "tests/unit/experiments/test_phase6_software_pipeline_reconstruction.py": (
        "2E6D844D24DB0A9326D84A19AEC56ED5BF6288B94C67AD5926AC05933FB6DF32"
    ),
    "experiments/runners/phase6_v12_champion_context_incidence.py": (
        "40942C7D96AFF54413ED9DD0E02ACB3FEA2B178B25EEB37C513573357ED3734B"
    ),
    "tests/unit/experiments/test_phase6_v12_champion_context_incidence.py": (
        "DCC8CCE479A3900375A339C78B2A11DE727D1B633A162DE9249098043E95C9DD"
    ),
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001.md": (
        "1A00F9FE8FE4004BF1E1F9FFD2D2441B2D57808CE6F7B3C87DA9CBB27DB964CF"
    ),
}

VALID_CLASSIFICATIONS = {
    "CONTEXT_INCIDENCE_SUPPORTED",
    "CONTEXT_INCIDENCE_COMPONENT_SUPPORTED",
    "CONTEXT_INCIDENCE_NOT_SUPPORTED",
    "INVALID_NO_CLAIM",
}
TERMINAL_CONTEXT_UPDATES = 256
MANUAL_SEED = 2_026_083_801


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
        raise ValueError("V18 report contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"V18 report value is not JSON-safe: {type(value).__name__}")


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
        raise RuntimeError(f"V18 one-shot output identity is already occupied: {occupied}")
    observed = {
        relative: sha256_file(ROOT / relative)
        for relative in EXPECTED_SOURCE_HASHES
    }
    if observed != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("V18 frozen launch source changed")
    if not SOURCE_CHECKPOINT.is_file():
        raise RuntimeError("V18 terminal V12 checkpoint is absent")
    if sha256_file(SOURCE_CHECKPOINT) != EXPECTED_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("V18 terminal V12 checkpoint hash changed")
    if v18.V12_CHECKPOINT_SHA256 != EXPECTED_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("V18 runner source-checkpoint binding changed")
    plan = v18.v12_champion_context_incidence_plan()
    if plan.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("V18 protocol identity changed")
    if plan.get("plan_digest") != EXPECTED_PLAN_DIGEST:
        raise RuntimeError("V18 plan digest changed")
    if plan.get("context_updates") != TERMINAL_CONTEXT_UPDATES:
        raise RuntimeError("V18 terminal update commitment changed")
    return observed


def create_run_claim(launch_hashes: Mapping[str, str]) -> dict[str, object]:
    claim = {
        "artifact_schema": "angler.phase6-v18-run-claim.v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "process_id": os.getpid(),
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "launch_hashes": dict(launch_hashes),
        "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
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
    system: v18.V12ChampionContextIncidenceSystem,
    *,
    require_terminal: bool = True,
) -> dict[str, object]:
    v18.save_v12_champion_context_incidence_checkpoint(CHECKPOINT_TEMP, system)
    restored = v18.load_v12_champion_context_incidence_checkpoint(
        CHECKPOINT_TEMP,
        device="cpu",
    )
    expected_digest = v18.context_incidence_system_digest(system)
    restored_digest = v18.context_incidence_system_digest(restored)
    if restored_digest != expected_digest:
        raise RuntimeError("V18 strict checkpoint reload changed the learned system")
    if restored.context_updates != system.context_updates:
        raise RuntimeError("V18 strict checkpoint reload changed the update count")
    if require_terminal and restored.context_updates != TERMINAL_CONTEXT_UPDATES:
        raise RuntimeError("V18 strict checkpoint reload lost terminal updates")
    if restored.optimizer_state is None:
        raise RuntimeError("V18 strict checkpoint reload lost optimizer state")
    expected_optimizer_digest = v18.context_incidence_optimizer_digest(
        system.optimizer_state
    )
    restored_optimizer_digest = v18.context_incidence_optimizer_digest(
        restored.optimizer_state
    )
    if restored_optimizer_digest != expected_optimizer_digest:
        raise RuntimeError("V18 strict checkpoint reload changed optimizer state")
    checkpoint_hash = sha256_file(CHECKPOINT_TEMP)
    checkpoint_bytes = CHECKPOINT_TEMP.stat().st_size
    CHECKPOINT_TEMP.replace(CHECKPOINT_PATH)
    return {
        "path": str(CHECKPOINT_PATH),
        "sha256": checkpoint_hash,
        "bytes": checkpoint_bytes,
        "strict_reload_verified": True,
        "system_digest": restored_digest,
        "mutable_digest": v18.context_incidence_mutable_digest(restored.controller),
        "context_updates": restored.context_updates,
        "optimizer_present": True,
        "optimizer_digest": restored_optimizer_digest,
    }


def main() -> int:
    launch_hashes = verify_launch()
    started_wall = utc_now()
    started = time.perf_counter()
    run_claim = create_run_claim(launch_hashes)
    system: v18.V12ChampionContextIncidenceSystem | None = None
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
        system = v18.load_v12_champion_context_incidence_source(
            SOURCE_CHECKPOINT,
            device="cpu",
        )
        load_seconds = time.perf_counter() - load_started
        initial_system_digest = v18.context_incidence_system_digest(system)
        fit_started = time.perf_counter()
        fit_report = v18.fit_v12_champion_context_incidence(system)
        fit_seconds = time.perf_counter() - fit_started
        classification = fit_report.get("classification")
        if (
            system.context_updates != TERMINAL_CONTEXT_UPDATES
            or system.optimizer_state is None
            or fit_report.get("context_updates") != TERMINAL_CONTEXT_UPDATES
            or fit_report.get("protocol_id") != PROTOCOL_ID
            or classification not in VALID_CLASSIFICATIONS
            or fit_report.get("passed")
            is not (classification == "CONTEXT_INCIDENCE_SUPPORTED")
        ):
            raise RuntimeError("V18 returned an invalid terminal semantic result")
        if torch.cuda.is_initialized():
            raise RuntimeError("V18 CPU-only semantic identity initialized CUDA")
        checkpoint_started = time.perf_counter()
        checkpoint = save_and_verify_checkpoint(system)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        report = {
            "artifact_schema": "angler.phase6-v18-context-incidence-report.v1",
            "protocol_id": PROTOCOL_ID,
            "classification": classification,
            "passed": fit_report["passed"],
            "run_identity": {
                "one_shot": True,
                "device": "cpu",
                "torch_threads": 1,
                "torch_interop_threads": 1,
                "deterministic_algorithms": True,
                "manual_seed": MANUAL_SEED,
                "plan_digest": EXPECTED_PLAN_DIGEST,
                "started_utc": started_wall,
                "completed_utc": utc_now(),
                "run_claim": run_claim,
            },
            "source_integrity": {
                "launch_hashes": launch_hashes,
                "v12_checkpoint_path": str(SOURCE_CHECKPOINT),
                "v12_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
                "source_binding": asdict(system.source),
                "initial_system_digest": initial_system_digest,
                "terminal_system_digest": v18.context_incidence_system_digest(system),
            },
            "pre_run_test_receipts": (
                {
                    "reviewer": "implementation",
                    "command": (
                        "python -B -m unittest "
                        "tests.unit.experiments."
                        "test_phase6_software_pipeline_reconstruction "
                        "tests.unit.experiments."
                        "test_phase6_v12_champion_context_incidence"
                    ),
                    "cases": 137,
                    "failures": 0,
                    "status": "PASS",
                    "seconds": 50.025,
                    "bound_test_sha256": {
                        relative: EXPECTED_SOURCE_HASHES[relative]
                        for relative in (
                            "tests/unit/experiments/"
                            "test_phase6_software_pipeline_reconstruction.py",
                            "tests/unit/experiments/"
                            "test_phase6_v12_champion_context_incidence.py",
                        )
                    },
                },
                {
                    "reviewer": "independent_root",
                    "command": (
                        "python -B -m unittest "
                        "tests.unit.experiments."
                        "test_phase6_software_pipeline_reconstruction "
                        "tests.unit.experiments."
                        "test_phase6_v12_champion_context_incidence"
                    ),
                    "cases": 137,
                    "failures": 0,
                    "skipped": 1,
                    "status": "PASS",
                    "seconds": 48.487,
                    "read_only_wsl_mount": True,
                    "bound_test_sha256": {
                        relative: EXPECTED_SOURCE_HASHES[relative]
                        for relative in (
                            "tests/unit/experiments/"
                            "test_phase6_software_pipeline_reconstruction.py",
                            "tests/unit/experiments/"
                            "test_phase6_v12_champion_context_incidence.py",
                        )
                    },
                },
            ),
            "pre_run_receipt_scope": {
                "implementation_receipt_embedded": True,
                "independent_receipt_available_at_harness_freeze": True,
                "independent_result_inferred": False,
            },
            "timings_seconds": {
                "source_load": load_seconds,
                "semantic_fit_and_evaluation": fit_seconds,
                "checkpoint_save_and_reload": checkpoint_seconds,
                "harness_total": time.perf_counter() - started,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "fit_report": fit_report,
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
                    "passed": fit_report["passed"],
                    "report": str(REPORT_PATH),
                    "checkpoint": str(CHECKPOINT_PATH),
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
                if CHECKPOINT_TEMP.exists():
                    preservation_error = (
                        "checkpoint temporary already exists and was preserved"
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
            "artifact_schema": "angler.phase6-v18-context-incidence-report.v1",
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
            "source_integrity": {"launch_hashes": launch_hashes},
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "context_updates_preserved": (
                    system.context_updates if system is not None else 0
                ),
                "checkpoint_preservation_error": preservation_error,
                "checkpoint_temporary_preserved": CHECKPOINT_TEMP.exists(),
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
