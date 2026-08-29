from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
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

# The recovery identity is CPU-only. Hide CUDA before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.evaluators import phase6_v19_paired_graph_context_recovery as recovery
from experiments.runners import phase6_v12_champion_paired_graph_context as v19


PROTOCOL_ID = "phase6.public-v12-champion-paired-graph-context-eval-recovery.v19r1"
SOURCE_PROTOCOL_ID = "phase6.public-v12-champion-paired-graph-context.v19"
EXPECTED_PLAN_DIGEST = (
    "sha256:e66d9e4e90e4c3b2ccb704144c7a591009cde57b6367c3e1cc0b9dd64b8d40d5"
)
TERMINAL_CONTEXT_UPDATES = 512
MANUAL_SEED = 2_026_083_901

SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v12-conflict.pt"
)
ORIGINAL_REPORT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.json"
)
ORIGINAL_CHECKPOINT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
ORIGINAL_CLAIM = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.claim.json"
)
ORIGINAL_FAILURE = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.failure.json"
)
RECOVERY_REPORT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context-"
    "eval-recovery-r1.json"
)
RECOVERY_CLAIM = RECOVERY_REPORT.with_name(
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context-"
    "eval-recovery-r1.claim.json"
)
RECOVERY_FAILURE = RECOVERY_REPORT.with_name(
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context-"
    "eval-recovery-r1.failure.json"
)
RECOVERY_REPORT_TEMP = RECOVERY_REPORT.with_suffix(".json.tmp")
RECOVERY_FAILURE_TEMP = RECOVERY_FAILURE.with_suffix(".json.tmp")

EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "B4DA4550D18C9F1480903DA087A8E7799341763F1EDD63061E8A04A7491BD62C"
)
EXPECTED_ORIGINAL_CHECKPOINT_SHA256 = (
    "10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F"
)
EXPECTED_ORIGINAL_CLAIM_SHA256 = (
    "E209F2075C59F2AD1087B2F11FFFABEAF31FC598E8B72B10D08F5F6F5E093C57"
)
EXPECTED_ORIGINAL_FAILURE_SHA256 = (
    "C297B861A26FF53EA489E70E537F6EECA7C20B54769394C85C042147838116EE"
)
EXPECTED_TERMINAL_SYSTEM_DIGEST = (
    "sha256:99712cfbc24140703203561f3ca42d904752aae92c8ec8d637128f7fe93bebc6"
)
EXPECTED_TERMINAL_MUTABLE_DIGEST = (
    "sha256:9cb6c11f5ff05fe75737227599094378cdacc9d914a3d558548780b26f7735ed"
)
EXPECTED_TERMINAL_OPTIMIZER_DIGEST = (
    "sha256:662fd334ecf56f0120e1b3023598099d7929289821df4a72b54a2ab74e83a388"
)

EXPECTED_REPOSITORY_HASHES = {
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
        "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
    ),
    "tests/unit/experiments/test_phase6_v12_champion_paired_graph_context.py": (
        "C0D1DBBDE81B628D8D9CCFA751DCB9CFE951B3809860BE5298494C103D1E12BD"
    ),
    ".angler_v19_once.py": (
        "099381C7AE58F1FBEEFCEC31B0FE1D53DA591D9D51D4E53549FA534F8D5D3123"
    ),
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001.md": (
        "B819DA5F6D10151E7613ADECBBA076DF7642559D35BEA2EA74551FD791C6668D"
    ),
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V19-EVAL-RECOVERY-001.md": (
        "4A068443C2FA8A7481576154575FDED9D08CD5ED4064FDE0DAA003A73F2B4A57"
    ),
    "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py": (
        "E9656044749805E626C2DD443EBB5C34E95656CE11128AB5B4D6A3425C927517"
    ),
    "tests/unit/experiments/test_phase6_v19_paired_graph_context_recovery.py": (
        "AD27A0015F555F41BC6D7299BB155D87FAB4CE8DEC09370D8F08FC81051DB4E9"
    ),
}

VALID_CLASSIFICATIONS = {
    "FULL_V12_REPLACEMENT",
    "PAIRED_GRAPH_COMPONENT_SUPPORTED",
    "PAIRED_GRAPH_CONTEXT_NOT_SUPPORTED",
}


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
        raise ValueError("V19 recovery report contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"V19 recovery report value is not JSON-safe: {type(value).__name__}"
    )


def write_json_atomic(
    path: Path,
    temporary: Path,
    value: Mapping[str, object],
) -> None:
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


def library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    for distribution in ("numpy", "scipy", "pytest"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def learned_state_identity(
    system: v19.V12ChampionPairedGraphContextSystem,
) -> dict[str, object]:
    return {
        "context_updates": system.context_updates,
        "system_digest": v19.paired_graph_system_digest(system),
        "mutable_digest": v19.paired_graph_mutable_digest(system.controller),
        "optimizer_digest": v19.paired_graph_optimizer_digest(
            system.optimizer_state
        ),
    }


def verify_launch() -> dict[str, object]:
    occupied = tuple(
        str(path)
        for path in (
            RECOVERY_CLAIM,
            RECOVERY_REPORT,
            RECOVERY_FAILURE,
            RECOVERY_REPORT_TEMP,
            RECOVERY_FAILURE_TEMP,
        )
        if path.exists()
    )
    if occupied:
        raise RuntimeError(
            f"V19 recovery one-shot output identity is already occupied: {occupied}"
        )
    observed = {
        relative: sha256_file(ROOT / relative)
        for relative in EXPECTED_REPOSITORY_HASHES
    }
    if observed != EXPECTED_REPOSITORY_HASHES:
        raise RuntimeError("V19 recovery frozen repository input changed")
    expected_files = {
        SOURCE_CHECKPOINT: EXPECTED_SOURCE_CHECKPOINT_SHA256,
        ORIGINAL_CHECKPOINT: EXPECTED_ORIGINAL_CHECKPOINT_SHA256,
        ORIGINAL_CLAIM: EXPECTED_ORIGINAL_CLAIM_SHA256,
        ORIGINAL_FAILURE: EXPECTED_ORIGINAL_FAILURE_SHA256,
    }
    original_evidence = {}
    for path, expected_hash in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"V19 recovery immutable evidence changed: {path}")
        original_evidence[str(path)] = {
            "sha256": expected_hash,
            "bytes": path.stat().st_size,
        }
    if ORIGINAL_REPORT.exists():
        raise RuntimeError("V19 original terminal report unexpectedly exists")
    failure_payload = json.loads(ORIGINAL_FAILURE.read_text(encoding="utf-8"))
    if failure_payload.get("classification") != "HARNESS_ERROR_PRESERVED":
        raise RuntimeError("V19 original failure classification changed")
    if recovery.PROTOCOL_ID != PROTOCOL_ID or v19.PROTOCOL_ID != SOURCE_PROTOCOL_ID:
        raise RuntimeError("V19 recovery protocol binding changed")
    plan = v19.v12_champion_paired_graph_context_plan()
    if (
        plan.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or plan.get("context_updates") != TERMINAL_CONTEXT_UPDATES
    ):
        raise RuntimeError("V19 frozen causal-evaluation identity changed")
    return {
        "repository_hashes": observed,
        "original_evidence": original_evidence,
        "original_terminal_report_absent": True,
        "original_failure_classification": "HARNESS_ERROR_PRESERVED",
    }


def create_run_claim(launch: Mapping[str, object]) -> dict[str, object]:
    claim = {
        "artifact_schema": "angler.phase6-v19-eval-recovery-claim.v1",
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "created_utc": utc_now(),
        "process_id": os.getpid(),
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "launch_integrity": dict(launch),
        "source_checkpoint_sha256": EXPECTED_ORIGINAL_CHECKPOINT_SHA256,
        "fit_calls": 0,
        "causal_evaluation_calls": 1,
        "new_checkpoint_permitted": False,
        "one_shot": True,
    }
    payload = json.dumps(
        json_ready(claim),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    with RECOVERY_CLAIM.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(RECOVERY_CLAIM),
        "sha256": sha256_file(RECOVERY_CLAIM),
        "bytes": RECOVERY_CLAIM.stat().st_size,
        "preserve_permanently": True,
    }


def validate_recovery_result(value: Mapping[str, object]) -> tuple[str, Mapping[str, object]]:
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or not isinstance(value.get("evaluation"), Mapping)
        or not isinstance(value.get("projection_audit"), Mapping)
    ):
        raise RuntimeError("V19 recovery evaluator returned an invalid envelope")
    evaluation = value["evaluation"]
    assert isinstance(evaluation, Mapping)
    classification = evaluation.get("classification")
    component_supported = evaluation.get("component_supported")
    full_replacement = evaluation.get("full_v12_replacement")
    if (
        evaluation.get("protocol_id") != SOURCE_PROTOCOL_ID
        or evaluation.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or classification not in VALID_CLASSIFICATIONS
        or type(component_supported) is not bool
        or type(full_replacement) is not bool
        or (classification == "FULL_V12_REPLACEMENT") is not full_replacement
        or (classification == "PAIRED_GRAPH_COMPONENT_SUPPORTED")
        is not (component_supported and not full_replacement)
        or (classification == "PAIRED_GRAPH_CONTEXT_NOT_SUPPORTED")
        is not (not component_supported)
    ):
        raise RuntimeError("V19 recovered causal classification is invalid")
    required = {
        "panels",
        "aggregate",
        "causal_delta",
        "attribution",
        "relation_exact_under_primary_lesion",
        "component_supported",
        "attribution_supported",
        "full_v12_replacement",
        "terminal_system_digest",
    }
    if not required.issubset(evaluation):
        raise RuntimeError("V19 recovered causal evaluation omitted required results")
    audit = value["projection_audit"]
    assert isinstance(audit, Mapping)
    if (
        audit.get("wrapper_restored") is not True
        or int(audit.get("zero_residual_calls", 0)) <= 0
        or int(audit.get("duplicate_rows_projected", 0)) <= 0
        or float(audit.get("maximum_raw_duplicate_logit_difference", math.inf))
        > 1.0e-6
    ):
        raise RuntimeError("V19 recovery projection audit is invalid")
    return str(classification), evaluation


def main() -> int:
    launch = verify_launch()
    started_utc = utc_now()
    started = time.perf_counter()
    claim = create_run_claim(launch)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(MANUAL_SEED)
        environment = {
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "versions": library_versions(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized(),
        }
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "event": "evaluation_recovery_started",
                    "device": "cpu",
                    "started_utc": started_utc,
                    "run_claim": claim,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        load_started = time.perf_counter()
        system = v19.load_v12_champion_paired_graph_context_checkpoint(
            ORIGINAL_CHECKPOINT,
            device="cpu",
        )
        load_seconds = time.perf_counter() - load_started
        before = learned_state_identity(system)
        expected_identity = {
            "context_updates": TERMINAL_CONTEXT_UPDATES,
            "system_digest": EXPECTED_TERMINAL_SYSTEM_DIGEST,
            "mutable_digest": EXPECTED_TERMINAL_MUTABLE_DIGEST,
            "optimizer_digest": EXPECTED_TERMINAL_OPTIMIZER_DIGEST,
        }
        if before != expected_identity:
            raise RuntimeError("V19 recovery loaded a changed terminal learned state")

        evaluation_started = time.perf_counter()
        recovered = recovery.evaluate_v19_paired_graph_context_recovery(system)
        evaluation_seconds = time.perf_counter() - evaluation_started
        classification, evaluation = validate_recovery_result(recovered)
        after = learned_state_identity(system)
        if after != before:
            raise RuntimeError("V19 recovery changed the terminal learned state")
        if evaluation["terminal_system_digest"] != after["system_digest"]:
            raise RuntimeError("V19 recovery evaluation misreported terminal state")
        if sha256_file(ORIGINAL_CHECKPOINT) != EXPECTED_ORIGINAL_CHECKPOINT_SHA256:
            raise RuntimeError("V19 recovery changed the preserved terminal checkpoint")
        if torch.cuda.is_initialized():
            raise RuntimeError("V19 CPU-only recovery initialized CUDA")

        report = {
            "artifact_schema": "angler.phase6-v19-eval-recovery-report.v1",
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "classification": classification,
            "passed": evaluation["full_v12_replacement"],
            "component_supported": evaluation["component_supported"],
            "original_run": {
                "classification": "HARNESS_ERROR_PRESERVED",
                "claim_path": str(ORIGINAL_CLAIM),
                "claim_sha256": EXPECTED_ORIGINAL_CLAIM_SHA256,
                "failure_path": str(ORIGINAL_FAILURE),
                "failure_sha256": EXPECTED_ORIGINAL_FAILURE_SHA256,
                "checkpoint_path": str(ORIGINAL_CHECKPOINT),
                "checkpoint_sha256": EXPECTED_ORIGINAL_CHECKPOINT_SHA256,
                "terminal_report_absent": True,
            },
            "recovered_run": {
                "classification": classification,
                "evaluation_only": True,
                "fit_calls": 0,
                "causal_evaluation_calls": 1,
                "new_checkpoint_created": False,
                "first_result_accepted_without_tuning": True,
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "run_claim": claim,
            },
            "environment": environment,
            "source_integrity": {
                "launch": launch,
                "harness_sha256": sha256_file(Path(__file__).resolve()),
                "learned_state_before": before,
                "learned_state_after": after,
                "checkpoint_sha256_after": sha256_file(ORIGINAL_CHECKPOINT),
            },
            "timings_seconds": {
                "checkpoint_load": load_seconds,
                "causal_evaluation": evaluation_seconds,
                "harness_total": time.perf_counter() - started,
            },
            "projection_audit": recovered["projection_audit"],
            "evaluation_report": evaluation,
            "scope_and_effects": {
                "gpu_used": False,
                "network_used": False,
                "package_install_used": False,
                "model_or_llm_used": False,
                "fit_or_optimizer_update_used": False,
                "new_checkpoint_created": False,
                "deterministic_solver_used": False,
                "threshold_or_panel_change_used": False,
                "external_effects": False,
            },
        }
        write_json_atomic(RECOVERY_REPORT, RECOVERY_REPORT_TEMP, report)
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "event": "evaluation_recovery_completed",
                    "classification": classification,
                    "component_supported": evaluation["component_supported"],
                    "full_v12_replacement": evaluation["full_v12_replacement"],
                    "report": {
                        "path": str(RECOVERY_REPORT),
                        "sha256": sha256_file(RECOVERY_REPORT),
                        "bytes": RECOVERY_REPORT.stat().st_size,
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        failure = {
            "artifact_schema": "angler.phase6-v19-eval-recovery-report.v1",
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "classification": "RECOVERY_HARNESS_ERROR_PRESERVED",
            "passed": False,
            "original_run": {
                "classification": "HARNESS_ERROR_PRESERVED",
                "failure_path": str(ORIGINAL_FAILURE),
                "failure_sha256": EXPECTED_ORIGINAL_FAILURE_SHA256,
                "checkpoint_path": str(ORIGINAL_CHECKPOINT),
                "checkpoint_sha256": EXPECTED_ORIGINAL_CHECKPOINT_SHA256,
            },
            "recovered_run": {
                "evaluation_only": True,
                "fit_calls": 0,
                "causal_evaluation_calls_planned": 1,
                "new_checkpoint_created": False,
                "started_utc": started_utc,
                "failed_utc": utc_now(),
                "run_claim": claim,
            },
            "environment": {
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "versions": library_versions(),
                "cuda_initialized": torch.cuda.is_initialized(),
            },
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "timings_seconds": {"harness_total": time.perf_counter() - started},
        }
        if not RECOVERY_FAILURE.exists() and not RECOVERY_FAILURE_TEMP.exists():
            write_json_atomic(RECOVERY_FAILURE, RECOVERY_FAILURE_TEMP, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
