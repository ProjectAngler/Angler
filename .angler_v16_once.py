from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch


PROTOCOL_ID = "phase6.public-anonymous-cross-variation-plasticity.paired.v16"
EXPECTED_PLAN_DIGEST = (
    "sha256:6d357f781843816287ca70017fe5a0fa76fe500293284bd421ade08e8ca20731"
)
EXPECTED_CUDA_PREFLIGHT_SHA256 = (
    "B9EE8AB1B445C36D64CCA34A253001D67B6AC6D924650EB0343D205E6A412952"
)
EXPECTED_ZERO_VJP_CUDA_SHA256 = (
    "B96F50BB9EEA5509BFBE40060434A74298EBA75235323EE17E01FC068F469C88"
)
EXPECTED_COMMITMENT_SCHEDULE_SHA256 = (
    "8B18860D42DB4DF6979EBA3148CE94E817CF98D2A25014C58E15D34D46F8F7D1"
)
EXPECTED_ADAPTATION_SCHEDULE_SHA256 = (
    "6B449614DC824EF71022B622FF8348D444942D2F00701E6810BD119965F1D04D"
)

REPORT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v16-plasticity.json"
)
CHECKPOINT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v16-plasticity.pt"
)
REPORT_TEMP = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
CHECKPOINT_TEMP = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
CUDA_PREFLIGHT_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v16-preflight-cuda.json"
)
ZERO_VJP_CUDA_PATH = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v16-zero-vjp-cuda.json"
)
LOCAL_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "phase6-software-pipeline-reconstruction-v16-plasticity-summary.json"
)

EXPECTED_LAUNCH_HASHES = {
    "experiments/evaluators/software_pipeline_reconstruction_suite.py": (
        "45D2282D5CC7FC504B817BA6ECB656B31DD568F85916B65E9145D1E1B0DFCE44"
    ),
    "tests/unit/experiments/test_software_pipeline_reconstruction_suite.py": (
        "CFB02F8D66CFFC9E326705969E6B3309025FB2BEE4A6D066A0D4780EB86586D3"
    ),
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "tests/unit/experiments/test_phase6_software_pipeline_reconstruction.py": (
        "2E6D844D24DB0A9326D84A19AEC56ED5BF6288B94C67AD5926AC05933FB6DF32"
    ),
    "experiments/runners/phase6_counterfactual_plasticity_router.py": (
        "1AA64AAC3716F5C2C8333EE46852F839D19FC80AD39B1F5ED041E1738210C068"
    ),
    "tests/unit/experiments/test_phase6_counterfactual_plasticity_router.py": (
        "5B8AA19F536EF1788D25CF5BF54D982BA1F1BA033176E09126DB7CCB8D23BFFF"
    ),
    "experiments/runners/phase6_cross_variation_plasticity.py": (
        "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
    ),
    "tests/unit/experiments/test_phase6_cross_variation_plasticity.py": (
        "D2560CC62D5C2031A35BE1CF951E14167CBE789AA8BACF03C86535622C40AA4E"
    ),
    "experiments/runners/phase6_cross_variation_plasticity_v16.py": (
        "EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB"
    ),
    "tests/unit/experiments/test_phase6_cross_variation_plasticity_v16.py": (
        "95A01F4C65EDF962422E6E379B05A2C1D98241F78FA6BC9F57D23A3903585637"
    ),
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001.md": (
        "3E34456165CA5A9E39BC2682F945F99A3964FC3ADF585479CEE541D0F434E524"
    ),
}

EXPECTED_CUDA_LAUNCH_HASHES = {
    relative: expected
    for relative, expected in EXPECTED_LAUNCH_HASHES.items()
    if not relative.startswith("docs/blueprints/")
}

EXPECTED_PROGRESS = (
    (0, "uniform_adamw_plasticity"),
    (0, "learned_episodic_plasticity"),
    (1, "learned_episodic_plasticity"),
    (1, "uniform_adamw_plasticity"),
    (2, "uniform_adamw_plasticity"),
    (2, "learned_episodic_plasticity"),
)

EXPECTED_PROGRESS_KEYS = {
    "protocol_id",
    "event",
    "replicate",
    "arm",
    "completed_arms",
    "total_arms",
    "optimizer_steps",
    "streams",
    "rows",
    "elapsed_seconds",
    "adaptive_metric_included",
}

EXPECTED_REPORT_KEYS = {
    "protocol_id",
    "status",
    "plasticity_router_supported",
    "plan",
    "plan_digest",
    "frozen_dependency_hashes",
    "structural_preflight",
    "replicates",
    "aggregate",
    "support_checks",
    "elapsed_seconds",
    "checkpoint_written",
    "semantic_fit_device",
    "semantic_fit_threads",
    "context_or_joint_training_performed",
    "development_or_final_access",
    "wrong_evidence_training_streams",
    "stored_examples_or_replay",
    "scalar_judge_calls",
    "deterministic_solver_used",
    "result_conditioned_continuation",
}

EXPECTED_SUPPORT_KEYS = {
    "integrity",
    "absolute_learned_competence",
    "paired_learned_advantage",
    "routing_specialization",
    "fresh_adaptation",
    "read_harmonization",
}

ALLOWED_STATUSES = {
    "INVALID_NO_CLAIM",
    "PLASTICITY_ROUTER_HARMFUL",
    "NO_COMPETENCE",
    "PLASTICITY_ROUTER_SUPPORTED",
    "PLASTICITY_ROUTER_NULL",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def validate_parameter_records(records: object, label: str) -> None:
    if not isinstance(records, (tuple, list)) or not records:
        raise RuntimeError(f"{label} records are absent")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("finite") is not True:
            raise RuntimeError(f"{label} record {index} is not finite")
        for key in ("numel", "nonzero", "zero"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"{label} record {index} has invalid {key}")
        if record["nonzero"] + record["zero"] != record["numel"]:
            raise RuntimeError(f"{label} record {index} counts changed")
        require_finite_number(record.get("max_abs"), f"{label} max_abs")
        require_finite_number(record.get("fp64_norm"), f"{label} fp64_norm")


def validate_structural_preflight(
    report: object,
    *,
    expected_device: str,
) -> None:
    if not isinstance(report, dict):
        raise RuntimeError("V16 structural preflight is not a mapping")
    required = {
        "protocol_id": PROTOCOL_ID,
        "kind": "NO_UPDATE_STRUCTURAL_PREFLIGHT",
        "device": expected_device,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "replicate_count": 3,
        "arms_checked": 6,
        "update_zero_batches_only": True,
        "optimizer_steps": 0,
        "evaluation_performed": False,
        "classification_performed": False,
        "checkpoint_written": False,
        "all_routes_exact_uniform": True,
        "all_upstream_meta_gradients_exact_zero": True,
        "all_tensors_finite": True,
        "all_before_after_digests_exact": True,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(f"V16 structural preflight changed {key}")
    replicates = report.get("replicates")
    if not isinstance(replicates, (tuple, list)) or len(replicates) != 3:
        raise RuntimeError("V16 structural preflight replicate set changed")
    for replicate_index, replicate in enumerate(replicates):
        if (
            not isinstance(replicate, dict)
            or replicate.get("replicate") != replicate_index
            or replicate.get("twins_exact") is not True
        ):
            raise RuntimeError("V16 structural preflight twin identity changed")
        arms = replicate.get("arms")
        if not isinstance(arms, (tuple, list)) or len(arms) != 2:
            raise RuntimeError("V16 structural preflight arm set changed")
        for arm in arms:
            if not isinstance(arm, dict):
                raise RuntimeError("V16 structural preflight arm is invalid")
            if arm.get("before_digests") != arm.get("after_digests"):
                raise RuntimeError("V16 structural preflight mutated an arm")
            routes = arm.get("routes")
            if not isinstance(routes, dict) or any(
                routes.get(key) is not True
                for key in (
                    "combined_exact_uniform",
                    "lane_a_exact_uniform",
                    "lane_b_exact_uniform",
                )
            ):
                raise RuntimeError("V16 structural preflight route changed")
            meta = arm.get("meta_gradient_diagnostics")
            if not isinstance(meta, dict) or meta.get("upstream_exact_zero") is not True:
                raise RuntimeError("V16 structural preflight upstream VJP changed")
            if arm.get("optimizer_steps") != 0 or arm.get("evaluation_performed") is not False:
                raise RuntimeError("V16 structural preflight performed an effect")
            for key in (
                "controller_parameters",
                "router_parameters",
                "routed_direction_parameters",
                "meta_gradient_parameters",
                "composer_gradient_parameters",
            ):
                validate_parameter_records(arm.get(key), f"preflight {key}")


def load_and_validate_cuda_preflight() -> dict[str, object]:
    if not CUDA_PREFLIGHT_PATH.is_file():
        raise RuntimeError("accepted V16 CUDA preflight is absent")
    observed_hash = file_sha256(CUDA_PREFLIGHT_PATH)
    if observed_hash != EXPECTED_CUDA_PREFLIGHT_SHA256:
        raise RuntimeError("accepted V16 CUDA preflight hash changed")
    envelope = json.loads(CUDA_PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(envelope, dict)
        or envelope.get("protocol_id") != PROTOCOL_ID
        or envelope.get("status") != "PASS"
        or envelope.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or envelope.get("semantic_fit_performed") is not False
        or envelope.get("evaluation_or_classification_performed") is not False
        or envelope.get("launch_hashes") != EXPECTED_CUDA_LAUNCH_HASHES
    ):
        raise RuntimeError("accepted V16 CUDA preflight envelope changed")
    validate_structural_preflight(envelope.get("preflight"), expected_device="cuda:0")
    return envelope


def load_and_validate_zero_vjp_cuda() -> dict[str, object]:
    if not ZERO_VJP_CUDA_PATH.is_file():
        raise RuntimeError("accepted V16 exact-zero CUDA VJP probe is absent")
    if file_sha256(ZERO_VJP_CUDA_PATH) != EXPECTED_ZERO_VJP_CUDA_SHA256:
        raise RuntimeError("accepted V16 exact-zero CUDA VJP probe hash changed")
    envelope = json.loads(ZERO_VJP_CUDA_PATH.read_text(encoding="utf-8"))
    original = envelope.get("original_v15") if isinstance(envelope, dict) else None
    repaired = envelope.get("repaired_v16") if isinstance(envelope, dict) else None
    parity = envelope.get("forward_parity") if isinstance(envelope, dict) else None
    if (
        not isinstance(envelope, dict)
        or envelope.get("protocol_id") != PROTOCOL_ID
        or envelope.get("kind")
        != "DIRECT_EXACT_ZERO_DIRECTION_VJP_CUDA_PROBE"
        or envelope.get("status") != "PASS"
        or envelope.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or envelope.get("launch_hashes") != EXPECTED_CUDA_LAUNCH_HASHES
        or envelope.get("semantic_fit_performed") is not False
        or envelope.get("evaluation_or_classification_performed") is not False
        or not isinstance(original, dict)
        or original.get("vjp_values") != 19
        or original.get("vjp_finite_values") != 0
        or original.get("vjp_nonfinite_values") != 19
        or original.get("vjp_all_finite") is not False
        or not isinstance(repaired, dict)
        or repaired.get("vjp_values") != 19
        or repaired.get("vjp_finite_values") != 19
        or repaired.get("vjp_nonfinite_values") != 0
        or repaired.get("vjp_all_finite") is not True
        or repaired.get("forward_equals_input") is not True
        or repaired.get("state_step") != 1
        or repaired.get("state_exp_avg_nonzero") != 0
        or repaired.get("state_exp_avg_sq_nonzero") != 0
        or not isinstance(parity, dict)
        or parity.get("all_parameter_and_state_values_bit_equal") is not True
    ):
        raise RuntimeError("accepted V16 exact-zero CUDA VJP probe changed")
    return envelope


def validate_progress_event(
    event: Mapping[str, object],
    progress: Sequence[dict[str, object]],
) -> dict[str, object]:
    if set(event) != EXPECTED_PROGRESS_KEYS:
        raise RuntimeError("V16 arm progress shape changed")
    index = len(progress)
    if index >= len(EXPECTED_PROGRESS):
        raise RuntimeError("V16 emitted too many arm progress events")
    replicate, arm = EXPECTED_PROGRESS[index]
    required = {
        "protocol_id": PROTOCOL_ID,
        "event": "ARM_BOUNDARY_COMPLETE",
        "replicate": replicate,
        "arm": arm,
        "completed_arms": index + 1,
        "total_arms": 6,
        "optimizer_steps": 80,
        "streams": 640,
        "rows": 2_560,
        "adaptive_metric_included": False,
    }
    for key, expected in required.items():
        if event.get(key) != expected:
            raise RuntimeError(f"V16 arm progress changed {key}")
    elapsed = require_finite_number(event.get("elapsed_seconds"), "progress elapsed")
    if elapsed < 0.0:
        raise RuntimeError("V16 arm progress elapsed time is negative")
    if progress and elapsed < float(progress[-1]["elapsed_seconds"]):
        raise RuntimeError("V16 arm progress elapsed time regressed")
    return dict(event)


def validate_fit_record(record: object, label: str) -> None:
    if not isinstance(record, dict) or record.get("arm") != label:
        raise RuntimeError(f"V16 {label} fit identity changed")
    required = {
        "optimizer_steps": 80,
        "streams": 640,
        "rows": 2_560,
        "virtual_folds_per_update": 2,
        "first_allocation_exact_uniform": True,
        "update0_upstream_exact_zero": True,
        "cell_update": "pure_functional_adamw_stable_zero_vjp_v2",
        "virtual_cell_update": "pure_functional_adamw_stable_zero_vjp_v2",
        "actual_and_virtual_cell_update_identical": True,
        "composer_update": "separately_owned_adamw",
        "router_update": "separately_owned_adamw",
        "router_affects_current_batch": False,
        "frozen_controller_parameters_unchanged": True,
        "controller_grad_fields_clear": True,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise RuntimeError(f"V16 {label} fit changed {key}")
    learned = label == "learned_episodic_plasticity"
    if (
        record.get("router_scores_applied") is not learned
        or record.get("sham_router_compute_matched") is not (not learned)
        or record.get("router_affects_next_batch") is not learned
    ):
        raise RuntimeError(f"V16 {label} routing ownership changed")
    updates = record.get("updates")
    if not isinstance(updates, (tuple, list)) or len(updates) != 80:
        raise RuntimeError(f"V16 {label} update set changed")
    for update_index, update in enumerate(updates):
        if not isinstance(update, dict) or update.get("update") != update_index:
            raise RuntimeError(f"V16 {label} update order changed")
        validate_parameter_records(
            update.get("cell_direction_parameters"),
            f"{label} update {update_index} cell direction",
        )
        validate_parameter_records(
            update.get("composer_gradient_parameters"),
            f"{label} update {update_index} composer gradient",
        )
        meta = update.get("meta")
        diagnostics = meta.get("gradient_diagnostics") if isinstance(meta, dict) else None
        if not isinstance(diagnostics, dict):
            raise RuntimeError(f"V16 {label} update {update_index} meta is absent")
        validate_parameter_records(
            diagnostics.get("parameters"),
            f"{label} update {update_index} meta gradient",
        )


def validate_result(
    report: object,
    *,
    plan: dict[str, object],
    progress: Sequence[dict[str, object]],
) -> dict[str, object]:
    if not isinstance(report, dict) or set(report) != EXPECTED_REPORT_KEYS:
        raise RuntimeError("V16 result top-level shape changed")
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("plan") != plan
        or report.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or report.get("status") not in ALLOWED_STATUSES
        or report.get("plasticity_router_supported")
        is not (report.get("status") == "PLASTICITY_ROUTER_SUPPORTED")
    ):
        raise RuntimeError("V16 result identity or classification changed")
    if len(progress) != len(EXPECTED_PROGRESS):
        raise RuntimeError("V16 result completed without all arm boundaries")
    frozen_expected = {
        relative: expected
        for relative, expected in EXPECTED_LAUNCH_HASHES.items()
        if relative
        in {
            "experiments/evaluators/software_pipeline_reconstruction_suite.py",
            "tests/unit/experiments/test_software_pipeline_reconstruction_suite.py",
            "experiments/runners/phase6_software_pipeline_reconstruction.py",
            "experiments/runners/phase6_counterfactual_plasticity_router.py",
            "experiments/runners/phase6_cross_variation_plasticity.py",
            "tests/unit/experiments/test_phase6_cross_variation_plasticity.py",
        }
    }
    if report.get("frozen_dependency_hashes") != frozen_expected:
        raise RuntimeError("V16 result frozen dependency binding changed")
    validate_structural_preflight(report.get("structural_preflight"), expected_device="cpu")
    support = report.get("support_checks")
    if (
        not isinstance(support, dict)
        or set(support) != EXPECTED_SUPPORT_KEYS
        or any(type(value) is not bool for value in support.values())
    ):
        raise RuntimeError("V16 support-check shape changed")
    if not isinstance(report.get("aggregate"), dict):
        raise RuntimeError("V16 aggregate diagnostics are absent")
    required = {
        "checkpoint_written": True,
        "semantic_fit_device": "cpu",
        "semantic_fit_threads": 1,
        "context_or_joint_training_performed": False,
        "development_or_final_access": False,
        "wrong_evidence_training_streams": 0,
        "stored_examples_or_replay": 0,
        "scalar_judge_calls": 0,
        "deterministic_solver_used": False,
        "result_conditioned_continuation": False,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(f"V16 result changed {key}")
    if require_finite_number(report.get("elapsed_seconds"), "result elapsed") < 0.0:
        raise RuntimeError("V16 result elapsed time is negative")
    replicates = report.get("replicates")
    if not isinstance(replicates, (tuple, list)) or len(replicates) != 3:
        raise RuntimeError("V16 result replicate set changed")
    for replicate_index, replicate in enumerate(replicates):
        if (
            not isinstance(replicate, dict)
            or replicate.get("replicate") != replicate_index
            or replicate.get("paired_start_exact") is not True
            or replicate.get("paired_exposure_exact") is not True
        ):
            raise RuntimeError("V16 result paired replicate identity changed")
        fits = replicate.get("fits")
        if not isinstance(fits, dict) or set(fits) != {
            "uniform_adamw_plasticity",
            "learned_episodic_plasticity",
        }:
            raise RuntimeError("V16 result fit-arm set changed")
        for label, fit in fits.items():
            validate_fit_record(fit, label)
        terminal = replicate.get("terminal_digests")
        if not isinstance(terminal, dict) or set(terminal) != {"uniform", "learned"}:
            raise RuntimeError("V16 result terminal digests are absent")
        for digest in terminal.values():
            if (
                not isinstance(digest, str)
                or not digest.startswith("sha256:")
                or len(digest) != 71
            ):
                raise RuntimeError("V16 result terminal digest is malformed")
        for key in ("evaluations", "adaptation", "specialization"):
            if not isinstance(replicate.get(key), dict):
                raise RuntimeError(f"V16 result replicate {key} is absent")
    return report


def validate_checkpoint(v16: object, report: Mapping[str, object]) -> tuple[str, int]:
    if not CHECKPOINT_TEMP.is_file() or CHECKPOINT_TEMP.stat().st_size <= 0:
        raise RuntimeError("V16 checkpoint temp is absent or empty")
    with CHECKPOINT_TEMP.open("rb") as handle:
        os.fsync(handle.fileno())
    systems = v16.load_cross_variation_checkpoint(CHECKPOINT_TEMP, device="cpu")
    if len(systems) != 3:
        raise RuntimeError("V16 checkpoint replicate count changed")
    for replicate_index, pair in enumerate(systems):
        if len(pair) != 2:
            raise RuntimeError("V16 checkpoint arm count changed")
        expected = report["replicates"][replicate_index]["terminal_digests"]
        for label, arm in zip(("uniform", "learned"), pair, strict=True):
            if v16.cross_variation_arm_digest(arm) != expected[label]:
                raise RuntimeError("V16 checkpoint terminal lineage changed")
            if any(
                slot.step != 80
                for cell in arm.cell_optimizer_state
                for slot in cell
            ):
                raise RuntimeError("V16 checkpoint cell optimizer step changed")
    del systems
    return file_sha256(CHECKPOINT_TEMP), CHECKPOINT_TEMP.stat().st_size


def main() -> None:
    observed = {
        relative: file_sha256(ROOT / relative)
        for relative in EXPECTED_LAUNCH_HASHES
    }
    if observed != EXPECTED_LAUNCH_HASHES:
        raise RuntimeError("V16 semantic launch hashes changed")
    cuda_preflight = load_and_validate_cuda_preflight()
    zero_vjp_cuda = load_and_validate_zero_vjp_cuda()
    harness_sha256 = file_sha256(Path(__file__).resolve())
    fresh_targets = (
        REPORT_PATH,
        CHECKPOINT_PATH,
        REPORT_TEMP,
        CHECKPOINT_TEMP,
        LOCAL_SUMMARY_PATH,
    )
    existing = tuple(str(path) for path in fresh_targets if path.exists())
    if existing:
        raise RuntimeError(f"V16 semantic launch targets are not fresh: {existing!r}")

    v16 = importlib.import_module(
        "experiments.runners.phase6_cross_variation_plasticity_v16"
    )
    first_plan_digest = v16.cross_variation_plan_digest()
    second_plan_digest = v16.cross_variation_plan_digest()
    if first_plan_digest != EXPECTED_PLAN_DIGEST or second_plan_digest != first_plan_digest:
        raise RuntimeError("V16 semantic launch plan digest changed")
    plan = v16.cross_variation_fit_plan()
    if (
        plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("commitment_schedule_sha256")
        != EXPECTED_COMMITMENT_SCHEDULE_SHA256
        or plan.get("adaptation_schedule_sha256")
        != EXPECTED_ADAPTATION_SCHEDULE_SHA256
    ):
        raise RuntimeError("V16 semantic launch plan identity changed")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress: list[dict[str, object]] = []
    started_wall = time.time()
    started_monotonic = time.perf_counter()

    def progress_callback(event: Mapping[str, object]) -> None:
        validated = validate_progress_event(event, progress)
        progress.append(validated)
        print(json.dumps(validated, sort_keys=True, allow_nan=False), flush=True)

    invocation = {
        "event": "V16_FIT_INVOKED",
        "protocol_id": PROTOCOL_ID,
        "device": "cpu",
        "semantic_fit_threads": 1,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "runner_sha256": observed[
            "experiments/runners/phase6_cross_variation_plasticity_v16.py"
        ],
        "test_sha256": observed[
            "tests/unit/experiments/test_phase6_cross_variation_plasticity_v16.py"
        ],
        "active_leaf_sha256": observed[
            "docs/blueprints/branches/learning/work/"
            "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-001.md"
        ],
        "harness_sha256": harness_sha256,
        "accepted_cuda_preflight_sha256": EXPECTED_CUDA_PREFLIGHT_SHA256,
        "accepted_zero_vjp_cuda_sha256": EXPECTED_ZERO_VJP_CUDA_SHA256,
    }
    report_handle = REPORT_TEMP.open("xb")
    try:
        report_handle.write(
            json.dumps(invocation, sort_keys=True, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        report_handle.flush()
        os.fsync(report_handle.fileno())
        print(json.dumps(invocation, sort_keys=True), flush=True)
        report = v16.fit_cross_variation_pilot(
            device="cpu",
            checkpoint_path=CHECKPOINT_TEMP,
            progress_callback=progress_callback,
        )
        report = validate_result(report, plan=plan, progress=progress)
        checkpoint_sha256, checkpoint_bytes = validate_checkpoint(v16, report)
        completed_wall = time.time()
        report["execution"] = {
            "started_unix_seconds": started_wall,
            "completed_unix_seconds": completed_wall,
            "harness_elapsed_seconds": time.perf_counter() - started_monotonic,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "semantic_fit_device": "cpu",
            "semantic_fit_threads": 1,
            "plan_digest": EXPECTED_PLAN_DIGEST,
            "harness_sha256": harness_sha256,
            "launch_hashes": observed,
            "accepted_cuda_preflight": {
                "path": str(CUDA_PREFLIGHT_PATH),
                "sha256": EXPECTED_CUDA_PREFLIGHT_SHA256,
                "status": cuda_preflight["status"],
                "device": cuda_preflight["device"],
                "plan_digest": cuda_preflight["plan_digest"],
                "arms_checked": cuda_preflight["preflight"]["arms_checked"],
                "optimizer_steps": cuda_preflight["preflight"]["optimizer_steps"],
            },
            "accepted_zero_vjp_cuda": {
                "path": str(ZERO_VJP_CUDA_PATH),
                "sha256": EXPECTED_ZERO_VJP_CUDA_SHA256,
                "status": zero_vjp_cuda["status"],
                "device": zero_vjp_cuda["device"],
                "original_v15_vjp_all_finite": zero_vjp_cuda["original_v15"][
                    "vjp_all_finite"
                ],
                "repaired_v16_vjp_all_finite": zero_vjp_cuda["repaired_v16"][
                    "vjp_all_finite"
                ],
                "forward_bit_equal": zero_vjp_cuda["forward_parity"][
                    "all_parameter_and_state_values_bit_equal"
                ],
            },
            "progress_events": tuple(progress),
            "checkpoint": {
                "path": str(CHECKPOINT_PATH),
                "sha256": checkpoint_sha256,
                "bytes": checkpoint_bytes,
                "reload_verified": True,
                "all_cell_optimizer_steps": 80,
            },
        }
        encoded = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        report_handle.seek(0)
        report_handle.truncate(0)
        report_handle.write(encoded)
        report_handle.flush()
        os.fsync(report_handle.fileno())
    finally:
        report_handle.close()

    if REPORT_PATH.exists() or CHECKPOINT_PATH.exists():
        raise RuntimeError("V16 final targets appeared before atomic publication")
    os.replace(CHECKPOINT_TEMP, CHECKPOINT_PATH)
    fsync_directory(CHECKPOINT_PATH.parent)
    if file_sha256(CHECKPOINT_PATH) != report["execution"]["checkpoint"]["sha256"]:
        raise RuntimeError("V16 published checkpoint hash changed")
    os.replace(REPORT_TEMP, REPORT_PATH)
    fsync_directory(REPORT_PATH.parent)
    report_sha256 = file_sha256(REPORT_PATH)
    print(
        json.dumps(
            {
                "event": "V16_FIT_COMPLETE",
                "protocol_id": PROTOCOL_ID,
                "status": report["status"],
                "support_checks": report["support_checks"],
                "elapsed_seconds": report["elapsed_seconds"],
                "harness_elapsed_seconds": report["execution"][
                    "harness_elapsed_seconds"
                ],
                "checkpoint_sha256": report["execution"]["checkpoint"]["sha256"],
                "report_sha256": report_sha256,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
