from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
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


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from experiments.runners import phase6_oml_persistent_lifelong_v21 as lifelong


PROTOCOL_ID = "phase6.public-oml-persistent-lifelong.v21a"
ARTIFACT_SCHEMA = "angler.phase6-v21a-persistent-lifelong-report.v1"
CLAIM_SCHEMA = "angler.phase6-v21a-oml-persistent-lifelong-run-claim.v1"

V20_CHECKPOINT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v20-oml.pt"
)
V20_REPORT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v20-oml.json"
)
V19_CHECKPOINT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)

PREFIX = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v21a-oml-persistent-lifelong"
)
EXPECTED_OUTPUT_PARENT = Path("/opt/angler/results")
CLAIM_PATH = PREFIX.with_suffix(".claim.json")
PROGRESS_PATH = PREFIX.with_suffix(".progress.pt")
CHECKPOINT_PATH = PREFIX.with_suffix(".pt")
REPORT_PATH = PREFIX.with_suffix(".json")
FAILURE_PATH = PREFIX.with_suffix(".failure.json")

# Every temporary has one literal identity so fresh-run output checks and
# resume cleanup cover the complete write surface.  The runner writes exactly
# the path supplied by this harness and does not introduce a nested temporary.
CLAIM_TEMP = CLAIM_PATH.with_suffix(CLAIM_PATH.suffix + ".tmp")
PROGRESS_TEMP = PROGRESS_PATH.with_suffix(PROGRESS_PATH.suffix + ".tmp")
CHECKPOINT_TEMP = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
REPORT_TEMP = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
FAILURE_TEMP = FAILURE_PATH.with_suffix(FAILURE_PATH.suffix + ".tmp")

EXPECTED_ARTIFACT_HASHES = {
    str(V20_CHECKPOINT): (
        "D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48"
    ),
    str(V20_REPORT): (
        "5CCCBF0CE8211E0CC99AEB856145BF4CD3D9EA30A1ECB3FAE8E9435B4689C498"
    ),
    str(V19_CHECKPOINT): (
        "10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F"
    ),
}

# The two new implementation hashes and the plan digest are finalized only
# after the independent implementation review.  Launch verification refuses a
# claim while any literal remains PENDING.
EXPECTED_REPOSITORY_HASHES = {
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-OML-PERSISTENT-LIFELONG-V21-001.md": (
        "B85D47AABE390761E932FE49EABD123109663802CC7BA3346E6607F6D29F092F"
    ),
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V20-OML-001.md": (
        "1C0130928F77D80F2AAA9047E44DD02B4DEDEE951CEB2E41837EAA2A086B66F5"
    ),
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_cross_variation_plasticity_v16.py": (
        "EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB"
    ),
    "experiments/evaluators/glyph_machine_trace_suite.py": (
        "259118357A042A9867DA90514EFD82292C36709A573EAD13AE956089DBD3BC7E"
    ),
    "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py": (
        "E9656044749805E626C2DD443EBB5C34E95656CE11128AB5B4D6A3425C927517"
    ),
    "experiments/evaluators/software_pipeline_reconstruction_suite.py": (
        "45D2282D5CC7FC504B817BA6ECB656B31DD568F85916B65E9145D1E1B0DFCE44"
    ),
    "experiments/runners/phase5_glyph_machine_trace.py": (
        "BCA01B3BA152200E0E1C79EAD993DEBD2D65CECC5D881DAA8152B863A6FA1066"
    ),
    "experiments/runners/phase5_skill_memory_stream.py": (
        "CAB8CB309D80083ED22E6BA86451B64A4A30BFD0CBE696B0B17E63966CAE1924"
    ),
    "experiments/runners/phase6_counterfactual_plasticity_router.py": (
        "1AA64AAC3716F5C2C8333EE46852F839D19FC80AD39B1F5ED041E1738210C068"
    ),
    "experiments/runners/phase6_cross_variation_plasticity.py": (
        "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
    ),
    "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
        "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
    ),
    "experiments/runners/phase6_oml_relation_representation.py": (
        "6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3"
    ),
    "src/angler/procedures/__init__.py": (
        "A6FC55919DBCE2D73A822F9F0A900C78CACEA31031068CEF251029A526FF637E"
    ),
    "src/angler/procedures/execution.py": (
        "DEC18CAB5A9F93F2463D2CBAED115848282C7D1E58DC87FFAFC81B6E2EB03AC8"
    ),
    "src/angler/procedures/learning.py": (
        "346791D1D9D81BF0827B40C20AE6C5A9931F6FDA30B682D198DE0ED203C63616"
    ),
    "src/angler/procedures/operators.py": (
        "1F6726BD66AF6666B44CD9B94C43FCE12F4CF4F03179FC3723C472C600A746EA"
    ),
    "src/angler/procedures/records.py": (
        "2FBEB4DD34249BDC9B5340F122C8C9AD2EF8DD0F8DF3232CFDFC9A3F8A4BFD0D"
    ),
    "src/angler/procedures/skill_memory.py": (
        "DDA7B7992970E6839A09E7C9C14A6B2E32E1F3B8ADB0DD4748EDC19A0672D36D"
    ),
    "src/angler/procedures/trunk.py": (
        "9C7730345FE036B4991E6C6187F620B26229F4421B9F29EF3B8CEC9E40B99D5B"
    ),
    "src/angler/reasoning/__init__.py": (
        "FF237005A2F80D98F02D7A6509F37F11FB4FF703B8A0A747E02FF5603499C79C"
    ),
    "src/angler/reasoning/adaptive_core.py": (
        "5C78F1F8E9CDF4ADE1DEEE40BBEADF48FD481A68834A3D484DB0221612858584"
    ),
    "src/angler/reasoning/recurrent_core.py": (
        "C6504E286A01E2E95F6F10F43BADAE2B2A8FB8DEBFD4E09C7EFB855D722FA0B6"
    ),
    "src/angler/reasoning/self_referential_memory.py": (
        "D87642BAC7F5073386AB88E4CCE6B29ED3C900EB5044D7ADDC95FB84420831B4"
    ),
    "experiments/runners/phase6_oml_persistent_lifelong_v21.py": (
        "PENDING"
    ),
    "tests/unit/experiments/test_phase6_oml_persistent_lifelong_v21.py": (
        "PENDING"
    ),
}

EXPECTED_PLAN_DIGEST = (
    "sha256:e1ffe10cb204dfae0af748abb93dcf6f19c4384fe1cdabf3d381760bcc23eccb"
)
EXPECTED_TOTAL_EXPERIENCES = 256
STAGE_A_EXPERIENCES = 192
STAGE_B_EXPERIENCES = 64
PROGRESS_INTERVAL = 32
PROGRESS_CURSORS = (32, 64, 96, 128, 160, 192, 224, 256)
MANUAL_SEED = 2_026_082_902
NUMERICAL_MODE = "fp32_no_tf32_no_autocast_deterministic"
MAX_ALLOCATED_BYTES = 2 * 1024**3
MAX_CHECKPOINT_BYTES = 16 * 1024**2
MAX_JSON_BYTES = 4 * 1024**2
MAX_SEMANTIC_SECONDS = 45 * 60.0
PERSISTENT_FLOAT_VALUES = 192
CLOCK_ROLLBACK_TOLERANCE_SECONDS = 1.0

VALID_CLASSIFICATIONS = {
    "INVALID_NO_CLAIM",
    "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED",
    "STAGE_A_NOT_ACQUIRED",
    "FAST_ACQUISITION_WITH_FORGETTING",
    "INHERITED_CAPABILITY_REGRESSION",
    "FAST_ACQUISITION_ATTRIBUTION_NOT_ESTABLISHED",
    "FAST_ACQUISITION_WITHOUT_PERSISTENT_TRANSFER",
    "STATIC_REPRESENTATION_DOMINATES",
    "PERSISTENT_OML_NOT_SUPPORTED",
}

_CUDA_CONFIGURED = False


def _lexists(path: Path) -> bool:
    """Treat broken links as occupied output identities."""

    return os.path.lexists(path)


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
        raise ValueError("V21-A JSON contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"V21-A JSON value is not safe: {type(value).__name__}")


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


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"V21-A expected a JSON object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _publish_new_bytes(
    path: Path,
    temporary: Path,
    payload: bytes,
    *,
    byte_ceiling: int,
    before_publish: Callable[[], object] | None = None,
) -> dict[str, object]:
    """Publish a complete immutable file without a partial final identity."""

    if not 0 < len(payload) <= byte_ceiling:
        raise RuntimeError(
            f"V21-A immutable payload exceeds its ceiling: {len(payload)}"
        )
    if _lexists(path) or _lexists(temporary):
        raise FileExistsError(f"V21-A immutable identity is occupied: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if before_publish is not None:
            before_publish()
        # A hard-link publication is atomic and fails rather than replacing an
        # independently published final identity.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if not _lexists(path) and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
            _fsync_directory(temporary.parent)
        raise
    temporary.unlink()
    _fsync_directory(path.parent)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_json_immutable(
    path: Path,
    temporary: Path,
    value: Mapping[str, object],
    *,
    before_publish: Callable[[], object] | None = None,
) -> dict[str, object]:
    return _publish_new_bytes(
        path,
        temporary,
        _json_bytes(value),
        byte_ceiling=MAX_JSON_BYTES,
        before_publish=before_publish,
    )


def _output_paths() -> tuple[Path, ...]:
    return (
        CLAIM_PATH,
        PROGRESS_PATH,
        CHECKPOINT_PATH,
        REPORT_PATH,
        FAILURE_PATH,
        CLAIM_TEMP,
        PROGRESS_TEMP,
        CHECKPOINT_TEMP,
        REPORT_TEMP,
        FAILURE_TEMP,
    )


def _output_state() -> dict[str, bool]:
    return {str(path): _lexists(path) for path in _output_paths()}


def _assert_output_parent() -> None:
    parent = PREFIX.parent
    if parent != EXPECTED_OUTPUT_PARENT:
        raise RuntimeError("V21-A output parent identity changed")
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("V21-A output parent must be an existing real directory")
    if any(path.parent != parent for path in _output_paths()):
        raise RuntimeError("V21-A output escaped its literal result directory")


def _assert_finalized_constants() -> None:
    pending = tuple(
        path
        for path, expected in EXPECTED_REPOSITORY_HASHES.items()
        if expected == "PENDING"
    )
    if pending or EXPECTED_PLAN_DIGEST == "PENDING":
        raise RuntimeError(
            "V21-A launch constants are not finalized: "
            f"paths={pending}, plan_digest={EXPECTED_PLAN_DIGEST}"
        )


def _validate_v20_report() -> dict[str, object]:
    report = _read_json(V20_REPORT)
    checkpoint = report.get("checkpoint")
    source_integrity = report.get("source_integrity")
    run_identity = report.get("run_identity")
    if (
        report.get("artifact_schema") != "angler.phase6-v20-oml-report.v1"
        or report.get("protocol_id")
        != "phase6.public-oml-relation-representation.v20"
        or report.get("classification") != "OML_V19_HARMONIZED_ADVANCEMENT"
        or report.get("passed") is not True
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != str(V20_CHECKPOINT)
        or checkpoint.get("sha256") != EXPECTED_ARTIFACT_HASHES[str(V20_CHECKPOINT)]
        or checkpoint.get("strict_reload_verified") is not True
        or not isinstance(source_integrity, Mapping)
        or source_integrity.get("terminal_system_digest")
        != "sha256:4c8e1f5df037956e01ab59353df45cf114c76385cca5d77c0c632e633d7614c3"
        or not isinstance(run_identity, Mapping)
        or run_identity.get("terminal_update") != 240
    ):
        raise RuntimeError("V21-A frozen V20 source report changed")
    return {
        "protocol_id": report["protocol_id"],
        "classification": report["classification"],
        "checkpoint_sha256": checkpoint["sha256"],
        "terminal_system_digest": source_integrity["terminal_system_digest"],
    }


def _validate_plan(plan: object, digest: object) -> dict[str, object]:
    if not isinstance(plan, Mapping) or not isinstance(digest, str):
        raise RuntimeError("V21-A runner returned no frozen plan")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "total_experiences": EXPECTED_TOTAL_EXPERIENCES,
        "stage_a_experiences": STAGE_A_EXPERIENCES,
        "stage_b_experiences": STAGE_B_EXPERIENCES,
        "progress_interval": PROGRESS_INTERVAL,
        "progress_cursors": PROGRESS_CURSORS,
        "allocated_memory_ceiling_bytes": MAX_ALLOCATED_BYTES,
        "semantic_wall_time_ceiling_seconds": MAX_SEMANTIC_SECONDS,
    }
    observed = {
        "protocol_id": plan.get("protocol_id"),
        "plan_digest": plan.get("plan_digest", digest),
        "total_experiences": plan.get("total_experiences"),
        "stage_a_experiences": plan.get("stage_a_experiences"),
        "stage_b_experiences": plan.get("stage_b_experiences"),
        "progress_interval": plan.get("progress_interval"),
        "progress_cursors": tuple(plan.get("progress_cursors", ())),
        "allocated_memory_ceiling_bytes": plan.get(
            "allocated_memory_ceiling_bytes"
        ),
        "semantic_wall_time_ceiling_seconds": plan.get(
            "semantic_wall_time_ceiling_seconds"
        ),
    }
    if digest != EXPECTED_PLAN_DIGEST or observed != expected:
        raise RuntimeError("V21-A frozen runner plan changed")
    return observed


def verify_launch(*, resume: bool = False) -> dict[str, object]:
    """Verify frozen inputs and exact output state without making a stream."""

    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    _assert_finalized_constants()
    _assert_output_parent()
    repository_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in EXPECTED_REPOSITORY_HASHES
    }
    if repository_hashes != EXPECTED_REPOSITORY_HASHES:
        raise RuntimeError("V21-A frozen repository bytes changed")
    artifact_hashes = {
        path: sha256_file(Path(path)) for path in EXPECTED_ARTIFACT_HASHES
    }
    if artifact_hashes != EXPECTED_ARTIFACT_HASHES:
        raise RuntimeError("V21-A frozen source artifact bytes changed")
    source_report = _validate_v20_report()
    dependencies = lifelong.verify_persistent_lifelong_dependencies(
        V20_CHECKPOINT,
        V20_REPORT,
        V19_CHECKPOINT,
    )
    if not isinstance(dependencies, Mapping):
        raise RuntimeError("V21-A dependency verifier returned no mapping")
    plan = lifelong.persistent_lifelong_plan()
    plan_digest = lifelong.persistent_lifelong_plan_digest()
    plan_identity = _validate_plan(plan, plan_digest)

    state = _output_state()
    terminal_count = int(state[str(REPORT_PATH)]) + int(state[str(FAILURE_PATH)])
    if terminal_count > 1:
        raise RuntimeError("V21-A has two terminal identities")
    if resume:
        if not state[str(CLAIM_PATH)] or terminal_count:
            raise RuntimeError(
                "V21-A resume requires one claim and no terminal artifact"
            )
        for path in (CLAIM_PATH, PROGRESS_PATH, CHECKPOINT_PATH):
            if state[str(path)] and (path.is_symlink() or not path.is_file()):
                raise RuntimeError(
                    f"V21-A resume identity is not a regular file: {path}"
                )
        for path in (
            CLAIM_TEMP,
            PROGRESS_TEMP,
            CHECKPOINT_TEMP,
            REPORT_TEMP,
            FAILURE_TEMP,
        ):
            if state[str(path)] and (path.is_symlink() or not path.is_file()):
                raise RuntimeError(
                    f"V21-A resume temporary is not a regular file: {path}"
                )
    elif any(state.values()):
        occupied = tuple(path for path, exists in state.items() if exists)
        raise RuntimeError(f"V21-A one-shot identity is occupied: {occupied}")
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_digest": plan_digest,
        "plan_identity": plan_identity,
        "repository_hashes": repository_hashes,
        "artifact_hashes": artifact_hashes,
        "source_report": source_report,
        "dependencies": dict(dependencies),
        "output_state": state,
        "resume": resume,
    }


def _configure_cuda() -> dict[str, object]:
    global _CUDA_CONFIGURED
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V21-A requires exactly one visible CUDA device")
    if not _CUDA_CONFIGURED:
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(MANUAL_SEED)
        torch.cuda.manual_seed_all(MANUAL_SEED)
        _CUDA_CONFIGURED = True
    torch.cuda.set_device(0)
    cuda_autocast = torch.is_autocast_enabled("cuda")
    cpu_autocast = torch.is_autocast_enabled("cpu")
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.backends.cudnn.benchmark
        or cuda_autocast
        or cpu_autocast
        or torch.get_float32_matmul_precision() != "highest"
    ):
        raise RuntimeError("V21-A CUDA numerical mode is not frozen FP32")
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "device_capability": tuple(torch.cuda.get_device_capability(0)),
        "device_total_memory_bytes": int(properties.total_memory),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_autocast_enabled": cuda_autocast,
        "cpu_autocast_enabled": cpu_autocast,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "manual_seed": MANUAL_SEED,
    }


def _restore_semantic_rng_after_load(resume_source: str) -> None:
    """Establish the claim seed only when no V21 checkpoint supplied RNG state."""

    if resume_source not in {"source", "progress", "final"}:
        raise RuntimeError("V21-A resume source is invalid")
    if resume_source == "source":
        torch.manual_seed(MANUAL_SEED)
        torch.cuda.manual_seed_all(MANUAL_SEED)


def _assert_memory_ceiling(label: str) -> dict[str, int]:
    allocated = int(torch.cuda.max_memory_allocated(0))
    reserved = int(torch.cuda.max_memory_reserved(0))
    if allocated > MAX_ALLOCATED_BYTES:
        raise RuntimeError(f"V21-A exceeded 2 GiB at {label}: {allocated}")
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}


def run_cuda_preflight() -> dict[str, object]:
    """Run only toy tensor checks; no V21 protocol stream may be constructed."""

    _assert_finalized_constants()
    before = _output_state()
    if any(before.values()):
        raise RuntimeError("V21-A preflight requires every output identity absent")
    _validate_plan(
        lifelong.persistent_lifelong_plan(),
        lifelong.persistent_lifelong_plan_digest(),
    )
    cuda = _configure_cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    result = lifelong.synthetic_cuda_preflight(device="cuda:0")
    torch.cuda.synchronize(0)
    memory = _assert_memory_ceiling("synthetic preflight")
    after = _output_state()
    if after != before:
        raise RuntimeError("V21-A synthetic preflight created an output artifact")
    if not isinstance(result, Mapping):
        raise RuntimeError("V21-A synthetic preflight returned no mapping")
    required = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "detach_continuation_exact": True,
        "functional_adamw_parity": True,
        "checkpoint_resume_exact": True,
        "selected_state_constant_capacity": True,
        "persistent_float_values": PERSISTENT_FLOAT_VALUES,
        "semantic_streams_generated": 0,
        "semantic_updates_performed": False,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise RuntimeError("V21-A synthetic CUDA preflight failed its frozen checks")
    reported_peak = result.get("maximum_allocated_bytes")
    if (
        type(reported_peak) is not int
        or reported_peak < 0
        or reported_peak > MAX_ALLOCATED_BYTES
        or reported_peak > memory["peak_allocated_bytes"]
    ):
        raise RuntimeError("V21-A synthetic preflight memory report is invalid")
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "cuda": cuda,
        "result": dict(result),
        **memory,
        "elapsed_seconds": time.perf_counter() - started,
        "semantic_streams_generated": 0,
        "semantic_updates_performed": False,
        "claim_created": False,
    }


def _validate_recorded_preflight(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("V21-A claim has no synthetic preflight")
    result = value.get("result")
    cuda = value.get("cuda")
    required_result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "detach_continuation_exact": True,
        "functional_adamw_parity": True,
        "checkpoint_resume_exact": True,
        "selected_state_constant_capacity": True,
        "persistent_float_values": PERSISTENT_FLOAT_VALUES,
        "semantic_streams_generated": 0,
        "semantic_updates_performed": False,
    }
    if (
        value.get("status") != "PASS"
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or value.get("semantic_streams_generated") != 0
        or value.get("semantic_updates_performed") is not False
        or value.get("claim_created") is not False
        or not isinstance(result, Mapping)
        or any(result.get(key) != expected for key, expected in required_result.items())
        or not isinstance(cuda, Mapping)
        or cuda.get("device") != "cuda:0"
        or cuda.get("device_count") != 1
        or cuda.get("torch_threads") != 1
        or cuda.get("torch_interop_threads") != 1
        or cuda.get("deterministic_algorithms") is not True
        or cuda.get("tf32_matmul") is not False
        or cuda.get("tf32_cudnn") is not False
        or cuda.get("cudnn_benchmark") is not False
        or cuda.get("cuda_autocast_enabled") is not False
        or cuda.get("cpu_autocast_enabled") is not False
        or cuda.get("float32_matmul_precision") != "highest"
        or cuda.get("manual_seed") != MANUAL_SEED
    ):
        raise RuntimeError("V21-A recorded preflight changed")
    peak = value.get("peak_allocated_bytes")
    reported = result.get("maximum_allocated_bytes")
    if (
        type(peak) is not int
        or type(reported) is not int
        or not 0 <= reported <= peak <= MAX_ALLOCATED_BYTES
    ):
        raise RuntimeError("V21-A recorded preflight memory changed")
    return dict(value)


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"V21-A {label} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"V21-A {label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"V21-A {label} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def create_run_claim(
    launch: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    if any(_output_state().values()):
        raise FileExistsError("V21-A output changed before claim publication")
    created = datetime.now(timezone.utc)
    deadline = created + timedelta(seconds=MAX_SEMANTIC_SECONDS)
    checked_preflight = _validate_recorded_preflight(preflight)
    claim = {
        "artifact_schema": CLAIM_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": created.isoformat(),
        "identity_deadline_utc": deadline.isoformat(),
        "process_id": os.getpid(),
        "one_shot": True,
        "resume_same_identity_only": True,
        "plan_digest": EXPECTED_PLAN_DIGEST,
        "run_identity": {
            "total_experiences": EXPECTED_TOTAL_EXPERIENCES,
            "stage_a_experiences": STAGE_A_EXPERIENCES,
            "stage_b_experiences": STAGE_B_EXPERIENCES,
            "progress_cursors": PROGRESS_CURSORS,
            "device": "cuda:0",
            "numerical_mode": NUMERICAL_MODE,
            "manual_seed": MANUAL_SEED,
        },
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "repository_hashes": dict(launch["repository_hashes"]),
        "artifact_hashes": dict(launch["artifact_hashes"]),
        "source_report": dict(launch["source_report"]),
        "dependencies": dict(launch["dependencies"]),
        "synthetic_cuda_preflight": checked_preflight,
        "ceilings": {
            "allocated_memory_bytes": MAX_ALLOCATED_BYTES,
            "identity_wall_seconds": MAX_SEMANTIC_SECONDS,
            "checkpoint_bytes": MAX_CHECKPOINT_BYTES,
            "terminal_json_bytes": MAX_JSON_BYTES,
        },
    }
    publication = _publish_new_bytes(
        CLAIM_PATH,
        CLAIM_TEMP,
        _json_bytes(claim),
        byte_ceiling=MAX_JSON_BYTES,
    )
    return {**publication, "record": claim}


def _validate_existing_claim(launch: Mapping[str, object]) -> dict[str, object]:
    claim = _read_json(CLAIM_PATH)
    created = _parse_utc(claim.get("created_utc"), "claim created_utc")
    deadline = _parse_utc(
        claim.get("identity_deadline_utc"), "claim identity_deadline_utc"
    )
    if abs((deadline - created).total_seconds() - MAX_SEMANTIC_SECONDS) > 1.0e-6:
        raise RuntimeError("V21-A claim deadline duration changed")
    identity = claim.get("run_identity")
    ceilings = claim.get("ceilings")
    preflight = claim.get("synthetic_cuda_preflight")
    normalized_identity = (
        {
            "total_experiences": identity.get("total_experiences"),
            "stage_a_experiences": identity.get("stage_a_experiences"),
            "stage_b_experiences": identity.get("stage_b_experiences"),
            "progress_cursors": tuple(identity.get("progress_cursors", ())),
            "device": identity.get("device"),
            "numerical_mode": identity.get("numerical_mode"),
            "manual_seed": identity.get("manual_seed"),
        }
        if isinstance(identity, Mapping)
        else None
    )
    expected_identity = {
        "total_experiences": EXPECTED_TOTAL_EXPERIENCES,
        "stage_a_experiences": STAGE_A_EXPERIENCES,
        "stage_b_experiences": STAGE_B_EXPERIENCES,
        "progress_cursors": PROGRESS_CURSORS,
        "device": "cuda:0",
        "numerical_mode": NUMERICAL_MODE,
        "manual_seed": MANUAL_SEED,
    }
    if (
        claim.get("artifact_schema") != CLAIM_SCHEMA
        or claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or claim.get("one_shot") is not True
        or claim.get("resume_same_identity_only") is not True
        or claim.get("harness_sha256") != sha256_file(Path(__file__).resolve())
        or claim.get("repository_hashes") != launch["repository_hashes"]
        or claim.get("artifact_hashes") != launch["artifact_hashes"]
        or claim.get("source_report") != launch["source_report"]
        or claim.get("dependencies") != launch["dependencies"]
        or not isinstance(identity, Mapping)
        or set(identity) != set(expected_identity)
        or normalized_identity != expected_identity
        or not isinstance(ceilings, Mapping)
        or ceilings.get("allocated_memory_bytes") != MAX_ALLOCATED_BYTES
        or ceilings.get("identity_wall_seconds") != MAX_SEMANTIC_SECONDS
        or ceilings.get("checkpoint_bytes") != MAX_CHECKPOINT_BYTES
        or ceilings.get("terminal_json_bytes") != MAX_JSON_BYTES
    ):
        raise RuntimeError("V21-A existing claim does not bind this identity")
    _validate_recorded_preflight(preflight)
    publication = {
        "path": str(CLAIM_PATH),
        "sha256": sha256_file(CLAIM_PATH),
        "bytes": CLAIM_PATH.stat().st_size,
        "record": claim,
    }
    return publication


def _identity_wall_snapshot(
    claim: Mapping[str, object],
    *,
    persisted_age_seconds: float = 0.0,
    invocation_started: float | None = None,
    invocation_start_age_seconds: float | None = None,
    enforce_remaining: bool = True,
) -> dict[str, object]:
    created = _parse_utc(claim.get("created_utc"), "claim created_utc")
    deadline = _parse_utc(
        claim.get("identity_deadline_utc"), "claim identity_deadline_utc"
    )
    if abs((deadline - created).total_seconds() - MAX_SEMANTIC_SECONDS) > 1.0e-6:
        raise RuntimeError("V21-A immutable claim deadline changed")
    if not math.isfinite(persisted_age_seconds) or persisted_age_seconds < 0.0:
        raise RuntimeError("V21-A persisted claim age is invalid")
    now = datetime.now(timezone.utc)
    observed_age = (now - created).total_seconds()
    if not math.isfinite(observed_age) or observed_age < 0.0:
        raise RuntimeError("V21-A clock is before claim creation")
    if (
        observed_age + CLOCK_ROLLBACK_TOLERANCE_SECONDS
        < persisted_age_seconds
    ):
        raise RuntimeError("V21-A clock rolled back below persisted claim age")
    invocation_elapsed = (
        0.0 if invocation_started is None else time.perf_counter() - invocation_started
    )
    if not math.isfinite(invocation_elapsed) or invocation_elapsed < 0.0:
        raise RuntimeError("V21-A invocation monotonic clock is invalid")
    effective_age = max(observed_age, persisted_age_seconds)
    if invocation_start_age_seconds is not None:
        if (
            not math.isfinite(invocation_start_age_seconds)
            or invocation_start_age_seconds < persisted_age_seconds
        ):
            raise RuntimeError("V21-A invocation start age is invalid")
        effective_age = max(
            effective_age,
            invocation_start_age_seconds + invocation_elapsed,
        )
    remaining = MAX_SEMANTIC_SECONDS - effective_age
    if enforce_remaining and (remaining <= 0.0 or now >= deadline):
        raise TimeoutError("V21-A exhausted its immutable 45-minute identity budget")
    return {
        "claim_created_utc": created.isoformat(),
        "identity_deadline_utc": deadline.isoformat(),
        "observed_wall_age_seconds": observed_age,
        "persisted_wall_age_seconds": persisted_age_seconds,
        "identity_wall_elapsed_seconds": effective_age,
        "invocation_elapsed_seconds": invocation_elapsed,
        "remaining_identity_wall_seconds": max(0.0, remaining),
    }


def _acquire_claim_execution_lock():
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


def _clear_stale_resume_temporaries() -> tuple[str, ...]:
    expected_parent = PREFIX.parent.resolve()
    removed = []
    for path in (
        CLAIM_TEMP,
        PROGRESS_TEMP,
        CHECKPOINT_TEMP,
        REPORT_TEMP,
        FAILURE_TEMP,
    ):
        if not _lexists(path):
            continue
        if (
            path.is_symlink()
            or path.resolve().parent != expected_parent
            or not path.is_file()
        ):
            raise RuntimeError(f"V21-A refuses unsafe stale temporary: {path}")
        path.unlink()
        removed.append(str(path))
    if removed:
        _fsync_directory(PREFIX.parent)
    return tuple(removed)


def _checkpoint_harness_state(
    claim: Mapping[str, object],
    claim_sha256: str,
    cursor: int,
    wall: Mapping[str, object],
) -> dict[str, object]:
    age = float(wall["identity_wall_elapsed_seconds"])
    if not math.isfinite(age) or age < 0.0:
        raise RuntimeError("V21-A cannot checkpoint an invalid claim age")
    return {
        "claim_sha256": claim_sha256,
        "claim_created_utc": claim["created_utc"],
        "identity_deadline_utc": claim["identity_deadline_utc"],
        "last_identity_age_seconds": age,
        "publication_cursor": cursor,
    }


def _validate_harness_state(
    value: object,
    *,
    claim: Mapping[str, object],
    claim_sha256: str,
    cursor: int,
) -> dict[str, object]:
    required = {
        "claim_sha256",
        "claim_created_utc",
        "identity_deadline_utc",
        "last_identity_age_seconds",
        "publication_cursor",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RuntimeError("V21-A checkpoint harness state fields changed")
    age = value.get("last_identity_age_seconds")
    if (
        value.get("claim_sha256") != claim_sha256
        or value.get("claim_created_utc") != claim.get("created_utc")
        or value.get("identity_deadline_utc")
        != claim.get("identity_deadline_utc")
        or value.get("publication_cursor") != cursor
        or not isinstance(age, (int, float))
        or isinstance(age, bool)
        or not math.isfinite(float(age))
        or float(age) < 0.0
    ):
        raise RuntimeError("V21-A checkpoint harness state changed")
    return dict(value)


def _validate_checkpoint_summary(
    summary: object,
    *,
    claim: Mapping[str, object],
    claim_sha256: str,
    expected_cursor: int | None = None,
    final: bool = False,
) -> dict[str, object]:
    if not isinstance(summary, Mapping):
        raise RuntimeError("V21-A checkpoint has no summary")
    cursor = summary.get("cursor")
    if type(cursor) is not int or cursor not in (0, *PROGRESS_CURSORS):
        raise RuntimeError("V21-A checkpoint cursor is invalid")
    if expected_cursor is not None and cursor != expected_cursor:
        raise RuntimeError("V21-A checkpoint cursor changed across publication")
    stage = summary.get("stage")
    end_a = summary.get("end_a_complete")
    end_b = summary.get("end_b_complete")
    reset = summary.get("boundary_reset_applied")
    probes = tuple(summary.get("probe_keys", ()))
    if cursor < 192:
        valid = (
            stage == "stage_a"
            and end_a is False
            and end_b is False
            and reset is False
            and probes == ("pre",)
        )
    elif cursor < 256:
        valid = (
            stage == "stage_b"
            and end_a is True
            and end_b is False
            and reset is True
            and probes == ("pre", "end_A")
        )
    elif final:
        valid = (
            stage == "complete"
            and end_a is True
            and end_b is True
            and reset is True
            and probes == ("pre", "end_A", "end_B")
        )
    else:
        valid = (
            stage == "complete"
            and end_a is True
            and end_b is False
            and reset is True
            and probes == ("pre", "end_A")
        )
    if not valid:
        raise RuntimeError("V21-A checkpoint chronology is impossible")
    counters = summary.get("arm_counters")
    persistent_names = (
        "second_order_oml_persistent",
        "first_order_meta_persistent",
        "source_v19_persistent",
    )
    boundary_name = "second_order_boundary_reset"
    if not isinstance(counters, Mapping) or set(counters) != {
        *persistent_names,
        boundary_name,
    }:
        raise RuntimeError("V21-A checkpoint arm counters are incomplete")
    for name in persistent_names:
        record = counters[name]
        if (
            not isinstance(record, Mapping)
            or record.get("lifetime_updates") != cursor
            or record.get("adamw_step") != cursor
            or record.get("reset_count") != 0
        ):
            raise RuntimeError(f"V21-A checkpoint counter changed: {name}")
    boundary = counters[boundary_name]
    expected_boundary_step = cursor if cursor < 192 else cursor - 192
    expected_boundary_resets = 0 if cursor < 192 else 1
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("lifetime_updates") != cursor
        or boundary.get("adamw_step") != expected_boundary_step
        or boundary.get("reset_count") != expected_boundary_resets
    ):
        raise RuntimeError("V21-A checkpoint boundary counter changed")
    harness_state = _validate_harness_state(
        summary.get("harness_state"),
        claim=claim,
        claim_sha256=claim_sha256,
        cursor=cursor,
    )
    digest = summary.get("system_digest")
    learned_digest = summary.get("learned_state_digest")
    harness_digest = summary.get("harness_state_digest")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or not isinstance(learned_digest, str)
        or not learned_digest.startswith("sha256:")
        or not isinstance(harness_digest, str)
        or not harness_digest.startswith("sha256:")
    ):
        raise RuntimeError("V21-A checkpoint digest is invalid")
    result = dict(summary)
    result["harness_state"] = harness_state
    return result


def _checkpoint_record(
    path: Path,
    system: object,
    *,
    claim: Mapping[str, object],
    claim_sha256: str,
    expected_cursor: int | None = None,
    final: bool = False,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"V21-A checkpoint is absent: {path}")
    size = path.stat().st_size
    if not 0 < size <= MAX_CHECKPOINT_BYTES:
        raise RuntimeError(f"V21-A checkpoint exceeds 16 MiB: {size}")
    summary = _validate_checkpoint_summary(
        lifelong.persistent_lifelong_checkpoint_summary(system),
        claim=claim,
        claim_sha256=claim_sha256,
        expected_cursor=expected_cursor,
        final=final,
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": size,
        "summary": summary,
    }


def _publish_progress(
    system: object,
    event: Mapping[str, object],
    *,
    claim_record: Mapping[str, object],
    claim_sha256: str,
    wall: Mapping[str, object],
) -> dict[str, object]:
    cursor = event.get("cursor")
    if type(cursor) is not int or cursor not in PROGRESS_CURSORS:
        raise RuntimeError("V21-A progress callback escaped a frozen cursor")
    if event.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("V21-A progress callback changed protocol identity")
    if event.get("plan_digest") != EXPECTED_PLAN_DIGEST:
        raise RuntimeError("V21-A progress callback changed plan identity")
    if cursor < 192:
        expected_event = {
            "stage": "stage_a",
            "end_a_complete": False,
            "end_b_complete": False,
            "boundary_reset_applied": False,
        }
    elif cursor < 256:
        expected_event = {
            "stage": "stage_b",
            "end_a_complete": True,
            "end_b_complete": False,
            "boundary_reset_applied": True,
        }
    else:
        expected_event = {
            "stage": "complete",
            "end_a_complete": True,
            "end_b_complete": False,
            "boundary_reset_applied": True,
        }
    if any(event.get(key) != value for key, value in expected_event.items()):
        raise RuntimeError("V21-A progress callback chronology changed")
    event_allocated = event.get("allocated_bytes")
    if (
        event.get("system_digest")
        != lifelong.persistent_lifelong_system_digest(system)
        or type(event_allocated) is not int
        or not 0 <= event_allocated <= MAX_ALLOCATED_BYTES
    ):
        raise RuntimeError("V21-A progress callback state changed")
    if _lexists(PROGRESS_TEMP):
        raise RuntimeError("V21-A progress temporary is occupied")
    harness_state = _checkpoint_harness_state(
        claim_record,
        claim_sha256,
        cursor,
        wall,
    )
    lifelong.save_persistent_lifelong_checkpoint(
        PROGRESS_TEMP,
        system,
        harness_state=harness_state,
    )
    _fsync_file(PROGRESS_TEMP)
    expected_digest = lifelong.persistent_lifelong_system_digest(system)
    restored = lifelong.load_persistent_lifelong_checkpoint(
        PROGRESS_TEMP,
        V20_CHECKPOINT,
        V19_CHECKPOINT,
        device="cuda:0",
    )
    record = _checkpoint_record(
        PROGRESS_TEMP,
        restored,
        claim=claim_record,
        claim_sha256=claim_sha256,
        expected_cursor=cursor,
        final=False,
    )
    if lifelong.persistent_lifelong_system_digest(restored) != expected_digest:
        raise RuntimeError("V21-A progress strict reload changed learned state")
    del restored
    gc.collect()
    torch.cuda.empty_cache()
    os.replace(PROGRESS_TEMP, PROGRESS_PATH)
    _fsync_directory(PROGRESS_PATH.parent)
    record["path"] = str(PROGRESS_PATH)
    record["strict_reload_verified"] = True
    record.update(_assert_memory_ceiling(f"progress cursor {cursor}"))
    print(
        json.dumps(
            {
                "event": "V21A_PROGRESS_CHECKPOINT",
                "protocol_id": PROTOCOL_ID,
                "cursor": cursor,
                "checkpoint_sha256": record["sha256"],
                "system_digest": expected_digest,
                "peak_allocated_bytes": record["peak_allocated_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return record


def _load_resume_system(
    *,
    claim_record: Mapping[str, object],
    claim_sha256: str,
) -> tuple[object, str, float]:
    if _lexists(CHECKPOINT_PATH):
        if CHECKPOINT_PATH.is_symlink() or not CHECKPOINT_PATH.is_file():
            raise RuntimeError("V21-A final checkpoint is not a regular file")
        system = lifelong.load_persistent_lifelong_checkpoint(
            CHECKPOINT_PATH,
            V20_CHECKPOINT,
            V19_CHECKPOINT,
            device="cuda:0",
        )
        summary = _validate_checkpoint_summary(
            lifelong.persistent_lifelong_checkpoint_summary(system),
            claim=claim_record,
            claim_sha256=claim_sha256,
            expected_cursor=EXPECTED_TOTAL_EXPERIENCES,
            final=True,
        )
        return (
            system,
            "final",
            float(summary["harness_state"]["last_identity_age_seconds"]),
        )
    if _lexists(PROGRESS_PATH):
        if PROGRESS_PATH.is_symlink() or not PROGRESS_PATH.is_file():
            raise RuntimeError("V21-A progress checkpoint is not a regular file")
        system = lifelong.load_persistent_lifelong_checkpoint(
            PROGRESS_PATH,
            V20_CHECKPOINT,
            V19_CHECKPOINT,
            device="cuda:0",
        )
        summary = _validate_checkpoint_summary(
            lifelong.persistent_lifelong_checkpoint_summary(system),
            claim=claim_record,
            claim_sha256=claim_sha256,
            final=False,
        )
        cursor = int(summary["cursor"])
        if cursor not in PROGRESS_CURSORS:
            raise RuntimeError("V21-A resume progress cursor is not publishable")
        return (
            system,
            "progress",
            float(summary["harness_state"]["last_identity_age_seconds"]),
        )
    system = lifelong.build_persistent_lifelong_system(
        V20_CHECKPOINT,
        V19_CHECKPOINT,
        device="cuda:0",
    )
    return system, "source", 0.0


def _save_reload_publish_final(
    system: object,
    *,
    claim_record: Mapping[str, object],
    claim_sha256: str,
    wall: Mapping[str, object],
) -> tuple[object, dict[str, object]]:
    if _lexists(CHECKPOINT_PATH) or _lexists(CHECKPOINT_TEMP):
        raise FileExistsError("V21-A final checkpoint identity is occupied")
    summary = lifelong.persistent_lifelong_checkpoint_summary(system)
    if not isinstance(summary, Mapping) or summary.get("cursor") != 256:
        raise RuntimeError("V21-A cannot publish a nonterminal checkpoint")
    harness_state = _checkpoint_harness_state(
        claim_record,
        claim_sha256,
        256,
        wall,
    )
    expected_digest = lifelong.persistent_lifelong_system_digest(system)
    lifelong.save_persistent_lifelong_checkpoint(
        CHECKPOINT_TEMP,
        system,
        harness_state=harness_state,
    )
    _fsync_file(CHECKPOINT_TEMP)
    del system
    gc.collect()
    torch.cuda.empty_cache()
    restored = lifelong.load_persistent_lifelong_checkpoint(
        CHECKPOINT_TEMP,
        V20_CHECKPOINT,
        V19_CHECKPOINT,
        device="cuda:0",
    )
    record = _checkpoint_record(
        CHECKPOINT_TEMP,
        restored,
        claim=claim_record,
        claim_sha256=claim_sha256,
        expected_cursor=256,
        final=True,
    )
    if lifelong.persistent_lifelong_system_digest(restored) != expected_digest:
        raise RuntimeError("V21-A final strict reload changed learned state")
    os.link(CHECKPOINT_TEMP, CHECKPOINT_PATH)
    _fsync_directory(CHECKPOINT_PATH.parent)
    CHECKPOINT_TEMP.unlink()
    _fsync_directory(CHECKPOINT_PATH.parent)
    record["path"] = str(CHECKPOINT_PATH)
    record["strict_reload_verified"] = True
    record.update(_assert_memory_ceiling("final checkpoint reload"))
    return restored, record


def _validate_fit_report(
    value: object,
    *,
    start_cursor: int,
    system: object,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("V21-A fit returned no report")
    emitted = tuple(value.get("progress_cursors", ()))
    expected_emitted = tuple(cursor for cursor in PROGRESS_CURSORS if cursor > start_cursor)
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or value.get("start_cursor") != start_cursor
        or value.get("terminal_cursor") != EXPECTED_TOTAL_EXPERIENCES
        or emitted != expected_emitted
        or value.get("system_digest")
        != lifelong.persistent_lifelong_system_digest(system)
    ):
        raise RuntimeError("V21-A fit report changed the frozen chronology")
    return value


def _validate_evaluation(value: object, system: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("V21-A evaluation returned no report")
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("plan_digest") != EXPECTED_PLAN_DIGEST
        or value.get("classification") not in VALID_CLASSIFICATIONS
        or value.get("first_result_accepted_without_tuning") is not True
        or value.get("system_digest")
        != lifelong.persistent_lifelong_system_digest(system)
    ):
        raise RuntimeError("V21-A evaluation identity changed")
    for key in ("gates", "comparisons", "mechanical_validity"):
        if key not in value:
            raise RuntimeError(f"V21-A evaluation omitted {key}")
    return value


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
        raise RuntimeError("V21-A semantic run has no immutable claim")
    claim_sha256 = str(claim["sha256"])
    cuda = _configure_cuda()
    # Preflight intentionally exercises toy RNG state before the claim. This
    # pre-load seed makes source construction deterministic; source checkpoint
    # loaders may restore predecessor RNG, so the claim seed is re-established
    # after source construction. A V21 checkpoint instead restores its own RNG.
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed_all(MANUAL_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started_utc = utc_now()

    if resume:
        system, resume_source, persisted_age = _load_resume_system(
            claim_record=claim_record,
            claim_sha256=claim_sha256,
        )
    else:
        system = lifelong.build_persistent_lifelong_system(
            V20_CHECKPOINT,
            V19_CHECKPOINT,
            device="cuda:0",
        )
        resume_source = "source"
        persisted_age = 0.0

    _restore_semantic_rng_after_load(resume_source)

    initial_summary = lifelong.persistent_lifelong_checkpoint_summary(system)
    if not isinstance(initial_summary, Mapping):
        raise RuntimeError("V21-A source system has no chronology summary")
    start_cursor = int(initial_summary.get("cursor", -1))
    identity_start = _identity_wall_snapshot(
        claim_record,
        persisted_age_seconds=persisted_age,
        invocation_started=invocation_started,
        enforce_remaining=True,
    )
    invocation_start_age = float(identity_start["identity_wall_elapsed_seconds"])
    identity_anchor_started = time.perf_counter()

    def identity_wall(*, enforce_remaining: bool = True) -> dict[str, object]:
        return _identity_wall_snapshot(
            claim_record,
            persisted_age_seconds=persisted_age,
            invocation_started=identity_anchor_started,
            invocation_start_age_seconds=invocation_start_age,
            enforce_remaining=enforce_remaining,
        )

    def deadline_callback() -> bool:
        identity_wall(enforce_remaining=True)
        _assert_memory_ceiling("semantic boundary")
        return True

    progress_records: list[dict[str, object]] = []

    if resume_source == "final":
        fit_report: dict[str, object] = {
            "protocol_id": PROTOCOL_ID,
            "plan_digest": EXPECTED_PLAN_DIGEST,
            "start_cursor": 256,
            "terminal_cursor": 256,
            "progress_cursors": (),
            "system_digest": lifelong.persistent_lifelong_system_digest(system),
            "resume_skipped_completed_fit": True,
        }
        checkpoint = _checkpoint_record(
            CHECKPOINT_PATH,
            system,
            claim=claim_record,
            claim_sha256=claim_sha256,
            expected_cursor=256,
            final=True,
        )
        checkpoint["strict_reload_verified"] = True
    else:

        def progress_callback(
            callback_system: object,
            event: Mapping[str, object],
        ) -> None:
            wall = identity_wall(enforce_remaining=True)
            progress_records.append(
                _publish_progress(
                    callback_system,
                    event,
                    claim_record=claim_record,
                    claim_sha256=claim_sha256,
                    wall=wall,
                )
            )
            identity_wall(enforce_remaining=True)

        raw_fit = lifelong.fit_persistent_lifelong(
            system,
            progress_callback=progress_callback,
            deadline_callback=deadline_callback,
        )
        fit_report = _validate_fit_report(
            raw_fit,
            start_cursor=start_cursor,
            system=system,
        )
        terminal_wall = identity_wall(enforce_remaining=True)
        system, checkpoint = _save_reload_publish_final(
            system,
            claim_record=claim_record,
            claim_sha256=claim_sha256,
            wall=terminal_wall,
        )

    identity_wall(enforce_remaining=True)
    evaluation_started = time.perf_counter()
    pre_evaluation_digest = lifelong.persistent_lifelong_system_digest(system)
    evaluation = _validate_evaluation(
        lifelong.evaluate_persistent_lifelong(
            system,
            deadline_callback=deadline_callback,
        ),
        system,
    )
    torch.cuda.synchronize(0)
    evaluation_seconds = time.perf_counter() - evaluation_started
    identity_end = identity_wall(enforce_remaining=True)
    memory = _assert_memory_ceiling("terminal evaluation")
    terminal_digest = lifelong.persistent_lifelong_system_digest(system)
    checkpoint_summary = checkpoint.get("summary")
    if (
        terminal_digest != pre_evaluation_digest
        or not isinstance(checkpoint_summary, Mapping)
        or checkpoint_summary.get("system_digest") != terminal_digest
    ):
        raise RuntimeError("V21-A terminal evaluation changed checkpointed state")
    classification = str(evaluation["classification"])
    report = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "classification": classification,
        "passed": classification
        == "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED",
        "run_identity": {
            "one_shot": True,
            "resume_used": resume,
            "resume_source": resume_source,
            "start_cursor": start_cursor,
            "terminal_cursor": 256,
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "identity_wall": identity_end,
            "claim": {
                "path": claim["path"],
                "sha256": claim["sha256"],
                "bytes": claim["bytes"],
            },
            "stale_temporaries_removed_on_resume": tuple(
                stale_temporaries_removed
            ),
        },
        "source_integrity": {
            "repository_hashes": dict(launch["repository_hashes"]),
            "artifact_hashes": dict(launch["artifact_hashes"]),
            "source_report": dict(launch["source_report"]),
            "dependencies": dict(launch["dependencies"]),
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "plan_digest": EXPECTED_PLAN_DIGEST,
            "terminal_system_digest": terminal_digest,
        },
        "environment": _environment_record(cuda),
        "resources": {
            **memory,
            "allocated_memory_ceiling_bytes": MAX_ALLOCATED_BYTES,
            "identity_wall_ceiling_seconds": MAX_SEMANTIC_SECONDS,
        },
        "timings_seconds": {
            "evaluation": evaluation_seconds,
            "invocation_elapsed": time.perf_counter() - invocation_started,
            "identity_wall_elapsed": identity_end[
                "identity_wall_elapsed_seconds"
            ],
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
            "public_synthetic_only": True,
            "replay_used": False,
            "deterministic_solver_used": False,
            "identity_router_used": False,
            "slow_training_used": False,
            "new_parameter_used": False,
            "external_effects": False,
            "deployment_used": False,
            "promoted_state_changed": False,
        },
    }
    if _lexists(REPORT_PATH) or _lexists(FAILURE_PATH):
        raise RuntimeError("V21-A terminal identity became occupied")
    publication_wall = identity_wall(enforce_remaining=True)
    report["run_identity"]["identity_wall"] = publication_wall
    report["timings_seconds"]["identity_wall_elapsed"] = publication_wall[
        "identity_wall_elapsed_seconds"
    ]
    publication = _write_json_immutable(
        REPORT_PATH,
        REPORT_TEMP,
        report,
        before_publish=lambda: identity_wall(enforce_remaining=True),
    )
    if not REPORT_PATH.is_file() or _lexists(FAILURE_PATH):
        raise RuntimeError("V21-A report/failure exclusivity failed")
    print(
        json.dumps(
            {
                "event": "V21A_OML_PERSISTENT_COMPLETE",
                "protocol_id": PROTOCOL_ID,
                "classification": classification,
                "report": publication,
                "checkpoint": checkpoint,
                "identity_wall_elapsed_seconds": publication_wall[
                    "identity_wall_elapsed_seconds"
                ],
                "peak_allocated_bytes": memory["peak_allocated_bytes"],
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
    if _lexists(REPORT_PATH) or _lexists(FAILURE_PATH):
        return None
    # Only terminal publication temporaries are removed so failure publication
    # cannot overwrite them.  Checkpoint temporaries remain preserved evidence.
    for temporary in (REPORT_TEMP, FAILURE_TEMP):
        if _lexists(temporary):
            if (
                temporary.is_symlink()
                or temporary.resolve().parent != PREFIX.parent.resolve()
                or not temporary.is_file()
            ):
                return None
            temporary.unlink()
    _fsync_directory(PREFIX.parent)
    claim_record = claim.get("record")
    wall: object = None
    if isinstance(claim_record, Mapping):
        try:
            wall = _identity_wall_snapshot(
                claim_record,
                invocation_started=invocation_started,
                enforce_remaining=False,
            )
        except BaseException as clock_error:
            wall = {
                "status": "INVALID",
                "error_type": type(clock_error).__name__,
                "error": str(clock_error)[:16_384],
            }
    preserved = {}
    for name, path in (
        ("progress", PROGRESS_PATH),
        ("progress_temporary", PROGRESS_TEMP),
        ("checkpoint", CHECKPOINT_PATH),
        ("checkpoint_temporary", CHECKPOINT_TEMP),
    ):
        occupied = _lexists(path)
        if occupied and (path.is_symlink() or not path.is_file()):
            preserved[f"{name}_exists"] = True
            preserved[f"{name}_sha256"] = None
            preserved[f"{name}_bytes"] = None
            preserved[f"{name}_invalid_type"] = True
            continue
        exists = path.is_file()
        preserved[f"{name}_exists"] = exists
        preserved[f"{name}_sha256"] = sha256_file(path) if exists else None
        preserved[f"{name}_bytes"] = path.stat().st_size if exists else None
    failure = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "classification": "INVALID_NO_CLAIM",
        "failure_subtype": "HARNESS_ERROR_PRESERVED",
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
        "preserved_state": preserved,
        "identity_wall": wall,
        "timings_seconds": {
            "invocation_elapsed": time.perf_counter() - invocation_started,
        },
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
    publication = _write_json_immutable(FAILURE_PATH, FAILURE_TEMP, failure)
    if not FAILURE_PATH.is_file() or _lexists(REPORT_PATH):
        raise RuntimeError("V21-A failure/report exclusivity failed")
    return publication


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project Angler V21-A persistent OML one-shot"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _same_launch_identity(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> bool:
    keys = (
        "protocol_id",
        "plan_digest",
        "plan_identity",
        "repository_hashes",
        "artifact_hashes",
        "source_report",
        "dependencies",
        "output_state",
    )
    return all(before.get(key) == after.get(key) for key in keys)


def _resume_failure_context() -> tuple[dict[str, object], dict[str, object]]:
    """Capture bounded, untrusted observations when resume validation fails."""

    def observed_hash(path: Path) -> str | None:
        return sha256_file(path) if path.is_file() and not path.is_symlink() else None

    repository_hashes = {
        relative: observed_hash(ROOT / relative)
        for relative in EXPECTED_REPOSITORY_HASHES
    }
    artifact_hashes = {
        path: observed_hash(Path(path)) for path in EXPECTED_ARTIFACT_HASHES
    }
    claim_record: dict[str, object] = {}
    claim_sha256 = None
    claim_bytes = None
    if CLAIM_PATH.is_file() and not CLAIM_PATH.is_symlink():
        claim_sha256 = sha256_file(CLAIM_PATH)
        claim_bytes = CLAIM_PATH.stat().st_size
        try:
            claim_record = _read_json(CLAIM_PATH)
        except BaseException:
            claim_record = {}
    return (
        {
            "repository_hashes": repository_hashes,
            "artifact_hashes": artifact_hashes,
        },
        {
            "path": str(CLAIM_PATH),
            "sha256": claim_sha256,
            "bytes": claim_bytes,
            "record": claim_record,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify:
        result = verify_launch(resume=False)
        print(json.dumps(json_ready(result), sort_keys=True), flush=True)
        return 0
    if args.preflight:
        before = verify_launch(resume=False)
        preflight = run_cuda_preflight()
        after = verify_launch(resume=False)
        if not _same_launch_identity(before, after):
            raise RuntimeError("V21-A frozen bytes changed across preflight")
        print(
            json.dumps(
                json_ready({"launch": after, "preflight": preflight}),
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    resume = bool(args.resume)
    if resume:
        invocation_started = time.perf_counter()
        try:
            launch = verify_launch(resume=True)
            claim = _validate_existing_claim(launch)
        except BaseException as error:
            fallback_launch, fallback_claim = _resume_failure_context()
            failure = _preserve_failure(
                error,
                launch=fallback_launch,
                claim=fallback_claim,
                resume=True,
                invocation_started=invocation_started,
            )
            print(
                json.dumps(
                    {
                        "event": "V21A_RESUME_VALIDATION_FAILED",
                        "protocol_id": PROTOCOL_ID,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failure": failure,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 1
    else:
        before = verify_launch(resume=False)
        preflight = run_cuda_preflight()
        launch = verify_launch(resume=False)
        if not _same_launch_identity(before, launch):
            raise RuntimeError("V21-A frozen bytes changed before claim")
        claim = create_run_claim(launch, preflight)
        invocation_started = time.perf_counter()
    try:
        claim_lock = _acquire_claim_execution_lock()
    except BlockingIOError:
        print(
            json.dumps(
                {
                    "event": "V21A_ALREADY_ACTIVE",
                    "protocol_id": PROTOCOL_ID,
                    "claim": claim["path"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    except BaseException as error:
        failure = _preserve_failure(
            error,
            launch=launch,
            claim=claim,
            resume=resume,
            invocation_started=invocation_started,
        )
        print(
            json.dumps(
                {
                    "event": "V21A_LOCK_FAILED",
                    "protocol_id": PROTOCOL_ID,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failure": failure,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    stale_temporaries_removed: tuple[str, ...] = ()
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
            failure = _preserve_failure(
                error,
                launch=launch,
                claim=claim,
                resume=resume,
                invocation_started=invocation_started,
            )
            print(
                json.dumps(
                    {
                        "event": "V21A_OML_PERSISTENT_FAILED",
                        "protocol_id": PROTOCOL_ID,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failure": failure,
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
