from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from types import ModuleType


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


PROTOCOL_ID = "phase6.public-representation-overlap.v11-d2"
SOURCE_PROTOCOL_ID = "phase6.public-relation-credit.v11"
ROOT = Path(__file__).resolve().parent
RESULTS = Path("/opt/angler/results")
CHECKPOINT = RESULTS / "phase6-software-pipeline-reconstruction-v11-terminal.pt"
SOURCE_REPORT = RESULTS / "phase6-software-pipeline-reconstruction-v11-report.json"
D1_REPORT = RESULTS / "phase6-software-pipeline-reconstruction-v11-d1-report.json"
CLAIM = RESULTS / (
    "phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.claim.json"
)
REPORT = RESULTS / (
    "phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.json"
)
FAILURE = RESULTS / (
    "phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.failure.json"
)
REPORT_TEMP = REPORT.with_suffix(".json.tmp")
FAILURE_TEMP = FAILURE.with_suffix(".json.tmp")
SOURCE_ARCHIVE_ENV = "ANGLER_V11_D2_SOURCE_ARCHIVE"

PRESERVED_REF = "refs/angler-preserved/v11-d1-source-tree"
PRESERVED_TREE = "6d54d3fe66d7e27b30e550b65c94d3f82c22bb1f"
RUNNER_BLOB = "cf7fe45fb31531435a2b51e4485ec3137d40ed4e"
RUNNER_RELATIVE = Path("experiments/runners/phase6_software_pipeline_reconstruction.py")
RUNNER_SHA256 = "305A083B4A108E5CA3784BD8834DDA74CA813D6E06C25CE08F75C61AE39D0B01"
RUNNER_BYTES = 308_885
PRESERVED_ARCHIVE_SHA256 = (
    "BD19CA89B24DAEA935190DBCCF63CFDC342EF7CAAC85CD44930743C7C5D9CEAD"
)
PRESERVED_ARCHIVE_MTIME = "2000-01-01T00:00:00Z"
SUITE_SHA256 = "45D2282D5CC7FC504B817BA6ECB656B31DD568F85916B65E9145D1E1B0DFCE44"
CHECKPOINT_SHA256 = "2CF650BA5C9B62F1205CBA7F096CF9A078B752E699B98584FBC436FE1F5F0694"
SOURCE_REPORT_SHA256 = "EFDAC9461F34BE20226F54B718FB7A6F29375F74D9EFAD293D847867E071AE43"
D1_REPORT_SHA256 = "3191E1D9962A11BCCE9E8664E315D13CADB2F893C02994B7AA8B25233DF142BA"
TERMINAL_MODEL_DIGEST = (
    "sha256:3833c9a01e986d5d7206802969b909747e34a5136266b54c72546f28436d9581"
)
MAXIMUM_REPORT_BYTES = 262_144
MANUAL_SEED = 2_026_082_911

# Filled from the reviewed repository files.  The harness itself records its
# own hash in the claim because a source file cannot contain its own digest.
EXPECTED_REPOSITORY_HASHES = {
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-REPRESENTATION-OVERLAP-V11-D2-001.md": (
        "0C6275639A03E1FEAFBBB1C96A2E93A8B388107CAD975121BF14FF8D05CD674D"
    ),
    "experiments/evaluators/phase6_v11_representation_overlap.py": (
        "C044E1AB88F57FB2577BF36416C3921CB854E70C9AA809B6BB74ED9F86A2419C"
    ),
    "tests/unit/experiments/test_phase6_v11_representation_overlap.py": (
        "655A14722BD657EBE6FE2D974DEB199318E560CD23A3BADD0E3186776634494F"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def library_versions() -> dict[str, str]:
    values = {"python": platform.python_version(), "torch": torch.__version__}
    for distribution in ("numpy", "scipy", "pytest"):
        try:
            values[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            values[distribution] = "not-installed"
    return values


def write_bytes_atomic(path: Path, temporary: Path, payload: bytes) -> None:
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    members = archive.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError("V11-D2 source archive attempted path traversal")
    archive.extractall(destination, members=members, filter="data")


def _archive_git_tree(destination: Path) -> None:
    tree = subprocess.run(
        ("git", "-C", str(ROOT), "rev-parse", f"{PRESERVED_REF}^{{tree}}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tree != PRESERVED_TREE:
        raise RuntimeError("V11-D2 preserved Git tree changed")
    with tempfile.TemporaryFile() as archive_bytes:
        subprocess.run(
            (
                "git",
                "-c",
                "core.autocrlf=false",
                "-C",
                str(ROOT),
                "archive",
                "--format=tar",
                f"--mtime={PRESERVED_ARCHIVE_MTIME}",
                PRESERVED_REF,
            ),
            check=True,
            stdout=archive_bytes,
        )
        archive_bytes.seek(0)
        with tarfile.open(fileobj=archive_bytes, mode="r:") as archive:
            _safe_extract(archive, destination)


@contextmanager
def frozen_source_tree():
    archive_value = os.environ.get(SOURCE_ARCHIVE_ENV)
    with tempfile.TemporaryDirectory(prefix="angler-v11-d2-source-") as temporary:
        destination = Path(temporary)
        if archive_value:
            archive_path = Path(archive_value).resolve()
            if (
                not archive_path.is_file()
                or sha256_file(archive_path) != PRESERVED_ARCHIVE_SHA256
            ):
                raise RuntimeError("V11-D2 supplied source archive identity changed")
            with tarfile.open(archive_path, mode="r:*") as archive:
                _safe_extract(archive, destination)
        elif (ROOT / ".git").exists():
            _archive_git_tree(destination)
        else:
            raise RuntimeError(
                f"V11-D2 requires {SOURCE_ARCHIVE_ENV} when the execution copy has no Git metadata"
            )
        runner = destination / RUNNER_RELATIVE
        if (
            not runner.is_file()
            or runner.stat().st_size != RUNNER_BYTES
            or sha256_file(runner) != RUNNER_SHA256
        ):
            raise RuntimeError("V11-D2 exported runner identity changed")
        yield destination


def _load_frozen_source(source_root: Path) -> ModuleType:
    collisions = tuple(
        name
        for name in sys.modules
        if name == "angler"
        or name.startswith("angler.")
        or name == "experiments"
        or name.startswith("experiments.")
    )
    if collisions:
        raise RuntimeError(f"V11-D2 refused preloaded project modules: {collisions}")
    source_paths = (str(source_root / "src"), str(source_root))
    sys.path[0:0] = source_paths
    try:
        import experiments.runners.phase6_software_pipeline_reconstruction as source
    finally:
        if tuple(sys.path[:2]) == source_paths:
            del sys.path[:2]
    if Path(source.__file__).resolve() != (source_root / RUNNER_RELATIVE).resolve():
        raise RuntimeError("V11-D2 imported a mutable runner substitution")
    source_root_resolved = source_root.resolve()
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "angler"
            or name.startswith("angler.")
            or name == "experiments"
            or name.startswith("experiments.")
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved = Path(module_file).resolve()
        if resolved != source_root_resolved and source_root_resolved not in resolved.parents:
            raise RuntimeError(f"V11-D2 imported mutable project module {name}")
    suite = source_root / "experiments/evaluators/software_pipeline_reconstruction_suite.py"
    if not suite.is_file() or sha256_file(suite) != SUITE_SHA256:
        raise RuntimeError("V11-D2 frozen evaluator suite identity changed")
    return source


def _load_current_evaluator() -> ModuleType:
    path = ROOT / "experiments/evaluators/phase6_v11_representation_overlap.py"
    specification = importlib.util.spec_from_file_location(
        "angler_v11_d2_representation_overlap_evaluator",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("V11-D2 evaluator could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_launch() -> dict[str, object]:
    occupied = tuple(
        str(path)
        for path in (CLAIM, REPORT, FAILURE, REPORT_TEMP, FAILURE_TEMP)
        if path.exists()
    )
    if occupied:
        raise RuntimeError(f"V11-D2 one-shot output identity is occupied: {occupied}")
    repository_hashes = {
        relative: sha256_file(ROOT / relative) for relative in EXPECTED_REPOSITORY_HASHES
    }
    if repository_hashes != EXPECTED_REPOSITORY_HASHES:
        raise RuntimeError("V11-D2 reviewed repository input changed")
    evidence = {
        CHECKPOINT: CHECKPOINT_SHA256,
        SOURCE_REPORT: SOURCE_REPORT_SHA256,
        D1_REPORT: D1_REPORT_SHA256,
    }
    observed_evidence = {}
    for path, expected in evidence.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V11-D2 immutable evidence changed: {path}")
        observed_evidence[str(path)] = {
            "sha256": expected,
            "bytes": path.stat().st_size,
        }
    archive_value = os.environ.get(SOURCE_ARCHIVE_ENV)
    if archive_value:
        archive_path = Path(archive_value).resolve()
        if (
            not archive_path.is_file()
            or sha256_file(archive_path) != PRESERVED_ARCHIVE_SHA256
        ):
            raise RuntimeError("V11-D2 supplied source archive identity changed")
        source_delivery = {
            "mode": "exact_git_archive",
            "path": str(archive_path),
            "sha256": PRESERVED_ARCHIVE_SHA256,
            "bytes": archive_path.stat().st_size,
            "creation_rule": (
                "git -c core.autocrlf=false archive --format=tar "
                f"--mtime={PRESERVED_ARCHIVE_MTIME} {PRESERVED_REF}"
            ),
        }
    elif (ROOT / ".git").exists():
        tree = subprocess.run(
            ("git", "-C", str(ROOT), "rev-parse", f"{PRESERVED_REF}^{{tree}}"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tree != PRESERVED_TREE:
            raise RuntimeError("V11-D2 preserved Git tree changed")
        source_delivery = {"mode": "local_preserved_ref", "tree": tree}
    else:
        raise RuntimeError(
            f"V11-D2 requires {SOURCE_ARCHIVE_ENV} when the execution copy has no Git metadata"
        )
    return {
        "repository_hashes": repository_hashes,
        "evidence": observed_evidence,
        "preserved_ref": PRESERVED_REF,
        "preserved_tree": PRESERVED_TREE,
        "runner_blob": RUNNER_BLOB,
        "preserved_archive_sha256": PRESERVED_ARCHIVE_SHA256,
        "preserved_archive_mtime": PRESERVED_ARCHIVE_MTIME,
        "source_delivery": source_delivery,
    }


def create_claim(launch: Mapping[str, object]) -> dict[str, object]:
    claim = {
        "artifact_schema": "angler.phase6-v11-d2-representation-overlap-claim.v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "process_id": os.getpid(),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "launch_integrity": dict(launch),
        "device": "cpu",
        "torch_threads": 1,
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "checkpoint_writes": 0,
        "one_shot": True,
    }
    write_json_exclusive(CLAIM, claim)
    return {
        "path": str(CLAIM),
        "sha256": sha256_file(CLAIM),
        "bytes": CLAIM.stat().st_size,
        "preserve_permanently": True,
    }


def _streams_from_d1(source: ModuleType, d1: Mapping[str, object]) -> dict[str, object]:
    identity = d1["identity"]
    commitments = tuple(identity["anonymous_commitments"])
    cells = {}
    for cell_id in ("t0_s0", "t0_s1", "t1_s0", "t1_s1"):
        seed_pairs = tuple(
            tuple(int(value) for value in pair)
            for pair in identity["seed_grid"][cell_id]["seed_pairs"]
        )
        cells[cell_id] = source._relation_credit_panel_streams(commitments, seed_pairs)
    return cells


def main() -> int:
    launch = verify_launch()
    started_utc = utc_now()
    started = time.perf_counter()
    claim = create_claim(launch)
    evaluator = None
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(MANUAL_SEED)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("V11-D2 failed to establish one-thread execution")
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
        source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
        d1_report = json.loads(D1_REPORT.read_text(encoding="utf-8"))
        evaluator = _load_current_evaluator()
        if evaluator.PROTOCOL_ID != PROTOCOL_ID:
            raise RuntimeError("V11-D2 evaluator protocol changed")
        frozen_identity = {
            "preserved_tree": PRESERVED_TREE,
            "runner_blob": RUNNER_BLOB,
            "runner_sha256": RUNNER_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_report_sha256": SOURCE_REPORT_SHA256,
            "d1_report_sha256": D1_REPORT_SHA256,
            "terminal_model_digest": TERMINAL_MODEL_DIGEST,
        }
        with frozen_source_tree() as source_root:
            source = _load_frozen_source(source_root)
            if getattr(source, "_RELATION_PROTOCOL_ID", None) != SOURCE_PROTOCOL_ID:
                raise RuntimeError("V11-D2 frozen source protocol changed")
            controller, _competence_state = source.load_software_pipeline_checkpoint(
                CHECKPOINT,
                device="cpu",
            )
            if source.software_pipeline_model_digest(controller) != TERMINAL_MODEL_DIGEST:
                raise RuntimeError("V11-D2 checkpoint model digest changed")
            streams = _streams_from_d1(source, d1_report)
            evaluation_started = time.perf_counter()
            evaluation = evaluator.evaluate_v11_representation_overlap(
                source,
                controller,
                source_report,
                d1_report,
                streams,
                frozen_identity,
            )
            evaluation_seconds = time.perf_counter() - evaluation_started
            if source.software_pipeline_model_digest(controller) != TERMINAL_MODEL_DIGEST:
                raise RuntimeError("V11-D2 changed the terminal model")
        if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
            raise RuntimeError("V11-D2 changed the terminal checkpoint")
        if sha256_file(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
            raise RuntimeError("V11-D2 changed the V11 report")
        if sha256_file(D1_REPORT) != D1_REPORT_SHA256:
            raise RuntimeError("V11-D2 changed the V11-D1 report")
        if torch.cuda.is_initialized():
            raise RuntimeError("V11-D2 CPU diagnostic initialized CUDA")

        report = {
            "artifact_schema": "angler.phase6-v11-d2-representation-overlap-report.v1",
            "protocol_id": PROTOCOL_ID,
            "classification": evaluation["classification"],
            "completed_utc": utc_now(),
            "first_result_accepted_without_tuning": True,
            "run_claim": claim,
            "environment": environment,
            "source_integrity": {
                "launch": launch,
                "harness_sha256": sha256_file(Path(__file__).resolve()),
                "checkpoint_sha256_after": sha256_file(CHECKPOINT),
                "source_report_sha256_after": sha256_file(SOURCE_REPORT),
                "d1_report_sha256_after": sha256_file(D1_REPORT),
                "temporary_source_removed_before_publication": True,
            },
            "timings_seconds": {
                "diagnostic": evaluation_seconds,
                "harness_total": time.perf_counter() - started,
            },
            "evaluation": evaluation,
            "scope_and_effects": {
                "gpu_used": False,
                "network_used": False,
                "package_install_used": False,
                "optimizer_or_parameter_update_used": False,
                "checkpoint_created_or_changed": False,
                "private_or_answer_data_used": False,
                "external_effects": False,
            },
        }
        payload = evaluator.serialize_bounded_report(
            report,
            maximum_bytes=MAXIMUM_REPORT_BYTES,
        )
        write_bytes_atomic(REPORT, REPORT_TEMP, payload)
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "event": "representation_overlap_completed",
                    "classification": evaluation["classification"],
                    "report": {
                        "path": str(REPORT),
                        "sha256": sha256_file(REPORT),
                        "bytes": REPORT.stat().st_size,
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
            "artifact_schema": "angler.phase6-v11-d2-representation-overlap-failure.v1",
            "protocol_id": PROTOCOL_ID,
            "classification": "REPRESENTATION_OVERLAP_HARNESS_ERROR_PRESERVED",
            "failed_utc": utc_now(),
            "run_claim": claim,
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
                "traceback": traceback.format_exc()[-16_384:],
            },
            "timings_seconds": {"harness_total": time.perf_counter() - started},
        }
        payload = (
            json.dumps(failure, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(payload) > MAXIMUM_REPORT_BYTES:
            raise RuntimeError("V11-D2 failure JSON exceeded its byte ceiling") from error
        # Once the terminal report has been published it is authoritative.
        # A nonsemantic stdout/hash/stat failure after publication must not
        # create a contradictory second terminal artifact.
        if (
            not REPORT.exists()
            and not FAILURE.exists()
            and not FAILURE_TEMP.exists()
        ):
            write_bytes_atomic(FAILURE, FAILURE_TEMP, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
