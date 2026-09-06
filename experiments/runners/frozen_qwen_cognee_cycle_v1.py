"""Bounded live qualification for the frozen-Qwen/Cognee successor cycle.

This is a technical composition witness, not a reasoning benchmark.  It uses
one authored synthetic live task plus one unrelated observed fixture seed,
accepts whatever objective disposition the external exact verifier observes,
and proves durable state/restart boundaries.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version as distribution_version
import io
import json
import os
import platform
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import torch

from angler.cognition.contracts import CognitiveEpisode, ProspectiveCommitment
from angler.cognition.prospective import ObjectiveFeedbackRecord
from angler.memory import (
    AcquisitionSituatedMemory,
    CogneeAcquisitionAdapter,
)
from angler.memory.cognee_subprocess_bindings import CogneeSubprocessBindings
from angler.memory.cognee_worker_protocol import (
    COGNEE_PYTHON,
    COGNEE_PYTHON_VERSION,
    SOFTWARE_VERSIONS as COGNEE_SOFTWARE_VERSIONS,
)
from angler.reasoning import (
    CompositeProspectiveCreditCore,
    ProspectiveDynamicsConfig,
    StructureKeyedCreditMemoryCore,
)
from angler.runtime import (
    CognitiveCycle,
    CognitiveTransactionStore,
    DurableAbilityLearner,
    FrozenQwenCognitiveExecutorV1,
    FrozenQwenCycleManifestV1,
    FrozenQwenProcedureAdapterV1,
    FrozenQwenProcedureRelationAdapterV1,
    LocalQwenIO,
    QwenReceiptObservedStateEncoderV1,
    SQLiteQwenExecutionJournal,
    foundation_tensor_digest,
    parse_qwen_procedure_proposals,
)


IDENTITY = "angler.frozen-qwen-cognee-cycle.v1-qualification"
HOST = "angler-workstation"
EXPECTED_OS_ID = "ubuntu"
EXPECTED_OS_VERSION = "24.04"
EXPECTED_OS_PRETTY_NAME = "Ubuntu 24.04.4 LTS"
EXPECTED_KERNEL = "7.0.0-30-generic"
EXPECTED_MACHINE = "x86_64"
EXPECTED_CPU_MODEL = "13th Gen Intel(R) Core(TM) i7-13700K"
EXPECTED_LOGICAL_CPUS = 24
EXPECTED_MEMORY_TOTAL_KIB = 65_596_216
MINIMUM_MEMORY_AVAILABLE_BYTES = 32 * 1024 * 1024 * 1024
EXPECTED_STORAGE_SOURCE = "/dev/nvme0n1p2"
EXPECTED_STORAGE_FSTYPE = "ext4"
EXPECTED_STORAGE_BYTES = 1_966_736_678_912
MINIMUM_STORAGE_AVAILABLE_BYTES = 1_600 * 1024 * 1024 * 1024
GPU_UUID = "GPU-df4bb978-e75f-08a0-6660-2b9ed69ee8ca"
GPU_NAME = "NVIDIA GeForce RTX 5080"
GPU_PCI = "00000000:01:00.0"
GPU_COMPUTE_CAPABILITY = "12.0"
GPU_TOTAL_MIB = 16_303
UNASSIGNED_GPU_UUID = "GPU-d9dd1ae0-f65d-ef22-f924-2c3e9c976c1e"
UNASSIGNED_GPU_NAME = "NVIDIA GeForce RTX 5070"
UNASSIGNED_GPU_PCI = "00000000:05:00.0"
UNASSIGNED_GPU_COMPUTE_CAPABILITY = "12.0"
UNASSIGNED_GPU_TOTAL_MIB = 12_227
MINIMUM_FREE_MIB = 14_336
MAXIMUM_CUDA_BYTES = 12_288 * 1024 * 1024
MAXIMUM_RSS_BYTES = 20 * 1024 * 1024 * 1024
MAXIMUM_WALL_SECONDS = 900.0
MAXIMUM_NEW_OUTPUT_BYTES = 1024 * 1024 * 1024

EXPECTED_PYTHON = "3.12.3"
EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_CUDNN_RUNTIME = 92_000
EXPECTED_DISTRIBUTIONS = {
    "accelerate": "1.14.0",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-cudnn-cu13": "9.20.0.48",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.1",
    "triton": "3.7.1",
}

MODEL_ROOT = Path("/opt/angler/models/Qwen3-4B")
MODEL_ROOT_MANIFEST = (
    "1ac705236348881b2fd46f4075b931b5137369d0c469b871aea749f6e0886d83"
)
TOKENIZER_MANIFEST = (
    "ba21c0913e7aa6dabc7b969aa5b8ced17de450b1b1169f9c474d206cbf485430"
)
MODEL_FILE_COUNT = 13
MODEL_TOTAL_BYTES = 8_060_926_626
TOKENIZER_FILES = (
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

STATE_ROOT = Path(
    "/opt/angler/state/project-angler/frozen-qwen-cognee-cycle-v1"
)
GENESIS_PATH = STATE_ROOT / "learner-genesis.pt"
STORE_PATH = STATE_ROOT / "cognitive.sqlite3"
JOURNAL_PATH = STATE_ROOT / "qwen-execution-journal.sqlite3"
RESULT_PATH = Path("/opt/angler/results/frozen-qwen-cognee-cycle-v1.json")
SCRATCH_ROOT = Path("/opt/angler/work/frozen-qwen-cognee-cycle-v1")
REPOSITORY_ROOT = Path("/opt/angler/src/angler")
RUNNER_PATH = REPOSITORY_ROOT / "experiments/runners/frozen_qwen_cognee_cycle_v1.py"
ANGLER_PYTHON = "/opt/angler/venvs/angler/bin/python"
PARENT_ENVIRONMENT = {
    "CUDA_CACHE_PATH": str(SCRATCH_ROOT / "cuda-cache"),
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": GPU_UUID,
    "DO_NOT_TRACK": "1",
    "HF_HOME": str(SCRATCH_ROOT / "huggingface"),
    "HF_HUB_CACHE": str(SCRATCH_ROOT / "huggingface/hub"),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": (
        "/opt/angler/venvs/angler/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
    "TELEMETRY_DISABLED": "1",
    "TMPDIR": str(SCRATCH_ROOT / "tmp"),
    "TOKENIZERS_PARALLELISM": "false",
    "TORCH_HOME": str(SCRATCH_ROOT / "torch"),
    "TORCHINDUCTOR_CACHE_DIR": str(SCRATCH_ROOT / "torchinductor"),
    "TRANSFORMERS_OFFLINE": "1",
    "TRITON_CACHE_DIR": str(SCRATCH_ROOT / "triton"),
    "XDG_CACHE_HOME": str(SCRATCH_ROOT / "xdg-cache"),
}
GENESIS_HELPER_TIMEOUT_SECONDS = 120
MAXIMUM_GENESIS_HELPER_BYTES = 2 * 1024 * 1024
GENESIS_HELPER_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "LC_CTYPE": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
    "TMPDIR": str(SCRATCH_ROOT / "genesis-tmp"),
    "TORCHINDUCTOR_CACHE_DIR": str(SCRATCH_ROOT / "genesis-torchinductor"),
}
COGNEE_STATE_ROOT = Path(
    "/opt/angler/state/project-angler/cognee-acquisition-live-v1"
)
FASTEMBED_CACHE_ROOT = Path("/opt/angler/models/fastembed-cache-v1")
FASTEMBED_REPOSITORY = "models--qdrant--bge-small-en-v1.5-onnx-q"
FASTEMBED_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
FASTEMBED_FILES = (
    "config.json",
    "model_optimized.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
FASTEMBED_TOTAL_BYTES = 67_179_163
FASTEMBED_MANIFEST_SHA256 = (
    "950932f40bdea47546ccc71dfb87585d1c2c74bf8f1e7d920ac0fddf53b1c148"
)
FASTEMBED_ONNX_BYTES = 66_465_124
FASTEMBED_ONNX_SHA256 = (
    "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
)
TIKTOKEN_CACHE_ROOT = Path(
    "/opt/angler/venvs/cognee/lib/python3.12/site-packages/"
    "litellm/litellm_core_utils/tokenizers"
)

GENESIS_CONFIG_REF = (
    "sha256:ec5df84ea8e3946ce9e2bc387ef874c5bd719d962ee0c9c08bd2ab6a279c05b7"
)
PROSPECTIVE_CONFIG_REF = (
    "sha256:f3bab0f86a3671588a3bd66538448cf276e3a6e965d7d866f7698a092d1e7313"
)
LEARNER_CHECKPOINT_REF = (
    "sha256:5b2de1bb50091570dd92a790fe179cf3681f1202a1660d843863c46845f14f4c"
)
INITIAL_COMPETENCE_DIGEST = (
    "sha256:5a8bac6f60ba9d0d73c4cbc3965479a7f1b1f8d4fe4b3e9c9bb4530a924a4286"
)
GENESIS_SNAPSHOT_SHA256 = (
    "93ab9deee7b0b67eabade8d43bb2b09e466f1dcd25405931d5a3d55eca91902d"
)
GENESIS_SNAPSHOT_BYTES = 139_913
GENESIS_SEED = 2_026_083_101

TASK_ID = "frozen-qwen-cognee-cycle-v1-synthetic"
TASK = "Return exactly the text ANGLER-QUALIFIED-V1 and nothing else."
EXPECTED_RESPONSE = "ANGLER-QUALIFIED-V1"


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1_048_576)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _bundle_manifest(root: Path, names: tuple[str, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for name in sorted(names, key=lambda value: value.encode("utf-8")):
        file_digest, size = _sha256_file(root / name)
        total += size
        digest.update(f"{file_digest}\t{size}\t{name}\n".encode("utf-8"))
    return digest.hexdigest(), total


def _tree_manifest(root: Path) -> dict[str, object]:
    """Hash one local tree without following directory symlinks."""

    digest = hashlib.sha256()
    total = 0
    files = 0
    if not root.exists():
        digest.update(b"ABSENT\n")
        return {
            "exists": False,
            "files": 0,
            "total_bytes": 0,
            "manifest_sha256": digest.hexdigest(),
        }
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"guarded tree is not an exact directory: {root}")
    paths = sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            digest.update(f"L\t{relative}\t{target}\n".encode("utf-8"))
        elif path.is_dir():
            digest.update(f"D\t{relative}\n".encode("utf-8"))
        elif path.is_file():
            file_digest, size = _sha256_file(path)
            digest.update(
                f"F\t{file_digest}\t{size}\t{relative}\n".encode("utf-8")
            )
            files += 1
            total += size
        else:
            raise RuntimeError(f"guarded tree contains a special path: {path}")
    return {
        "exists": True,
        "files": files,
        "total_bytes": total,
        "manifest_sha256": digest.hexdigest(),
    }


def _memory_inventory() -> tuple[int, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="ascii") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) == 3 and fields[0] in ("MemTotal:", "MemAvailable:"):
                values[fields[0]] = int(fields[1])
    try:
        return values["MemTotal:"], values["MemAvailable:"]
    except KeyError as exc:
        raise RuntimeError("Linux memory inventory is incomplete") from exc


def _host_preflight() -> dict[str, object]:
    release = platform.freedesktop_os_release()
    cpu_model = None
    with Path("/proc/cpuinfo").open("r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    total_kib, available_kib = _memory_inventory()
    completed = subprocess.run(
        (
            "/usr/bin/findmnt",
            "--bytes",
            "--noheadings",
            "--output",
            "SOURCE,FSTYPE,SIZE",
            "--target",
            str(REPOSITORY_ROOT),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    storage = tuple(completed.stdout.split())
    if len(storage) != 3:
        raise RuntimeError("working-storage inventory is malformed")
    source, filesystem, raw_size = storage
    storage_bytes = int(raw_size)
    storage_status = os.statvfs(REPOSITORY_ROOT)
    available_bytes = int(storage_status.f_bavail * storage_status.f_frsize)
    observed = (
        socket.gethostname(),
        release.get("ID"),
        release.get("VERSION_ID"),
        release.get("PRETTY_NAME"),
        platform.release(),
        platform.machine(),
        cpu_model,
        os.cpu_count(),
        total_kib,
        source,
        filesystem,
        storage_bytes,
    )
    expected = (
        HOST,
        EXPECTED_OS_ID,
        EXPECTED_OS_VERSION,
        EXPECTED_OS_PRETTY_NAME,
        EXPECTED_KERNEL,
        EXPECTED_MACHINE,
        EXPECTED_CPU_MODEL,
        EXPECTED_LOGICAL_CPUS,
        EXPECTED_MEMORY_TOTAL_KIB,
        EXPECTED_STORAGE_SOURCE,
        EXPECTED_STORAGE_FSTYPE,
        EXPECTED_STORAGE_BYTES,
    )
    if observed != expected:
        raise RuntimeError("frozen workstation inventory differs")
    if available_kib * 1024 < MINIMUM_MEMORY_AVAILABLE_BYTES:
        raise RuntimeError("workstation available memory is below the qualification floor")
    if available_bytes < MINIMUM_STORAGE_AVAILABLE_BYTES:
        raise RuntimeError("working storage is below the qualification floor")
    return {
        "hostname": observed[0],
        "os_id": observed[1],
        "os_version": observed[2],
        "os_pretty_name": observed[3],
        "kernel": observed[4],
        "machine": observed[5],
        "cpu_model": observed[6],
        "logical_cpus": observed[7],
        "memory_total_kib": total_kib,
        "memory_available_bytes": available_kib * 1024,
        "storage_source": source,
        "storage_fstype": filesystem,
        "storage_bytes": storage_bytes,
        "storage_available_bytes": available_bytes,
    }


def _software_preflight() -> dict[str, object]:
    try:
        distributions = {
            name: distribution_version(name) for name in EXPECTED_DISTRIBUTIONS
        }
    except PackageNotFoundError as exc:
        raise RuntimeError("frozen Angler software distribution is absent") from exc
    if distributions != EXPECTED_DISTRIBUTIONS:
        raise RuntimeError("frozen Angler software versions differ")
    if (
        Path(sys.executable) != Path("/opt/angler/venvs/angler/bin/python")
        or platform.python_version() != EXPECTED_PYTHON
        or torch.__version__ != EXPECTED_TORCH
        or torch.version.cuda != EXPECTED_CUDA
        or torch.backends.cudnn.version() != EXPECTED_CUDNN_RUNTIME
    ):
        raise RuntimeError("frozen Python, Torch, CUDA, or cuDNN identity differs")
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_runtime": torch.backends.cudnn.version(),
        "distributions": distributions,
        "cognee_python_executable": COGNEE_PYTHON,
        "cognee_python": COGNEE_PYTHON_VERSION,
        "cognee_worker_distributions": dict(COGNEE_SOFTWARE_VERSIONS),
    }


def _verify_embedding_files() -> dict[str, object]:
    repository = FASTEMBED_CACHE_ROOT / FASTEMBED_REPOSITORY
    snapshot = repository / "snapshots" / FASTEMBED_REVISION
    for directory in ("blobs", "refs", "snapshots", "trees"):
        if not (repository / directory).is_dir():
            raise RuntimeError("frozen FastEmbed cache tree is incomplete")
    if (repository / "refs" / "main").read_bytes() != FASTEMBED_REVISION.encode(
        "ascii"
    ):
        raise RuntimeError("frozen FastEmbed revision binding differs")
    if not snapshot.is_dir() or tuple(
        sorted(item.name for item in snapshot.iterdir())
    ) != tuple(sorted(FASTEMBED_FILES)):
        raise RuntimeError("frozen FastEmbed snapshot file set differs")
    digest = hashlib.sha256()
    total = 0
    observed: dict[str, tuple[str, int]] = {}
    for name in FASTEMBED_FILES:
        file_digest, size = _sha256_file(snapshot / name)
        observed[name] = file_digest, size
        total += size
        digest.update(f"{file_digest}  {name}\n".encode("ascii"))
    if (
        total != FASTEMBED_TOTAL_BYTES
        or digest.hexdigest() != FASTEMBED_MANIFEST_SHA256
        or observed["model_optimized.onnx"]
        != (FASTEMBED_ONNX_SHA256, FASTEMBED_ONNX_BYTES)
    ):
        raise RuntimeError("frozen FastEmbed model identity differs")
    return {
        "revision": FASTEMBED_REVISION,
        "snapshot_files": len(FASTEMBED_FILES),
        "snapshot_bytes": total,
        "snapshot_manifest_sha256": digest.hexdigest(),
        "cache_tree": _tree_manifest(FASTEMBED_CACHE_ROOT),
    }


def _validate_early_scratch() -> None:
    """Admit only the empty declared temp/import cache dirs before the gate."""

    if not SCRATCH_ROOT.exists():
        return
    if SCRATCH_ROOT.is_symlink() or not SCRATCH_ROOT.is_dir():
        raise RuntimeError("qualification scratch root is not an exact directory")
    root_metadata = SCRATCH_ROOT.stat()
    if (
        root_metadata.st_uid != os.getuid()
        or root_metadata.st_gid != os.getgid()
        or root_metadata.st_mode & 0o7777 != 0o700
    ):
        raise RuntimeError("qualification scratch root metadata differs")
    entries = tuple(SCRATCH_ROOT.iterdir())
    if {item.name for item in entries} - {"cuda-cache", "tmp", "torchinductor"}:
        raise RuntimeError("qualification scratch root contains prior state")
    for item in entries:
        metadata = item.stat()
        if (
            item.is_symlink()
            or not item.is_dir()
            or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or any(item.iterdir())
        ):
            raise RuntimeError("qualification import-time scratch is not empty")


def _write_scope_baseline() -> dict[str, dict[str, object]]:
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    if RESULT_PATH.exists() or temporary.exists():
        raise RuntimeError("qualification result path is not fresh")
    if STATE_ROOT.exists():
        names = {item.name for item in STATE_ROOT.iterdir()}
        if not names.issubset({GENESIS_PATH.name}):
            raise RuntimeError("qualification state root contains undeclared state")
    _validate_early_scratch()
    return {
        "state": _tree_manifest(STATE_ROOT),
        "cognee": _tree_manifest(COGNEE_STATE_ROOT),
        "scratch": _tree_manifest(SCRATCH_ROOT),
    }


def _verify_write_scope(
    baseline: dict[str, dict[str, object]],
    *,
    repository_before: dict[str, object],
    embedding_before: dict[str, object],
    tiktoken_before: dict[str, object],
) -> dict[str, object]:
    repository_after = _tree_manifest(REPOSITORY_ROOT)
    embedding_after = _verify_embedding_files()
    tiktoken_after = _tree_manifest(TIKTOKEN_CACHE_ROOT)
    if repository_after != repository_before:
        raise RuntimeError("repository changed during the live qualification")
    if embedding_after != embedding_before:
        raise RuntimeError("FastEmbed cache changed during the live qualification")
    if tiktoken_after != tiktoken_before:
        raise RuntimeError("Tiktoken cache changed during the live qualification")
    state_names = {item.name for item in STATE_ROOT.iterdir()}
    if state_names != {GENESIS_PATH.name, STORE_PATH.name, JOURNAL_PATH.name}:
        raise RuntimeError("qualification state root contains an undeclared output")
    after = {
        "state": _tree_manifest(STATE_ROOT),
        "cognee": _tree_manifest(COGNEE_STATE_ROOT),
        "scratch": _tree_manifest(SCRATCH_ROOT),
    }
    new_bytes = sum(
        max(
            0,
            int(after[name]["total_bytes"])
            - int(baseline[name]["total_bytes"]),
        )
        for name in after
    )
    if new_bytes > MAXIMUM_NEW_OUTPUT_BYTES:
        raise RuntimeError("qualification outputs exceed the one-GiB ceiling")
    return {
        "repository_before": repository_before,
        "repository_after": repository_after,
        "fastembed_before": embedding_before,
        "fastembed_after": embedding_after,
        "tiktoken_before": tiktoken_before,
        "tiktoken_after": tiktoken_after,
        "output_baseline": baseline,
        "output_after": after,
        "new_bytes_before_result": new_bytes,
    }


def _verify_model_files() -> dict[str, object]:
    if not MODEL_ROOT.is_dir():
        raise RuntimeError("frozen Qwen root is absent")
    root_names = tuple(
        item.name for item in MODEL_ROOT.iterdir() if item.is_file()
    )
    manifest, total = _bundle_manifest(MODEL_ROOT, root_names)
    tokenizer_manifest, _ = _bundle_manifest(MODEL_ROOT, TOKENIZER_FILES)
    if (
        len(root_names) != MODEL_FILE_COUNT
        or total != MODEL_TOTAL_BYTES
        or manifest != MODEL_ROOT_MANIFEST
        or tokenizer_manifest != TOKENIZER_MANIFEST
    ):
        raise RuntimeError("frozen Qwen file identity differs")
    return {
        "file_count": len(root_names),
        "total_bytes": total,
        "root_manifest_sha256": manifest,
        "tokenizer_manifest_sha256": tokenizer_manifest,
    }


def _gpu_preflight() -> dict[str, object]:
    if socket.gethostname() != HOST:
        raise RuntimeError("qualification host differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the exact RTX 5080 UUID")
    command = (
        "/usr/bin/nvidia-smi",
        "--query-gpu=uuid,name,pci.bus_id,compute_cap,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: dict[str, tuple[str, str, str, int, int]] = {}
    for line in completed.stdout.splitlines():
        values = tuple(item.strip() for item in line.split(","))
        if len(values) != 6:
            raise RuntimeError("nvidia-smi inventory is malformed")
        if values[0] in rows:
            raise RuntimeError("nvidia-smi inventory contains a duplicate GPU UUID")
        rows[values[0]] = (
            values[1],
            values[2],
            values[3],
            int(values[4]),
            int(values[5]),
        )
    expected = {
        GPU_UUID: (
            GPU_NAME,
            GPU_PCI,
            GPU_COMPUTE_CAPABILITY,
            GPU_TOTAL_MIB,
        ),
        UNASSIGNED_GPU_UUID: (
            UNASSIGNED_GPU_NAME,
            UNASSIGNED_GPU_PCI,
            UNASSIGNED_GPU_COMPUTE_CAPABILITY,
            UNASSIGNED_GPU_TOTAL_MIB,
        ),
    }
    if set(rows) != set(expected):
        raise RuntimeError("physical GPU UUID inventory differs")
    for uuid, identity in expected.items():
        if rows[uuid][:4] != identity:
            raise RuntimeError(f"physical GPU identity differs for {uuid}")
    assigned_free_mib = rows[GPU_UUID][4]
    if assigned_free_mib < MINIMUM_FREE_MIB:
        raise RuntimeError("assigned RTX 5080 free-memory preflight failed")
    return {
        "assigned": {
            "uuid": GPU_UUID,
            "name": GPU_NAME,
            "pci_bus_id": GPU_PCI,
            "compute_capability": GPU_COMPUTE_CAPABILITY,
            "total_mib": GPU_TOTAL_MIB,
            "free_mib": assigned_free_mib,
            "logical_device": "cuda:0",
        },
        "unassigned": {
            "uuid": UNASSIGNED_GPU_UUID,
            "name": UNASSIGNED_GPU_NAME,
            "pci_bus_id": UNASSIGNED_GPU_PCI,
            "compute_capability": UNASSIGNED_GPU_COMPUTE_CAPABILITY,
            "total_mib": UNASSIGNED_GPU_TOTAL_MIB,
            "free_mib": rows[UNASSIGNED_GPU_UUID][4],
        },
    }


def _genesis_record() -> dict[str, object]:
    genesis_record = {
        "checkpoint_constructor": (
            "CompositeProspectiveCreditCore(StructureKeyedCreditMemoryCore(**credit_config), "
            "ProspectiveDynamicsConfig(**prospective_config))"
        ),
        "credit_config": {
            "maximum_residual": 4.0,
            "memory_slots": 512,
            "rank": 32,
            "relation_width": 64,
            "temporal_width": 8,
        },
        "device": "cpu",
        "dtype": "torch.float32",
        "initial_state_constructor": "core.initial_state()",
        "prospective_config": {
            "latent_width": 16,
            "maximum_branches": 64,
            "maximum_recurrent_steps": 4,
            "maximum_residual": 2.0,
            "maximum_update_backtracks": 12,
            "maximum_world_reads": 4,
            "relation_width": 64,
            "temporal_width": 8,
            "update_rate": 0.5,
            "world_slots": 8,
        },
        "prospective_lesion": False,
        "schema": "angler.frozen-qwen-learner-genesis.v1",
        "torch_manual_seed": GENESIS_SEED,
    }
    material = json.dumps(
        genesis_record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    derived_genesis_ref = "sha256:" + hashlib.sha256(
        b"angler.frozen-qwen-learner-genesis-config.v1\0" + material
    ).hexdigest()
    if derived_genesis_ref != GENESIS_CONFIG_REF:
        raise RuntimeError("learner genesis configuration record differs")
    return genesis_record


def _build_genesis_core() -> CompositeProspectiveCreditCore:
    """Construct the exact CPU topology without changing the parent RNG policy."""

    genesis_record = _genesis_record()
    credit_config = genesis_record["credit_config"]
    prospective_config = genesis_record["prospective_config"]
    if type(credit_config) is not dict or type(prospective_config) is not dict:
        raise RuntimeError("learner genesis configuration shape differs")
    # Ordering is identity-bearing: the credit core consumes RNG first.
    credit = StructureKeyedCreditMemoryCore(**credit_config)
    config = ProspectiveDynamicsConfig(**prospective_config)
    core = CompositeProspectiveCreditCore(credit, config)
    if (
        "sha256:" + config.digest != PROSPECTIVE_CONFIG_REF
        or core.parameter_count != 32_484
    ):
        raise RuntimeError("learner genesis topology differs")
    return core


def _new_genesis_learner(
    core: CompositeProspectiveCreditCore,
) -> DurableAbilityLearner:
    return DurableAbilityLearner(
        core,
        core.initial_state(),
        checkpoint_identity=core.checkpoint_identity,
        prospective_lesion=False,
    )


def _verify_exact_genesis_learner(
    learner: DurableAbilityLearner,
    *,
    expected_snapshot: bytes | None = None,
) -> bytes:
    core = learner.core
    if type(core) is not CompositeProspectiveCreditCore:
        raise RuntimeError("learner genesis core class differs")
    integrity = learner.component_state_integrity()
    snapshot = learner.capture_state()
    if (
        core.checkpoint_identity != LEARNER_CHECKPOINT_REF
        or learner.checkpoint_identity != LEARNER_CHECKPOINT_REF
        or integrity.checkpoint_ref != LEARNER_CHECKPOINT_REF
        or integrity.config_ref != PROSPECTIVE_CONFIG_REF
        or integrity.step != 0
        or learner.state_digest() != INITIAL_COMPETENCE_DIGEST
        or learner.events
        or learner.tombstoned_evidence_refs
        or learner.pending_decision is not None
        or len(snapshot) != GENESIS_SNAPSHOT_BYTES
        or hashlib.sha256(snapshot).hexdigest() != GENESIS_SNAPSHOT_SHA256
        or (expected_snapshot is not None and snapshot != expected_snapshot)
    ):
        raise RuntimeError("learner genesis reconstruction differs")
    return snapshot


def _read_exact_genesis_artifact() -> bytes:
    if GENESIS_PATH.is_symlink() or not GENESIS_PATH.is_file():
        raise RuntimeError("learner genesis artifact is absent or not a regular file")
    metadata = GENESIS_PATH.stat()
    if metadata.st_uid != os.getuid() or metadata.st_gid != os.getgid():
        raise RuntimeError("learner genesis artifact ownership differs")
    if metadata.st_mode & 0o7777 != 0o600:
        raise RuntimeError("learner genesis artifact permissions differ")
    snapshot = GENESIS_PATH.read_bytes()
    if (
        len(snapshot) != GENESIS_SNAPSHOT_BYTES
        or hashlib.sha256(snapshot).hexdigest() != GENESIS_SNAPSHOT_SHA256
    ):
        raise RuntimeError("learner genesis artifact identity differs")
    return snapshot


def _materialize_genesis(snapshot: bytes) -> str:
    if (
        type(snapshot) is not bytes
        or len(snapshot) != GENESIS_SNAPSHOT_BYTES
        or hashlib.sha256(snapshot).hexdigest() != GENESIS_SNAPSHOT_SHA256
    ):
        raise RuntimeError("learner genesis bytes differ before materialization")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir():
        raise RuntimeError("learner genesis state root is not an exact directory")
    if GENESIS_PATH.exists() or GENESIS_PATH.is_symlink():
        if _read_exact_genesis_artifact() != snapshot:
            raise RuntimeError("existing learner genesis bytes differ")
        return "VERIFIED_EXISTING"
    temporary = GENESIS_PATH.with_suffix(".pt.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("learner genesis temporary path is not fresh")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    linked = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(snapshot)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, GENESIS_PATH)
            linked = True
        except FileExistsError:
            if _read_exact_genesis_artifact() != snapshot:
                raise RuntimeError("concurrent learner genesis bytes differ")
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(STATE_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return "CREATED" if linked else "VERIFIED_CONCURRENT"


def _validate_genesis_helper_environment() -> None:
    if dict(os.environ) != GENESIS_HELPER_ENVIRONMENT:
        raise RuntimeError("learner genesis helper environment differs")
    if Path(sys.executable) != Path(ANGLER_PYTHON):
        raise RuntimeError("learner genesis helper Python differs")
    if Path(__file__).resolve() != RUNNER_PATH:
        raise RuntimeError("learner genesis helper runner path differs")


def _materialize_learner_genesis_helper() -> None:
    _validate_genesis_helper_environment()
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("learner genesis helper unexpectedly exposes CUDA")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # This is the sole manual seed in the runner and precedes core construction.
    torch.manual_seed(GENESIS_SEED)
    core = _build_genesis_core()
    if core.checkpoint_identity != LEARNER_CHECKPOINT_REF:
        raise RuntimeError("seeded learner checkpoint identity differs")
    learner = DurableAbilityLearner(
        core,
        core.initial_state(),
        checkpoint_identity=core.checkpoint_identity,
        prospective_lesion=False,
    )
    snapshot = _verify_exact_genesis_learner(learner)
    materialization = _materialize_genesis(snapshot)
    envelope = {
        "version": 1,
        "artifact_status": materialization,
        "checkpoint_ref": core.checkpoint_identity,
        "config_ref": "sha256:" + core.config.digest,
        "core_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in core.state_dict().items()
        },
    }
    buffer = io.BytesIO()
    torch.save(envelope, buffer)
    encoded = buffer.getvalue()
    if not 1 <= len(encoded) <= MAXIMUM_GENESIS_HELPER_BYTES:
        raise RuntimeError("learner genesis helper payload exceeds its ceiling")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _restore_parent_genesis() -> tuple[DurableAbilityLearner, dict[str, object]]:
    if Path(sys.executable) != Path(ANGLER_PYTHON):
        raise RuntimeError("live parent Python differs before genesis helper")
    if Path(__file__).resolve() != RUNNER_PATH:
        raise RuntimeError("live parent runner path differs before genesis helper")
    helper_temporary = Path(GENESIS_HELPER_ENVIRONMENT["TMPDIR"])
    helper_temporary.mkdir(exist_ok=True, mode=0o700)
    helper_metadata = helper_temporary.stat()
    if (
        helper_temporary.is_symlink()
        or not helper_temporary.is_dir()
        or any(helper_temporary.iterdir())
        or helper_metadata.st_uid != os.getuid()
        or helper_metadata.st_gid != os.getgid()
        or helper_metadata.st_mode & 0o7777 != 0o700
    ):
        raise RuntimeError("learner genesis helper temporary directory differs")
    command = (
        ANGLER_PYTHON,
        str(RUNNER_PATH),
        "--materialize-learner-genesis",
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=dict(GENESIS_HELPER_ENVIRONMENT),
        check=False,
        capture_output=True,
        timeout=GENESIS_HELPER_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        error = " ".join(completed.stderr.decode("utf-8", "replace").split())
        raise RuntimeError(
            "learner genesis helper failed "
            f"with exit {completed.returncode}: {error[:1_024] or 'no stderr'}"
        )
    if completed.stderr:
        raise RuntimeError("learner genesis helper emitted unexpected stderr")
    if not 1 <= len(completed.stdout) <= MAXIMUM_GENESIS_HELPER_BYTES:
        raise RuntimeError("learner genesis helper payload size is invalid")
    try:
        envelope = torch.load(
            io.BytesIO(completed.stdout),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError("learner genesis helper payload cannot be decoded") from exc
    if type(envelope) is not dict or set(envelope) != {
        "version",
        "artifact_status",
        "checkpoint_ref",
        "config_ref",
        "core_state_dict",
    }:
        raise RuntimeError("learner genesis helper payload shape differs")
    if (
        envelope["version"] != 1
        or envelope["artifact_status"]
        not in {"CREATED", "VERIFIED_EXISTING", "VERIFIED_CONCURRENT"}
        or envelope["checkpoint_ref"] != LEARNER_CHECKPOINT_REF
        or envelope["config_ref"] != PROSPECTIVE_CONFIG_REF
        or type(envelope["core_state_dict"]) is not dict
    ):
        raise RuntimeError("learner genesis helper payload identity differs")

    # The live parent does not seed: it builds a topology shell, then installs
    # the helper-proven frozen parameters before restoring canonical state.
    core = _build_genesis_core()
    raw_state = envelope["core_state_dict"]
    expected_keys = set(core.state_dict())
    if set(raw_state) != expected_keys or any(
        type(name) is not str
        or not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.requires_grad
        for name, value in raw_state.items()
    ):
        raise RuntimeError("learner genesis helper core state differs")
    core.load_state_dict(raw_state, strict=True)
    if core.checkpoint_identity != LEARNER_CHECKPOINT_REF:
        raise RuntimeError("restored learner checkpoint identity differs")
    learner = _new_genesis_learner(core)
    snapshot = _read_exact_genesis_artifact()
    learner.restore_state(snapshot)
    _verify_exact_genesis_learner(learner, expected_snapshot=snapshot)
    return learner, {
        "argv": list(command),
        "environment": dict(GENESIS_HELPER_ENVIRONMENT),
        "helper_payload_bytes": len(completed.stdout),
        "helper_payload_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "artifact_status": envelope["artifact_status"],
        "parent_restore_verified": True,
    }


def _manifest() -> FrozenQwenCycleManifestV1:
    return FrozenQwenCycleManifestV1(
        model_ref="sha256:" + MODEL_ROOT_MANIFEST,
        tokenizer_ref="sha256:" + TOKENIZER_MANIFEST,
        genesis_config_ref=GENESIS_CONFIG_REF,
        prospective_config_ref=PROSPECTIVE_CONFIG_REF,
        learner_checkpoint_ref=LEARNER_CHECKPOINT_REF,
        initial_competence_state_digest=INITIAL_COMPETENCE_DIGEST,
        genesis_snapshot_sha256=GENESIS_SNAPSHOT_SHA256,
        genesis_snapshot_bytes=GENESIS_SNAPSHOT_BYTES,
        genesis_seed=GENESIS_SEED,
        prospective_lesion=False,
    )


def _seed_episode(state_digest: str, manifest: FrozenQwenCycleManifestV1) -> CognitiveEpisode:
    """Create unrelated historical evidence without leaking the live answer."""

    evidence_ref = _ref(
        "frozen-qwen-cognee-cycle-v1-unrelated-observed-fixture-evidence"
    )
    proposals = (
        "Read the two supplied color words and preserve their given order.",
        "Return only the supplied color words separated by one space.",
    )
    return CognitiveEpisode(
        task_id="frozen-qwen-cognee-cycle-v1-unrelated-observed-fixture",
        request=(
            "For an unrelated synthetic fixture, return the two supplied color "
            "words in their given order: cobalt then amber."
        ),
        recalled_refs=(evidence_ref,),
        proposals=proposals,
        selected_index=1,
        commitment=ProspectiveCommitment(
            parent_event_ref=None,
            task_id="frozen-qwen-cognee-cycle-v1-unrelated-observed-fixture",
            candidate_index=1,
            candidate_trace=proposals[1],
            predicted_score=0.0,
            uncertainty=1.0,
            horizon=1,
            competence_state_digest=state_digest,
        ),
        response="cobalt amber",
        observations=(
            "Authored synthetic fixture transcript unrelated to the live task.",
        ),
        outcome="success",
        feedback_text=(
            "The fixture-only exact-order check accepted the unrelated transcript."
        ),
        feedback_source_ref=_ref(
            "frozen-qwen-cognee-cycle-v1-unrelated-fixture-disposition"
        ),
        parent_state_digest=state_digest,
        child_state_digest=state_digest,
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
        supporting_evidence_refs=(evidence_ref,),
    )


def _initialize_store(
    learner: DurableAbilityLearner,
    manifest: FrozenQwenCycleManifestV1,
) -> CognitiveTransactionStore:
    if STORE_PATH.exists() or JOURNAL_PATH.exists():
        raise RuntimeError("qualification cognitive state is not fresh")
    store = CognitiveTransactionStore(STORE_PATH)
    snapshot = learner.capture_state()
    state_digest = learner.state_digest()
    store.initialize(
        state_digest,
        snapshot,
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
    )
    store.commit_episode(
        _seed_episode(state_digest, manifest),
        snapshot,
        expected_parent_digest=state_digest,
    )
    return store


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _linux_task_count() -> int:
    tasks = tuple(Path("/proc/self/task").iterdir())
    if not tasks:
        raise RuntimeError("Linux task inventory is unexpectedly empty")
    return len(tasks)


def _atomic_result(
    payload: dict[str, object],
    *,
    existing_output_bytes: int = 0,
) -> None:
    if type(existing_output_bytes) is not int or existing_output_bytes < 0:
        raise ValueError("existing_output_bytes must be a non-negative integer")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULT_PATH.exists():
        raise RuntimeError("qualification result already exists")
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    if temporary.exists():
        raise RuntimeError("qualification temporary result already exists")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if existing_output_bytes + len(encoded) > MAXIMUM_NEW_OUTPUT_BYTES:
        raise RuntimeError("qualification evidence exceeds the one-GiB ceiling")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, RESULT_PATH)
        replaced = True
        directory = os.open(RESULT_PATH.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if replaced:
            RESULT_PATH.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise


def _stable_error(error: BaseException) -> str:
    return " ".join(str(error).split())[:1_024] or type(error).__name__


def _canonical_state_observation() -> dict[str, object]:
    """Record identities without asserting post-failure semantic preservation."""

    artifacts: dict[str, dict[str, object]] = {}
    for label, path in (("store", STORE_PATH), ("journal", JOURNAL_PATH)):
        try:
            if not path.exists():
                artifacts[label] = {"status": "ABSENT", "path": str(path)}
            elif not path.is_file():
                artifacts[label] = {"status": "NOT_A_FILE", "path": str(path)}
            else:
                digest, size = _sha256_file(path)
                artifacts[label] = {
                    "status": "IDENTITY_OBSERVED_ONLY",
                    "path": str(path),
                    "bytes": size,
                    "sha256": digest,
                }
        except BaseException as error:
            artifacts[label] = {
                "status": "OBSERVATION_FAILED",
                "path": str(path),
                "error_type": type(error).__name__,
                "error": _stable_error(error),
            }
    return {
        "status": "NOT_VERIFIED_AFTER_FAILURE",
        "artifacts": artifacts,
    }


def _cycle(
    io: LocalQwenIO,
    manifest: FrozenQwenCycleManifestV1,
    bindings: CogneeSubprocessBindings,
    learner: DurableAbilityLearner,
) -> tuple[
    CognitiveCycle,
    DurableAbilityLearner,
    AcquisitionSituatedMemory,
    CognitiveTransactionStore,
    FrozenQwenCognitiveExecutorV1,
]:
    store = CognitiveTransactionStore(STORE_PATH)
    adapter = CogneeAcquisitionAdapter(
        bindings=bindings,
        tenant_id=bindings.tenant_id,
        dataset_id=bindings.dataset_id,
        dataset_name=bindings.dataset_name,
        node_set_name=bindings.node_set_name,
        local_embeddings_configured=True,
        external_embedding_calls_authorized=False,
        telemetry_authorized=False,
    )
    memory = AcquisitionSituatedMemory(source=store, backend=adapter)
    text = FrozenQwenProcedureAdapterV1(io, manifest)
    relation = FrozenQwenProcedureRelationAdapterV1(io, manifest)
    executor = FrozenQwenCognitiveExecutorV1(
        io,
        manifest,
        SQLiteQwenExecutionJournal(JOURNAL_PATH),
    )
    observation = QwenReceiptObservedStateEncoderV1(
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
        manifest_ref=manifest.manifest_ref,
        latent_width=manifest.latent_width,
    )
    cycle = CognitiveCycle(
        text_adapter=text,
        relation_adapter=relation,
        learner=learner,
        memory=memory,
        executor=executor,
        transaction_store=store,
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
        agent_ref=_ref("frozen-qwen-cognee-cycle-v1-agent"),
        world_ref=_ref("frozen-qwen-cognee-cycle-v1-world"),
        recall_limit=12,
        proposal_count=2,
        successor_mode=True,
        reality_mode="SIMULATED",
        subject_ref=_ref("frozen-qwen-cognee-cycle-v1-subject"),
        scope_ref=_ref("frozen-qwen-cognee-cycle-v1-scope"),
        bootstrap_evidence_refs=(
            _ref("frozen-qwen-cognee-cycle-v1-bootstrap"),
        ),
        observation_encoder=observation,
    )
    return cycle, learner, memory, store, executor


def _validate_offline_environment() -> dict[str, object]:
    """Require one clean environment inside a loopback-only net namespace."""

    observed = dict(os.environ)
    if observed != PARENT_ENVIRONMENT:
        raise RuntimeError("bounded offline parent environment differs")
    if Path(sys.executable) != Path(ANGLER_PYTHON) or Path(__file__).resolve() != RUNNER_PATH:
        raise RuntimeError("bounded offline parent executable identity differs")
    temporary = Path(PARENT_ENVIRONMENT["TMPDIR"])
    if temporary.is_symlink() or not temporary.is_dir() or any(temporary.iterdir()):
        raise RuntimeError("bounded offline parent temporary directory is not fresh")
    metadata = temporary.stat()
    if (
        metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or metadata.st_mode & 0o7777 != 0o700
    ):
        raise RuntimeError("bounded offline parent temporary directory metadata differs")
    interfaces = tuple(socket.if_nameindex())
    raw_ipv4_lines = tuple(
        Path("/proc/net/route").read_text(encoding="ascii").splitlines()
    )
    ipv4_routes = (
        raw_ipv4_lines[1:]
        if raw_ipv4_lines and raw_ipv4_lines[0].split()[:2] == ["Iface", "Destination"]
        else raw_ipv4_lines
    )
    ipv6_routes = tuple(
        tuple(line.split())
        for line in Path("/proc/net/ipv6_route")
        .read_text(encoding="ascii")
        .splitlines()
    )
    expected_ipv6_route = (
        "00000000000000000000000000000000",
        "00",
        "00000000000000000000000000000000",
        "00",
        "00000000000000000000000000000000",
        "ffffffff",
        "00000001",
        "00000000",
        "00200200",
        "lo",
    )
    if (
        interfaces != ((1, "lo"),)
        or ipv4_routes
        or ipv6_routes != (expected_ipv6_route, expected_ipv6_route)
    ):
        raise RuntimeError("live parent is not inside the exact network namespace")
    loopback_state = Path("/sys/class/net/lo/operstate").read_text(
        encoding="ascii"
    ).strip()
    if loopback_state != "unknown":
        raise RuntimeError("live parent loopback state differs from the isolated namespace")
    material = json.dumps(
        observed,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "admitted_environment": observed,
        "environment_sha256": hashlib.sha256(material).hexdigest(),
        "interfaces": [list(item) for item in interfaces],
        "ipv4_routes": list(ipv4_routes),
        "ipv6_routes": [list(item) for item in ipv6_routes],
        "loopback_operstate": loopback_state,
        "network_namespace": os.readlink("/proc/self/ns/net"),
        "temporary_directory": str(temporary),
    }


async def _finish_binding(
    bindings: CogneeSubprocessBindings,
    *,
    forget: bool,
    operation_error: BaseException | None,
) -> dict[str, object]:
    """Close one worker and preserve both operation and cleanup failures."""

    cleanup_errors: list[BaseException] = []
    namespace_status = "PRESERVED_FOR_RESTART"
    if forget:
        namespace_status = "FORGET_FAILED"
        try:
            await bindings.forget_namespace()
            namespace_status = "FORGOTTEN_AND_RECREATED"
        except BaseException as error:
            cleanup_errors.append(error)
    try:
        await bindings.close()
    except BaseException as error:
        cleanup_errors.append(error)
    if operation_error is not None:
        if cleanup_errors:
            raise BaseExceptionGroup(
                "Cognee operation and rollback both failed",
                [operation_error, *cleanup_errors],
            )
        raise operation_error.with_traceback(operation_error.__traceback__)
    if cleanup_errors:
        if len(cleanup_errors) == 1:
            error = cleanup_errors[0]
            raise error.with_traceback(error.__traceback__)
        raise BaseExceptionGroup("Cognee rollback failed", cleanup_errors)
    return {
        "namespace_status": namespace_status,
        "dataset_id": bindings.dataset_id,
    }


async def _rollback_after_closed_binding(
    dataset_id: str,
    operation_error: BaseException,
) -> None:
    """Rebind solely to forget an owned namespace after a closed-stage error."""

    cleanup_errors: list[BaseException] = []
    try:
        rebound = await CogneeSubprocessBindings.start(
            expected_dataset_id=dataset_id
        )
    except BaseException as error:
        cleanup_errors.append(error)
    else:
        try:
            await _finish_binding(
                rebound,
                forget=True,
                operation_error=None,
            )
        except BaseException as error:
            cleanup_errors.append(error)
    if cleanup_errors:
        raise BaseExceptionGroup(
            "closed-stage Cognee operation and rollback both failed",
            [operation_error, *cleanup_errors],
        )
    raise operation_error.with_traceback(operation_error.__traceback__)


def _cognee_runtime_evidence(
    bindings: CogneeSubprocessBindings,
) -> dict[str, object]:
    evidence = {
        "execution_providers": list(bindings.embedding_execution_providers),
        "intra_op_threads": bindings.embedding_intra_op_threads,
        "inter_op_threads": bindings.embedding_inter_op_threads,
    }
    if evidence != {
        "execution_providers": ["CPUExecutionProvider"],
        "intra_op_threads": 2,
        "inter_op_threads": 2,
    }:
        raise RuntimeError("Cognee embedding runtime evidence differs")
    return evidence


async def _integrated_cycle(
    io: LocalQwenIO,
    manifest: FrozenQwenCycleManifestV1,
    learner: DurableAbilityLearner,
) -> dict[str, object]:
    if io.generation_calls != 0:
        raise RuntimeError("live Qwen I/O already contains a generation attempt")
    first_bindings = await CogneeSubprocessBindings.start()
    dataset_id = first_bindings.dataset_id
    user_id = first_bindings.user_id
    tenant_id = first_bindings.tenant_id
    first_error: BaseException | None = None
    try:
        cognee_runtime = _cognee_runtime_evidence(first_bindings)
        cycle, _learner, _memory, store, executor = _cycle(
            io, manifest, first_bindings, learner
        )
        turn = await cycle.begin_turn(TASK_ID, TASK)
        proposal_record = cycle.text_adapter.last_proposal_generation
        if proposal_record is None:
            raise RuntimeError("Qwen proposal generation was not retained")
        reparsed_proposals = parse_qwen_procedure_proposals(
            proposal_record.response,
            count=2,
        )
        if reparsed_proposals != turn.decision.proposals:
            raise RuntimeError("durable proposal generation does not reparse exactly")
        completed = await cycle.execute_turn(
            turn,
            observations=("No external tool or hidden observation is available.",),
        )
        journal_entry = executor.journal.get(turn.reservation.idempotency_key)
        if (
            journal_entry is None
            or journal_entry.receipt != completed.execution_receipt
        ):
            raise RuntimeError("Qwen execution journal differs from the cycle receipt")
        if (
            io.generation_calls != 2
            or proposal_record.generation_ref
            == journal_entry.generation.generation_ref
        ):
            raise RuntimeError("live Qwen generation-attempt accounting differs")
        parent_digest = turn.decision.parent_state_digest
    except BaseException as error:
        first_error = error
    await _finish_binding(
        first_bindings,
        forget=first_error is not None,
        operation_error=first_error,
    )

    # Restart once from the canonical receipt, then stage objective feedback
    # without applying it.  This isolates the feedback-staged restart witness.
    second_bindings = await CogneeSubprocessBindings.start(
        expected_dataset_id=dataset_id
    )
    second_error: BaseException | None = None
    try:
        if _cognee_runtime_evidence(second_bindings) != cognee_runtime:
            raise RuntimeError("Cognee embedding runtime drifted on receipt restart")
        second, _second_learner, _second_memory, second_store, _second_executor = (
            _cycle(io, manifest, second_bindings, learner)
        )
        resumed = second.executed_turn
        if resumed is None or resumed.execution_receipt != completed.execution_receipt:
            raise RuntimeError("receipt-stage cycle restart did not rejoin exactly")
        verifier_passed = resumed.execution.response == EXPECTED_RESPONSE
        outcome = "success" if verifier_passed else "failure"
        feedback = ObjectiveFeedbackRecord.from_execution(
            resumed.turn.reservation,
            resumed.execution_request,
            resumed.execution_receipt,
            outcome=outcome,
            feedback_text=(
                "External exact-string verifier matched the required response."
                if verifier_passed
                else "External exact-string verifier did not match the required response."
            ),
            feedback_source_ref=_ref(
                "frozen-qwen-cognee-cycle-v1-objective-exact-string-verifier"
            ),
        )
        staged = second_store.stage_prospective_feedback(feedback)
        if staged.record.feedback != feedback:
            raise RuntimeError("objective feedback did not stage exactly")
    except BaseException as error:
        second_error = error
    await _finish_binding(
        second_bindings,
        forget=second_error is not None,
        operation_error=second_error,
    )

    third_bindings = await CogneeSubprocessBindings.start(
        expected_dataset_id=dataset_id
    )
    third_error: BaseException | None = None
    third_closed = False
    try:
        if _cognee_runtime_evidence(third_bindings) != cognee_runtime:
            raise RuntimeError("Cognee embedding runtime drifted on feedback restart")
        third, third_learner, third_memory, third_store, third_executor = _cycle(
            io, manifest, third_bindings, learner
        )
        if third.executed_turn is None:
            raise RuntimeError("feedback-stage restart lost the executed turn")
        # Close the ordinary worker before the canonical commit.  The canonical
        # learner/store transition must succeed while disposable projection
        # fails closed and remains retryable by a new worker.
        await third_bindings.close()
        third_closed = True
        committed = await third.resume_staged_outcome()
        if not committed.projection_pending:
            raise RuntimeError("closed-worker commit did not retain pending projection")
        pending_before_retry = third_store.pending_acquisition_projections(limit=64)
        if (
            not pending_before_retry
            or pending_before_retry[-1].record_ref
            != third_store.acquisition_head().record_ref
        ):
            raise RuntimeError("canonical pending projection does not bind the new head")
        third_store.audit_integrity()
        third_executor.journal.audit_integrity()
        child_digest = third_learner.state_digest()
        if child_digest == parent_digest:
            raise RuntimeError("objective feedback did not advance learner state")
        lineage = third.current_lineage
        if lineage is None or lineage.competence_state_digest != child_digest:
            raise RuntimeError("child Moving-Origin lineage differs from learner state")
        active = third_store.active_prospective_turn()
        if active is not None:
            raise RuntimeError("committed cycle retained an active reservation")
    except BaseException as error:
        third_error = error
    if third_error is not None:
        if third_closed:
            await _rollback_after_closed_binding(dataset_id, third_error)
        await _finish_binding(
            third_bindings,
            forget=True,
            operation_error=third_error,
        )
    await _finish_binding(
        third_bindings,
        forget=False,
        operation_error=None,
    )

    fourth_bindings = await CogneeSubprocessBindings.start(
        expected_dataset_id=dataset_id
    )
    fourth_error: BaseException | None = None
    try:
        if _cognee_runtime_evidence(fourth_bindings) != cognee_runtime:
            raise RuntimeError("Cognee embedding runtime drifted on retry restart")
        fourth, fourth_learner, fourth_memory, fourth_store, fourth_executor = _cycle(
            io, manifest, fourth_bindings, learner
        )
        observed_pending = fourth_store.pending_acquisition_projections(limit=64)
        if observed_pending != pending_before_retry:
            raise RuntimeError("pending projection drifted across worker restart")
        rebuilt_refs = await fourth_memory.rebuild(limit=64)
        if not rebuilt_refs:
            raise RuntimeError("worker restart did not rebuild canonical acquisition history")
        retry_refs = await fourth.retry_pending_projections(limit=64)
        expected_retry_refs = tuple(item.record_ref for item in observed_pending)
        if retry_refs != expected_retry_refs:
            raise RuntimeError("bounded projection retry acknowledgements differ")
        if fourth_store.pending_acquisition_projections(limit=1):
            raise RuntimeError("bounded projection retry left canonical work pending")
        fourth_memory.assert_synchronized(fourth_store.acquisition_head())
        recalled = await fourth_memory.recall(TASK, limit=12)
        if not recalled.items:
            raise RuntimeError("post-retry Cognee recall did not canonically rejoin")
        fourth_store.audit_integrity()
        fourth_executor.journal.audit_integrity()
        if (
            fourth_learner.state_digest() != child_digest
            or fourth.current_lineage != lineage
            or io.generation_calls != 2
        ):
            raise RuntimeError("post-retry learner, lineage, or generation state drifted")
        result = {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "cognee_embedding_runtime": cognee_runtime,
            "worker_starts": 4,
            "cycle_starts": 4,
            "proposal_generation_ref": proposal_record.generation_ref,
            "proposal_generation": proposal_record.to_payload(),
            "proposal_response_reparsed_exactly": True,
            "proposal_recalled_refs": list(turn.recalled_refs),
            "proposal_prompt_tokens": proposal_record.prompt_tokens,
            "execution_generation_ref": journal_entry.generation.generation_ref,
            "execution_prompt_tokens": journal_entry.generation.prompt_tokens,
            "execution_request_ref": completed.execution_request.execution_request_ref,
            "execution_receipt_ref": completed.execution_receipt.execution_receipt_ref,
            "selected_index": turn.decision.selected_index,
            "selected_trace": turn.decision.selected_trace,
            "proposal_count": len(turn.decision.proposals),
            "response": completed.execution.response,
            "objective_outcome": outcome,
            "objective_verifier_passed": verifier_passed,
            "feedback_ref": feedback.feedback_ref,
            "episode_ref": committed.episode_ref,
            "parent_competence_digest": parent_digest,
            "child_competence_digest": child_digest,
            "child_lineage_ref": lineage.lineage_ref,
            "canonical_sequence": committed.sequence,
            "acquisition_ordinal": fourth_store.acquisition_head().next_ordinal - 1,
            "projection_pending_before_retry": committed.projection_pending,
            "pending_projection_refs_before_retry": [
                item.projection_ref for item in observed_pending
            ],
            "rebuild_record_refs": list(rebuilt_refs),
            "retry_acknowledged_record_refs": list(retry_refs),
            "projection_pending_after_retry": False,
            "post_commit_recall_refs": [item.artifact_ref for item in recalled.items],
            "generation_calls_observed": io.generation_calls,
        }
    except BaseException as error:
        fourth_error = error
    namespace_cleanup = await _finish_binding(
        fourth_bindings,
        forget=True,
        operation_error=fourth_error,
    )
    result["cognee_namespace_cleanup"] = namespace_cleanup
    return result


async def run_live() -> dict[str, object]:
    if RESULT_PATH.exists():
        raise RuntimeError("qualification result already exists")
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (SCRATCH_ROOT / "tmp").mkdir(exist_ok=True, mode=0o700)
    parent_boundary = _validate_offline_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 2 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("parent Torch CPU thread pools differ")
    parent_tasks_before_model = _linux_task_count()
    host = _host_preflight()
    software = _software_preflight()
    repository_before = _tree_manifest(REPOSITORY_ROOT)
    embedding_before = _verify_embedding_files()
    tiktoken_before = _tree_manifest(TIKTOKEN_CACHE_ROOT)
    write_baseline = _write_scope_baseline()
    _validate_early_scratch()

    manifest = _manifest()
    learner, genesis_helper = _restore_parent_genesis()
    _initialize_store(learner, manifest)

    before_files = _verify_model_files()
    gpu = _gpu_preflight()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one assigned CUDA device is required")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) != GPU_NAME:
        raise RuntimeError("logical cuda:0 is not the frozen RTX 5080")
    torch.cuda.reset_peak_memory_stats(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.eval()
    if (
        type(model).__name__ != "Qwen3ForCausalLM"
        or int(model.config.hidden_size) != manifest.input_width
        or int(model.config.num_hidden_layers) != 36
        or int(model.config.max_position_embeddings) != 40_960
        or any(parameter.requires_grad for parameter in model.parameters())
        or any(parameter.dtype is not torch.bfloat16 for parameter in model.parameters())
        or model.training
    ):
        raise RuntimeError("loaded Qwen architecture or frozen state differs")
    io = LocalQwenIO(
        model,
        tokenizer,
        embedding_batch_size=manifest.embedding_batch_size,
        max_input_tokens=manifest.max_input_tokens,
        max_new_tokens=manifest.max_new_tokens,
        enable_thinking=False,
        model_ref=manifest.model_ref,
        tokenizer_ref=manifest.tokenizer_ref,
    )
    manifest.assert_io(io)
    started = time.monotonic()
    tensor_before = foundation_tensor_digest(model)
    integrated = await _integrated_cycle(io, manifest, learner)
    tensor_after = foundation_tensor_digest(model)
    wall_seconds = time.monotonic() - started
    if tensor_after != tensor_before or model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError("frozen Qwen tensors or mode changed")
    torch.cuda.synchronize(0)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    rss = _rss_bytes()
    parent_tasks_after_cycle = _linux_task_count()
    if (
        peak_allocated > MAXIMUM_CUDA_BYTES
        or peak_reserved > MAXIMUM_CUDA_BYTES
        or rss > MAXIMUM_RSS_BYTES
        or wall_seconds > MAXIMUM_WALL_SECONDS
    ):
        raise RuntimeError("live qualification exceeded its resource ceiling")

    del io
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    after_files = _verify_model_files()
    if after_files != before_files:
        raise RuntimeError("Qwen model files changed during qualification")
    write_scope = _verify_write_scope(
        write_baseline,
        repository_before=repository_before,
        embedding_before=embedding_before,
        tiktoken_before=tiktoken_before,
    )

    return {
        "identity": IDENTITY,
        "classification": "TECHNICAL_PASS",
        "scientific_claim": False,
        "model_ref": manifest.model_ref,
        "tokenizer_ref": manifest.tokenizer_ref,
        "encoder_ref": manifest.encoder_ref,
        "manifest_ref": manifest.manifest_ref,
        "learner": {
            "genesis_config_ref": manifest.genesis_config_ref,
            "prospective_config_ref": manifest.prospective_config_ref,
            "checkpoint_ref": manifest.learner_checkpoint_ref,
            "initial_competence_digest": manifest.initial_competence_state_digest,
            "genesis_snapshot_sha256": manifest.genesis_snapshot_sha256,
            "genesis_snapshot_bytes": manifest.genesis_snapshot_bytes,
            "genesis_seed": manifest.genesis_seed,
            "genesis_helper": genesis_helper,
        },
        "cycle": integrated,
        "resource": {
            "host": host,
            "software": software,
            "gpu": gpu,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "peak_rss_bytes": rss,
            "wall_seconds_after_model_load": wall_seconds,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "parent_boundary": parent_boundary,
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "configured_compute_pools": {
                "parent_torch_intraop_threads": 2,
                "parent_torch_interop_threads": 1,
                "cognee_fastembed_onnx_intraop_threads": 2,
                "cognee_fastembed_onnx_interop_threads": 2,
                "configured_pool_sum": 7,
                "ceiling": 8,
            },
            "linux_task_count_observed": {
                "before_model_load": parent_tasks_before_model,
                "after_integrated_cycle": parent_tasks_after_cycle,
                "interpretation": (
                    "observational OS tasks, not configured compute-pool workers"
                ),
            },
            "ceilings": {
                "minimum_memory_available_bytes": MINIMUM_MEMORY_AVAILABLE_BYTES,
                "minimum_storage_available_bytes": MINIMUM_STORAGE_AVAILABLE_BYTES,
                "minimum_assigned_gpu_free_mib": MINIMUM_FREE_MIB,
                "maximum_cuda_allocated_bytes": MAXIMUM_CUDA_BYTES,
                "maximum_cuda_reserved_bytes": MAXIMUM_CUDA_BYTES,
                "maximum_process_rss_bytes": MAXIMUM_RSS_BYTES,
                "maximum_wall_seconds_after_model_load": MAXIMUM_WALL_SECONDS,
                "maximum_new_output_bytes": MAXIMUM_NEW_OUTPUT_BYTES,
                "maximum_configured_compute_pool_threads": 8,
                "maximum_model_generation_calls": 16,
                "maximum_live_cycles": 8,
            },
        },
        "model_files_before": before_files,
        "model_files_after": after_files,
        "write_scope": write_scope,
        "foundation_tensor_digest_before": tensor_before,
        "foundation_tensor_digest_after": tensor_after,
        "limits": [
            (
                "One authored synthetic live task plus an unrelated fixture seed; "
                "this is not a reasoning-quality evaluation."
            ),
            "The objective outcome may be failure without invalidating composition.",
            "No AGI, autonomy, consciousness, feeling, readiness, or deployment claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--execute-live-qualification",
        action="store_true",
        help="run the exact local effectful qualification",
    )
    modes.add_argument(
        "--materialize-learner-genesis",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.materialize_learner_genesis:
        _materialize_learner_genesis_helper()
        return
    if not args.execute_live_qualification:
        raise RuntimeError("live qualification requires the explicit execution flag")
    if RESULT_PATH.exists():
        raise RuntimeError("qualification result already exists")
    started = time.monotonic()
    try:
        result = asyncio.run(run_live())
        write_scope = result.get("write_scope")
        if type(write_scope) is not dict or type(
            write_scope.get("new_bytes_before_result")
        ) is not int:
            raise RuntimeError("qualification write-scope evidence is malformed")
        _atomic_result(
            result,
            existing_output_bytes=write_scope["new_bytes_before_result"],
        )
    except BaseException as error:
        failure = {
            "identity": IDENTITY,
            "classification": "TECHNICAL_FAILURE",
            "scientific_claim": False,
            "error_type": type(error).__name__,
            "error": _stable_error(error),
            "elapsed_seconds": time.monotonic() - started,
            "canonical_state_preservation": _canonical_state_observation(),
        }
        if not RESULT_PATH.exists():
            try:
                _atomic_result(failure)
            except BaseException as record_error:
                print(
                    json.dumps(
                        {
                            **failure,
                            "failure_record_error_type": type(record_error).__name__,
                            "failure_record_error": _stable_error(record_error),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                raise BaseExceptionGroup(
                    "qualification and failure-result write both failed",
                    [error, record_error],
                ) from None
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        raise
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "result": str(RESULT_PATH),
                "resource": result["resource"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
