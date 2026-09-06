"""Bounded runner for the frozen high-level multi-domain experiment.

The module is deliberately importable with the Python standard library only.
Evaluator, Angler, Cognee, Torch, and Transformers imports occur only inside
the effectful worker or live execution boundaries.  Deterministic code here is
limited to identity, isolation, accounting, evidence, and orchestration; it
does not contain a solution for any evaluation task.

This experiment is local, synthetic, non-promotional evidence.  None of its
records are authorization-bearing Action, Feedback, Episode, PlasticState,
EvaluationReceipt, or PromotionDecision contracts.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import errno
from enum import Enum
import gc
import hashlib
import io
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import selectors
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias
import unicodedata


SUITE_SCHEMA = "angler.high-level-multidomain.v1"
RUNNER_SCHEMA = "angler.high-level-multidomain.runner.v1"
IPC_SCHEMA = "angler.high-level-multidomain.evaluator-ipc.v1"
SEED_ARTIFACT_SCHEMA = "angler.high-level-multidomain.sealed-seeds.v1"
SEED_SEAL_SCHEMA = "angler.high-level-multidomain.seed-seal-receipt.v1"
MANIFEST_SCHEMA = "angler.high-level-multidomain.manifest.v1"
RUN_INTENT_SCHEMA = "angler.high-level-multidomain.run-intent.v1"
EVIDENCE_SCHEMA = "angler.high-level-multidomain.evidence-ledger.v1"
ATTEMPT_STAGE_SCHEMA = "angler.high-level-multidomain.attempt-stage.v1"
ATTEMPT_FINAL_SCHEMA = "angler.high-level-multidomain.attempt-final.v1"
ATTEMPT_RECEIPT_SCHEMA = "angler.high-level-multidomain.attempt-receipt.v1"
CONTROL_SELECTION_SCHEMA = "angler.high-level-multidomain.control-selection.v1"
FROZEN_RECALL_SCHEMA = "angler.high-level-multidomain.frozen-recall.v1"
CLONE_SCHEMA = "angler.high-level-multidomain.probe-clone.v1"
PROBE_INTEGRITY_SCHEMA = "angler.high-level-multidomain.probe-integrity.v1"
MANAGED_RUNTIME_DISPOSITION_SCHEMA = (
    "angler.high-level-multidomain.managed-runtime-disposition.v1"
)
LINEAGE_INTEGRITY_SCHEMA = "angler.high-level-multidomain.lineage-integrity.v1"
ADAPTATION_LINEAGE_SCHEMA = (
    "angler.high-level-multidomain.adaptation-lineage-integrity.v1"
)
FOUNDATION_INTEGRITY_SCHEMA = (
    "angler.high-level-multidomain.foundation-integrity.v1"
)
FRESH_GENESIS_SCHEMA = "angler.high-level-multidomain.fresh-genesis.v1"
FOUNDATION_LOAD_SCHEMA = "angler.high-level-multidomain.foundation-load.v1"
REPOSITORY_TREE_SCHEMA = "angler.repository-tree-manifest.v1"
RUN_INTEGRITY_SCHEMA = "angler.high-level-multidomain.run-integrity.v1"
REMOVAL_FAIRNESS_SCHEMA = "angler.high-level-multidomain.removal-fairness.v1"
RESULT_SCHEMA = "angler.high-level-multidomain.result.v1"
QUALIFICATION_RESULT_SCHEMA = (
    "angler.high-level-multidomain.qualification-result.v1"
)
EVALUATION_RESULT_SCHEMA = "angler.high-level-multidomain.evaluation-result.v1"
RUN_CLASSIFIER_SCHEMA = "angler.high-level-multidomain.integer-classifier.v1"
FAILURE_DISPOSITION_SCHEMA = "angler.high-level-multidomain.failure-disposition.v1"
EVALUATION_ADMISSION_CLAIM_SCHEMA = (
    "angler.high-level-multidomain.evaluation-admission-claim.v1"
)
QUALIFICATION_RELEASE_CLAIM_SCHEMA = (
    "angler.high-level-multidomain.qualification-release-claim.v1"
)

REMOVAL_EVIDENCE_FIELDS = frozenset(
    {
        "backend_raw_hit_count",
        "backend_rejected_hit_count",
        "backend_search_calls",
        "evidence_final_ref",
        "evidence_stage_ref",
        "execution_prompt_ref",
        "execution_receipt_bytes",
        "execution_receipt_ref",
        "execution_request_bytes",
        "execution_request_ref",
        "frozen_recall_ref",
        "proposal_generation_bytes",
        "proposal_generation_ref",
        "proposal_prompt_ref",
        "proposal_request_bytes",
        "probe_integrity_ref",
        "proposals",
        "public_task_ref",
        "raw_response",
        "recalled_record_bytes_sha256",
        "recalled_record_refs",
        "runtime_quiescence_ref",
        "score",
        "selection_bytes",
        "selection_ref",
        "selected_trace",
        "task_id",
        "task_response_generation_bytes",
        "task_response_generation_ref",
    }
)

CLONED_BASELINE_HASH_NAMES = frozenset({"journal", "learner", "store"})
PROBE_BASELINE_HASH_NAMES = frozenset(
    {"acquisition_sequence", "journal", "learner", "store"}
)
FOUNDATION_TENSOR_HASH_NAME = "foundation_tensor"
FOUNDATION_GUARD_SEMANTICS = (
    "CACHED_INITIAL_TENSOR_DIGEST_PLUS_PARAMETER_IDENTITY_SHAPE_DTYPE_DEVICE_"
    "STORAGE_POINTER_AND_VERSION_PER_PROBE"
)

EVALUATOR_QUALIFICATION_IDENTITY = f"{SUITE_SCHEMA}-harness-qualification"
EVALUATOR_EVALUATION_IDENTITY = f"{SUITE_SCHEMA}-evaluation"
QUALIFICATION_IDENTITY = f"{SUITE_SCHEMA}-r2-harness-qualification"
EVALUATION_IDENTITY = f"{SUITE_SCHEMA}-r2-evaluation"
QUALIFICATION_SEED = 2_026_083_190
QUALIFICATION_REPLICATE_COMMITMENT = (
    "sha256:de7d98486deeca217a0088c069234d442d9a4dff1857c585ae1f853a1355b18a"
)
PROTOCOL_VERSION = 1
MAXIMUM_IPC_BYTES = 4 * 1024 * 1024
MAXIMUM_RESULT_BYTES = 64 * 1024 * 1024
MAXIMUM_EVIDENCE_RECORD_BYTES = 4 * 1024 * 1024
MAXIMUM_SEED_ARTIFACT_BYTES = 4_096
MAXIMUM_RECALL_ITEMS = 12
MAXIMUM_INPUT_TOKENS = 4_096
MAXIMUM_OUTPUT_TOKENS = 64
PROPOSAL_COUNT = 2

NO_PERSISTENT_RECALLED_EVIDENCE_V1 = "NO_PERSISTENT_RECALLED_EVIDENCE_V1"
QWEN_JOURNAL_QUIESCENCE_KIND = "QWEN_ONLY_JOURNAL"
LITERAL_FIRST_PROPOSAL = "LITERAL_FIRST_PROPOSAL"
CONTROL_NONAUTHORIZATION = (
    "LOCAL_NONAUTHORIZING_CONTROL_SELECTION; not an approved action, learned "
    "selection, authorization-bearing prospective reservation, promotion "
    "decision, or external-effect permission"
)
ATTEMPT_RECEIPT_NONAUTHORIZATION = (
    "LOCAL_NONAUTHORIZING_EVALUATION_ATTEMPT_IDENTITY; not an approved Action, "
    "authorization-bearing prospective reservation, EvaluationReceipt, "
    "PromotionDecision, or external-effect permission"
)

REPOSITORY_ROOT = Path("/opt/angler/src/angler")
RUNNER_PATH = REPOSITORY_ROOT / "experiments/runners/high_level_multidomain_v1_r2.py"
MANIFEST_PATH = (
    REPOSITORY_ROOT / "experiments/manifests/high-level-multidomain-v1-r2.json"
)
ACTIVE_LEAF_PATH = REPOSITORY_ROOT / (
    "docs/blueprints/branches/science/work/"
    "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-"
    "LINEAGE-SEQUENCE-RECOVERY-001.md"
)
EXPECTED_ACTIVE_LEAF_SHA256 = (
    "a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455"
)
STATE_ROOT = Path("/opt/angler/state/project-angler/high-level-multidomain-v1-r2")
SEALED_SEED_PATH = STATE_ROOT / "sealed/seeds-v1-r2.json"
QUALIFICATION_RELEASE_CLAIM_PATH = (
    STATE_ROOT / "sealed/qualification-release-claim-v1-r2.json"
)
EVALUATION_ADMISSION_CLAIM_PATH = (
    STATE_ROOT / "sealed/evaluation-admission-claim-v1-r2.json"
)
_ACTIVE_EVALUATION_PERMITS: dict[str, object] = {}
SCRATCH_ROOT = Path("/opt/angler/scratch/high-level-multidomain-v1-r2")
QUALIFICATION_RESULT_PATH = Path(
    "/opt/angler/results/high-level-multidomain-v1-r2-qualification.json"
)
EVALUATION_RESULT_PATH = Path(
    "/opt/angler/results/high-level-multidomain-v1-r2.json"
)
QUALIFICATION_RUNTIME_ROOT = STATE_ROOT / "qualification-v1-r2"
QUALIFICATION_GENESIS_ROOT = QUALIFICATION_RUNTIME_ROOT / "genesis"
QUALIFICATION_ARM_FACTORY_ROOT = QUALIFICATION_RUNTIME_ROOT / "arm-runtime"
QUALIFICATION_EVIDENCE_LEDGER_PATH = (
    QUALIFICATION_RUNTIME_ROOT / "attempt-evidence.sqlite3"
)
QUALIFICATION_COGNEE_SCOPES_ROOT = STATE_ROOT / "cognee-scopes"
EVALUATION_RUNTIME_ROOT = STATE_ROOT / "evaluation-v1-r2"
EVALUATION_GENESIS_ROOT = EVALUATION_RUNTIME_ROOT / "genesis"
EVALUATION_ARM_FACTORY_ROOT = EVALUATION_RUNTIME_ROOT / "arm-runtime"
EVALUATION_EVIDENCE_LEDGER_PATH = (
    EVALUATION_RUNTIME_ROOT / "attempt-evidence.sqlite3"
)
EVALUATION_COGNEE_SCOPES_ROOT = STATE_ROOT / "evaluation-cognee-scopes"
ANGLER_PYTHON = Path("/opt/angler/venvs/angler/bin/python")
MODEL_ROOT = Path("/opt/angler/models/Qwen3-4B")
FASTEMBED_CACHE_ROOT = Path("/opt/angler/models/fastembed-cache-v1")
TIKTOKEN_CACHE_ROOT = Path(
    "/opt/angler/venvs/cognee/lib/python3.12/site-packages/"
    "litellm/litellm_core_utils/tokenizers"
)
EXPECTED_HOSTNAME = "angler-workstation"
EXPECTED_PYTHON_VERSION = "3.12.3"
EXPECTED_TORCH_VERSION = "2.13.0+cu130"
EXPECTED_CUDA_VERSION = "13.0"
EXPECTED_CUDNN_RUNTIME = 92_000
EXPECTED_PARENT_DISTRIBUTIONS = {
    "accelerate": "1.14.0",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-cudnn-cu13": "9.20.0.48",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.1",
    "triton": "3.7.1",
}
EXPECTED_FASTEMBED_CACHE_TREE = {
    "exists": True,
    "files": 8,
    "manifest_sha256": (
        "f789cc0003b36cadefa04b54d2570bba8451fdcd92e37b4a5b50a1753d1fb2f6"
    ),
    "total_bytes": 67_181_082,
}
EXPECTED_TIKTOKEN_CACHE_TREE = {
    "exists": True,
    "files": 6,
    "manifest_sha256": (
        "0a7cce3442abd34e688feede705e6f71702229ac651330c1d3d90f22478b4a9c"
    ),
    "total_bytes": 7_905_647,
}
ASSIGNED_GPU_UUID = "GPU-df4bb978-e75f-08a0-6660-2b9ed69ee8ca"
UNASSIGNED_GPU_UUID = "GPU-d9dd1ae0-f65d-ef22-f924-2c3e9c976c1e"
EXPECTED_PHYSICAL_GPUS = {
    ASSIGNED_GPU_UUID: {
        "compute_capability": "12.0",
        "name": "NVIDIA GeForce RTX 5080",
        "pci_bus_id": "00000000:01:00.0",
        "total_mib": 16_303,
    },
    UNASSIGNED_GPU_UUID: {
        "compute_capability": "12.0",
        "name": "NVIDIA GeForce RTX 5070",
        "pci_bus_id": "00000000:05:00.0",
        "total_mib": 12_227,
    },
}
MINIMUM_ASSIGNED_GPU_FREE_MIB = 14_336
MAXIMUM_NEW_STATE_SCRATCH_BYTES = 4 * 1024 * 1024 * 1024
MAXIMUM_PENDING_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAXIMUM_PROJECTION_RETRY = 64

# Exact qualified predecessor identities.  These values are copied literally
# rather than imported from the predecessor runner so this module retains its
# standard-library-only import surface and never inherits that runner's live
# paths or private fifth CLI mode.
QUALIFIED_MODEL_ROOT_SHA256 = (
    "1ac705236348881b2fd46f4075b931b5137369d0c469b871aea749f6e0886d83"
)
QUALIFIED_TOKENIZER_SHA256 = (
    "ba21c0913e7aa6dabc7b969aa5b8ced17de450b1b1169f9c474d206cbf485430"
)
QUALIFIED_MODEL_REF = "sha256:" + QUALIFIED_MODEL_ROOT_SHA256
QUALIFIED_TOKENIZER_REF = "sha256:" + QUALIFIED_TOKENIZER_SHA256
QUALIFIED_MODEL_FILE_COUNT = 13
QUALIFIED_MODEL_TOTAL_BYTES = 8_060_926_626
QUALIFIED_TOKENIZER_FILES = (
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
QUALIFIED_GENESIS_CONFIG_REF = (
    "sha256:ec5df84ea8e3946ce9e2bc387ef874c5bd719d962ee0c9c08bd2ab6a279c05b7"
)
QUALIFIED_PROSPECTIVE_CONFIG_REF = (
    "sha256:f3bab0f86a3671588a3bd66538448cf276e3a6e965d7d866f7698a092d1e7313"
)
QUALIFIED_LEARNER_CHECKPOINT_REF = (
    "sha256:5b2de1bb50091570dd92a790fe179cf3681f1202a1660d843863c46845f14f4c"
)
QUALIFIED_INITIAL_COMPETENCE_DIGEST = (
    "sha256:5a8bac6f60ba9d0d73c4cbc3965479a7f1b1f8d4fe4b3e9c9bb4530a924a4286"
)
# The qualified persistent-lineage baseline contains transaction-store genesis
# at sequence 0 followed by exactly one neutral unrelated bootstrap commit.
QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE = 1
QUALIFIED_GENESIS_SNAPSHOT_SHA256 = (
    "93ab9deee7b0b67eabade8d43bb2b09e466f1dcd25405931d5a3d55eca91902d"
)
QUALIFIED_GENESIS_SNAPSHOT_BYTES = 139_913
QUALIFIED_GENESIS_SEED = 2_026_083_101
QUALIFIED_GENESIS_PARAMETER_COUNT = 32_484
QUALIFIED_GENESIS_CREDIT_CONFIG: tuple[tuple[str, int | float], ...] = (
    ("maximum_residual", 4.0),
    ("memory_slots", 512),
    ("rank", 32),
    ("relation_width", 64),
    ("temporal_width", 8),
)
QUALIFIED_GENESIS_PROSPECTIVE_CONFIG: tuple[
    tuple[str, int | float], ...
] = (
    ("latent_width", 16),
    ("maximum_branches", 64),
    ("maximum_recurrent_steps", 4),
    ("maximum_residual", 2.0),
    ("maximum_update_backtracks", 12),
    ("maximum_world_reads", 4),
    ("relation_width", 64),
    ("temporal_width", 8),
    ("update_rate", 0.5),
    ("world_slots", 8),
)
MAXIMUM_GENESIS_ARTIFACT_BYTES = 2 * 1024 * 1024

FROZEN_PARENT_ENVIRONMENT: dict[str, str] = {
    "CUDA_CACHE_PATH": str(SCRATCH_ROOT / "cuda-cache"),
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": ASSIGNED_GPU_UUID,
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
    "PYTHONPATH": f"{REPOSITORY_ROOT}:{REPOSITORY_ROOT / 'src'}",
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

FROZEN_EVALUATOR_WORKER_ENVIRONMENT: dict[str, str] = {
    "CUDA_VISIBLE_DEVICES": "",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/opt/angler/venvs/angler/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": f"{REPOSITORY_ROOT}:{REPOSITORY_ROOT / 'src'}",
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
    "TELEMETRY_DISABLED": "1",
}

FROZEN_PATHS: dict[str, str] = {
    "active_leaf": str(ACTIVE_LEAF_PATH),
    "evaluation_arm_factory_root": str(EVALUATION_ARM_FACTORY_ROOT),
    "evaluation_cognee_scopes_root": str(EVALUATION_COGNEE_SCOPES_ROOT),
    "evaluation_evidence_ledger": str(EVALUATION_EVIDENCE_LEDGER_PATH),
    "evaluation_genesis_root": str(EVALUATION_GENESIS_ROOT),
    "evaluation_runtime_root": str(EVALUATION_RUNTIME_ROOT),
    "evaluation_result": str(EVALUATION_RESULT_PATH),
    "evaluation_admission_claim": str(EVALUATION_ADMISSION_CLAIM_PATH),
    "manifest": str(MANIFEST_PATH),
    "model_root": str(MODEL_ROOT),
    "qualification_release_claim": str(QUALIFICATION_RELEASE_CLAIM_PATH),
    "qualification_result": str(QUALIFICATION_RESULT_PATH),
    "repository_root": str(REPOSITORY_ROOT),
    "runner": str(RUNNER_PATH),
    "scratch_root": str(SCRATCH_ROOT),
    "sealed_seed": str(SEALED_SEED_PATH),
    "state_root": str(STATE_ROOT),
}

FROZEN_GPU_ASSIGNMENT: dict[str, str] = {
    "assigned_model": "Qwen3-4B BF16",
    "assigned_name": "NVIDIA GeForce RTX 5080",
    "assigned_uuid": ASSIGNED_GPU_UUID,
    "unassigned_name": "NVIDIA GeForce RTX 5070",
    "unassigned_uuid": UNASSIGNED_GPU_UUID,
}

FROZEN_RESOURCE_CEILINGS: dict[str, object] = {
    "batch_size": 1,
    "candidate_count": PROPOSAL_COUNT,
    "configured_pool_sum": 8,
    "cuda_allocated_bytes": 12 * 1024 * 1024 * 1024,
    "cuda_reserved_bytes": 12 * 1024 * 1024 * 1024,
    "evaluation_admitted_execution_attempts": 512,
    "evaluation_generation_attempts": 1_024,
    "ipc_message_bytes": MAXIMUM_IPC_BYTES,
    "max_input_tokens": MAXIMUM_INPUT_TOKENS,
    "max_output_tokens": MAXIMUM_OUTPUT_TOKENS,
    "new_state_scratch_bytes": MAXIMUM_NEW_STATE_SCRATCH_BYTES,
    "pending_snapshot_bytes": MAXIMUM_PENDING_SNAPSHOT_BYTES,
    "process_rss_bytes": 24 * 1024 * 1024 * 1024,
    "projection_retry_limit": MAXIMUM_PROJECTION_RETRY,
    "proposal_attempts_per_task": 1,
    "qualification_generation_attempts": 64,
    "recall_records": MAXIMUM_RECALL_ITEMS,
    "result_bytes": MAXIMUM_RESULT_BYTES,
    "successful_proposal_execution_attempts_per_task": 1,
    "store_page_limit": 64,
    "wall_seconds_after_model_load": 7_200.0,
}

FROZEN_WRITE_SCOPE: tuple[str, ...] = (
    f"{STATE_ROOT}/**",
    f"{SCRATCH_ROOT}/**",
    str(QUALIFICATION_RESULT_PATH),
    str(EVALUATION_RESULT_PATH),
)

FROZEN_SOURCE_SHA256: dict[str, str] = {
    "experiments/evaluators/symbolic_procedure_transfer_suite.py": (
        "3a225e8747aab85336f9525eca0a272c86367b9bf68048ffbaec050645abeaa3"
    ),
    "experiments/evaluators/glyph_machine_trace_suite.py": (
        "259118357a042a9867da90514efd82292c36709a573ead13ae956089dbd3bc7e"
    ),
    "experiments/evaluators/causal_operator_suite.py": (
        "0d4ceeae48a849450d239a259a2c3e9e888ee68ab34e41ac30f40d538dba3c93"
    ),
    "experiments/evaluators/high_level_multidomain_v1.py": (
        "2ed9df622aa3f548fc35b4f5d83f7a0aeb70a2b351a93f297d09bae30e495f0d"
    ),
    "tests/unit/experiments/test_high_level_multidomain_v1.py": (
        "f89faa20621842727a70b9bdb7bd798ef78c0f7cb2f245e41ea4464fbb804ba4"
    ),
    "src/angler/runtime/qwen_cognitive.py": (
        "ab2573d6c15174562c93091fcff1cc7b1bb71ac9d534be39b68157cc42cc4406"
    ),
    "src/angler/runtime/situated_qwen.py": (
        "7cf744ce35ba8d0240f748f9efce9a374278ab291e188b67b62a44969babb328"
    ),
    "src/angler/runtime/qwen_peft.py": (
        "b762dbbe0f83aec1dacf8b18a30661d60de900bdf0636df54295945a00d657ca"
    ),
    "src/angler/reasoning/prospective_dynamics.py": (
        "fb0d49e9775f626e8fadb6172bb8cdcb1dffc06f02d07a2103bcb8be2f81354d"
    ),
    "src/angler/reasoning/structure_keyed_credit_memory.py": (
        "f159691121bd638244888dc6d3513c9324fa040d50c00d8b2b86c241e8735326"
    ),
    "src/angler/cognition/contracts.py": (
        "cb2fe5d597c10550425a03951aaecf2e9b104065e6ea32adaba30d1d61b1eb77"
    ),
    "src/angler/cognition/prospective.py": (
        "3678685b4228db615087a5d6807fe30b3ce1c0570c144f05db22d61e8bdb1860"
    ),
    "src/angler/runtime/cognitive_cycle.py": (
        "b738fafe06a071e739f331ca3cd7b7011285db66b2db29ec35b9472df4d26c3d"
    ),
    "src/angler/runtime/cognitive_transaction_store.py": (
        "4c8add9e1b49649b381b0f4dad439709f7b4691cec514c88b39b386bc8a28c0a"
    ),
    "src/angler/runtime/durable_ability_bridge.py": (
        "5f19ac9c23e34af44fc858fbacc6293261a164ddfb067834532107882757e4b2"
    ),
    "src/angler/runtime/prospective_observation.py": (
        "706c539b208f3d0864475fe22d6b01ca42ead4f8b16cb297cd59c23d07706e2c"
    ),
    "src/angler/memory/cognitive_acquisition_graph.py": (
        "62587b8dea7b83e28606dcd83f3062b716094630033e475ddf96ba3f3a57fa3b"
    ),
    "src/angler/memory/cognitive_acquisition.py": (
        "e93c9f181fe1938a4c033a55a88827ba3d0ca5b8b9f46cb1b5569bc4987d36e7"
    ),
    "src/angler/memory/cognee_acquisition_adapter.py": (
        "950c995f87079bd3520987b1038e9d1322b178ffa3893a4d8a9ce8952dedda5b"
    ),
    "src/angler/memory/cognee_worker_protocol.py": (
        "eed327f4d3567356b33cba262b2a96ea9d8dcfc43b0f9b15bb976b2357eb9394"
    ),
    "src/angler/memory/cognee_subprocess_bindings.py": (
        "1295eca35268073164f12f95f2ed1782a9413da743a86becc4c4b3708c822cc0"
    ),
    "src/angler/memory/cognee_acquisition_worker.py": (
        "36925c251fb20726c3f860522c6faaca8c37928997ede30ca0cb29aaa259aa69"
    ),
    (
        "docs/blueprints/branches/runtime/work/"
        "ANG-WORK-RUNTIME-COGNEE-NAMESPACE-SCOPE-V2-001.md"
    ): "26fc14728738099f3b71f694e7e315553c29df867f55be222009ba20ffa31833",
    "experiments/runners/frozen_qwen_cognee_cycle_v1.py": (
        "93e4f1a31cee5f4b738c647a888bea26601b3d9ed7f83e93bfc4557ce487d202"
    ),
    (
        "docs/blueprints/branches/science/work/"
        "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-001.md"
    ): "b8678803ac2ea63f44d9f1e386dffefb907be619d208c18aba71de31e39e629a",
    "experiments/manifests/high-level-multidomain-v1.json": (
        "91542cc7557495c7793f82247772e71b0f00f6995e3b4b3fbf2185ee38b470f8"
    ),
    "experiments/runners/high_level_multidomain_v1.py": (
        "4c9dc1551d5902f4132ac8d7262a025c17d72299de79d872f0b300d0299356d7"
    ),
    "tests/integration/runtime/test_high_level_multidomain_qwen_v1.py": (
        "05836ee77994f8905a8a2523135872a81b8b67bcdae77e102e730ac9ef55025d"
    ),
    "tests/unit/experiments/test_high_level_multidomain_runner_v1.py": (
        "5a225fdd57b8690cf33b63eed5d0d8cf25d575ee46d929a7e9e67f7329d9ae92"
    ),
    "experiments/runners/high_level_multidomain_v1_r1.py": (
        "78152a1502ca8317c16c9aa77f913d46ec031229fa51e8aaf2d0de25f2c2398a"
    ),
    "tests/unit/experiments/test_high_level_multidomain_runner_v1_r1.py": (
        "9a929950b7f64b0f8220716f39187040ab52f7e3198dd92a39794c72b624e372"
    ),
    "tests/integration/runtime/test_high_level_multidomain_qwen_v1_r1.py": (
        "bb8aa7e931b6ab197741a56d7a54de68f975fcbb2cc1c9c4840424ff11b9cc43"
    ),
    (
        "docs/blueprints/branches/science/work/"
        "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R1-"
        "LIVE-EVALUATION-COMPLETION-001.md"
    ): "a54aa78d48fe5435b035e3c057470c915b87c8f4a4de9d001997a1dcd57d39de",
    "experiments/manifests/high-level-multidomain-v1-r1.json": (
        "2b8adadf402d52f1c0cd7a4523c855a0951fc04333fe43350009b1a8bd72c36d"
    ),
    "docs/reports/HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_R1_RESULT.md": (
        "114fd31e35e4bb0778eaeb7b25421c41b297b1b443a3dc95d5b5d6755954be7f"
    ),
}
FROZEN_SOURCE_HASHES = FROZEN_SOURCE_SHA256
REQUIRED_CONSTRUCTION_SOURCES = (
    "experiments/runners/high_level_multidomain_v1_r2.py",
    "tests/unit/experiments/test_high_level_multidomain_runner_v1_r2.py",
    "tests/integration/runtime/test_high_level_multidomain_qwen_v1_r2.py",
)

IMMUTABLE_R1_EVIDENCE_SHA256 = MappingProxyType(
    {
        "active_leaf": "a54aa78d48fe5435b035e3c057470c915b87c8f4a4de9d001997a1dcd57d39de",
        "arm_factory_audit": "01cd9c91e2407584e4b5a9db5dc9f818029f7c9341cac176121507cabd460183",
        "attempt_ledger": "f167a08029baa46eeff7c028e73c3e16d4f5973c5bbd5234fe6a4da61c0190f5",
        "exact_qwen_integration": "bb8aa7e931b6ab197741a56d7a54de68f975fcbb2cc1c9c4840424ff11b9cc43",
        "release_claim": "e5c4483e7162244bb5c2101961e253a94c8f628c44599cad4807ed2959683c68",
        "result_report": "114fd31e35e4bb0778eaeb7b25421c41b297b1b443a3dc95d5b5d6755954be7f",
        "retained_scratch_tree": "a9191c6653c9b5417c80e1779e821e318768a3dba8a732313d821bbbf979d43f",
        "retained_state_tree": "136f04653ed9dc32b1d8eff6c09921b12ab7ca6b258a33fde9f58e714a46a30c",
        "runner": "78152a1502ca8317c16c9aa77f913d46ec031229fa51e8aaf2d0de25f2c2398a",
        "runner_tests": "9a929950b7f64b0f8220716f39187040ab52f7e3198dd92a39794c72b624e372",
        "seed_seal": "4f7360f91a82c57a0212bb9fa5e95acd743d84b63a8d08bb32c7d724451f0d4f",
        "source_manifest": "2b8adadf402d52f1c0cd7a4523c855a0951fc04333fe43350009b1a8bd72c36d",
        "terminal_result": "547dbce00c9175fb97d4abcdf61cc8fb8ad0c2dd91f293c350a7c4700814c16c",
    }
)
IMMUTABLE_R1_RETAINED_TREE_WITNESSES = MappingProxyType(
    {
        "scratch": MappingProxyType(
            {
                "directories": 14,
                "files": 65,
                "logical_bytes": 812_194,
                "records": 79,
                "sha256": IMMUTABLE_R1_EVIDENCE_SHA256["retained_scratch_tree"],
            }
        ),
        "state": MappingProxyType(
            {
                "directories": 36,
                "files": 53,
                "logical_bytes": 11_487_875,
                "records": 89,
                "sha256": IMMUTABLE_R1_EVIDENCE_SHA256["retained_state_tree"],
            }
        ),
    }
)

CONSUMED_V1_SEED_COMMITMENTS = frozenset(
    {
        "sha256:86383b64d268bc944abffa1481d81cb730b75766d7d9f0f06d153ca9837cf4f9",
        "sha256:4ec98ef2fed6da44b2593301876d59ee051112e28d9c9caabc243f76481d7c4d",
    }
)
CONSUMED_R1_SEED_COMMITMENTS = frozenset(
    {
        "sha256:869675ad3b43bdd300f6ed58b8bf0af2b87520318b42e36d721565a7b7e9d9f5",
        "sha256:44ea52eb9a4085e0e803a965b53ad0fa6ac58cd642fc82ae130a583371b9316f",
    }
)

Arm = Literal[
    "FULL",
    "QWEN_ONLY",
    "RETRIEVAL_ONLY",
    "FROZEN_ORIGIN",
    "PROSPECTIVE_REMOVAL",
    "BACKEND_REMOVAL",
    "RANDOM_FEEDBACK",
    "QUALIFICATION",
]
Phase = Literal["adaptation", "development", "final"]
Purpose = Literal["qualification", "evaluation"]

EVALUATION_ARMS: tuple[Arm, ...] = (
    "FULL",
    "QWEN_ONLY",
    "RETRIEVAL_ONLY",
    "FROZEN_ORIGIN",
    "PROSPECTIVE_REMOVAL",
    "BACKEND_REMOVAL",
    "RANDOM_FEEDBACK",
)
STATEFUL_ARMS: tuple[Arm, ...] = (
    "FULL",
    "FROZEN_ORIGIN",
    "PROSPECTIVE_REMOVAL",
    "BACKEND_REMOVAL",
    "RANDOM_FEEDBACK",
)
CONTROL_ARMS: tuple[Arm, ...] = ("QWEN_ONLY", "RETRIEVAL_ONLY")
PHASES: tuple[Phase, ...] = ("adaptation", "development", "final")
FAMILIES: tuple[str, ...] = (
    "symbolic-demonstration-transfer",
    "glyph-machine",
    "causal-operator",
)

QUALIFICATION_COUNTS = (31, 62)
EVALUATION_COUNTS = (468, 936)

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SEED_COMMITMENT_DOMAIN = (
    b"angler.high-level-multidomain.v1\x00replicate-seed\x00"
)
_CONTROL_RECEIPT_DOMAIN = (
    b"angler.high-level-multidomain.control-selection.v1\x00"
)
_ADMISSION_DOMAIN = b"angler.high-level-multidomain.v1\x00final-admission\x00"
COMMITMENT_ONLY_ADMISSION_DIGEST = "sha256:" + hashlib.sha256(
    b"angler.high-level-multidomain.v1\x00commitment-only-admission-placeholder\x00"
).hexdigest()


class RunnerInvariantError(ValueError):
    """A frozen local integrity invariant was violated."""


class EvaluatorProtocolError(RunnerInvariantError):
    """The evaluator IPC peer violated the bounded canonical protocol."""


class EvaluatorTransportError(RunnerInvariantError):
    """The evaluator subprocess channel is no longer usable."""


class InfrastructureFailure(Exception):
    """Explicitly typed infrastructure failure; never inferred from messages."""

    def __init__(self, code: str, stage: str) -> None:
        self.code = _canonical_identifier(code, "infrastructure failure code")
        self.stage = _canonical_identifier(stage, "infrastructure failure stage")
        super().__init__(self.code)


class CleanupFailure(RunnerInvariantError):
    """Typed cleanup integrity failure with a bounded public code."""

    def __init__(self, code: str, stage: str) -> None:
        self.code = _canonical_identifier(code, "cleanup failure code")
        self.stage = _canonical_identifier(stage, "cleanup failure stage")
        super().__init__(self.code)


class AdmissionClaimPublicationFailure(RunnerInvariantError):
    """This invocation passed O_EXCL, so the evaluation identity is consumed."""

    code = "EVALUATION_ADMISSION_PUBLICATION_UNCERTAIN"
    stage = "EVALUATION_ADMISSION"

    def __init__(self) -> None:
        super().__init__(self.code)


class EvaluationIdentityConsumed(RunnerInvariantError):
    """Another invocation already owns the one-use evaluation identity."""


class QualificationIdentityConsumed(RunnerInvariantError):
    """Another invocation already owns the one-use qualification identity."""


class QualificationReleasePublicationFailure(RunnerInvariantError):
    """Qualification O_EXCL succeeded, so release is consumed but uncertain."""

    code = "QUALIFICATION_RELEASE_PUBLICATION_UNCERTAIN"
    stage = "QUALIFICATION_RELEASE"

    def __init__(self) -> None:
        super().__init__(self.code)


class RunMode(str, Enum):
    EVALUATOR_WORKER = "evaluator-worker"
    SEAL_EVALUATION_SEEDS = "seal-evaluation-seeds"
    QUALIFICATION = "qualification"
    EVALUATION = "evaluation"


_MODE_FLAGS: dict[str, RunMode] = {
    "--evaluator-worker": RunMode.EVALUATOR_WORKER,
    "--seal-evaluation-seeds": RunMode.SEAL_EVALUATION_SEEDS,
    "--qualification": RunMode.QUALIFICATION,
    "--evaluation": RunMode.EVALUATION,
}


def _text(value: object, label: str, maximum: int = 16_384) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(f"{label} must be bounded nonempty text")
    if value != value.strip() or value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{label} must be stripped Unicode NFC text")
    return value


def _digest(value: object, label: str = "digest") -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _raw_digest(value: object, label: str = "sha256") -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _canonical_identifier(value: object, label: str) -> str:
    if type(value) is not str or _CANONICAL_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical bounded identifier")
    return value


def _validate_exact_json(value: object, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("canonical JSON exceeds the nesting ceiling")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return
    if type(value) is str:
        if value != unicodedata.normalize("NFC", value):
            raise ValueError("canonical JSON strings must be Unicode NFC")
        return
    if type(value) is list:
        for item in value:
            _validate_exact_json(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or key != unicodedata.normalize("NFC", key):
                raise TypeError("canonical JSON object keys must be exact NFC strings")
            _validate_exact_json(item, depth=depth + 1)
        return
    raise TypeError("canonical JSON admits only exact JSON-native values")


def canonical_json_bytes(value: object) -> bytes:
    """Encode one finite, deterministic JSON value without repair."""

    _validate_exact_json(value)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("value is not finite canonical JSON") from error
    return encoded


def decode_canonical_json(
    data: bytes | str,
    *,
    maximum_bytes: int = MAXIMUM_IPC_BYTES,
) -> object:
    """Decode only an exact canonical JSON encoding."""

    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif type(data) is bytes:
        raw = data
    else:
        raise TypeError("canonical JSON input must be exact bytes or text")
    if not 1 <= len(raw) <= maximum_bytes:
        raise ValueError("canonical JSON size is outside its bound")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("canonical JSON cannot be decoded") from error
    if canonical_json_bytes(value) != raw:
        raise ValueError("JSON input is not in canonical form")
    return value


def content_ref(kind: str, value: object) -> str:
    kind = _text(kind, "content kind", 128)
    return "sha256:" + hashlib.sha256(
        kind.encode("ascii") + b"\x00" + canonical_json_bytes(value)
    ).hexdigest()


def _validate_qwen_journal_quiescence(
    value: object,
    quiescence_ref: object,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "entries",
        "journal_bytes",
        "journal_sha256",
        "kind",
    }:
        raise RunnerInvariantError("QWEN_ONLY journal quiescence fields differ")
    canonical = dict(value)
    if (
        canonical["kind"] != QWEN_JOURNAL_QUIESCENCE_KIND
        or type(canonical["entries"]) is not int
        or canonical["entries"] < 0
        or type(canonical["journal_bytes"]) is not int
        or canonical["journal_bytes"] < 1
    ):
        raise RunnerInvariantError("QWEN_ONLY journal quiescence values differ")
    _raw_digest(canonical["journal_sha256"], "QWEN_ONLY journal sha256")
    expected_ref = content_ref("runtime-quiescence", canonical)
    if quiescence_ref != expected_ref:
        raise RunnerInvariantError("QWEN_ONLY journal quiescence ref differs")
    return canonical


def attempt_receipt_payload(
    *,
    purpose: Purpose,
    phase: Phase,
    task_id: str,
    arm: Arm,
    parser_disposition: Literal["ADMITTED", "MALFORMED"],
    source_receipt_ref: str,
    source_kind: Literal["EXECUTION_RECEIPT", "PROPOSAL_GENERATION"],
) -> dict[str, object]:
    """Bind one consumed local attempt without changing its runtime receipt."""

    if purpose not in ("qualification", "evaluation") or phase not in PHASES:
        raise ValueError("attempt purpose or phase differs")
    _digest(task_id, "attempt task_id")
    if arm not in EVALUATION_ARMS:
        raise ValueError("attempt arm is not declared")
    if parser_disposition not in ("ADMITTED", "MALFORMED"):
        raise ValueError("attempt parser disposition differs")
    if source_kind not in ("EXECUTION_RECEIPT", "PROPOSAL_GENERATION") or (
        (parser_disposition == "ADMITTED")
        != (source_kind == "EXECUTION_RECEIPT")
    ):
        raise ValueError("attempt underlying receipt kind differs")
    _digest(source_receipt_ref, "attempt source receipt")
    payload = {
        "arm": arm,
        "nonauthorization": ATTEMPT_RECEIPT_NONAUTHORIZATION,
        "parser_disposition": parser_disposition,
        "phase": phase,
        "purpose": purpose,
        "schema": ATTEMPT_RECEIPT_SCHEMA,
        "source_kind": source_kind,
        "source_receipt_ref": source_receipt_ref,
        "task_id": task_id,
    }
    canonical_json_bytes(payload)
    return payload


def attempt_receipt_reference(payload: Mapping[str, object]) -> str:
    canonical = dict(payload)
    expected = attempt_receipt_payload(
        purpose=canonical.get("purpose"),  # type: ignore[arg-type]
        phase=canonical.get("phase"),  # type: ignore[arg-type]
        task_id=canonical.get("task_id"),  # type: ignore[arg-type]
        arm=canonical.get("arm"),  # type: ignore[arg-type]
        parser_disposition=canonical.get("parser_disposition"),  # type: ignore[arg-type]
        source_receipt_ref=canonical.get("source_receipt_ref"),  # type: ignore[arg-type]
        source_kind=canonical.get("source_kind"),  # type: ignore[arg-type]
    )
    if canonical != expected:
        raise ValueError("attempt receipt payload fields differ")
    return content_ref("arm-attempt-receipt", canonical)


def evaluator_record_digest(kind: str, payload: Mapping[str, object]) -> str:
    """Recompute the evaluator-owned public record digest domain exactly."""

    _text(kind, "evaluator digest kind", 128)
    material = {
        "kind": kind,
        "payload": dict(payload),
        "schema": SUITE_SCHEMA,
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def sha256_file(path: str | Path) -> tuple[str, int]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RunnerInvariantError(f"required regular file is absent: {target}")
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _canonical_source_path(value: object) -> str:
    name = _text(value, "source path", 512)
    candidate = Path(name)
    if (
        not name
        or "\\" in name
        or candidate.is_absolute()
        or candidate.as_posix() != name
        or name == "."
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise ValueError("source paths must be canonical repository-relative POSIX paths")
    try:
        name.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("source paths must be ASCII") from error
    return name


def _source_inventory_paths() -> tuple[str, ...]:
    expected_construction = (
        "experiments/runners/high_level_multidomain_v1_r2.py",
        "tests/unit/experiments/test_high_level_multidomain_runner_v1_r2.py",
        "tests/integration/runtime/test_high_level_multidomain_qwen_v1_r2.py",
    )
    if (
        type(REQUIRED_CONSTRUCTION_SOURCES) is not tuple
        or REQUIRED_CONSTRUCTION_SOURCES != expected_construction
        or len(set(REQUIRED_CONSTRUCTION_SOURCES)) != 3
    ):
        raise RunnerInvariantError("R2 construction source inventory differs")
    frozen = tuple(FROZEN_SOURCE_SHA256)
    if len(frozen) != 35 or len(set(frozen)) != 35:
        raise RunnerInvariantError("frozen source inventory differs")
    if set(frozen).intersection(REQUIRED_CONSTRUCTION_SOURCES):
        raise RunnerInvariantError("frozen and R2 construction sources overlap")
    combined = frozen + REQUIRED_CONSTRUCTION_SOURCES
    if len(combined) != 38 or len(set(combined)) != 38:
        raise RunnerInvariantError("completed source inventory differs")
    for name in combined:
        _canonical_source_path(name)
    return combined


def _hash_source_file(path: Path) -> tuple[str, int, tuple[int, int]]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise RunnerInvariantError(f"required regular source is absent: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RunnerInvariantError("source must be a real single-link regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            size != before.st_size
            or after.st_size != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_nlink != 1
        ):
            raise RunnerInvariantError("source changed while hashing")
        return digest.hexdigest(), size, (before.st_dev, before.st_ino)
    finally:
        os.close(descriptor)


def validate_frozen_sources(
    root: str | Path = REPOSITORY_ROOT,
    expected: Mapping[str, str] = FROZEN_SOURCE_SHA256,
) -> dict[str, dict[str, object]]:
    """Validate every frozen source before any adaptive/model result."""

    base = Path(root)
    if not base.is_absolute() or base.is_symlink() or not base.is_dir():
        raise ValueError("frozen source repository root must be exact and absolute")
    observed: dict[str, dict[str, object]] = {}
    identities: set[tuple[int, int]] = set()
    for relative, expected_digest in sorted(expected.items()):
        relative = _canonical_source_path(relative)
        _raw_digest(expected_digest, f"expected source hash for {relative}")
        path = base / relative
        digest, size, identity = _hash_source_file(path)
        if identity in identities:
            raise RunnerInvariantError("frozen sources share a filesystem identity")
        identities.add(identity)
        if digest != expected_digest:
            raise RunnerInvariantError(f"frozen source drift: {relative}")
        observed[relative] = {"bytes": size, "sha256": digest}
    return observed


def frozen_source_manifest(
    repository_root: str | Path = REPOSITORY_ROOT,
    expected: Mapping[str, str] = FROZEN_SOURCE_HASHES,
) -> dict[str, str]:
    """Return the exact canonical path-to-hash mapping after byte validation."""

    observed = validate_frozen_sources(repository_root, expected)
    return {
        path: str(values["sha256"])
        for path, values in sorted(observed.items())
    }


def scan_for_hidden_material(
    value: object,
    *,
    forbidden_values: Iterable[bytes | str] = (),
    maximum_bytes: int = MAXIMUM_IPC_BYTES,
) -> int:
    """Fail if a public envelope contains seed/answer/route-shaped material.

    This is defense in depth, not a semantic hidden-answer detector.  The
    evaluator process remains the primary role boundary.
    """

    encoded = canonical_json_bytes(value)
    if not 1 <= len(encoded) <= maximum_bytes:
        raise RunnerInvariantError("public material exceeds its byte ceiling")
    lowered = encoded.lower()
    forbidden_keys = (
        b'"raw_seed"',
        b'"seed_hex"',
        b'"hidden_answer"',
        b'"solution"',
        b'"route"',
        b'"mechanism_bytes"',
    )
    if any(token in lowered for token in forbidden_keys):
        raise RunnerInvariantError("public material exposes a forbidden hidden field")
    for forbidden in forbidden_values:
        raw = forbidden.encode("utf-8") if isinstance(forbidden, str) else forbidden
        if raw and raw in encoded:
            raise RunnerInvariantError("public material contains forbidden hidden bytes")
    return len(encoded)


@dataclass(frozen=True, slots=True)
class RunIntent:
    mode: RunMode

    @classmethod
    def parse(cls, argv: Sequence[str]) -> "RunIntent":
        values = tuple(argv)
        if len(values) != 1 or values[0] not in _MODE_FLAGS:
            raise RunnerInvariantError(
                "runner requires exactly one of --evaluator-worker, "
                "--seal-evaluation-seeds, --qualification, or --evaluation"
            )
        return cls(_MODE_FLAGS[values[0]])

    @property
    def intent_ref(self) -> str:
        return content_ref("run-intent", self.to_canonical())

    def to_canonical(self) -> dict[str, object]:
        return {"mode": self.mode.value, "schema": RUN_INTENT_SCHEMA}


@dataclass(frozen=True, slots=True)
class RunBudget:
    arm_tasks: int
    generation_attempts: int
    execution_attempts: int

    def __post_init__(self) -> None:
        for label, value in (
            ("arm_tasks", self.arm_tasks),
            ("generation_attempts", self.generation_attempts),
            ("execution_attempts", self.execution_attempts),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a non-negative exact integer")
        if self.generation_attempts != self.arm_tasks * 2:
            raise ValueError("generation ceiling must provide two slots per arm-task")
        if self.execution_attempts > self.arm_tasks:
            raise ValueError("execution ceiling exceeds the arm-task count")


def expected_run_budget(mode: RunMode | str) -> RunBudget:
    resolved = RunMode(mode)
    if resolved is RunMode.QUALIFICATION:
        return RunBudget(31, 62, 31)
    if resolved is RunMode.EVALUATION:
        return RunBudget(468, 936, 468)
    raise ValueError("only qualification and evaluation have arm-task budgets")


def seed_commitment(raw_seed: bytes) -> str:
    if type(raw_seed) is not bytes or len(raw_seed) != 32:
        raise ValueError("replicate seed must contain exactly 32 bytes")
    return "sha256:" + hashlib.sha256(
        _SEED_COMMITMENT_DOMAIN + raw_seed
    ).hexdigest()


def _validate_fresh_r2_seed_commitments(
    commitments: Sequence[str],
) -> tuple[str, str]:
    values = tuple(_digest(value, "R2 seed commitment") for value in commitments)
    if len(values) != 2 or len(set(values)) != 2:
        raise RunnerInvariantError("R2 seed commitments must be exactly two and distinct")
    if set(values) & CONSUMED_V1_SEED_COMMITMENTS:
        raise RunnerInvariantError("R2 seed commitments overlap consumed V1")
    if set(values) & CONSUMED_R1_SEED_COMMITMENTS:
        raise RunnerInvariantError("R2 seed commitments overlap consumed R1")
    return values  # type: ignore[return-value]


def qualification_replicate_commitment() -> str:
    """Return the frozen public commitment of the fixed qualification seed."""

    observed = seed_commitment(QUALIFICATION_SEED.to_bytes(32, "big"))
    if observed != QUALIFICATION_REPLICATE_COMMITMENT:
        raise RunnerInvariantError("qualification replicate commitment drifted")
    return observed


@dataclass(frozen=True, slots=True)
class SeedSeal:
    schema: str
    seed_commitments: tuple[str, str]
    artifact_sha256: str
    artifact_bytes: int
    path: str

    def __post_init__(self) -> None:
        if self.schema != SEED_SEAL_SCHEMA:
            raise ValueError("seed seal schema differs")
        if type(self.seed_commitments) is not tuple or len(self.seed_commitments) != 2:
            raise ValueError("seed seal requires exactly two commitments")
        for value in self.seed_commitments:
            _digest(value, "seed commitment")
        if len(set(self.seed_commitments)) != 2:
            raise ValueError("replicate seed commitments must be distinct")
        _raw_digest(self.artifact_sha256, "seed artifact sha256")
        if type(self.artifact_bytes) is not int or not 1 <= self.artifact_bytes <= MAXIMUM_SEED_ARTIFACT_BYTES:
            raise ValueError("seed artifact byte count is invalid")
        if not Path(self.path).is_absolute():
            raise ValueError("seed artifact path must be absolute")

    @property
    def seal_ref(self) -> str:
        return content_ref("seed-seal", self.to_canonical())

    def to_canonical(self) -> dict[str, object]:
        return {
            "artifact_bytes": self.artifact_bytes,
            "artifact_sha256": self.artifact_sha256,
            "path": self.path,
            "schema": self.schema,
            "seed_commitments": list(self.seed_commitments),
        }


def _seed_artifact_payload(seeds: tuple[bytes, bytes]) -> dict[str, object]:
    commitments = tuple(seed_commitment(seed) for seed in seeds)
    return {
        "identity": EVALUATION_IDENTITY,
        "replicates": [
            {
                "commitment": commitment,
                "label": f"replicate-{index:02d}",
                "seed_hex": seed.hex(),
            }
            for index, (seed, commitment) in enumerate(
                zip(seeds, commitments, strict=True), start=1
            )
        ],
        "schema": SEED_ARTIFACT_SCHEMA,
    }


def _decode_seed_artifact(raw: bytes, path: Path) -> tuple[SeedSeal, tuple[bytes, bytes]]:
    value = decode_canonical_json(raw, maximum_bytes=MAXIMUM_SEED_ARTIFACT_BYTES)
    if type(value) is not dict or set(value) != {"identity", "replicates", "schema"}:
        raise RunnerInvariantError("seed artifact fields differ")
    if value["schema"] != SEED_ARTIFACT_SCHEMA or value["identity"] != EVALUATION_IDENTITY:
        raise RunnerInvariantError("seed artifact identity differs")
    rows = value["replicates"]
    if type(rows) is not list or len(rows) != 2:
        raise RunnerInvariantError("seed artifact replicate count differs")
    seeds: list[bytes] = []
    commitments: list[str] = []
    for index, row in enumerate(rows, start=1):
        if type(row) is not dict or set(row) != {"commitment", "label", "seed_hex"}:
            raise RunnerInvariantError("seed artifact replicate fields differ")
        if row["label"] != f"replicate-{index:02d}":
            raise RunnerInvariantError("seed artifact replicate order differs")
        seed_hex = row["seed_hex"]
        if type(seed_hex) is not str or re.fullmatch(r"[0-9a-f]{64}", seed_hex) is None:
            raise RunnerInvariantError("seed artifact contains malformed seed bytes")
        seed = bytes.fromhex(seed_hex)
        commitment = seed_commitment(seed)
        if row["commitment"] != commitment:
            raise RunnerInvariantError("seed artifact commitment differs")
        seeds.append(seed)
        commitments.append(commitment)
    if seeds[0] == seeds[1]:
        raise RunnerInvariantError("replicate seeds must be distinct")
    _validate_fresh_r2_seed_commitments(commitments)
    seal = SeedSeal(
        schema=SEED_SEAL_SCHEMA,
        seed_commitments=(commitments[0], commitments[1]),
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        artifact_bytes=len(raw),
        path=str(path),
    )
    return seal, (seeds[0], seeds[1])


def _validate_private_file_metadata(
    path: Path,
    *,
    exact_mode: int = 0o600,
) -> os.stat_result:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise RunnerInvariantError(
            f"private artifact is absent or not regular: {path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or stat.S_IMODE(metadata.st_mode) != exact_mode
    ):
        raise RunnerInvariantError("private artifact ownership or mode differs")
    return metadata


def _validate_private_file(
    path: Path,
    *,
    exact_mode: int = 0o600,
    maximum_bytes: int = MAXIMUM_SEED_ARTIFACT_BYTES,
) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise RunnerInvariantError(
            f"private artifact is absent or not regular: {path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != exact_mode
        ):
            raise RunnerInvariantError("private artifact ownership or mode differs")
        if metadata.st_size > maximum_bytes:
            raise RunnerInvariantError("private artifact exceeds its byte ceiling")
        raw = os.read(descriptor, maximum_bytes + 1)
        if len(raw) != metadata.st_size or os.read(descriptor, 1):
            raise RunnerInvariantError("private artifact changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def load_seed_seal(
    path: str | Path,
    expected_commitments: Sequence[str] | None = None,
) -> SeedSeal:
    target = Path(path)
    raw = _validate_private_file(target)
    seal, _ = _decode_seed_artifact(raw, target)
    if expected_commitments is not None and tuple(expected_commitments) != seal.seed_commitments:
        raise RunnerInvariantError("seed artifact differs from published commitments")
    return seal


def _load_raw_seeds_for_worker(
    path: Path,
    expected_commitments: tuple[str, ...],
) -> tuple[bytes, bytes]:
    raw = _validate_private_file(path)
    seal, seeds = _decode_seed_artifact(raw, path)
    if seal.seed_commitments != expected_commitments:
        raise RunnerInvariantError("worker seed commitments differ")
    return seeds


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_private(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    for item in reversed(missing):
        item.mkdir(mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise RunnerInvariantError(f"private directory is not exact: {path}")
    metadata = path.stat()
    if (
        metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RunnerInvariantError("private directory ownership or mode differs")


def _require_owned_real_parent(path: Path) -> None:
    """Validate an existing output parent without changing its metadata."""

    if path.is_symlink() or not path.is_dir():
        raise RunnerInvariantError(f"output parent is not an exact directory: {path}")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_gid != os.getgid():
        raise RunnerInvariantError("output parent ownership differs")


def _create_fresh_private_root(root: str | Path) -> Path:
    """Create one caller-selected empty 0700 construction root."""

    target = Path(root)
    if (
        not target.is_absolute()
        or "." in target.parts
        or ".." in target.parts
        or target.parent == target
    ):
        raise ValueError("fresh construction root must be an exact absolute child path")
    if target.exists() or target.is_symlink():
        raise RunnerInvariantError("fresh construction root is already consumed")
    _require_owned_real_parent(target.parent)
    try:
        if target.parent.resolve(strict=True) != target.parent:
            raise RunnerInvariantError("fresh construction parent is not canonical")
    except OSError as error:
        raise RunnerInvariantError("fresh construction parent cannot be resolved") from error
    target.mkdir(mode=0o700)
    metadata = target.stat()
    if (
        target.is_symlink()
        or not target.is_dir()
        or any(target.iterdir())
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RunnerInvariantError("fresh construction root metadata differs")
    _fsync_directory(target.parent)
    return target


def _read_live_boundary_text(path: Path) -> str:
    return path.read_text(encoding="ascii")


def _read_live_namespace(path: Path) -> str:
    return os.readlink(path)


def validate_live_parent_boundary(
    *,
    configure_pools: bool = True,
    environment: Mapping[str, str] | None = None,
    executable: str | Path | None = None,
    runner_path: str | Path | None = None,
    working_directory: str | Path | None = None,
    interface_inventory: Callable[[], Sequence[tuple[int, str]]] | None = None,
    text_reader: Callable[[Path], str] | None = None,
    namespace_reader: Callable[[Path], str] | None = None,
    gpu_inventory_runner: Callable[..., object] | None = None,
    hostname_reader: Callable[[], str] | None = None,
    python_version_reader: Callable[[], str] | None = None,
    distribution_reader: Callable[[str], str] | None = None,
    torch_module: object | None = None,
) -> dict[str, object]:
    """Validate the exact offline parent and assigned workstation devices."""

    if type(configure_pools) is not bool:
        raise TypeError("pool configuration flag must be exact")
    observed_environment = dict(os.environ if environment is None else environment)
    if observed_environment != FROZEN_PARENT_ENVIRONMENT:
        raise RunnerInvariantError("live parent environment differs")
    observed_executable = Path(sys.executable if executable is None else executable)
    observed_runner = Path(__file__ if runner_path is None else runner_path).resolve()
    observed_cwd = Path.cwd() if working_directory is None else Path(working_directory)
    if (
        observed_executable != ANGLER_PYTHON
        or observed_runner != RUNNER_PATH
        or observed_cwd != REPOSITORY_ROOT
    ):
        raise RunnerInvariantError("live parent executable or working directory differs")
    temporary = Path(observed_environment["TMPDIR"])
    if temporary.is_symlink() or not temporary.is_dir() or any(temporary.iterdir()):
        raise RunnerInvariantError("live parent temporary directory is not fresh")
    temporary_metadata = temporary.stat()
    if (
        temporary_metadata.st_uid != os.getuid()
        or temporary_metadata.st_gid != os.getgid()
        or stat.S_IMODE(temporary_metadata.st_mode) != 0o700
    ):
        raise RunnerInvariantError("live parent temporary metadata differs")

    inventory = interface_inventory or socket.if_nameindex
    reader = text_reader or _read_live_boundary_text
    read_namespace = namespace_reader or _read_live_namespace
    interfaces = tuple(inventory())
    ipv4_lines = tuple(
        line
        for line in reader(Path("/proc/net/route")).splitlines()[1:]
        if line.strip()
    )
    ipv6_rows = tuple(
        tuple(line.split())
        for line in reader(Path("/proc/net/ipv6_route")).splitlines()
        if line.strip()
    )
    loopback_state = reader(Path("/sys/class/net/lo/operstate")).strip()
    namespace = read_namespace(Path("/proc/self/ns/net"))
    if (
        interfaces != ((1, "lo"),)
        or ipv4_lines
        or not ipv6_rows
        or any(len(row) != 10 or row[-1] != "lo" for row in ipv6_rows)
        or loopback_state != "unknown"
        or re.fullmatch(r"net:\[[0-9]+\]", namespace) is None
    ):
        raise RunnerInvariantError("live parent is not loopback-only and offline")

    get_hostname = hostname_reader or socket.gethostname
    get_python_version = python_version_reader or platform.python_version
    get_distribution = distribution_reader or distribution_version
    hostname = get_hostname()
    python_identity = get_python_version()
    if hostname != EXPECTED_HOSTNAME or python_identity != EXPECTED_PYTHON_VERSION:
        raise RunnerInvariantError("live host or Python identity differs")
    try:
        distributions = {
            name: get_distribution(name) for name in EXPECTED_PARENT_DISTRIBUTIONS
        }
    except PackageNotFoundError as error:
        raise RunnerInvariantError("live software distribution is absent") from error
    if distributions != EXPECTED_PARENT_DISTRIBUTIONS:
        raise RunnerInvariantError("live software distribution versions differ")

    command_runner = gpu_inventory_runner or subprocess.run
    completed = command_runner(
        (
            "/usr/bin/nvidia-smi",
            "--query-gpu=uuid,name,pci.bus_id,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = getattr(completed, "stdout", None)
    if type(output) is not str:
        raise RunnerInvariantError("physical GPU inventory output differs")
    physical: dict[str, dict[str, object]] = {}
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 6 or fields[0] in physical:
            raise RunnerInvariantError("physical GPU inventory is malformed")
        try:
            physical[fields[0]] = {
                "compute_capability": fields[3],
                "free_mib": int(fields[5]),
                "name": fields[1],
                "pci_bus_id": fields[2],
                "total_mib": int(fields[4]),
            }
        except ValueError as error:
            raise RunnerInvariantError("physical GPU inventory is malformed") from error
    if set(physical) != set(EXPECTED_PHYSICAL_GPUS) or any(
        {
            key: value
            for key, value in physical[uuid].items()
            if key != "free_mib"
        }
        != expected
        for uuid, expected in EXPECTED_PHYSICAL_GPUS.items()
    ):
        raise RunnerInvariantError("physical GPU inventory differs")
    if physical[ASSIGNED_GPU_UUID]["free_mib"] < MINIMUM_ASSIGNED_GPU_FREE_MIB:
        raise RunnerInvariantError("assigned GPU free-memory floor is not met")

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    set_threads = getattr(torch_module, "set_num_threads", None)
    set_interop = getattr(torch_module, "set_num_interop_threads", None)
    get_threads = getattr(torch_module, "get_num_threads", None)
    get_interop = getattr(torch_module, "get_num_interop_threads", None)
    cuda = getattr(torch_module, "cuda", None)
    required_pools = (get_threads, get_interop)
    if configure_pools:
        required_pools += (set_threads, set_interop)
    if not all(callable(item) for item in required_pools):
        raise TypeError("live Torch CPU pool boundary differs")
    if configure_pools:
        set_threads(2)
        set_interop(1)
    if get_threads() != 2 or get_interop() != 1:
        raise RunnerInvariantError("live Torch CPU pool configuration differs")
    torch_version = getattr(torch_module, "__version__", None)
    torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    cudnn_version = getattr(
        getattr(getattr(torch_module, "backends", None), "cudnn", None),
        "version",
        None,
    )
    observed_cudnn_version = cudnn_version() if callable(cudnn_version) else None
    if (
        not isinstance(torch_version, str)
        or torch_version != EXPECTED_TORCH_VERSION
        or type(torch_cuda_version) is not str
        or torch_cuda_version != EXPECTED_CUDA_VERSION
        or type(observed_cudnn_version) is not int
        or observed_cudnn_version != EXPECTED_CUDNN_RUNTIME
    ):
        raise RunnerInvariantError("live Torch, CUDA, or cuDNN identity differs")
    required_cuda = (
        "is_available",
        "device_count",
        "set_device",
        "current_device",
        "get_device_name",
    )
    if cuda is None or any(not callable(getattr(cuda, name, None)) for name in required_cuda):
        raise TypeError("live CUDA inventory boundary differs")
    if not cuda.is_available() or cuda.device_count() != 1:
        raise RunnerInvariantError("CUDA does not expose exactly one logical GPU")
    cuda.set_device(0)
    if cuda.current_device() != 0 or cuda.get_device_name(0) != FROZEN_GPU_ASSIGNMENT[
        "assigned_name"
    ]:
        raise RunnerInvariantError("logical cuda:0 is not the assigned RTX 5080")
    pool_sum = (
        int(observed_environment["OMP_NUM_THREADS"])
        + int(observed_environment["MKL_NUM_THREADS"])
        + int(observed_environment["OPENBLAS_NUM_THREADS"])
        + 2
        + 2
    )
    if pool_sum != 8:
        raise RunnerInvariantError("configured compute-pool sum differs")
    environment_sha256 = hashlib.sha256(
        canonical_json_bytes(observed_environment)
    ).hexdigest()
    result = {
        "configured_pool_sum": pool_sum,
        "environment_sha256": environment_sha256,
        "hostname": hostname,
        "interfaces": [list(item) for item in interfaces],
        "logical_cuda_device": 0,
        "network_namespace": namespace,
        "physical_gpus": [
            {**EXPECTED_PHYSICAL_GPUS[uuid], "uuid": uuid}
            for uuid in sorted(physical)
        ],
        "python": python_identity,
        "python_executable": str(observed_executable),
        "runner": str(observed_runner),
        "software": {
            # torch.__version__ is torch.torch_version.TorchVersion, a str
            # subclass.  Preserve equality against the observed identities
            # above, but emit only exact JSON-native frozen strings.
            "cuda": EXPECTED_CUDA_VERSION,
            "cudnn_runtime": EXPECTED_CUDNN_RUNTIME,
            "distributions": distributions,
            "torch": EXPECTED_TORCH_VERSION,
        },
        "temporary_directory": str(temporary),
    }
    canonical_json_bytes(result)
    return result


def _write_private_artifact(
    path: Path,
    data: bytes,
    *,
    maximum_bytes: int = MAXIMUM_GENESIS_ARTIFACT_BYTES,
) -> None:
    if (
        type(maximum_bytes) is not int
        or maximum_bytes < 1
        or type(data) is not bytes
        or not 1 <= len(data) <= maximum_bytes
    ):
        raise ValueError("private construction artifact bytes are outside their bound")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_private_artifact(
    path: Path,
    *,
    maximum_bytes: int = MAXIMUM_GENESIS_ARTIFACT_BYTES,
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("private construction artifact ceiling must be positive")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise RunnerInvariantError(f"private construction artifact is absent: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 1 <= before.st_size <= maximum_bytes
        ):
            raise RunnerInvariantError("private construction artifact metadata differs")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RunnerInvariantError("private construction artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RunnerInvariantError("private construction artifact grew while read")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise RunnerInvariantError("private construction artifact changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _replace_private_artifact(
    path: Path,
    data: bytes,
    *,
    maximum_bytes: int,
) -> None:
    """Atomically replace one owned mutable private snapshot."""

    _read_private_artifact(path, maximum_bytes=maximum_bytes)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RunnerInvariantError("private snapshot temporary identity is consumed")
    try:
        _write_private_artifact(
            temporary,
            data,
            maximum_bytes=maximum_bytes,
        )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _qualified_genesis_record() -> dict[str, object]:
    record = {
        "checkpoint_constructor": (
            "CompositeProspectiveCreditCore(StructureKeyedCreditMemoryCore(**credit_config), "
            "ProspectiveDynamicsConfig(**prospective_config))"
        ),
        "credit_config": dict(QUALIFIED_GENESIS_CREDIT_CONFIG),
        "device": "cpu",
        "dtype": "torch.float32",
        "initial_state_constructor": "core.initial_state()",
        "prospective_config": dict(QUALIFIED_GENESIS_PROSPECTIVE_CONFIG),
        "prospective_lesion": False,
        "schema": "angler.frozen-qwen-learner-genesis.v1",
        "torch_manual_seed": QUALIFIED_GENESIS_SEED,
    }
    material = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    observed = "sha256:" + hashlib.sha256(
        b"angler.frozen-qwen-learner-genesis-config.v1\0" + material
    ).hexdigest()
    if observed != QUALIFIED_GENESIS_CONFIG_REF:
        raise RunnerInvariantError("qualified learner genesis configuration differs")
    return record


def _build_qualified_genesis_core() -> object:
    from angler.reasoning.prospective_dynamics import (
        CompositeProspectiveCreditCore,
        ProspectiveDynamicsConfig,
    )
    from angler.reasoning.structure_keyed_credit_memory import (
        StructureKeyedCreditMemoryCore,
    )

    record = _qualified_genesis_record()
    credit_config = record["credit_config"]
    prospective_config = record["prospective_config"]
    if type(credit_config) is not dict or type(prospective_config) is not dict:
        raise RunnerInvariantError("qualified learner genesis record shape differs")
    # Ordering is identity-bearing: the credit core consumes CPU RNG first.
    credit = StructureKeyedCreditMemoryCore(**credit_config)
    config = ProspectiveDynamicsConfig(**prospective_config)
    core = CompositeProspectiveCreditCore(credit, config)
    if (
        "sha256:" + config.digest != QUALIFIED_PROSPECTIVE_CONFIG_REF
        or core.checkpoint_identity != QUALIFIED_LEARNER_CHECKPOINT_REF
        or core.parameter_count != QUALIFIED_GENESIS_PARAMETER_COUNT
    ):
        raise RunnerInvariantError("qualified learner genesis topology differs")
    return core


def _new_qualified_genesis_learner(
    core: object,
    *,
    prospective_lesion: bool,
) -> object:
    from angler.runtime.durable_ability_bridge import DurableAbilityLearner

    if type(prospective_lesion) is not bool:
        raise TypeError("prospective_lesion must be an exact bool")
    return DurableAbilityLearner(
        core,
        core.initial_state(),  # type: ignore[attr-defined]
        checkpoint_identity=core.checkpoint_identity,  # type: ignore[attr-defined]
        prospective_lesion=prospective_lesion,
    )


def _verify_qualified_genesis_learner(
    learner: object,
    *,
    expected_snapshot: bytes | None = None,
    prospective_lesion: bool,
) -> bytes:
    from angler.reasoning.prospective_dynamics import CompositeProspectiveCreditCore

    if type(prospective_lesion) is not bool:
        raise TypeError("prospective_lesion must be an exact bool")
    core = getattr(learner, "core", None)
    if type(core) is not CompositeProspectiveCreditCore:
        raise RunnerInvariantError("qualified learner genesis core class differs")
    integrity = learner.component_state_integrity()
    snapshot = learner.capture_state()
    if (
        core.checkpoint_identity != QUALIFIED_LEARNER_CHECKPOINT_REF
        or learner.checkpoint_identity != QUALIFIED_LEARNER_CHECKPOINT_REF
        or learner.prospective_lesion is not prospective_lesion
        or integrity.checkpoint_ref != QUALIFIED_LEARNER_CHECKPOINT_REF
        or integrity.config_ref != QUALIFIED_PROSPECTIVE_CONFIG_REF
        or integrity.step != 0
        or learner.state_digest() != QUALIFIED_INITIAL_COMPETENCE_DIGEST
        or learner.events
        or learner.tombstoned_evidence_refs
        or learner.pending_decision is not None
        or len(snapshot) != QUALIFIED_GENESIS_SNAPSHOT_BYTES
        or hashlib.sha256(snapshot).hexdigest()
        != QUALIFIED_GENESIS_SNAPSHOT_SHA256
        or (expected_snapshot is not None and snapshot != expected_snapshot)
    ):
        raise RunnerInvariantError("qualified learner genesis reconstruction differs")
    return snapshot


@dataclass(frozen=True, slots=True)
class FreshGenesisBundle:
    """One fresh, exact CPU genesis and its root-independent binding."""

    root: Path
    core_state_bytes: int
    core_state_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("fresh genesis root must be an absolute Path")
        if (
            type(self.core_state_bytes) is not int
            or not 1 <= self.core_state_bytes <= MAXIMUM_GENESIS_ARTIFACT_BYTES
        ):
            raise ValueError("fresh genesis core-state size is outside its bound")
        _raw_digest(self.core_state_sha256, "fresh genesis core-state sha256")

    @property
    def snapshot_path(self) -> Path:
        return self.root / "learner-snapshot.pt"

    @property
    def core_state_path(self) -> Path:
        return self.root / "core-state.pt"

    @property
    def binding_path(self) -> Path:
        return self.root / "binding.json"

    def to_canonical(self) -> dict[str, object]:
        return {
            "checkpoint_ref": QUALIFIED_LEARNER_CHECKPOINT_REF,
            "core_state_bytes": self.core_state_bytes,
            "core_state_sha256": self.core_state_sha256,
            "genesis_config_ref": QUALIFIED_GENESIS_CONFIG_REF,
            "initial_competence_state_digest": (
                QUALIFIED_INITIAL_COMPETENCE_DIGEST
            ),
            "parameter_count": QUALIFIED_GENESIS_PARAMETER_COUNT,
            "prospective_config_ref": QUALIFIED_PROSPECTIVE_CONFIG_REF,
            "schema": FRESH_GENESIS_SCHEMA,
            "seed": QUALIFIED_GENESIS_SEED,
            "snapshot_bytes": QUALIFIED_GENESIS_SNAPSHOT_BYTES,
            "snapshot_sha256": QUALIFIED_GENESIS_SNAPSHOT_SHA256,
        }

    @property
    def binding_ref(self) -> str:
        return content_ref("fresh-genesis", self.to_canonical())

    def validate_artifacts(self) -> tuple[bytes, bytes]:
        if self.root.is_symlink() or not self.root.is_dir():
            raise RunnerInvariantError("fresh genesis root is no longer exact")
        metadata = self.root.stat()
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or {item.name for item in self.root.iterdir()}
            != {"binding.json", "core-state.pt", "learner-snapshot.pt"}
        ):
            raise RunnerInvariantError("fresh genesis root contents differ")
        snapshot = _read_private_artifact(self.snapshot_path)
        core_state = _read_private_artifact(self.core_state_path)
        binding = decode_canonical_json(
            _read_private_artifact(self.binding_path),
            maximum_bytes=MAXIMUM_GENESIS_ARTIFACT_BYTES,
        )
        expected_binding = {
            **self.to_canonical(),
            "binding_ref": self.binding_ref,
        }
        if (
            binding != expected_binding
            or len(snapshot) != QUALIFIED_GENESIS_SNAPSHOT_BYTES
            or hashlib.sha256(snapshot).hexdigest()
            != QUALIFIED_GENESIS_SNAPSHOT_SHA256
            or len(core_state) != self.core_state_bytes
            or hashlib.sha256(core_state).hexdigest() != self.core_state_sha256
        ):
            raise RunnerInvariantError("fresh genesis artifact binding differs")
        return snapshot, core_state


def reconstruct_fresh_cpu_genesis(root: str | Path) -> FreshGenesisBundle:
    """Reconstruct the qualified CPU genesis without changing caller RNG."""

    import torch

    target = _create_fresh_private_root(root)
    caller_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[], enabled=True):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(QUALIFIED_GENESIS_SEED)
        torch.set_rng_state(generator.get_state())
        core = _build_qualified_genesis_core()
        learner = _new_qualified_genesis_learner(
            core,
            prospective_lesion=False,
        )
        snapshot = _verify_qualified_genesis_learner(
            learner,
            prospective_lesion=False,
        )
        state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in core.state_dict().items()
        }
    if not torch.equal(torch.get_rng_state(), caller_rng):
        raise RunnerInvariantError("fresh genesis reconstruction changed caller CPU RNG")
    if not state or any(
        type(name) is not str
        or not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.requires_grad
        for name, value in state.items()
    ):
        raise RunnerInvariantError("fresh genesis core state differs")
    stream = io.BytesIO()
    torch.save(
        {
            "checkpoint_ref": QUALIFIED_LEARNER_CHECKPOINT_REF,
            "config_ref": QUALIFIED_PROSPECTIVE_CONFIG_REF,
            "core_state_dict": state,
            "version": 1,
        },
        stream,
    )
    core_state = stream.getvalue()
    if not 1 <= len(core_state) <= MAXIMUM_GENESIS_ARTIFACT_BYTES:
        raise RunnerInvariantError("fresh genesis core-state artifact exceeds its bound")
    bundle = FreshGenesisBundle(
        root=target,
        core_state_bytes=len(core_state),
        core_state_sha256=hashlib.sha256(core_state).hexdigest(),
    )
    _write_private_artifact(bundle.snapshot_path, snapshot)
    _write_private_artifact(bundle.core_state_path, core_state)
    _write_private_artifact(
        bundle.binding_path,
        canonical_json_bytes(
            {**bundle.to_canonical(), "binding_ref": bundle.binding_ref}
        ),
    )
    _fsync_directory(target)
    bundle.validate_artifacts()
    return bundle


def restore_fresh_cpu_genesis(
    bundle: FreshGenesisBundle,
    *,
    prospective_lesion: bool = False,
) -> object:
    """Return one independently owned learner from an exact fresh bundle."""

    import torch

    if type(bundle) is not FreshGenesisBundle:
        raise TypeError("fresh genesis bundle must be exact FreshGenesisBundle")
    if type(prospective_lesion) is not bool:
        raise TypeError("prospective_lesion must be an exact bool")
    snapshot, core_state = bundle.validate_artifacts()
    try:
        envelope = torch.load(
            io.BytesIO(core_state),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RunnerInvariantError("fresh genesis core-state cannot be decoded") from error
    if type(envelope) is not dict or set(envelope) != {
        "checkpoint_ref",
        "config_ref",
        "core_state_dict",
        "version",
    }:
        raise RunnerInvariantError("fresh genesis core-state envelope differs")
    raw_state = envelope["core_state_dict"]
    if (
        envelope["version"] != 1
        or envelope["checkpoint_ref"] != QUALIFIED_LEARNER_CHECKPOINT_REF
        or envelope["config_ref"] != QUALIFIED_PROSPECTIVE_CONFIG_REF
        or type(raw_state) is not dict
    ):
        raise RunnerInvariantError("fresh genesis core-state identity differs")
    caller_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[], enabled=True):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(QUALIFIED_GENESIS_SEED)
        torch.set_rng_state(generator.get_state())
        core = _build_qualified_genesis_core()
    if not torch.equal(torch.get_rng_state(), caller_rng):
        raise RunnerInvariantError("fresh genesis restore changed caller CPU RNG")
    expected_keys = set(core.state_dict())
    if set(raw_state) != expected_keys or any(
        type(name) is not str
        or not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.requires_grad
        for name, value in raw_state.items()
    ):
        raise RunnerInvariantError("fresh genesis restored core state differs")
    core.load_state_dict(raw_state, strict=True)
    learner = _new_qualified_genesis_learner(
        core,
        prospective_lesion=prospective_lesion,
    )
    learner.restore_state(snapshot)
    _verify_qualified_genesis_learner(
        learner,
        expected_snapshot=snapshot,
        prospective_lesion=prospective_lesion,
    )
    return learner


@dataclass(frozen=True, slots=True)
class _FoundationParameterIdentity:
    name: str
    parameter: object = field(repr=False, compare=False)
    shape: tuple[int, ...]
    dtype: str
    device: str
    storage_pointer: int
    storage_offset: int
    stride: tuple[int, ...]
    version: int
    requires_grad: bool


class FoundationGuard:
    """Cheap per-probe identity guard plus exact whole-boundary tensor hash."""

    def __init__(
        self,
        model: object,
        *,
        tensor_digest: Callable[[object], str] | None = None,
    ) -> None:
        if tensor_digest is None:
            from angler.runtime.qwen_peft import foundation_tensor_digest

            tensor_digest = foundation_tensor_digest
        if not callable(tensor_digest):
            raise TypeError("foundation tensor digest boundary must be callable")
        self._model = model
        self._tensor_digest = tensor_digest
        self._identities = self._capture()
        self._initial_tensor_digest = _digest(
            tensor_digest(model),
            "initial foundation tensor digest",
        )
        self._validate(self._capture())

    @staticmethod
    def _identity(name: object, parameter: object) -> _FoundationParameterIdentity:
        name = _text(name, "foundation parameter name", 2_048)
        try:
            shape = tuple(int(value) for value in parameter.shape)  # type: ignore[attr-defined]
            dtype = str(parameter.dtype)  # type: ignore[attr-defined]
            device = str(parameter.device)  # type: ignore[attr-defined]
            storage = parameter.untyped_storage()  # type: ignore[attr-defined]
            storage_pointer = int(storage.data_ptr())
            storage_offset = int(parameter.storage_offset())  # type: ignore[attr-defined]
            stride = tuple(int(value) for value in parameter.stride())  # type: ignore[attr-defined]
            version = int(parameter._version)  # type: ignore[attr-defined]
            requires_grad = parameter.requires_grad  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise RunnerInvariantError("foundation parameter metadata is unavailable") from error
        if (
            not dtype
            or not device
            or storage_pointer < 0
            or storage_offset < 0
            or version < 0
            or type(requires_grad) is not bool
        ):
            raise RunnerInvariantError("foundation parameter metadata is invalid")
        return _FoundationParameterIdentity(
            name=name,
            parameter=parameter,
            shape=shape,
            dtype=dtype,
            device=device,
            storage_pointer=storage_pointer,
            storage_offset=storage_offset,
            stride=stride,
            version=version,
            requires_grad=requires_grad,
        )

    def _capture(self) -> tuple[_FoundationParameterIdentity, ...]:
        if getattr(self._model, "training", None) is not False:
            raise RunnerInvariantError("foundation model training mode drifted")
        named_parameters = getattr(self._model, "named_parameters", None)
        if not callable(named_parameters):
            raise TypeError("foundation model lacks named_parameters")
        try:
            rows = tuple(named_parameters())
        except Exception as error:
            raise RunnerInvariantError("foundation parameter inventory failed") from error
        if (
            not rows
            or any(type(row) is not tuple or len(row) != 2 for row in rows)
        ):
            raise RunnerInvariantError("foundation parameter inventory is empty or malformed")
        identities = tuple(self._identity(name, parameter) for name, parameter in rows)
        if len({identity.name for identity in identities}) != len(identities):
            raise RunnerInvariantError("foundation parameter names are duplicated")
        if any(identity.requires_grad for identity in identities):
            raise RunnerInvariantError("foundation parameter requires_grad drifted")
        return identities

    def _validate(
        self,
        observed: tuple[_FoundationParameterIdentity, ...],
    ) -> None:
        expected = self._identities
        if tuple(row.name for row in observed) != tuple(row.name for row in expected):
            raise RunnerInvariantError("foundation parameter name/order drifted")
        for prior, current in zip(expected, observed, strict=True):
            if current.parameter is not prior.parameter:
                raise RunnerInvariantError("foundation parameter was replaced")
            for label in (
                "shape",
                "dtype",
                "device",
                "storage_pointer",
                "storage_offset",
                "stride",
                "version",
                "requires_grad",
            ):
                if getattr(current, label) != getattr(prior, label):
                    raise RunnerInvariantError(
                        f"foundation parameter {label} drifted"
                    )

    @property
    def initial_tensor_digest(self) -> str:
        return self._initial_tensor_digest

    def probe(self) -> str:
        self._validate(self._capture())
        return self._initial_tensor_digest

    def verify_boundary_digest(self) -> str:
        """Perform the one expensive whole-tensor recomputation at a boundary."""

        self.probe()
        observed = _digest(
            self._tensor_digest(self._model),
            "boundary foundation tensor digest",
        )
        self.probe()
        if observed != self._initial_tensor_digest:
            raise RunnerInvariantError("foundation tensor bytes changed")
        return observed


def qualified_qwen_cycle_manifest() -> object:
    from angler.runtime.qwen_cognitive import FrozenQwenCycleManifestV1

    return FrozenQwenCycleManifestV1(
        model_ref=QUALIFIED_MODEL_REF,
        tokenizer_ref=QUALIFIED_TOKENIZER_REF,
        genesis_config_ref=QUALIFIED_GENESIS_CONFIG_REF,
        prospective_config_ref=QUALIFIED_PROSPECTIVE_CONFIG_REF,
        learner_checkpoint_ref=QUALIFIED_LEARNER_CHECKPOINT_REF,
        initial_competence_state_digest=QUALIFIED_INITIAL_COMPETENCE_DIGEST,
        genesis_snapshot_sha256=QUALIFIED_GENESIS_SNAPSHOT_SHA256,
        genesis_snapshot_bytes=QUALIFIED_GENESIS_SNAPSHOT_BYTES,
        genesis_seed=QUALIFIED_GENESIS_SEED,
        prospective_lesion=False,
    )


def _bundle_manifest(root: Path, names: Sequence[str]) -> tuple[str, int]:
    records: list[str] = []
    total = 0
    for name in sorted(names):
        _text(name, "model bundle file name", 512)
        if Path(name).name != name or "\t" in name or "\n" in name:
            raise RunnerInvariantError("model bundle file name is not exact")
        digest, size = sha256_file(root / name)
        records.append(f"{digest}\t{size}\t{name}\n")
        total += size
    return hashlib.sha256("".join(records).encode("utf-8")).hexdigest(), total


def _guarded_tree_manifest(root: Path) -> dict[str, object]:
    """Content-address one frozen cache tree with predecessor-compatible bytes."""

    digest = hashlib.sha256()
    total = 0
    files = 0
    if root.is_symlink():
        raise RunnerInvariantError("guarded cache root is a symlink")
    if not root.exists():
        digest.update(b"ABSENT\n")
        return {
            "exists": False,
            "files": 0,
            "manifest_sha256": digest.hexdigest(),
            "total_bytes": 0,
        }
    if not root.is_dir():
        raise RunnerInvariantError("guarded cache root is not a directory")
    paths = sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(
                f"L\t{relative}\t{os.readlink(path)}\n".encode("utf-8")
            )
        elif path.is_dir():
            digest.update(f"D\t{relative}\n".encode("utf-8"))
        elif path.is_file():
            file_digest, size = sha256_file(path)
            digest.update(
                f"F\t{file_digest}\t{size}\t{relative}\n".encode("utf-8")
            )
            files += 1
            total += size
        else:
            raise RunnerInvariantError("guarded cache tree contains a special path")
    return {
        "exists": True,
        "files": files,
        "manifest_sha256": digest.hexdigest(),
        "total_bytes": total,
    }


def _repository_tree_manifest(root: Path) -> dict[str, object]:
    """Bind the preserved dirty repository without following symlinks."""

    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("repository guard root must be an absolute Path")
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
        root_metadata.st_mode
    ):
        raise RunnerInvariantError("repository guard root differs")
    paths: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in directory.iterdir():
            metadata = child.lstat()
            paths.append(child)
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
    paths.sort(key=lambda item: item.relative_to(root).as_posix().encode("utf-8"))
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0

    def add_record(record: Mapping[str, object]) -> None:
        digest.update(canonical_json_bytes(dict(record)))
        digest.update(b"\n")

    add_record(
        {
            "device": int(root_metadata.st_dev),
            "gid": int(root_metadata.st_gid),
            "inode": int(root_metadata.st_ino),
            "kind": "directory",
            "links": int(root_metadata.st_nlink),
            "mode": stat.S_IMODE(root_metadata.st_mode),
            "path": "",
            "uid": int(root_metadata.st_uid),
        }
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        before = path.lstat()
        common = {
            "device": int(before.st_dev),
            "gid": int(before.st_gid),
            "inode": int(before.st_ino),
            "links": int(before.st_nlink),
            "mode": stat.S_IMODE(before.st_mode),
            "path": relative,
            "uid": int(before.st_uid),
        }
        if stat.S_ISDIR(before.st_mode):
            add_record({**common, "kind": "directory"})
        elif stat.S_ISLNK(before.st_mode):
            add_record(
                {
                    **common,
                    "kind": "symlink",
                    "target": os.readlink(path),
                }
            )
        elif stat.S_ISREG(before.st_mode):
            file_digest, size = sha256_file(path)
            after = path.lstat()
            stable_fields = (
                "st_dev",
                "st_gid",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_uid",
            )
            if any(
                getattr(before, name) != getattr(after, name)
                for name in stable_fields
            ) or size != before.st_size:
                raise RunnerInvariantError(
                    "repository file changed while its baseline was captured"
                )
            add_record(
                {
                    **common,
                    "kind": "file",
                    "sha256": file_digest,
                    "size": size,
                }
            )
            files += 1
            total_bytes += size
        else:
            raise RunnerInvariantError("repository tree contains a special path")
    return {
        "entries": len(paths) + 1,
        "files": files,
        "manifest_sha256": digest.hexdigest(),
        "schema": REPOSITORY_TREE_SCHEMA,
        "total_bytes": total_bytes,
    }


def _validated_repository_tree_manifest(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "entries",
        "files",
        "manifest_sha256",
        "schema",
        "total_bytes",
    }:
        raise RunnerInvariantError("repository tree manifest fields differ")
    if (
        value["schema"] != REPOSITORY_TREE_SCHEMA
        or type(value["entries"]) is not int
        or value["entries"] < 1
        or type(value["files"]) is not int
        or not 0 <= value["files"] < value["entries"]
        or type(value["total_bytes"]) is not int
        or value["total_bytes"] < 0
    ):
        raise RunnerInvariantError("repository tree manifest values differ")
    _raw_digest(value["manifest_sha256"], "repository tree manifest sha256")
    return dict(value)


def _validated_tree_manifest(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "exists",
        "files",
        "manifest_sha256",
        "total_bytes",
    }:
        raise RunnerInvariantError(f"{label} tree manifest fields differ")
    if (
        type(value["exists"]) is not bool
        or type(value["files"]) is not int
        or value["files"] < 0
        or type(value["total_bytes"]) is not int
        or value["total_bytes"] < 0
        or type(value["manifest_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["manifest_sha256"]) is None
        or (not value["exists"] and (value["files"] or value["total_bytes"]))
    ):
        raise RunnerInvariantError(f"{label} tree manifest values differ")
    canonical_json_bytes(value)
    return dict(value)


def _verify_qualified_model_files(root: Path) -> dict[str, object]:
    if root != MODEL_ROOT or root.is_symlink() or not root.is_dir():
        raise RunnerInvariantError("qualified Qwen root differs")
    names = tuple(item.name for item in root.iterdir() if item.is_file())
    manifest, total = _bundle_manifest(root, names)
    tokenizer_manifest, _ = _bundle_manifest(root, QUALIFIED_TOKENIZER_FILES)
    if (
        len(names) != QUALIFIED_MODEL_FILE_COUNT
        or total != QUALIFIED_MODEL_TOTAL_BYTES
        or manifest != QUALIFIED_MODEL_ROOT_SHA256
        or tokenizer_manifest != QUALIFIED_TOKENIZER_SHA256
    ):
        raise RunnerInvariantError("qualified Qwen file identity differs")
    fastembed = _guarded_tree_manifest(FASTEMBED_CACHE_ROOT)
    tiktoken = _guarded_tree_manifest(TIKTOKEN_CACHE_ROOT)
    if (
        fastembed != EXPECTED_FASTEMBED_CACHE_TREE
        or tiktoken != EXPECTED_TIKTOKEN_CACHE_TREE
    ):
        raise RunnerInvariantError("qualified embedding/tokenizer cache identity differs")
    return {
        "fastembed_cache_tree": fastembed,
        "file_count": len(names),
        "model_cache_tree": _guarded_tree_manifest(root / ".cache"),
        "root_manifest_sha256": manifest,
        "schema": FOUNDATION_LOAD_SCHEMA,
        "tiktoken_cache_tree": tiktoken,
        "tokenizer_manifest_sha256": tokenizer_manifest,
        "total_bytes": total,
    }


@dataclass(frozen=True, slots=True)
class LoadedQwenFoundation:
    model: object = field(repr=False, compare=False)
    tokenizer: object = field(repr=False, compare=False)
    io: object = field(repr=False, compare=False)
    manifest: object = field(repr=False, compare=False)
    guard: FoundationGuard = field(repr=False, compare=False)
    production_file_evidence: Mapping[str, object] | None

    @property
    def production_file_identity_verified(self) -> bool:
        return self.production_file_evidence is not None


def load_qualified_qwen_foundation(
    model_root: str | Path = MODEL_ROOT,
    *,
    model_factory: Callable[..., object] | None = None,
    tokenizer_factory: Callable[..., object] | None = None,
) -> LoadedQwenFoundation:
    """Load exactly one frozen Qwen/LocalQwenIO owner, lazily.

    Injected factories are a CPU construction-test seam only.  They require a
    non-live root and the returned owner is explicitly not production-file
    verified; the default path performs the qualified 8-GB file binding.
    """

    import torch
    from angler.runtime.situated_qwen import LocalQwenIO

    root = Path(model_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Qwen model root must be an exact absolute directory")
    injected = model_factory is not None or tokenizer_factory is not None
    if injected != (model_factory is not None and tokenizer_factory is not None):
        raise ValueError("model and tokenizer factories must be injected together")
    if injected:
        if root == MODEL_ROOT or root in MODEL_ROOT.parents or MODEL_ROOT in root.parents:
            raise RunnerInvariantError("injected factories cannot target the live model tree")
        file_evidence: Mapping[str, object] | None = None
    else:
        file_evidence = MappingProxyType(_verify_qualified_model_files(root))
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_factory = AutoModelForCausalLM.from_pretrained
        tokenizer_factory = AutoTokenizer.from_pretrained
    assert model_factory is not None and tokenizer_factory is not None
    tokenizer: object | None = None
    model: object | None = None
    qwen_io: object | None = None
    try:
        tokenizer = tokenizer_factory(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = model_factory(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        )
        freeze = getattr(model, "requires_grad_", None)
        evaluate = getattr(model, "eval", None)
        if not callable(freeze) or not callable(evaluate):
            raise TypeError("loaded Qwen is not a mutable module boundary")
        freeze(False)
        evaluate()
        parameters = tuple(model.parameters())  # type: ignore[attr-defined]
        config = getattr(model, "config", None)
        expected_device = "cpu" if injected else "cuda:0"
        if (
            type(model).__name__ != "Qwen3ForCausalLM"
            or config is None
            or int(getattr(config, "hidden_size", -1)) != 2_560
            or int(getattr(config, "num_hidden_layers", -1)) != 36
            or int(getattr(config, "max_position_embeddings", -1)) != 40_960
            or not parameters
            or any(parameter.requires_grad for parameter in parameters)
            or any(parameter.dtype is not torch.bfloat16 for parameter in parameters)
            or any(str(parameter.device) != expected_device for parameter in parameters)
            or getattr(model, "training", None) is not False
        ):
            raise RunnerInvariantError("loaded Qwen architecture or frozen state differs")
        manifest = qualified_qwen_cycle_manifest()
        qwen_io = LocalQwenIO(
            model,
            tokenizer,
            embedding_batch_size=manifest.embedding_batch_size,
            max_input_tokens=manifest.max_input_tokens,
            max_new_tokens=manifest.max_new_tokens,
            enable_thinking=False,
            model_ref=manifest.model_ref,
            tokenizer_ref=manifest.tokenizer_ref,
        )
        manifest.assert_io(qwen_io)
        guard = FoundationGuard(model)
        return LoadedQwenFoundation(
            model=model,
            tokenizer=tokenizer,
            io=qwen_io,
            manifest=manifest,
            guard=guard,
            production_file_evidence=file_evidence,
        )
    except BaseException as primary:
        tokenizer = None
        model = None
        qwen_io = None
        _clear_exception_tracebacks(primary)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException:
            cleanup = CleanupFailure(
                "QWEN_PARTIAL_LOAD_CLEANUP_FAILED",
                "LIVE_CONSTRUCTION_CLEANUP",
            )
            raise BaseExceptionGroup(
                "Qwen load and partial cleanup failed",
                [primary, cleanup],
            ) from None
        raise primary from None


def seal_evaluation_seeds(
    path: str | Path,
    entropy: Callable[[int], bytes] = os.urandom,
    expected_commitments: Sequence[str] | None = None,
) -> SeedSeal:
    """Create or verify the exact two-seed artifact without exposing raw seeds."""

    target = Path(path)
    if not target.is_absolute():
        raise ValueError("seed path must be absolute")
    if target.exists() or target.is_symlink():
        return load_seed_seal(target, expected_commitments)
    if target == SEALED_SEED_PATH and (
        QUALIFICATION_RESULT_PATH.exists() or EVALUATION_RESULT_PATH.exists()
    ):
        raise RunnerInvariantError("seed commitments cannot first publish after a result")
    first = entropy(32)
    second = entropy(32)
    if (
        type(first) is not bytes
        or type(second) is not bytes
        or len(first) != 32
        or len(second) != 32
    ):
        raise RunnerInvariantError(
            "entropy boundary must return exactly 32 bytes on each of two calls"
        )
    seeds = (first, second)
    if seeds[0] == seeds[1]:
        raise RunnerInvariantError("entropy boundary returned duplicate replicate seeds")
    payload = _seed_artifact_payload(seeds)
    encoded = canonical_json_bytes(payload)
    if len(encoded) > MAXIMUM_SEED_ARTIFACT_BYTES:
        raise RunnerInvariantError("seed artifact exceeds its ceiling")
    commitments = tuple(row["commitment"] for row in payload["replicates"])
    _validate_fresh_r2_seed_commitments(commitments)
    if expected_commitments is not None and tuple(expected_commitments) != commitments:
        raise RunnerInvariantError("new seeds differ from expected commitments")
    _mkdir_private(target.parent)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    # Once O_EXCL creates this identity, every failure consumes it.  Preserving
    # a partial or file-synced target makes a retry fail closed instead of
    # drawing different entropy under the same experiment identity.
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(target.parent)
    return load_seed_seal(target, commitments)


def random_feedback_scalar(
    schedule: Sequence[Sequence[object]],
    task_id: str,
) -> float:
    """Read a precommitted schedule without accepting a truth/judgment input."""

    _digest(task_id, "task_id")
    selected: list[float] = []
    seen: set[str] = set()
    for row in schedule:
        if type(row) not in (tuple, list) or len(row) != 2:
            raise ValueError("random-feedback schedule row is malformed")
        row_task, scalar = row
        _digest(row_task, "random-feedback task_id")
        if row_task in seen or type(scalar) is not float or scalar not in (0.0, 1.0):
            raise ValueError("random-feedback schedule is not exact and unique")
        seen.add(row_task)
        if row_task == task_id:
            selected.append(scalar)
    if len(selected) != 1:
        raise ValueError("task_id has no unique random-feedback assignment")
    return selected[0]


def evidence_for_arm(
    arm: Arm,
    recalled_records: Sequence[str],
) -> tuple[str, ...]:
    """Return model-facing evidence while keeping QWEN_ONLY history-free."""

    if arm not in EVALUATION_ARMS:
        raise ValueError("arm is not one of the seven execution arms")
    if type(recalled_records) not in (tuple, list):
        raise TypeError("recalled_records must be an exact bounded sequence")
    records = tuple(recalled_records)
    if arm == "QWEN_ONLY":
        if records:
            raise RunnerInvariantError("QWEN_ONLY cannot receive persistent history")
        return (NO_PERSISTENT_RECALLED_EVIDENCE_V1,)
    if not records or len(records) > MAXIMUM_RECALL_ITEMS:
        raise ValueError("history arms require one bounded nonempty recall tuple")
    for value in records:
        _text(value, "recalled record", 16_384)
        if value == NO_PERSISTENT_RECALLED_EVIDENCE_V1:
            raise ValueError("no-history sentinel cannot masquerade as recalled evidence")
    return records


def validate_foundation_hashes(
    before: Mapping[str, str],
    after: Mapping[str, str],
    expected: Mapping[str, str],
) -> dict[str, str]:
    """Require exact model-file/tensor identities before and after execution."""

    rows = (dict(before), dict(after), dict(expected))
    if not rows[2] or set(rows[0]) != set(rows[1]) or set(rows[0]) != set(rows[2]):
        raise RunnerInvariantError("foundation identity key sets differ")
    for row in rows:
        for name, digest in row.items():
            _text(name, "foundation identity name", 512)
            _raw_digest(digest, "foundation sha256")
    if rows[0] != rows[1] or rows[0] != rows[2]:
        raise RunnerInvariantError("frozen foundation identity drifted")
    return dict(sorted(rows[0].items()))


@dataclass(frozen=True, slots=True)
class ControlSelectionReceipt:
    schema: str
    arm: Literal["QWEN_ONLY", "RETRIEVAL_ONLY"]
    policy: str
    selected_index: int
    proposal_generation_ref: str
    proposals: tuple[str, ...]
    selected_trace: str
    task_id: str
    nonauthorization: str

    def __post_init__(self) -> None:
        if self.schema != CONTROL_SELECTION_SCHEMA:
            raise ValueError("control selection schema differs")
        if self.arm not in CONTROL_ARMS:
            raise ValueError("control selection belongs only to control arms")
        if self.policy != LITERAL_FIRST_PROPOSAL or self.selected_index != 0:
            raise ValueError("control selection must be literal first proposal")
        _digest(self.proposal_generation_ref, "proposal_generation_ref")
        _digest(self.task_id, "task_id")
        if (
            type(self.proposals) is not tuple
            or len(self.proposals) != PROPOSAL_COUNT
            or len(set(self.proposals)) != PROPOSAL_COUNT
        ):
            raise ValueError("control receipt requires two distinct proposals")
        for proposal in self.proposals:
            _text(proposal, "proposal", 8_192)
        if self.selected_trace != self.proposals[0]:
            raise ValueError("control selected trace is not proposal zero")
        if self.nonauthorization != CONTROL_NONAUTHORIZATION:
            raise ValueError("control nonauthorization semantics differ")

    @classmethod
    def create(
        cls,
        *,
        arm: Literal["QWEN_ONLY", "RETRIEVAL_ONLY"],
        proposal_generation_ref: str,
        proposals: Sequence[str],
        task_id: str,
    ) -> "ControlSelectionReceipt":
        values = tuple(proposals)
        return cls(
            schema=CONTROL_SELECTION_SCHEMA,
            arm=arm,
            policy=LITERAL_FIRST_PROPOSAL,
            selected_index=0,
            proposal_generation_ref=proposal_generation_ref,
            proposals=values,
            selected_trace=values[0] if values else "",
            task_id=task_id,
            nonauthorization=CONTROL_NONAUTHORIZATION,
        )

    @property
    def receipt_ref(self) -> str:
        return "sha256:" + hashlib.sha256(
            _CONTROL_RECEIPT_DOMAIN + canonical_json_bytes(self.to_canonical())
        ).hexdigest()

    def to_canonical(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "nonauthorization": self.nonauthorization,
            "policy": self.policy,
            "proposal_generation_ref": self.proposal_generation_ref,
            "proposals": list(self.proposals),
            "schema": self.schema,
            "selected_index": self.selected_index,
            "selected_trace": self.selected_trace,
            "task_id": self.task_id,
        }


@dataclass(slots=True)
class AttemptBudget:
    proposal_ceiling: int
    execution_ceiling: int
    arm_task_ceiling: int
    _proposals: dict[tuple[str, str], str | None] = field(default_factory=dict, init=False)
    _executions: set[tuple[str, str]] = field(default_factory=set, init=False)
    _finished: set[tuple[str, str]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("proposal_ceiling", self.proposal_ceiling),
            ("execution_ceiling", self.execution_ceiling),
            ("arm_task_ceiling", self.arm_task_ceiling),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a non-negative exact integer")
        if self.proposal_ceiling < self.arm_task_ceiling:
            raise ValueError("proposal ceiling cannot be below the arm-task ceiling")
        if self.execution_ceiling > self.arm_task_ceiling:
            raise ValueError("execution ceiling cannot exceed arm tasks")

    @classmethod
    def for_mode(cls, mode: RunMode | str) -> "AttemptBudget":
        budget = expected_run_budget(mode)
        return cls(budget.arm_tasks, budget.execution_attempts, budget.arm_tasks)

    @staticmethod
    def _key(task_id: str, arm: str) -> tuple[str, str]:
        _digest(task_id, "task_id")
        if arm not in EVALUATION_ARMS:
            raise ValueError("arm is not declared")
        return task_id, arm

    def consume_proposal(
        self,
        task_id: str,
        arm: str,
        phase: Phase | None = None,
    ) -> None:
        key = self._key(task_id, arm)
        if phase is not None and phase not in PHASES:
            raise ValueError("phase is not declared")
        if key in self._proposals:
            raise RunnerInvariantError("proposal retry is forbidden")
        if len(self._proposals) >= self.proposal_ceiling:
            raise RunnerInvariantError("proposal generation ceiling exceeded")
        self._proposals[key] = phase

    def consume_execution(self, task_id: str, arm: str) -> None:
        key = self._key(task_id, arm)
        if key not in self._proposals:
            raise RunnerInvariantError("execution cannot precede proposal generation")
        if key in self._finished:
            raise RunnerInvariantError("a finished arm-task cannot execute")
        if key in self._executions:
            raise RunnerInvariantError("execution retry is forbidden")
        if len(self._executions) >= self.execution_ceiling:
            raise RunnerInvariantError("execution generation ceiling exceeded")
        self._executions.add(key)

    def finish_arm_task(self, task_id: str, arm: str) -> None:
        key = self._key(task_id, arm)
        if key not in self._proposals or key in self._finished:
            raise RunnerInvariantError("arm-task completion is missing or duplicated")
        if len(self._finished) >= self.arm_task_ceiling:
            raise RunnerInvariantError("arm-task ceiling exceeded")
        self._finished.add(key)

    def snapshot(self) -> dict[str, object]:
        return {
            "arm_tasks": len(self._finished),
            "execution_attempts": len(self._executions),
            "generation_attempts": len(self._proposals) + len(self._executions),
            "proposal_attempts": len(self._proposals),
        }

    def detailed_snapshot(self) -> dict[str, object]:
        """Return ceilings and phase totals without changing the frozen summary."""

        return {
            **self.snapshot(),
            "arm_task_totals": {
                arm: sum(1 for _, item_arm in self._finished if item_arm == arm)
                for arm in EVALUATION_ARMS
            },
            "arm_task_ceiling": self.arm_task_ceiling,
            "execution_ceiling": self.execution_ceiling,
            "phase_arm_tasks": {
                phase: sum(
                    1
                    for key in self._finished
                    if self._proposals.get(key) == phase
                )
                for phase in PHASES
            },
            "phase_execution_attempts": {
                phase: sum(
                    1
                    for key in self._executions
                    if self._proposals.get(key) == phase
                )
                for phase in PHASES
            },
            "phase_generation_attempts": {
                phase: sum(1 for value in self._proposals.values() if value == phase)
                + sum(
                    1
                    for key in self._executions
                    if self._proposals.get(key) == phase
                )
                for phase in PHASES
            },
            "phase_proposal_attempts": {
                phase: sum(1 for value in self._proposals.values() if value == phase)
                for phase in PHASES
            },
            "proposal_ceiling": self.proposal_ceiling,
        }


@dataclass(frozen=True, slots=True)
class ResourceCeilings:
    maximum_prompt_tokens: int = MAXIMUM_INPUT_TOKENS
    maximum_output_tokens: int = MAXIMUM_OUTPUT_TOKENS
    maximum_rss_bytes: int = 24 * 1024 * 1024 * 1024
    maximum_cuda_allocated_bytes: int = 12 * 1024 * 1024 * 1024
    maximum_cuda_reserved_bytes: int = 12 * 1024 * 1024 * 1024
    maximum_wall_seconds: float = 7_200.0
    maximum_configured_pool_sum: int = 8

    def __post_init__(self) -> None:
        integer_fields = (
            self.maximum_prompt_tokens,
            self.maximum_output_tokens,
            self.maximum_rss_bytes,
            self.maximum_cuda_allocated_bytes,
            self.maximum_cuda_reserved_bytes,
            self.maximum_configured_pool_sum,
        )
        if any(type(value) is not int or value < 1 for value in integer_fields):
            raise ValueError("resource ceilings must be positive exact integers")
        if (
            type(self.maximum_wall_seconds) not in (int, float)
            or not math.isfinite(float(self.maximum_wall_seconds))
            or float(self.maximum_wall_seconds) <= 0.0
        ):
            raise ValueError("wall-time ceiling must be positive and finite")


@dataclass(slots=True)
class ResourceLedger:
    ceilings: ResourceCeilings = field(default_factory=ResourceCeilings)
    generation_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    peak_rss_bytes: int = 0
    peak_cuda_allocated_bytes: int = 0
    peak_cuda_reserved_bytes: int = 0
    wall_seconds: float = 0.0
    configured_pool_sum: int = 0
    peak_state_scratch_bytes: int = 0
    memory_samples: int = 0
    wall_samples: int = 0
    configured_pool_samples: int = 0
    state_scratch_samples: int = 0

    def __post_init__(self) -> None:
        if type(self.ceilings) is not ResourceCeilings:
            raise TypeError("resource ledger ceilings must be exact ResourceCeilings")

    def record_generation(self, prompt_tokens: int, output_tokens: int) -> None:
        if (
            type(prompt_tokens) is not int
            or type(output_tokens) is not int
            or not 1 <= prompt_tokens <= self.ceilings.maximum_prompt_tokens
            or not 1 <= output_tokens <= self.ceilings.maximum_output_tokens
        ):
            raise RunnerInvariantError("generation token evidence exceeds its ceiling")
        self.generation_calls += 1
        self.prompt_tokens += prompt_tokens
        self.output_tokens += output_tokens

    def record_memory(
        self,
        rss_bytes: int,
        cuda_allocated_bytes: int = 0,
        cuda_reserved_bytes: int = 0,
    ) -> None:
        values = (rss_bytes, cuda_allocated_bytes, cuda_reserved_bytes)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("memory evidence must contain non-negative integers")
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        self.memory_samples += 1
        self.peak_cuda_allocated_bytes = max(
            self.peak_cuda_allocated_bytes, cuda_allocated_bytes
        )
        self.peak_cuda_reserved_bytes = max(
            self.peak_cuda_reserved_bytes, cuda_reserved_bytes
        )
        if (
            self.peak_rss_bytes > self.ceilings.maximum_rss_bytes
            or self.peak_cuda_allocated_bytes
            > self.ceilings.maximum_cuda_allocated_bytes
            or self.peak_cuda_reserved_bytes > self.ceilings.maximum_cuda_reserved_bytes
        ):
            raise RunnerInvariantError("memory resource ceiling exceeded")

    def record_wall(self, wall_seconds: float) -> None:
        if (
            type(wall_seconds) not in (int, float)
            or not math.isfinite(float(wall_seconds))
            or float(wall_seconds) < 0.0
        ):
            raise ValueError("wall time must be finite and non-negative")
        self.wall_seconds = max(self.wall_seconds, float(wall_seconds))
        self.wall_samples += 1
        if self.wall_seconds > self.ceilings.maximum_wall_seconds:
            raise RunnerInvariantError("wall-time ceiling exceeded")

    def record_configured_pools(self, pool_sum: int) -> None:
        if type(pool_sum) is not int or pool_sum < 0:
            raise ValueError("configured pool sum must be a non-negative integer")
        self.configured_pool_sum = max(self.configured_pool_sum, pool_sum)
        self.configured_pool_samples += 1
        if self.configured_pool_sum > self.ceilings.maximum_configured_pool_sum:
            raise RunnerInvariantError("configured compute-pool ceiling exceeded")

    def record_state_scratch(self, used_bytes: int) -> None:
        if type(used_bytes) is not int or used_bytes < 0:
            raise ValueError("state/scratch usage must be a non-negative integer")
        self.peak_state_scratch_bytes = max(
            self.peak_state_scratch_bytes,
            used_bytes,
        )
        self.state_scratch_samples += 1
        if self.peak_state_scratch_bytes > MAXIMUM_NEW_STATE_SCRATCH_BYTES:
            raise RunnerInvariantError("state/scratch resource ceiling exceeded")

    def snapshot(self) -> dict[str, object]:
        return {
            "configured_pool_sum": self.configured_pool_sum,
            "configured_pool_samples": self.configured_pool_samples,
            "generation_calls": self.generation_calls,
            "memory_samples": self.memory_samples,
            "output_tokens": self.output_tokens,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_state_scratch_bytes": self.peak_state_scratch_bytes,
            "prompt_tokens": self.prompt_tokens,
            "wall_seconds": self.wall_seconds,
            "state_scratch_samples": self.state_scratch_samples,
            "wall_samples": self.wall_samples,
        }


def _regular_tree_inventory(
    root: Path,
    *,
    hash_all_files: bool = False,
    hashed_relative_paths: frozenset[str] = frozenset(),
) -> dict[
    str,
    tuple[Literal["directory", "file"], int, int, int, int, str | None],
]:
    if type(hash_all_files) is not bool or type(hashed_relative_paths) is not frozenset:
        raise TypeError("resource inventory hashing boundary differs")
    if root.is_symlink() or not root.is_dir():
        raise RunnerInvariantError("resource root is not an exact directory")
    root_metadata = root.stat()
    if root_metadata.st_uid != os.getuid() or root_metadata.st_gid != os.getgid():
        raise RunnerInvariantError("resource root ownership differs")
    observed: dict[
        str,
        tuple[Literal["directory", "file"], int, int, int, int, str | None],
    ] = {
        ".": (
            "directory",
            int(root_metadata.st_dev),
            int(root_metadata.st_ino),
            stat.S_IMODE(root_metadata.st_mode),
            0,
            None,
        )
    }
    pending = [root]
    while pending:
        current = pending.pop()
        for child in current.iterdir():
            metadata = child.lstat()
            relative = child.relative_to(root).as_posix()
            if (
                metadata.st_uid != os.getuid()
                or metadata.st_gid != os.getgid()
            ):
                raise RunnerInvariantError("resource tree ownership differs")
            if stat.S_ISLNK(metadata.st_mode):
                raise RunnerInvariantError("resource tree contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                observed[relative] = (
                    "directory",
                    int(metadata.st_dev),
                    int(metadata.st_ino),
                    stat.S_IMODE(metadata.st_mode),
                    0,
                    None,
                )
                pending.append(child)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise RunnerInvariantError("resource tree contains a hardlink")
                observed[relative] = (
                    "file",
                    int(metadata.st_dev),
                    int(metadata.st_ino),
                    stat.S_IMODE(metadata.st_mode),
                    int(metadata.st_size),
                    (
                        sha256_file(child)[0]
                        if hash_all_files or relative in hashed_relative_paths
                        else None
                    ),
                )
            else:
                raise RunnerInvariantError("resource tree contains a special file")
    return dict(sorted(observed.items()))


def _regular_tree_bytes(root: Path) -> int:
    return sum(
        row[4]
        for row in _regular_tree_inventory(root).values()
        if row[0] == "file"
    )


def _validate_qualification_preconstruction_roots(
    state_root: Path,
    scratch_root: Path,
    *,
    release_required: bool = True,
) -> None:
    """Require only the sealed seed/release claim before qualification effects."""

    if type(release_required) is not bool:
        raise TypeError("qualification release requirement must be exact")
    for root in (state_root, scratch_root):
        if root.is_symlink() or not root.is_dir():
            raise RunnerInvariantError("qualification resource root differs")
        metadata = root.stat()
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RunnerInvariantError("qualification resource root metadata differs")
    expected_scratch = tuple(
        sorted(
            (
                scratch_root / "cuda-cache",
                scratch_root / "tmp",
            )
        )
    )
    if tuple(sorted(scratch_root.iterdir())) != expected_scratch:
        raise RunnerInvariantError("qualification scratch root shape differs")
    for path in expected_scratch:
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise RunnerInvariantError("qualification scratch directory is not fresh")
        metadata = path.stat()
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RunnerInvariantError(
                "qualification scratch directory metadata differs"
            )
    state_children = tuple(sorted(state_root.iterdir()))
    sealed = state_root / "sealed"
    if state_children != (sealed,) or sealed.is_symlink() or not sealed.is_dir():
        raise RunnerInvariantError("qualification state root contains stale state")
    sealed_metadata = sealed.stat()
    if (
        sealed_metadata.st_uid != os.getuid()
        or sealed_metadata.st_gid != os.getgid()
        or stat.S_IMODE(sealed_metadata.st_mode) != 0o700
    ):
        raise RunnerInvariantError("qualification sealed root metadata differs")
    expected_files = tuple(
        sorted(
            (
                sealed / SEALED_SEED_PATH.name,
                *((sealed / QUALIFICATION_RELEASE_CLAIM_PATH.name,) if release_required else ()),
            )
        )
    )
    if tuple(sorted(sealed.iterdir())) != expected_files:
        raise RunnerInvariantError("qualification sealed state fields differ")
    for path in expected_files:
        metadata = _validate_private_file_metadata(path)
        if metadata.st_nlink != 1:
            raise RunnerInvariantError("qualification sealed artifact is hardlinked")


def _validate_evaluation_preconstruction_roots(
    state_root: Path,
    scratch_root: Path,
    *,
    qualification_runtime_root: Path = QUALIFICATION_RUNTIME_ROOT,
    qualification_scope_root: Path = QUALIFICATION_COGNEE_SCOPES_ROOT,
    evaluation_runtime_root: Path = EVALUATION_RUNTIME_ROOT,
    evaluation_scope_root: Path = EVALUATION_COGNEE_SCOPES_ROOT,
    seed_path: Path = SEALED_SEED_PATH,
    qualification_release_path: Path = QUALIFICATION_RELEASE_CLAIM_PATH,
    admission_claim_path: Path = EVALUATION_ADMISSION_CLAIM_PATH,
    admission_required: bool,
) -> None:
    """Validate exact retained qualification state before evaluation effects."""

    if type(admission_required) is not bool:
        raise TypeError("evaluation admission requirement must be exact")
    paths = (
        state_root,
        scratch_root,
        qualification_runtime_root,
        qualification_scope_root,
        evaluation_runtime_root,
        evaluation_scope_root,
        seed_path,
        qualification_release_path,
        admission_claim_path,
    )
    if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
        raise ValueError("evaluation preconstruction paths must be absolute Paths")
    if len(set(paths)) != len(paths):
        raise RunnerInvariantError("evaluation preconstruction paths overlap")
    if any(
        path.parent != state_root
        for path in (
            qualification_runtime_root,
            qualification_scope_root,
            evaluation_runtime_root,
            evaluation_scope_root,
        )
    ):
        raise RunnerInvariantError("evaluation owned roots leave the state root")

    for root, label in ((state_root, "state"), (scratch_root, "scratch")):
        if root.is_symlink() or not root.is_dir():
            raise RunnerInvariantError(f"evaluation {label} root differs")
        metadata = root.stat()
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RunnerInvariantError(
                f"evaluation {label} root metadata differs"
            )

    allowed_scratch_names = frozenset(
        {
            "cuda-cache",
            "huggingface",
            "tmp",
            "torch",
            "torchinductor",
            "triton",
            "xdg-cache",
        }
    )
    scratch_children = tuple(sorted(scratch_root.iterdir()))
    observed_scratch_names = {path.name for path in scratch_children}
    if not {"cuda-cache", "tmp"}.issubset(observed_scratch_names) or not (
        observed_scratch_names <= allowed_scratch_names
    ):
        raise RunnerInvariantError("evaluation retained scratch shape differs")
    for child in scratch_children:
        if child.is_symlink() or not child.is_dir():
            raise RunnerInvariantError(
                "evaluation retained scratch contains a non-directory"
            )
        metadata = child.stat()
        if metadata.st_uid != os.getuid() or metadata.st_gid != os.getgid():
            raise RunnerInvariantError("evaluation retained scratch ownership differs")
    temporary = scratch_root / "tmp"
    cuda_cache = scratch_root / "cuda-cache"
    for path in (temporary, cuda_cache):
        if stat.S_IMODE(path.stat().st_mode) != 0o700:
            raise RunnerInvariantError(
                "evaluation externally owned scratch metadata differs"
            )
    if any(temporary.iterdir()):
        raise RunnerInvariantError("evaluation temporary directory is not empty")
    _regular_tree_inventory(scratch_root)

    sealed = state_root / "sealed"
    if sealed.is_symlink() or not sealed.is_dir():
        raise RunnerInvariantError("evaluation sealed root differs")
    sealed_metadata = sealed.stat()
    if (
        sealed_metadata.st_uid != os.getuid()
        or sealed_metadata.st_gid != os.getgid()
        or stat.S_IMODE(sealed_metadata.st_mode) != 0o700
    ):
        raise RunnerInvariantError("evaluation sealed root metadata differs")
    if any(
        path.parent != sealed
        for path in (seed_path, qualification_release_path, admission_claim_path)
    ):
        raise RunnerInvariantError("evaluation sealed artifact parent differs")
    expected_sealed = {
        seed_path,
        qualification_release_path,
        *( (admission_claim_path,) if admission_required else () ),
    }
    if set(sealed.iterdir()) != expected_sealed:
        raise RunnerInvariantError("evaluation sealed artifacts differ")
    for path in expected_sealed:
        metadata = _validate_private_file_metadata(path)
        if metadata.st_nlink != 1:
            raise RunnerInvariantError("evaluation sealed artifact is hardlinked")

    expected_state_children = {
        sealed,
        qualification_runtime_root,
        qualification_scope_root,
    }
    if set(state_root.iterdir()) != expected_state_children:
        raise RunnerInvariantError("evaluation retained state shape differs")
    for path in (qualification_runtime_root, qualification_scope_root):
        if path.is_symlink() or not path.is_dir():
            raise RunnerInvariantError("qualification-owned retained root differs")
        metadata = path.stat()
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RunnerInvariantError(
                "qualification-owned retained root metadata differs"
            )
        _regular_tree_inventory(path)
    for path in (evaluation_runtime_root, evaluation_scope_root):
        if path.exists() or path.is_symlink():
            raise RunnerInvariantError("evaluation runtime identity is consumed")


@dataclass(slots=True)
class _ProductionResourceSampler:
    """Observe production resource ceilings at every live arm boundary."""

    accounting: "RunAccounting"
    state_root: Path
    scratch_root: Path
    torch_module: object = field(repr=False)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    rss_reader: Callable[[], int] = field(
        default=lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        repr=False,
    )
    _baseline: dict[
        tuple[Literal["state", "scratch"], str],
        tuple[Literal["directory", "file"], int, int, int, int, str | None],
    ] = field(init=False, repr=False)
    _started: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.accounting) is not RunAccounting:
            raise TypeError("production sampler accounting differs")
        for path in (self.state_root, self.scratch_root):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError("production sampler roots must be absolute Paths")
        if not callable(self.clock) or not callable(self.rss_reader):
            raise TypeError("production sampler observation boundary differs")
        self._baseline = {
            (label, relative): row
            for label, root in (
                ("state", self.state_root),
                ("scratch", self.scratch_root),
            )
            for relative, row in _regular_tree_inventory(
                root,
                hash_all_files=True,
            ).items()
        }
        self.accounting.resources.record_configured_pools(8)
        cuda = getattr(self.torch_module, "cuda", None)
        reset = getattr(cuda, "reset_peak_memory_stats", None)
        if not callable(reset):
            raise TypeError("CUDA peak-memory reset boundary differs")
        reset(0)

    def start_after_model_load(self) -> None:
        if self._started is not None:
            raise RunnerInvariantError("resource sampling was already started")
        self._started = float(self.clock())

    def sample(self) -> None:
        if self._started is None:
            raise RunnerInvariantError("resource sampling has not started")
        cuda = getattr(self.torch_module, "cuda", None)
        allocated = getattr(cuda, "max_memory_allocated", None)
        reserved = getattr(cuda, "max_memory_reserved", None)
        if not callable(allocated) or not callable(reserved):
            raise TypeError("CUDA peak-memory observation boundary differs")
        rss = self.rss_reader()
        if type(rss) is not int or rss < 0:
            raise RunnerInvariantError("RSS observation differs")
        self.accounting.resources.record_memory(
            rss,
            int(allocated(0)),
            int(reserved(0)),
        )
        self.accounting.resources.record_wall(float(self.clock()) - self._started)
        current = {
            (label, relative): row
            for label, root in (
                ("state", self.state_root),
                ("scratch", self.scratch_root),
            )
            for relative, row in _regular_tree_inventory(
                root,
                hashed_relative_paths=frozenset(
                    relative
                    for (baseline_label, relative), row in self._baseline.items()
                    if baseline_label == label and row[0] == "file"
                ),
            ).items()
        }
        for key, expected in self._baseline.items():
            if current.get(key) != expected:
                raise RunnerInvariantError(
                    "pre-existing state/scratch entry changed during the run"
                )
        new_bytes = sum(
            row[4]
            for key, row in current.items()
            if key not in self._baseline and row[0] == "file"
        )
        self.accounting.resources.record_state_scratch(new_bytes)


@dataclass(slots=True)
class RunAccounting:
    """One shared attempt/resource boundary for a complete injected run."""

    mode: RunMode
    budget: AttemptBudget = field(init=False)
    resources: ResourceLedger = field(default_factory=ResourceLedger)
    _finalizer: Callable[[], Any] | None = field(default=None, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.mode) is not RunMode or self.mode not in (
            RunMode.QUALIFICATION,
            RunMode.EVALUATION,
        ):
            raise ValueError("run accounting mode must be qualification or evaluation")
        if type(self.resources) is not ResourceLedger or self.resources.ceilings != ResourceCeilings():
            raise ValueError("run accounting resource ceilings differ")
        expected = expected_run_budget(self.mode)
        self.budget = AttemptBudget(
            arm_task_ceiling=expected.arm_tasks,
            proposal_ceiling=expected.arm_tasks,
            execution_ceiling=expected.execution_attempts,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "attempts": self.budget.detailed_snapshot(),
            "resources": self.resources.snapshot(),
        }

    def bind_finalizer(self, finalizer: Callable[[], Any]) -> None:
        if self._finalizer is not None or self._finalized:
            raise RunnerInvariantError("run accounting finalizer is already bound")
        if not callable(finalizer):
            raise TypeError("run accounting finalizer must be callable")
        self._finalizer = finalizer

    def validate_complete(self) -> dict[str, object]:
        if not self._finalized:
            self._finalized = True
            if self._finalizer is not None:
                result = self._finalizer()
                if hasattr(result, "__await__"):
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
                    raise TypeError("run accounting finalizer must be synchronous")
        return validate_serialized_run_accounting(self.snapshot(), self.mode)


def validate_serialized_run_accounting(
    value: Mapping[str, object],
    mode: RunMode,
) -> dict[str, object]:
    """Strictly revalidate a completed canonical accounting snapshot."""

    if type(mode) is not RunMode or mode not in (
        RunMode.QUALIFICATION,
        RunMode.EVALUATION,
    ):
        raise ValueError("serialized accounting mode differs")
    if type(value) is not dict or set(value) != {"attempts", "resources"}:
        raise RunnerInvariantError("serialized accounting fields differ")
    attempts = value["attempts"]
    resources = value["resources"]
    attempt_fields = {
        "arm_task_ceiling",
        "arm_task_totals",
        "arm_tasks",
        "execution_attempts",
        "execution_ceiling",
        "generation_attempts",
        "phase_arm_tasks",
        "phase_execution_attempts",
        "phase_generation_attempts",
        "phase_proposal_attempts",
        "proposal_attempts",
        "proposal_ceiling",
    }
    resource_fields = {
        "configured_pool_samples",
        "configured_pool_sum",
        "generation_calls",
        "memory_samples",
        "output_tokens",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "peak_rss_bytes",
        "peak_state_scratch_bytes",
        "prompt_tokens",
        "state_scratch_samples",
        "wall_samples",
        "wall_seconds",
    }
    if (
        type(attempts) is not dict
        or set(attempts) != attempt_fields
        or type(resources) is not dict
        or set(resources) != resource_fields
    ):
        raise RunnerInvariantError("serialized accounting nested fields differ")
    expected = expected_run_budget(mode)
    expected_phases = (
        {"adaptation": 24, "development": 7, "final": 0}
        if mode is RunMode.QUALIFICATION
        else {"adaptation": 48, "development": 140, "final": 280}
    )
    expected_arms = (
        {
            "FULL": 13,
            "QWEN_ONLY": 1,
            "RETRIEVAL_ONLY": 1,
            "FROZEN_ORIGIN": 1,
            "PROSPECTIVE_REMOVAL": 1,
            "BACKEND_REMOVAL": 1,
            "RANDOM_FEEDBACK": 13,
        }
        if mode is RunMode.QUALIFICATION
        else {
            "FULL": 84,
            "QWEN_ONLY": 60,
            "RETRIEVAL_ONLY": 60,
            "FROZEN_ORIGIN": 60,
            "PROSPECTIVE_REMOVAL": 60,
            "BACKEND_REMOVAL": 60,
            "RANDOM_FEEDBACK": 84,
        }
    )
    nested_phase_fields = (
        "phase_arm_tasks",
        "phase_execution_attempts",
        "phase_generation_attempts",
        "phase_proposal_attempts",
    )
    if any(
        type(attempts[name]) is not dict
        or set(attempts[name]) != set(PHASES)
        or any(type(attempts[name][phase]) is not int for phase in PHASES)
        for name in nested_phase_fields
    ) or (
        type(attempts["arm_task_totals"]) is not dict
        or set(attempts["arm_task_totals"]) != set(EVALUATION_ARMS)
        or any(
            type(attempts["arm_task_totals"][arm]) is not int
            for arm in EVALUATION_ARMS
        )
    ):
        raise RunnerInvariantError("serialized accounting subtotal fields differ")
    integer_attempt_fields = (
        "arm_task_ceiling",
        "arm_tasks",
        "execution_attempts",
        "execution_ceiling",
        "generation_attempts",
        "proposal_attempts",
        "proposal_ceiling",
    )
    if any(
        type(attempts[name]) is not int or attempts[name] < 0
        for name in integer_attempt_fields
    ):
        raise RunnerInvariantError("serialized accounting attempt values differ")
    phase_executions = attempts["phase_execution_attempts"]
    phase_generations = attempts["phase_generation_attempts"]
    executions = attempts["execution_attempts"]
    generations = attempts["generation_attempts"]
    proposals = attempts["proposal_attempts"]
    if (
        attempts["arm_task_ceiling"] != expected.arm_tasks
        or attempts["proposal_ceiling"] != expected.arm_tasks
        or attempts["execution_ceiling"] != expected.execution_attempts
        or attempts["arm_tasks"] != expected.arm_tasks
        or proposals != expected.arm_tasks
        or not 0 <= executions <= expected.execution_attempts
        or generations != proposals + executions
        or attempts["phase_arm_tasks"] != expected_phases
        or attempts["phase_proposal_attempts"] != expected_phases
        or attempts["arm_task_totals"] != expected_arms
        or any(
            not 0 <= phase_executions[phase] <= expected_phases[phase]
            or phase_generations[phase]
            != expected_phases[phase] + phase_executions[phase]
            for phase in PHASES
        )
        or sum(phase_executions.values()) != executions
        or sum(phase_generations.values()) != generations
        or resources["generation_calls"] != generations
    ):
        raise RunnerInvariantError("serialized run attempt accounting differs")
    live_generation_ceiling = (
        FROZEN_RESOURCE_CEILINGS["qualification_generation_attempts"]
        if mode is RunMode.QUALIFICATION
        else FROZEN_RESOURCE_CEILINGS["evaluation_generation_attempts"]
    )
    integer_resources = resource_fields - {"wall_seconds"}
    if any(
        type(resources[name]) is not int or resources[name] < 0
        for name in integer_resources
    ) or (
        type(resources["wall_seconds"]) is not float
        or not math.isfinite(resources["wall_seconds"])
        or resources["wall_seconds"] < 0.0
        or generations > live_generation_ceiling
        or resources["configured_pool_sum"] > 8
        or resources["peak_cuda_allocated_bytes"] > 12 * 1024**3
        or resources["peak_cuda_reserved_bytes"] > 12 * 1024**3
        or resources["peak_rss_bytes"] > 24 * 1024**3
        or resources["peak_state_scratch_bytes"]
        > MAXIMUM_NEW_STATE_SCRATCH_BYTES
        or resources["wall_seconds"] > 7_200.0
        or resources["memory_samples"] < 1
        or resources["wall_samples"] < 1
        or resources["configured_pool_samples"] < 1
        or resources["state_scratch_samples"] < 1
        or not generations
        <= resources["prompt_tokens"]
        <= generations * MAXIMUM_INPUT_TOKENS
        or not generations
        <= resources["output_tokens"]
        <= generations * MAXIMUM_OUTPUT_TOKENS
    ):
        raise RunnerInvariantError("serialized run resource accounting differs")
    encoded = canonical_json_bytes(value)
    result = decode_canonical_json(encoded, maximum_bytes=MAXIMUM_RESULT_BYTES)
    if type(result) is not dict:
        raise RunnerInvariantError("serialized accounting canonical copy differs")
    return result


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _sqlite_connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


class EvidenceLedger:
    """Synchronous canonical attempt ledger with immutable finalized rows."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, run_intent_ref: str) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("evidence ledger path must be absolute")
        self.run_intent_ref = _digest(run_intent_ref, "run_intent_ref")
        _mkdir_private(self.path.parent)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise RunnerInvariantError("evidence ledger path is not a regular file")
        self._initialize()

    def _initialize(self) -> None:
        fresh = not self.path.exists()
        if not fresh:
            _validate_private_file_metadata(self.path)
            connection = _sqlite_connect_readonly(self.path)
            try:
                row = connection.execute(
                    "SELECT schema, run_intent_ref FROM ledger_identity WHERE singleton=1"
                ).fetchone()
                version = connection.execute("PRAGMA user_version").fetchone()
            except sqlite3.Error as error:
                raise RunnerInvariantError("existing evidence ledger schema differs") from error
            finally:
                connection.close()
            if row != (EVIDENCE_SCHEMA, self.run_intent_ref) or version != (
                self.SCHEMA_VERSION,
            ):
                raise RunnerInvariantError("evidence ledger identity differs")
            self.audit()
            return
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError as error:
            raise RunnerInvariantError("evidence ledger creation raced") from error
        os.close(descriptor)
        connection = _sqlite_connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_identity (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema TEXT NOT NULL,
                    run_intent_ref TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    task_id TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    attempt_receipt_ref TEXT NOT NULL UNIQUE,
                    stage_bytes BLOB NOT NULL,
                    stage_ref TEXT NOT NULL UNIQUE,
                    judgment_bytes BLOB,
                    judgment_ref TEXT UNIQUE,
                    PRIMARY KEY (task_id, arm)
                ) WITHOUT ROWID
                """
            )
            expected = (EVIDENCE_SCHEMA, self.run_intent_ref)
            connection.execute(
                "INSERT INTO ledger_identity VALUES (1, ?, ?)", expected
            )
            connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            self.path.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
        _fsync_directory(self.path.parent)
        _validate_private_file_metadata(self.path)
        self.audit()

    @staticmethod
    def _attempt_key(record: Mapping[str, object]) -> tuple[str, str, str]:
        if type(record) is not dict:
            record = dict(record)
        required = {"task_id", "arm", "attempt_receipt_ref"}
        if not required.issubset(record):
            raise ValueError("attempt record lacks its identity fields")
        task_id = _digest(record["task_id"], "task_id")
        arm = record["arm"]
        if arm not in (*EVALUATION_ARMS, "QUALIFICATION"):
            raise ValueError("attempt arm is not declared")
        receipt = _digest(record["attempt_receipt_ref"], "attempt_receipt_ref")
        return task_id, arm, receipt

    @staticmethod
    def _validate_stage(record: dict[str, object]) -> None:
        fields = {
            "arm",
            "attempt_receipt",
            "attempt_receipt_ref",
            "feedback_source",
            "foundation_tensor_digest",
            "frozen_recall_ref",
            "learner_parent_digest",
            "judge_arm",
            "parser_disposition",
            "phase",
            "proposal_generation",
            "proposal_request",
            "proposals",
            "public_task_ref",
            "purpose",
            "raw_response",
            "resource_counters",
            "schema",
            "selection",
            "execution_request",
            "execution_receipt",
            "task_id",
            "task_response_generation",
        }
        if set(record) != fields or record["schema"] != ATTEMPT_STAGE_SCHEMA:
            raise ValueError("attempt stage fields or schema differ")
        EvidenceLedger._attempt_key(record)
        if record["phase"] not in PHASES:
            raise ValueError("attempt stage phase differs")
        if record["purpose"] not in ("qualification", "evaluation"):
            raise ValueError("attempt stage purpose differs")
        judge_arm = record["judge_arm"]
        if record["purpose"] == "evaluation":
            if judge_arm != record["arm"]:
                raise ValueError("evaluation stage judge arm differs")
        elif judge_arm not in (None, "QUALIFICATION") or (
            judge_arm == "QUALIFICATION" and record["arm"] != "FULL"
        ):
            raise ValueError("qualification stage judge arm differs")
        _digest(record["public_task_ref"], "public_task_ref")
        for label in (
            "proposal_generation",
            "proposal_request",
            "feedback_source",
            "resource_counters",
        ):
            if type(record[label]) is not dict or not record[label]:
                raise ValueError(f"attempt stage {label} must be a nonempty object")
        proposal_generation = record["proposal_generation"]
        generation_ref = proposal_generation.get("generation_ref")  # type: ignore[union-attr]
        _digest(generation_ref, "proposal generation_ref")
        parser = record["parser_disposition"]
        if parser not in ("ADMITTED", "MALFORMED"):
            raise ValueError("parser disposition differs")
        proposals = record["proposals"]
        if type(proposals) is not list:
            raise TypeError("proposals must be an exact list")
        raw_response = record["raw_response"]
        if type(raw_response) is not str:
            raise TypeError("raw_response must be text")
        for label in (
            "foundation_tensor_digest",
            "frozen_recall_ref",
            "learner_parent_digest",
        ):
            value = record[label]
            if value is not None:
                _digest(value, label)
        if parser == "MALFORMED":
            if proposals or record["selection"] is not None or any(
                record[label] is not None
                for label in (
                    "execution_request",
                    "execution_receipt",
                    "task_response_generation",
                )
            ) or raw_response != "":
                raise ValueError("malformed proposal stage contains downstream evidence")
            source_receipt_ref = generation_ref
            source_kind = "PROPOSAL_GENERATION"
        else:
            if (
                len(proposals) != PROPOSAL_COUNT
                or len(set(proposals)) != PROPOSAL_COUNT
                or any(type(item) is not str or not item for item in proposals)
            ):
                raise ValueError("admitted proposal evidence differs")
            for label in (
                "selection",
                "execution_request",
                "execution_receipt",
                "task_response_generation",
            ):
                if type(record[label]) is not dict or not record[label]:
                    raise ValueError(f"admitted attempt lacks {label}")
            receipt_ref = record["execution_receipt"].get(  # type: ignore[union-attr]
                "execution_receipt_ref"
            )
            _digest(receipt_ref, "execution_receipt_ref")
            source_receipt_ref = receipt_ref
            source_kind = "EXECUTION_RECEIPT"
        attempt_receipt = record["attempt_receipt"]
        if type(attempt_receipt) is not dict:
            raise TypeError("attempt receipt payload must be an exact object")
        expected_attempt_receipt = attempt_receipt_payload(
            purpose=record["purpose"],  # type: ignore[arg-type]
            phase=record["phase"],  # type: ignore[arg-type]
            task_id=record["task_id"],  # type: ignore[arg-type]
            arm=record["arm"],  # type: ignore[arg-type]
            parser_disposition=parser,
            source_receipt_ref=source_receipt_ref,
            source_kind=source_kind,  # type: ignore[arg-type]
        )
        if (
            attempt_receipt != expected_attempt_receipt
            or record["attempt_receipt_ref"]
            != attempt_receipt_reference(attempt_receipt)
        ):
            raise RunnerInvariantError("attempt receipt payload or reference differs")

    @staticmethod
    def _validate_finalization(
        record: dict[str, object],
        stage: dict[str, object],
    ) -> None:
        fields = {
            "arm",
            "attempt_receipt_ref",
            "feedback_record",
            "learner_transition",
            "objective_judgment",
            "probe_integrity",
            "probe_integrity_ref",
            "resource_counters",
            "runtime_quiescence",
            "runtime_quiescence_ref",
            "schema",
            "task_id",
            "unevaluated_resolution",
        }
        if set(record) != fields or record["schema"] != ATTEMPT_FINAL_SCHEMA:
            raise ValueError("attempt finalization fields or schema differ")
        if EvidenceLedger._attempt_key(record) != EvidenceLedger._attempt_key(stage):
            raise RunnerInvariantError("finalization identity differs from stage")
        judgment = record["objective_judgment"]
        random_diagnostic = (
            judgment is None
            and stage["purpose"] == "qualification"
            and stage["judge_arm"] is None
            and stage["arm"] == "RANDOM_FEEDBACK"
            and stage["phase"] == "adaptation"
            and stage["feedback_source"].get("kind") == "RANDOM_FEEDBACK_SCHEDULE"
        )
        unevaluated_diagnostic = (
            judgment is None
            and stage["purpose"] == "qualification"
            and stage["judge_arm"] is None
            and not random_diagnostic
        )
        if not (random_diagnostic or unevaluated_diagnostic):
            if type(judgment) is not dict or set(judgment) != {
                "arm",
                "attempt_receipt_ref",
                "disposition",
                "raw_response",
                "response_commitment",
                "score",
                "task_id",
            }:
                raise ValueError("objective judgment evidence fields differ")
            if (
                judgment["task_id"],
                judgment["arm"],
                judgment["attempt_receipt_ref"],
                judgment["raw_response"],
            ) != (
                stage["task_id"],
                stage["judge_arm"],
                stage["attempt_receipt_ref"],
                stage["raw_response"],
            ):
                raise RunnerInvariantError("objective judgment does not bind stage")
            expected_response = evaluator_record_digest(
                "response",
                {
                    "arm": judgment["arm"],
                    "attempt_receipt_ref": judgment["attempt_receipt_ref"],
                    "raw_response": judgment["raw_response"],
                    "task_id": judgment["task_id"],
                },
            )
            score = judgment["score"]
            if (
                judgment["response_commitment"] != expected_response
                or type(score) is not float
                or score not in (0.0, 1.0)
                or judgment["disposition"]
                != ("SUCCESS" if score == 1.0 else "UNSUCCESSFUL")
            ):
                raise ValueError("objective judgment scalar/disposition differs")
        if type(record["resource_counters"]) is not dict:
            raise TypeError("final resource counters must be an object")
        quiescence = record["runtime_quiescence"]
        quiescence_ref = record["runtime_quiescence_ref"]
        if stage["arm"] == "QWEN_ONLY":
            _validate_qwen_journal_quiescence(quiescence, quiescence_ref)
        elif quiescence is not None or quiescence_ref is not None:
            raise RunnerInvariantError("non-QWEN arm claimed auxiliary quiescence")
        integrity = record["probe_integrity"]
        if (
            type(integrity) is not dict
            or record["probe_integrity_ref"]
            != content_ref("probe-integrity", integrity)
            or (
                integrity.get("task_id"),
                integrity.get("arm"),
                integrity.get("purpose"),
                integrity.get("phase"),
            )
            != (
                stage["task_id"],
                stage["arm"],
                stage["purpose"],
                stage["phase"],
            )
        ):
            raise RunnerInvariantError("probe integrity does not bind finalization")
        malformed = stage["parser_disposition"] == "MALFORMED"
        control = stage["arm"] in CONTROL_ARMS
        feedback = record["feedback_record"]
        transition = record["learner_transition"]
        resolution = record["unevaluated_resolution"]
        if malformed or control or unevaluated_diagnostic:
            if feedback is not None or transition is not None:
                raise ValueError("control/malformed attempt cannot launder a learner update")
        elif type(feedback) is not dict or type(transition) is not dict:
            raise ValueError("stateful admitted attempt requires feedback and transition")
        if random_diagnostic and not malformed and (not feedback or not transition):
            raise ValueError("random-feedback diagnostic lacks its learner transition")
        if unevaluated_diagnostic and not malformed and not control:
            if (
                type(resolution) is not dict
                or set(resolution)
                != {"acquisition_ref", "projection_pending", "resolution_ref"}
                or type(resolution["projection_pending"]) is not bool
            ):
                raise RunnerInvariantError("unevaluated diagnostic resolution differs")
            _digest(resolution["acquisition_ref"], "unevaluated acquisition ref")
            _digest(resolution["resolution_ref"], "unevaluated resolution ref")
        elif resolution is not None:
            raise RunnerInvariantError("attempt claimed an inapplicable unevaluated resolution")

    def stage_attempt(self, record: Mapping[str, object]) -> str:
        _validate_private_file_metadata(self.path)
        canonical = dict(record)
        self._validate_stage(canonical)
        task_id, arm, receipt = self._attempt_key(canonical)
        encoded = canonical_json_bytes(canonical)
        if len(encoded) > MAXIMUM_EVIDENCE_RECORD_BYTES:
            raise RunnerInvariantError("attempt evidence exceeds its record ceiling")
        stage_ref = content_ref("attempt-stage", canonical)
        connection = _sqlite_connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT stage_bytes, stage_ref FROM attempts WHERE task_id=? AND arm=?",
                (task_id, arm),
            ).fetchone()
            if existing is not None:
                if existing != (encoded, stage_ref):
                    raise RunnerInvariantError("attempt task/arm identity conflict")
                connection.commit()
                return stage_ref
            connection.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                (task_id, arm, receipt, encoded, stage_ref),
            )
            connection.commit()
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise RunnerInvariantError("attempt receipt is already consumed") from error
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return stage_ref

    def finalize_attempt(
        self,
        key: tuple[str, str, str],
        judgment: Mapping[str, object],
    ) -> str:
        _validate_private_file_metadata(self.path)
        if type(key) is not tuple or len(key) != 3:
            raise TypeError("attempt key must be (task_id, arm, attempt_receipt_ref)")
        task_id, arm, receipt = self._attempt_key(
            {"task_id": key[0], "arm": key[1], "attempt_receipt_ref": key[2]}
        )
        connection = _sqlite_connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempt_receipt_ref, stage_bytes, judgment_bytes, judgment_ref "
                "FROM attempts WHERE task_id=? AND arm=?",
                (task_id, arm),
            ).fetchone()
            if row is None or row[0] != receipt:
                raise RunnerInvariantError("judgment has no exact staged attempt")
            stage = decode_canonical_json(
                row[1], maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES
            )
            if type(stage) is not dict:
                raise RunnerInvariantError("stored attempt stage is malformed")
            canonical = dict(judgment)
            self._validate_finalization(canonical, stage)
            encoded = canonical_json_bytes(canonical)
            if len(encoded) > MAXIMUM_EVIDENCE_RECORD_BYTES:
                raise RunnerInvariantError("judgment evidence exceeds its record ceiling")
            judgment_ref = content_ref("attempt-judgment", canonical)
            if row[2] is not None:
                if row[2:] != (encoded, judgment_ref):
                    raise RunnerInvariantError("finalized judgment is immutable")
                connection.commit()
                return judgment_ref
            connection.execute(
                "UPDATE attempts SET judgment_bytes=?, judgment_ref=? "
                "WHERE task_id=? AND arm=? AND judgment_bytes IS NULL",
                (encoded, judgment_ref, task_id, arm),
            )
            connection.commit()
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise RunnerInvariantError("judgment identity is already consumed") from error
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return judgment_ref

    def read_all(self) -> tuple[dict[str, object], ...]:
        _validate_private_file_metadata(self.path)
        connection = _sqlite_connect_readonly(self.path)
        try:
            rows = connection.execute(
                "SELECT stage_bytes, judgment_bytes FROM attempts "
                "ORDER BY task_id, arm"
            ).fetchall()
        finally:
            connection.close()
        result: list[dict[str, object]] = []
        for stage_bytes, judgment_bytes in rows:
            stage = decode_canonical_json(stage_bytes, maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES)
            if type(stage) is not dict:
                raise RunnerInvariantError("stored attempt is not an object")
            result.append(
                {
                    "attempt": stage,
                    "judgment": (
                        None
                        if judgment_bytes is None
                        else decode_canonical_json(
                            judgment_bytes,
                            maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
                        )
                    ),
                }
            )
        return tuple(result)

    def audit(self) -> None:
        _validate_private_file_metadata(self.path)
        connection = _sqlite_connect_readonly(self.path)
        try:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise RunnerInvariantError("evidence ledger SQLite integrity failed")
            identity = connection.execute(
                "SELECT schema, run_intent_ref FROM ledger_identity WHERE singleton=1"
            ).fetchone()
            if identity != (EVIDENCE_SCHEMA, self.run_intent_ref) or connection.execute(
                "PRAGMA user_version"
            ).fetchone() != (self.SCHEMA_VERSION,):
                raise RunnerInvariantError("evidence ledger identity differs")
            rows = connection.execute(
                "SELECT task_id, arm, attempt_receipt_ref, stage_bytes, stage_ref, "
                "judgment_bytes, judgment_ref FROM attempts ORDER BY task_id, arm"
            ).fetchall()
        finally:
            connection.close()
        seen_receipts: set[str] = set()
        for task_id, arm, receipt, stage_bytes, stage_ref, judgment_bytes, judgment_ref in rows:
            stage = decode_canonical_json(stage_bytes, maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES)
            if type(stage) is not dict:
                raise RunnerInvariantError("evidence stage is not an object")
            self._validate_stage(stage)
            if self._attempt_key(stage) != (task_id, arm, receipt):
                raise RunnerInvariantError("evidence stage index differs")
            if stage_ref != content_ref("attempt-stage", stage) or receipt in seen_receipts:
                raise RunnerInvariantError("evidence stage identity differs")
            seen_receipts.add(receipt)
            if judgment_bytes is None:
                if judgment_ref is not None:
                    raise RunnerInvariantError("empty judgment has a reference")
                continue
            judgment = decode_canonical_json(
                judgment_bytes, maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES
            )
            if type(judgment) is not dict or (
                judgment.get("task_id"),
                judgment.get("arm"),
                judgment.get("attempt_receipt_ref"),
            ) != (task_id, arm, receipt):
                raise RunnerInvariantError("evidence judgment index differs")
            self._validate_finalization(judgment, stage)
            if judgment_ref != content_ref("attempt-judgment", judgment):
                raise RunnerInvariantError("evidence judgment reference differs")


_RESULT_CLASSIFICATIONS = frozenset(
    {
        "SUCCESS",
        "QUALIFICATION_PASS",
        "QUALIFICATION_FAILURE",
        "EXPERIMENTALLY_SUPPORTED",
        "NOT_SUPPORTED",
        "INVALID",
        "INCONCLUSIVE",
    }
)


def _publish_canonical_create_once(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    maximum_bytes: int,
) -> Path:
    """Publish canonical 0600 bytes once, preserving a linked consumed target."""

    if not isinstance(payload, Mapping) or type(maximum_bytes) is not int or maximum_bytes < 1:
        raise TypeError("canonical publisher inputs differ")
    target = Path(path)
    if not target.is_absolute() or target.parent == target:
        raise ValueError("canonical publisher target must be an exact absolute path")
    encoded = canonical_json_bytes(dict(payload))
    if not 1 <= len(encoded) <= maximum_bytes:
        raise RunnerInvariantError("canonical publication exceeds its byte ceiling")
    _require_owned_real_parent(target.parent)
    temporary = target.with_name(target.name + ".tmp")
    if (
        target.exists()
        or target.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise RunnerInvariantError("publication identity is already consumed")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    linked = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
            linked = True
        except FileExistsError as error:
            raise RunnerInvariantError(
                "publication identity is already consumed"
            ) from error
        temporary.unlink()
        _fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        # Once the hard link exists it is the consumed identity even if the
        # parent-directory durability acknowledgment fails.
        if linked and not target.exists():
            raise RunnerInvariantError("linked publication target disappeared")
        raise
    return target


def atomic_result(
    path: str | Path,
    payload: Mapping[str, object],
    classification: str,
    max_bytes: int = MAXIMUM_RESULT_BYTES,
) -> Path:
    """Create one result atomically; never overwrite or silently rerun."""

    if classification not in _RESULT_CLASSIFICATIONS:
        raise ValueError("result classification is not declared")
    if classification in (
        "QUALIFICATION_PASS",
        "EXPERIMENTALLY_SUPPORTED",
        "NOT_SUPPORTED",
    ):
        raise RunnerInvariantError(
            "successful results require their strict typed publisher"
        )
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("result byte ceiling must be positive")
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("result path must be absolute")
    result = dict(payload)
    if "classification" in result and result["classification"] != classification:
        raise ValueError("payload classification differs")
    result["classification"] = classification
    return _publish_canonical_create_once(
        target,
        result,
        maximum_bytes=max_bytes,
    )


def _validate_completed_source_map(
    source_hashes: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(source_hashes, Mapping):
        raise TypeError("completed source hashes must be a mapping")
    required_paths = _source_inventory_paths()
    sources = dict(sorted(source_hashes.items()))
    for name in sources:
        _canonical_source_path(name)
    required = set(required_paths)
    if set(sources) != required:
        missing = required - set(sources)
        extra = set(sources) - required
        reason = "missing" if missing else "extra"
        raise RunnerInvariantError(
            f"completed source map has {reason} source identities"
        )
    for name, digest in sources.items():
        _raw_digest(digest, "source sha256")
    for name, expected in FROZEN_SOURCE_SHA256.items():
        if sources.get(name) != expected:
            raise RunnerInvariantError(f"frozen predecessor source drift: {name}")
    return sources


def completed_source_manifest(
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    """Hash exactly the completed frozen and construction source inventory."""

    root = Path(repository_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("completed source repository root must be exact and absolute")
    required = _source_inventory_paths()
    identities: set[tuple[int, int]] = set()
    observed: dict[str, str] = {}
    for relative in sorted(required):
        digest, _, identity = _hash_source_file(root / relative)
        if identity in identities:
            raise RunnerInvariantError("completed sources share a filesystem identity")
        identities.add(identity)
        observed[relative] = digest
    return _validate_completed_source_map(observed)


def _frozen_manifest_budgets() -> dict[str, object]:
    return {
        "evaluation": {"arm_tasks": 468, "generation_attempts": 936},
        "qualification": {"arm_tasks": 31, "generation_attempts": 62},
        "max_input_tokens": MAXIMUM_INPUT_TOKENS,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKENS,
        "proposal_count": PROPOSAL_COUNT,
    }


def _frozen_manifest_thresholds() -> dict[str, object]:
    return {
        "full_final_minimum": 0.35,
        "full_minus_frozen_origin": 0.025,
        "full_minus_prospective_removal": 0.05,
        "full_minus_qwen_only": 0.075,
        "full_minus_random_feedback": 0.05,
        "full_minus_retrieval_only": 0.075,
        "maximum_family_deficit": 0.125,
        "strict_family_wins_minimum": 2,
    }


def _frozen_manifest_environment() -> dict[str, object]:
    return {
        "cognee_worker": {
            "cuda_visible_devices": "",
            "execution_providers": ["CPUExecutionProvider"],
            "fastembed_onnx_inter_op_threads": 2,
            "fastembed_onnx_intra_op_threads": 2,
            "inheritance": "env -i",
            "network_namespace": "unshare --net loopback-only",
            "scope_state_parents": {
                "evaluation": str(EVALUATION_COGNEE_SCOPES_ROOT),
                "qualification": str(QUALIFICATION_COGNEE_SCOPES_ROOT),
            },
            "source_path": "src/angler/memory/cognee_worker_protocol.py",
            "source_sha256": FROZEN_SOURCE_SHA256[
                "src/angler/memory/cognee_worker_protocol.py"
            ],
            "telemetry": "disabled",
            "worker_cwd": "private-empty-0700-child-of-each-scope",
            "worker_tmpdir": "private-empty-0700-child-of-each-scope",
        },
        "evaluator_worker": {
            "cwd_parent": str(SCRATCH_ROOT / "evaluator-cwd"),
            "directories": "fresh-empty-0700",
            "fixed_variables": dict(FROZEN_EVALUATOR_WORKER_ENVIRONMENT),
            "inheritance": "env -i",
            "network": "inherits-offline-loopback-only-parent",
            "tmpdir_parent": str(SCRATCH_ROOT / "evaluator-tmp"),
        },
        "parent": {
            "inheritance": "env -i",
            "network": "offline-loopback-only",
            "variables": dict(FROZEN_PARENT_ENVIRONMENT),
        },
    }


def _validate_manifest_commitments(
    evaluator_commitments: object,
    seed_commitments: object,
) -> tuple[str, str]:
    if type(seed_commitments) is not list or len(seed_commitments) != 2:
        raise RunnerInvariantError("manifest requires two seed commitments")
    seeds = tuple(_digest(value, "seed commitment") for value in seed_commitments)
    if len(set(seeds)) != 2:
        raise RunnerInvariantError("manifest seed commitments must be distinct")
    if type(evaluator_commitments) is not dict:
        raise RunnerInvariantError("manifest evaluator commitments are malformed")
    required = {
        "identity",
        "purpose",
        "qualification_seed",
        "random_feedback",
        "replicate_commitments",
        "schema",
        "tasks",
    }
    if set(evaluator_commitments) != required or (
        evaluator_commitments["identity"] != EVALUATOR_EVALUATION_IDENTITY
        or evaluator_commitments["purpose"] != "evaluation"
        or evaluator_commitments["qualification_seed"] is not None
        or evaluator_commitments["replicate_commitments"] != list(seeds)
        or evaluator_commitments["schema"] != SUITE_SCHEMA
        or type(evaluator_commitments["random_feedback"]) is not list
        or type(evaluator_commitments["tasks"]) is not list
    ):
        raise RunnerInvariantError("manifest evaluator commitment identity differs")
    tasks = evaluator_commitments["tasks"]
    task_fields = {
        "family",
        "ordinal",
        "phase",
        "private_commitment",
        "public_commitment",
        "replicate_commitment",
        "task_id",
    }
    counts = {
        "symbolic-demonstration-transfer": {
            "adaptation": 4,
            "development": 2,
            "final": 4,
        },
        "glyph-machine": {
            "adaptation": 2,
            "development": 2,
            "final": 4,
        },
        "causal-operator": {
            "adaptation": 6,
            "development": 6,
            "final": 12,
        },
    }
    expected_order: list[tuple[str, str, str, int]] = []
    for replicate in seeds:
        for phase in PHASES:
            maximum = max(counts[family][phase] for family in FAMILIES)
            for ordinal in range(maximum):
                expected_order.extend(
                    (replicate, family, phase, ordinal)
                    for family in FAMILIES
                    if ordinal < counts[family][phase]
                )
    observed_order: list[tuple[str, str, str, int]] = []
    task_ids: set[str] = set()
    public_refs: set[str] = set()
    private_refs: set[str] = set()
    for task in tasks:
        if type(task) is not dict or set(task) != task_fields:
            raise RunnerInvariantError("manifest evaluator task fields differ")
        family = task["family"]
        phase = task["phase"]
        ordinal = task["ordinal"]
        replicate = task["replicate_commitment"]
        if (
            family not in FAMILIES
            or phase not in PHASES
            or type(ordinal) is not int
            or ordinal < 0
            or replicate not in seeds
        ):
            raise RunnerInvariantError("manifest evaluator task identity differs")
        task_id = _digest(task["task_id"], "manifest task_id")
        public_ref = _digest(
            task["public_commitment"],
            "manifest public commitment",
        )
        private_ref = _digest(
            task["private_commitment"],
            "manifest private commitment",
        )
        if (
            task_id in task_ids
            or public_ref in public_refs
            or private_ref in private_refs
        ):
            raise RunnerInvariantError("manifest evaluator task identity was reused")
        task_ids.add(task_id)
        public_refs.add(public_ref)
        private_refs.add(private_ref)
        observed_order.append((replicate, family, phase, ordinal))
    if observed_order != expected_order:
        raise RunnerInvariantError("manifest evaluator task coverage/order differs")
    random_rows = evaluator_commitments["random_feedback"]
    if type(random_rows) is not list or len(random_rows) != 2:
        raise RunnerInvariantError("manifest random-feedback cardinality differs")
    for index, row in enumerate(random_rows):
        if type(row) is not dict or set(row) != {
            "replicate_commitment",
            "schedule_commitment",
        } or row["replicate_commitment"] != seeds[index]:
            raise RunnerInvariantError("manifest random-feedback identity differs")
        _digest(row["schedule_commitment"], "manifest random-feedback commitment")
    scan_for_hidden_material(evaluator_commitments)
    return seeds  # type: ignore[return-value]


def build_source_manifest(
    *,
    evaluator_commitments: Mapping[str, object],
    seed_commitments: Sequence[str],
    accepted_leaf_sha256: str,
    completed_source_hashes: Mapping[str, str],
    qualification_result_sha256: str | None = None,
) -> dict[str, object]:
    commitments = tuple(seed_commitments)
    if len(commitments) != 2 or len(set(commitments)) != 2:
        raise ValueError("manifest requires two distinct evaluation seed commitments")
    for value in commitments:
        _digest(value, "seed commitment")
    if (
        _raw_digest(accepted_leaf_sha256, "accepted leaf sha256")
        != EXPECTED_ACTIVE_LEAF_SHA256
    ):
        raise RunnerInvariantError("accepted R2 science leaf identity differs")
    if qualification_result_sha256 is not None:
        raise RunnerInvariantError(
            "qualification result identity cannot mutate the pre-result manifest"
        )
    sources = _validate_completed_source_map(completed_source_hashes)
    _validate_manifest_commitments(
        dict(evaluator_commitments),
        list(commitments),
    )
    evaluator_commitment_digest = evaluator_record_digest(
        "evaluator-commitments",
        evaluator_commitments,
    )
    return {
        "arms": list(EVALUATION_ARMS),
        "budgets": _frozen_manifest_budgets(),
        "environment": _frozen_manifest_environment(),
        "evaluator_commitments": dict(evaluator_commitments),
        "evaluator_commitment_digest": evaluator_commitment_digest,
        "evaluation_identity": EVALUATION_IDENTITY,
        "frozen_sources": sources,
        "gpu_assignment": dict(FROZEN_GPU_ASSIGNMENT),
        "host": "angler-workstation",
        "leaf_sha256": accepted_leaf_sha256,
        "paths": dict(FROZEN_PATHS),
        "qualification_identity": QUALIFICATION_IDENTITY,
        "qualification_result_sha256": qualification_result_sha256,
        "schema": MANIFEST_SCHEMA,
        "seed_commitments": list(commitments),
        "resource_ceilings": dict(FROZEN_RESOURCE_CEILINGS),
        "thresholds": _frozen_manifest_thresholds(),
        "write_scope": list(FROZEN_WRITE_SCOPE),
    }


def validate_source_manifest(
    manifest: Mapping[str, object],
    *,
    expected_hash: str | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> str:
    canonical = dict(manifest)
    required = {
        "arms",
        "budgets",
        "environment",
        "evaluator_commitments",
        "evaluator_commitment_digest",
        "evaluation_identity",
        "frozen_sources",
        "gpu_assignment",
        "host",
        "leaf_sha256",
        "paths",
        "qualification_identity",
        "qualification_result_sha256",
        "schema",
        "seed_commitments",
        "resource_ceilings",
        "thresholds",
        "write_scope",
    }
    if set(canonical) != required:
        raise RunnerInvariantError("manifest fields differ")
    if (
        canonical["schema"] != MANIFEST_SCHEMA
        or canonical["arms"] != list(EVALUATION_ARMS)
        or canonical["budgets"] != _frozen_manifest_budgets()
        or canonical["environment"] != _frozen_manifest_environment()
        or canonical["evaluation_identity"] != EVALUATION_IDENTITY
        or canonical["gpu_assignment"] != FROZEN_GPU_ASSIGNMENT
        or canonical["host"] != "angler-workstation"
        or canonical["paths"] != FROZEN_PATHS
        or canonical["qualification_identity"] != QUALIFICATION_IDENTITY
        or canonical["evaluation_identity"] == EVALUATOR_EVALUATION_IDENTITY
        or canonical["qualification_identity"]
        == EVALUATOR_QUALIFICATION_IDENTITY
        or canonical["qualification_result_sha256"] is not None
        or canonical["resource_ceilings"] != FROZEN_RESOURCE_CEILINGS
        or canonical["thresholds"] != _frozen_manifest_thresholds()
        or canonical["write_scope"] != list(FROZEN_WRITE_SCOPE)
    ):
        raise RunnerInvariantError("manifest frozen identity differs")
    _validate_manifest_commitments(
        canonical["evaluator_commitments"],
        canonical["seed_commitments"],
    )
    if canonical["evaluator_commitment_digest"] != evaluator_record_digest(
        "evaluator-commitments",
        canonical["evaluator_commitments"],  # type: ignore[arg-type]
    ):
        raise RunnerInvariantError("manifest evaluator commitment digest differs")
    if type(canonical["frozen_sources"]) is not dict:
        raise RunnerInvariantError("manifest completed source map is malformed")
    sources = _validate_completed_source_map(canonical["frozen_sources"])
    if sources != canonical["frozen_sources"]:
        raise RunnerInvariantError("manifest completed source order differs")
    accepted_leaf = _raw_digest(canonical["leaf_sha256"], "accepted leaf sha256")
    if accepted_leaf != EXPECTED_ACTIVE_LEAF_SHA256:
        raise RunnerInvariantError("accepted R2 science leaf identity differs")
    leaf_path = Path(repository_root) / ACTIVE_LEAF_PATH.relative_to(REPOSITORY_ROOT)
    observed_leaf, _ = sha256_file(leaf_path)
    if observed_leaf != accepted_leaf:
        raise RunnerInvariantError("accepted science leaf bytes differ from manifest")
    validate_frozen_sources(repository_root, sources)
    scan_for_hidden_material(canonical)
    raw = canonical_json_bytes(canonical)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != _raw_digest(expected_hash, "manifest hash"):
        raise RunnerInvariantError("manifest hash differs")
    return digest


def load_source_manifest(
    path: str | Path,
    *,
    expected_hash: str | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Load one canonical regular 0600 manifest and revalidate every source."""

    target = Path(path)
    if not target.is_absolute():
        raise ValueError("manifest path must be absolute")
    raw = _read_private_artifact(target, maximum_bytes=MAXIMUM_IPC_BYTES)
    value = decode_canonical_json(raw, maximum_bytes=MAXIMUM_IPC_BYTES)
    if type(value) is not dict:
        raise RunnerInvariantError("source manifest is not an exact object")
    observed_hash = hashlib.sha256(raw).hexdigest()
    validated_hash = validate_source_manifest(
        value,
        expected_hash=(observed_hash if expected_hash is None else expected_hash),
        repository_root=repository_root,
    )
    if validated_hash != observed_hash:
        raise RunnerInvariantError("source manifest canonical file hash differs")
    return value


def publish_source_manifest(
    path: str | Path,
    manifest: Mapping[str, object],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> Path:
    """Validate and create the immutable pre-result source manifest once."""

    canonical = dict(manifest)
    manifest_hash = validate_source_manifest(
        canonical,
        repository_root=repository_root,
    )
    target = _publish_canonical_create_once(
        path,
        canonical,
        maximum_bytes=MAXIMUM_IPC_BYTES,
    )
    loaded = load_source_manifest(
        target,
        expected_hash=manifest_hash,
        repository_root=repository_root,
    )
    if loaded != canonical:
        raise RunnerInvariantError("published source manifest bytes differ")
    return target


def admission_digest(
    *,
    manifest_sha256: str,
    qualification_result_sha256: str,
    evaluator_commitment_digest: str,
    source_hashes: Mapping[str, str],
) -> str:
    _raw_digest(manifest_sha256, "manifest sha256")
    _raw_digest(qualification_result_sha256, "qualification result sha256")
    _digest(evaluator_commitment_digest, "evaluator commitment digest")
    sources = _validate_completed_source_map(source_hashes)
    payload = {
        "evaluator_commitment_digest": evaluator_commitment_digest,
        "manifest_sha256": manifest_sha256,
        "qualification_result_sha256": qualification_result_sha256,
        "source_hashes": sources,
    }
    return "sha256:" + hashlib.sha256(
        _ADMISSION_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()


def _build_qualification_release_claim(
    *,
    manifest_sha256: str,
    manifest_path: str | Path,
    result_path: str | Path,
    claim_path: str | Path,
    repository_root: str | Path,
    path_binding_mode: Literal["FROZEN_LIVE", "INJECTED_CPU_TEST"],
) -> tuple[dict[str, object], str]:
    if path_binding_mode not in ("FROZEN_LIVE", "INJECTED_CPU_TEST"):
        raise ValueError("qualification release path mode differs")
    paths = {
        "claim": str(Path(claim_path)),
        "manifest": str(Path(manifest_path)),
        "repository_root": str(Path(repository_root)),
        "result": str(Path(result_path)),
    }
    if any(
        not Path(value).is_absolute() or str(Path(value)) != value
        for value in paths.values()
    ):
        raise ValueError("qualification release paths differ")
    claim = {
        "identity": QUALIFICATION_IDENTITY,
        "manifest_sha256": _raw_digest(
            manifest_sha256,
            "qualification release manifest sha256",
        ),
        "path_binding_mode": path_binding_mode,
        "paths": paths,
        "purpose": "qualification",
        "schema": QUALIFICATION_RELEASE_CLAIM_SCHEMA,
    }
    return claim, content_ref("qualification-release-claim", claim)


def _validate_qualification_release_claim(
    value: Mapping[str, object],
    expected_ref: str | None = None,
) -> tuple[dict[str, object], str]:
    if type(value) is not dict or set(value) != {
        "identity",
        "manifest_sha256",
        "path_binding_mode",
        "paths",
        "purpose",
        "schema",
    } or (
        value["schema"] != QUALIFICATION_RELEASE_CLAIM_SCHEMA
        or value["identity"] != QUALIFICATION_IDENTITY
        or value["purpose"] != "qualification"
        or value["path_binding_mode"] not in ("FROZEN_LIVE", "INJECTED_CPU_TEST")
    ):
        raise RunnerInvariantError("qualification release claim fields differ")
    _raw_digest(value["manifest_sha256"], "qualification release manifest")
    paths = value["paths"]
    if type(paths) is not dict or set(paths) != {
        "claim",
        "manifest",
        "repository_root",
        "result",
    } or any(
        type(path) is not str
        or not Path(path).is_absolute()
        or str(Path(path)) != path
        for path in paths.values()
    ):
        raise RunnerInvariantError("qualification release claim paths differ")
    claim_ref = content_ref("qualification-release-claim", value)
    if expected_ref is not None and claim_ref != _digest(
        expected_ref,
        "qualification release claim ref",
    ):
        raise RunnerInvariantError("qualification release claim ref differs")
    return dict(value), claim_ref


def _load_qualification_release_claim(
    path: str | Path,
    *,
    expected_ref: str | None = None,
) -> tuple[dict[str, object], str]:
    target = Path(path)
    raw = _read_private_artifact(target, maximum_bytes=MAXIMUM_IPC_BYTES)
    value = decode_canonical_json(raw, maximum_bytes=MAXIMUM_IPC_BYTES)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise RunnerInvariantError("qualification release claim bytes differ")
    canonical, claim_ref = _validate_qualification_release_claim(value, expected_ref)
    if target != Path(canonical["paths"]["claim"]):  # type: ignore[index]
        raise RunnerInvariantError("qualification release claim path differs")
    return canonical, claim_ref


def _publish_qualification_release_claim(
    path: str | Path,
    claim: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    canonical, claim_ref = _validate_qualification_release_claim(claim)
    target = Path(path)
    encoded = canonical_json_bytes(canonical)
    if not target.is_absolute() or not 1 <= len(encoded) <= MAXIMUM_IPC_BYTES:
        raise ValueError("qualification release publication inputs differ")
    _mkdir_private(target.parent)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as error:
        raise QualificationIdentityConsumed(
            "one-use qualification identity is already consumed"
        ) from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(target.parent)
        loaded, loaded_ref = _load_qualification_release_claim(
            target,
            expected_ref=claim_ref,
        )
        if loaded != canonical or loaded_ref != claim_ref:
            raise RunnerInvariantError("qualification release claim differs")
    except BaseException as error:
        raise QualificationReleasePublicationFailure() from error
    return loaded, loaded_ref


def _load_evaluation_preconstruction_evidence(
    *,
    manifest_path: str | Path,
    seed_path: str | Path,
    qualification_result_path: str | Path,
    qualification_release_path: str | Path,
    repository_root: str | Path,
    frozen_live: bool,
) -> tuple[
    dict[str, object],
    str,
    dict[str, object],
    str,
    SeedSeal,
    dict[str, object],
    str,
]:
    """Reload the exact PASS/release/seed/source join before evaluation."""

    if type(frozen_live) is not bool:
        raise TypeError("evaluation preconstruction path mode must be exact")
    manifest_target = Path(manifest_path)
    seed_target = Path(seed_path)
    qualification_target = Path(qualification_result_path)
    release_target = Path(qualification_release_path)
    root = Path(repository_root)
    if any(
        not path.is_absolute()
        for path in (
            manifest_target,
            seed_target,
            qualification_target,
            release_target,
            root,
        )
    ):
        raise ValueError("evaluation preconstruction evidence paths must be absolute")
    manifest = load_source_manifest(manifest_target, repository_root=root)
    manifest_raw = _read_private_artifact(
        manifest_target,
        maximum_bytes=MAXIMUM_IPC_BYTES,
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    seal = load_seed_seal(seed_target, manifest["seed_commitments"])
    qualification, qualification_sha256 = load_qualification_result(
        qualification_target,
        expected_manifest_sha256=manifest_sha256,
    )
    if qualification["classification"] != "QUALIFICATION_PASS":
        raise RunnerInvariantError(
            "evaluation preconstruction requires qualification PASS"
        )
    release, release_ref = _load_qualification_release_claim(release_target)
    expected_mode = "FROZEN_LIVE" if frozen_live else "INJECTED_CPU_TEST"
    expected_release_paths = {
        "claim": str(release_target),
        "manifest": str(manifest_target),
        "repository_root": str(root),
        "result": str(qualification_target),
    }
    if (
        release["path_binding_mode"] != expected_mode
        or release["manifest_sha256"] != manifest_sha256
        or release["paths"] != expected_release_paths
    ):
        raise RunnerInvariantError(
            "evaluation preconstruction qualification release differs"
        )
    if frozen_live and (
        manifest_target != MANIFEST_PATH
        or seed_target != SEALED_SEED_PATH
        or qualification_target != QUALIFICATION_RESULT_PATH
        or release_target != QUALIFICATION_RELEASE_CLAIM_PATH
        or root != REPOSITORY_ROOT
        or manifest.get("paths") != FROZEN_PATHS
    ):
        raise RunnerInvariantError(
            "evaluation preconstruction paths are not frozen"
        )
    return (
        manifest,
        manifest_sha256,
        qualification,
        qualification_sha256,
        seal,
        release,
        release_ref,
    )


def build_evaluation_admission_claim(
    *,
    manifest_sha256: str,
    qualification_result_sha256: str,
    evaluator_commitment_digest: str,
    source_hashes: Mapping[str, str],
    manifest_path: str | Path,
    seed_path: str | Path,
    qualification_result_path: str | Path,
    evaluation_result_path: str | Path,
    claim_path: str | Path,
    repository_root: str | Path,
    seed_seal_ref: str,
    path_binding_mode: Literal["FROZEN_LIVE", "INJECTED_CPU_TEST"],
) -> tuple[dict[str, object], str]:
    """Build the canonical non-authorizing evaluation admission identity."""

    sources = _validate_completed_source_map(source_hashes)
    if path_binding_mode not in ("FROZEN_LIVE", "INJECTED_CPU_TEST"):
        raise ValueError("evaluation claim path-binding mode differs")
    paths = {
        "claim": str(Path(claim_path)),
        "evaluation_result": str(Path(evaluation_result_path)),
        "manifest": str(Path(manifest_path)),
        "qualification_result": str(Path(qualification_result_path)),
        "repository_root": str(Path(repository_root)),
        "seed": str(Path(seed_path)),
    }
    if any(
        not Path(value).is_absolute() or str(Path(value)) != value
        for value in paths.values()
    ):
        raise ValueError("evaluation admission claim paths differ")
    claim = {
        "admission_digest": admission_digest(
            manifest_sha256=manifest_sha256,
            qualification_result_sha256=qualification_result_sha256,
            evaluator_commitment_digest=evaluator_commitment_digest,
            source_hashes=sources,
        ),
        "evaluator_commitment_digest": _digest(
            evaluator_commitment_digest,
            "evaluation claim commitment digest",
        ),
        "manifest_sha256": _raw_digest(
            manifest_sha256,
            "evaluation claim manifest sha256",
        ),
        "path_binding_mode": path_binding_mode,
        "paths": paths,
        "qualification_result_sha256": _raw_digest(
            qualification_result_sha256,
            "evaluation claim qualification sha256",
        ),
        "schema": EVALUATION_ADMISSION_CLAIM_SCHEMA,
        "seed_seal_ref": _digest(seed_seal_ref, "evaluation claim seed seal"),
        "source_hashes": sources,
        "source_hashes_ref": content_ref("completed-source-map", sources),
    }
    return claim, content_ref("evaluation-admission-claim", claim)


def validate_evaluation_admission_claim(
    value: Mapping[str, object],
    expected_ref: str | None = None,
) -> tuple[dict[str, object], str]:
    if type(value) is not dict or set(value) != {
        "admission_digest",
        "evaluator_commitment_digest",
        "manifest_sha256",
        "path_binding_mode",
        "paths",
        "qualification_result_sha256",
        "schema",
        "seed_seal_ref",
        "source_hashes",
        "source_hashes_ref",
    } or value["schema"] != EVALUATION_ADMISSION_CLAIM_SCHEMA:
        raise RunnerInvariantError("evaluation admission-claim fields differ")
    sources = _validate_completed_source_map(value["source_hashes"])  # type: ignore[arg-type]
    paths = value["paths"]
    if (
        value["path_binding_mode"] not in ("FROZEN_LIVE", "INJECTED_CPU_TEST")
        or type(paths) is not dict
        or set(paths) != {
            "claim",
            "evaluation_result",
            "manifest",
            "qualification_result",
            "repository_root",
            "seed",
        }
        or any(
            type(path) is not str
            or not Path(path).is_absolute()
            or str(Path(path)) != path
            for path in paths.values()
        )
    ):
        raise RunnerInvariantError("evaluation admission path binding differs")
    _digest(value["seed_seal_ref"], "evaluation admission seed seal")
    if value["source_hashes_ref"] != content_ref("completed-source-map", sources):
        raise RunnerInvariantError("evaluation admission source-map reference differs")
    expected_digest = admission_digest(
        manifest_sha256=value["manifest_sha256"],  # type: ignore[arg-type]
        qualification_result_sha256=value["qualification_result_sha256"],  # type: ignore[arg-type]
        evaluator_commitment_digest=value["evaluator_commitment_digest"],  # type: ignore[arg-type]
        source_hashes=sources,
    )
    if value["admission_digest"] != expected_digest:
        raise RunnerInvariantError("evaluation admission digest differs")
    claim_ref = content_ref("evaluation-admission-claim", value)
    if expected_ref is not None and claim_ref != _digest(
        expected_ref,
        "evaluation admission claim reference",
    ):
        raise RunnerInvariantError("evaluation admission claim reference differs")
    return dict(value), claim_ref


def _frozen_live_claim_paths() -> dict[str, str]:
    return {
        "claim": FROZEN_PATHS["evaluation_admission_claim"],
        "evaluation_result": FROZEN_PATHS["evaluation_result"],
        "manifest": FROZEN_PATHS["manifest"],
        "qualification_result": FROZEN_PATHS["qualification_result"],
        "repository_root": FROZEN_PATHS["repository_root"],
        "seed": FROZEN_PATHS["sealed_seed"],
    }


def _validate_live_claim_manifest_binding(
    claim: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_sha256: str,
) -> None:
    """Join a result claim to the frozen live manifest, not a synthetic twin."""

    if (
        claim.get("path_binding_mode") != "FROZEN_LIVE"
        or claim.get("paths") != _frozen_live_claim_paths()
        or claim.get("manifest_sha256") != manifest_sha256
        or claim.get("source_hashes") != manifest.get("frozen_sources")
        or claim.get("evaluator_commitment_digest")
        != manifest.get("evaluator_commitment_digest")
        or manifest.get("paths") != FROZEN_PATHS
    ):
        raise RunnerInvariantError(
            "evaluation admission claim differs from frozen manifest"
        )


def _claim_target_is_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise RunnerInvariantError("evaluation admission identity is consumed")


def _require_identity_absent(
    path: Path,
    error_type: type[RunnerInvariantError],
) -> None:
    if path.exists() or path.is_symlink():
        raise error_type("one-use experiment identity is already consumed")


def _write_consuming_admission_claim(
    path: Path,
    claim: Mapping[str, object],
) -> None:
    """Create the claim itself with O_EXCL; uncertainty permanently consumes it."""

    if not path.is_absolute():
        raise ValueError("evaluation admission claim path must be absolute")
    canonical, _ = validate_evaluation_admission_claim(claim)
    encoded = canonical_json_bytes(canonical)
    if not 1 <= len(encoded) <= MAXIMUM_IPC_BYTES:
        raise RunnerInvariantError("evaluation admission claim exceeds its bound")
    _mkdir_private(path.parent)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as error:
        raise EvaluationIdentityConsumed(
            "one-use evaluation identity is already consumed"
        ) from error
    # Never unlink this target after O_EXCL succeeds.  A partial write or either
    # fsync uncertainty is a consumed one-use evaluation identity.
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException as error:
        raise AdmissionClaimPublicationFailure() from error


def load_evaluation_admission_claim(
    path: str | Path,
    *,
    expected_ref: str | None = None,
) -> tuple[dict[str, object], str]:
    target = Path(path)
    raw = _read_private_artifact(target, maximum_bytes=MAXIMUM_IPC_BYTES)
    value = decode_canonical_json(raw, maximum_bytes=MAXIMUM_IPC_BYTES)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise RunnerInvariantError("evaluation admission claim bytes differ")
    canonical, claim_ref = validate_evaluation_admission_claim(value, expected_ref)
    if target != Path(canonical["paths"]["claim"]):  # type: ignore[index]
        raise RunnerInvariantError("evaluation admission claim path differs")
    return canonical, claim_ref


@dataclass(slots=True)
class _EvaluationAdmissionPermit:
    """Non-reconstructable in-process handoff from O_EXCL to live release."""

    receipt: dict[str, object]
    _token: object = field(default_factory=object, repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not dict:
            raise TypeError("evaluation admission permit receipt differs")
        claim_ref = _digest(self.receipt.get("claim_ref"), "permit claim ref")
        if claim_ref in _ACTIVE_EVALUATION_PERMITS:
            raise RunnerInvariantError("evaluation admission permit already exists")
        _ACTIVE_EVALUATION_PERMITS[claim_ref] = self._token

    def consume(self) -> dict[str, object]:
        claim_ref = _digest(self.receipt.get("claim_ref"), "permit claim ref")
        if (
            self._consumed
            or _ACTIVE_EVALUATION_PERMITS.get(claim_ref) is not self._token
        ):
            raise RunnerInvariantError("evaluation admission permit is unavailable")
        self._consumed = True
        del _ACTIVE_EVALUATION_PERMITS[claim_ref]
        return dict(self.receipt)


async def _claim_evaluation_admission(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    seed_path: str | Path = SEALED_SEED_PATH,
    qualification_result_path: str | Path = QUALIFICATION_RESULT_PATH,
    evaluation_result_path: str | Path = EVALUATION_RESULT_PATH,
    claim_path: str | Path = EVALUATION_ADMISSION_CLAIM_PATH,
    repository_root: str | Path = REPOSITORY_ROOT,
    evaluator_handshake: Callable[[], Any],
    injected_paths: bool = False,
) -> _EvaluationAdmissionPermit:
    """Validate all public preimages, then durably consume evaluation once."""

    paths = tuple(
        Path(value)
        for value in (
            manifest_path,
            seed_path,
            qualification_result_path,
            evaluation_result_path,
            claim_path,
        )
    )
    if any(not path.is_absolute() for path in paths):
        raise ValueError("evaluation admission paths must be exact absolute paths")
    if type(injected_paths) is not bool:
        raise TypeError("injected_paths must be an exact bool")
    manifest_target, seed_target, qualification_target, result_target, claim_target = paths
    result_temporary = result_target.with_name(result_target.name + ".tmp")
    claim_temporary = claim_target.with_name(claim_target.name + ".tmp")
    _require_owned_real_parent(result_target.parent)
    _mkdir_private(claim_target.parent)
    for target in (result_target, result_temporary, claim_target, claim_temporary):
        _require_identity_absent(target, EvaluationIdentityConsumed)

    manifest = load_source_manifest(
        manifest_target,
        repository_root=repository_root,
    )
    manifest_raw = _read_private_artifact(
        manifest_target,
        maximum_bytes=MAXIMUM_IPC_BYTES,
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    seed_seal = load_seed_seal(seed_target, manifest["seed_commitments"])
    qualification, qualification_sha256 = load_qualification_result(
        qualification_target,
        expected_manifest_sha256=manifest_sha256,
    )
    if qualification["classification"] != "QUALIFICATION_PASS":
        raise RunnerInvariantError("evaluation admission requires qualification PASS")
    observed_handshake = await _maybe_await(evaluator_handshake())
    if hasattr(observed_handshake, "to_canonical"):
        observed_handshake = observed_handshake.to_canonical()
    expected_handshake = {
        "commitments": manifest["evaluator_commitments"],
        "digest": manifest["evaluator_commitment_digest"],
    }
    if observed_handshake != expected_handshake:
        raise RunnerInvariantError("evaluation admission evaluator handshake differs")
    # The awaited commitment-only handshake is an effect boundary.  Re-read
    # every preimage before the consuming O_EXCL so a concurrent mutation
    # cannot be admitted from stale bytes.
    reloaded_manifest = load_source_manifest(
        manifest_target,
        expected_hash=manifest_sha256,
        repository_root=repository_root,
    )
    reloaded_seed = load_seed_seal(
        seed_target,
        reloaded_manifest["seed_commitments"],
    )
    reloaded_qualification, reloaded_qualification_sha256 = (
        load_qualification_result(
            qualification_target,
            expected_manifest_sha256=manifest_sha256,
        )
    )
    if (
        reloaded_manifest != manifest
        or reloaded_seed != seed_seal
        or reloaded_qualification != qualification
        or reloaded_qualification_sha256 != qualification_sha256
    ):
        raise RunnerInvariantError("evaluation admission preimages changed")
    if not injected_paths:
        qualification_release, _ = _load_qualification_release_claim(
            QUALIFICATION_RELEASE_CLAIM_PATH
        )
        if (
            qualification_release["path_binding_mode"] != "FROZEN_LIVE"
            or qualification_release["manifest_sha256"] != manifest_sha256
            or qualification_release["paths"]
            != {
                "claim": str(QUALIFICATION_RELEASE_CLAIM_PATH),
                "manifest": str(manifest_target),
                "repository_root": str(Path(repository_root)),
                "result": str(qualification_target),
            }
        ):
            raise RunnerInvariantError(
                "evaluation admission qualification release differs"
            )
    sources = manifest["frozen_sources"]
    if type(sources) is not dict:
        raise RunnerInvariantError("evaluation admission source map differs")
    if not injected_paths:
        manifest_paths = manifest.get("paths")
        if type(manifest_paths) is not dict or {
            "claim": str(claim_target),
            "evaluation_result": str(result_target),
            "manifest": str(manifest_target),
            "qualification_result": str(qualification_target),
            "repository_root": str(Path(repository_root)),
            "seed": str(seed_target),
        } != {
            "claim": manifest_paths.get("evaluation_admission_claim"),
            "evaluation_result": manifest_paths.get("evaluation_result"),
            "manifest": manifest_paths.get("manifest"),
            "qualification_result": manifest_paths.get("qualification_result"),
            "repository_root": manifest_paths.get("repository_root"),
            "seed": manifest_paths.get("sealed_seed"),
        }:
            raise RunnerInvariantError("evaluation admission paths differ from manifest")
    claim, claim_ref = build_evaluation_admission_claim(
        manifest_sha256=manifest_sha256,
        qualification_result_sha256=qualification_sha256,
        evaluator_commitment_digest=manifest["evaluator_commitment_digest"],  # type: ignore[arg-type]
        source_hashes=sources,
        manifest_path=manifest_target,
        seed_path=seed_target,
        qualification_result_path=qualification_target,
        evaluation_result_path=result_target,
        claim_path=claim_target,
        repository_root=repository_root,
        seed_seal_ref=seed_seal.seal_ref,
        path_binding_mode=(
            "INJECTED_CPU_TEST" if injected_paths else "FROZEN_LIVE"
        ),
    )
    # Recheck every one-use output immediately before the consuming O_EXCL.
    for target in (result_target, result_temporary, claim_target, claim_temporary):
        _require_identity_absent(target, EvaluationIdentityConsumed)
    try:
        _write_consuming_admission_claim(claim_target, claim)
        durable_claim, durable_ref = load_evaluation_admission_claim(
            claim_target,
            expected_ref=claim_ref,
        )
        if durable_claim != claim or durable_ref != claim_ref:
            raise RunnerInvariantError("durable evaluation admission claim differs")
        # A result racing the admission claim consumes the identity and cannot run.
        for target in (result_target, result_temporary):
            _claim_target_is_absent(target)
        receipt = {
            "claim": durable_claim,
            "claim_path": str(claim_target),
            "claim_ref": durable_ref,
            "manifest_sha256": manifest_sha256,
            "qualification_result_sha256": qualification_sha256,
            "result_path": str(result_target),
            "seed_seal_ref": seed_seal.seal_ref,
        }
        canonical_json_bytes(receipt)
        permit = _EvaluationAdmissionPermit(receipt)
    except EvaluationIdentityConsumed:
        # Only the genuine O_EXCL loser is not the owner of this consumed
        # identity.  It must never enter terminal-result publication.
        raise
    except AdmissionClaimPublicationFailure:
        raise
    except BaseException as error:
        raise AdmissionClaimPublicationFailure() from error
    return permit


@dataclass(frozen=True, slots=True)
class ClonePaths:
    root: Path
    store: Path
    journal: Path
    learner: Path

    def __post_init__(self) -> None:
        paths = (self.root, self.store, self.journal, self.learner)
        if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
            raise ValueError("clone paths must be exact absolute Path values")
        if any(path != self.root and path.parent != self.root for path in paths):
            raise ValueError("clone artifacts must be direct children of clone root")
        if len({path.name for path in paths[1:]}) != 3:
            raise ValueError("clone artifact names must be distinct")

    @classmethod
    def for_task(
        cls,
        parent: str | Path,
        task_id: str,
        arm: str,
    ) -> "ClonePaths":
        _digest(task_id, "task_id")
        if arm not in EVALUATION_ARMS:
            raise ValueError("clone arm is not declared")
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            raise ValueError("clone parent must be absolute")
        slug = hashlib.sha256(f"{task_id}\x00{arm}".encode("ascii")).hexdigest()
        root = parent_path / slug
        return cls(
            root=root,
            store=root / "cognitive.sqlite3",
            journal=root / "qwen-execution-journal.sqlite3",
            learner=root / "learner-state.bin",
        )


@dataclass(frozen=True, slots=True)
class CloneAudit:
    schema: str
    source_hashes: tuple[tuple[str, str], ...]
    destination_hashes: tuple[tuple[str, str], ...]
    root: str

    def __post_init__(self) -> None:
        if self.schema != CLONE_SCHEMA:
            raise ValueError("clone audit schema differs")
        if (
            set(dict(self.source_hashes)) != CLONED_BASELINE_HASH_NAMES
            or set(dict(self.destination_hashes)) != CLONED_BASELINE_HASH_NAMES
            or _hash_pairs(dict(self.source_hashes), "clone source")
            != self.source_hashes
            or _hash_pairs(dict(self.destination_hashes), "clone destination")
            != self.destination_hashes
            or self.source_hashes != self.destination_hashes
        ):
            raise RunnerInvariantError("clone audit hashes differ")
        root = Path(self.root)
        if not root.is_absolute() or str(root) != self.root:
            raise ValueError("clone audit root must be a canonical absolute path")

    @property
    def audit_ref(self) -> str:
        return content_ref("probe-clone", self.to_canonical())

    def to_canonical(self) -> dict[str, object]:
        return {
            "destination_hashes": dict(self.destination_hashes),
            "root": self.root,
            "schema": self.schema,
            "source_hashes": dict(self.source_hashes),
        }


def _audit_closed_sqlite(path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        if Path(str(path) + suffix).exists():
            raise RunnerInvariantError("SQLite baseline has an open/sidecar state")
    uri = f"file:{path}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RunnerInvariantError("SQLite baseline integrity check failed")
    finally:
        connection.close()


def clone_closed_baseline(
    source: ClonePaths,
    destination: ClonePaths,
    expected_hashes: Mapping[str, str],
) -> CloneAudit:
    """Byte-copy only a closed store, journal, and learner snapshot."""

    if type(source) is not ClonePaths or type(destination) is not ClonePaths:
        raise TypeError("source and destination must be exact ClonePaths")
    if destination.root.exists() or destination.root.is_symlink():
        raise RunnerInvariantError("probe clone destination is not fresh")
    source_files = {
        "journal": source.journal,
        "learner": source.learner,
        "store": source.store,
    }
    if set(expected_hashes) != set(source_files):
        raise ValueError("expected clone hashes must name store, journal, and learner")
    observed: dict[str, str] = {}
    for label, path in source_files.items():
        digest, _ = sha256_file(path)
        if digest != _raw_digest(expected_hashes[label], f"expected {label} hash"):
            raise RunnerInvariantError(f"closed baseline {label} hash differs")
        observed[label] = digest
    _audit_closed_sqlite(source.store)
    _audit_closed_sqlite(source.journal)
    _mkdir_private(destination.root)
    destination_files = {
        "journal": destination.journal,
        "learner": destination.learner,
        "store": destination.store,
    }
    try:
        for label, target in destination_files.items():
            source_path = source_files[label]
            with source_path.open("rb") as reader:
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                with os.fdopen(descriptor, "wb", closefd=True) as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    writer.flush()
                    os.fsync(writer.fileno())
        _fsync_directory(destination.root)
        destination_hashes = {
            label: sha256_file(path)[0] for label, path in destination_files.items()
        }
        if destination_hashes != observed:
            raise RunnerInvariantError("probe clone bytes differ from baseline")
        source_after = {
            label: sha256_file(path)[0] for label, path in source_files.items()
        }
        if source_after != observed:
            raise RunnerInvariantError("closed baseline changed while it was cloned")
        _audit_closed_sqlite(source.store)
        _audit_closed_sqlite(source.journal)
        _audit_closed_sqlite(destination.store)
        _audit_closed_sqlite(destination.journal)
    except BaseException:
        shutil.rmtree(destination.root, ignore_errors=True)
        raise
    return CloneAudit(
        schema=CLONE_SCHEMA,
        source_hashes=tuple(sorted(observed.items())),
        destination_hashes=tuple(sorted(destination_hashes.items())),
        root=str(destination.root),
    )


def _fixture_ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(_text(label, "fixture label").encode("utf-8")).hexdigest()


def _qualified_unrelated_bootstrap_episode(
    state_digest: str,
    manifest: object,
) -> object:
    """Recreate the qualified neutral observed fixture without task aliases."""

    from angler.cognition.contracts import CognitiveEpisode
    from angler.cognition.prospective import ProspectiveCommitment

    _digest(state_digest, "bootstrap state digest")
    model_ref = _digest(getattr(manifest, "model_ref", None), "bootstrap model_ref")
    encoder_ref = _digest(
        getattr(manifest, "encoder_ref", None),
        "bootstrap encoder_ref",
    )
    evidence_ref = _fixture_ref(
        "frozen-qwen-cognee-cycle-v1-unrelated-observed-fixture-evidence"
    )
    task_id = "frozen-qwen-cognee-cycle-v1-unrelated-observed-fixture"
    proposals = (
        "Read the two supplied color words and preserve their given order.",
        "Return only the supplied color words separated by one space.",
    )
    return CognitiveEpisode(
        task_id=task_id,
        request=(
            "For an unrelated synthetic fixture, return the two supplied color "
            "words in their given order: cobalt then amber."
        ),
        recalled_refs=(evidence_ref,),
        proposals=proposals,
        selected_index=1,
        commitment=ProspectiveCommitment(
            parent_event_ref=None,
            task_id=task_id,
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
        feedback_source_ref=_fixture_ref(
            "frozen-qwen-cognee-cycle-v1-unrelated-fixture-disposition"
        ),
        parent_state_digest=state_digest,
        child_state_digest=state_digest,
        model_ref=model_ref,
        encoder_ref=encoder_ref,
        supporting_evidence_refs=(evidence_ref,),
    )


def _acquisition_sequence_sha256(store: object) -> str:
    digest = hashlib.sha256(
        b"angler.high-level-multidomain.acquisition-sequence.v1\x00"
    )
    after = -1
    count = 0
    while True:
        page = tuple(store.acquisition_items(after_ordinal=after, limit=64))
        if len(page) > 64:
            raise RunnerInvariantError("acquisition sequence page exceeds its bound")
        if not page:
            break
        for item in page:
            ordinal = getattr(item, "ordinal", None)
            acquisition = getattr(item, "acquisition", None)
            canonical = getattr(acquisition, "canonical_bytes", None)
            if type(ordinal) is not int or ordinal != count or not callable(canonical):
                raise RunnerInvariantError("acquisition sequence is not exact and contiguous")
            raw = canonical()
            if type(raw) is not bytes or not raw:
                raise RunnerInvariantError("acquisition sequence bytes differ")
            digest.update(ordinal.to_bytes(8, "big", signed=False))
            digest.update(len(raw).to_bytes(8, "big", signed=False))
            digest.update(raw)
            after = ordinal
            count += 1
        if len(page) < 64:
            break
    head = store.acquisition_head()
    if getattr(head, "next_ordinal", None) != count:
        raise RunnerInvariantError("acquisition sequence head differs")
    digest.update(count.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _closed_baseline_hashes(
    paths: ClonePaths,
    *,
    store: object | None = None,
    journal: object | None = None,
) -> dict[str, str]:
    from angler.runtime.cognitive_transaction_store import CognitiveTransactionStore
    from angler.runtime.qwen_cognitive import SQLiteQwenExecutionJournal

    if type(paths) is not ClonePaths:
        raise TypeError("baseline paths must be exact ClonePaths")
    _audit_closed_sqlite(paths.store)
    _audit_closed_sqlite(paths.journal)
    snapshot = _read_private_artifact(
        paths.learner,
        maximum_bytes=MAXIMUM_PENDING_SNAPSHOT_BYTES,
    )
    if not snapshot:
        raise RunnerInvariantError("baseline learner snapshot is empty")
    if store is None:
        store = CognitiveTransactionStore(paths.store)
    elif getattr(store, "path", None) != paths.store:
        raise RunnerInvariantError("baseline store owner path differs")
    head = store.audit_integrity()
    if (
        head is None
        or store.active_prospective_turn() is not None
        or store.pending_acquisition_projections(limit=1)
        or store.load_head_state() != snapshot
        or head.snapshot_sha256 != "sha256:" + hashlib.sha256(snapshot).hexdigest()
    ):
        raise RunnerInvariantError("closed baseline is not a quiescent exact snapshot")
    if journal is None:
        journal = SQLiteQwenExecutionJournal(paths.journal)
    elif getattr(journal, "path", None) != paths.journal:
        raise RunnerInvariantError("baseline journal owner path differs")
    journal.audit_integrity()
    _audit_closed_sqlite(paths.store)
    _audit_closed_sqlite(paths.journal)
    return {
        "acquisition_sequence": _acquisition_sequence_sha256(store),
        "journal": sha256_file(paths.journal)[0],
        "learner": hashlib.sha256(snapshot).hexdigest(),
        "store": sha256_file(paths.store)[0],
    }


def _assert_successor_quiescent(
    *,
    learner: object,
    store: object,
    journal: object,
    cycle: object | None = None,
) -> None:
    if (
        getattr(learner, "pending_decision", None) is not None
        or getattr(learner, "pending_prospective_material", None) is not None
        or store.active_prospective_turn() is not None  # type: ignore[attr-defined]
        or store.pending_acquisition_projections(limit=1)  # type: ignore[attr-defined]
    ):
        raise RunnerInvariantError("successor state is not quiescent")
    if cycle is not None and (
        getattr(cycle, "pending_turn", None) is not None
        or getattr(cycle, "executed_turn", None) is not None
        or getattr(cycle, "_beginning", None) is not False
        or getattr(cycle, "_executing", None) is not False
    ):
        raise RunnerInvariantError("cognitive cycle is not quiescent")
    head = store.audit_integrity()  # type: ignore[attr-defined]
    snapshot = learner.capture_state()  # type: ignore[attr-defined]
    if (
        head is None
        or head.state_digest != learner.state_digest()  # type: ignore[attr-defined]
        or store.load_head_state() != snapshot  # type: ignore[attr-defined]
    ):
        raise RunnerInvariantError("learner and canonical store head differ")
    journal.audit_integrity()  # type: ignore[attr-defined]


async def _quiesce_successor_runtime(
    *,
    cycle: object,
    learner: object,
    store: object,
    journal: object,
    memory: object,
) -> None:
    if (
        getattr(cycle, "pending_turn", None) is not None
        or getattr(cycle, "executed_turn", None) is not None
        or getattr(cycle, "_beginning", None) is not False
        or getattr(cycle, "_executing", None) is not False
        or getattr(learner, "pending_decision", None) is not None
        or getattr(learner, "pending_prospective_material", None) is not None
    ):
        raise RunnerInvariantError("cycle cannot be quiesced with pending material")
    await cycle.retry_pending_projections(limit=MAXIMUM_PROJECTION_RETRY)  # type: ignore[attr-defined]
    memory.assert_synchronized(store.acquisition_head())  # type: ignore[attr-defined]
    _assert_successor_quiescent(
        learner=learner,
        store=store,
        journal=journal,
        cycle=cycle,
    )


@dataclass(slots=True)
class LineageBaselineOwner:
    purpose: Purpose
    replicate_commitment: str
    arm: Literal["FULL", "RANDOM_FEEDBACK"]
    paths: ClonePaths
    learner: object = field(repr=False)
    store: object = field(repr=False)
    journal: object = field(repr=False)
    genesis_digest: str
    scope_spec: AcquisitionScopeSpec
    dataset_id: str
    initial_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.purpose not in ("qualification", "evaluation"):
            raise ValueError("lineage baseline purpose differs")
        _digest(self.replicate_commitment, "lineage replicate commitment")
        if self.arm not in ("FULL", "RANDOM_FEEDBACK"):
            raise ValueError("lineage baseline arm differs")
        if type(self.paths) is not ClonePaths:
            raise TypeError("lineage baseline paths differ")
        _digest(self.genesis_digest, "lineage genesis digest")
        if type(self.scope_spec) is not AcquisitionScopeSpec:
            raise TypeError("lineage acquisition scope differs")
        _text(self.dataset_id, "lineage acquisition dataset id", 512)
        if _hash_pairs(dict(self.initial_hashes), "initial lineage baseline") != self.initial_hashes:
            raise ValueError("initial lineage baseline hashes are not canonical")
        if set(dict(self.initial_hashes)) != PROBE_BASELINE_HASH_NAMES:
            raise ValueError("initial lineage baseline hash names differ")

    @classmethod
    async def create(
        cls,
        root: str | Path,
        *,
        purpose: Purpose,
        replicate_commitment: str,
        arm: Literal["FULL", "RANDOM_FEEDBACK"],
        genesis: FreshGenesisBundle,
        manifest: object,
        scope_parent: str | Path,
        acquisition_opener: AcquisitionOpener,
    ) -> "LineageBaselineOwner":
        from angler.memory.cognitive_acquisition_graph import AcquisitionSituatedMemory
        from angler.runtime.cognitive_transaction_store import CognitiveTransactionStore
        from angler.runtime.qwen_cognitive import SQLiteQwenExecutionJournal

        target = _create_fresh_private_root(root)
        paths = ClonePaths(
            root=target,
            store=target / "cognitive.sqlite3",
            journal=target / "qwen-execution-journal.sqlite3",
            learner=target / "learner-state.bin",
        )
        learner = restore_fresh_cpu_genesis(genesis)
        snapshot = learner.capture_state()
        state_digest = learner.state_digest()
        if state_digest != QUALIFIED_INITIAL_COMPETENCE_DIGEST:
            raise RunnerInvariantError("lineage baseline does not begin at genesis")
        store = CognitiveTransactionStore(paths.store)
        store.initialize(
            state_digest,
            snapshot,
            model_ref=getattr(manifest, "model_ref", None),
            encoder_ref=getattr(manifest, "encoder_ref", None),
        )
        bootstrap_commit = store.commit_episode(
            _qualified_unrelated_bootstrap_episode(state_digest, manifest),
            snapshot,
            expected_parent_digest=state_digest,
        )
        journal = SQLiteQwenExecutionJournal(paths.journal)
        bootstrap_head = store.audit_integrity()
        if (
            bootstrap_commit.sequence != QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
            or bootstrap_commit.head.sequence
            != QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
            or bootstrap_commit.head.state_digest != state_digest
            or bootstrap_head != bootstrap_commit.head
        ):
            raise RunnerInvariantError(
                "qualified neutral bootstrap transaction baseline differs"
            )
        journal.audit_integrity()
        os.chmod(paths.store, 0o600)
        os.chmod(paths.journal, 0o600)
        _write_private_artifact(
            paths.learner,
            snapshot,
            maximum_bytes=MAXIMUM_PENDING_SNAPSHOT_BYTES,
        )
        _fsync_directory(target)
        scope_spec = AcquisitionScopeSpec.for_lineage(
            purpose=purpose,
            replicate=replicate_commitment,
            arm=arm,
            state_parent=scope_parent,
        )
        session = await _open_scoped_session(
            scope_spec,
            opener=acquisition_opener,
        )
        primary: BaseException | None = None
        try:
            memory = AcquisitionSituatedMemory(source=store, backend=session.backend)
            await memory.rebuild(limit=MAXIMUM_PROJECTION_RETRY)
            await memory.retry_pending_projections(limit=MAXIMUM_PROJECTION_RETRY)
            memory.assert_synchronized(store.acquisition_head())
            if store.pending_acquisition_projections(limit=1):
                raise RunnerInvariantError("fresh lineage acquisition outbox was not drained")
        except BaseException as error:
            primary = error
        cleanup: BaseException | None = None
        try:
            await session.forget_then_close(opener=acquisition_opener)
        except BaseException as error:
            cleanup = error
        if primary is not None and cleanup is not None:
            raise BaseExceptionGroup(
                "fresh lineage construction and cleanup failed",
                [primary, cleanup],
            )
        if primary is not None:
            raise primary.with_traceback(primary.__traceback__)
        if cleanup is not None:
            raise cleanup.with_traceback(cleanup.__traceback__)
        _assert_successor_quiescent(
            learner=learner,
            store=store,
            journal=journal,
        )
        hashes = _closed_baseline_hashes(
            paths,
            store=store,
            journal=journal,
        )
        return cls(
            purpose=purpose,
            replicate_commitment=replicate_commitment,
            arm=arm,
            paths=paths,
            learner=learner,
            store=store,
            journal=journal,
            genesis_digest=state_digest,
            scope_spec=scope_spec,
            dataset_id=session.dataset_id,
            initial_hashes=tuple(sorted(hashes.items())),
        )

    def scope_binding(self) -> dict[str, object]:
        record = {
            "arm": self.arm,
            "dataset_id": self.dataset_id,
            "dataset_name": self.scope_spec.dataset_name,
            "node_set_name": self.scope_spec.node_set_name,
            "purpose": self.purpose,
            "replicate_commitment": self.replicate_commitment,
            "state_root": self.scope_spec.state_root,
            "tenant_name": self.scope_spec.tenant_name,
        }
        canonical_json_bytes(record)
        return record

    @property
    def scope_binding_ref(self) -> str:
        return content_ref("lineage-acquisition-scope", self.scope_binding())

    def audit_binding(self) -> dict[str, object]:
        record = {
            "hashes": self.hashes(),
            "scope": self.scope_binding(),
            "scope_ref": self.scope_binding_ref,
        }
        canonical_json_bytes(record)
        return record

    def hashes(self) -> dict[str, str]:
        return _closed_baseline_hashes(
            self.paths,
            store=self.store,
            journal=self.journal,
        )

    def persist_and_audit(self, *, cycle: object | None = None) -> dict[str, str]:
        _assert_successor_quiescent(
            learner=self.learner,
            store=self.store,
            journal=self.journal,
            cycle=cycle,
        )
        snapshot = self.learner.capture_state()
        if not 1 <= len(snapshot) <= MAXIMUM_PENDING_SNAPSHOT_BYTES:
            raise RunnerInvariantError("lineage learner snapshot exceeds its bound")
        head = self.store.audit_integrity()
        if (
            head is None
            or head.state_digest != self.learner.state_digest()
            or self.store.load_head_state() != snapshot
        ):
            raise RunnerInvariantError("lineage learner and canonical store differ")
        self.journal.audit_integrity()
        _replace_private_artifact(
            self.paths.learner,
            snapshot,
            maximum_bytes=MAXIMUM_PENDING_SNAPSHOT_BYTES,
        )
        return self.hashes()


async def create_fresh_lineage_baselines(
    parent: str | Path,
    *,
    purpose: Purpose,
    replicate_commitments: Sequence[str],
    genesis: FreshGenesisBundle,
    manifest: object,
    scope_parent: str | Path,
    acquisition_opener: AcquisitionOpener,
) -> dict[tuple[str, str], LineageBaselineOwner]:
    target = Path(parent)
    if not target.is_absolute() or target.is_symlink() or not target.is_dir():
        raise ValueError("lineage baseline parent must be an exact absolute directory")
    replicates = tuple(replicate_commitments)
    if not replicates or len(set(replicates)) != len(replicates):
        raise ValueError("lineage replicate commitments must be nonempty and unique")
    owners: dict[tuple[str, str], LineageBaselineOwner] = {}
    initial_reference: dict[str, str] | None = None
    for replicate in replicates:
        _digest(replicate, "lineage replicate commitment")
        for arm in ("FULL", "RANDOM_FEEDBACK"):
            slug = hashlib.sha256(f"{replicate}\x00{arm}".encode("ascii")).hexdigest()
            owner = await LineageBaselineOwner.create(
                target / slug,
                purpose=purpose,
                replicate_commitment=replicate,
                arm=arm,
                genesis=genesis,
                manifest=manifest,
                scope_parent=scope_parent,
                acquisition_opener=acquisition_opener,
            )
            observed = owner.hashes()
            if initial_reference is None:
                initial_reference = observed
            elif observed != initial_reference:
                raise RunnerInvariantError(
                    "fresh lineage learner/store/journal/acquisition bytes differ"
                )
            owners[(replicate, arm)] = owner
    return owners


def _record_bytes(record: object) -> bytes:
    canonical = getattr(record, "canonical_bytes", None)
    if callable(canonical):
        value = canonical()
        if type(value) is not bytes or not value:
            raise ValueError("canonical record bytes are invalid")
        return value
    payload = getattr(record, "to_payload", None)
    if callable(payload):
        return canonical_json_bytes(payload())
    if type(record) is bytes and record:
        return record
    if isinstance(record, Mapping):
        return canonical_json_bytes(dict(record))
    raise TypeError("resolved record lacks canonical bytes")


@dataclass(frozen=True, slots=True)
class FrozenRecallItem:
    record_ref: str
    adjacent_record_refs: tuple[str, ...]
    score_hex: str | None
    backend_ref: str | None
    record_bytes_sha256: str
    record_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _digest(self.record_ref, "record_ref")
        if (
            type(self.adjacent_record_refs) is not tuple
            or tuple(sorted(set(self.adjacent_record_refs))) != self.adjacent_record_refs
        ):
            raise ValueError("adjacent references must be canonical and unique")
        for value in self.adjacent_record_refs:
            _digest(value, "adjacent_record_ref")
        if self.score_hex is not None:
            try:
                value = float.fromhex(self.score_hex)
            except (TypeError, ValueError) as error:
                raise ValueError("backend score hex is invalid") from error
            if not math.isfinite(value):
                raise ValueError("backend score must be finite")
        if self.backend_ref is not None:
            _text(self.backend_ref, "backend_ref", 512)
        _raw_digest(self.record_bytes_sha256, "record bytes sha256")
        if type(self.record_bytes) is not bytes or not self.record_bytes:
            raise ValueError("resolved record bytes must be nonempty")
        if hashlib.sha256(self.record_bytes).hexdigest() != self.record_bytes_sha256:
            raise ValueError("resolved record bytes hash differs")

    def to_canonical(self) -> dict[str, object]:
        return {
            "adjacent_record_refs": list(self.adjacent_record_refs),
            "backend_ref": self.backend_ref,
            "record_bytes_sha256": self.record_bytes_sha256,
            "record_ref": self.record_ref,
            "score_hex": self.score_hex,
        }

    def recalled_text(self) -> str:
        try:
            payload = json.loads(self.record_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunnerInvariantError("canonical recall record cannot be decoded") from error
        if type(payload) is not dict or type(payload.get("content")) is not str:
            raise RunnerInvariantError("canonical recall record lacks public content")
        return _text(payload["content"], "recalled record content", 16_384)


@dataclass(frozen=True, slots=True)
class FrozenRecallBatch:
    items: tuple[FrozenRecallItem, ...]
    search_calls: int

    def __post_init__(self) -> None:
        if (
            type(self.items) is not tuple
            or not self.items
            or len(self.items) > MAXIMUM_RECALL_ITEMS
            or len({item.record_ref for item in self.items}) != len(self.items)
        ):
            raise ValueError("frozen recall items are empty, duplicated, or over limit")
        if type(self.search_calls) is not int or self.search_calls < 0:
            raise ValueError("search_calls must be a non-negative integer")

    @classmethod
    def from_hits(
        cls,
        hits: Sequence[object],
        resolved_records: Sequence[object],
        search_calls: int,
    ) -> "FrozenRecallBatch":
        hit_values = tuple(hits)
        record_values = tuple(resolved_records)
        if not hit_values or len(hit_values) != len(record_values):
            raise ValueError("hits and resolved records must align one-to-one")
        items: list[FrozenRecallItem] = []
        for hit, record in zip(hit_values, record_values, strict=True):
            record_ref = getattr(hit, "record_ref", None)
            _digest(record_ref, "hit record_ref")
            resolved_ref = getattr(record, "record_ref", record_ref)
            if resolved_ref != record_ref:
                raise ValueError("resolved record does not match hit order")
            adjacent = tuple(getattr(hit, "adjacent_record_refs", ()))
            score = getattr(hit, "score", None)
            if score is not None and (
                type(score) not in (int, float) or not math.isfinite(float(score))
            ):
                raise ValueError("hit score is not finite")
            raw = _record_bytes(record)
            items.append(
                FrozenRecallItem(
                    record_ref=record_ref,
                    adjacent_record_refs=adjacent,
                    score_hex=None if score is None else float(score).hex(),
                    backend_ref=getattr(hit, "backend_ref", None),
                    record_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                    record_bytes=raw,
                )
            )
        return cls(tuple(items), search_calls)

    @property
    def batch_ref(self) -> str:
        return content_ref("frozen-recall", self.to_canonical())

    def to_canonical(self) -> dict[str, object]:
        return {
            "items": [item.to_canonical() for item in self.items],
            "schema": FROZEN_RECALL_SCHEMA,
            "search_calls": self.search_calls,
        }

    def runtime_hits(self) -> tuple[object, ...]:
        from angler.memory.cognitive_acquisition_graph import AcquisitionReferenceHit

        return tuple(
            AcquisitionReferenceHit(
                record_ref=item.record_ref,
                adjacent_record_refs=item.adjacent_record_refs,
                score=None if item.score_hex is None else float.fromhex(item.score_hex),
                backend_ref=item.backend_ref,
            )
            for item in self.items
        )


@dataclass(slots=True)
class ReplayOnlyRecallProvider:
    backend: object
    search_calls: int = 0
    _disposed: bool = field(default=False, init=False, repr=False)

    def dispose(self) -> None:
        if self._disposed:
            raise RunnerInvariantError("replay-only provider is already disposed")
        self._disposed = True

    async def project(self, projection: object) -> object:
        if self._disposed:
            raise RunnerInvariantError("replay-only provider is disposed")
        if not _projection_is_recall_eligible(projection):
            return None
        return await self.backend.project(projection)  # type: ignore[attr-defined]

    async def search(self, query: str, *, limit: int) -> Sequence[object]:
        if self._disposed:
            raise RunnerInvariantError("replay-only provider is disposed")
        self.search_calls += 1
        raise RunnerInvariantError("BACKEND_REMOVAL must make zero search calls")

    async def forget_namespace(self) -> None:
        if self._disposed:
            raise RunnerInvariantError("replay-only provider is disposed")
        await self.backend.forget_namespace()  # type: ignore[attr-defined]


def _projection_is_recall_eligible(projection: object) -> bool:
    record = getattr(projection, "record", None)
    status = getattr(getattr(record, "epistemic_status", None), "value", None)
    if status not in ("OBSERVED", "VALIDATED", "PROPOSED", "RETRACTED"):
        raise RunnerInvariantError("projection epistemic status differs")
    return status in ("OBSERVED", "VALIDATED")


@dataclass(slots=True)
class CaptureOnceRecallProvider:
    """Capture the cycle's sole real search before returning its exact hits."""

    backend: object
    expected_query: str
    expected_limit: int = MAXIMUM_RECALL_ITEMS
    expected_batch: FrozenRecallBatch | None = None
    underlying_calls: int = 0
    forget_calls: int = 0
    captured_batch: FrozenRecallBatch | None = field(default=None, init=False)
    raw_hit_count: int = field(default=0, init=False)
    rejected_hit_count: int = field(default=0, init=False)
    _resolver: Callable[
        [Sequence[object], int],
        tuple[Sequence[object], Sequence[object], Sequence[str]],
    ] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _search_attempted: bool = field(default=False, init=False, repr=False)
    _disposed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.expected_query = _text(
            self.expected_query,
            "capture-once expected query",
            16_384,
        )
        if type(self.expected_limit) is not int or not 1 <= self.expected_limit <= 256:
            raise ValueError("capture-once limit must be an integer from 1 through 256")
        if self.expected_batch is not None and type(self.expected_batch) is not FrozenRecallBatch:
            raise TypeError("capture-once expected batch differs")
        for method in ("project", "search", "forget_namespace"):
            if not callable(getattr(self.backend, method, None)):
                raise TypeError(f"capture-once backend lacks {method}")

    @property
    def search_calls(self) -> int:
        return self.underlying_calls

    def bind_resolver(
        self,
        resolver: Callable[
            [Sequence[object], int],
            tuple[Sequence[object], Sequence[object], Sequence[str]],
        ],
    ) -> None:
        if (
            not callable(resolver)
            or self._resolver is not None
            or self._search_attempted
            or self._disposed
        ):
            raise RunnerInvariantError("capture-once resolver binding differs")
        self._resolver = resolver

    def dispose(self) -> None:
        if self._disposed:
            raise RunnerInvariantError("capture-once provider is already disposed")
        self._disposed = True

    async def project(self, projection: object) -> object:
        if self._disposed:
            raise RunnerInvariantError("capture-once provider is disposed")
        if not _projection_is_recall_eligible(projection):
            return None
        return await self.backend.project(projection)  # type: ignore[attr-defined]

    async def search(self, query: str, *, limit: int) -> Sequence[object]:
        validated = _text(query, "capture-once query", 16_384)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("capture-once limit must be an integer from 1 through 256")
        if self._disposed or self._search_attempted:
            raise RunnerInvariantError("capture-once permits exactly one search")
        self._search_attempted = True
        if (
            validated.encode("utf-8") != self.expected_query.encode("utf-8")
            or limit != self.expected_limit
            or self._resolver is None
        ):
            raise RunnerInvariantError("capture-once query bytes, limit, or resolver differ")
        self.underlying_calls += 1
        observed = await self.backend.search(query, limit=limit)  # type: ignore[attr-defined]
        if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
            raise TypeError("capture-once backend returned a non-sequence")
        raw_hits = tuple(observed)
        if not raw_hits or len(raw_hits) > limit:
            raise RunnerInvariantError("capture-once backend returned invalid cardinality")
        self.raw_hit_count = len(raw_hits)
        admitted_values, record_values, rejected_values = self._resolver(
            raw_hits,
            limit,
        )
        hits = tuple(admitted_values)
        records = tuple(record_values)
        rejected = tuple(rejected_values)
        self.rejected_hit_count = len(rejected)
        if (
            not hits
            or len(hits) != len(records)
            or len(hits) > len(raw_hits)
            or not set(getattr(hit, "record_ref", None) for hit in hits).issubset(
                getattr(hit, "record_ref", None) for hit in raw_hits
            )
            or any(type(reason) is not str or not reason for reason in rejected)
        ):
            raise RunnerInvariantError("capture-once canonical admission differs")
        captured = FrozenRecallBatch.from_hits(
            hits,
            records,
            search_calls=self.underlying_calls,
        )
        if self.expected_batch is not None and captured != self.expected_batch:
            raise RunnerInvariantError("live recall differs from the frozen FULL batch")
        self.captured_batch = captured
        return hits

    async def forget_namespace(self) -> None:
        if self._disposed:
            raise RunnerInvariantError("capture-once provider is disposed")
        self.forget_calls += 1
        await self.backend.forget_namespace()  # type: ignore[attr-defined]


def _validated_removal_fairness_preimage(
    full: Mapping[str, object],
    frozen_origin: Mapping[str, object],
    prospective_removal: Mapping[str, object],
    backend_removal: Mapping[str, object],
    retrieval_only: Mapping[str, object],
    *,
    require_score: bool = True,
) -> dict[str, dict[str, object]]:
    """Return the canonical predicate-8 preimage after exact validation."""

    if type(require_score) is not bool:
        raise TypeError("require_score must be an exact bool")
    rows = {
        "FULL": dict(full),
        "FROZEN_ORIGIN": dict(frozen_origin),
        "PROSPECTIVE_REMOVAL": dict(prospective_removal),
        "BACKEND_REMOVAL": dict(backend_removal),
        "RETRIEVAL_ONLY": dict(retrieval_only),
    }
    for arm, row in rows.items():
        if set(row) != REMOVAL_EVIDENCE_FIELDS:
            raise RunnerInvariantError(f"{arm} removal evidence fields differ")
        _digest(row["task_id"], f"{arm} removal task_id")
        for name in (
            "backend_raw_hit_count",
            "backend_rejected_hit_count",
            "backend_search_calls",
        ):
            if type(row[name]) is not int or row[name] < 0:
                raise RunnerInvariantError(f"{arm} removal counter differs")
        for name in (
            "evidence_final_ref",
            "evidence_stage_ref",
            "frozen_recall_ref",
            "proposal_generation_ref",
            "proposal_prompt_ref",
            "probe_integrity_ref",
            "public_task_ref",
        ):
            _digest(row[name], f"{arm} {name}")
        for name in (
            "execution_prompt_ref",
            "execution_receipt_ref",
            "execution_request_ref",
            "runtime_quiescence_ref",
            "selection_ref",
            "task_response_generation_ref",
        ):
            if row[name] is not None:
                _digest(row[name], f"{arm} {name}")
        for name in (
            "execution_receipt_bytes",
            "execution_request_bytes",
            "proposal_generation_bytes",
            "proposal_request_bytes",
            "selection_bytes",
            "task_response_generation_bytes",
        ):
            if type(row[name]) is not str or re.fullmatch(
                r"(?:[0-9a-f]{2})*",
                row[name],
            ) is None:
                raise RunnerInvariantError(f"{arm} {name} differs")
        if (
            type(row["proposals"]) is not list
            or any(type(item) is not str for item in row["proposals"])
            or type(row["recalled_record_refs"]) is not list
            or type(row["recalled_record_bytes_sha256"]) is not list
            or len(row["recalled_record_refs"])
            != len(row["recalled_record_bytes_sha256"])
            or any(
                type(item) is not str or _DIGEST.fullmatch(item) is None
                for item in row["recalled_record_refs"]
            )
            or any(
                type(item) is not str or _RAW_DIGEST.fullmatch(item) is None
                for item in row["recalled_record_bytes_sha256"]
            )
            or type(row["raw_response"]) is not str
            or row["selected_trace"] is not None
            and type(row["selected_trace"]) is not str
            or row["score"] is not None
            and (
                type(row["score"]) is not float
                or row["score"] not in (0.0, 1.0)
            )
        ):
            raise RunnerInvariantError(f"{arm} removal public payload differs")
        if require_score and (
            type(row["score"]) is not float or row["score"] not in (0.0, 1.0)
        ):
            raise RunnerInvariantError(f"{arm} removal score differs")
    common_fields = (
        "task_id",
        "public_task_ref",
        "frozen_recall_ref",
        "recalled_record_refs",
        "recalled_record_bytes_sha256",
        "proposal_request_bytes",
        "proposal_generation_ref",
        "proposal_generation_bytes",
        "proposal_prompt_ref",
        "proposals",
    )
    reference = tuple(rows["FULL"].get(name) for name in common_fields)
    for arm, row in rows.items():
        if tuple(row.get(name) for name in common_fields) != reference:
            raise RunnerInvariantError(f"{arm} differs before its declared intervention")
    search_expectations = {
        "FULL": 1,
        "FROZEN_ORIGIN": 1,
        "PROSPECTIVE_REMOVAL": 1,
        "BACKEND_REMOVAL": 0,
        "RETRIEVAL_ONLY": 0,
    }
    for arm, expected_calls in search_expectations.items():
        if rows[arm].get("backend_search_calls") != expected_calls:
            raise RunnerInvariantError(f"{arm} backend search accounting differs")
    backend_exact_fields = (
        "selection_bytes",
        "selection_ref",
        "selected_trace",
        "execution_prompt_ref",
        "execution_request_bytes",
        "execution_request_ref",
        "task_response_generation_bytes",
        "task_response_generation_ref",
        "execution_receipt_bytes",
        "execution_receipt_ref",
        "raw_response",
    )
    if require_score:
        backend_exact_fields = (*backend_exact_fields, "score")
    for name in backend_exact_fields:
        if name not in rows["FULL"] or name not in rows["BACKEND_REMOVAL"]:
            raise RunnerInvariantError(f"removal evidence lacks exact {name}")
        if rows["BACKEND_REMOVAL"].get(name) != rows["FULL"].get(name):
            raise RunnerInvariantError(f"BACKEND_REMOVAL differs from FULL {name}")
    canonical_json_bytes(rows)
    scan_for_hidden_material(rows)
    return rows


def validate_removal_fairness(
    full: Mapping[str, object],
    frozen_origin: Mapping[str, object],
    prospective_removal: Mapping[str, object],
    backend_removal: Mapping[str, object],
    retrieval_only: Mapping[str, object],
    *,
    require_score: bool = True,
) -> str:
    """Validate and commit the complete canonical predicate-8 preimage."""

    rows = _validated_removal_fairness_preimage(
        full,
        frozen_origin,
        prospective_removal,
        backend_removal,
        retrieval_only,
        require_score=require_score,
    )
    return content_ref(
        "removal-fairness",
        {"arms": rows, "require_score": require_score},
    )


@dataclass(frozen=True, slots=True)
class AcquisitionScopeSpec:
    dataset_name: str
    tenant_name: str
    node_set_name: str
    state_root: str

    @classmethod
    def for_lineage(
        cls,
        *,
        purpose: Purpose,
        replicate: str,
        arm: str,
        state_parent: str | Path = STATE_ROOT / "cognee-scopes",
    ) -> "AcquisitionScopeSpec":
        if purpose not in ("qualification", "evaluation"):
            raise ValueError("scope purpose is not declared")
        _digest(replicate, "replicate commitment")
        if arm not in ("FULL", "RANDOM_FEEDBACK"):
            raise ValueError("persistent Cognee scopes belong only to adaptation lineages")
        identity = hashlib.sha256(
            f"{purpose}\x00{replicate}\x00{arm}".encode("ascii")
        ).hexdigest()[:24]
        dataset = f"hlmd-{identity}"
        root = Path(state_parent) / dataset
        return cls(
            dataset_name=dataset,
            tenant_name=f"{dataset}-tenant",
            node_set_name=f"{dataset}-records",
            state_root=str(root),
        )

    @classmethod
    def for_clone(
        cls,
        *,
        run_label: str,
        replicate: str,
        arm: str,
        task_id: str,
        state_parent: str | Path = STATE_ROOT / "cognee-scopes",
    ) -> "AcquisitionScopeSpec":
        """Return one task/arm-specific disposable acquisition namespace."""

        purpose: Purpose
        if run_label in ("qualification", "evaluation"):
            purpose = run_label  # type: ignore[assignment]
        else:
            raise ValueError("run_label must be qualification or evaluation")
        _digest(replicate, "replicate commitment")
        _digest(task_id, "task_id")
        if arm not in EVALUATION_ARMS or arm == "QWEN_ONLY":
            raise ValueError("disposable Cognee scope arm differs")
        comparison_arms = {
            "FULL",
            "RETRIEVAL_ONLY",
            "FROZEN_ORIGIN",
            "PROSPECTIVE_REMOVAL",
            "BACKEND_REMOVAL",
        }
        scope_lane = "comparison" if arm in comparison_arms else "random-feedback"
        identity = hashlib.sha256(
            f"{purpose}\x00{replicate}\x00{scope_lane}\x00{task_id}".encode(
                "ascii"
            )
        ).hexdigest()[:24]
        dataset = f"hlmd-{identity}"
        return cls(
            dataset_name=dataset,
            tenant_name=f"{dataset}-tenant",
            node_set_name=f"{dataset}-records",
            state_root=str(Path(state_parent) / dataset),
        )

    def runtime_scope(self) -> object:
        from angler.memory.cognee_worker_protocol import CogneeWorkerScope

        return CogneeWorkerScope(
            dataset_name=self.dataset_name,
            tenant_name=self.tenant_name,
            node_set_name=self.node_set_name,
            state_root=self.state_root,
        )


async def open_scoped_acquisition(
    scope_spec: AcquisitionScopeSpec,
    *,
    expected_dataset_id: str | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[object, object]:
    """Open the exact public scoped binding and acquisition adapter."""

    from angler.memory.cognee_acquisition_adapter import CogneeAcquisitionAdapter
    from angler.memory.cognee_subprocess_bindings import CogneeSubprocessBindings

    scope = scope_spec.runtime_scope()
    binding = await CogneeSubprocessBindings.start(
        scope=scope,
        expected_dataset_id=expected_dataset_id,
        timeout_seconds=timeout_seconds,
    )
    try:
        adapter = CogneeAcquisitionAdapter(
            bindings=binding,
            tenant_id=binding.tenant_id,
            dataset_id=binding.dataset_id,
            dataset_name=binding.dataset_name,
            node_set_name=binding.node_set_name,
            local_embeddings_configured=True,
            external_embedding_calls_authorized=False,
            telemetry_authorized=False,
        )
    except BaseException as primary:
        try:
            await binding.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "scoped acquisition construction and binding cleanup failed",
                [primary, cleanup],
            ) from None
        raise
    return binding, adapter


AcquisitionOpener: TypeAlias = Callable[..., Any]


@dataclass(slots=True)
class ScopedAcquisitionSession:
    """Own one scoped adapter/binding pair through forget-then-close cleanup."""

    scope_spec: AcquisitionScopeSpec
    binding: object = field(repr=False)
    backend: object = field(repr=False)
    dataset_id: str
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.scope_spec) is not AcquisitionScopeSpec:
            raise TypeError("scoped acquisition session spec differs")
        self.dataset_id = _text(self.dataset_id, "scoped dataset id", 512)
        if not callable(getattr(self.binding, "close", None)):
            raise TypeError("scoped acquisition binding lacks close")
        for method in ("project", "search", "forget_namespace"):
            if not callable(getattr(self.backend, method, None)):
                raise TypeError(f"scoped acquisition backend lacks {method}")

    async def forget_then_close(self, *, opener: AcquisitionOpener) -> None:
        if self._closed:
            raise RunnerInvariantError("scoped acquisition session is already closed")
        forget_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            await self.backend.forget_namespace()  # type: ignore[attr-defined]
        except BaseException as error:
            forget_error = error
        try:
            await self.binding.close()  # type: ignore[attr-defined]
        except BaseException as error:
            close_error = error

        recovery_failures: list[BaseException] = []
        if forget_error is not None:
            recovery: ScopedAcquisitionSession | None = None
            try:
                recovery = await _open_scoped_session(
                    self.scope_spec,
                    opener=opener,
                    expected_dataset_id=self.dataset_id,
                )
            except BaseException as error:
                recovery_failures.append(error)
            if recovery is not None:
                try:
                    await recovery.backend.forget_namespace()  # type: ignore[attr-defined]
                except BaseException as error:
                    recovery_failures.append(error)
                try:
                    await recovery.binding.close()  # type: ignore[attr-defined]
                except BaseException as error:
                    recovery_failures.append(error)
        failures: list[BaseException] = []
        if close_error is not None:
            failures.append(close_error)
        if recovery_failures:
            if forget_error is not None:
                failures.append(forget_error)
            failures.extend(recovery_failures)
        if failures:
            raise BaseExceptionGroup(
                "scoped acquisition cleanup failed",
                failures,
            )
        self._closed = True


async def _open_scoped_session(
    scope_spec: AcquisitionScopeSpec,
    *,
    opener: AcquisitionOpener,
    expected_dataset_id: str | None = None,
) -> ScopedAcquisitionSession:
    if type(scope_spec) is not AcquisitionScopeSpec or not callable(opener):
        raise TypeError("scoped acquisition opener inputs differ")
    value = await _maybe_await(
        opener(scope_spec, expected_dataset_id=expected_dataset_id)
    )
    if type(value) is ScopedAcquisitionSession:
        if value.scope_spec != scope_spec or (
            expected_dataset_id is not None
            and value.dataset_id != expected_dataset_id
        ):
            primary = RunnerInvariantError("opened acquisition session identity differs")
            try:
                await value.forget_then_close(opener=opener)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "opened acquisition session mismatch and cleanup failed",
                    [primary, cleanup],
                ) from None
            raise primary
        return value
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("acquisition opener must return a binding/backend pair")
    binding, backend = value
    try:
        dataset_id = _text(
            getattr(binding, "dataset_id", None),
            "opened acquisition dataset id",
            512,
        )
        if expected_dataset_id is not None and dataset_id != expected_dataset_id:
            raise RunnerInvariantError("reopened acquisition dataset id differs")
        return ScopedAcquisitionSession(scope_spec, binding, backend, dataset_id)
    except BaseException as primary:
        try:
            await binding.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "opened acquisition validation and binding cleanup failed",
                [primary, cleanup],
            ) from None
        raise


@dataclass(frozen=True, slots=True)
class ArmTaskSpec:
    purpose: Purpose
    phase: Phase
    arm: Arm
    task: Mapping[str, object]
    judge_arm: Arm | None
    feedback_value: float | None = None

    def __post_init__(self) -> None:
        if self.purpose not in ("qualification", "evaluation"):
            raise ValueError("arm-task purpose differs")
        if self.phase not in PHASES or self.arm not in EVALUATION_ARMS:
            raise ValueError("arm-task phase or arm is not declared")
        canonical = dict(self.task)
        if type(self.task) is not dict or canonical.get("phase") != self.phase:
            raise ValueError("arm-task public task does not bind its phase")
        _digest(canonical.get("task_id"), "arm-task task_id")
        _digest(canonical.get("public_commitment"), "public task commitment")
        _digest(canonical.get("replicate_commitment"), "replicate commitment")
        scan_for_hidden_material(canonical)
        if self.judge_arm is not None and self.judge_arm not in (
            *EVALUATION_ARMS,
            "QUALIFICATION",
        ):
            raise ValueError("arm-task judge arm differs")
        if self.purpose == "evaluation" and self.judge_arm != self.arm:
            raise ValueError("evaluation attempts must be judged under their exact arm")
        if self.purpose == "qualification" and self.judge_arm not in (
            None,
            "QUALIFICATION",
        ):
            raise ValueError("qualification diagnostics cannot masquerade as eval arms")
        if self.feedback_value is not None and (
            type(self.feedback_value) is not float
            or self.feedback_value not in (0.0, 1.0)
        ):
            raise ValueError("learner-visible feedback must be exact binary scalar")
        if self.arm == "RANDOM_FEEDBACK" and self.phase == "adaptation":
            if self.feedback_value is None:
                raise ValueError("random-feedback adaptation needs its frozen scalar")
        elif self.feedback_value is not None:
            raise ValueError("only random-feedback adaptation accepts injected feedback")

    @property
    def task_id(self) -> str:
        return str(self.task["task_id"])

    @property
    def replicate_commitment(self) -> str:
        return str(self.task["replicate_commitment"])


def _hash_pairs(
    value: Mapping[str, str],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty hash mapping")
    rows = tuple(sorted(value.items()))
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError(f"{label} names must be unique")
    for name, digest in rows:
        _text(name, f"{label} name", 512)
        _raw_digest(digest, f"{label} sha256")
    return rows


@dataclass(frozen=True, slots=True)
class ProbeIntegrityEvidence:
    schema: str
    purpose: Purpose
    phase: Phase
    task_id: str
    arm: Arm
    replicate_commitment: str
    foundation_tensor_digest: str
    foundation_guard_semantics: str
    source_baseline_before: tuple[tuple[str, str], ...] | None
    source_baseline_after: tuple[tuple[str, str], ...] | None
    clone_before: tuple[tuple[str, str], ...] | None
    clone_after: tuple[tuple[str, str], ...] | None
    clone_audit: CloneAudit | None
    learner_genesis_digest: str | None
    learner_parent_digest: str | None
    learner_child_digest: str | None
    learner_sequence: int | None

    def __post_init__(self) -> None:
        if self.schema != PROBE_INTEGRITY_SCHEMA:
            raise ValueError("probe integrity schema differs")
        if self.purpose not in ("qualification", "evaluation"):
            raise ValueError("probe integrity purpose differs")
        if self.phase not in PHASES or self.arm not in EVALUATION_ARMS:
            raise ValueError("probe integrity phase or arm differs")
        _digest(self.task_id, "probe integrity task_id")
        _digest(self.replicate_commitment, "probe replicate commitment")
        _digest(self.foundation_tensor_digest, "probe foundation tensor digest")
        if self.foundation_guard_semantics != FOUNDATION_GUARD_SEMANTICS:
            raise ValueError("probe foundation guard semantics differ")
        for label, rows in (
            ("source baseline before", self.source_baseline_before),
            ("source baseline after", self.source_baseline_after),
            ("clone before", self.clone_before),
            ("clone after", self.clone_after),
        ):
            if rows is not None and (
                type(rows) is not tuple
                or set(dict(rows)) != PROBE_BASELINE_HASH_NAMES
                or _hash_pairs(dict(rows), label) != rows
            ):
                raise ValueError(f"{label} hashes are not canonical")
        disposable = self.phase in ("development", "final") and self.arm != "QWEN_ONLY"
        clone_values = (
            self.source_baseline_before,
            self.source_baseline_after,
            self.clone_before,
            self.clone_after,
            self.clone_audit,
        )
        if disposable:
            if any(value is None for value in clone_values):
                raise ValueError("disposable probe lacks clone/source evidence")
            if type(self.clone_audit) is not CloneAudit:
                raise TypeError("disposable probe clone audit differs")
            source_before = dict(self.source_baseline_before)  # type: ignore[arg-type]
            clone_before = dict(self.clone_before)  # type: ignore[arg-type]
            cloned_source = dict(self.clone_audit.source_hashes)
            cloned_destination = dict(self.clone_audit.destination_hashes)
            if (
                self.source_baseline_before != self.source_baseline_after
                or self.clone_before != self.source_baseline_before
                or any(
                    source_before[name] != digest
                    for name, digest in cloned_source.items()
                )
                or any(
                    clone_before[name] != digest
                    for name, digest in cloned_destination.items()
                )
            ):
                raise RunnerInvariantError("disposable probe clone/source binding differs")
        elif any(value is not None for value in clone_values):
            raise ValueError("adaptation/QWEN_ONLY cannot claim a disposable clone")
        learner_values = (
            self.learner_genesis_digest,
            self.learner_parent_digest,
            self.learner_child_digest,
        )
        for value in learner_values:
            if value is not None:
                _digest(value, "probe learner digest")
        if self.arm in CONTROL_ARMS:
            if any(value is not None for value in learner_values) or self.learner_sequence is not None:
                raise ValueError("control probe cannot claim a learner lineage")
        else:
            if self.learner_genesis_digest is None or self.learner_parent_digest is None:
                raise ValueError("stateful probe lacks genesis or parent lineage")
            if (self.learner_child_digest is None) != (self.learner_sequence is None):
                raise ValueError("learner child and sequence must be present together")
            if (
                self.learner_child_digest is not None
                and self.learner_child_digest == self.learner_parent_digest
            ):
                raise RunnerInvariantError("learner transition did not change state")
            if self.learner_sequence is not None and (
                type(self.learner_sequence) is not int or self.learner_sequence < 1
            ):
                raise ValueError("learner sequence must be a positive exact integer")

    @classmethod
    def create(
        cls,
        *,
        purpose: Purpose,
        phase: Phase,
        task_id: str,
        arm: Arm,
        replicate_commitment: str,
        foundation_tensor_digest: str,
        source_baseline_before: Mapping[str, str] | None = None,
        source_baseline_after: Mapping[str, str] | None = None,
        clone_before: Mapping[str, str] | None = None,
        clone_after: Mapping[str, str] | None = None,
        clone_audit: CloneAudit | None = None,
        learner_genesis_digest: str | None = None,
        learner_parent_digest: str | None = None,
        learner_child_digest: str | None = None,
        learner_sequence: int | None = None,
    ) -> "ProbeIntegrityEvidence":
        return cls(
            schema=PROBE_INTEGRITY_SCHEMA,
            purpose=purpose,
            phase=phase,
            task_id=task_id,
            arm=arm,
            replicate_commitment=replicate_commitment,
            foundation_tensor_digest=foundation_tensor_digest,
            foundation_guard_semantics=FOUNDATION_GUARD_SEMANTICS,
            source_baseline_before=(
                None
                if source_baseline_before is None
                else _hash_pairs(source_baseline_before, "source baseline before")
            ),
            source_baseline_after=(
                None
                if source_baseline_after is None
                else _hash_pairs(source_baseline_after, "source baseline after")
            ),
            clone_before=(
                None if clone_before is None else _hash_pairs(clone_before, "clone before")
            ),
            clone_after=(
                None if clone_after is None else _hash_pairs(clone_after, "clone after")
            ),
            clone_audit=clone_audit,
            learner_genesis_digest=learner_genesis_digest,
            learner_parent_digest=learner_parent_digest,
            learner_child_digest=learner_child_digest,
            learner_sequence=learner_sequence,
        )

    def validate_for(
        self,
        spec: ArmTaskSpec,
        parser_disposition: str,
        expected_foundation_tensor_digest: str,
        expected_learner_genesis_digest: str,
    ) -> None:
        if (
            self.purpose,
            self.phase,
            self.task_id,
            self.arm,
            self.replicate_commitment,
        ) != (
            spec.purpose,
            spec.phase,
            spec.task_id,
            spec.arm,
            spec.replicate_commitment,
        ):
            raise RunnerInvariantError("probe integrity identity differs from schedule")
        if self.foundation_tensor_digest != _digest(
            expected_foundation_tensor_digest,
            "expected foundation tensor digest",
        ):
            raise RunnerInvariantError("probe foundation differs from frozen identity")
        expected_genesis = _digest(
            expected_learner_genesis_digest,
            "expected learner genesis digest",
        )
        if (
            self.arm in STATEFUL_ARMS
            and self.learner_genesis_digest != expected_genesis
        ):
            raise RunnerInvariantError("probe learner genesis differs from frozen identity")
        should_transition = (
            self.arm in STATEFUL_ARMS
            and parser_disposition == "ADMITTED"
            and (
                self.purpose == "evaluation"
                or self.phase == "adaptation"
                or self.arm == "FULL"
            )
        )
        has_transition = self.learner_child_digest is not None
        if should_transition != has_transition:
            raise RunnerInvariantError("probe learner transition applicability differs")

    def to_canonical(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "clone_after": None if self.clone_after is None else dict(self.clone_after),
            "clone_audit": (
                None if self.clone_audit is None else self.clone_audit.to_canonical()
            ),
            "clone_audit_ref": (
                None if self.clone_audit is None else self.clone_audit.audit_ref
            ),
            "clone_before": None if self.clone_before is None else dict(self.clone_before),
            "foundation_guard_semantics": self.foundation_guard_semantics,
            "foundation_tensor_digest": self.foundation_tensor_digest,
            "learner_child_digest": self.learner_child_digest,
            "learner_genesis_digest": self.learner_genesis_digest,
            "learner_parent_digest": self.learner_parent_digest,
            "learner_sequence": self.learner_sequence,
            "phase": self.phase,
            "purpose": self.purpose,
            "replicate_commitment": self.replicate_commitment,
            "schema": self.schema,
            "source_baseline_after": (
                None
                if self.source_baseline_after is None
                else dict(self.source_baseline_after)
            ),
            "source_baseline_before": (
                None
                if self.source_baseline_before is None
                else dict(self.source_baseline_before)
            ),
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class ProbeIntegrityContext:
    """Pre-execution identities and bounded post-execution hash probes."""

    foundation_tensor_digest: str
    foundation_identity_probe: Callable[[], str] = field(
        repr=False,
        compare=False,
    )
    learner_genesis_digest: str | None = None
    source_baseline_before: tuple[tuple[str, str], ...] | None = None
    clone_before: tuple[tuple[str, str], ...] | None = None
    clone_audit: CloneAudit | None = None
    source_baseline_probe: Callable[[], Mapping[str, str]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    clone_probe: Callable[[], Mapping[str, str]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _digest(self.foundation_tensor_digest, "context foundation tensor digest")
        if not callable(self.foundation_identity_probe):
            raise TypeError("context foundation identity probe must be callable")
        if self.learner_genesis_digest is not None:
            _digest(self.learner_genesis_digest, "context learner genesis digest")
        values = (
            self.source_baseline_before,
            self.clone_before,
            self.clone_audit,
            self.source_baseline_probe,
            self.clone_probe,
        )
        if any(value is None for value in values) and any(
            value is not None for value in values
        ):
            raise ValueError("probe integrity clone context is incomplete")
        if self.source_baseline_before is not None:
            source_before = dict(self.source_baseline_before)
            clone_before = dict(self.clone_before)  # type: ignore[arg-type]
            if (
                set(source_before) != PROBE_BASELINE_HASH_NAMES
                or set(clone_before) != PROBE_BASELINE_HASH_NAMES
                or _hash_pairs(source_before, "context source baseline")
                != self.source_baseline_before
                or _hash_pairs(clone_before, "context clone")
                != self.clone_before
                or type(self.clone_audit) is not CloneAudit
                or any(
                    source_before[name] != digest
                    for name, digest in self.clone_audit.source_hashes
                )
                or any(
                    clone_before[name] != digest
                    for name, digest in self.clone_audit.destination_hashes
                )
                or not callable(self.source_baseline_probe)
                or not callable(self.clone_probe)
            ):
                raise RunnerInvariantError("probe integrity clone context differs")

    def finalize(
        self,
        spec: ArmTaskSpec,
        *,
        learner_parent_digest: str | None,
        learner_child_digest: str | None,
        learner_sequence: int | None,
    ) -> ProbeIntegrityEvidence:
        disposable = spec.phase in ("development", "final") and spec.arm != "QWEN_ONLY"
        if disposable != (self.source_baseline_before is not None):
            raise RunnerInvariantError("probe clone context applicability differs")
        observed_foundation = self.foundation_identity_probe()
        if (
            _digest(observed_foundation, "observed probe foundation identity")
            != self.foundation_tensor_digest
        ):
            raise RunnerInvariantError("probe foundation identity changed")
        source_after = (
            None
            if self.source_baseline_probe is None
            else dict(self.source_baseline_probe())
        )
        clone_after = None if self.clone_probe is None else dict(self.clone_probe())
        return ProbeIntegrityEvidence.create(
            purpose=spec.purpose,
            phase=spec.phase,
            task_id=spec.task_id,
            arm=spec.arm,
            replicate_commitment=spec.replicate_commitment,
            foundation_tensor_digest=observed_foundation,
            source_baseline_before=(
                None
                if self.source_baseline_before is None
                else dict(self.source_baseline_before)
            ),
            source_baseline_after=source_after,
            clone_before=None if self.clone_before is None else dict(self.clone_before),
            clone_after=clone_after,
            clone_audit=self.clone_audit,
            learner_genesis_digest=self.learner_genesis_digest,
            learner_parent_digest=learner_parent_digest,
            learner_child_digest=learner_child_digest,
            learner_sequence=learner_sequence,
        )


@dataclass(frozen=True, slots=True)
class ArmTaskResult:
    task_id: str
    arm: Arm
    attempt_receipt_ref: str
    raw_response: str
    parser_disposition: Literal["ADMITTED", "MALFORMED"]
    judgment: Mapping[str, object] | None
    evidence: Mapping[str, object]
    integrity: ProbeIntegrityEvidence
    frozen_recall: FrozenRecallBatch | None = None
    proposal_generation: object | None = field(default=None, repr=False, compare=False)
    execution_request: object | None = field(default=None, repr=False, compare=False)
    execution_receipt: object | None = field(default=None, repr=False, compare=False)
    journal_entry: object | None = field(default=None, repr=False, compare=False)
    learner_pending_material: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _digest(self.task_id, "arm result task_id")
        if self.arm not in EVALUATION_ARMS:
            raise ValueError("arm result arm differs")
        _digest(self.attempt_receipt_ref, "arm result attempt receipt")
        if type(self.raw_response) is not str:
            raise TypeError("arm result raw response must be text")
        if self.parser_disposition not in ("ADMITTED", "MALFORMED"):
            raise ValueError("arm result parser disposition differs")
        if type(self.evidence) is not dict:
            raise TypeError("arm result evidence must be an exact object")
        canonical_json_bytes(dict(self.evidence))
        if type(self.integrity) is not ProbeIntegrityEvidence:
            raise TypeError("arm result integrity must be exact ProbeIntegrityEvidence")
        if self.judgment is not None and type(self.judgment) is not dict:
            raise TypeError("arm result judgment must be an exact object or None")


@dataclass(slots=True)
class ArmRuntime:
    """Injected exact runtime components for one isolated arm-task."""

    proposal_adapter: object
    executor: object
    foundation_tensor_digest: str
    integrity_context: ProbeIntegrityContext
    frozen_recall: FrozenRecallBatch | None = None
    cycle: object | None = None
    backend: object | None = None
    learner: object | None = None
    store: object | None = None
    deferred_recall_provider: CaptureOnceRecallProvider | None = None
    quiesce: Callable[[], Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _digest(self.foundation_tensor_digest, "foundation tensor digest")
        if (
            type(self.integrity_context) is not ProbeIntegrityContext
            or self.integrity_context.foundation_tensor_digest
            != self.foundation_tensor_digest
        ):
            raise RunnerInvariantError("arm runtime integrity context differs")
        if not callable(getattr(self.proposal_adapter, "propose_procedure_traces", None)):
            raise TypeError("arm runtime proposal adapter differs")
        if not callable(getattr(self.executor, "execute", None)):
            raise TypeError("arm runtime executor differs")
        if self.cycle is not None and not all(
            callable(getattr(self.cycle, name, None))
            for name in ("begin_turn", "execute_turn", "record_outcome")
        ):
            raise TypeError("stateful arm runtime cycle differs")
        if self.deferred_recall_provider is not None and (
            type(self.deferred_recall_provider) is not CaptureOnceRecallProvider
            or self.cycle is None
            or self.frozen_recall is not None
        ):
            raise RunnerInvariantError("deferred recall provider applicability differs")
        if self.quiesce is not None and not callable(self.quiesce):
            raise TypeError("arm runtime quiesce callback differs")

    @property
    def backend_search_calls(self) -> int | None:
        if self.backend is None:
            return None
        value = getattr(self.backend, "search_calls", None)
        if type(value) is int and value >= 0:
            return value
        queries = getattr(self.backend, "queries", None)
        if isinstance(queries, (tuple, list)):
            return len(queries)
        return None


def _public_task_object(task: Mapping[str, object]) -> object:
    from experiments.evaluators.high_level_multidomain_v1 import PublicEvaluationTask

    value = dict(task)
    if set(value) != {
        "family",
        "ordinal",
        "payload",
        "phase",
        "public_commitment",
        "replicate",
        "replicate_commitment",
        "schema",
        "task_id",
    } or value["schema"] != SUITE_SCHEMA:
        raise RunnerInvariantError("public task canonical fields differ")
    if type(value["payload"]) is not dict:
        raise TypeError("public task payload must be an exact object")
    return PublicEvaluationTask(
        task_id=value["task_id"],
        replicate=value["replicate"],
        replicate_commitment=value["replicate_commitment"],
        family=value["family"],
        phase=value["phase"],
        ordinal=value["ordinal"],
        payload_json=canonical_json_bytes(value["payload"]).decode("utf-8"),
        public_commitment=value["public_commitment"],
    )


def _canonical_payload(value: object) -> dict[str, object]:
    for name in ("to_payload", "to_canonical"):
        method = getattr(value, name, None)
        if callable(method):
            payload = method()
            if type(payload) is not dict:
                raise TypeError("runtime evidence payload is not an exact object")
            canonical_json_bytes(payload)
            return payload
    if type(value) is dict:
        canonical_json_bytes(value)
        return dict(value)
    raise TypeError("runtime evidence lacks a canonical public payload")


def _canonical_payload_bytes(value: object) -> bytes:
    method = getattr(value, "canonical_bytes", None)
    if callable(method):
        encoded = method()
        if type(encoded) is not bytes or not encoded:
            raise TypeError("runtime canonical bytes differ")
        return encoded
    return canonical_json_bytes(_canonical_payload(value))


def _generation_evidence(generation: object) -> dict[str, object]:
    payload = _canonical_payload(generation)
    generation_ref = getattr(generation, "generation_ref", None)
    _digest(generation_ref, "generation_ref")
    return {"generation_ref": generation_ref, **payload}


def _generation_resource_record(
    generation: object,
    resources: ResourceLedger | None,
) -> None:
    prompt_tokens = getattr(generation, "prompt_tokens", None)
    token_ids = getattr(generation, "generated_token_ids", None)
    if type(prompt_tokens) is not int or type(token_ids) is not tuple:
        raise RunnerInvariantError("generation resource evidence differs")
    if resources is not None:
        resources.record_generation(prompt_tokens, len(token_ids))


def _proposal_request_evidence(
    rendered_task: str,
    recalled_evidence: tuple[str, ...],
    generation: object,
) -> dict[str, object]:
    return {
        "count": PROPOSAL_COUNT,
        "prompt_ref": getattr(generation, "prompt_ref", None),
        "recalled_evidence": list(recalled_evidence),
        "task": rendered_task,
    }


def _runtime_from_factory(value: object, *, stateful: bool) -> ArmRuntime:
    if type(value) is ArmRuntime:
        runtime = value
    elif stateful:
        cycle = value
        runtime = ArmRuntime(
            proposal_adapter=getattr(cycle, "text_adapter", None),
            executor=getattr(cycle, "executor", None),
            foundation_tensor_digest=getattr(cycle, "foundation_tensor_digest", None),
            integrity_context=getattr(cycle, "integrity_context", None),
            cycle=cycle,
            backend=getattr(getattr(cycle, "memory", None), "backend", None),
            learner=getattr(cycle, "learner", None),
            store=getattr(cycle, "transaction_store", None),
        )
    else:
        raise TypeError("control factory must return an exact ArmRuntime")
    if stateful != (runtime.cycle is not None):
        raise RunnerInvariantError("arm runtime statefulness differs from its arm")
    return runtime


def validate_objective_judgment(
    judgment: Mapping[str, object],
    *,
    task_id: str,
    arm: Arm,
    attempt_receipt_ref: str,
    raw_response: str,
) -> dict[str, object]:
    canonical = dict(judgment)
    if set(canonical) != {
        "arm",
        "attempt_receipt_ref",
        "disposition",
        "raw_response",
        "response_commitment",
        "score",
        "task_id",
    } or (
        canonical.get("task_id"),
        canonical.get("arm"),
        canonical.get("attempt_receipt_ref"),
        canonical.get("raw_response"),
    ) != (task_id, arm, attempt_receipt_ref, raw_response):
        raise RunnerInvariantError("objective judgment does not bind the exact attempt")
    expected_response = evaluator_record_digest(
        "response",
        {
            "arm": arm,
            "attempt_receipt_ref": attempt_receipt_ref,
            "raw_response": raw_response,
            "task_id": task_id,
        },
    )
    score = canonical["score"]
    if (
        canonical["response_commitment"] != expected_response
        or type(score) is not float
        or score not in (0.0, 1.0)
        or canonical["disposition"]
        != ("SUCCESS" if score == 1.0 else "UNSUCCESSFUL")
    ):
        raise RunnerInvariantError("objective judgment canonical binding differs")
    scan_for_hidden_material(canonical)
    return canonical


async def execute_arm_task(
    spec: ArmTaskSpec,
    *,
    cycle_factory: Callable[[ArmTaskSpec], Any],
    proposal_adapter_factory: Callable[[ArmTaskSpec], Any],
    evaluator: object | None,
    ledger: EvidenceLedger,
    budget: AttemptBudget,
    resources: ResourceLedger | None = None,
) -> ArmTaskResult:
    """Execute one arm-task with no proposal repair or generation retry."""

    if type(spec) is not ArmTaskSpec:
        raise TypeError("spec must be an exact ArmTaskSpec")
    if type(ledger) is not EvidenceLedger or type(budget) is not AttemptBudget:
        raise TypeError("ledger and budget must be exact runner boundaries")
    stateful = spec.arm in STATEFUL_ARMS
    factory = cycle_factory if stateful else proposal_adapter_factory
    runtime = _runtime_from_factory(
        await _maybe_await(factory(spec)), stateful=stateful
    )
    deferred_recall = (
        spec.arm in ("FULL", "RANDOM_FEEDBACK")
        and runtime.frozen_recall is None
        and type(runtime.deferred_recall_provider) is CaptureOnceRecallProvider
    )
    if spec.arm != "QWEN_ONLY" and runtime.frozen_recall is None and not deferred_recall:
        raise RunnerInvariantError("history arm lacks its frozen canonical recall batch")
    if spec.arm == "QWEN_ONLY" and runtime.frozen_recall is not None:
        raise RunnerInvariantError("QWEN_ONLY unexpectedly received a recall batch")

    from angler.runtime.qwen_cognitive import parse_qwen_procedure_proposals
    from experiments.evaluators.high_level_multidomain_v1 import render_public_task

    public_task = _public_task_object(spec.task)
    rendered_task = render_public_task(public_task)
    public_task_ref = str(spec.task["public_commitment"])
    frozen = runtime.frozen_recall
    recalled_text: tuple[str, ...] = (
        evidence_for_arm("QWEN_ONLY", ())
        if spec.arm == "QWEN_ONLY"
        else ()
        if frozen is None
        else evidence_for_arm(
            spec.arm,
            tuple(item.recalled_text() for item in frozen.items),
        )
    )
    budget.consume_proposal(spec.task_id, spec.arm, spec.phase)
    turn = None
    proposals: tuple[str, ...] = ()
    malformed = False
    try:
        if stateful:
            turn = await _maybe_await(
                runtime.cycle.begin_turn(spec.task_id, rendered_task)  # type: ignore[union-attr]
            )
            capture = (
                runtime.deferred_recall_provider
                if deferred_recall
                else runtime.backend
            )
            if type(capture) is CaptureOnceRecallProvider:
                captured = capture.captured_batch
                if captured is None:
                    raise RunnerInvariantError("cycle search did not capture canonical recall")
                if frozen is None:
                    frozen = captured
                    runtime.frozen_recall = captured
                    recalled_text = evidence_for_arm(
                        spec.arm,
                        tuple(item.recalled_text() for item in captured.items),
                    )
                elif captured != frozen:
                    raise RunnerInvariantError("cycle recall differs from its frozen batch")
            if frozen is None or turn.recalled_refs != tuple(
                item.record_ref for item in frozen.items
            ):
                raise RunnerInvariantError("cycle recalled references differ from evidence")
            proposals = tuple(turn.decision.proposals)
        else:
            proposals = tuple(
                runtime.proposal_adapter.propose_procedure_traces(
                    rendered_task,
                    recalled_text,
                    count=PROPOSAL_COUNT,
                )
            )
    except ValueError as error:
        if deferred_recall:
            captured = runtime.deferred_recall_provider.captured_batch  # type: ignore[union-attr]
            if captured is None:
                raise
            frozen = captured
            runtime.frozen_recall = captured
            recalled_text = evidence_for_arm(
                spec.arm,
                tuple(item.recalled_text() for item in captured.items),
            )
        generation = getattr(runtime.proposal_adapter, "last_proposal_generation", None)
        if generation is None:
            raise
        try:
            parse_qwen_procedure_proposals(
                generation.response,
                count=PROPOSAL_COUNT,
            )
        except ValueError:
            malformed = True
        else:
            raise error.with_traceback(error.__traceback__)
    proposal_generation = getattr(
        runtime.proposal_adapter, "last_proposal_generation", None
    )
    if proposal_generation is None:
        raise RunnerInvariantError("proposal generation was not retained before parsing")
    _generation_resource_record(proposal_generation, resources)
    proposal_evidence = _generation_evidence(proposal_generation)
    proposal_request = _proposal_request_evidence(
        rendered_task,
        recalled_text,
        proposal_generation,
    )

    selection: dict[str, object] | None = None
    execution_request = None
    execution_receipt = None
    task_generation = None
    journal_entry = None
    completed = None
    pending_material = (
        None
        if turn is None or runtime.learner is None
        else runtime.learner.pending_prospective_material
    )
    raw_response = ""
    if malformed:
        source_attempt_receipt_ref = proposal_generation.generation_ref
    else:
        if len(proposals) != PROPOSAL_COUNT or len(set(proposals)) != PROPOSAL_COUNT:
            raise RunnerInvariantError("admitted proposals differ from exact cardinality")
        budget.consume_execution(spec.task_id, spec.arm)
        if stateful:
            if turn is None:
                raise RunnerInvariantError("stateful admitted proposal has no turn")
            selection = {
                "decision_evidence_ref": turn.decision.evidence_ref,
                "reservation_ref": turn.reservation.reservation_ref,
                "selected_index": turn.decision.selected_index,
                "selected_trace": turn.decision.selected_trace,
            }
            completed = await _maybe_await(
                runtime.cycle.execute_turn(  # type: ignore[union-attr]
                    turn,
                    observations=("No hidden or external observation is available.",),
                )
            )
            execution_request = completed.execution_request
            execution_receipt = completed.execution_receipt
            journal_entry = runtime.executor.journal.get(
                turn.reservation.idempotency_key
            )
        else:
            from angler.cognition.prospective import CognitiveExecutionRequest

            control = ControlSelectionReceipt.create(
                arm=spec.arm,  # type: ignore[arg-type]
                proposal_generation_ref=proposal_generation.generation_ref,
                proposals=proposals,
                task_id=spec.task_id,
            )
            selection = {"selection_ref": control.receipt_ref, **control.to_canonical()}
            execution_request = CognitiveExecutionRequest(
                reservation_ref=control.receipt_ref,
                commitment_ref=control.receipt_ref,
                idempotency_key=control.receipt_ref,
                task_id=spec.task_id,
                request=rendered_task,
                selected_trace=control.selected_trace,
                input_observations=(
                    "No hidden or external observation is available.",
                ),
            )
            runtime.executor.execute(execution_request)
            journal_entry = runtime.executor.journal.get(control.receipt_ref)
            if journal_entry is None:
                raise RunnerInvariantError("control execution was not journaled")
            execution_receipt = journal_entry.receipt
        if journal_entry is None or execution_receipt is None or execution_request is None:
            raise RunnerInvariantError("admitted execution evidence is incomplete")
        task_generation = journal_entry.generation
        _generation_resource_record(task_generation, resources)
        raw_response = execution_receipt.response
        source_attempt_receipt_ref = execution_receipt.execution_receipt_ref
    attempt_receipt = attempt_receipt_payload(
        purpose=spec.purpose,
        phase=spec.phase,
        task_id=spec.task_id,
        arm=spec.arm,
        parser_disposition="MALFORMED" if malformed else "ADMITTED",
        source_receipt_ref=source_attempt_receipt_ref,
        source_kind=(
            "PROPOSAL_GENERATION" if malformed else "EXECUTION_RECEIPT"
        ),
    )
    attempt_receipt_ref = attempt_receipt_reference(attempt_receipt)

    budget.finish_arm_task(spec.task_id, spec.arm)
    recall_ref = None if frozen is None else frozen.batch_ref
    feedback_kind = (
        "RANDOM_FEEDBACK_SCHEDULE"
        if spec.arm == "RANDOM_FEEDBACK" and spec.phase == "adaptation"
        else "OBJECTIVE_EVALUATOR"
    )
    feedback_source_ref = content_ref(
        "feedback-source",
        {
            "arm": spec.arm,
            "kind": feedback_kind,
            "phase": spec.phase,
            "task_id": spec.task_id,
        },
    )
    feedback_source = {"kind": feedback_kind, "source_ref": feedback_source_ref}
    learner_parent = None
    if turn is not None:
        learner_parent = turn.decision.parent_state_digest
    elif runtime.learner is not None:
        learner_parent = runtime.learner.state_digest()

    def payload_with_ref(value: object, attribute: str) -> dict[str, object]:
        payload = _canonical_payload(value)
        reference = getattr(value, attribute, None)
        _digest(reference, attribute)
        return {attribute: reference, **payload}

    execution_request_payload = (
        None
        if execution_request is None
        else payload_with_ref(execution_request, "execution_request_ref")
    )
    execution_receipt_payload = (
        None
        if execution_receipt is None
        else payload_with_ref(execution_receipt, "execution_receipt_ref")
    )
    task_generation_payload = (
        None if task_generation is None else _generation_evidence(task_generation)
    )
    resource_snapshot = budget.detailed_snapshot()
    if resources is not None:
        resource_snapshot = {**resource_snapshot, "measured": resources.snapshot()}
    stage = {
        "arm": spec.arm,
        "attempt_receipt": attempt_receipt,
        "attempt_receipt_ref": attempt_receipt_ref,
        "execution_receipt": execution_receipt_payload,
        "execution_request": execution_request_payload,
        "feedback_source": feedback_source,
        "foundation_tensor_digest": runtime.foundation_tensor_digest,
        "frozen_recall_ref": recall_ref,
        "learner_parent_digest": learner_parent,
        "judge_arm": spec.judge_arm,
        "parser_disposition": "MALFORMED" if malformed else "ADMITTED",
        "phase": spec.phase,
        "proposal_generation": proposal_evidence,
        "proposal_request": proposal_request,
        "proposals": list(proposals),
        "public_task_ref": public_task_ref,
        "purpose": spec.purpose,
        "raw_response": raw_response,
        "resource_counters": resource_snapshot,
        "schema": ATTEMPT_STAGE_SCHEMA,
        "selection": selection,
        "task_id": spec.task_id,
        "task_response_generation": task_generation_payload,
    }
    stage_ref = ledger.stage_attempt(stage)

    judgment: dict[str, object] | None = None
    if spec.judge_arm is not None:
        if evaluator is None or not callable(getattr(evaluator, "judge_response", None)):
            raise TypeError("judged arm-task requires an evaluator client")
        observed = await _maybe_await(
            evaluator.judge_response(
                spec.task_id,
                spec.judge_arm,
                attempt_receipt_ref,
                raw_response,
            )
        )
        if hasattr(observed, "to_canonical"):
            observed = observed.to_canonical()
        if type(observed) is not dict:
            raise RunnerInvariantError("evaluator judgment is not canonical")
        judgment = validate_objective_judgment(
            observed,
            task_id=spec.task_id,
            arm=spec.judge_arm,
            attempt_receipt_ref=attempt_receipt_ref,
            raw_response=raw_response,
        )

    feedback_record: dict[str, object] | None = None
    learner_transition: dict[str, object] | None = None
    unevaluated_resolution: dict[str, object] | None = None
    should_update = stateful and not malformed and (
        judgment is not None or spec.feedback_value is not None
    )
    if should_update:
        scalar = (
            spec.feedback_value
            if spec.feedback_value is not None
            else judgment.get("score")  # type: ignore[union-attr]
        )
        if type(scalar) is not float or scalar not in (0.0, 1.0):
            raise RunnerInvariantError("learner-visible feedback scalar differs")
        outcome = "success" if scalar == 1.0 else "failure"
        committed = await _maybe_await(
            runtime.cycle.record_outcome(  # type: ignore[union-attr]
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=spec.task_id,
                outcome=outcome,
                feedback_text=(
                    "The bounded objective source returned its declared binary disposition."
                ),
                feedback_source_ref=feedback_source_ref,
            )
        )
        child_digest = runtime.learner.state_digest()
        feedback_record = {
            "feedback_source_ref": feedback_source_ref,
            "learner_visible_scalar": scalar,
            "outcome": outcome,
        }
        learner_transition = {
            "child_state_digest": child_digest,
            "episode_ref": committed.episode_ref,
            "parent_state_digest": learner_parent,
            "projection_pending": committed.projection_pending,
            "sequence": committed.sequence,
        }
    elif stateful and not malformed:
        if completed is None or turn is None:
            raise RunnerInvariantError("unevaluated stateful execution is incomplete")
        resolved = await _maybe_await(
            runtime.cycle.resolve_completed_unevaluated(  # type: ignore[union-attr]
                completed,
                reservation_ref=turn.reservation.reservation_ref,
            )
        )
        unevaluated_resolution = {
            "acquisition_ref": _digest(
                getattr(resolved, "acquisition_ref", None),
                "unevaluated acquisition ref",
            ),
            "projection_pending": getattr(resolved, "projection_pending", None),
            "resolution_ref": _digest(
                getattr(resolved, "resolution_ref", None),
                "unevaluated resolution ref",
            ),
        }
        if type(unevaluated_resolution["projection_pending"]) is not bool:
            raise RunnerInvariantError("unevaluated projection disposition differs")

    runtime_quiescence: dict[str, object] | None = None
    if runtime.quiesce is not None:
        observed_quiescence = await _maybe_await(runtime.quiesce())
        if observed_quiescence is not None:
            if not isinstance(observed_quiescence, Mapping):
                raise TypeError("runtime quiescence evidence must be a mapping")
            runtime_quiescence = dict(observed_quiescence)
            canonical_json_bytes(runtime_quiescence)

    integrity = runtime.integrity_context.finalize(
        spec,
        learner_parent_digest=learner_parent,
        learner_child_digest=(
            None
            if learner_transition is None
            else learner_transition["child_state_digest"]  # type: ignore[arg-type]
        ),
        learner_sequence=(
            None
            if learner_transition is None
            else learner_transition["sequence"]  # type: ignore[arg-type]
        ),
    )
    integrity_ref = content_ref("probe-integrity", integrity.to_canonical())
    final = {
        "arm": spec.arm,
        "attempt_receipt_ref": attempt_receipt_ref,
        "feedback_record": feedback_record,
        "learner_transition": learner_transition,
        "objective_judgment": judgment,
        "probe_integrity": integrity.to_canonical(),
        "probe_integrity_ref": integrity_ref,
        "resource_counters": (
            budget.detailed_snapshot()
            if resources is None
            else {**budget.detailed_snapshot(), "measured": resources.snapshot()}
        ),
        "runtime_quiescence": runtime_quiescence,
        "runtime_quiescence_ref": (
            None
            if runtime_quiescence is None
            else content_ref("runtime-quiescence", runtime_quiescence)
        ),
        "schema": ATTEMPT_FINAL_SCHEMA,
        "task_id": spec.task_id,
        "unevaluated_resolution": unevaluated_resolution,
    }
    final_ref = ledger.finalize_attempt(
        (spec.task_id, spec.arm, attempt_receipt_ref), final
    )
    frozen_refs = [] if frozen is None else [item.record_ref for item in frozen.items]
    frozen_bytes = (
        [] if frozen is None else [item.record_bytes_sha256 for item in frozen.items]
    )
    proposal_request_bytes = canonical_json_bytes(proposal_request)
    proposal_generation_bytes = canonical_json_bytes(proposal_evidence)
    selection_bytes = b"" if selection is None else canonical_json_bytes(selection)
    request_bytes = (
        b"" if execution_request is None else _canonical_payload_bytes(execution_request)
    )
    receipt_bytes = (
        b"" if execution_receipt is None else _canonical_payload_bytes(execution_receipt)
    )
    task_generation_bytes = (
        b"" if task_generation is None else canonical_json_bytes(task_generation_payload)
    )
    evidence = {
        "backend_raw_hit_count": getattr(runtime.backend, "raw_hit_count", 0),
        "backend_rejected_hit_count": getattr(
            runtime.backend,
            "rejected_hit_count",
            0,
        ),
        "backend_search_calls": runtime.backend_search_calls,
        "evidence_final_ref": final_ref,
        "evidence_stage_ref": stage_ref,
        "execution_prompt_ref": (
            None if task_generation is None else task_generation.prompt_ref
        ),
        "execution_receipt_bytes": receipt_bytes.hex(),
        "execution_receipt_ref": (
            None
            if execution_receipt is None
            else execution_receipt.execution_receipt_ref
        ),
        "execution_request_bytes": request_bytes.hex(),
        "execution_request_ref": (
            None
            if execution_request is None
            else execution_request.execution_request_ref
        ),
        "frozen_recall_ref": recall_ref,
        "proposal_generation_bytes": proposal_generation_bytes.hex(),
        "proposal_generation_ref": proposal_generation.generation_ref,
        "proposal_prompt_ref": proposal_generation.prompt_ref,
        "proposal_request_bytes": proposal_request_bytes.hex(),
        "probe_integrity_ref": integrity_ref,
        "proposals": list(proposals),
        "public_task_ref": public_task_ref,
        "raw_response": raw_response,
        "recalled_record_bytes_sha256": frozen_bytes,
        "recalled_record_refs": frozen_refs,
        "runtime_quiescence_ref": (
            None
            if runtime_quiescence is None
            else content_ref("runtime-quiescence", runtime_quiescence)
        ),
        "score": None if judgment is None else judgment["score"],
        "selection_bytes": selection_bytes.hex(),
        "selection_ref": (
            None
            if selection is None
            else selection.get("selection_ref", selection.get("reservation_ref"))
        ),
        "selected_trace": None if selection is None else selection["selected_trace"],
        "task_id": spec.task_id,
        "task_response_generation_bytes": task_generation_bytes.hex(),
        "task_response_generation_ref": (
            None if task_generation is None else task_generation.generation_ref
        ),
    }
    return ArmTaskResult(
        task_id=spec.task_id,
        arm=spec.arm,
        attempt_receipt_ref=attempt_receipt_ref,
        raw_response=raw_response,
        parser_disposition="MALFORMED" if malformed else "ADMITTED",
        judgment=judgment,
        evidence=evidence,
        integrity=integrity,
        frozen_recall=frozen,
        proposal_generation=proposal_generation,
        execution_request=execution_request,
        execution_receipt=execution_receipt,
        journal_entry=journal_entry,
        learner_pending_material=pending_material,
    )


_CYCLE_AGENT_REF = _fixture_ref("frozen-qwen-cognee-cycle-v1-agent")
_CYCLE_WORLD_REF = _fixture_ref("frozen-qwen-cognee-cycle-v1-world")
_CYCLE_SUBJECT_REF = _fixture_ref("frozen-qwen-cognee-cycle-v1-subject")
_CYCLE_SCOPE_REF = _fixture_ref("frozen-qwen-cognee-cycle-v1-scope")
_CYCLE_BOOTSTRAP_REFS = (
    _fixture_ref("frozen-qwen-cognee-cycle-v1-bootstrap"),
)


def _restore_clone_learner(
    paths: ClonePaths,
    *,
    genesis: FreshGenesisBundle,
    prospective_lesion: bool,
) -> tuple[object, object]:
    from angler.runtime.cognitive_transaction_store import CognitiveTransactionStore

    snapshot = _read_private_artifact(
        paths.learner,
        maximum_bytes=MAXIMUM_PENDING_SNAPSHOT_BYTES,
    )
    store = CognitiveTransactionStore(paths.store)
    head = store.audit_integrity()
    if (
        head is None
        or store.load_head_state() != snapshot
        or head.snapshot_sha256 != "sha256:" + hashlib.sha256(snapshot).hexdigest()
    ):
        raise RunnerInvariantError("clone learner artifact and store head differ")
    learner = restore_fresh_cpu_genesis(
        genesis,
        prospective_lesion=prospective_lesion,
    )
    learner.restore_state(snapshot)
    if learner.capture_state() != snapshot or learner.state_digest() != head.state_digest:
        raise RunnerInvariantError("clone learner restore differs before cycle construction")
    if (
        learner.pending_decision is not None
        or learner.pending_prospective_material is not None
    ):
        raise RunnerInvariantError("restored clone learner has pending material")
    return learner, store


def _admit_recall_hits(
    memory: object,
    hits: Sequence[object],
    limit: int,
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[str, ...]]:
    from angler.memory.cognitive_acquisition_graph import AcquisitionReferenceHit

    recall = memory.recall_from_hits(hits, limit=limit)  # type: ignore[attr-defined]
    items = tuple(getattr(recall, "items", ()))
    rejected = tuple(getattr(recall, "rejected", ()))
    raw_hits = tuple(hits)
    if not items:
        raise RunnerInvariantError("canonical recall admitted no captured hit")
    item_refs = tuple(getattr(item, "artifact_ref", None) for item in items)
    if (
        len(set(item_refs)) != len(item_refs)
        or any(type(value) is not str for value in item_refs)
    ):
        raise RunnerInvariantError("canonical recall item references differ")
    hit_by_ref: dict[str, object] = {}
    for hit in raw_hits:
        record_ref = getattr(hit, "record_ref", None)
        if type(record_ref) is str and record_ref not in hit_by_ref:
            hit_by_ref[record_ref] = hit
    try:
        raw_admitted = tuple(hit_by_ref[record_ref] for record_ref in item_refs)
    except KeyError as error:
        raise RunnerInvariantError(
            "canonical recall item does not rejoin its raw hit"
        ) from error
    if tuple(getattr(hit, "record_ref", None) for hit in raw_admitted) != item_refs:
        raise RunnerInvariantError("canonical recall admission order differs")
    admitted_hits = tuple(
        AcquisitionReferenceHit(
            record_ref=item_ref,
            adjacent_record_refs=tuple(
                getattr(record, "record_ref", None)
                for record in getattr(item, "adjacent_records", ())
            ),
            score=getattr(raw_hit, "score", None),
            backend_ref=getattr(raw_hit, "backend_ref", None),
        )
        for item_ref, item, raw_hit in zip(
            item_refs,
            items,
            raw_admitted,
            strict=True,
        )
    )
    records = tuple(getattr(item, "record", None) for item in items)
    if any(record is None for record in records):
        raise RunnerInvariantError("canonical recall omitted resolved records")
    return admitted_hits, records, rejected


def _validate_replayed_batch(
    memory: object,
    batch: FrozenRecallBatch,
) -> None:
    hits = batch.runtime_hits()
    admitted, records, rejected = _admit_recall_hits(
        memory,
        hits,
        MAXIMUM_RECALL_ITEMS,
    )
    if admitted != hits or rejected:
        raise RunnerInvariantError("frozen admitted hits did not replay exactly")
    replayed = FrozenRecallBatch.from_hits(
        hits,
        records,
        search_calls=batch.search_calls,
    )
    if replayed != batch:
        raise RunnerInvariantError("replayed recall bytes differ from frozen FULL batch")


def _remove_disposable_runtime_root(root: Path, parent: Path) -> None:
    if (
        not root.is_absolute()
        or root.parent != parent
        or parent.is_symlink()
        or not parent.is_dir()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise RunnerInvariantError("disposable runtime root identity differs")
    shutil.rmtree(root)
    _fsync_directory(parent)


@dataclass(slots=True)
class ManagedArmRuntime:
    """Own one arm runtime, its namespace, and any disposable physical clone."""

    factory: "HighLevelArmFactory" = field(repr=False)
    spec: ArmTaskSpec
    runtime: ArmRuntime
    source_owner: LineageBaselineOwner | None = field(default=None, repr=False)
    session: ScopedAcquisitionSession | None = field(default=None, repr=False)
    scope_key: str | None = None
    lineage_key: tuple[str, str] | None = None
    disposable_root: Path | None = None
    disposable_parent: Path | None = None
    recall_provider: CaptureOnceRecallProvider | ReplayOnlyRecallProvider | None = field(
        default=None,
        repr=False,
    )
    namespace_rebuild_calls: int = 0
    persistent_lineage: bool = False
    disposition_ref: str | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _poisoned: bool = field(default=False, init=False, repr=False)

    async def close(
        self,
        *,
        result: ArmTaskResult | None,
        ledger: EvidenceLedger,
    ) -> None:
        if self._closed:
            state = "terminally poisoned" if self._poisoned else "already closed"
            raise RunnerInvariantError(f"managed arm runtime is {state}")
        if type(ledger) is not EvidenceLedger:
            raise TypeError("managed arm close requires its exact evidence ledger")
        failures: list[BaseException] = []
        integrity_safe = result is not None
        disposition: dict[str, object] | None = None
        if result is not None:
            try:
                if (result.task_id, result.arm) != (
                    self.spec.task_id,
                    self.spec.arm,
                ):
                    raise RunnerInvariantError("managed result identity differs")
                if self.disposable_root is not None and result.arm != "QWEN_ONLY":
                    clone_audit = result.integrity.clone_audit
                    if (
                        type(clone_audit) is not CloneAudit
                        or clone_audit.root != str(self.disposable_root)
                    ):
                        raise RunnerInvariantError("managed clone root binding differs")
                final_ref = _digest(
                    result.evidence.get("evidence_final_ref"),
                    "managed final evidence ref",
                )
                expected_integrity_ref = content_ref(
                    "probe-integrity",
                    result.integrity.to_canonical(),
                )
                if result.evidence.get("probe_integrity_ref") != expected_integrity_ref:
                    raise RunnerInvariantError("managed probe integrity ref differs")
                final_rows = tuple(
                    row
                    for row in ledger.read_all()
                    if (
                        row["attempt"].get("task_id"),
                        row["attempt"].get("arm"),
                    )
                    == (result.task_id, result.arm)
                )
                if len(final_rows) != 1:
                    raise RunnerInvariantError("managed final ledger row differs")
                final_record = final_rows[0]["judgment"]
                if (
                    type(final_record) is not dict
                    or content_ref("attempt-judgment", final_record) != final_ref
                    or final_record.get("probe_integrity")
                    != result.integrity.to_canonical()
                    or final_record.get("probe_integrity_ref")
                    != expected_integrity_ref
                ):
                    raise RunnerInvariantError(
                        "managed result differs from its final ledger record"
                    )
                disposition = {
                    "arm": result.arm,
                    "backend_raw_hit_count": result.evidence.get(
                        "backend_raw_hit_count"
                    ),
                    "backend_rejected_hit_count": result.evidence.get(
                        "backend_rejected_hit_count"
                    ),
                    "backend_search_calls": result.evidence.get(
                        "backend_search_calls"
                    ),
                    "disposable_root": (
                        None
                        if self.disposable_root is None
                        else str(self.disposable_root)
                    ),
                    "evidence_final_ref": final_ref,
                    "namespace_rebuild_calls": self.namespace_rebuild_calls,
                    "phase": self.spec.phase,
                    "probe_integrity_ref": expected_integrity_ref,
                    "purpose": self.spec.purpose,
                    "schema": MANAGED_RUNTIME_DISPOSITION_SCHEMA,
                    "task_id": result.task_id,
                }
                if self.persistent_lineage:
                    if self.source_owner is None or self.runtime.cycle is None:
                        raise RunnerInvariantError("persistent lineage owner differs")
                    hashes = self.source_owner.persist_and_audit(
                        cycle=self.runtime.cycle,
                    )
                    disposition["lineage_hashes"] = hashes
                    disposition["lineage_scope_ref"] = (
                        self.source_owner.scope_binding_ref
                    )
                elif self.disposable_root is not None:
                    if result.integrity.clone_after is not None:
                        clone = ClonePaths(
                            root=self.disposable_root,
                            store=self.disposable_root / "cognitive.sqlite3",
                            journal=(
                                self.disposable_root
                                / "qwen-execution-journal.sqlite3"
                            ),
                            learner=self.disposable_root / "learner-state.bin",
                        )
                        store = self.runtime.store
                        journal = getattr(self.runtime.executor, "journal", None)
                        if store is None or journal is None:
                            raise RunnerInvariantError(
                                "managed clone lacks its exact open owners"
                            )
                        observed = _closed_baseline_hashes(
                            clone,
                            store=store,
                            journal=journal,
                        )
                        if observed != dict(result.integrity.clone_after):
                            raise RunnerInvariantError(
                                "managed clone bytes differ from durable integrity"
                            )
                        disposition["clone_hashes"] = observed
                    elif result.arm == "QWEN_ONLY":
                        journal_path = self.disposable_root / "qwen-journal.sqlite3"
                        digest, size = sha256_file(journal_path)
                        quiescence_ref = final_record.get(
                            "runtime_quiescence_ref"
                        )
                        quiescence = _validate_qwen_journal_quiescence(
                            final_record.get("runtime_quiescence"),
                            quiescence_ref,
                        )
                        if (
                            result.evidence.get("runtime_quiescence_ref")
                            != quiescence_ref
                            or (
                                quiescence["journal_sha256"],
                                quiescence["journal_bytes"],
                            )
                            != (digest, size)
                        ):
                            raise RunnerInvariantError(
                                "QWEN_ONLY journal changed after finalization"
                            )
                        disposition["journal_bytes"] = size
                        disposition["journal_sha256"] = digest
                        disposition["runtime_quiescence_ref"] = quiescence_ref
                    else:
                        raise RunnerInvariantError(
                            "disposable runtime lacks durable byte evidence"
                        )
            except BaseException as error:
                integrity_safe = False
                failures.append(error)
        session_present = self.session is not None
        namespace: dict[str, object] | None = None
        if self.session is not None:
            namespace = {
                "dataset_id": self.session.dataset_id,
                "dataset_name": self.session.scope_spec.dataset_name,
                "node_set_name": self.session.scope_spec.node_set_name,
                "state_root": self.session.scope_spec.state_root,
                "tenant_name": self.session.scope_spec.tenant_name,
            }
            namespace["scope_ref"] = content_ref(
                "managed-acquisition-scope",
                namespace,
            )
        session_clean = self.session is None
        if self.session is not None:
            try:
                await self.session.forget_then_close(
                    opener=self.factory.acquisition_opener,
                )
                session_clean = True
                self.session = None
            except BaseException as error:
                failures.append(error)
        provider_present = self.recall_provider is not None
        provider_kind: str | None = None
        if type(self.recall_provider) is CaptureOnceRecallProvider:
            provider_kind = "CAPTURE_ONCE"
        elif type(self.recall_provider) is ReplayOnlyRecallProvider:
            provider_kind = "REPLAY_ONLY"
        elif self.recall_provider is not None:
            failures.append(RunnerInvariantError("managed recall provider differs"))
        provider_clean = self.recall_provider is None
        if self.recall_provider is not None:
            try:
                self.recall_provider.dispose()
                provider_clean = True
                self.recall_provider = None
            except BaseException as error:
                failures.append(error)
        scope_released = self.scope_key is None
        if session_clean and self.scope_key is not None:
            try:
                self.factory._release_scope(self.scope_key)
                self.scope_key = None
                scope_released = True
            except BaseException as error:
                failures.append(error)
        if disposition is not None:
            disposition.update(
                {
                    "binding_close_completed": (
                        None if not session_present else session_clean
                    ),
                    "namespace": namespace,
                    "namespace_forget_completed": (
                        None if not session_present else session_clean
                    ),
                    "recall_provider_disposed": (
                        None if not provider_present else provider_clean
                    ),
                    "recall_provider_kind": provider_kind,
                    "scope_lease_released": (
                        None if not session_present else scope_released
                    ),
                }
            )
        if (
            integrity_safe
            and session_clean
            and provider_clean
            and scope_released
            and not failures
            and disposition is not None
        ):
            try:
                self.disposition_ref = self.factory._persist_managed_disposition(
                    self.spec,
                    disposition,
                )
            except BaseException as error:
                failures.append(error)
        if (
            integrity_safe
            and session_clean
            and provider_clean
            and scope_released
            and not failures
            and self.disposition_ref is not None
            and self.disposable_root is not None
            and self.disposable_parent is not None
        ):
            try:
                _remove_disposable_runtime_root(
                    self.disposable_root,
                    self.disposable_parent,
                )
            except BaseException as error:
                failures.append(error)
        if (
            integrity_safe
            and session_clean
            and provider_clean
            and scope_released
            and not failures
            and self.disposition_ref is not None
            and self.lineage_key is not None
        ):
            try:
                self.factory._release_lineage(self.lineage_key)
                self.lineage_key = None
            except BaseException as error:
                failures.append(error)
        if failures:
            self._poisoned = True
            self._closed = True
            raise BaseExceptionGroup("managed arm cleanup failed", failures)
        if result is None:
            self._poisoned = True
        self._closed = True


@dataclass(slots=True)
class HighLevelArmFactory:
    """Injected, one-run owner for persistent lineages and disposable arm clones."""

    root: Path
    purpose: Purpose
    genesis: FreshGenesisBundle
    io: object = field(repr=False)
    manifest: object = field(repr=False)
    foundation_guard: FoundationGuard = field(repr=False)
    acquisition_opener: AcquisitionOpener = field(repr=False)
    lineages: dict[tuple[str, str], LineageBaselineOwner] = field(repr=False)
    lineages_parent: Path
    clones_parent: Path
    controls_parent: Path
    dispositions_parent: Path
    scopes_parent: Path
    _active_scopes: set[str] = field(default_factory=set, init=False, repr=False)
    _active_lineages: set[tuple[str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _comparison_dataset_ids: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _full_recall: dict[tuple[str, str], FrozenRecallBatch] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _consumed: set[tuple[str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _dispositions: dict[tuple[str, str], tuple[Path, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.purpose not in ("qualification", "evaluation"):
            raise ValueError("arm factory purpose differs")
        if (
            not self.root.is_absolute()
            or self.root.is_symlink()
            or not self.root.is_dir()
            or type(self.genesis) is not FreshGenesisBundle
            or type(self.foundation_guard) is not FoundationGuard
            or not callable(self.acquisition_opener)
        ):
            raise RunnerInvariantError("arm factory construction differs")
        expected_local_parents = {
            self.lineages_parent,
            self.clones_parent,
            self.controls_parent,
            self.dispositions_parent,
        }
        if any(
            path.parent != self.root
            or path.is_symlink()
            or not path.is_dir()
            or stat.S_IMODE(path.stat().st_mode) != 0o700
            for path in expected_local_parents
        ):
            raise RunnerInvariantError("arm factory private parents differ")
        if (
            not self.scopes_parent.is_absolute()
            or self.scopes_parent.is_symlink()
            or not self.scopes_parent.is_dir()
            or stat.S_IMODE(self.scopes_parent.stat().st_mode) != 0o700
        ):
            raise RunnerInvariantError("arm factory scope parent differs")

    @classmethod
    async def create(
        cls,
        root: str | Path,
        *,
        purpose: Purpose,
        replicate_commitments: Sequence[str],
        genesis: FreshGenesisBundle,
        io: object,
        manifest: object,
        foundation_guard: FoundationGuard,
        acquisition_opener: AcquisitionOpener,
        scope_parent: str | Path | None = None,
    ) -> "HighLevelArmFactory":
        target = _create_fresh_private_root(root)
        lineages_parent = target / "lineages"
        clones_parent = target / "clones"
        controls_parent = target / "controls"
        dispositions_parent = target / "dispositions"
        scopes_parent = (
            target / "scopes" if scope_parent is None else Path(scope_parent)
        )
        for path in (
            lineages_parent,
            clones_parent,
            controls_parent,
            dispositions_parent,
        ):
            _mkdir_private(path)
        if scope_parent is None:
            _mkdir_private(scopes_parent)
        else:
            _create_fresh_private_root(scopes_parent)
        lineages = await create_fresh_lineage_baselines(
            lineages_parent,
            purpose=purpose,
            replicate_commitments=replicate_commitments,
            genesis=genesis,
            manifest=manifest,
            scope_parent=scopes_parent,
            acquisition_opener=acquisition_opener,
        )
        return cls(
            root=target,
            purpose=purpose,
            genesis=genesis,
            io=io,
            manifest=manifest,
            foundation_guard=foundation_guard,
            acquisition_opener=acquisition_opener,
            lineages=lineages,
            lineages_parent=lineages_parent,
            clones_parent=clones_parent,
            controls_parent=controls_parent,
            dispositions_parent=dispositions_parent,
            scopes_parent=scopes_parent,
        )

    def _claim_scope(self, scope_spec: AcquisitionScopeSpec) -> str:
        key = scope_spec.state_root
        if key in self._active_scopes:
            raise RunnerInvariantError("shared acquisition scope lifetime overlaps")
        self._active_scopes.add(key)
        return key

    def _release_scope(self, key: str) -> None:
        if key not in self._active_scopes:
            raise RunnerInvariantError("acquisition scope release differs")
        self._active_scopes.remove(key)

    def _claim_lineage(self, key: tuple[str, str]) -> None:
        if key in self._active_lineages:
            raise RunnerInvariantError("lineage baseline lifetime overlaps")
        self._active_lineages.add(key)

    def _release_lineage(self, key: tuple[str, str]) -> None:
        if key not in self._active_lineages:
            raise RunnerInvariantError("lineage baseline release differs")
        self._active_lineages.remove(key)

    def _persist_managed_disposition(
        self,
        spec: ArmTaskSpec,
        disposition: Mapping[str, object],
    ) -> str:
        if type(spec) is not ArmTaskSpec or type(disposition) is not dict:
            raise TypeError("managed disposition inputs differ")
        canonical = dict(disposition)
        if (
            canonical.get("schema") != MANAGED_RUNTIME_DISPOSITION_SCHEMA
            or (
                canonical.get("purpose"),
                canonical.get("phase"),
                canonical.get("task_id"),
                canonical.get("arm"),
            )
            != (spec.purpose, spec.phase, spec.task_id, spec.arm)
        ):
            raise RunnerInvariantError("managed disposition identity differs")
        key = (spec.task_id, spec.arm)
        if key not in self._consumed or key in self._dispositions:
            raise RunnerInvariantError("managed disposition identity is not one-use")
        disposition_ref = content_ref("managed-runtime-disposition", canonical)
        envelope = {
            "disposition": canonical,
            "disposition_ref": disposition_ref,
        }
        encoded = canonical_json_bytes(envelope)
        if len(encoded) > MAXIMUM_EVIDENCE_RECORD_BYTES:
            raise RunnerInvariantError("managed disposition exceeds its byte ceiling")
        slug = hashlib.sha256(
            canonical_json_bytes({"arm": spec.arm, "task_id": spec.task_id})
        ).hexdigest()
        path = self.dispositions_parent / f"{slug}.json"
        _write_private_artifact(
            path,
            encoded,
            maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
        )
        _fsync_directory(self.dispositions_parent)
        observed = decode_canonical_json(
            _read_private_artifact(
                path,
                maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
            ),
            maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
        )
        if observed != envelope:
            raise RunnerInvariantError("managed disposition durable bytes differ")
        self._dispositions[key] = (path, disposition_ref)
        return disposition_ref

    def _owner_for(self, spec: ArmTaskSpec) -> LineageBaselineOwner:
        lineage_arm = "RANDOM_FEEDBACK" if spec.arm == "RANDOM_FEEDBACK" else "FULL"
        key = (spec.replicate_commitment, lineage_arm)
        owner = self.lineages.get(key)
        if owner is None:
            raise RunnerInvariantError("arm task has no exact lineage baseline")
        return owner

    async def _open_owned_scope(
        self,
        scope_spec: AcquisitionScopeSpec,
        *,
        expected_dataset_id: str | None,
    ) -> tuple[ScopedAcquisitionSession, str]:
        key = self._claim_scope(scope_spec)
        try:
            session = await _open_scoped_session(
                scope_spec,
                opener=self.acquisition_opener,
                expected_dataset_id=expected_dataset_id,
            )
        except BaseException:
            self._release_scope(key)
            raise
        observed = self._comparison_dataset_ids.get(key)
        if observed is None:
            self._comparison_dataset_ids[key] = session.dataset_id
        elif observed != session.dataset_id:
            primary = RunnerInvariantError(
                "shared acquisition dataset identity drifted"
            )
            cleanup: BaseException | None = None
            try:
                await session.forget_then_close(opener=self.acquisition_opener)
            except BaseException as error:
                cleanup = error
            if cleanup is None:
                self._release_scope(key)
                raise primary
            raise BaseExceptionGroup(
                "shared acquisition drift and cleanup failed",
                [primary, cleanup],
            )
        return session, key

    def _clone_owner(
        self,
        spec: ArmTaskSpec,
        owner: LineageBaselineOwner,
    ) -> tuple[ClonePaths, CloneAudit, dict[str, str], dict[str, str]]:
        source_hashes = owner.persist_and_audit()
        clone = ClonePaths.for_task(self.clones_parent, spec.task_id, spec.arm)
        audit = clone_closed_baseline(
            owner.paths,
            clone,
            {name: source_hashes[name] for name in CLONED_BASELINE_HASH_NAMES},
        )
        clone_hashes = {
            "acquisition_sequence": source_hashes["acquisition_sequence"],
            **dict(audit.destination_hashes),
        }
        if clone_hashes != source_hashes:
            raise RunnerInvariantError("disposable clone four-way hashes differ")
        return clone, audit, source_hashes, clone_hashes

    def _integrity_context(
        self,
        *,
        owner: LineageBaselineOwner | None,
        clone: ClonePaths | None,
        clone_audit: CloneAudit | None,
        source_hashes: Mapping[str, str] | None,
        clone_hashes: Mapping[str, str] | None,
        learner: object | None,
        store: object | None,
        journal: object | None,
        cycle_box: dict[str, object] | None,
    ) -> ProbeIntegrityContext:
        common = {
            "foundation_tensor_digest": self.foundation_guard.initial_tensor_digest,
            "foundation_identity_probe": self.foundation_guard.probe,
            "learner_genesis_digest": (
                None if learner is None else QUALIFIED_INITIAL_COMPETENCE_DIGEST
            ),
        }
        if clone is None:
            return ProbeIntegrityContext(**common)
        if (
            owner is None
            or clone_audit is None
            or source_hashes is None
            or clone_hashes is None
        ):
            raise RunnerInvariantError("clone integrity construction is incomplete")

        def clone_probe() -> Mapping[str, str]:
            if learner is not None:
                if store is None or journal is None or cycle_box is None:
                    raise RunnerInvariantError("stateful clone probe lacks its owner")
                cycle = cycle_box.get("cycle")
                if cycle is None:
                    raise RunnerInvariantError("stateful clone probe lacks its cycle")
                _assert_successor_quiescent(
                    learner=learner,
                    store=store,
                    journal=journal,
                    cycle=cycle,
                )
                snapshot = learner.capture_state()  # type: ignore[attr-defined]
                _replace_private_artifact(
                    clone.learner,
                    snapshot,
                    maximum_bytes=MAXIMUM_PENDING_SNAPSHOT_BYTES,
                )
            return _closed_baseline_hashes(
                clone,
                store=store,
                journal=journal,
            )

        return ProbeIntegrityContext(
            **common,
            source_baseline_before=_hash_pairs(source_hashes, "source baseline"),
            clone_before=_hash_pairs(clone_hashes, "clone baseline"),
            clone_audit=clone_audit,
            source_baseline_probe=owner.hashes,
            clone_probe=clone_probe,
        )

    @staticmethod
    def _rendered_task(spec: ArmTaskSpec) -> str:
        from experiments.evaluators.high_level_multidomain_v1 import render_public_task

        return render_public_task(_public_task_object(spec.task))

    def _full_batch_for(self, spec: ArmTaskSpec) -> FrozenRecallBatch:
        batch = self._full_recall.get((spec.replicate_commitment, spec.task_id))
        if batch is None:
            raise RunnerInvariantError("comparison arm preceded its exact FULL recall")
        return batch

    async def _build_stateful(self, spec: ArmTaskSpec) -> ManagedArmRuntime:
        from angler.memory.cognitive_acquisition_graph import AcquisitionSituatedMemory
        from angler.runtime.cognitive_cycle import CognitiveCycle
        from angler.runtime.prospective_observation import (
            QwenReceiptObservedStateEncoderV1,
        )
        from angler.runtime.qwen_cognitive import (
            FrozenQwenCognitiveExecutorV1,
            FrozenQwenProcedureAdapterV1,
            FrozenQwenProcedureRelationAdapterV1,
            SQLiteQwenExecutionJournal,
        )

        expected_batch = (
            None
            if spec.arm in ("FULL", "RANDOM_FEEDBACK")
            else self._full_batch_for(spec)
        )
        owner = self._owner_for(spec)
        lineage_key = (owner.replicate_commitment, owner.arm)
        self._claim_lineage(lineage_key)
        persistent = spec.phase == "adaptation"
        if persistent and spec.arm not in ("FULL", "RANDOM_FEEDBACK"):
            self._release_lineage(lineage_key)
            raise RunnerInvariantError("only learned comparison lineages adapt persistently")
        clone: ClonePaths | None = None
        clone_audit: CloneAudit | None = None
        source_hashes: dict[str, str] | None = None
        clone_hashes: dict[str, str] | None = None
        try:
            if persistent:
                owner.persist_and_audit()
                learner = owner.learner
                store = owner.store
                journal = owner.journal
                paths = owner.paths
                scope_spec = owner.scope_spec
                expected_dataset_id = owner.dataset_id
            else:
                clone, clone_audit, source_hashes, clone_hashes = self._clone_owner(
                    spec,
                    owner,
                )
                learner, store = _restore_clone_learner(
                    clone,
                    genesis=self.genesis,
                    prospective_lesion=spec.arm == "PROSPECTIVE_REMOVAL",
                )
                journal = SQLiteQwenExecutionJournal(clone.journal)
                paths = clone
                scope_spec = AcquisitionScopeSpec.for_clone(
                    run_label=spec.purpose,
                    replicate=spec.replicate_commitment,
                    arm=spec.arm,
                    task_id=spec.task_id,
                    state_parent=self.scopes_parent,
                )
                expected_dataset_id = self._comparison_dataset_ids.get(
                    scope_spec.state_root
                )
            session, scope_key = await self._open_owned_scope(
                scope_spec,
                expected_dataset_id=expected_dataset_id,
            )
        except BaseException:
            self._release_lineage(lineage_key)
            raise

        rendered_task = self._rendered_task(spec)
        capture: CaptureOnceRecallProvider | None = None
        if spec.arm == "BACKEND_REMOVAL":
            backend: object = ReplayOnlyRecallProvider(session.backend)
        else:
            capture = CaptureOnceRecallProvider(
                session.backend,
                expected_query=rendered_task,
                expected_limit=MAXIMUM_RECALL_ITEMS,
                expected_batch=expected_batch,
            )
            backend = capture
        try:
            memory = AcquisitionSituatedMemory(source=store, backend=backend)
            if capture is not None:
                capture.bind_resolver(
                    lambda hits, limit: _admit_recall_hits(
                        memory,
                        hits,
                        limit,
                    )
                )
            await memory.rebuild(limit=MAXIMUM_PROJECTION_RETRY)
            memory.assert_synchronized(store.acquisition_head())
            if store.pending_acquisition_projections(limit=1):
                raise RunnerInvariantError("stateful task began with a pending outbox")
            if expected_batch is not None and spec.arm == "BACKEND_REMOVAL":
                _validate_replayed_batch(memory, expected_batch)

            text_adapter = FrozenQwenProcedureAdapterV1(self.io, self.manifest)
            relation_adapter = FrozenQwenProcedureRelationAdapterV1(
                self.io,
                self.manifest,
            )
            executor = FrozenQwenCognitiveExecutorV1(
                self.io,
                self.manifest,
                journal,
            )
            observation = QwenReceiptObservedStateEncoderV1(
                model_ref=self.manifest.model_ref,
                encoder_ref=self.manifest.encoder_ref,
                manifest_ref=self.manifest.manifest_ref,
                latent_width=self.manifest.latent_width,
            )
            cycle_box: dict[str, object] = {}
            integrity = self._integrity_context(
                owner=owner if not persistent else None,
                clone=clone,
                clone_audit=clone_audit,
                source_hashes=source_hashes,
                clone_hashes=clone_hashes,
                learner=learner,
                store=store,
                journal=journal,
                cycle_box=cycle_box,
            )
            removal_condition = (
                spec.arm
                if spec.arm
                in (
                    "FULL",
                    "FROZEN_ORIGIN",
                    "PROSPECTIVE_REMOVAL",
                    "BACKEND_REMOVAL",
                )
                else "FULL"
            )
            cycle = CognitiveCycle(
                text_adapter=text_adapter,
                relation_adapter=relation_adapter,
                learner=learner,
                memory=memory,
                executor=executor,
                transaction_store=store,
                model_ref=self.manifest.model_ref,
                encoder_ref=self.manifest.encoder_ref,
                agent_ref=_CYCLE_AGENT_REF,
                world_ref=_CYCLE_WORLD_REF,
                recall_limit=MAXIMUM_RECALL_ITEMS,
                proposal_count=PROPOSAL_COUNT,
                successor_mode=True,
                reality_mode="SIMULATED",
                subject_ref=_CYCLE_SUBJECT_REF,
                scope_ref=_CYCLE_SCOPE_REF,
                bootstrap_evidence_refs=_CYCLE_BOOTSTRAP_REFS,
                observation_encoder=observation,
                removal_condition=removal_condition,
                frozen_recall_hits=(
                    () if expected_batch is None else expected_batch.runtime_hits()
                ),
            )
            cycle_box["cycle"] = cycle

            async def quiesce() -> None:
                await _quiesce_successor_runtime(
                    cycle=cycle,
                    learner=learner,
                    store=store,
                    journal=journal,
                    memory=memory,
                )
                if persistent:
                    owner.persist_and_audit(cycle=cycle)

            runtime = ArmRuntime(
                proposal_adapter=text_adapter,
                executor=executor,
                foundation_tensor_digest=self.foundation_guard.initial_tensor_digest,
                integrity_context=integrity,
                frozen_recall=expected_batch,
                cycle=cycle,
                backend=backend,
                learner=learner,
                store=store,
                deferred_recall_provider=(
                    capture if expected_batch is None else None
                ),
                quiesce=quiesce,
            )
        except BaseException as primary:
            cleanup: BaseException | None = None
            try:
                await session.forget_then_close(opener=self.acquisition_opener)
            except BaseException as error:
                cleanup = error
            if cleanup is None:
                self._release_scope(scope_key)
            self._release_lineage(lineage_key)
            if primary is not None and cleanup is not None:
                raise BaseExceptionGroup(
                    "stateful runtime construction and cleanup failed",
                    [primary, cleanup],
                )
            raise
        return ManagedArmRuntime(
            factory=self,
            spec=spec,
            runtime=runtime,
            source_owner=owner,
            session=session,
            scope_key=scope_key,
            lineage_key=lineage_key,
            disposable_root=None if clone is None else clone.root,
            disposable_parent=None if clone is None else self.clones_parent,
            recall_provider=backend,
            namespace_rebuild_calls=1,
            persistent_lineage=persistent,
        )

    async def _build_control(self, spec: ArmTaskSpec) -> ManagedArmRuntime:
        from angler.memory.cognitive_acquisition_graph import AcquisitionSituatedMemory
        from angler.runtime.qwen_cognitive import (
            FrozenQwenCognitiveExecutorV1,
            FrozenQwenProcedureAdapterV1,
            SQLiteQwenExecutionJournal,
        )

        text_adapter = FrozenQwenProcedureAdapterV1(self.io, self.manifest)
        if spec.arm == "QWEN_ONLY":
            slug = hashlib.sha256(
                f"{spec.task_id}\x00{spec.arm}".encode("ascii")
            ).hexdigest()
            control_root = _create_fresh_private_root(self.controls_parent / slug)
            journal = SQLiteQwenExecutionJournal(control_root / "qwen-journal.sqlite3")
            executor = FrozenQwenCognitiveExecutorV1(
                self.io,
                self.manifest,
                journal,
            )

            async def qwen_quiesce() -> Mapping[str, object]:
                from angler.runtime.qwen_cognitive import (
                    parse_qwen_procedure_proposals,
                )

                journal.audit_integrity()
                generation = text_adapter.last_proposal_generation
                if generation is None:
                    raise RunnerInvariantError("QWEN_ONLY lacks proposal generation")
                try:
                    parse_qwen_procedure_proposals(
                        generation.response,
                        count=PROPOSAL_COUNT,
                    )
                except ValueError:
                    expected_entries = 0
                else:
                    expected_entries = 1
                with sqlite3.connect(journal.path) as connection:
                    observed_entries = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM qwen_execution_journal"
                        ).fetchone()[0]
                    )
                if observed_entries != expected_entries:
                    raise RunnerInvariantError("QWEN_ONLY journal entry count differs")
                digest, size = sha256_file(journal.path)
                return {
                    "entries": observed_entries,
                    "journal_bytes": size,
                    "journal_sha256": digest,
                    "kind": QWEN_JOURNAL_QUIESCENCE_KIND,
                }

            runtime = ArmRuntime(
                proposal_adapter=text_adapter,
                executor=executor,
                foundation_tensor_digest=self.foundation_guard.initial_tensor_digest,
                integrity_context=self._integrity_context(
                    owner=None,
                    clone=None,
                    clone_audit=None,
                    source_hashes=None,
                    clone_hashes=None,
                    learner=None,
                    store=None,
                    journal=None,
                    cycle_box=None,
                ),
                quiesce=qwen_quiesce,
            )
            return ManagedArmRuntime(
                factory=self,
                spec=spec,
                runtime=runtime,
                disposable_root=control_root,
                disposable_parent=self.controls_parent,
            )

        frozen = self._full_batch_for(spec)
        owner = self._owner_for(spec)
        lineage_key = (owner.replicate_commitment, owner.arm)
        self._claim_lineage(lineage_key)
        try:
            clone, clone_audit, source_hashes, clone_hashes = self._clone_owner(
                spec,
                owner,
            )
            _learner, store = _restore_clone_learner(
                clone,
                genesis=self.genesis,
                prospective_lesion=False,
            )
            journal = SQLiteQwenExecutionJournal(clone.journal)
            scope_spec = AcquisitionScopeSpec.for_clone(
                run_label=spec.purpose,
                replicate=spec.replicate_commitment,
                arm=spec.arm,
                task_id=spec.task_id,
                state_parent=self.scopes_parent,
            )
            session, scope_key = await self._open_owned_scope(
                scope_spec,
                expected_dataset_id=self._comparison_dataset_ids.get(
                    scope_spec.state_root
                ),
            )
        except BaseException:
            self._release_lineage(lineage_key)
            raise
        backend = ReplayOnlyRecallProvider(session.backend)
        try:
            memory = AcquisitionSituatedMemory(source=store, backend=backend)
            await memory.rebuild(limit=MAXIMUM_PROJECTION_RETRY)
            memory.assert_synchronized(store.acquisition_head())
            _validate_replayed_batch(memory, frozen)
            executor = FrozenQwenCognitiveExecutorV1(
                self.io,
                self.manifest,
                journal,
            )
            runtime = ArmRuntime(
                proposal_adapter=text_adapter,
                executor=executor,
                foundation_tensor_digest=self.foundation_guard.initial_tensor_digest,
                integrity_context=self._integrity_context(
                    owner=owner,
                    clone=clone,
                    clone_audit=clone_audit,
                    source_hashes=source_hashes,
                    clone_hashes=clone_hashes,
                    learner=None,
                    store=store,
                    journal=journal,
                    cycle_box=None,
                ),
                frozen_recall=frozen,
                backend=backend,
                store=store,
            )
        except BaseException as primary:
            cleanup: BaseException | None = None
            try:
                await session.forget_then_close(opener=self.acquisition_opener)
            except BaseException as error:
                cleanup = error
            if cleanup is None:
                self._release_scope(scope_key)
            self._release_lineage(lineage_key)
            if cleanup is not None:
                raise BaseExceptionGroup(
                    "retrieval runtime construction and cleanup failed",
                    [primary, cleanup],
                )
            raise
        return ManagedArmRuntime(
            factory=self,
            spec=spec,
            runtime=runtime,
            source_owner=owner,
            session=session,
            scope_key=scope_key,
            lineage_key=lineage_key,
            disposable_root=clone.root,
            disposable_parent=self.clones_parent,
            recall_provider=backend,
            namespace_rebuild_calls=1,
        )

    async def _build(self, spec: ArmTaskSpec) -> ManagedArmRuntime:
        if type(spec) is not ArmTaskSpec or spec.purpose != self.purpose:
            raise RunnerInvariantError("arm factory task purpose differs")
        key = (spec.task_id, spec.arm)
        if key in self._consumed:
            raise RunnerInvariantError("arm factory task/arm identity is already consumed")
        self._consumed.add(key)
        if spec.arm in STATEFUL_ARMS:
            return await self._build_stateful(spec)
        return await self._build_control(spec)

    async def execute(
        self,
        spec: ArmTaskSpec,
        *,
        evaluator: object | None,
        ledger: EvidenceLedger,
        budget: AttemptBudget,
        resources: ResourceLedger | None = None,
        before_managed_cleanup: Callable[[], Any] | None = None,
    ) -> ArmTaskResult:
        if before_managed_cleanup is not None and not callable(
            before_managed_cleanup
        ):
            raise TypeError("pre-cleanup observation boundary must be callable")
        managed = await self._build(spec)
        result: ArmTaskResult | None = None
        primary: BaseException | None = None
        try:
            result = await execute_arm_task(
                spec,
                cycle_factory=lambda _spec: managed.runtime,
                proposal_adapter_factory=lambda _spec: managed.runtime,
                evaluator=evaluator,
                ledger=ledger,
                budget=budget,
                resources=resources,
            )
            if spec.arm == "FULL" and spec.phase in ("development", "final"):
                if result.frozen_recall is None:
                    raise RunnerInvariantError("FULL comparison recall was not captured")
                cache_key = (spec.replicate_commitment, spec.task_id)
                if cache_key in self._full_recall:
                    raise RunnerInvariantError("FULL comparison recall was reused")
                self._full_recall[cache_key] = result.frozen_recall
        except BaseException as error:
            primary = error
        observation: BaseException | None = None
        if before_managed_cleanup is not None:
            try:
                await _maybe_await(before_managed_cleanup())
            except BaseException as error:
                observation = error
        cleanup: BaseException | None = None
        try:
            await managed.close(result=result, ledger=ledger)
        except BaseException as error:
            cleanup = error
        failures = tuple(
            error
            for error in (primary, observation, cleanup)
            if error is not None
        )
        if len(failures) > 1:
            raise BaseExceptionGroup(
                "arm execution, observation, or managed cleanup failed",
                list(failures),
            )
        if primary is not None:
            raise primary.with_traceback(primary.__traceback__)
        if observation is not None:
            raise observation.with_traceback(observation.__traceback__)
        if cleanup is not None:
            raise cleanup.with_traceback(cleanup.__traceback__)
        if result is None:
            raise RunnerInvariantError("arm execution returned no result")
        return result

    def audit_lineages(self) -> dict[str, object]:
        if self._active_scopes or self._active_lineages:
            raise RunnerInvariantError("arm factory still has active owned lifetimes")
        if set(self._dispositions) != self._consumed:
            raise RunnerInvariantError("arm factory disposition set differs from consumption")
        records = [
            {
                "arm": owner.arm,
                "audit": owner.audit_binding(),
                "replicate_commitment": owner.replicate_commitment,
            }
            for owner in self.lineages.values()
        ]
        records.sort(key=lambda row: (row["replicate_commitment"], row["arm"]))
        dispositions: list[dict[str, object]] = []
        for key in sorted(self._dispositions):
            path, expected_ref = self._dispositions[key]
            observed = decode_canonical_json(
                _read_private_artifact(
                    path,
                    maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
                ),
                maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
            )
            if type(observed) is not dict or set(observed) != {
                "disposition",
                "disposition_ref",
            }:
                raise RunnerInvariantError("managed disposition envelope differs")
            disposition = observed["disposition"]
            if (
                type(disposition) is not dict
                or observed["disposition_ref"] != expected_ref
                or content_ref("managed-runtime-disposition", disposition)
                != expected_ref
                or (disposition.get("task_id"), disposition.get("arm")) != key
            ):
                raise RunnerInvariantError("managed disposition audit differs")
            disposable_root = disposition.get("disposable_root")
            if disposable_root is not None and (
                type(disposable_root) is not str or Path(disposable_root).exists()
            ):
                raise RunnerInvariantError("managed disposable root was not removed")
            dispositions.append(observed)
        result = {
            "lineages": records,
            "managed_dispositions": dispositions,
            "purpose": self.purpose,
        }
        canonical_json_bytes(result)
        return result


_LIVE_FOUNDATION_HASH_NAMES = frozenset(
    {"foundation_tensor", "model_root", "tokenizer"}
)


def _cleanup_unverified_loaded_foundation(torch_module: object) -> None:
    """Best-effort bounded release when owner validation rejects after load."""

    failures: list[BaseException] = []
    try:
        gc.collect()
    except BaseException:
        failures.append(CleanupFailure("QWEN_UNLOAD_FAILED", "LIVE_OWNER_CLEANUP"))
    cuda = getattr(torch_module, "cuda", None)
    synchronize = getattr(cuda, "synchronize", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if not callable(synchronize) or not callable(empty_cache):
        failures.append(CleanupFailure("QWEN_UNLOAD_FAILED", "LIVE_OWNER_CLEANUP"))
    else:
        for operation, arguments in (
            (synchronize, (0,)),
            (empty_cache, ()),
        ):
            try:
                operation(*arguments)
            except BaseException:
                failures.append(
                    CleanupFailure("QWEN_UNLOAD_FAILED", "LIVE_OWNER_CLEANUP")
                )
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("unverified Qwen cleanup failed", failures)


@dataclass(slots=True)
class _LiveFoundationOwner:
    loaded: LoadedQwenFoundation | None = field(repr=False)
    model_root: Path
    torch_module: object = field(repr=False)
    file_verifier: Callable[[Path], Mapping[str, object]] = field(repr=False)
    _probe_calls: int = field(default=0, init=False, repr=False)
    _boundary_verified: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.loaded) is not LoadedQwenFoundation:
            raise TypeError("live foundation owner requires an exact loaded foundation")
        if not self.model_root.is_absolute() or not callable(self.file_verifier):
            raise ValueError("live foundation file boundary differs")
        self._validated_file_evidence(self.loaded.production_file_evidence)

    @staticmethod
    def _validated_file_evidence(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise RunnerInvariantError("live foundation lacks production file evidence")
        evidence = dict(value)
        if set(evidence) != {
            "fastembed_cache_tree",
            "file_count",
            "model_cache_tree",
            "root_manifest_sha256",
            "schema",
            "tiktoken_cache_tree",
            "tokenizer_manifest_sha256",
            "total_bytes",
        } or (
            evidence["schema"] != FOUNDATION_LOAD_SCHEMA
            or evidence["file_count"] != QUALIFIED_MODEL_FILE_COUNT
            or evidence["total_bytes"] != QUALIFIED_MODEL_TOTAL_BYTES
            or evidence["root_manifest_sha256"] != QUALIFIED_MODEL_ROOT_SHA256
            or evidence["tokenizer_manifest_sha256"] != QUALIFIED_TOKENIZER_SHA256
        ):
            raise RunnerInvariantError("live foundation production file evidence differs")
        model_cache = _validated_tree_manifest(
            evidence["model_cache_tree"],
            "qualified model cache",
        )
        fastembed = _validated_tree_manifest(
            evidence["fastembed_cache_tree"],
            "qualified FastEmbed cache",
        )
        tiktoken = _validated_tree_manifest(
            evidence["tiktoken_cache_tree"],
            "qualified Tiktoken cache",
        )
        if (
            fastembed != EXPECTED_FASTEMBED_CACHE_TREE
            or tiktoken != EXPECTED_TIKTOKEN_CACHE_TREE
        ):
            raise RunnerInvariantError("live foundation cache evidence differs")
        evidence["model_cache_tree"] = model_cache
        evidence["fastembed_cache_tree"] = fastembed
        evidence["tiktoken_cache_tree"] = tiktoken
        return evidence

    def _hashes(self, *, final: bool) -> dict[str, str]:
        loaded = self.loaded
        if loaded is None or self._closed:
            raise RunnerInvariantError("live foundation owner is unavailable")
        before = self._validated_file_evidence(loaded.production_file_evidence)
        if final:
            after = self._validated_file_evidence(self.file_verifier(self.model_root))
            if after != before:
                raise RunnerInvariantError("qualified model files changed during the run")
            tensor = loaded.guard.verify_boundary_digest()
            self._boundary_verified = True
        else:
            tensor = loaded.guard.probe()
        hashes = {
            "foundation_tensor": tensor.removeprefix("sha256:"),
            "model_root": str(before["root_manifest_sha256"]),
            "tokenizer": str(before["tokenizer_manifest_sha256"]),
        }
        if set(hashes) != _LIVE_FOUNDATION_HASH_NAMES:
            raise RunnerInvariantError("live foundation hash names differ")
        validate_foundation_hashes(hashes, hashes, hashes)
        return hashes

    @property
    def expected_hashes(self) -> dict[str, str]:
        return self._hashes(final=False)

    @property
    def tensor_digest(self) -> str:
        if self.loaded is None:
            raise RunnerInvariantError("live foundation owner is unavailable")
        return self.loaded.guard.initial_tensor_digest

    def probe(self) -> dict[str, str]:
        if self._probe_calls == 0:
            result = self._hashes(final=False)
        elif self._probe_calls == 1:
            result = self._hashes(final=True)
        else:
            raise RunnerInvariantError("whole-run foundation probe count differs")
        self._probe_calls += 1
        return result

    def close(self) -> None:
        if self._closed:
            return
        loaded = self.loaded
        if loaded is None:
            raise RunnerInvariantError("live foundation ownership differs")
        failures: list[BaseException] = []
        if not self._boundary_verified:
            try:
                self._hashes(final=True)
            except BaseException as error:
                _clear_exception_tracebacks(error)
                failures.append(error)
        self.loaded = None
        loaded = None
        try:
            gc.collect()
        except BaseException:
            failures.append(
                CleanupFailure(
                    "QWEN_UNLOAD_FAILED",
                    "LIVE_OWNER_CLEANUP",
                )
            )
        cuda = getattr(self.torch_module, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if not callable(synchronize) or not callable(empty_cache):
            failures.append(
                CleanupFailure(
                    "QWEN_UNLOAD_FAILED",
                    "LIVE_OWNER_CLEANUP",
                )
            )
        else:
            for operation, arguments in (
                (synchronize, (0,)),
                (empty_cache, ()),
            ):
                try:
                    operation(*arguments)
                except BaseException:
                    failures.append(
                        CleanupFailure(
                            "QWEN_UNLOAD_FAILED",
                            "LIVE_OWNER_CLEANUP",
                        )
                    )
        self._closed = True
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("live foundation cleanup failed", failures)


@dataclass(slots=True)
class _LiveArmExecutorOwner:
    factory: HighLevelArmFactory | None = field(repr=False)
    evaluator: object = field(repr=False)
    ledger: EvidenceLedger = field(repr=False)
    accounting: RunAccounting
    sampler: _ProductionResourceSampler = field(repr=False)
    audit_path: Path
    _finalized: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    async def __call__(self, spec: ArmTaskSpec) -> ArmTaskResult:
        if self._closed or self.factory is None:
            raise RunnerInvariantError("live arm executor owner is unavailable")
        result: ArmTaskResult | None = None
        primary: BaseException | None = None
        try:
            result = await self.factory.execute(
                spec,
                evaluator=self.evaluator,
                ledger=self.ledger,
                budget=self.accounting.budget,
                resources=self.accounting.resources,
                before_managed_cleanup=self.sampler.sample,
            )
        except BaseException as error:
            primary = error
        sample_error: BaseException | None = None
        try:
            self.sampler.sample()
        except BaseException as error:
            sample_error = error
        if primary is not None and sample_error is not None:
            raise BaseExceptionGroup(
                "live arm execution and resource sampling failed",
                [primary, sample_error],
            ) from None
        if primary is not None:
            raise primary.with_traceback(primary.__traceback__)
        if sample_error is not None:
            raise sample_error.with_traceback(sample_error.__traceback__)
        if result is None:
            raise RunnerInvariantError("live arm executor returned no result")
        return result

    def finalize_evidence(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        factory = self.factory
        if factory is None:
            raise RunnerInvariantError("live arm factory ownership differs")
        self.sampler.sample()
        self.ledger.audit()
        audit = factory.audit_lineages()
        ledger_sha256, ledger_bytes = sha256_file(self.ledger.path)
        envelope = {
            "factory_audit": audit,
            "factory_audit_ref": content_ref("live-arm-factory-audit", audit),
            "ledger_bytes": ledger_bytes,
            "ledger_sha256": ledger_sha256,
            "run_intent_ref": self.ledger.run_intent_ref,
        }
        _write_private_artifact(
            self.audit_path,
            canonical_json_bytes(envelope),
            maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
        )
        _fsync_directory(self.audit_path.parent)
        # The terminal sample occurs after the final audit bytes are durable so
        # the result's accounting binds every live-owner output.
        self.sampler.sample()

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        if not self._finalized:
            try:
                self.finalize_evidence()
            except BaseException as error:
                failures.append(error)
        self.factory = None
        self._closed = True
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("live arm factory cleanup failed", failures)


ArmExecutor: TypeAlias = Callable[[ArmTaskSpec], Any]


@dataclass(frozen=True, slots=True)
class OrchestrationDependencies:
    evaluator: object
    execute_arm: ArmExecutor
    manifest: Mapping[str, object]
    expected_manifest_sha256: str
    foundation_probe: Callable[[], Mapping[str, str]]
    expected_foundation_hashes: Mapping[str, str]
    expected_foundation_tensor_digest: str
    expected_learner_genesis_digests: Mapping[str, str]
    accounting: RunAccounting
    repository_root: Path = REPOSITORY_ROOT

    def __post_init__(self) -> None:
        if not callable(self.execute_arm):
            raise TypeError("execute_arm must be callable")
        if not callable(self.foundation_probe):
            raise TypeError("foundation_probe must be callable")
        if type(self.accounting) is not RunAccounting:
            raise TypeError("accounting must be an exact shared RunAccounting")
        validate_foundation_hashes(
            self.expected_foundation_hashes,
            self.expected_foundation_hashes,
            self.expected_foundation_hashes,
        )
        _digest(
            self.expected_foundation_tensor_digest,
            "expected foundation tensor digest",
        )
        if self.expected_foundation_hashes.get(FOUNDATION_TENSOR_HASH_NAME) != (
            self.expected_foundation_tensor_digest.removeprefix("sha256:")
        ):
            raise RunnerInvariantError(
                "whole-run foundation map does not bind the expected tensor digest"
            )
        if not isinstance(self.expected_learner_genesis_digests, Mapping):
            raise TypeError("expected learner genesis identities must be a mapping")
        expected_geneses = dict(self.expected_learner_genesis_digests)
        expected_replicates = 1 if self.accounting.mode is RunMode.QUALIFICATION else 2
        if len(expected_geneses) != expected_replicates:
            raise RunnerInvariantError("expected learner genesis cardinality differs")
        for replicate_commitment, genesis_digest in expected_geneses.items():
            _digest(replicate_commitment, "expected genesis replicate commitment")
            _digest(genesis_digest, "expected learner genesis digest")
        if len(set(expected_geneses.values())) != 1:
            raise RunnerInvariantError("fresh replicate genesis bytes are not identical")
        if self.accounting.mode is RunMode.EVALUATION and set(
            expected_geneses
        ) != set(self.manifest.get("seed_commitments", ())):
            raise RunnerInvariantError(
                "expected learner genesis replicates differ from manifest"
            )
        object.__setattr__(
            self,
            "expected_foundation_hashes",
            MappingProxyType(dict(sorted(self.expected_foundation_hashes.items()))),
        )
        object.__setattr__(
            self,
            "expected_learner_genesis_digests",
            MappingProxyType(dict(sorted(expected_geneses.items()))),
        )
        _raw_digest(self.expected_manifest_sha256, "expected manifest sha256")
        if not isinstance(self.repository_root, Path) or not self.repository_root.is_absolute():
            raise ValueError("repository_root must be an absolute Path")


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


@dataclass(slots=True)
class _PartialConstructionCleanup:
    """One-shot rollback registrations owned until a factory returns an owner."""

    _entries: list[tuple[str, Callable[[], Any]]] = field(default_factory=list)
    _finished: bool = False

    def register(self, code: str, cleanup: Callable[[], Any]) -> None:
        if self._finished:
            raise RunnerInvariantError("partial cleanup registry is finished")
        canonical_code = _canonical_identifier(code, "partial cleanup code")
        if not callable(cleanup):
            raise TypeError("partial construction cleanup must be callable")
        self._entries.append((canonical_code, cleanup))

    def transfer(self) -> tuple[tuple[str, Callable[[], Any]], ...]:
        if self._finished:
            raise RunnerInvariantError("partial cleanup registry is finished")
        self._finished = True
        entries = tuple(self._entries)
        self._entries.clear()
        return entries

    async def rollback(self) -> None:
        if self._finished:
            return
        self._finished = True
        failures: list[BaseException] = []
        for code, cleanup in reversed(self._entries):
            try:
                await _maybe_await(cleanup())
            except BaseException:
                failures.append(
                    CleanupFailure(code, "LIVE_CONSTRUCTION_CLEANUP")
                )
        self._entries.clear()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "partial live construction cleanup failed",
                failures,
            )


@dataclass(slots=True)
class _LiveDependenciesOwner:
    """Exact live-owner transfer; cleanup owns every constructed live resource."""

    dependencies: OrchestrationDependencies
    cleanup: Callable[[], Any]
    cleanup_code: str
    _closed: bool = field(default=False, init=False, repr=False)
    _transferred_cleanup: tuple[tuple[str, Callable[[], Any]], ...] = field(
        default=(),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.dependencies) is not OrchestrationDependencies:
            raise TypeError("live owner dependencies differ")
        if not callable(self.cleanup):
            raise TypeError("live owner cleanup must be callable")
        self.cleanup_code = _canonical_identifier(
            self.cleanup_code,
            "live owner cleanup code",
        )

    def _adopt_partial_cleanup(
        self,
        entries: tuple[tuple[str, Callable[[], Any]], ...],
    ) -> None:
        if self._closed or self._transferred_cleanup:
            raise RunnerInvariantError("live owner cleanup transfer differs")
        self._transferred_cleanup = entries

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        try:
            await _maybe_await(self.cleanup())
        except BaseException:
            failures.append(
                CleanupFailure(
                    self.cleanup_code,
                    "LIVE_OWNER_CLEANUP",
                )
            )
        for code, cleanup in reversed(self._transferred_cleanup):
            try:
                await _maybe_await(cleanup())
            except BaseException:
                failures.append(
                    CleanupFailure(code, "LIVE_CONSTRUCTION_CLEANUP")
                )
        self._transferred_cleanup = ()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("live owner cleanup failed", failures)


def _clear_exception_tracebacks(error: BaseException) -> None:
    """Release execution frames while retaining the semantic exception graph."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        current.__traceback__ = None
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if isinstance(current.__cause__, BaseException):
            pending.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            pending.append(current.__context__)


async def _raise_after_partial_cleanup(
    cleanup: _PartialConstructionCleanup,
    primary: BaseException,
) -> None:
    _clear_exception_tracebacks(primary)
    try:
        await cleanup.rollback()
    except BaseException as cleanup_error:
        raise BaseExceptionGroup(
            "live construction primary and cleanup failure",
            [primary, cleanup_error],
        ) from None
    raise primary from None


async def _construct_live_dependencies_owner(
    factory: Callable[[Mapping[str, object]], Any],
    payload: Mapping[str, object],
    *,
    purpose: Literal["qualification", "evaluation"],
) -> _LiveDependenciesOwner:
    partial = _PartialConstructionCleanup()
    try:
        owner = await _maybe_await(
            factory(
                {
                    **dict(payload),
                    "partial_cleanup": partial,
                }
            )
        )
        if type(owner) is not _LiveDependenciesOwner:
            raise TypeError("live factory returned another owner type")
    except BaseException as primary:
        _clear_exception_tracebacks(primary)
        typed_primary = _typed_live_infrastructure(
            primary,
            code=(
                "QUALIFICATION_DEPENDENCY_CONSTRUCTION_FAILED"
                if purpose == "qualification"
                else "LIVE_DEPENDENCY_CONSTRUCTION_FAILED"
            ),
            stage="LIVE_CONSTRUCTION",
        )
        _clear_exception_tracebacks(typed_primary)
        del primary
        await _raise_after_partial_cleanup(partial, typed_primary)
        raise AssertionError("unreachable")
    owner._adopt_partial_cleanup(partial.transfer())
    return owner


async def _raise_after_owner_cleanup(
    owner: _LiveDependenciesOwner,
    primary: BaseException,
) -> None:
    # A failed arm may retain the runtime, model IO, or the model itself through
    # its traceback (including nested group/cause/context tracebacks).  Release
    # those frames before the owner's CUDA cleanup boundary while retaining the
    # semantic exception graph used by the disposition classifier.
    _clear_exception_tracebacks(primary)
    try:
        await owner.close()
    except BaseException as cleanup_error:
        raise BaseExceptionGroup(
            "live run primary and owner cleanup failure",
            [primary, cleanup_error],
        ) from None
    raise primary from None


def _task_canonical(value: object) -> dict[str, object]:
    if type(value) is dict:
        task = dict(value)
    else:
        method = getattr(value, "to_canonical", None)
        if not callable(method):
            raise TypeError("released task lacks a canonical projection")
        task = method()
    if type(task) is not dict:
        raise TypeError("released canonical task is not an object")
    scan_for_hidden_material(task)
    return task


async def _release(evaluator: object, phase: Phase) -> tuple[dict[str, object], ...]:
    method = getattr(evaluator, "release_phase", None)
    if not callable(method):
        raise TypeError("evaluator lacks release_phase")
    values = await _maybe_await(method(phase))
    return tuple(_task_canonical(value) for value in values)


async def _complete(evaluator: object, phase: Phase) -> None:
    method = getattr(evaluator, "complete_phase", None)
    if not callable(method):
        raise TypeError("evaluator lacks complete_phase")
    await _maybe_await(method(phase))


async def _random_schedule(
    evaluator: object,
    replicate_commitment: str,
) -> tuple[tuple[str, float], ...]:
    method = getattr(evaluator, "random_feedback", None)
    if method is None:
        method = getattr(evaluator, "random_feedback_schedule", None)
    if not callable(method):
        raise TypeError("evaluator lacks random-feedback schedule access")
    rows = await _maybe_await(method(replicate_commitment))
    schedule = tuple(tuple(row) for row in rows)
    for task_id, _ in schedule:
        random_feedback_scalar(schedule, task_id)
    return schedule  # type: ignore[return-value]


async def _evaluator_commitment_snapshot(
    evaluator: object,
) -> dict[str, object]:
    boundary = getattr(evaluator, "commitments", None)
    if callable(boundary):
        observed = await _maybe_await(boundary())
    else:
        observed = boundary
    if type(observed) is dict and set(observed) == {"commitments", "digest"}:
        payload = observed["commitments"]
        digest = observed["digest"]
    else:
        canonical = getattr(observed, "to_canonical", None)
        if not callable(canonical):
            raise TypeError("evaluator lacks its public commitment snapshot")
        payload = canonical()
        digest = getattr(observed, "digest", None)
    if type(payload) is not dict or digest != evaluator_record_digest(
        "evaluator-commitments",
        payload,
    ):
        raise RunnerInvariantError("evaluator commitment snapshot is malformed")
    scan_for_hidden_material(payload)
    return {"commitments": payload, "digest": digest}


async def _execute_scheduled(
    dependencies: OrchestrationDependencies,
    spec: ArmTaskSpec,
) -> ArmTaskResult:
    result = await _maybe_await(dependencies.execute_arm(spec))
    if type(result) is not ArmTaskResult:
        raise TypeError("execute_arm must return an exact ArmTaskResult")
    if result.task_id != spec.task_id or result.arm != spec.arm:
        raise RunnerInvariantError("arm executor returned another scheduled attempt")
    result.integrity.validate_for(
        spec,
        result.parser_disposition,
        dependencies.expected_foundation_tensor_digest,
        dependencies.expected_learner_genesis_digests.get(
            spec.replicate_commitment,
            "",
        ),
    )
    if result.evidence.get("probe_integrity_ref") != content_ref(
        "probe-integrity",
        result.integrity.to_canonical(),
    ):
        raise RunnerInvariantError("arm result does not bind its probe integrity")
    if spec.judge_arm is None and result.judgment is not None:
        raise RunnerInvariantError("diagnostic attempt acquired an undeclared judgment")
    if spec.judge_arm is not None:
        judgment = result.judgment
        if type(judgment) is not dict:
            raise RunnerInvariantError("arm result lacks its exact evaluator judgment")
        validate_objective_judgment(
            judgment,
            task_id=spec.task_id,
            arm=spec.judge_arm,
            attempt_receipt_ref=result.attempt_receipt_ref,
            raw_response=result.raw_response,
        )
    return result


async def _close_evaluator(evaluator: object) -> None:
    method = getattr(evaluator, "close", None)
    if not callable(method):
        raise TypeError("evaluator lacks close")
    await _maybe_await(method())


def _validate_unique_attempts(attempts: Sequence[ArmTaskResult]) -> None:
    refs = tuple(item.attempt_receipt_ref for item in attempts)
    keys = tuple((item.task_id, item.arm) for item in attempts)
    clone_roots = tuple(
        item.integrity.clone_audit.root
        for item in attempts
        if item.integrity.clone_audit is not None
    )
    if (
        len(set(refs)) != len(refs)
        or len(set(keys)) != len(keys)
        or len(set(clone_roots)) != len(clone_roots)
    ):
        raise RunnerInvariantError(
            "orchestration reused an attempt receipt, task/arm, or clone root"
        )


def _append_unique_attempt(
    attempts: list[ArmTaskResult],
    result: ArmTaskResult,
) -> None:
    clone_root = (
        None
        if result.integrity.clone_audit is None
        else result.integrity.clone_audit.root
    )
    if any(
        item.attempt_receipt_ref == result.attempt_receipt_ref
        or (item.task_id, item.arm) == (result.task_id, result.arm)
        or (
            clone_root is not None
            and item.integrity.clone_audit is not None
            and item.integrity.clone_audit.root == clone_root
        )
        for item in attempts
    ):
        raise RunnerInvariantError(
            "orchestration reused an attempt receipt, task/arm, or clone root"
        )
    attempts.append(result)


def _validate_removal_group(
    results: Sequence[ArmTaskResult],
    *,
    require_score: bool = True,
) -> dict[str, object]:
    by_arm = {result.arm: result for result in results}
    required = {
        "FULL",
        "RETRIEVAL_ONLY",
        "FROZEN_ORIGIN",
        "PROSPECTIVE_REMOVAL",
        "BACKEND_REMOVAL",
    }
    if not required.issubset(by_arm):
        raise RunnerInvariantError("arm group lacks a declared removal comparison")
    identities = {
        (
            result.integrity.purpose,
            result.integrity.phase,
            result.task_id,
        )
        for result in by_arm.values()
    }
    if len(identities) != 1:
        raise RunnerInvariantError("removal group identities differ")
    arms = _validated_removal_fairness_preimage(
        by_arm["FULL"].evidence,
        by_arm["FROZEN_ORIGIN"].evidence,
        by_arm["PROSPECTIVE_REMOVAL"].evidence,
        by_arm["BACKEND_REMOVAL"].evidence,
        by_arm["RETRIEVAL_ONLY"].evidence,
        require_score=require_score,
    )
    purpose, phase, task_id = next(iter(identities))
    preimage = {"arms": arms, "require_score": require_score}
    record = {
        "arms": arms,
        "evidence_ref": content_ref("removal-fairness", preimage),
        "phase": phase,
        "purpose": purpose,
        "require_score": require_score,
        "schema": REMOVAL_FAIRNESS_SCHEMA,
        "task_id": task_id,
    }
    canonical_json_bytes(record)
    return record


def _validated_adaptation_lineage_rows(
    attempts: Sequence[ArmTaskResult],
    expected_learner_genesis_digests: Mapping[str, str],
    *,
    purpose: Purpose,
    arms: tuple[Arm, ...],
    require_advance: frozenset[Arm],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    if purpose not in ("qualification", "evaluation"):
        raise ValueError("adaptation lineage purpose differs")
    if not arms or any(arm not in STATEFUL_ARMS for arm in arms):
        raise ValueError("adaptation lineage arms differ")
    if not require_advance.issubset(arms):
        raise ValueError("adaptation advance requirement differs")
    if not isinstance(expected_learner_genesis_digests, Mapping):
        raise TypeError("expected adaptation genesis identities must be a mapping")
    expected_geneses = dict(expected_learner_genesis_digests)
    for replicate_commitment, genesis_digest in expected_geneses.items():
        _digest(replicate_commitment, "expected adaptation replicate commitment")
        _digest(genesis_digest, "expected adaptation genesis digest")
    selected = tuple(
        item
        for item in attempts
        if item.arm in arms
        and item.integrity.purpose == purpose
        and item.integrity.phase == "adaptation"
    )
    groups: dict[tuple[Arm, str], list[ArmTaskResult]] = {}
    for item in selected:
        groups.setdefault(
            (item.arm, item.integrity.replicate_commitment),
            [],
        ).append(item)
    expected_groups = {
        (arm, replicate_commitment)
        for arm in arms
        for replicate_commitment in expected_geneses
    }
    if (
        set(groups) != expected_groups
        or any(len(rows) != 12 for rows in groups.values())
        or len(set(expected_geneses.values())) != 1
    ):
        raise RunnerInvariantError("adaptation lineage cardinality differs")
    canonical_groups: list[dict[str, object]] = []
    for arm, replicate_commitment in sorted(groups):
        rows = groups[(arm, replicate_commitment)]
        transitions: list[dict[str, object]] = []
        genesis: str | None = None
        previous_child: str | None = None
        previous_sequence = QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
        for item in rows:
            integrity = item.integrity
            row_genesis = integrity.learner_genesis_digest
            parent = integrity.learner_parent_digest
            child = integrity.learner_child_digest
            sequence = integrity.learner_sequence
            if row_genesis is None or parent is None:
                raise RunnerInvariantError("FULL adaptation lineage is incomplete")
            if genesis is None:
                genesis = row_genesis
                previous_child = genesis
            if (
                row_genesis != genesis
                or row_genesis != _digest(
                    expected_geneses[replicate_commitment],
                    "expected adaptation genesis digest",
                )
                or parent != previous_child
            ):
                raise RunnerInvariantError("adaptation lineage reset or forked")
            if item.parser_disposition == "MALFORMED":
                if child is not None or sequence is not None:
                    raise RunnerInvariantError(
                        "malformed adaptation attempt advanced its learner"
                    )
            elif (
                child is None
                or sequence != previous_sequence + 1
                or child == parent
            ):
                raise RunnerInvariantError(
                    "adaptation learner transition did not advance"
                )
            transitions.append(
                {
                    "child_digest": child,
                    "parent_digest": parent,
                    "parser_disposition": item.parser_disposition,
                    "sequence": sequence,
                    "task_id": item.task_id,
                }
            )
            if child is not None:
                previous_child = child
                previous_sequence = sequence  # type: ignore[assignment]
        advanced = (
            previous_sequence > QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
            and previous_child != genesis
        )
        if arm in require_advance and not advanced:
            raise RunnerInvariantError(f"{arm} adaptation lineage never advanced")
        canonical_groups.append(
            {
                "advanced": advanced,
                "arm": arm,
                "final_child_digest": previous_child,
                "genesis_digest": genesis,
                "replicate_commitment": replicate_commitment,
                "transitions": transitions,
            }
        )
    return canonical_groups, expected_geneses


def validate_adaptation_lineages(
    attempts: Sequence[ArmTaskResult],
    expected_learner_genesis_digests: Mapping[str, str],
    *,
    purpose: Purpose,
) -> tuple[dict[str, object], str]:
    """Validate FULL and RANDOM_FEEDBACK as separate continuous lineages."""

    lineages, expected_geneses = _validated_adaptation_lineage_rows(
        attempts,
        expected_learner_genesis_digests,
        purpose=purpose,
        arms=("FULL", "RANDOM_FEEDBACK"),
        require_advance=(
            frozenset({"FULL"})
            if purpose == "qualification"
            else frozenset()
        ),
    )
    aggregate = {
        "expected_genesis_by_replicate": dict(sorted(expected_geneses.items())),
        "lineages": lineages,
        "purpose": purpose,
        "schema": ADAPTATION_LINEAGE_SCHEMA,
    }
    lineage_ref = content_ref("adaptation-lineages", aggregate)
    return aggregate, lineage_ref


def validate_full_adaptation_lineages(
    attempts: Sequence[ArmTaskResult],
    expected_learner_genesis_digests: Mapping[str, str],
) -> tuple[dict[str, object], str]:
    """Return the exact two-replicate FULL predicate-9 lineage subset."""

    lineages, expected_geneses = _validated_adaptation_lineage_rows(
        attempts,
        expected_learner_genesis_digests,
        purpose="evaluation",
        arms=("FULL",),
        require_advance=frozenset(),
    )
    replicates = []
    for row in lineages:
        if row["arm"] != "FULL":
            raise RunnerInvariantError("FULL lineage subset contains another arm")
        replicates.append({name: value for name, value in row.items() if name != "arm"})
    aggregate = {
        "all_replicates_advanced": all(
            row["advanced"] is True for row in replicates
        ),
        "expected_genesis_by_replicate": dict(sorted(expected_geneses.items())),
        "replicates": replicates,
        "schema": LINEAGE_INTEGRITY_SCHEMA,
    }
    lineage_ref = content_ref("full-adaptation-lineages", aggregate)
    return aggregate, lineage_ref


def _foundation_integrity_evidence(
    before: Mapping[str, str],
    after: Mapping[str, str],
    expected: Mapping[str, str],
    expected_tensor_digest: str,
) -> tuple[dict[str, object], str]:
    frozen = validate_foundation_hashes(before, after, expected)
    tensor_digest = _digest(
        expected_tensor_digest,
        "foundation integrity tensor digest",
    )
    if frozen.get(FOUNDATION_TENSOR_HASH_NAME) != tensor_digest.removeprefix(
        "sha256:"
    ):
        raise RunnerInvariantError("foundation evidence does not bind its tensor digest")
    evidence = {
        "after": dict(sorted(after.items())),
        "before": dict(sorted(before.items())),
        "expected": frozen,
        "guard_semantics": FOUNDATION_GUARD_SEMANTICS,
        "schema": FOUNDATION_INTEGRITY_SCHEMA,
        "tensor_digest": tensor_digest,
    }
    return evidence, content_ref("foundation-integrity", evidence)


def _validate_removal_records(
    records: Sequence[Mapping[str, object]],
    *,
    purpose: Purpose,
) -> list[dict[str, object]]:
    expected_count = 1 if purpose == "qualification" else 60
    if len(records) != expected_count:
        raise RunnerInvariantError("removal-fairness record cardinality differs")
    canonical_records: list[dict[str, object]] = []
    identities: set[tuple[object, object]] = set()
    for value in records:
        record = dict(value)
        if set(record) != {
            "arms",
            "evidence_ref",
            "phase",
            "purpose",
            "require_score",
            "schema",
            "task_id",
        } or (
            record["schema"] != REMOVAL_FAIRNESS_SCHEMA
            or record["purpose"] != purpose
            or record["phase"] not in ("development", "final")
            or record["require_score"] != (purpose == "evaluation")
            or type(record["arms"]) is not dict
        ):
            raise RunnerInvariantError("removal-fairness record identity differs")
        _digest(record["task_id"], "removal-fairness task_id")
        expected_ref = content_ref(
            "removal-fairness",
            {
                "arms": record["arms"],
                "require_score": record["require_score"],
            },
        )
        if record["evidence_ref"] != expected_ref:
            raise RunnerInvariantError("removal-fairness preimage reference differs")
        arms = record["arms"]
        if set(arms) != {
            "BACKEND_REMOVAL",
            "FROZEN_ORIGIN",
            "FULL",
            "PROSPECTIVE_REMOVAL",
            "RETRIEVAL_ONLY",
        } or any(
            type(row) is not dict or row.get("task_id") != record["task_id"]
            for row in arms.values()
        ):
            raise RunnerInvariantError("removal-fairness arm preimage differs")
        revalidated_arms = _validated_removal_fairness_preimage(
            arms["FULL"],
            arms["FROZEN_ORIGIN"],
            arms["PROSPECTIVE_REMOVAL"],
            arms["BACKEND_REMOVAL"],
            arms["RETRIEVAL_ONLY"],
            require_score=record["require_score"],
        )
        if revalidated_arms != arms:
            raise RunnerInvariantError("removal-fairness canonical preimage differs")
        identity = (record["phase"], record["task_id"])
        if identity in identities:
            raise RunnerInvariantError("removal-fairness task identity was reused")
        identities.add(identity)
        canonical_json_bytes(record)
        canonical_records.append(record)
    phase_counts = {
        phase: sum(record["phase"] == phase for record in canonical_records)
        for phase in ("development", "final")
    }
    expected_phase_counts = (
        {"development": 1, "final": 0}
        if purpose == "qualification"
        else {"development": 20, "final": 40}
    )
    if phase_counts != expected_phase_counts:
        raise RunnerInvariantError("removal-fairness phase coverage differs")
    return canonical_records


def build_run_integrity_evidence(
    *,
    purpose: Purpose,
    attempts: Sequence[ArmTaskResult],
    removal_fairness: Sequence[Mapping[str, object]],
    foundation_before: Mapping[str, str],
    foundation_after: Mapping[str, str],
    expected_foundation: Mapping[str, str],
    expected_foundation_tensor_digest: str,
    expected_learner_genesis_digests: Mapping[str, str],
    adaptation_lineage: Mapping[str, object],
    adaptation_lineage_ref: str,
    resource_accounting: Mapping[str, object],
    full_adaptation_lineage: Mapping[str, object] | None = None,
    full_adaptation_lineage_ref: str | None = None,
) -> tuple[dict[str, object], str, dict[str, object], str]:
    """Build the bounded canonical preimage for predicates 8 and 9."""

    if purpose not in ("qualification", "evaluation"):
        raise ValueError("run integrity purpose differs")
    expected_attempts = (
        QUALIFICATION_COUNTS[0]
        if purpose == "qualification"
        else EVALUATION_COUNTS[0]
    )
    if len(attempts) != expected_attempts:
        raise RunnerInvariantError("run integrity attempt cardinality differs")
    _validate_unique_attempts(attempts)
    attempt_records: list[dict[str, object]] = []
    clone_roots: list[str] = []
    tensor_digest = _digest(
        expected_foundation_tensor_digest,
        "run integrity tensor digest",
    )
    for item in attempts:
        integrity = item.integrity.to_canonical()
        integrity_ref = content_ref("probe-integrity", integrity)
        if (
            item.integrity.purpose != purpose
            or item.integrity.foundation_tensor_digest != tensor_digest
            or item.integrity.foundation_guard_semantics
            != FOUNDATION_GUARD_SEMANTICS
            or item.evidence.get("probe_integrity_ref") != integrity_ref
        ):
            raise RunnerInvariantError("run probe integrity binding differs")
        if item.integrity.clone_audit is not None:
            clone_roots.append(item.integrity.clone_audit.root)
        attempt_records.append(
            {
                "arm": item.arm,
                "attempt_receipt_ref": item.attempt_receipt_ref,
                "parser_disposition": item.parser_disposition,
                "probe_integrity": integrity,
                "probe_integrity_ref": integrity_ref,
                "task_id": item.task_id,
            }
        )
    expected_clone_count = 6 if purpose == "qualification" else 360
    if (
        len(clone_roots) != expected_clone_count
        or len(set(clone_roots)) != len(clone_roots)
    ):
        raise RunnerInvariantError("disposable clone-root coverage differs")
    removal_records = _validate_removal_records(
        removal_fairness,
        purpose=purpose,
    )
    adaptation = dict(adaptation_lineage)
    revalidated_adaptation, revalidated_adaptation_ref = (
        validate_adaptation_lineages(
            attempts,
            expected_learner_genesis_digests,
            purpose=purpose,
        )
    )
    if (
        adaptation.get("schema") != ADAPTATION_LINEAGE_SCHEMA
        or adaptation.get("purpose") != purpose
        or adaptation_lineage_ref
        != content_ref("adaptation-lineages", adaptation)
        or adaptation != revalidated_adaptation
        or adaptation_lineage_ref != revalidated_adaptation_ref
    ):
        raise RunnerInvariantError("adaptation-lineage aggregate binding differs")
    full_lineage: dict[str, object] | None = None
    if purpose == "evaluation":
        if full_adaptation_lineage is None or full_adaptation_lineage_ref is None:
            raise RunnerInvariantError("evaluation lacks its FULL lineage subset")
        full_lineage = dict(full_adaptation_lineage)
        revalidated_full, revalidated_full_ref = (
            validate_full_adaptation_lineages(
                attempts,
                expected_learner_genesis_digests,
            )
        )
        if (
            full_lineage.get("schema") != LINEAGE_INTEGRITY_SCHEMA
            or full_adaptation_lineage_ref
            != content_ref("full-adaptation-lineages", full_lineage)
            or full_lineage != revalidated_full
            or full_adaptation_lineage_ref != revalidated_full_ref
        ):
            raise RunnerInvariantError("FULL lineage subset binding differs")
    elif full_adaptation_lineage is not None or full_adaptation_lineage_ref is not None:
        raise RunnerInvariantError("qualification cannot claim evaluation FULL lineages")
    accounting = dict(resource_accounting)
    canonical_json_bytes(accounting)
    foundation, foundation_ref = _foundation_integrity_evidence(
        foundation_before,
        foundation_after,
        expected_foundation,
        tensor_digest,
    )
    aggregate = {
        "adaptation_lineage": adaptation,
        "adaptation_lineage_ref": adaptation_lineage_ref,
        "attempts": attempt_records,
        "clone_roots": sorted(clone_roots),
        "foundation": foundation,
        "foundation_ref": foundation_ref,
        "foundation_guard_checks": len(attempt_records),
        "full_adaptation_lineage": full_lineage,
        "full_adaptation_lineage_ref": full_adaptation_lineage_ref,
        "purpose": purpose,
        "removal_fairness": removal_records,
        "removal_fairness_refs": [
            record["evidence_ref"] for record in removal_records
        ],
        "resource_accounting": accounting,
        "schema": RUN_INTEGRITY_SCHEMA,
    }
    encoded = canonical_json_bytes(aggregate)
    if len(encoded) > MAXIMUM_RESULT_BYTES:
        raise RunnerInvariantError("run integrity evidence exceeds the result ceiling")
    scan_for_hidden_material(aggregate, maximum_bytes=MAXIMUM_RESULT_BYTES)
    return (
        aggregate,
        content_ref("run-integrity", aggregate),
        foundation,
        foundation_ref,
    )


def _validate_final_metrics(value: Mapping[str, object]) -> dict[str, object]:
    metrics = dict(value)
    if set(metrics) != {"aggregates", "identity", "purpose", "schema"}:
        raise RunnerInvariantError("final metrics fields differ")
    if (
        metrics["schema"] != SUITE_SCHEMA
        or metrics["identity"] != EVALUATOR_EVALUATION_IDENTITY
        or metrics["purpose"] != "evaluation"
        or type(metrics["aggregates"]) is not list
        or not metrics["aggregates"]
    ):
        raise RunnerInvariantError("final metrics identity differs")
    family_counts = {
        "adaptation": {
            "symbolic-demonstration-transfer": 8,
            "glyph-machine": 4,
            "causal-operator": 12,
        },
        "development": {
            "symbolic-demonstration-transfer": 4,
            "glyph-machine": 4,
            "causal-operator": 12,
        },
        "final": {
            "symbolic-demonstration-transfer": 8,
            "glyph-machine": 8,
            "causal-operator": 24,
        },
    }
    expected_keys = {
        (phase, arm, family)
        for phase, arms in (
            ("adaptation", ("FULL", "RANDOM_FEEDBACK")),
            ("development", EVALUATION_ARMS),
            ("final", EVALUATION_ARMS),
        )
        for arm in arms
        for family in FAMILIES
    }
    seen: set[tuple[object, object, object]] = set()
    for row in metrics["aggregates"]:
        if type(row) is not dict or set(row) != {
            "arm",
            "attempts",
            "binary_success_total",
            "family",
            "pairwise_agreement_total",
            "phase",
        }:
            raise RunnerInvariantError("final metric aggregate fields differ")
        key = (row["phase"], row["arm"], row["family"])
        if key not in expected_keys or key in seen:
            raise RunnerInvariantError("final metric aggregate key is duplicated")
        seen.add(key)
        if (
            type(row["attempts"]) is not int
            or row["attempts"] <= 0
            or type(row["binary_success_total"]) is not int
            or not 0 <= row["binary_success_total"] <= row["attempts"]
            or row["attempts"]
            != family_counts[row["phase"]][row["family"]]
        ):
            raise RunnerInvariantError("final metric aggregate count differs")
        pairwise = row["pairwise_agreement_total"]
        if row["family"] == "symbolic-demonstration-transfer":
            if (
                type(pairwise) is not float
                or not math.isfinite(pairwise)
                or not 0.0 <= pairwise <= float(row["attempts"])
            ):
                raise RunnerInvariantError("symbolic metric pairwise total differs")
        elif pairwise is not None:
            raise RunnerInvariantError("pairwise metric belongs only to symbolic tasks")
    if seen != expected_keys:
        raise RunnerInvariantError("final metric aggregate coverage differs")
    scan_for_hidden_material(metrics)
    return metrics


def _probe_integrity_from_canonical(
    value: Mapping[str, object],
) -> ProbeIntegrityEvidence:
    """Reconstruct and strictly validate one serialized probe-integrity record."""

    if type(value) is not dict or set(value) != {
        "arm",
        "clone_after",
        "clone_audit",
        "clone_audit_ref",
        "clone_before",
        "foundation_guard_semantics",
        "foundation_tensor_digest",
        "learner_child_digest",
        "learner_genesis_digest",
        "learner_parent_digest",
        "learner_sequence",
        "phase",
        "purpose",
        "replicate_commitment",
        "schema",
        "source_baseline_after",
        "source_baseline_before",
        "task_id",
    }:
        raise RunnerInvariantError("serialized probe-integrity fields differ")

    def optional_hashes(name: str) -> tuple[tuple[str, str], ...] | None:
        raw = value[name]
        if raw is None:
            return None
        if type(raw) is not dict:
            raise RunnerInvariantError("serialized probe hash map differs")
        return _hash_pairs(raw, f"serialized probe {name}")

    clone_value = value["clone_audit"]
    clone: CloneAudit | None = None
    if clone_value is not None:
        if type(clone_value) is not dict or set(clone_value) != {
            "destination_hashes",
            "root",
            "schema",
            "source_hashes",
        }:
            raise RunnerInvariantError("serialized clone-audit fields differ")
        if (
            type(clone_value["source_hashes"]) is not dict
            or type(clone_value["destination_hashes"]) is not dict
        ):
            raise RunnerInvariantError("serialized clone-audit hashes differ")
        clone = CloneAudit(
            schema=clone_value["schema"],  # type: ignore[arg-type]
            source_hashes=_hash_pairs(
                clone_value["source_hashes"],
                "serialized clone source",
            ),
            destination_hashes=_hash_pairs(
                clone_value["destination_hashes"],
                "serialized clone destination",
            ),
            root=clone_value["root"],  # type: ignore[arg-type]
        )
        if clone.to_canonical() != clone_value:
            raise RunnerInvariantError("serialized clone-audit is not canonical")
        if value["clone_audit_ref"] != clone.audit_ref:
            raise RunnerInvariantError("serialized clone-audit reference differs")
    elif value["clone_audit_ref"] is not None:
        raise RunnerInvariantError("serialized clone-audit reference is orphaned")

    probe = ProbeIntegrityEvidence(
        schema=value["schema"],  # type: ignore[arg-type]
        purpose=value["purpose"],  # type: ignore[arg-type]
        phase=value["phase"],  # type: ignore[arg-type]
        task_id=value["task_id"],  # type: ignore[arg-type]
        arm=value["arm"],  # type: ignore[arg-type]
        replicate_commitment=value["replicate_commitment"],  # type: ignore[arg-type]
        foundation_tensor_digest=value["foundation_tensor_digest"],  # type: ignore[arg-type]
        foundation_guard_semantics=value["foundation_guard_semantics"],  # type: ignore[arg-type]
        source_baseline_before=optional_hashes("source_baseline_before"),
        source_baseline_after=optional_hashes("source_baseline_after"),
        clone_before=optional_hashes("clone_before"),
        clone_after=optional_hashes("clone_after"),
        clone_audit=clone,
        learner_genesis_digest=value["learner_genesis_digest"],  # type: ignore[arg-type]
        learner_parent_digest=value["learner_parent_digest"],  # type: ignore[arg-type]
        learner_child_digest=value["learner_child_digest"],  # type: ignore[arg-type]
        learner_sequence=value["learner_sequence"],  # type: ignore[arg-type]
    )
    if probe.to_canonical() != value:
        raise RunnerInvariantError("serialized probe-integrity is not canonical")
    return probe


def _validate_serialized_adaptation_lineage(
    value: Mapping[str, object],
    *,
    purpose: Purpose,
) -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    """Validate lineage continuity without requiring live learner objects."""

    if type(value) is not dict or set(value) != {
        "expected_genesis_by_replicate",
        "lineages",
        "purpose",
        "schema",
    } or value["schema"] != ADAPTATION_LINEAGE_SCHEMA or value["purpose"] != purpose:
        raise RunnerInvariantError("serialized adaptation-lineage identity differs")
    expected = value["expected_genesis_by_replicate"]
    lineages = value["lineages"]
    expected_replicates = 1 if purpose == "qualification" else 2
    if (
        type(expected) is not dict
        or len(expected) != expected_replicates
        or type(lineages) is not list
        or len(lineages) != expected_replicates * 2
    ):
        raise RunnerInvariantError("serialized adaptation-lineage cardinality differs")
    for replicate, genesis in expected.items():
        _digest(replicate, "serialized lineage replicate")
        _digest(genesis, "serialized lineage genesis")
    if len(set(expected.values())) != 1:
        raise RunnerInvariantError("serialized lineages do not share exact genesis")
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    seen_tasks: set[tuple[str, str]] = set()
    for raw_row in lineages:
        if type(raw_row) is not dict or set(raw_row) != {
            "advanced",
            "arm",
            "final_child_digest",
            "genesis_digest",
            "replicate_commitment",
            "transitions",
        }:
            raise RunnerInvariantError("serialized adaptation-lineage row differs")
        row = dict(raw_row)
        arm = row["arm"]
        replicate = row["replicate_commitment"]
        if (
            arm not in ("FULL", "RANDOM_FEEDBACK")
            or replicate not in expected
            or (arm, replicate) in indexed
            or row["genesis_digest"] != expected[replicate]
            or type(row["transitions"]) is not list
            or len(row["transitions"]) != 12
        ):
            raise RunnerInvariantError("serialized adaptation-lineage membership differs")
        previous_child = row["genesis_digest"]
        previous_sequence = QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
        for raw_transition in row["transitions"]:
            if type(raw_transition) is not dict or set(raw_transition) != {
                "child_digest",
                "parent_digest",
                "parser_disposition",
                "sequence",
                "task_id",
            }:
                raise RunnerInvariantError("serialized lineage transition fields differ")
            transition = raw_transition
            _digest(transition["task_id"], "serialized lineage task_id")
            task_identity = (arm, transition["task_id"])
            if task_identity in seen_tasks or transition["parent_digest"] != previous_child:
                raise RunnerInvariantError("serialized adaptation lineage reset or forked")
            seen_tasks.add(task_identity)
            disposition = transition["parser_disposition"]
            if disposition == "MALFORMED":
                if transition["child_digest"] is not None or transition["sequence"] is not None:
                    raise RunnerInvariantError("malformed serialized lineage advanced")
            elif disposition == "ADMITTED":
                child = transition["child_digest"]
                sequence = transition["sequence"]
                if (
                    not isinstance(child, str)
                    or type(sequence) is not int
                    or sequence != previous_sequence + 1
                    or child == previous_child
                ):
                    raise RunnerInvariantError("admitted serialized lineage did not advance")
                _digest(child, "serialized lineage child")
                previous_child = child
                previous_sequence = sequence
            else:
                raise RunnerInvariantError("serialized lineage parser disposition differs")
        advanced = (
            previous_sequence > QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
            and previous_child != row["genesis_digest"]
        )
        if (
            type(row["advanced"]) is not bool
            or row["advanced"] != advanced
            or row["final_child_digest"] != previous_child
        ):
            raise RunnerInvariantError("serialized lineage terminal state differs")
        if purpose == "qualification" and arm == "FULL" and not advanced:
            raise RunnerInvariantError("qualification FULL lineage never advanced")
        indexed[(arm, replicate)] = row
    expected_keys = {
        (arm, replicate)
        for arm in ("FULL", "RANDOM_FEEDBACK")
        for replicate in expected
    }
    if set(indexed) != expected_keys:
        raise RunnerInvariantError("serialized adaptation-lineage coverage differs")
    if [
        (row["arm"], row["replicate_commitment"])
        for row in lineages
    ] != sorted(expected_keys):
        raise RunnerInvariantError("serialized adaptation-lineage order differs")
    canonical = decode_canonical_json(
        canonical_json_bytes(value),
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )
    if type(canonical) is not dict:
        raise RunnerInvariantError("serialized adaptation-lineage copy differs")
    return canonical, indexed


def _validate_serialized_full_lineage(
    value: Mapping[str, object],
    adaptation_rows: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[dict[str, object], int]:
    if type(value) is not dict or set(value) != {
        "all_replicates_advanced",
        "expected_genesis_by_replicate",
        "replicates",
        "schema",
    } or value["schema"] != LINEAGE_INTEGRITY_SCHEMA:
        raise RunnerInvariantError("serialized FULL-lineage identity differs")
    expected = value["expected_genesis_by_replicate"]
    replicates = value["replicates"]
    if type(expected) is not dict or len(expected) != 2 or type(replicates) is not list or len(replicates) != 2:
        raise RunnerInvariantError("serialized FULL-lineage cardinality differs")
    observed: dict[str, dict[str, object]] = {}
    for raw_row in replicates:
        if type(raw_row) is not dict or set(raw_row) != {
            "advanced",
            "final_child_digest",
            "genesis_digest",
            "replicate_commitment",
            "transitions",
        }:
            raise RunnerInvariantError("serialized FULL-lineage row differs")
        replicate = raw_row["replicate_commitment"]
        if replicate in observed or ("FULL", replicate) not in adaptation_rows:
            raise RunnerInvariantError("serialized FULL-lineage replicate differs")
        expected_row = {
            name: item
            for name, item in adaptation_rows[("FULL", replicate)].items()
            if name != "arm"
        }
        if raw_row != expected_row:
            raise RunnerInvariantError("FULL-lineage subset differs from adaptation evidence")
        observed[replicate] = raw_row
    if set(observed) != set(expected) or any(
        expected[replicate] != observed[replicate]["genesis_digest"]
        for replicate in observed
    ):
        raise RunnerInvariantError("serialized FULL-lineage genesis binding differs")
    if [row["replicate_commitment"] for row in replicates] != sorted(expected):
        raise RunnerInvariantError("serialized FULL-lineage order differs")
    advanced_count = sum(row["advanced"] is True for row in observed.values())
    if (
        type(value["all_replicates_advanced"]) is not bool
        or value["all_replicates_advanced"] != (advanced_count == 2)
    ):
        raise RunnerInvariantError("serialized FULL-lineage aggregate differs")
    canonical = decode_canonical_json(
        canonical_json_bytes(value),
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )
    if type(canonical) is not dict:
        raise RunnerInvariantError("serialized FULL-lineage copy differs")
    return canonical, advanced_count


def validate_serialized_run_integrity(
    value: Mapping[str, object],
    *,
    purpose: Purpose,
) -> tuple[dict[str, object], int]:
    """Strictly revalidate predicate-8/9 evidence from result bytes."""

    expected_fields = {
        "adaptation_lineage",
        "adaptation_lineage_ref",
        "attempts",
        "clone_roots",
        "foundation",
        "foundation_ref",
        "foundation_guard_checks",
        "full_adaptation_lineage",
        "full_adaptation_lineage_ref",
        "purpose",
        "removal_fairness",
        "removal_fairness_refs",
        "resource_accounting",
        "schema",
    }
    if (
        type(value) is not dict
        or set(value) != expected_fields
        or value["schema"] != RUN_INTEGRITY_SCHEMA
        or value["purpose"] != purpose
    ):
        raise RunnerInvariantError("serialized run-integrity identity differs")
    attempts = value["attempts"]
    clone_roots = value["clone_roots"]
    expected_attempts = QUALIFICATION_COUNTS[0] if purpose == "qualification" else EVALUATION_COUNTS[0]
    expected_clones = 6 if purpose == "qualification" else 360
    if (
        type(attempts) is not list
        or len(attempts) != expected_attempts
        or type(clone_roots) is not list
        or len(clone_roots) != expected_clones
        or clone_roots != sorted(clone_roots)
        or len(set(clone_roots)) != expected_clones
        or value["foundation_guard_checks"] != expected_attempts
    ):
        raise RunnerInvariantError("serialized run-integrity cardinality differs")
    seen_pairs: set[tuple[str, str]] = set()
    seen_receipts: set[str] = set()
    observed_clone_roots: list[str] = []
    observed_phase_arms: Counter[tuple[str, str]] = Counter()
    observed_probe_identities: set[tuple[str, str, str]] = set()
    observed_probes: list[ProbeIntegrityEvidence] = []
    observed_task_groups: dict[
        tuple[str, str],
        tuple[set[str], set[str]],
    ] = {}
    adaptation_probes: dict[
        tuple[str, str],
        list[tuple[ProbeIntegrityEvidence, str]],
    ] = {}
    for raw_attempt in attempts:
        if type(raw_attempt) is not dict or set(raw_attempt) != {
            "arm",
            "attempt_receipt_ref",
            "parser_disposition",
            "probe_integrity",
            "probe_integrity_ref",
            "task_id",
        }:
            raise RunnerInvariantError("serialized run-integrity attempt fields differ")
        probe_value = raw_attempt["probe_integrity"]
        if type(probe_value) is not dict:
            raise RunnerInvariantError("serialized probe-integrity payload differs")
        probe = _probe_integrity_from_canonical(probe_value)
        receipt = _digest(raw_attempt["attempt_receipt_ref"], "serialized attempt receipt")
        pair = (raw_attempt["task_id"], raw_attempt["arm"])
        if (
            pair in seen_pairs
            or receipt in seen_receipts
            or probe.purpose != purpose
            or probe.task_id != raw_attempt["task_id"]
            or probe.arm != raw_attempt["arm"]
            or raw_attempt["parser_disposition"] not in ("ADMITTED", "MALFORMED")
            or raw_attempt["probe_integrity_ref"]
            != content_ref("probe-integrity", probe_value)
        ):
            raise RunnerInvariantError("serialized run-integrity attempt binding differs")
        disposition = raw_attempt["parser_disposition"]
        should_transition = (
            disposition == "ADMITTED"
            and probe.arm in STATEFUL_ARMS
            and (
                purpose == "evaluation"
                or probe.phase == "adaptation"
                or probe.arm == "FULL"
            )
        )
        if (probe.learner_child_digest is not None) != should_transition:
            raise RunnerInvariantError(
                "serialized parser/learner transition binding differs"
            )
        seen_pairs.add(pair)
        seen_receipts.add(receipt)
        observed_phase_arms[(probe.phase, probe.arm)] += 1
        observed_probe_identities.add((probe.phase, probe.task_id, probe.arm))
        observed_probes.append(probe)
        group_arms, group_replicates = observed_task_groups.setdefault(
            (probe.phase, probe.task_id),
            (set(), set()),
        )
        group_arms.add(probe.arm)
        group_replicates.add(probe.replicate_commitment)
        if probe.phase == "adaptation" and probe.arm in (
            "FULL",
            "RANDOM_FEEDBACK",
        ):
            adaptation_probes.setdefault(
                (probe.arm, probe.replicate_commitment),
                [],
            ).append((probe, raw_attempt["parser_disposition"]))
        if probe.clone_audit is not None:
            observed_clone_roots.append(probe.clone_audit.root)
    if sorted(observed_clone_roots) != clone_roots:
        raise RunnerInvariantError("serialized clone-root preimages differ")
    accounting_mode = RunMode.QUALIFICATION if purpose == "qualification" else RunMode.EVALUATION
    accounting = validate_serialized_run_accounting(
        value["resource_accounting"],  # type: ignore[arg-type]
        accounting_mode,
    )
    expected_phase_arms: Counter[tuple[str, str]] = Counter()
    if purpose == "qualification":
        expected_phase_arms.update(
            {
                ("adaptation", "FULL"): 12,
                ("adaptation", "RANDOM_FEEDBACK"): 12,
                **{
                    ("development", arm): 1
                    for arm in EVALUATION_ARMS
                },
            }
        )
    else:
        expected_phase_arms.update(
            {
                ("adaptation", "FULL"): 24,
                ("adaptation", "RANDOM_FEEDBACK"): 24,
                **{
                    (phase, arm): count
                    for phase, count in (("development", 20), ("final", 40))
                    for arm in EVALUATION_ARMS
                },
            }
        )
    if observed_phase_arms != expected_phase_arms:
        raise RunnerInvariantError("serialized run-integrity phase/arm schedule differs")
    expected_task_group_counts = (
        {"adaptation": 12, "development": 1, "final": 0}
        if purpose == "qualification"
        else {"adaptation": 24, "development": 20, "final": 40}
    )
    for phase in PHASES:
        selected = {
            identity: membership
            for identity, membership in observed_task_groups.items()
            if identity[0] == phase
        }
        expected_arms = (
            {"FULL", "RANDOM_FEEDBACK"}
            if phase == "adaptation"
            else set(EVALUATION_ARMS)
        )
        if len(selected) != expected_task_group_counts[phase] or any(
            arms != expected_arms or len(replicates) != 1
            for arms, replicates in selected.values()
        ):
            raise RunnerInvariantError(
                "serialized run-integrity per-task arm grouping differs"
            )
    foundation = value["foundation"]
    if type(foundation) is not dict or set(foundation) != {
        "after",
        "before",
        "expected",
        "guard_semantics",
        "schema",
        "tensor_digest",
    } or foundation["schema"] != FOUNDATION_INTEGRITY_SCHEMA or foundation["guard_semantics"] != FOUNDATION_GUARD_SEMANTICS:
        raise RunnerInvariantError("serialized foundation evidence differs")
    for name in ("before", "after", "expected"):
        if type(foundation[name]) is not dict:
            raise RunnerInvariantError("serialized foundation hash map differs")
    validated_foundation = validate_foundation_hashes(
        foundation["before"],
        foundation["after"],
        foundation["expected"],
    )
    tensor_digest = _digest(foundation["tensor_digest"], "serialized foundation tensor")
    if validated_foundation.get(FOUNDATION_TENSOR_HASH_NAME) != tensor_digest.removeprefix("sha256:"):
        raise RunnerInvariantError("serialized foundation tensor binding differs")
    if value["foundation_ref"] != content_ref("foundation-integrity", foundation):
        raise RunnerInvariantError("serialized foundation reference differs")
    adaptation, adaptation_rows = _validate_serialized_adaptation_lineage(
        value["adaptation_lineage"],  # type: ignore[arg-type]
        purpose=purpose,
    )
    if value["adaptation_lineage_ref"] != content_ref("adaptation-lineages", adaptation):
        raise RunnerInvariantError("serialized adaptation-lineage reference differs")
    expected_geneses = adaptation["expected_genesis_by_replicate"]
    if any(
        probe.replicate_commitment not in expected_geneses
        or probe.learner_genesis_digest
        != expected_geneses[probe.replicate_commitment]
        for probe in observed_probes
        if probe.arm in STATEFUL_ARMS
    ):
        raise RunnerInvariantError("serialized stateful probe genesis binding differs")
    if set(adaptation_probes) != set(adaptation_rows):
        raise RunnerInvariantError("serialized adaptation attempts differ from lineages")
    for identity, lineage in adaptation_rows.items():
        rows = adaptation_probes[identity]
        derived_transitions = [
            {
                "child_digest": probe.learner_child_digest,
                "parent_digest": probe.learner_parent_digest,
                "parser_disposition": disposition,
                "sequence": probe.learner_sequence,
                "task_id": probe.task_id,
            }
            for probe, disposition in rows
        ]
        final_child = next(
            (
                probe.learner_child_digest
                for probe, _ in reversed(rows)
                if probe.learner_child_digest is not None
            ),
            rows[0][0].learner_genesis_digest,
        )
        if (
            lineage["transitions"] != derived_transitions
            or lineage["genesis_digest"] != rows[0][0].learner_genesis_digest
            or lineage["final_child_digest"] != final_child
            or lineage["advanced"] != (
                final_child != rows[0][0].learner_genesis_digest
            )
            or any(
                probe.learner_genesis_digest != lineage["genesis_digest"]
                for probe, _ in rows
            )
        ):
            raise RunnerInvariantError(
                "serialized lineage differs from adaptation probe evidence"
            )
    advanced_count = 0
    if purpose == "evaluation":
        if value["full_adaptation_lineage"] is None or value["full_adaptation_lineage_ref"] is None:
            raise RunnerInvariantError("serialized evaluation lacks FULL lineages")
        full, advanced_count = _validate_serialized_full_lineage(
            value["full_adaptation_lineage"],  # type: ignore[arg-type]
            adaptation_rows,
        )
        if value["full_adaptation_lineage_ref"] != content_ref("full-adaptation-lineages", full):
            raise RunnerInvariantError("serialized FULL-lineage reference differs")
    elif value["full_adaptation_lineage"] is not None or value["full_adaptation_lineage_ref"] is not None:
        raise RunnerInvariantError("serialized qualification claims FULL-lineage subset")
    removal = _validate_removal_records(
        value["removal_fairness"],  # type: ignore[arg-type]
        purpose=purpose,
    )
    refs = [record["evidence_ref"] for record in removal]
    if value["removal_fairness_refs"] != refs:
        raise RunnerInvariantError("serialized removal-fairness references differ")
    removal_identities = {
        (record["phase"], record["task_id"])
        for record in removal
    }
    expected_removal_identities = {
        (probe.phase, probe.task_id)
        for probe in observed_probes
        if probe.arm == "FULL" and probe.phase in ("development", "final")
    }
    if removal_identities != expected_removal_identities or any(
        not all(
            (phase, task_id, arm) in observed_probe_identities
            for arm in (
                "FULL",
                "FROZEN_ORIGIN",
                "PROSPECTIVE_REMOVAL",
                "BACKEND_REMOVAL",
                "RETRIEVAL_ONLY",
            )
        )
        for phase, task_id in removal_identities
    ):
        raise RunnerInvariantError("serialized removal evidence differs from attempts")
    if any(
        probe.foundation_tensor_digest != tensor_digest
        for probe in observed_probes
    ):
        raise RunnerInvariantError("serialized probes differ from foundation tensor")
    canonical = decode_canonical_json(
        canonical_json_bytes(value),
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )
    if type(canonical) is not dict:
        raise RunnerInvariantError("serialized run-integrity copy differs")
    scan_for_hidden_material(canonical, maximum_bytes=MAXIMUM_RESULT_BYTES)
    return canonical, advanced_count


def classify_evaluation_result(
    metrics: Mapping[str, object],
    run_integrity: Mapping[str, object],
) -> tuple[str, dict[str, object], str]:
    """Apply only the frozen integer predicates to a validated aggregate."""

    canonical_metrics = _validate_final_metrics(metrics)
    canonical_integrity, advanced_count = validate_serialized_run_integrity(
        run_integrity,
        purpose="evaluation",
    )
    final_rows = {
        (row["arm"], row["family"]): row
        for row in canonical_metrics["aggregates"]
        if row["phase"] == "final"
    }
    final_totals = {
        arm: sum(
            final_rows[(arm, family)]["binary_success_total"]
            for family in FAMILIES
        )
        for arm in EVALUATION_ARMS
    }
    full = final_totals["FULL"]
    predicate_values = {
        "p1_full_at_least_14": full >= 14,
        "p2_full_minus_qwen_at_least_3": full - final_totals["QWEN_ONLY"] >= 3,
        "p3_full_minus_retrieval_at_least_3": full - final_totals["RETRIEVAL_ONLY"] >= 3,
        "p4_full_minus_random_at_least_2": full - final_totals["RANDOM_FEEDBACK"] >= 2,
        "p5_full_minus_prospective_at_least_2": full - final_totals["PROSPECTIVE_REMOVAL"] >= 2,
        "p6_full_minus_frozen_origin_at_least_1": full - final_totals["FROZEN_ORIGIN"] >= 1,
    }
    family_rows: list[dict[str, object]] = []
    strictly_better = 0
    no_family_worse = True
    family_denominators = {
        "symbolic-demonstration-transfer": 8,
        "glyph-machine": 8,
        "causal-operator": 24,
    }
    for family in FAMILIES:
        denominator = family_denominators[family]
        full_family = final_rows[("FULL", family)]["binary_success_total"]
        qwen_family = final_rows[("QWEN_ONLY", family)]["binary_success_total"]
        retrieval_family = final_rows[("RETRIEVAL_ONLY", family)]["binary_success_total"]
        better_control = max(qwen_family, retrieval_family)
        is_better = full_family > better_control
        within_floor = 8 * (better_control - full_family) <= denominator
        strictly_better += int(is_better)
        no_family_worse = no_family_worse and within_floor
        family_rows.append(
            {
                "better_control_total": better_control,
                "denominator": denominator,
                "family": family,
                "full_total": full_family,
                "full_strictly_better": is_better,
                "within_one_eighth_floor": within_floor,
            }
        )
    predicate_values["p7_family_control_margin"] = strictly_better >= 2 and no_family_worse
    predicate_values["p8_removal_integrity"] = (
        len(canonical_integrity["removal_fairness"]) == 60
    )
    predicate_values["p9_two_advanced_full_lineages"] = advanced_count == 2
    all_passed = all(predicate_values.values())
    classification = (
        "EXPERIMENTALLY_SUPPORTED"
        if all_passed
        else "NOT_SUPPORTED"
    )
    witness = {
        "advanced_full_lineages": advanced_count,
        "all_predicates_passed": all_passed,
        "classification": classification,
        "family_witnesses": family_rows,
        "final_success_totals": final_totals,
        "metrics_ref": content_ref("final-metrics", canonical_metrics),
        "predicates": predicate_values,
        "run_integrity_ref": content_ref("run-integrity", canonical_integrity),
        "schema": RUN_CLASSIFIER_SCHEMA,
        "strictly_better_family_count": strictly_better,
        "thresholds": {
            "p1_full_minimum_numerator_over_40": 14,
            "p2_full_minus_qwen_minimum": 3,
            "p3_full_minus_retrieval_minimum": 3,
            "p4_full_minus_random_minimum": 2,
            "p5_full_minus_prospective_minimum": 2,
            "p6_full_minus_frozen_origin_minimum": 1,
            "p7_family_deficit_denominator": 8,
            "p7_minimum_strict_wins": 2,
        },
    }
    canonical_json_bytes(witness)
    return classification, witness, content_ref("integer-classifier", witness)


def validate_qualification_result(
    value: Mapping[str, object],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Strictly validate canonical qualification-PASS result content."""

    expected_fields = {
        "accounting",
        "adaptation_lineage",
        "adaptation_lineage_ref",
        "arm_tasks",
        "attempt_receipt_refs",
        "classification",
        "foundation_integrity",
        "foundation_integrity_ref",
        "generation_attempt_ceiling",
        "identity",
        "manifest_sha256",
        "purpose",
        "removal_fairness_refs",
        "run_integrity",
        "run_integrity_ref",
        "schema",
        "scientific_claim",
    }
    if type(value) is not dict or set(value) != expected_fields or (
        value["schema"] != QUALIFICATION_RESULT_SCHEMA
        or value["purpose"] != "qualification"
        or value["identity"] != QUALIFICATION_IDENTITY
        or value["classification"] != "QUALIFICATION_PASS"
        or value["scientific_claim"] is not False
        or value["arm_tasks"] != QUALIFICATION_COUNTS[0]
        or value["generation_attempt_ceiling"] != QUALIFICATION_COUNTS[1]
    ):
        raise RunnerInvariantError("qualification result identity differs")
    manifest_hash = _raw_digest(
        value["manifest_sha256"],
        "qualification manifest sha256",
    )
    if expected_manifest_sha256 is not None and manifest_hash != _raw_digest(
        expected_manifest_sha256,
        "expected qualification manifest sha256",
    ):
        raise RunnerInvariantError("qualification result manifest differs")
    accounting = validate_serialized_run_accounting(
        value["accounting"],  # type: ignore[arg-type]
        RunMode.QUALIFICATION,
    )
    integrity, _ = validate_serialized_run_integrity(
        value["run_integrity"],  # type: ignore[arg-type]
        purpose="qualification",
    )
    if value["run_integrity_ref"] != content_ref("run-integrity", integrity):
        raise RunnerInvariantError("qualification run-integrity reference differs")
    attempts = integrity["attempts"]
    attempt_refs = value["attempt_receipt_refs"]
    if (
        type(attempt_refs) is not list
        or attempt_refs != [row["attempt_receipt_ref"] for row in attempts]
        or len(set(attempt_refs)) != QUALIFICATION_COUNTS[0]
        or value["accounting"] != accounting
        or integrity["resource_accounting"] != accounting
        or value["adaptation_lineage"] != integrity["adaptation_lineage"]
        or value["adaptation_lineage_ref"] != integrity["adaptation_lineage_ref"]
        or value["foundation_integrity"] != integrity["foundation"]
        or value["foundation_integrity_ref"] != integrity["foundation_ref"]
        or value["removal_fairness_refs"] != integrity["removal_fairness_refs"]
    ):
        raise RunnerInvariantError("qualification result duplicated evidence differs")
    canonical = decode_canonical_json(
        canonical_json_bytes(value),
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )
    if type(canonical) is not dict:
        raise RunnerInvariantError("qualification result canonical copy differs")
    scan_for_hidden_material(canonical, maximum_bytes=MAXIMUM_RESULT_BYTES)
    return canonical


def _validate_evaluation_result(
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_manifest_sha256: str | None = None,
    expected_admission_claim_ref: str | None = None,
    allow_injected_claim: bool,
) -> dict[str, object]:
    """Strictly recompute every public completed-evaluation result binding."""

    expected_fields = {
        "accounting",
        "adaptation_lineage",
        "adaptation_lineage_ref",
        "admission_claim",
        "admission_claim_ref",
        "arm_tasks",
        "attempt_receipt_refs",
        "classification",
        "classifier",
        "classifier_ref",
        "evaluation_metrics",
        "evaluation_metrics_ref",
        "foundation_integrity",
        "foundation_integrity_ref",
        "full_adaptation_lineage",
        "full_adaptation_lineage_ref",
        "generation_attempt_ceiling",
        "identity",
        "manifest_sha256",
        "purpose",
        "removal_fairness_refs",
        "run_integrity",
        "run_integrity_ref",
        "schema",
        "scientific_claim",
    }
    if type(value) is not dict or set(value) != expected_fields or (
        value["schema"] != EVALUATION_RESULT_SCHEMA
        or value["purpose"] != "evaluation"
        or value["identity"] != EVALUATION_IDENTITY
        or value["classification"]
        not in ("EXPERIMENTALLY_SUPPORTED", "NOT_SUPPORTED")
        or value["scientific_claim"] != "BOUNDED_SYNTHETIC_EXPERIMENT_ONLY"
        or value["arm_tasks"] != EVALUATION_COUNTS[0]
        or value["generation_attempt_ceiling"] != EVALUATION_COUNTS[1]
    ):
        raise RunnerInvariantError("evaluation result identity differs")
    manifest_hash = _raw_digest(
        value["manifest_sha256"],
        "evaluation result manifest sha256",
    )
    if expected_manifest_sha256 is not None and manifest_hash != _raw_digest(
        expected_manifest_sha256,
        "expected evaluation manifest sha256",
    ):
        raise RunnerInvariantError("evaluation result manifest differs")
    if not isinstance(manifest, Mapping):
        raise TypeError("evaluation result audit requires its exact manifest")
    observed_manifest_hash = validate_source_manifest(
        manifest,
        expected_hash=manifest_hash,
        repository_root=repository_root,
    )
    if observed_manifest_hash != manifest_hash:
        raise RunnerInvariantError("evaluation result manifest audit differs")
    claim, claim_ref = validate_evaluation_admission_claim(
        value["admission_claim"],  # type: ignore[arg-type]
        value["admission_claim_ref"],  # type: ignore[arg-type]
    )
    if (
        claim["manifest_sha256"] != manifest_hash
        or claim["source_hashes"] != manifest.get("frozen_sources")
        or claim["evaluator_commitment_digest"]
        != manifest.get("evaluator_commitment_digest")
    ):
        raise RunnerInvariantError("evaluation result admission manifest differs")
    if allow_injected_claim:
        if claim["path_binding_mode"] != "INJECTED_CPU_TEST":
            raise RunnerInvariantError("injected result claim mode differs")
    else:
        _validate_live_claim_manifest_binding(
            claim,
            manifest=manifest,
            manifest_sha256=manifest_hash,
        )
    if expected_admission_claim_ref is not None and claim_ref != _digest(
        expected_admission_claim_ref,
        "expected evaluation admission claim reference",
    ):
        raise RunnerInvariantError("evaluation result admission identity differs")
    accounting = validate_serialized_run_accounting(
        value["accounting"],  # type: ignore[arg-type]
        RunMode.EVALUATION,
    )
    integrity, _ = validate_serialized_run_integrity(
        value["run_integrity"],  # type: ignore[arg-type]
        purpose="evaluation",
    )
    integrity_ref = content_ref("run-integrity", integrity)
    if value["run_integrity_ref"] != integrity_ref:
        raise RunnerInvariantError("evaluation run-integrity reference differs")
    commitments = manifest.get("evaluator_commitments")
    if type(commitments) is not dict or type(commitments.get("tasks")) is not list:
        raise RunnerInvariantError("evaluation manifest task schedule differs")
    manifest_tasks = commitments["tasks"]
    expected_schedule: list[tuple[object, object, str, object]] = []
    expected_removal_order: list[tuple[object, object]] = []
    phase_counts: Counter[str] = Counter()
    task_ids: set[object] = set()
    for task in manifest_tasks:
        if type(task) is not dict or set(task) != {
            "family",
            "ordinal",
            "phase",
            "private_commitment",
            "public_commitment",
            "replicate_commitment",
            "task_id",
        }:
            raise RunnerInvariantError("evaluation manifest task fields differ")
        phase = task["phase"]
        task_id = task["task_id"]
        replicate = task["replicate_commitment"]
        if (
            phase not in PHASES
            or task["family"] not in FAMILIES
            or type(task["ordinal"]) is not int
            or task["ordinal"] < 0
            or task_id in task_ids
        ):
            raise RunnerInvariantError("evaluation manifest task identity differs")
        _digest(task_id, "evaluation manifest task_id")
        _digest(replicate, "evaluation manifest replicate")
        _digest(task["private_commitment"], "evaluation private commitment")
        _digest(task["public_commitment"], "evaluation public commitment")
        task_ids.add(task_id)
        phase_counts[phase] += 1
    if phase_counts != Counter({"adaptation": 24, "development": 20, "final": 40}):
        raise RunnerInvariantError("evaluation manifest phase coverage differs")
    for phase in PHASES:
        for task in manifest_tasks:
            if task["phase"] != phase:
                continue
            arms = (
                ("FULL", "RANDOM_FEEDBACK")
                if phase == "adaptation"
                else EVALUATION_ARMS
            )
            expected_schedule.extend(
                (
                    phase,
                    task["task_id"],
                    arm,
                    task["replicate_commitment"],
                )
                for arm in arms
            )
            if phase in ("development", "final"):
                expected_removal_order.append((phase, task["task_id"]))
    observed_schedule = [
        (
            row["probe_integrity"]["phase"],
            row["task_id"],
            row["arm"],
            row["probe_integrity"]["replicate_commitment"],
        )
        for row in integrity["attempts"]
    ]
    observed_removal_order = [
        (row["phase"], row["task_id"])
        for row in integrity["removal_fairness"]
    ]
    if (
        observed_schedule != expected_schedule
        or observed_removal_order != expected_removal_order
    ):
        raise RunnerInvariantError("evaluation result schedule differs from manifest")
    metrics = _validate_final_metrics(value["evaluation_metrics"])  # type: ignore[arg-type]
    metrics_ref = content_ref("final-metrics", metrics)
    classification, classifier, classifier_ref = classify_evaluation_result(
        metrics,
        integrity,
    )
    attempt_refs = value["attempt_receipt_refs"]
    if (
        type(attempt_refs) is not list
        or attempt_refs != [
            row["attempt_receipt_ref"] for row in integrity["attempts"]
        ]
        or len(set(attempt_refs)) != EVALUATION_COUNTS[0]
        or value["classification"] != classification
        or value["classifier"] != classifier
        or value["classifier_ref"] != classifier_ref
        or value["evaluation_metrics_ref"] != metrics_ref
        or value["accounting"] != accounting
        or integrity["resource_accounting"] != accounting
        or value["adaptation_lineage"] != integrity["adaptation_lineage"]
        or value["adaptation_lineage_ref"] != integrity["adaptation_lineage_ref"]
        or value["full_adaptation_lineage"]
        != integrity["full_adaptation_lineage"]
        or value["full_adaptation_lineage_ref"]
        != integrity["full_adaptation_lineage_ref"]
        or value["foundation_integrity"] != integrity["foundation"]
        or value["foundation_integrity_ref"] != integrity["foundation_ref"]
        or value["removal_fairness_refs"] != integrity["removal_fairness_refs"]
    ):
        raise RunnerInvariantError("evaluation result recomputation differs")
    canonical = decode_canonical_json(
        canonical_json_bytes(value),
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )
    if type(canonical) is not dict:
        raise RunnerInvariantError("evaluation result canonical copy differs")
    scan_for_hidden_material(canonical, maximum_bytes=MAXIMUM_RESULT_BYTES)
    return canonical


def validate_evaluation_result(
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_manifest_sha256: str | None = None,
    expected_admission_claim_ref: str | None = None,
) -> dict[str, object]:
    return _validate_evaluation_result(
        value,
        manifest=manifest,
        repository_root=repository_root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_admission_claim_ref=expected_admission_claim_ref,
        allow_injected_claim=False,
    )


def _validate_evaluation_result_injected(
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_manifest_sha256: str | None = None,
    expected_admission_claim_ref: str | None = None,
) -> dict[str, object]:
    """CPU/fake validator; its output is deliberately not publishable."""

    return _validate_evaluation_result(
        value,
        manifest=manifest,
        repository_root=repository_root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_admission_claim_ref=expected_admission_claim_ref,
        allow_injected_claim=True,
    )


def load_qualification_result(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    target = Path(path)
    raw = _read_private_artifact(target, maximum_bytes=MAXIMUM_RESULT_BYTES)
    value = decode_canonical_json(raw, maximum_bytes=MAXIMUM_RESULT_BYTES)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise RunnerInvariantError("qualification result bytes are not canonical")
    canonical = validate_qualification_result(
        value,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return canonical, hashlib.sha256(raw).hexdigest()


def publish_qualification_result(
    path: str | Path,
    value: Mapping[str, object],
    *,
    expected_manifest_sha256: str,
) -> Path:
    canonical = validate_qualification_result(
        value,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    target = _publish_canonical_create_once(
        path,
        canonical,
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )
    loaded, _ = load_qualification_result(
        target,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if loaded != canonical:
        raise RunnerInvariantError("published qualification result differs")
    return target


def load_evaluation_result(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_manifest_sha256: str | None = None,
    expected_admission_claim_ref: str,
    admission_claim_path: str | Path = EVALUATION_ADMISSION_CLAIM_PATH,
) -> tuple[dict[str, object], str]:
    target = Path(path)
    manifest_hash = validate_source_manifest(
        manifest,
        expected_hash=expected_manifest_sha256,
        repository_root=repository_root,
    )
    durable_claim, durable_ref = load_evaluation_admission_claim(
        admission_claim_path,
        expected_ref=expected_admission_claim_ref,
    )
    _validate_live_claim_manifest_binding(
        durable_claim,
        manifest=manifest,
        manifest_sha256=manifest_hash,
    )
    if target != Path(durable_claim["paths"]["evaluation_result"]):  # type: ignore[index]
        raise RunnerInvariantError("evaluation result path differs from admission")
    raw = _read_private_artifact(target, maximum_bytes=MAXIMUM_RESULT_BYTES)
    value = decode_canonical_json(raw, maximum_bytes=MAXIMUM_RESULT_BYTES)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise RunnerInvariantError("evaluation result bytes are not canonical")
    canonical = validate_evaluation_result(
        value,
        manifest=manifest,
        repository_root=repository_root,
        expected_manifest_sha256=manifest_hash,
        expected_admission_claim_ref=durable_ref,
    )
    if (
        canonical["admission_claim"] != durable_claim
        or canonical["admission_claim_ref"] != durable_ref
    ):
        raise RunnerInvariantError("evaluation result differs from durable admission")
    return canonical, hashlib.sha256(raw).hexdigest()


def publish_evaluation_result(
    path: str | Path,
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_admission_claim_ref: str,
    admission_claim_path: str | Path = EVALUATION_ADMISSION_CLAIM_PATH,
) -> Path:
    manifest_hash = validate_source_manifest(
        manifest,
        repository_root=repository_root,
    )
    durable_claim, durable_ref = load_evaluation_admission_claim(
        admission_claim_path,
        expected_ref=expected_admission_claim_ref,
    )
    _validate_live_claim_manifest_binding(
        durable_claim,
        manifest=manifest,
        manifest_sha256=manifest_hash,
    )
    if Path(path) != Path(durable_claim["paths"]["evaluation_result"]):  # type: ignore[index]
        raise RunnerInvariantError("evaluation publication path differs from admission")
    canonical = validate_evaluation_result(
        value,
        manifest=manifest,
        repository_root=repository_root,
        expected_manifest_sha256=manifest_hash,
        expected_admission_claim_ref=durable_ref,
    )
    if (
        canonical["admission_claim"] != durable_claim
        or canonical["admission_claim_ref"] != durable_ref
    ):
        raise RunnerInvariantError("evaluation publication differs from durable claim")
    return _publish_canonical_create_once(
        path,
        canonical,
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )


def validate_failure_disposition(
    value: Mapping[str, object],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "admission_validated",
        "category",
        "cleanup_codes",
        "code",
        "evaluator_cleanup_completed",
        "exception_type",
        "schema",
        "stage",
    } or value["schema"] != FAILURE_DISPOSITION_SCHEMA:
        raise RunnerInvariantError("failure disposition fields differ")
    if (
        value["category"]
        not in ("INVARIANT", "INFRASTRUCTURE", "CANCELLED", "CLEANUP")
        or type(value["admission_validated"]) is not bool
        or type(value["evaluator_cleanup_completed"]) is not bool
    ):
        raise RunnerInvariantError("failure disposition values differ")
    cleanup_codes = value["cleanup_codes"]
    if type(cleanup_codes) is not list or any(
        type(code) is not str or _CANONICAL_ID.fullmatch(code) is None
        for code in cleanup_codes
    ):
        raise RunnerInvariantError("failure cleanup codes differ")
    if not (
        (value["category"] == "CLEANUP")
        == (value["evaluator_cleanup_completed"] is False)
        == bool(cleanup_codes)
    ):
        raise RunnerInvariantError("failure cleanup completion contradicts category")
    _canonical_identifier(value["stage"], "failure stage")
    _canonical_identifier(value["code"], "failure code")
    _canonical_identifier(value["exception_type"], "failure exception type")
    canonical = decode_canonical_json(canonical_json_bytes(value))
    if type(canonical) is not dict:
        raise RunnerInvariantError("failure disposition canonical copy differs")
    scan_for_hidden_material(canonical)
    return canonical


def _exception_leaves(error: BaseException) -> list[BaseException]:
    leaves: list[BaseException] = []
    visited: set[int] = set()

    def walk(current: BaseException) -> None:
        identity = id(current)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(current, BaseExceptionGroup):
            for nested in current.exceptions:
                walk(nested)
        else:
            leaves.append(current)
        # Explicit bounded wrappers are semantic effect-boundary leaves.  Their
        # suppressed implementation causes may contain raw OS/private detail
        # but cannot change the declared disposition.
        if isinstance(
            current,
            (InfrastructureFailure, EvaluatorTransportError, CleanupFailure),
        ):
            return
        cause = current.__cause__
        context = current.__context__
        if isinstance(cause, BaseException):
            walk(cause)
        elif isinstance(context, BaseException) and not current.__suppress_context__:
            walk(context)

    walk(error)
    return leaves


def _typed_live_infrastructure(
    error: BaseException,
    *,
    code: str,
    stage: str,
) -> BaseException:
    """Erase raw infrastructure detail at the explicit live effect boundary."""

    if isinstance(error, BaseExceptionGroup):
        transformed = tuple(
            _typed_live_infrastructure(item, code=code, stage=stage)
            for item in error.exceptions
        )
        if all(
            observed is original
            for observed, original in zip(
                transformed,
                error.exceptions,
                strict=True,
            )
        ):
            return error
        return error.derive(transformed)
    if isinstance(error, InfrastructureFailure):
        return error
    if isinstance(error, EvaluatorTransportError):
        return InfrastructureFailure("EVALUATOR_TRANSPORT_FAILED", "EVALUATOR_IO")
    if isinstance(error, (OSError, TimeoutError, MemoryError)):
        return InfrastructureFailure(code, stage)
    error_type = type(error)
    if (
        error_type.__name__ == "OutOfMemoryError"
        and error_type.__module__.startswith("torch")
    ):
        return InfrastructureFailure("CUDA_OUT_OF_MEMORY", "RESOURCE_GUARD")
    return error


def classify_exception_disposition(
    error: BaseException,
    *,
    admission_validated: bool,
    evaluator_cleanup_completed: bool,
) -> tuple[str | None, dict[str, object] | None]:
    """Classify recursively without serializing messages, tracebacks, or bytes."""

    if not isinstance(error, BaseException):
        raise TypeError("failure classifier requires an exception")
    if type(admission_validated) is not bool or type(evaluator_cleanup_completed) is not bool:
        raise TypeError("failure classifier state must be exact bools")
    leaves = _exception_leaves(error)
    cleanup = [item for item in leaves if isinstance(item, CleanupFailure)]
    if admission_validated and not evaluator_cleanup_completed and not cleanup:
        cleanup = [
            CleanupFailure(
                "EVALUATOR_CLEANUP_INCOMPLETE",
                "EVALUATOR_CLEANUP",
            )
        ]
    non_cleanup = [item for item in leaves if not isinstance(item, CleanupFailure)]
    infrastructure_types = (
        InfrastructureFailure,
        EvaluatorTransportError,
        asyncio.CancelledError,
    )
    invariant_leaves = [
        item for item in non_cleanup if not isinstance(item, infrastructure_types)
    ]
    primary = (
        invariant_leaves[0]
        if invariant_leaves
        else non_cleanup[0]
        if non_cleanup
        else leaves[0]
    )
    typed_infrastructure = bool(non_cleanup) and not invariant_leaves
    cancelled = isinstance(primary, asyncio.CancelledError)
    if cleanup:
        classification: str | None = "INVALID"
        category = "CLEANUP"
    elif typed_infrastructure and admission_validated:
        classification = "INCONCLUSIVE"
        category = "CANCELLED" if cancelled else "INFRASTRUCTURE"
    elif typed_infrastructure:
        # Pre-admission infrastructure is retryable and has no evaluation result.
        return None, None
    else:
        classification = "INVALID"
        category = "INVARIANT"
    stage = getattr(primary, "stage", "ORCHESTRATION")
    code = getattr(primary, "code", type(primary).__name__)
    failure = {
        "admission_validated": admission_validated,
        "category": category,
        "cleanup_codes": [item.code for item in cleanup],
        "code": _canonical_identifier(code, "classified failure code"),
        "evaluator_cleanup_completed": evaluator_cleanup_completed and not cleanup,
        "exception_type": _canonical_identifier(
            type(primary).__name__,
            "classified exception type",
        ),
        "schema": FAILURE_DISPOSITION_SCHEMA,
        "stage": _canonical_identifier(stage, "classified failure stage"),
    }
    return classification, validate_failure_disposition(failure)


def validate_qualification_failure_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "classification",
        "failure",
        "identity",
        "manifest_sha256",
        "purpose",
        "schema",
        "scientific_claim",
    } or (
        value["schema"] != QUALIFICATION_RESULT_SCHEMA
        or value["purpose"] != "qualification"
        or value["identity"] != QUALIFICATION_IDENTITY
        or value["classification"] != "QUALIFICATION_FAILURE"
        or value["scientific_claim"] is not False
    ):
        raise RunnerInvariantError("qualification failure identity differs")
    if value["manifest_sha256"] is not None:
        _raw_digest(value["manifest_sha256"], "qualification failure manifest")
    failure = validate_failure_disposition(value["failure"])  # type: ignore[arg-type]
    if failure["admission_validated"] is not False:
        raise RunnerInvariantError(
            "qualification failure cannot carry evaluation admission"
        )
    canonical = decode_canonical_json(canonical_json_bytes(value))
    if type(canonical) is not dict:
        raise RunnerInvariantError("qualification failure canonical copy differs")
    return canonical


def validate_evaluation_failure_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "admission_claim",
        "admission_claim_ref",
        "classification",
        "failure",
        "identity",
        "manifest_sha256",
        "purpose",
        "schema",
        "scientific_claim",
    } or (
        value["schema"] != EVALUATION_RESULT_SCHEMA
        or value["purpose"] != "evaluation"
        or value["identity"] != EVALUATION_IDENTITY
        or value["classification"] not in ("INVALID", "INCONCLUSIVE")
        or value["scientific_claim"] != "BOUNDED_SYNTHETIC_EXPERIMENT_ONLY"
    ):
        raise RunnerInvariantError("evaluation failure identity differs")
    failure = validate_failure_disposition(value["failure"])  # type: ignore[arg-type]
    claim = value["admission_claim"]
    claim_ref = value["admission_claim_ref"]
    if claim is None or claim_ref is None:
        if claim is not None or claim_ref is not None:
            raise RunnerInvariantError("evaluation failure admission is partial")
        if failure["admission_validated"] is not False:
            raise RunnerInvariantError("evaluation failure admission state differs")
    else:
        validated_claim, validated_ref = validate_evaluation_admission_claim(
            claim,  # type: ignore[arg-type]
            claim_ref,  # type: ignore[arg-type]
        )
        if failure["admission_validated"] is not True:
            raise RunnerInvariantError("evaluation failure omits validated admission")
        if value["manifest_sha256"] != validated_claim["manifest_sha256"]:
            raise RunnerInvariantError("evaluation failure manifest/claim differ")
        if validated_ref != claim_ref:
            raise RunnerInvariantError("evaluation failure claim reference differs")
        if (
            validated_claim["path_binding_mode"] != "FROZEN_LIVE"
            or validated_claim["paths"] != _frozen_live_claim_paths()
        ):
            raise RunnerInvariantError(
                "evaluation failure admission is not the frozen live claim"
            )
    if value["manifest_sha256"] is not None:
        _raw_digest(value["manifest_sha256"], "evaluation failure manifest")
    if value["classification"] == "INCONCLUSIVE" and (
        failure["admission_validated"] is not True
        or failure["category"] not in ("INFRASTRUCTURE", "CANCELLED")
        or failure["evaluator_cleanup_completed"] is not True
    ):
        raise RunnerInvariantError("INCONCLUSIVE failure boundary differs")
    if failure["category"] in ("INFRASTRUCTURE", "CANCELLED") and (
        failure["admission_validated"] is not True
    ):
        raise RunnerInvariantError(
            "pre-admission infrastructure cannot publish an evaluation result"
        )
    if value["classification"] == "INVALID" and failure["category"] in (
        "INFRASTRUCTURE",
        "CANCELLED",
    ) and failure["admission_validated"] is True:
        raise RunnerInvariantError("post-admission infrastructure was misclassified")
    canonical = decode_canonical_json(canonical_json_bytes(value))
    if type(canonical) is not dict:
        raise RunnerInvariantError("evaluation failure canonical copy differs")
    return canonical


def build_evaluation_failure_result(
    *,
    classification: Literal["INVALID", "INCONCLUSIVE"],
    failure: Mapping[str, object],
    admission_claim: Mapping[str, object] | None,
    admission_claim_ref: str | None,
    manifest_sha256: str | None,
) -> dict[str, object]:
    result = {
        "admission_claim": (
            None if admission_claim is None else dict(admission_claim)
        ),
        "admission_claim_ref": admission_claim_ref,
        "classification": classification,
        "failure": dict(failure),
        "identity": EVALUATION_IDENTITY,
        "manifest_sha256": manifest_sha256,
        "purpose": "evaluation",
        "schema": EVALUATION_RESULT_SCHEMA,
        "scientific_claim": "BOUNDED_SYNTHETIC_EXPERIMENT_ONLY",
    }
    return validate_evaluation_failure_result(result)


def build_qualification_failure_result(
    *,
    failure: Mapping[str, object],
    manifest_sha256: str | None,
) -> dict[str, object]:
    result = {
        "classification": "QUALIFICATION_FAILURE",
        "failure": dict(failure),
        "identity": QUALIFICATION_IDENTITY,
        "manifest_sha256": manifest_sha256,
        "purpose": "qualification",
        "schema": QUALIFICATION_RESULT_SCHEMA,
        "scientific_claim": False,
    }
    return validate_qualification_failure_result(result)


def publish_qualification_failure_result(
    path: str | Path,
    value: Mapping[str, object],
) -> Path:
    return _publish_canonical_create_once(
        path,
        validate_qualification_failure_result(value),
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )


def publish_evaluation_failure_result(
    path: str | Path,
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object] | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
    expected_admission_claim_ref: str | None = None,
    admission_claim_path: str | Path = EVALUATION_ADMISSION_CLAIM_PATH,
) -> Path:
    canonical = validate_evaluation_failure_result(value)
    admitted = canonical["failure"]["admission_validated"]  # type: ignore[index]
    if admitted:
        if expected_admission_claim_ref is None:
            raise RunnerInvariantError(
                "post-admission failure publication requires durable admission"
            )
        durable_claim, durable_ref = load_evaluation_admission_claim(
            admission_claim_path,
            expected_ref=expected_admission_claim_ref,
        )
        if manifest is not None:
            manifest_hash = validate_source_manifest(
                manifest,
                expected_hash=canonical["manifest_sha256"],  # type: ignore[arg-type]
                repository_root=repository_root,
            )
            _validate_live_claim_manifest_binding(
                durable_claim,
                manifest=manifest,
                manifest_sha256=manifest_hash,
            )
        if (
            canonical["admission_claim"] != durable_claim
            or canonical["admission_claim_ref"] != durable_ref
            or Path(path)
            != Path(durable_claim["paths"]["evaluation_result"])  # type: ignore[index]
        ):
            raise RunnerInvariantError(
                "post-admission failure differs from durable claim"
            )
    elif manifest is not None or expected_admission_claim_ref is not None:
        raise RunnerInvariantError(
            "pre-admission failure cannot borrow durable admission"
        )
    return _publish_canonical_create_once(
        path,
        canonical,
        maximum_bytes=MAXIMUM_RESULT_BYTES,
    )


async def _foundation_snapshot(
    dependencies: OrchestrationDependencies,
) -> dict[str, str]:
    observed = await _maybe_await(dependencies.foundation_probe())
    if not isinstance(observed, Mapping):
        raise TypeError("foundation probe must return a hash mapping")
    snapshot = dict(observed)
    validate_foundation_hashes(snapshot, snapshot, snapshot)
    return snapshot


async def _run_qualification(
    dependencies: OrchestrationDependencies,
) -> dict[str, object]:
    """Execute the exact 31-arm-task non-scientific qualification schedule."""

    if type(dependencies) is not OrchestrationDependencies:
        raise TypeError("dependencies must be exact OrchestrationDependencies")
    if dependencies.accounting.mode is not RunMode.QUALIFICATION:
        raise RunnerInvariantError("qualification accounting mode differs")
    manifest_sha256 = validate_source_manifest(
        dependencies.manifest,
        expected_hash=dependencies.expected_manifest_sha256,
        repository_root=dependencies.repository_root,
    )
    foundation_before = await _foundation_snapshot(dependencies)
    validate_foundation_hashes(
        foundation_before,
        foundation_before,
        dependencies.expected_foundation_hashes,
    )
    evaluator = dependencies.evaluator
    attempts: list[ArmTaskResult] = []
    adaptation = await _release(evaluator, "adaptation")
    if len(adaptation) != 12:
        raise RunnerInvariantError("qualification adaptation count differs")
    schedules: dict[str, tuple[tuple[str, float], ...]] = {}
    for task in adaptation:
        replicate = str(task["replicate_commitment"])
        if replicate not in schedules:
            schedules[replicate] = await _random_schedule(evaluator, replicate)
        _append_unique_attempt(
            attempts,
            await _execute_scheduled(
                dependencies,
                ArmTaskSpec(
                    purpose="qualification",
                    phase="adaptation",
                    arm="FULL",
                    task=task,
                    judge_arm="QUALIFICATION",
                ),
            )
        )
        _append_unique_attempt(
            attempts,
            await _execute_scheduled(
                dependencies,
                ArmTaskSpec(
                    purpose="qualification",
                    phase="adaptation",
                    arm="RANDOM_FEEDBACK",
                    task=task,
                    judge_arm=None,
                    feedback_value=random_feedback_scalar(
                        schedules[replicate], str(task["task_id"])
                    ),
                ),
            )
        )
    await _complete(evaluator, "adaptation")
    development = await _release(evaluator, "development")
    if len(development) != 10:
        raise RunnerInvariantError("qualification development count differs")
    first = development[0]
    removal_group: list[ArmTaskResult] = []
    removal_fairness: list[dict[str, object]] = []
    for arm in EVALUATION_ARMS:
        result = await _execute_scheduled(
                dependencies,
                ArmTaskSpec(
                    purpose="qualification",
                    phase="development",
                    arm=arm,
                    task=first,
                    judge_arm=("QUALIFICATION" if arm == "FULL" else None),
                ),
            )
        _append_unique_attempt(attempts, result)
        removal_group.append(result)
    removal_fairness.append(
        _validate_removal_group(removal_group, require_score=False)
    )
    if len(attempts) != QUALIFICATION_COUNTS[0]:
        raise RunnerInvariantError("qualification arm-task schedule differs")
    _validate_unique_attempts(attempts)
    adaptation_lineage, adaptation_lineage_ref = validate_adaptation_lineages(
        attempts,
        dependencies.expected_learner_genesis_digests,
        purpose="qualification",
    )
    foundation_after = await _foundation_snapshot(dependencies)
    validate_foundation_hashes(
        foundation_before,
        foundation_after,
        dependencies.expected_foundation_hashes,
    )
    accounting = dependencies.accounting.validate_complete()
    (
        run_integrity,
        run_integrity_ref,
        foundation_integrity,
        foundation_integrity_ref,
    ) = build_run_integrity_evidence(
        purpose="qualification",
        attempts=attempts,
        removal_fairness=removal_fairness,
        foundation_before=foundation_before,
        foundation_after=foundation_after,
        expected_foundation=dependencies.expected_foundation_hashes,
        expected_foundation_tensor_digest=(
            dependencies.expected_foundation_tensor_digest
        ),
        expected_learner_genesis_digests=(
            dependencies.expected_learner_genesis_digests
        ),
        adaptation_lineage=adaptation_lineage,
        adaptation_lineage_ref=adaptation_lineage_ref,
        resource_accounting=accounting,
    )
    await _close_evaluator(evaluator)
    return {
        "accounting": accounting,
        "adaptation_lineage": adaptation_lineage,
        "adaptation_lineage_ref": adaptation_lineage_ref,
        "arm_tasks": len(attempts),
        "attempt_receipt_refs": [item.attempt_receipt_ref for item in attempts],
        "classification": "QUALIFICATION_PASS",
        "foundation_integrity": foundation_integrity,
        "foundation_integrity_ref": foundation_integrity_ref,
        "generation_attempt_ceiling": QUALIFICATION_COUNTS[1],
        "identity": QUALIFICATION_IDENTITY,
        "manifest_sha256": manifest_sha256,
        "purpose": "qualification",
        "removal_fairness_refs": run_integrity["removal_fairness_refs"],
        "run_integrity": run_integrity,
        "run_integrity_ref": run_integrity_ref,
        "schema": QUALIFICATION_RESULT_SCHEMA,
        "scientific_claim": False,
    }


async def _run_evaluation(
    dependencies: OrchestrationDependencies,
    *,
    qualification_result_sha256: str,
    evaluator_commitment_digest: str,
    expected_admission_digest: str,
    durable_admission_claim: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute adaptation, development, and final without an operator gap."""

    if type(dependencies) is not OrchestrationDependencies:
        raise TypeError("dependencies must be exact OrchestrationDependencies")
    if dependencies.accounting.mode is not RunMode.EVALUATION:
        raise RunnerInvariantError("evaluation accounting mode differs")
    manifest_sha256 = validate_source_manifest(
        dependencies.manifest,
        expected_hash=dependencies.expected_manifest_sha256,
        repository_root=dependencies.repository_root,
    )
    foundation_before = await _foundation_snapshot(dependencies)
    validate_foundation_hashes(
        foundation_before,
        foundation_before,
        dependencies.expected_foundation_hashes,
    )
    sources = dependencies.manifest.get("frozen_sources")
    if type(sources) is not dict:
        raise RunnerInvariantError("manifest source map is malformed")
    if evaluator_commitment_digest != dependencies.manifest.get(
        "evaluator_commitment_digest"
    ):
        raise RunnerInvariantError("evaluation commitment digest differs from manifest")
    evaluator = dependencies.evaluator
    evaluator_snapshot = await _evaluator_commitment_snapshot(evaluator)
    if evaluator_snapshot != {
        "commitments": dependencies.manifest.get("evaluator_commitments"),
        "digest": dependencies.manifest.get("evaluator_commitment_digest"),
    }:
        raise RunnerInvariantError("live evaluator differs from manifest commitments")
    derived_admission = admission_digest(
        manifest_sha256=manifest_sha256,
        qualification_result_sha256=qualification_result_sha256,
        evaluator_commitment_digest=evaluator_commitment_digest,
        source_hashes=sources,
    )
    if derived_admission != _digest(expected_admission_digest, "expected admission"):
        raise RunnerInvariantError("evaluation admission binding differs")
    if durable_admission_claim is None:
        admission_claim, admission_claim_ref = build_evaluation_admission_claim(
            manifest_sha256=manifest_sha256,
            qualification_result_sha256=qualification_result_sha256,
            evaluator_commitment_digest=evaluator_commitment_digest,
            source_hashes=sources,
            manifest_path=MANIFEST_PATH,
            seed_path=SEALED_SEED_PATH,
            qualification_result_path=QUALIFICATION_RESULT_PATH,
            evaluation_result_path=EVALUATION_RESULT_PATH,
            claim_path=EVALUATION_ADMISSION_CLAIM_PATH,
            repository_root=dependencies.repository_root,
            seed_seal_ref=_digest(
                "sha256:" + hashlib.sha256(
                    b"INJECTED_CPU_TEST_SEED_SEAL"
                ).hexdigest(),
                "injected seed seal",
            ),
            path_binding_mode="INJECTED_CPU_TEST",
        )
    else:
        admission_claim, admission_claim_ref = validate_evaluation_admission_claim(
            durable_admission_claim,
        )
    if admission_claim["admission_digest"] != derived_admission:
        raise RunnerInvariantError("evaluation admission claim differs")
    attempts: list[ArmTaskResult] = []
    removal_fairness: list[dict[str, object]] = []
    adaptation = await _release(evaluator, "adaptation")
    if len(adaptation) != 24:
        raise RunnerInvariantError("evaluation adaptation count differs")
    schedules: dict[str, tuple[tuple[str, float], ...]] = {}
    for task in adaptation:
        replicate = str(task["replicate_commitment"])
        if replicate not in schedules:
            schedules[replicate] = await _random_schedule(evaluator, replicate)
        for arm in ("FULL", "RANDOM_FEEDBACK"):
            _append_unique_attempt(
                attempts,
                await _execute_scheduled(
                    dependencies,
                    ArmTaskSpec(
                        purpose="evaluation",
                        phase="adaptation",
                        arm=arm,
                        task=task,
                        judge_arm=arm,
                        feedback_value=(
                            random_feedback_scalar(
                                schedules[replicate], str(task["task_id"])
                            )
                            if arm == "RANDOM_FEEDBACK"
                            else None
                        ),
                    ),
                )
            )
    await _complete(evaluator, "adaptation")
    development = await _release(evaluator, "development")
    if len(development) != 20:
        raise RunnerInvariantError("evaluation development count differs")
    for task in development:
        removal_group = []
        for arm in EVALUATION_ARMS:
            result = await _execute_scheduled(
                    dependencies,
                    ArmTaskSpec(
                        purpose="evaluation",
                        phase="development",
                        arm=arm,
                        task=task,
                        judge_arm=arm,
                    ),
                )
            _append_unique_attempt(attempts, result)
            removal_group.append(result)
        removal_fairness.append(_validate_removal_group(removal_group))
    await _complete(evaluator, "development")
    pre_final_manifest_sha256 = validate_source_manifest(
        dependencies.manifest,
        expected_hash=dependencies.expected_manifest_sha256,
        repository_root=dependencies.repository_root,
    )
    if pre_final_manifest_sha256 != manifest_sha256:
        raise RunnerInvariantError("pre-final manifest identity changed")
    admit = getattr(evaluator, "admit_final", None)
    if not callable(admit):
        raise TypeError("evaluator lacks admit_final")
    await _maybe_await(admit(derived_admission))
    final = await _release(evaluator, "final")
    if len(final) != 40:
        raise RunnerInvariantError("evaluation final count differs")
    for task in final:
        removal_group = []
        for arm in EVALUATION_ARMS:
            result = await _execute_scheduled(
                    dependencies,
                    ArmTaskSpec(
                        purpose="evaluation",
                        phase="final",
                        arm=arm,
                        task=task,
                        judge_arm=arm,
                    ),
                )
            _append_unique_attempt(attempts, result)
            removal_group.append(result)
        removal_fairness.append(_validate_removal_group(removal_group))
    await _complete(evaluator, "final")
    metrics_method = getattr(evaluator, "final_metrics", None)
    if not callable(metrics_method):
        raise TypeError("evaluator lacks final_metrics")
    metrics = await _maybe_await(metrics_method())
    if hasattr(metrics, "to_canonical"):
        metrics = metrics.to_canonical()
    if type(metrics) is not dict:
        raise RunnerInvariantError("evaluator final metrics are malformed")
    metrics = _validate_final_metrics(metrics)
    if len(attempts) != EVALUATION_COUNTS[0]:
        raise RunnerInvariantError("evaluation arm-task schedule differs")
    _validate_unique_attempts(attempts)
    adaptation_lineage, adaptation_lineage_ref = validate_adaptation_lineages(
        attempts,
        dependencies.expected_learner_genesis_digests,
        purpose="evaluation",
    )
    lineage_evidence, lineage_ref = validate_full_adaptation_lineages(
        attempts,
        dependencies.expected_learner_genesis_digests,
    )
    foundation_after = await _foundation_snapshot(dependencies)
    validate_foundation_hashes(
        foundation_before,
        foundation_after,
        dependencies.expected_foundation_hashes,
    )
    accounting = dependencies.accounting.validate_complete()
    (
        run_integrity,
        run_integrity_ref,
        foundation_integrity,
        foundation_integrity_ref,
    ) = build_run_integrity_evidence(
        purpose="evaluation",
        attempts=attempts,
        removal_fairness=removal_fairness,
        foundation_before=foundation_before,
        foundation_after=foundation_after,
        expected_foundation=dependencies.expected_foundation_hashes,
        expected_foundation_tensor_digest=(
            dependencies.expected_foundation_tensor_digest
        ),
        expected_learner_genesis_digests=(
            dependencies.expected_learner_genesis_digests
        ),
        adaptation_lineage=adaptation_lineage,
        adaptation_lineage_ref=adaptation_lineage_ref,
        resource_accounting=accounting,
        full_adaptation_lineage=lineage_evidence,
        full_adaptation_lineage_ref=lineage_ref,
    )
    classification, classifier, classifier_ref = classify_evaluation_result(
        metrics,
        run_integrity,
    )
    await _close_evaluator(evaluator)
    return {
        "accounting": accounting,
        "adaptation_lineage": adaptation_lineage,
        "adaptation_lineage_ref": adaptation_lineage_ref,
        "admission_claim": admission_claim,
        "admission_claim_ref": admission_claim_ref,
        "arm_tasks": len(attempts),
        "attempt_receipt_refs": [item.attempt_receipt_ref for item in attempts],
        "classification": classification,
        "classifier": classifier,
        "classifier_ref": classifier_ref,
        "evaluation_metrics": metrics,
        "evaluation_metrics_ref": content_ref("final-metrics", metrics),
        "foundation_integrity": foundation_integrity,
        "foundation_integrity_ref": foundation_integrity_ref,
        "full_adaptation_lineage": lineage_evidence,
        "full_adaptation_lineage_ref": lineage_ref,
        "generation_attempt_ceiling": EVALUATION_COUNTS[1],
        "identity": EVALUATION_IDENTITY,
        "manifest_sha256": manifest_sha256,
        "purpose": "evaluation",
        "removal_fairness_refs": run_integrity["removal_fairness_refs"],
        "run_integrity": run_integrity,
        "run_integrity_ref": run_integrity_ref,
        "schema": EVALUATION_RESULT_SCHEMA,
        "scientific_claim": "BOUNDED_SYNTHETIC_EXPERIMENT_ONLY",
    }


async def _invalidate_or_reap_evaluator(evaluator: object) -> None:
    invalidate = getattr(evaluator, "invalidate", None)
    if callable(invalidate):
        await _maybe_await(invalidate())
        return
    close = getattr(evaluator, "close", None)
    if callable(close):
        await _maybe_await(close())
        return
    raise TypeError("evaluator has no cleanup boundary")


async def _raise_after_evaluator_cleanup(
    evaluator: object,
    primary: BaseException,
) -> None:
    try:
        await _invalidate_or_reap_evaluator(evaluator)
    except BaseException:
        cleanup = CleanupFailure(
            "EVALUATOR_CLEANUP_FAILED",
            "EVALUATOR_CLEANUP",
        )
        raise BaseExceptionGroup(
            "evaluation primary and cleanup failure",
            [primary, cleanup],
        ) from None
    raise primary.with_traceback(primary.__traceback__) from None


async def _run_qualification_injected(
    dependencies: OrchestrationDependencies,
) -> dict[str, object]:
    try:
        return await _run_qualification(dependencies)
    except BaseException as primary:
        typed_primary = _typed_live_infrastructure(
            primary,
            code="LIVE_QUALIFICATION_INFRASTRUCTURE_FAILED",
            stage="LIVE_QUALIFICATION",
        )
        await _raise_after_evaluator_cleanup(
            dependencies.evaluator,
            typed_primary,
        )
        raise AssertionError("unreachable")


def _qualification_failure_disposition(
    error: BaseException,
) -> dict[str, object]:
    leaves = _exception_leaves(error)
    cleanup_complete = not any(
        isinstance(item, CleanupFailure) for item in leaves
    )
    _, failure = classify_exception_disposition(
        error,
        admission_validated=True,
        evaluator_cleanup_completed=cleanup_complete,
    )
    if failure is None:
        raise RunnerInvariantError("qualification failure was not classifiable")
    failure = dict(failure)
    failure["admission_validated"] = False
    return validate_failure_disposition(failure)


async def _run_and_publish_qualification_once(
    dependencies_factory: Callable[[Mapping[str, object]], Any],
    *,
    manifest_path: str | Path,
    result_path: str | Path,
    release_claim_path: str | Path,
    repository_root: str | Path,
    injected_paths: bool,
) -> dict[str, object]:
    """Private CPU/fake-capable core; preflight precedes owner construction."""

    if not callable(dependencies_factory):
        raise TypeError("qualification dependencies factory must be callable")
    if type(injected_paths) is not bool:
        raise TypeError("injected_paths must be an exact bool")
    manifest_target = Path(manifest_path)
    result_target = Path(result_path)
    release_target = Path(release_claim_path)
    root = Path(repository_root)
    if any(
        not path.is_absolute()
        for path in (manifest_target, result_target, release_target, root)
    ):
        raise ValueError("qualification coordinator paths must be absolute")
    _require_owned_real_parent(result_target.parent)
    _mkdir_private(release_target.parent)
    temporary = result_target.with_name(result_target.name + ".tmp")
    release_temporary = release_target.with_name(release_target.name + ".tmp")
    for target in (result_target, temporary, release_target, release_temporary):
        _require_identity_absent(target, QualificationIdentityConsumed)
    manifest = load_source_manifest(
        manifest_target,
        repository_root=root,
    )
    manifest_raw = _read_private_artifact(
        manifest_target,
        maximum_bytes=MAXIMUM_IPC_BYTES,
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if not injected_paths and (
        str(manifest_target) != FROZEN_PATHS["manifest"]
        or str(result_target) != FROZEN_PATHS["qualification_result"]
        or str(release_target) != FROZEN_PATHS["qualification_release_claim"]
        or str(root) != FROZEN_PATHS["repository_root"]
        or manifest.get("paths") != FROZEN_PATHS
    ):
        raise RunnerInvariantError("qualification coordinator paths are not frozen")
    release_claim, release_claim_ref = _build_qualification_release_claim(
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_target,
        result_path=result_target,
        claim_path=release_target,
        repository_root=root,
        path_binding_mode=(
            "INJECTED_CPU_TEST" if injected_paths else "FROZEN_LIVE"
        ),
    )
    for target in (result_target, temporary, release_target, release_temporary):
        _require_identity_absent(target, QualificationIdentityConsumed)
    try:
        durable_release, durable_release_ref = (
            _publish_qualification_release_claim(
                release_target,
                release_claim,
            )
        )
    except QualificationReleasePublicationFailure as primary:
        failure_result = build_qualification_failure_result(
            failure=_qualification_failure_disposition(primary),
            manifest_sha256=manifest_sha256,
        )
        try:
            publish_qualification_failure_result(
                result_target,
                failure_result,
            )
        except BaseException:
            publication = CleanupFailure(
                "TERMINAL_RESULT_PUBLICATION_FAILED",
                "RESULT_PUBLICATION",
            )
            raise BaseExceptionGroup(
                "qualification release and terminal publication failure",
                [primary, publication],
            ) from None
        return failure_result
    if (
        durable_release != release_claim
        or durable_release_ref != release_claim_ref
    ):
        raise RunnerInvariantError("qualification release reservation differs")

    released = True
    try:
        owner = await _construct_live_dependencies_owner(
            dependencies_factory,
            {
                "manifest": manifest,
                "manifest_sha256": manifest_sha256,
                "purpose": "qualification",
            },
            purpose="qualification",
        )
        dependencies = owner.dependencies
        if (
            dependencies.expected_manifest_sha256 != manifest_sha256
            or dependencies.manifest != manifest
            or Path(dependencies.repository_root) != root
        ):
            primary = RunnerInvariantError(
                "qualification dependencies differ from preflight"
            )
            await _raise_after_owner_cleanup(owner, primary)
            raise AssertionError("unreachable")
        try:
            result = await _run_qualification(dependencies)
        except BaseException as primary:
            _clear_exception_tracebacks(primary)
            typed_primary = _typed_live_infrastructure(
                primary,
                code="LIVE_QUALIFICATION_INFRASTRUCTURE_FAILED",
                stage="LIVE_QUALIFICATION",
            )
            _clear_exception_tracebacks(typed_primary)
            del primary
            await _raise_after_owner_cleanup(owner, typed_primary)
            raise AssertionError("unreachable")
        await owner.close()
        current_manifest = load_source_manifest(
            manifest_target,
            expected_hash=manifest_sha256,
            repository_root=root,
        )
        if current_manifest != manifest:
            raise RunnerInvariantError(
                "qualification source manifest changed before publication"
            )
        publish_qualification_result(
            result_target,
            result,
            expected_manifest_sha256=manifest_sha256,
        )
        published, _ = load_qualification_result(
            result_target,
            expected_manifest_sha256=manifest_sha256,
        )
        return published
    except BaseException as primary:
        if not released:
            raise
        failure_result = build_qualification_failure_result(
            failure=_qualification_failure_disposition(primary),
            manifest_sha256=manifest_sha256,
        )
        try:
            publish_qualification_failure_result(
                result_target,
                failure_result,
            )
        except BaseException:
            publication = CleanupFailure(
                "TERMINAL_RESULT_PUBLICATION_FAILED",
                "RESULT_PUBLICATION",
            )
            raise BaseExceptionGroup(
                "qualification failure and terminal publication failure",
                [primary, publication],
            ) from None
        return failure_result


async def run_and_publish_qualification_once(
) -> dict[str, object]:
    """Run the sole frozen live qualification release-to-terminal path."""

    return await _run_live_qualification_cli()


async def _run_evaluation_injected(
    dependencies: OrchestrationDependencies,
    *,
    qualification_result_sha256: str,
    evaluator_commitment_digest: str,
    expected_admission_digest: str,
) -> dict[str, object]:
    """CPU/fake construction seam; never the live one-use entrypoint."""

    try:
        return await _run_evaluation(
            dependencies,
            qualification_result_sha256=qualification_result_sha256,
            evaluator_commitment_digest=evaluator_commitment_digest,
            expected_admission_digest=expected_admission_digest,
        )
    except BaseException as primary:
        typed_primary = _typed_live_infrastructure(
            primary,
            code="LIVE_EVALUATION_INFRASTRUCTURE_FAILED",
            stage="LIVE_EVALUATION",
        )
        await _raise_after_evaluator_cleanup(
            dependencies.evaluator,
            typed_primary,
        )
        raise AssertionError("unreachable")


async def _run_admitted_evaluation(
    dependencies_factory: Callable[[Mapping[str, object]], Any],
    *,
    admission_permit: _EvaluationAdmissionPermit,
    allow_injected_paths: bool = False,
) -> dict[str, object]:
    """Run only after reloading a durable claim, then construct live owners."""

    if not callable(dependencies_factory):
        raise TypeError("evaluation dependencies factory must be callable")
    if type(admission_permit) is not _EvaluationAdmissionPermit:
        raise TypeError("live evaluation requires its exact in-process permit")
    if type(allow_injected_paths) is not bool:
        raise TypeError("allow_injected_paths must be an exact bool")
    # O_EXCL has already consumed the experiment identity.  Consume the
    # non-reconstructable release permit before any further fallible read.
    admission_receipt = admission_permit.consume()
    if set(admission_receipt) != {
        "claim",
        "claim_path",
        "claim_ref",
        "manifest_sha256",
        "qualification_result_sha256",
        "result_path",
        "seed_seal_ref",
    }:
        raise RunnerInvariantError("durable evaluation admission receipt differs")
    claim_path = Path(admission_receipt["claim_path"])
    if not claim_path.is_absolute():
        raise RunnerInvariantError("durable evaluation claim path differs")
    claim, claim_ref = load_evaluation_admission_claim(
        claim_path,
        expected_ref=admission_receipt["claim_ref"],  # type: ignore[arg-type]
    )
    if (
        claim != admission_receipt["claim"]
        or claim_ref != admission_receipt["claim_ref"]
        or claim["manifest_sha256"] != admission_receipt["manifest_sha256"]
        or claim["qualification_result_sha256"]
        != admission_receipt["qualification_result_sha256"]
        or claim["seed_seal_ref"] != admission_receipt["seed_seal_ref"]
        or claim["paths"]["claim"] != admission_receipt["claim_path"]
        or claim["paths"]["evaluation_result"]
        != admission_receipt["result_path"]
    ):
        raise RunnerInvariantError("durable evaluation admission receipt was altered")
    expected_mode = "INJECTED_CPU_TEST" if allow_injected_paths else "FROZEN_LIVE"
    if claim["path_binding_mode"] != expected_mode:
        raise RunnerInvariantError("admitted evaluation claim mode differs")
    paths = claim["paths"]
    if not allow_injected_paths and paths != _frozen_live_claim_paths():
        raise RunnerInvariantError("public evaluation claim paths are not frozen")
    manifest = load_source_manifest(
        paths["manifest"],
        expected_hash=claim["manifest_sha256"],
        repository_root=paths["repository_root"],
    )
    qualification, qualification_sha256 = load_qualification_result(
        paths["qualification_result"],
        expected_manifest_sha256=claim["manifest_sha256"],
    )
    seed_seal = load_seed_seal(
        paths["seed"],
        manifest["seed_commitments"],
    )
    if (
        qualification_sha256 != claim["qualification_result_sha256"]
        or seed_seal.seal_ref != claim["seed_seal_ref"]
        or manifest.get("frozen_sources") != claim["source_hashes"]
        or manifest.get("evaluator_commitment_digest")
        != claim["evaluator_commitment_digest"]
        or (not allow_injected_paths and manifest.get("paths") != FROZEN_PATHS)
    ):
        raise RunnerInvariantError("durable evaluation preimage reload differs")
    result_path = Path(admission_receipt["result_path"])
    if not result_path.is_absolute():
        raise RunnerInvariantError("durable evaluation result path differs")
    result_temporary = result_path.with_name(result_path.name + ".tmp")
    for target in (result_path, result_temporary):
        _claim_target_is_absent(target)
    # This is the first permitted construction of model, state, Cognee, or live
    # evaluator owners.  The factory is deliberately not invoked on any prior
    # validation path.
    owner = await _construct_live_dependencies_owner(
        dependencies_factory,
        {
            **admission_receipt,
            "manifest": manifest,
            "qualification_result": qualification,
            "seed_seal": seed_seal.to_canonical(),
        },
        purpose="evaluation",
    )
    dependencies = owner.dependencies
    if (
        dependencies.expected_manifest_sha256 != claim["manifest_sha256"]
        or dependencies.manifest.get("evaluator_commitment_digest")
        != claim["evaluator_commitment_digest"]
        or dependencies.manifest.get("frozen_sources") != claim["source_hashes"]
    ):
        primary = RunnerInvariantError(
            "live dependencies differ from durable admission"
        )
        await _raise_after_owner_cleanup(owner, primary)
        raise AssertionError("unreachable")
    try:
        result = await _run_evaluation(
            dependencies,
            qualification_result_sha256=claim["qualification_result_sha256"],  # type: ignore[arg-type]
            evaluator_commitment_digest=claim["evaluator_commitment_digest"],  # type: ignore[arg-type]
            expected_admission_digest=claim["admission_digest"],  # type: ignore[arg-type]
            durable_admission_claim=claim,
        )
    except BaseException as primary:
        _clear_exception_tracebacks(primary)
        typed_primary = _typed_live_infrastructure(
            primary,
            code="LIVE_EVALUATION_INFRASTRUCTURE_FAILED",
            stage="LIVE_EVALUATION",
        )
        _clear_exception_tracebacks(typed_primary)
        del primary
        await _raise_after_owner_cleanup(owner, typed_primary)
        raise AssertionError("unreachable")
    await owner.close()
    return result


async def _run_and_publish_evaluation_once(
    dependencies_factory: Callable[[Mapping[str, object]], Any],
    *,
    evaluator_handshake: Callable[[], Any],
) -> dict[str, object]:
    """Private claim-to-terminal core used by the zero-injection live CLI."""

    permit: _EvaluationAdmissionPermit | None = None
    claim: dict[str, object] | None = None
    claim_ref: str | None = None
    manifest: dict[str, object] | None = None
    for target in (
        EVALUATION_ADMISSION_CLAIM_PATH,
        EVALUATION_RESULT_PATH,
        EVALUATION_RESULT_PATH.with_name(EVALUATION_RESULT_PATH.name + ".tmp"),
    ):
        _require_identity_absent(target, EvaluationIdentityConsumed)
    try:
        try:
            permit = await _claim_evaluation_admission(
                manifest_path=MANIFEST_PATH,
                seed_path=SEALED_SEED_PATH,
                qualification_result_path=QUALIFICATION_RESULT_PATH,
                evaluation_result_path=EVALUATION_RESULT_PATH,
                claim_path=EVALUATION_ADMISSION_CLAIM_PATH,
                repository_root=REPOSITORY_ROOT,
                evaluator_handshake=evaluator_handshake,
            )
        except AdmissionClaimPublicationFailure:
            raise
        except (OSError, TimeoutError, MemoryError):
            raise InfrastructureFailure(
                "EVALUATION_PREFLIGHT_INFRASTRUCTURE_FAILED",
                "EVALUATION_ADMISSION",
            ) from None
        claim = dict(permit.receipt["claim"])
        claim_ref = permit.receipt["claim_ref"]  # type: ignore[assignment]
        result = await _run_admitted_evaluation(
            dependencies_factory,
            admission_permit=permit,
        )
        manifest = load_source_manifest(
            claim["paths"]["manifest"],  # type: ignore[index]
            expected_hash=claim["manifest_sha256"],  # type: ignore[arg-type]
            repository_root=claim["paths"]["repository_root"],  # type: ignore[index]
        )
        publish_evaluation_result(
            claim["paths"]["evaluation_result"],  # type: ignore[index]
            result,
            manifest=manifest,
            repository_root=claim["paths"]["repository_root"],  # type: ignore[index]
            expected_admission_claim_ref=claim_ref,
            admission_claim_path=claim["paths"]["claim"],  # type: ignore[index]
        )
        published, _ = load_evaluation_result(
            claim["paths"]["evaluation_result"],  # type: ignore[index]
            manifest=manifest,
            repository_root=claim["paths"]["repository_root"],  # type: ignore[index]
            expected_manifest_sha256=claim["manifest_sha256"],  # type: ignore[arg-type]
            expected_admission_claim_ref=claim_ref,
            admission_claim_path=claim["paths"]["claim"],  # type: ignore[index]
        )
        return published
    except BaseException as primary:
        if isinstance(primary, EvaluationIdentityConsumed):
            raise
        leaves = _exception_leaves(primary)
        cleanup_complete = not any(
            isinstance(item, CleanupFailure) for item in leaves
        )
        admission_validated = claim is not None and claim_ref is not None
        classification, failure = classify_exception_disposition(
            primary,
            admission_validated=admission_validated,
            evaluator_cleanup_completed=cleanup_complete,
        )
        if classification is None or failure is None:
            raise
        if classification not in ("INVALID", "INCONCLUSIVE"):
            raise RunnerInvariantError("terminal evaluation classification differs")
        failure_result = build_evaluation_failure_result(
            classification=classification,  # type: ignore[arg-type]
            failure=failure,
            admission_claim=claim,
            admission_claim_ref=claim_ref,
            manifest_sha256=(
                None if claim is None else claim["manifest_sha256"]  # type: ignore[arg-type]
            ),
        )
        try:
            publish_evaluation_failure_result(
                EVALUATION_RESULT_PATH,
                failure_result,
                manifest=None,
                repository_root=(
                    REPOSITORY_ROOT
                    if claim is None
                    else claim["paths"]["repository_root"]  # type: ignore[index]
                ),
                expected_admission_claim_ref=(
                    claim_ref if admission_validated else None
                ),
                admission_claim_path=(
                    EVALUATION_ADMISSION_CLAIM_PATH
                    if claim is None
                    else claim["paths"]["claim"]  # type: ignore[index]
                ),
            )
        except BaseException:
            publication = CleanupFailure(
                "TERMINAL_RESULT_PUBLICATION_FAILED",
                "RESULT_PUBLICATION",
            )
            raise BaseExceptionGroup(
                "evaluation failure and terminal publication failure",
                [primary, publication],
            ) from None
        return failure_result
@dataclass(frozen=True, slots=True)
class EvaluatorWorkerSpec:
    purpose: Purpose
    expected_commitments: tuple[str, ...]
    expected_final_admission: str | None = None
    seed_path: str | None = None
    qualification_seed: int = QUALIFICATION_SEED
    expected_evaluator_commitments_json: str | None = None
    expected_evaluator_commitment_digest: str | None = None
    commitment_only: bool = False

    def __post_init__(self) -> None:
        if self.purpose not in ("qualification", "evaluation"):
            raise ValueError("worker purpose differs")
        expected_count = 1 if self.purpose == "qualification" else 2
        if type(self.expected_commitments) is not tuple or len(self.expected_commitments) != expected_count:
            raise ValueError("worker commitment count differs")
        for value in self.expected_commitments:
            _digest(value, "expected replicate commitment")
        if len(set(self.expected_commitments)) != expected_count:
            raise ValueError("worker commitments must be distinct")
        if self.qualification_seed != QUALIFICATION_SEED:
            raise ValueError("qualification seed differs")
        if type(self.commitment_only) is not bool:
            raise TypeError("commitment_only must be an exact bool")
        if self.purpose == "qualification":
            if (
                self.commitment_only
                or self.expected_final_admission is not None
                or self.seed_path is not None
                or self.expected_evaluator_commitments_json is not None
                or self.expected_evaluator_commitment_digest is not None
            ):
                raise ValueError("qualification cannot receive final seed/admission inputs")
        else:
            _digest(self.expected_final_admission, "expected_final_admission")
            if self.seed_path != str(SEALED_SEED_PATH):
                raise ValueError("evaluation worker needs the canonical absolute seed path")
            if self.commitment_only:
                if (
                    self.expected_final_admission
                    != COMMITMENT_ONLY_ADMISSION_DIGEST
                    or self.expected_evaluator_commitments_json is not None
                    or self.expected_evaluator_commitment_digest is not None
                ):
                    raise ValueError(
                        "commitment-only worker inputs differ from its placeholder"
                    )
                return
            if type(self.expected_evaluator_commitments_json) is not str:
                raise ValueError("evaluation worker needs canonical evaluator commitments")
            payload = decode_canonical_json(self.expected_evaluator_commitments_json)
            _validate_manifest_commitments(payload, list(self.expected_commitments))
            expected_digest = evaluator_record_digest(
                "evaluator-commitments",
                payload,  # type: ignore[arg-type]
            )
            if self.expected_evaluator_commitment_digest != expected_digest:
                raise ValueError("evaluation worker commitment digest differs")

    @property
    def expected_evaluator_commitments(self) -> dict[str, object] | None:
        if self.expected_evaluator_commitments_json is None:
            return None
        value = decode_canonical_json(self.expected_evaluator_commitments_json)
        if type(value) is not dict:  # pragma: no cover - constructor validated this
            raise RunnerInvariantError("worker commitments are not an object")
        return value

    def to_canonical(self) -> dict[str, object]:
        return {
            "commitment_only": self.commitment_only,
            "expected_commitments": list(self.expected_commitments),
            "expected_evaluator_commitment_digest": (
                self.expected_evaluator_commitment_digest
            ),
            "expected_evaluator_commitments": self.expected_evaluator_commitments,
            "expected_final_admission": self.expected_final_admission,
            "purpose": self.purpose,
            "qualification_seed": self.qualification_seed,
            "seed_path": self.seed_path,
        }

    @classmethod
    def from_canonical(cls, value: Mapping[str, object]) -> "EvaluatorWorkerSpec":
        if type(value) is not dict or set(value) != {
            "commitment_only",
            "expected_commitments",
            "expected_evaluator_commitment_digest",
            "expected_evaluator_commitments",
            "expected_final_admission",
            "purpose",
            "qualification_seed",
            "seed_path",
        }:
            raise EvaluatorProtocolError("worker specification fields differ")
        commitments = value["expected_commitments"]
        if type(commitments) is not list:
            raise EvaluatorProtocolError("worker commitments are not a list")
        evaluator_commitments = value["expected_evaluator_commitments"]
        if evaluator_commitments is not None and type(evaluator_commitments) is not dict:
            raise EvaluatorProtocolError("expected evaluator commitments are malformed")
        return cls(
            purpose=value["purpose"],  # type: ignore[arg-type]
            expected_commitments=tuple(commitments),  # type: ignore[arg-type]
            expected_final_admission=value["expected_final_admission"],  # type: ignore[arg-type]
            seed_path=value["seed_path"],  # type: ignore[arg-type]
            qualification_seed=value["qualification_seed"],  # type: ignore[arg-type]
            expected_evaluator_commitments_json=(
                None
                if evaluator_commitments is None
                else canonical_json_bytes(evaluator_commitments).decode("utf-8")
            ),
            expected_evaluator_commitment_digest=value[
                "expected_evaluator_commitment_digest"
            ],  # type: ignore[arg-type]
            commitment_only=value["commitment_only"],  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class EvaluatorProtocolState:
    initialized: bool = False
    next_request_id: int = 1
    spec: EvaluatorWorkerSpec | None = None
    evaluator: object | None = None
    _closed: bool = False
    _valid: bool = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def valid(self) -> bool:
        return self._valid

    def invalidate(self) -> None:
        self._valid = False
        self._closed = True


EvaluatorDispatchState = EvaluatorProtocolState


_EVALUATOR_OPS = frozenset(
    {
        "commitments",
        "release_phase",
        "complete_phase",
        "admit_final",
        "judge_response",
        "random_feedback",
        "final_metrics",
        "close",
    }
)


def _request(request_id: int, op: str, payload: Mapping[str, object]) -> dict[str, object]:
    if type(request_id) is not int or request_id < 0:
        raise ValueError("request ID must be a non-negative integer")
    if op not in _EVALUATOR_OPS:
        raise ValueError("evaluator operation is not declared")
    return {"id": request_id, "op": op, "payload": dict(payload), "v": PROTOCOL_VERSION}


def _validate_request_envelope(value: object, state: EvaluatorProtocolState) -> tuple[int, str, dict[str, object]]:
    if not state.valid:
        raise EvaluatorProtocolError("evaluator protocol state is invalid")
    if type(value) is not dict or set(value) != {"id", "op", "payload", "v"}:
        raise EvaluatorProtocolError("request envelope fields differ")
    if value["v"] != PROTOCOL_VERSION:
        raise EvaluatorProtocolError("request protocol version differs")
    request_id = value["id"]
    if type(request_id) is not int or request_id != state.next_request_id:
        raise EvaluatorProtocolError("request ID is not exact and monotonic")
    op = value["op"]
    if op not in _EVALUATOR_OPS:
        raise EvaluatorProtocolError("request operation is not declared")
    payload = value["payload"]
    if type(payload) is not dict:
        raise EvaluatorProtocolError("request payload must be an exact object")
    return request_id, op, payload


def _success(request_id: int, result: object) -> dict[str, object]:
    envelope = {"id": request_id, "ok": True, "result": result, "v": PROTOCOL_VERSION}
    scan_for_hidden_material(envelope)
    return envelope


def _generic_error(request_id: int, code: str = "OPERATION_FAILED") -> dict[str, object]:
    if code not in {"MALFORMED_REQUEST", "OPERATION_FAILED", "INTERNAL_FAILURE"}:
        code = "INTERNAL_FAILURE"
    return {
        "error": {"code": code, "message": "EVALUATOR_OPERATION_FAILED"},
        "id": request_id,
        "ok": False,
        "v": PROTOCOL_VERSION,
    }


def _construct_evaluator(spec: EvaluatorWorkerSpec) -> object:
    from experiments.evaluators.high_level_multidomain_v1 import (
        make_evaluation_evaluator,
        make_qualification_evaluator,
    )

    if spec.purpose == "qualification":
        evaluator = make_qualification_evaluator()
    else:
        seeds = _load_raw_seeds_for_worker(
            Path(spec.seed_path), spec.expected_commitments  # type: ignore[arg-type]
        )
        evaluator = make_evaluation_evaluator(
            seeds,
            spec.expected_commitments,
            spec.expected_final_admission,  # type: ignore[arg-type]
        )
    observed = tuple(evaluator.commitments.replicate_commitments)
    if observed != spec.expected_commitments:
        raise RunnerInvariantError("constructed evaluator commitments differ")
    if spec.purpose == "evaluation" and not spec.commitment_only and (
        evaluator.commitments.to_canonical()
        != spec.expected_evaluator_commitments
        or evaluator.commitments.digest
        != spec.expected_evaluator_commitment_digest
    ):
        raise RunnerInvariantError("constructed evaluator manifest binding differs")
    return evaluator


def _dispatch_evaluator_request(
    evaluator: object | None,
    request: Mapping[str, object],
    state: EvaluatorProtocolState,
) -> dict[str, object]:
    """Dispatch one exact request; suitable for in-memory fake protocol tests."""

    request_id, op, payload = _validate_request_envelope(dict(request), state)
    if state.closed:
        raise EvaluatorProtocolError("post-close request is forbidden")
    if not state.initialized:
        if op != "commitments" or set(payload) != {"spec"}:
            raise EvaluatorProtocolError("first operation must construct commitments")
        spec_value = payload["spec"]
        if type(spec_value) is not dict:
            raise EvaluatorProtocolError("worker specification is not an object")
        spec = EvaluatorWorkerSpec.from_canonical(spec_value)
        actual = evaluator if evaluator is not None else _construct_evaluator(spec)
        state.spec = spec
        state.evaluator = actual
        state.initialized = True
        result = {
            "commitments": actual.commitments.to_canonical(),
            "digest": actual.commitments.digest,
        }
        return _success(request_id, result)
    if evaluator is not None and evaluator is not state.evaluator:
        raise EvaluatorProtocolError("worker evaluator instance changed")
    active = state.evaluator
    if active is None:
        raise EvaluatorProtocolError("worker evaluator is absent")
    if state.spec is not None and state.spec.commitment_only and op != "close":
        raise EvaluatorProtocolError(
            "commitment-only worker permits only commitments then close"
        )
    if op == "commitments":
        if payload:
            raise EvaluatorProtocolError("repeated commitments payload must be empty")
        result = {
            "commitments": active.commitments.to_canonical(),
            "digest": active.commitments.digest,
        }
    elif op == "release_phase":
        if set(payload) != {"phase"}:
            raise EvaluatorProtocolError("release_phase fields differ")
        tasks = active.release_phase(payload["phase"])
        result = {"tasks": [task.to_canonical() for task in tasks]}
    elif op == "complete_phase":
        if set(payload) != {"phase"}:
            raise EvaluatorProtocolError("complete_phase fields differ")
        active.complete_phase(payload["phase"])
        result = {}
    elif op == "admit_final":
        if set(payload) != {"admission_digest"}:
            raise EvaluatorProtocolError("admit_final fields differ")
        active.admit_final(payload["admission_digest"])
        result = {}
    elif op == "judge_response":
        if set(payload) != {"arm", "attempt_receipt_ref", "raw_response", "task_id"}:
            raise EvaluatorProtocolError("judge_response fields differ")
        judgment = active.judge_response(
            payload["task_id"],
            payload["arm"],
            payload["attempt_receipt_ref"],
            payload["raw_response"],
        )
        result = judgment.to_canonical()
    elif op == "random_feedback":
        if set(payload) != {"replicate_commitment"}:
            raise EvaluatorProtocolError("random_feedback fields differ")
        schedule = active.random_feedback_schedule(payload["replicate_commitment"])
        result = {
            "assignments": [
                {"task_id": task_id, "value": value}
                for task_id, value in schedule
            ],
            "replicate_commitment": payload["replicate_commitment"],
        }
    elif op == "final_metrics":
        if payload:
            raise EvaluatorProtocolError("final_metrics payload must be empty")
        metrics = active.final_metrics()
        result = metrics.to_canonical()
    elif op == "close":
        if payload:
            raise EvaluatorProtocolError("close payload must be empty")
        completed = tuple(active.completed_phases)
        released = tuple(active.released_phases)
        if state.spec is None:
            raise EvaluatorProtocolError("close has no worker specification")
        if state.spec.commitment_only:
            if completed or released:
                raise EvaluatorProtocolError(
                    "commitment-only worker acquired phase state"
                )
        elif state.spec.purpose == "evaluation":
            if completed != PHASES or released != PHASES:
                raise EvaluatorProtocolError(
                    "evaluation close requires every phase complete"
                )
        elif completed != ("adaptation",) or released != (
            "adaptation",
            "development",
        ):
            raise EvaluatorProtocolError(
                "qualification close requires the frozen partial schedule"
            )
        state._closed = True
        result = {}
    else:  # pragma: no cover - operation allowlist is exhaustive
        raise EvaluatorProtocolError("unsupported evaluator operation")
    return _success(request_id, result)


def dispatch_evaluator_request(
    evaluator: object | None,
    request: Mapping[str, object],
    state: EvaluatorProtocolState,
) -> dict[str, object]:
    """Dispatch once and permanently invalidate state on any failure."""

    if type(state) is not EvaluatorProtocolState:
        raise TypeError("state must be an exact EvaluatorProtocolState")
    try:
        response = _dispatch_evaluator_request(evaluator, request, state)
    except BaseException:
        state.invalidate()
        raise
    state.next_request_id += 1
    return response


class _ProcessLike(Protocol):
    stdin: Any
    stdout: Any
    stderr: Any

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory: TypeAlias = Callable[[EvaluatorWorkerSpec], _ProcessLike]


def _validate_evaluator_handshake_identity(
    commitments: Mapping[str, object],
    spec: EvaluatorWorkerSpec,
) -> None:
    expected_identity = (
        EVALUATOR_QUALIFICATION_IDENTITY
        if spec.purpose == "qualification"
        else EVALUATOR_EVALUATION_IDENTITY
    )
    expected_qualification_seed = (
        QUALIFICATION_SEED if spec.purpose == "qualification" else None
    )
    if (
        type(commitments) is not dict
        or set(commitments)
        != {
            "identity",
            "purpose",
            "qualification_seed",
            "random_feedback",
            "replicate_commitments",
            "schema",
            "tasks",
        }
        or commitments["identity"] != expected_identity
        or commitments["purpose"] != spec.purpose
        or commitments["schema"] != SUITE_SCHEMA
        or commitments["qualification_seed"] != expected_qualification_seed
        or commitments["replicate_commitments"]
        != list(spec.expected_commitments)
    ):
        raise EvaluatorProtocolError("worker commitment identity differs")


def _validated_evaluator_timeout(value: object) -> float:
    if type(value) not in (int, float) or not 0.1 <= float(value) <= 300.0:
        raise ValueError("evaluator timeout must be from 0.1 through 300 seconds")
    return float(value)


def _close_process_pipes(process: _ProcessLike) -> None:
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                pass


def _dispose_evaluator_process(
    process: _ProcessLike,
    *,
    timeout_seconds: float,
    terminate_first: bool,
) -> int | None:
    return_code: int | None = None
    try:
        if terminate_first and process.poll() is None:
            process.terminate()
        try:
            return_code = process.wait(timeout=min(timeout_seconds, 5.0))
        except BaseException:
            kill = getattr(process, "kill", None)
            if not callable(kill):
                raise
            try:
                kill()
            except BaseException:
                pass
            return_code = process.wait(timeout=min(timeout_seconds, 5.0))
    except BaseException:
        return_code = None
    finally:
        _close_process_pipes(process)
    return return_code


def _raise_after_client_invalidation(
    client: "EvaluatorClient",
    primary: BaseException,
) -> None:
    """Preserve protocol/transport primary when process reaping also fails."""

    try:
        client.invalidate()
    except BaseException:
        cleanup = CleanupFailure(
            "EVALUATOR_PROCESS_REAP_FAILED",
            "EVALUATOR_CLEANUP",
        )
        raise BaseExceptionGroup(
            "evaluator client primary and cleanup failure",
            [primary, cleanup],
        ) from None
    raise primary.with_traceback(primary.__traceback__) from None


class EvaluatorClient:
    """Fail-closed synchronous JSONL client around one evaluator subprocess."""

    def __init__(
        self,
        process: _ProcessLike,
        spec: EvaluatorWorkerSpec,
        *,
        timeout_seconds: float,
    ) -> None:
        validated_timeout = _validated_evaluator_timeout(timeout_seconds)
        self._process = process
        self.spec = spec
        self.timeout_seconds = validated_timeout
        self._next_id = 1
        self._usable = True
        self._closed = False
        self._reaped = False
        self._commitment_snapshot: dict[str, object] | None = None

    @classmethod
    def start(
        cls,
        spec: EvaluatorWorkerSpec,
        *,
        process_factory: ProcessFactory | None = None,
        timeout_seconds: float = 30.0,
    ) -> "EvaluatorClient":
        if type(spec) is not EvaluatorWorkerSpec:
            raise TypeError("spec must be an exact EvaluatorWorkerSpec")
        validated_timeout = _validated_evaluator_timeout(timeout_seconds)
        factory = process_factory or _default_evaluator_process
        process = factory(spec)
        client = cls(process, spec, timeout_seconds=validated_timeout)
        try:
            result = client._exchange("commitments", {"spec": spec.to_canonical()})
            if type(result) is not dict or set(result) != {"commitments", "digest"}:
                raise EvaluatorProtocolError("commitments response is not an object")
            _digest(result["digest"], "evaluator commitment digest")
            commitments = result.get("commitments")
            if type(commitments) is not dict:
                raise EvaluatorProtocolError("worker commitment payload is malformed")
            _validate_evaluator_handshake_identity(commitments, spec)
            observed = commitments.get("replicate_commitments")
            if (
                observed != list(spec.expected_commitments)
                or result["digest"]
                != evaluator_record_digest("evaluator-commitments", commitments)
            ):
                raise EvaluatorProtocolError("worker commitment handshake differs")
            if spec.purpose == "evaluation" and not spec.commitment_only and (
                commitments != spec.expected_evaluator_commitments
                or result["digest"]
                != spec.expected_evaluator_commitment_digest
            ):
                raise EvaluatorProtocolError(
                    "worker commitment handshake differs from manifest"
                )
            snapshot = decode_canonical_json(canonical_json_bytes(result))
            if type(snapshot) is not dict:
                raise EvaluatorProtocolError("commitment snapshot canonical copy differs")
            client._commitment_snapshot = snapshot
            return client
        except BaseException as primary:
            _raise_after_client_invalidation(client, primary)
            raise AssertionError("unreachable")

    @property
    def usable(self) -> bool:
        return self._usable and not self._closed

    @property
    def commitment_snapshot(self) -> dict[str, object]:
        if self._commitment_snapshot is None:
            raise RunnerInvariantError("evaluator commitment snapshot is unavailable")
        value = decode_canonical_json(canonical_json_bytes(self._commitment_snapshot))
        if type(value) is not dict:  # pragma: no cover - exact cache invariant
            raise RunnerInvariantError("evaluator commitment snapshot differs")
        return value

    def _protocol_failure(self, message: str) -> None:
        primary = EvaluatorProtocolError(message)
        _raise_after_client_invalidation(self, primary)
        raise AssertionError("unreachable")

    def invalidate(self) -> None:
        if self._reaped:
            return
        self._usable = False
        return_code = _dispose_evaluator_process(
            self._process,
            timeout_seconds=self.timeout_seconds,
            terminate_first=True,
        )
        if return_code is None:
            raise EvaluatorTransportError("evaluator worker could not be reaped")
        self._reaped = True

    def _exchange(self, op: str, payload: Mapping[str, object]) -> object:
        if not self.usable:
            raise EvaluatorTransportError("evaluator client is unusable")
        request_id = self._next_id
        request = _request(request_id, op, payload)
        encoded = canonical_json_bytes(request)
        if len(encoded) + 1 > MAXIMUM_IPC_BYTES:
            self._protocol_failure("evaluator request exceeds IPC ceiling")
        try:
            self._process.stdin.write(encoded + b"\n")
            self._process.stdin.flush()
            raw = _readline_with_timeout(
                self._process.stdout,
                MAXIMUM_IPC_BYTES + 1,
                self.timeout_seconds,
            )
        except BaseException:
            primary = EvaluatorTransportError("evaluator exchange failed")
            _raise_after_client_invalidation(self, primary)
            raise AssertionError("unreachable")
        if not raw or len(raw) > MAXIMUM_IPC_BYTES or not raw.endswith(b"\n"):
            primary = EvaluatorTransportError(
                "evaluator EOF or oversized response"
            )
            _raise_after_client_invalidation(self, primary)
            raise AssertionError("unreachable")
        try:
            response = decode_canonical_json(raw[:-1])
            if type(response) is not dict:
                raise EvaluatorProtocolError("evaluator response is not an object")
            if response.get("ok") is True:
                if set(response) != {"id", "ok", "result", "v"}:
                    raise EvaluatorProtocolError("success response fields differ")
            elif response.get("ok") is False:
                if set(response) != {"error", "id", "ok", "v"}:
                    raise EvaluatorProtocolError("error response fields differ")
            else:
                raise EvaluatorProtocolError("response disposition is malformed")
            if response["v"] != PROTOCOL_VERSION or response["id"] != request_id:
                raise EvaluatorProtocolError("response identity differs")
            self._next_id += 1
            if response["ok"] is False:
                error_value = response["error"]
                if type(error_value) is not dict or set(error_value) != {"code", "message"}:
                    raise EvaluatorProtocolError("generic error fields differ")
                if error_value["message"] != "EVALUATOR_OPERATION_FAILED":
                    raise EvaluatorProtocolError("generic error message differs")
                raise RunnerInvariantError("evaluator operation failed closed")
            return response["result"]
        except BaseException as primary:
            _raise_after_client_invalidation(self, primary)
            raise AssertionError("unreachable")

    def commitments(self) -> dict[str, object]:
        result = self._exchange("commitments", {})
        if type(result) is not dict or set(result) != {"commitments", "digest"}:
            self._protocol_failure("commitments result is malformed")
        _digest(result["digest"], "evaluator commitment digest")
        if type(result["commitments"]) is not dict:
            self._protocol_failure("commitments payload is malformed")
        try:
            _validate_evaluator_handshake_identity(
                result["commitments"],
                self.spec,
            )
        except BaseException as primary:
            _raise_after_client_invalidation(self, primary)
            raise AssertionError("unreachable")
        if result["digest"] != evaluator_record_digest(
            "evaluator-commitments", result["commitments"]
        ):
            self._protocol_failure("evaluator commitment digest differs")
        if self.spec.purpose == "evaluation" and not self.spec.commitment_only and (
            result["commitments"] != self.spec.expected_evaluator_commitments
            or result["digest"]
            != self.spec.expected_evaluator_commitment_digest
        ):
            self._protocol_failure("evaluator commitments differ from manifest")
        return result

    def release_phase(self, phase: Phase) -> tuple[dict[str, object], ...]:
        result = self._exchange("release_phase", {"phase": phase})
        if type(result) is not dict or set(result) != {"tasks"} or type(result["tasks"]) is not list:
            self._protocol_failure("release_phase result is malformed")
        tasks = tuple(result["tasks"])
        if any(type(task) is not dict for task in tasks):
            self._protocol_failure("released task is not an object")
        for task in tasks:
            scan_for_hidden_material(task)
        return tasks  # type: ignore[return-value]

    def complete_phase(self, phase: Phase) -> None:
        result = self._exchange("complete_phase", {"phase": phase})
        if result != {}:
            self._protocol_failure("complete_phase acknowledgment differs")

    def admit_final(self, digest: str) -> None:
        _digest(digest, "admission digest")
        result = self._exchange("admit_final", {"admission_digest": digest})
        if result != {}:
            self._protocol_failure("admit_final acknowledgment differs")

    def judge_response(
        self,
        task_id: str,
        arm: Arm,
        attempt_receipt_ref: str,
        raw_response: str,
    ) -> dict[str, object]:
        result = self._exchange(
            "judge_response",
            {
                "arm": arm,
                "attempt_receipt_ref": attempt_receipt_ref,
                "raw_response": raw_response,
                "task_id": task_id,
            },
        )
        if type(result) is not dict or set(result) != {
            "arm",
            "attempt_receipt_ref",
            "disposition",
            "raw_response",
            "response_commitment",
            "score",
            "task_id",
        }:
            self._protocol_failure("judgment fields differ")
        if (
            result["task_id"],
            result["arm"],
            result["attempt_receipt_ref"],
            result["raw_response"],
        ) != (task_id, arm, attempt_receipt_ref, raw_response):
            self._protocol_failure("judgment does not bind the exact attempt")
        try:
            return validate_objective_judgment(
                result,
                task_id=task_id,
                arm=arm,
                attempt_receipt_ref=attempt_receipt_ref,
                raw_response=raw_response,
            )
        except BaseException as primary:
            _raise_after_client_invalidation(self, primary)
            raise AssertionError("unreachable")

    def random_feedback(self, replicate_commitment: str) -> tuple[tuple[str, float], ...]:
        result = self._exchange(
            "random_feedback", {"replicate_commitment": replicate_commitment}
        )
        if type(result) is not dict or set(result) != {"assignments", "replicate_commitment"}:
            self._protocol_failure("random-feedback result fields differ")
        assignments = result["assignments"]
        if result["replicate_commitment"] != replicate_commitment or type(assignments) is not list:
            self._protocol_failure("random-feedback result identity differs")
        if any(
            type(row) is not dict or set(row) != {"task_id", "value"}
            for row in assignments
        ):
            self._protocol_failure("random-feedback assignment fields differ")
        schedule = tuple((row["task_id"], row["value"]) for row in assignments)
        for task_id, _ in schedule:
            random_feedback_scalar(schedule, task_id)
        return schedule  # type: ignore[return-value]

    def final_metrics(self) -> dict[str, object]:
        result = self._exchange("final_metrics", {})
        if type(result) is not dict:
            self._protocol_failure("final metrics are malformed")
        try:
            return _validate_final_metrics(result)
        except BaseException as primary:
            _raise_after_client_invalidation(self, primary)
            raise AssertionError("unreachable")

    def close(self) -> None:
        if self._closed:
            return
        try:
            result = self._exchange("close", {})
            if result != {}:
                raise EvaluatorProtocolError("close acknowledgment differs")
            self._closed = True
            self._usable = False
            if self._process.stdin is not None:
                self._process.stdin.close()
            return_code = _dispose_evaluator_process(
                self._process,
                timeout_seconds=self.timeout_seconds,
                terminate_first=False,
            )
            if return_code is None:
                raise EvaluatorTransportError("evaluator worker could not be reaped")
            self._reaped = True
            if return_code != 0:
                raise EvaluatorTransportError("evaluator worker exit differs")
        except BaseException as primary:
            _raise_after_client_invalidation(self, primary)
            raise AssertionError("unreachable")

    def __enter__(self) -> "EvaluatorClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc is None:
            self.close()
        elif isinstance(exc, BaseException):
            _raise_after_client_invalidation(self, exc)
        else:
            self.invalidate()


def _readline_with_timeout(stream: Any, limit: int, timeout_seconds: float) -> bytes:
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, ValueError):
        # In-memory test streams are nonblocking by construction.
        return stream.readline(limit)
    deadline = time.monotonic() + timeout_seconds
    buffered = bytearray()
    selector = selectors.DefaultSelector()
    try:
        selector.register(descriptor, selectors.EVENT_READ)
        while len(buffered) < limit:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not selector.select(remaining):
                raise TimeoutError("evaluator response timeout")
            chunk = os.read(descriptor, min(65_536, limit - len(buffered)))
            if not chunk:
                break
            newline = chunk.find(b"\n")
            if newline >= 0:
                buffered.extend(chunk[: newline + 1])
                if newline + 1 != len(chunk):
                    raise EvaluatorProtocolError(
                        "evaluator sent bytes after the response frame"
                    )
                break
            buffered.extend(chunk)
    finally:
        selector.close()
    return bytes(buffered)


@dataclass(slots=True)
class _ManagedEvaluatorProcess:
    process: Any
    worker_cwd: Path
    worker_tmpdir: Path

    @property
    def stdin(self) -> Any:
        return self.process.stdin

    @property
    def stdout(self) -> Any:
        return self.process.stdout

    @property
    def stderr(self) -> Any:
        return self.process.stderr

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    def wait(self, timeout: float | None = None) -> int:
        result = self.process.wait(timeout=timeout)
        for path in (self.worker_tmpdir, self.worker_cwd):
            try:
                path.rmdir()
            except FileNotFoundError:
                pass
        # Separate pre-admission handshakes may briefly share these two parent
        # directories.  Each worker must remove its own private children; a
        # nonempty shared parent is retained for the other bounded worker.
        for path in (self.worker_tmpdir.parent, self.worker_cwd.parent):
            try:
                path.rmdir()
            except FileNotFoundError:
                pass
            except OSError as error:
                if error.errno != errno.ENOTEMPTY:
                    raise
        return result


def _default_evaluator_process(spec: EvaluatorWorkerSpec) -> _ProcessLike:
    del spec  # The first canonical request carries the complete public spec.
    _mkdir_private(SCRATCH_ROOT)
    cwd_parent = SCRATCH_ROOT / "evaluator-cwd"
    tmp_parent = SCRATCH_ROOT / "evaluator-tmp"
    _mkdir_private(cwd_parent)
    _mkdir_private(tmp_parent)
    worker_cwd = Path(tempfile.mkdtemp(prefix="worker-", dir=cwd_parent))
    worker_tmpdir = Path(tempfile.mkdtemp(prefix="worker-", dir=tmp_parent))
    environment = {
        **FROZEN_EVALUATOR_WORKER_ENVIRONMENT,
        "ANGLER_EVALUATOR_CWD": str(worker_cwd),
        "TMPDIR": str(worker_tmpdir),
    }
    try:
        process = subprocess.Popen(
            (str(ANGLER_PYTHON), str(RUNNER_PATH), "--evaluator-worker"),
            cwd=worker_cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
    except BaseException:
        worker_tmpdir.rmdir()
        worker_cwd.rmdir()
        tmp_parent.rmdir()
        cwd_parent.rmdir()
        raise
    return _ManagedEvaluatorProcess(process, worker_cwd, worker_tmpdir)


def _validate_evaluator_worker_environment() -> None:
    expected_keys = {
        "ANGLER_EVALUATOR_CWD",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        "PYTHONUTF8",
        "TELEMETRY_DISABLED",
        "TMPDIR",
    }
    observed = dict(os.environ)
    if set(observed) != expected_keys:
        raise RunnerInvariantError("evaluator worker environment fields differ")
    fixed = FROZEN_EVALUATOR_WORKER_ENVIRONMENT
    if any(observed.get(name) != value for name, value in fixed.items()):
        raise RunnerInvariantError("evaluator worker fixed environment differs")
    cwd = Path(observed["ANGLER_EVALUATOR_CWD"])
    temporary = Path(observed["TMPDIR"])
    if Path.cwd() != cwd or cwd == temporary:
        raise RunnerInvariantError("evaluator worker cwd/TMPDIR binding differs")
    for path in (cwd, temporary):
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise RunnerInvariantError("evaluator worker private directory is not empty")
        metadata = path.stat()
        if (
            metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RunnerInvariantError("evaluator worker directory metadata differs")


async def _ensure_live_source_manifest(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    seed_path: str | Path = SEALED_SEED_PATH,
    qualification_result_path: str | Path = QUALIFICATION_RESULT_PATH,
    qualification_release_path: str | Path = QUALIFICATION_RELEASE_CLAIM_PATH,
    repository_root: str | Path = REPOSITORY_ROOT,
    evaluator_client_factory: Callable[[EvaluatorWorkerSpec], Any] | None = None,
    injected_paths: bool = False,
) -> tuple[dict[str, object], str]:
    """Create or rejoin the pre-result manifest without materializing a seed."""

    if type(injected_paths) is not bool:
        raise TypeError("manifest path injection flag must be exact")
    manifest_target = Path(manifest_path)
    seed_target = Path(seed_path)
    result_target = Path(qualification_result_path)
    release_target = Path(qualification_release_path)
    root = Path(repository_root)
    paths = (manifest_target, seed_target, result_target, release_target, root)
    if any(not path.is_absolute() for path in paths):
        raise ValueError("live manifest inputs must be absolute")
    if not injected_paths and (
        manifest_target != MANIFEST_PATH
        or seed_target != SEALED_SEED_PATH
        or result_target != QUALIFICATION_RESULT_PATH
        or release_target != QUALIFICATION_RELEASE_CLAIM_PATH
        or root != REPOSITORY_ROOT
    ):
        raise RunnerInvariantError("live manifest inputs differ from frozen paths")
    for target in (
        result_target,
        result_target.with_name(result_target.name + ".tmp"),
        release_target,
        release_target.with_name(release_target.name + ".tmp"),
    ):
        _require_identity_absent(target, QualificationIdentityConsumed)
    seal = load_seed_seal(seed_target)
    if manifest_target.exists() or manifest_target.is_symlink():
        manifest = load_source_manifest(manifest_target, repository_root=root)
        if manifest.get("seed_commitments") != list(seal.seed_commitments):
            raise RunnerInvariantError("existing manifest differs from the seed seal")
        raw = _read_private_artifact(
            manifest_target,
            maximum_bytes=MAXIMUM_IPC_BYTES,
        )
        return manifest, hashlib.sha256(raw).hexdigest()

    # Freeze and validate every repository-owned construction input before the
    # commitment worker is allowed to start.  The same source map is rejoined
    # again by publish_source_manifest/load_source_manifest below.
    completed_sources = completed_source_manifest(root)
    accepted_leaf_sha256 = sha256_file(
        root / ACTIVE_LEAF_PATH.relative_to(REPOSITORY_ROOT)
    )[0]

    spec = EvaluatorWorkerSpec(
        purpose="evaluation",
        expected_commitments=seal.seed_commitments,
        expected_final_admission=COMMITMENT_ONLY_ADMISSION_DIGEST,
        seed_path=str(seed_target),
        commitment_only=True,
    )
    factory = evaluator_client_factory or EvaluatorClient.start
    client = await _maybe_await(factory(spec))
    if type(client) is not EvaluatorClient:
        raise TypeError("commitment-only factory returned another client type")
    snapshot: dict[str, object] | None = None
    primary: BaseException | None = None
    try:
        snapshot = client.commitment_snapshot
    except BaseException as error:
        primary = error
    cleanup: BaseException | None = None
    try:
        client.close()
    except BaseException as error:
        cleanup = error
    if primary is not None and cleanup is not None:
        raise BaseExceptionGroup(
            "commitment-only snapshot and cleanup failed",
            [primary, cleanup],
        ) from None
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    if cleanup is not None:
        raise cleanup.with_traceback(cleanup.__traceback__)
    if snapshot is None or set(snapshot) != {"commitments", "digest"}:
        raise RunnerInvariantError("commitment-only snapshot differs")
    commitments = snapshot["commitments"]
    if type(commitments) is not dict or snapshot["digest"] != evaluator_record_digest(
        "evaluator-commitments",
        commitments,
    ):
        raise RunnerInvariantError("commitment-only digest differs")
    manifest = build_source_manifest(
        evaluator_commitments=commitments,
        seed_commitments=seal.seed_commitments,
        accepted_leaf_sha256=accepted_leaf_sha256,
        completed_source_hashes=completed_sources,
    )
    for target in (
        result_target,
        result_target.with_name(result_target.name + ".tmp"),
        release_target,
        release_target.with_name(release_target.name + ".tmp"),
    ):
        _require_identity_absent(target, QualificationIdentityConsumed)
    publish_source_manifest(
        manifest_target,
        manifest,
        repository_root=root,
    )
    loaded = load_source_manifest(manifest_target, repository_root=root)
    if loaded != manifest or loaded.get("seed_commitments") != list(
        seal.seed_commitments
    ):
        raise RunnerInvariantError("published live manifest differs")
    raw = _read_private_artifact(
        manifest_target,
        maximum_bytes=MAXIMUM_IPC_BYTES,
    )
    return loaded, hashlib.sha256(raw).hexdigest()


async def _run_live_evaluation_commitment_handshake(
    manifest: Mapping[str, object],
    seed_seal: SeedSeal,
    *,
    evaluator_client_factory: Callable[[EvaluatorWorkerSpec], Any] | None = None,
    injected_boundaries: bool = False,
) -> dict[str, object]:
    """Start, snapshot, and reap one commitment-only evaluation worker."""

    if type(manifest) is not dict or type(seed_seal) is not SeedSeal:
        raise TypeError("live evaluation commitment inputs differ")
    if type(injected_boundaries) is not bool:
        raise TypeError("live evaluation commitment injection flag differs")
    commitments = tuple(manifest.get("seed_commitments", ()))
    if commitments != seed_seal.seed_commitments:
        raise RunnerInvariantError(
            "live evaluation commitment seed binding differs"
        )
    expected = {
        "commitments": manifest.get("evaluator_commitments"),
        "digest": manifest.get("evaluator_commitment_digest"),
    }
    if (
        type(expected["commitments"]) is not dict
        or expected["digest"]
        != evaluator_record_digest(
            "evaluator-commitments",
            expected["commitments"],  # type: ignore[arg-type]
        )
        or (not injected_boundaries and seed_seal.path != str(SEALED_SEED_PATH))
    ):
        raise RunnerInvariantError(
            "live evaluation commitment manifest binding differs"
        )
    spec = EvaluatorWorkerSpec(
        purpose="evaluation",
        expected_commitments=seed_seal.seed_commitments,
        expected_final_admission=COMMITMENT_ONLY_ADMISSION_DIGEST,
        seed_path=seed_seal.path,
        commitment_only=True,
    )
    factory = evaluator_client_factory or EvaluatorClient.start
    client = await _maybe_await(factory(spec))
    primary: BaseException | None = None
    snapshot: object = None
    try:
        if not injected_boundaries and type(client) is not EvaluatorClient:
            raise TypeError(
                "live commitment-only factory returned another client type"
            )
        snapshot = getattr(client, "commitment_snapshot")
        if callable(snapshot):
            snapshot = await _maybe_await(snapshot())
        if snapshot != expected:
            raise RunnerInvariantError(
                "live commitment-only evaluator differs from manifest"
            )
    except BaseException as error:
        primary = error
    if primary is not None:
        try:
            await _invalidate_or_reap_evaluator(client)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "commitment-only evaluation handshake and cleanup failed",
                [primary, cleanup],
            ) from None
        raise primary.with_traceback(primary.__traceback__)
    close = getattr(client, "close", None)
    if not callable(close):
        try:
            await _invalidate_or_reap_evaluator(client)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "commitment-only evaluator lacks close and cleanup failed",
                [
                    TypeError("commitment-only evaluator lacks close"),
                    cleanup,
                ],
            ) from None
        raise TypeError("commitment-only evaluator lacks close")
    try:
        await _maybe_await(close())
    except BaseException as primary:
        try:
            await _invalidate_or_reap_evaluator(client)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "commitment-only evaluator close and reap failed",
                [primary, cleanup],
            ) from None
        raise primary.with_traceback(primary.__traceback__)
    canonical = decode_canonical_json(canonical_json_bytes(expected))
    if type(canonical) is not dict:
        raise RunnerInvariantError("commitment-only canonical snapshot differs")
    return canonical


async def _build_live_qualification_owner(
    payload: Mapping[str, object],
    *,
    parent_boundary_validator: Callable[[], Mapping[str, object]] | None = None,
    expected_parent_boundary: Mapping[str, object] | None = None,
    evaluator_client_factory: Callable[[EvaluatorWorkerSpec], Any] | None = None,
    genesis_factory: Callable[[str | Path], FreshGenesisBundle] | None = None,
    foundation_loader: Callable[[], LoadedQwenFoundation] | None = None,
    foundation_model_root: str | Path = MODEL_ROOT,
    foundation_file_verifier: Callable[[Path], Mapping[str, object]] = (
        _verify_qualified_model_files
    ),
    arm_factory_builder: Callable[..., Any] | None = None,
    acquisition_opener: AcquisitionOpener = open_scoped_acquisition,
    torch_module: object | None = None,
    state_root: str | Path = STATE_ROOT,
    scratch_root: str | Path = SCRATCH_ROOT,
    runtime_root: str | Path = QUALIFICATION_RUNTIME_ROOT,
    scope_root: str | Path = QUALIFICATION_COGNEE_SCOPES_ROOT,
    expected_repository_tree: Mapping[str, object] | None = None,
    repository_tree_probe: Callable[[], Mapping[str, object]] | None = None,
    injected_boundaries: bool = False,
) -> _LiveDependenciesOwner:
    """Construct the sole live qualification dependency owner after release."""

    if type(payload) is not dict or set(payload) != {
        "manifest",
        "manifest_sha256",
        "partial_cleanup",
        "purpose",
    }:
        raise RunnerInvariantError("live qualification factory payload differs")
    if payload["purpose"] != "qualification" or type(injected_boundaries) is not bool:
        raise RunnerInvariantError("live qualification factory purpose differs")
    partial = payload["partial_cleanup"]
    if type(partial) is not _PartialConstructionCleanup:
        raise TypeError("live qualification partial cleanup owner differs")
    if expected_repository_tree is None:
        raise RunnerInvariantError("live qualification repository baseline is absent")
    repository_before = _validated_repository_tree_manifest(
        dict(expected_repository_tree)
    )
    repository_probe = repository_tree_probe or (
        lambda: _repository_tree_manifest(REPOSITORY_ROOT)
    )
    if not callable(repository_probe):
        raise TypeError("live qualification repository probe differs")
    manifest = payload["manifest"]
    manifest_sha256 = payload["manifest_sha256"]
    if type(manifest) is not dict:
        raise TypeError("live qualification source manifest differs")
    _raw_digest(manifest_sha256, "live qualification manifest sha256")

    def terminal_repository_validation() -> None:
        validate_source_manifest(
            manifest,
            expected_hash=manifest_sha256,
            repository_root=REPOSITORY_ROOT,
        )
        repository_after = _validated_repository_tree_manifest(
            dict(repository_probe())
        )
        if repository_after != repository_before:
            raise RunnerInvariantError(
                "repository changed during live qualification"
            )

    # First registration is the final LIFO rollback action.  It therefore
    # observes evaluator/Qwen/arm cleanup writes on success and construction
    # failure alike.
    partial.register(
        "QUALIFICATION_REPOSITORY_REVALIDATION_FAILED",
        terminal_repository_validation,
    )
    terminal_repository_validation()
    validate_source_manifest(
        manifest,
        expected_hash=manifest_sha256,
        repository_root=REPOSITORY_ROOT,
    )
    boundary_validator = parent_boundary_validator or validate_live_parent_boundary
    boundary = dict(await _maybe_await(boundary_validator()))
    canonical_json_bytes(boundary)
    if expected_parent_boundary is not None:
        expected_boundary = dict(expected_parent_boundary)
        canonical_json_bytes(expected_boundary)
        if boundary != expected_boundary:
            raise RunnerInvariantError(
                "live parent boundary changed before dependency construction"
            )
    elif not injected_boundaries:
        raise RunnerInvariantError("live parent boundary was not independently rebound")

    owned_state = Path(state_root)
    owned_scratch = Path(scratch_root)
    owned_runtime = Path(runtime_root)
    owned_scopes = Path(scope_root)
    owned_model = Path(foundation_model_root)
    if any(
        not path.is_absolute()
        for path in (
            owned_state,
            owned_scratch,
            owned_runtime,
            owned_scopes,
            owned_model,
        )
    ):
        raise ValueError("live qualification owner paths must be absolute")
    if not injected_boundaries and (
        owned_state != STATE_ROOT
        or owned_scratch != SCRATCH_ROOT
        or owned_runtime != QUALIFICATION_RUNTIME_ROOT
        or owned_scopes != QUALIFICATION_COGNEE_SCOPES_ROOT
        or owned_model != MODEL_ROOT
        or foundation_file_verifier is not _verify_qualified_model_files
    ):
        raise RunnerInvariantError("live qualification owner paths are not frozen")
    if not callable(foundation_file_verifier):
        raise TypeError("live qualification model-file verifier differs")
    _validate_qualification_preconstruction_roots(owned_state, owned_scratch)
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    accounting = RunAccounting(RunMode.QUALIFICATION)
    sampler = _ProductionResourceSampler(
        accounting,
        owned_state,
        owned_scratch,
        torch_module,
    )
    _create_fresh_private_root(owned_runtime)
    _write_private_artifact(
        owned_runtime / "parent-boundary.json",
        canonical_json_bytes(boundary),
        maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
    )
    _fsync_directory(owned_runtime)
    ledger = EvidenceLedger(
        owned_runtime / "attempt-evidence.sqlite3",
        RunIntent(RunMode.QUALIFICATION).intent_ref,
    )

    evaluator_spec = EvaluatorWorkerSpec(
        purpose="qualification",
        expected_commitments=(qualification_replicate_commitment(),),
    )
    client_factory = evaluator_client_factory or EvaluatorClient.start
    evaluator_client = await _maybe_await(client_factory(evaluator_spec))
    partial.register(
        "QUALIFICATION_EVALUATOR_CLEANUP_FAILED",
        lambda: _invalidate_or_reap_evaluator(evaluator_client),
    )
    if not injected_boundaries and type(evaluator_client) is not EvaluatorClient:
        raise TypeError("live qualification evaluator client differs")
    for name in (
        "release_phase",
        "complete_phase",
        "judge_response",
        "random_feedback",
        "close",
        "invalidate",
    ):
        if not callable(getattr(evaluator_client, name, None)):
            raise TypeError(f"live qualification evaluator lacks {name}")
    make_genesis = genesis_factory or reconstruct_fresh_cpu_genesis
    genesis = make_genesis(owned_runtime / "genesis")
    if not injected_boundaries and type(genesis) is not FreshGenesisBundle:
        raise TypeError("live qualification genesis differs")

    load_foundation = foundation_loader or load_qualified_qwen_foundation
    foundation_state: dict[str, object | None] = {
        "loaded": None,
        "owner": None,
    }

    def cleanup_foundation() -> None:
        owner = foundation_state.pop("owner", None)
        foundation_state["loaded"] = None
        if isinstance(owner, _LiveFoundationOwner):
            owner.close()
        else:
            _cleanup_unverified_loaded_foundation(torch_module)

    # Register before invoking the loader so tokenizer/model/OOM partial
    # construction still reaches GC and both CUDA cleanup operations.
    partial.register(
        "QUALIFICATION_QWEN_CLEANUP_FAILED",
        cleanup_foundation,
    )
    loaded_foundation = load_foundation()
    foundation_state["loaded"] = loaded_foundation
    if type(loaded_foundation) is not LoadedQwenFoundation:
        loaded_foundation = None
        foundation_state["loaded"] = None
        raise TypeError("live qualification foundation differs")
    try:
        foundation_owner = _LiveFoundationOwner(
            loaded_foundation,
            owned_model,
            torch_module,
            foundation_file_verifier,
        )
    except BaseException as primary:
        loaded_foundation = None
        foundation_state["loaded"] = None
        _clear_exception_tracebacks(primary)
        raise primary from None
    foundation_state["owner"] = foundation_owner
    loaded_foundation = None

    async def finish_construction() -> _LiveDependenciesOwner:
        sampler.start_after_model_load()
        active_foundation = foundation_owner.loaded
        if active_foundation is None:
            raise RunnerInvariantError("live foundation transfer differs")
        build_arms = arm_factory_builder or HighLevelArmFactory.create
        arm_factory = await _maybe_await(
            build_arms(
                owned_runtime / "arm-runtime",
                purpose="qualification",
                replicate_commitments=(qualification_replicate_commitment(),),
                genesis=genesis,
                io=active_foundation.io,
                manifest=active_foundation.manifest,
                foundation_guard=active_foundation.guard,
                acquisition_opener=acquisition_opener,
                scope_parent=owned_scopes,
            )
        )
        active_foundation = None
        if not injected_boundaries and type(arm_factory) is not HighLevelArmFactory:
            raise TypeError("live qualification arm factory differs")
        executor_owner = _LiveArmExecutorOwner(
            arm_factory,
            evaluator_client,
            ledger,
            accounting,
            sampler,
            owned_runtime / "arm-factory-audit.json",
        )
        partial.register(
            "QUALIFICATION_ARM_FACTORY_CLEANUP_FAILED",
            executor_owner.close,
        )
        accounting.bind_finalizer(executor_owner.finalize_evidence)
        sampler.sample()
        expected_foundation = foundation_owner.expected_hashes

        def foundation_probe() -> dict[str, str]:
            sampler.sample()
            observed = foundation_owner.probe()
            sampler.sample()
            return observed

        dependencies = OrchestrationDependencies(
            evaluator=evaluator_client,
            execute_arm=executor_owner,
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            foundation_probe=foundation_probe,
            expected_foundation_hashes=expected_foundation,
            expected_foundation_tensor_digest=foundation_owner.tensor_digest,
            expected_learner_genesis_digests={
                qualification_replicate_commitment(): (
                    QUALIFIED_INITIAL_COMPETENCE_DIGEST
                )
            },
            accounting=accounting,
            repository_root=REPOSITORY_ROOT,
        )

        def final_source_validation() -> None:
            validate_source_manifest(
                manifest,
                expected_hash=manifest_sha256,
                repository_root=REPOSITORY_ROOT,
            )

        return _LiveDependenciesOwner(
            dependencies,
            final_source_validation,
            "QUALIFICATION_SOURCE_REVALIDATION_FAILED",
        )

    try:
        return await finish_construction()
    except BaseException as primary:
        _clear_exception_tracebacks(primary)
        raise primary from None


async def _build_live_evaluation_owner(
    payload: Mapping[str, object],
    *,
    parent_boundary_validator: Callable[[], Mapping[str, object]] | None = None,
    expected_parent_boundary: Mapping[str, object] | None = None,
    evaluator_client_factory: Callable[[EvaluatorWorkerSpec], Any] | None = None,
    genesis_factory: Callable[[str | Path], FreshGenesisBundle] | None = None,
    foundation_loader: Callable[[], LoadedQwenFoundation] | None = None,
    foundation_model_root: str | Path = MODEL_ROOT,
    foundation_file_verifier: Callable[[Path], Mapping[str, object]] = (
        _verify_qualified_model_files
    ),
    arm_factory_builder: Callable[..., Any] | None = None,
    acquisition_opener: AcquisitionOpener = open_scoped_acquisition,
    torch_module: object | None = None,
    state_root: str | Path = STATE_ROOT,
    scratch_root: str | Path = SCRATCH_ROOT,
    qualification_runtime_root: str | Path = QUALIFICATION_RUNTIME_ROOT,
    qualification_scope_root: str | Path = QUALIFICATION_COGNEE_SCOPES_ROOT,
    qualification_release_path: str | Path = QUALIFICATION_RELEASE_CLAIM_PATH,
    runtime_root: str | Path = EVALUATION_RUNTIME_ROOT,
    scope_root: str | Path = EVALUATION_COGNEE_SCOPES_ROOT,
    expected_repository_tree: Mapping[str, object] | None = None,
    repository_tree_probe: Callable[[], Mapping[str, object]] | None = None,
    injected_boundaries: bool = False,
) -> _LiveDependenciesOwner:
    """Construct the exact two-replicate evaluation owner after admission."""

    expected_payload_fields = {
        "claim",
        "claim_path",
        "claim_ref",
        "manifest",
        "manifest_sha256",
        "partial_cleanup",
        "qualification_result",
        "qualification_result_sha256",
        "result_path",
        "seed_seal",
        "seed_seal_ref",
    }
    if type(payload) is not dict or set(payload) != expected_payload_fields:
        raise RunnerInvariantError("live evaluation factory payload differs")
    if type(injected_boundaries) is not bool:
        raise TypeError("live evaluation injection flag differs")
    partial = payload["partial_cleanup"]
    if type(partial) is not _PartialConstructionCleanup:
        raise TypeError("live evaluation partial cleanup owner differs")
    claim, claim_ref = validate_evaluation_admission_claim(
        payload["claim"],  # type: ignore[arg-type]
        payload["claim_ref"],  # type: ignore[arg-type]
    )
    claim_path = Path(payload["claim_path"])
    result_path = Path(payload["result_path"])
    if (
        not claim_path.is_absolute()
        or not result_path.is_absolute()
        or claim_path != Path(claim["paths"]["claim"])  # type: ignore[index]
        or result_path != Path(claim["paths"]["evaluation_result"])  # type: ignore[index]
        or claim_ref != payload["claim_ref"]
        or claim["manifest_sha256"] != payload["manifest_sha256"]
        or claim["qualification_result_sha256"]
        != payload["qualification_result_sha256"]
        or claim["seed_seal_ref"] != payload["seed_seal_ref"]
    ):
        raise RunnerInvariantError("live evaluation admission payload differs")
    expected_claim_mode = (
        "INJECTED_CPU_TEST" if injected_boundaries else "FROZEN_LIVE"
    )
    if claim["path_binding_mode"] != expected_claim_mode:
        raise RunnerInvariantError("live evaluation admission mode differs")
    if not injected_boundaries:
        _validate_live_claim_manifest_binding(
            claim,
            manifest=payload["manifest"],  # type: ignore[arg-type]
            manifest_sha256=payload["manifest_sha256"],  # type: ignore[arg-type]
        )

    owned_state = Path(state_root)
    owned_scratch = Path(scratch_root)
    owned_qualification_runtime = Path(qualification_runtime_root)
    owned_qualification_scopes = Path(qualification_scope_root)
    owned_qualification_release = Path(qualification_release_path)
    owned_runtime = Path(runtime_root)
    owned_scopes = Path(scope_root)
    owned_model = Path(foundation_model_root)
    owned_seed = Path(claim["paths"]["seed"])  # type: ignore[index]
    owned_qualification_result = Path(
        claim["paths"]["qualification_result"]  # type: ignore[index]
    )
    owned_manifest = Path(claim["paths"]["manifest"])  # type: ignore[index]
    owned_repository = Path(claim["paths"]["repository_root"])  # type: ignore[index]
    if any(
        not path.is_absolute()
        for path in (
            owned_state,
            owned_scratch,
            owned_qualification_runtime,
            owned_qualification_scopes,
            owned_qualification_release,
            owned_runtime,
            owned_scopes,
            owned_model,
            owned_seed,
            owned_qualification_result,
            owned_manifest,
            owned_repository,
        )
    ):
        raise ValueError("live evaluation owner paths must be absolute")
    if not injected_boundaries and (
        owned_state != STATE_ROOT
        or owned_scratch != SCRATCH_ROOT
        or owned_qualification_runtime != QUALIFICATION_RUNTIME_ROOT
        or owned_qualification_scopes != QUALIFICATION_COGNEE_SCOPES_ROOT
        or owned_qualification_release != QUALIFICATION_RELEASE_CLAIM_PATH
        or owned_runtime != EVALUATION_RUNTIME_ROOT
        or owned_scopes != EVALUATION_COGNEE_SCOPES_ROOT
        or owned_model != MODEL_ROOT
        or owned_seed != SEALED_SEED_PATH
        or owned_qualification_result != QUALIFICATION_RESULT_PATH
        or owned_manifest != MANIFEST_PATH
        or owned_repository != REPOSITORY_ROOT
        or claim_path != EVALUATION_ADMISSION_CLAIM_PATH
        or result_path != EVALUATION_RESULT_PATH
        or foundation_file_verifier is not _verify_qualified_model_files
    ):
        raise RunnerInvariantError("live evaluation owner paths are not frozen")
    if not callable(foundation_file_verifier):
        raise TypeError("live evaluation model-file verifier differs")

    (
        manifest,
        manifest_sha256,
        qualification,
        qualification_sha256,
        seed_seal,
        _qualification_release,
        _qualification_release_ref,
    ) = _load_evaluation_preconstruction_evidence(
        manifest_path=owned_manifest,
        seed_path=owned_seed,
        qualification_result_path=owned_qualification_result,
        qualification_release_path=owned_qualification_release,
        repository_root=owned_repository,
        frozen_live=not injected_boundaries,
    )
    if (
        manifest != payload["manifest"]
        or manifest_sha256 != payload["manifest_sha256"]
        or qualification != payload["qualification_result"]
        or qualification_sha256 != payload["qualification_result_sha256"]
        or seed_seal.to_canonical() != payload["seed_seal"]
        or seed_seal.seal_ref != payload["seed_seal_ref"]
        or tuple(manifest["seed_commitments"])
        != seed_seal.seed_commitments
        or claim["source_hashes"] != manifest["frozen_sources"]
        or claim["evaluator_commitment_digest"]
        != manifest["evaluator_commitment_digest"]
    ):
        raise RunnerInvariantError("live evaluation durable preimage join differs")
    loaded_claim, loaded_claim_ref = load_evaluation_admission_claim(
        claim_path,
        expected_ref=claim_ref,
    )
    if loaded_claim != claim or loaded_claim_ref != claim_ref:
        raise RunnerInvariantError("live evaluation durable claim changed")
    for target in (result_path, result_path.with_name(result_path.name + ".tmp")):
        _claim_target_is_absent(target)
    _validate_evaluation_preconstruction_roots(
        owned_state,
        owned_scratch,
        qualification_runtime_root=owned_qualification_runtime,
        qualification_scope_root=owned_qualification_scopes,
        evaluation_runtime_root=owned_runtime,
        evaluation_scope_root=owned_scopes,
        seed_path=owned_seed,
        qualification_release_path=owned_qualification_release,
        admission_claim_path=claim_path,
        admission_required=True,
    )
    if expected_repository_tree is None:
        raise RunnerInvariantError("live evaluation repository baseline is absent")
    repository_before = _validated_repository_tree_manifest(
        dict(expected_repository_tree)
    )
    repository_probe = repository_tree_probe or (
        lambda: _repository_tree_manifest(REPOSITORY_ROOT)
    )
    if not callable(repository_probe):
        raise TypeError("live evaluation repository probe differs")

    def terminal_repository_validation() -> None:
        validate_source_manifest(
            manifest,
            expected_hash=manifest_sha256,
            repository_root=owned_repository,
        )
        repository_after = _validated_repository_tree_manifest(
            dict(repository_probe())
        )
        if repository_after != repository_before:
            raise RunnerInvariantError("repository changed during live evaluation")

    # Registered first, so it executes after all resource cleanup on rollback.
    partial.register(
        "EVALUATION_REPOSITORY_REVALIDATION_FAILED",
        terminal_repository_validation,
    )
    terminal_repository_validation()
    boundary_validator = parent_boundary_validator or validate_live_parent_boundary
    boundary = dict(await _maybe_await(boundary_validator()))
    canonical_json_bytes(boundary)
    if expected_parent_boundary is not None:
        expected_boundary = dict(expected_parent_boundary)
        canonical_json_bytes(expected_boundary)
        if boundary != expected_boundary:
            raise RunnerInvariantError(
                "live parent boundary changed before evaluation construction"
            )
    elif not injected_boundaries:
        raise RunnerInvariantError(
            "live evaluation parent boundary was not independently rebound"
        )

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    accounting = RunAccounting(RunMode.EVALUATION)
    sampler = _ProductionResourceSampler(
        accounting,
        owned_state,
        owned_scratch,
        torch_module,
    )
    _create_fresh_private_root(owned_runtime)
    _write_private_artifact(
        owned_runtime / "parent-boundary.json",
        canonical_json_bytes(boundary),
        maximum_bytes=MAXIMUM_EVIDENCE_RECORD_BYTES,
    )
    _fsync_directory(owned_runtime)
    ledger = EvidenceLedger(
        owned_runtime / "attempt-evidence.sqlite3",
        RunIntent(RunMode.EVALUATION).intent_ref,
    )

    evaluator_spec = EvaluatorWorkerSpec(
        purpose="evaluation",
        expected_commitments=seed_seal.seed_commitments,
        expected_final_admission=claim["admission_digest"],  # type: ignore[arg-type]
        seed_path=str(owned_seed),
        expected_evaluator_commitments_json=canonical_json_bytes(
            manifest["evaluator_commitments"]
        ).decode("utf-8"),
        expected_evaluator_commitment_digest=manifest[
            "evaluator_commitment_digest"
        ],  # type: ignore[arg-type]
    )
    client_factory = evaluator_client_factory or EvaluatorClient.start
    evaluator_client = await _maybe_await(client_factory(evaluator_spec))
    partial.register(
        "EVALUATION_EVALUATOR_CLEANUP_FAILED",
        lambda: _invalidate_or_reap_evaluator(evaluator_client),
    )
    if not injected_boundaries and type(evaluator_client) is not EvaluatorClient:
        raise TypeError("live evaluation evaluator client differs")
    for name in (
        "commitments",
        "release_phase",
        "complete_phase",
        "admit_final",
        "judge_response",
        "random_feedback",
        "final_metrics",
        "close",
        "invalidate",
    ):
        if not callable(getattr(evaluator_client, name, None)):
            raise TypeError(f"live evaluation evaluator lacks {name}")
    client_snapshot = getattr(evaluator_client, "commitment_snapshot", None)
    if callable(client_snapshot):
        client_snapshot = await _maybe_await(client_snapshot())
    if client_snapshot != {
        "commitments": manifest["evaluator_commitments"],
        "digest": manifest["evaluator_commitment_digest"],
    }:
        raise RunnerInvariantError("full evaluation client differs from manifest")

    make_genesis = genesis_factory or reconstruct_fresh_cpu_genesis
    genesis = make_genesis(owned_runtime / "genesis")
    if not injected_boundaries and type(genesis) is not FreshGenesisBundle:
        raise TypeError("live evaluation genesis differs")

    load_foundation = foundation_loader or load_qualified_qwen_foundation
    foundation_state: dict[str, object | None] = {
        "loaded": None,
        "owner": None,
    }

    def cleanup_foundation() -> None:
        owner = foundation_state.pop("owner", None)
        foundation_state["loaded"] = None
        if isinstance(owner, _LiveFoundationOwner):
            owner.close()
        else:
            _cleanup_unverified_loaded_foundation(torch_module)

    partial.register(
        "EVALUATION_QWEN_CLEANUP_FAILED",
        cleanup_foundation,
    )
    loaded_foundation = load_foundation()
    foundation_state["loaded"] = loaded_foundation
    if type(loaded_foundation) is not LoadedQwenFoundation:
        loaded_foundation = None
        foundation_state["loaded"] = None
        raise TypeError("live evaluation foundation differs")
    try:
        foundation_owner = _LiveFoundationOwner(
            loaded_foundation,
            owned_model,
            torch_module,
            foundation_file_verifier,
        )
    except BaseException as primary:
        loaded_foundation = None
        foundation_state["loaded"] = None
        _clear_exception_tracebacks(primary)
        raise primary from None
    foundation_state["owner"] = foundation_owner
    loaded_foundation = None

    async def finish_construction() -> _LiveDependenciesOwner:
        sampler.start_after_model_load()
        active_foundation = foundation_owner.loaded
        if active_foundation is None:
            raise RunnerInvariantError("live evaluation foundation transfer differs")
        build_arms = arm_factory_builder or HighLevelArmFactory.create
        arm_factory = await _maybe_await(
            build_arms(
                owned_runtime / "arm-runtime",
                purpose="evaluation",
                replicate_commitments=seed_seal.seed_commitments,
                genesis=genesis,
                io=active_foundation.io,
                manifest=active_foundation.manifest,
                foundation_guard=active_foundation.guard,
                acquisition_opener=acquisition_opener,
                scope_parent=owned_scopes,
            )
        )
        active_foundation = None
        if not injected_boundaries and type(arm_factory) is not HighLevelArmFactory:
            raise TypeError("live evaluation arm factory differs")
        if type(arm_factory) is HighLevelArmFactory and (
            arm_factory.purpose != "evaluation"
            or set(arm_factory.lineages)
            != {
                (replicate, arm)
                for replicate in seed_seal.seed_commitments
                for arm in ("FULL", "RANDOM_FEEDBACK")
            }
        ):
            raise RunnerInvariantError("live evaluation lineage owners differ")
        executor_owner = _LiveArmExecutorOwner(
            arm_factory,
            evaluator_client,
            ledger,
            accounting,
            sampler,
            owned_runtime / "arm-factory-audit.json",
        )
        partial.register(
            "EVALUATION_ARM_FACTORY_CLEANUP_FAILED",
            executor_owner.close,
        )
        accounting.bind_finalizer(executor_owner.finalize_evidence)
        sampler.sample()
        expected_foundation = foundation_owner.expected_hashes

        def foundation_probe() -> dict[str, str]:
            sampler.sample()
            observed = foundation_owner.probe()
            sampler.sample()
            return observed

        dependencies = OrchestrationDependencies(
            evaluator=evaluator_client,
            execute_arm=executor_owner,
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            foundation_probe=foundation_probe,
            expected_foundation_hashes=expected_foundation,
            expected_foundation_tensor_digest=foundation_owner.tensor_digest,
            expected_learner_genesis_digests={
                replicate: QUALIFIED_INITIAL_COMPETENCE_DIGEST
                for replicate in seed_seal.seed_commitments
            },
            accounting=accounting,
            repository_root=owned_repository,
        )

        def final_source_validation() -> None:
            validate_source_manifest(
                manifest,
                expected_hash=manifest_sha256,
                repository_root=owned_repository,
            )

        return _LiveDependenciesOwner(
            dependencies,
            final_source_validation,
            "EVALUATION_SOURCE_REVALIDATION_FAILED",
        )

    try:
        return await finish_construction()
    except BaseException as primary:
        _clear_exception_tracebacks(primary)
        raise primary from None


async def _run_live_qualification_cli() -> dict[str, object]:
    """Validate/freeze public inputs, then enter the sole qualification path."""

    for target in (
        QUALIFICATION_RUNTIME_ROOT,
        QUALIFICATION_COGNEE_SCOPES_ROOT,
        QUALIFICATION_RESULT_PATH,
        QUALIFICATION_RESULT_PATH.with_name(QUALIFICATION_RESULT_PATH.name + ".tmp"),
        QUALIFICATION_RELEASE_CLAIM_PATH,
        QUALIFICATION_RELEASE_CLAIM_PATH.with_name(
            QUALIFICATION_RELEASE_CLAIM_PATH.name + ".tmp"
        ),
    ):
        _require_identity_absent(target, QualificationIdentityConsumed)
    _validate_qualification_preconstruction_roots(
        STATE_ROOT,
        SCRATCH_ROOT,
        release_required=False,
    )
    boundary = validate_live_parent_boundary(configure_pools=True)
    # Parent-boundary validation must not mutate either externally provisioned
    # scratch directory before manifest construction is admitted.
    _validate_qualification_preconstruction_roots(
        STATE_ROOT,
        SCRATCH_ROOT,
        release_required=False,
    )
    await _ensure_live_source_manifest()
    # The commitment-only worker must leave the exact pre-release shape it
    # observed.  Recheck before the O_EXCL qualification release is created.
    _validate_qualification_preconstruction_roots(
        STATE_ROOT,
        SCRATCH_ROOT,
        release_required=False,
    )
    repository_before = _repository_tree_manifest(REPOSITORY_ROOT)

    async def factory(payload: Mapping[str, object]) -> _LiveDependenciesOwner:
        return await _build_live_qualification_owner(
            payload,
            parent_boundary_validator=lambda: validate_live_parent_boundary(
                configure_pools=False
            ),
            expected_parent_boundary=boundary,
            expected_repository_tree=repository_before,
        )

    return await _run_and_publish_qualification_once(
        factory,
        manifest_path=MANIFEST_PATH,
        result_path=QUALIFICATION_RESULT_PATH,
        release_claim_path=QUALIFICATION_RELEASE_CLAIM_PATH,
        repository_root=REPOSITORY_ROOT,
        injected_paths=False,
    )


async def _run_live_evaluation_cli() -> dict[str, object]:
    """Enter the one-use evaluation only from exact retained PASS evidence."""

    for target in (
        EVALUATION_RUNTIME_ROOT,
        EVALUATION_COGNEE_SCOPES_ROOT,
        EVALUATION_RESULT_PATH,
        EVALUATION_RESULT_PATH.with_name(EVALUATION_RESULT_PATH.name + ".tmp"),
        EVALUATION_ADMISSION_CLAIM_PATH,
        EVALUATION_ADMISSION_CLAIM_PATH.with_name(
            EVALUATION_ADMISSION_CLAIM_PATH.name + ".tmp"
        ),
    ):
        _require_identity_absent(target, EvaluationIdentityConsumed)
    _validate_evaluation_preconstruction_roots(
        STATE_ROOT,
        SCRATCH_ROOT,
        admission_required=False,
    )
    evidence = _load_evaluation_preconstruction_evidence(
        manifest_path=MANIFEST_PATH,
        seed_path=SEALED_SEED_PATH,
        qualification_result_path=QUALIFICATION_RESULT_PATH,
        qualification_release_path=QUALIFICATION_RELEASE_CLAIM_PATH,
        repository_root=REPOSITORY_ROOT,
        frozen_live=True,
    )
    manifest, _manifest_sha256, _qualification, _qualification_sha256, seal, *_ = (
        evidence
    )
    boundary = validate_live_parent_boundary(configure_pools=True)
    _validate_evaluation_preconstruction_roots(
        STATE_ROOT,
        SCRATCH_ROOT,
        admission_required=False,
    )
    if _load_evaluation_preconstruction_evidence(
        manifest_path=MANIFEST_PATH,
        seed_path=SEALED_SEED_PATH,
        qualification_result_path=QUALIFICATION_RESULT_PATH,
        qualification_release_path=QUALIFICATION_RELEASE_CLAIM_PATH,
        repository_root=REPOSITORY_ROOT,
        frozen_live=True,
    ) != evidence:
        raise RunnerInvariantError(
            "evaluation preconstruction evidence changed before handshake"
        )
    repository_before = _repository_tree_manifest(REPOSITORY_ROOT)

    async def handshake() -> dict[str, object]:
        observed = await _run_live_evaluation_commitment_handshake(
            manifest,
            seal,
        )
        _validate_evaluation_preconstruction_roots(
            STATE_ROOT,
            SCRATCH_ROOT,
            admission_required=False,
        )
        if validate_live_parent_boundary(configure_pools=False) != boundary:
            raise RunnerInvariantError(
                "live parent boundary changed during evaluation handshake"
            )
        if _load_evaluation_preconstruction_evidence(
            manifest_path=MANIFEST_PATH,
            seed_path=SEALED_SEED_PATH,
            qualification_result_path=QUALIFICATION_RESULT_PATH,
            qualification_release_path=QUALIFICATION_RELEASE_CLAIM_PATH,
            repository_root=REPOSITORY_ROOT,
            frozen_live=True,
        ) != evidence:
            raise RunnerInvariantError(
                "evaluation evidence changed during commitment handshake"
            )
        repository_after = _repository_tree_manifest(REPOSITORY_ROOT)
        if repository_after != repository_before:
            raise RunnerInvariantError(
                "repository changed during evaluation commitment handshake"
            )
        return observed

    async def factory(payload: Mapping[str, object]) -> _LiveDependenciesOwner:
        return await _build_live_evaluation_owner(
            payload,
            parent_boundary_validator=lambda: validate_live_parent_boundary(
                configure_pools=False
            ),
            expected_parent_boundary=boundary,
            expected_repository_tree=repository_before,
        )

    return await _run_and_publish_evaluation_once(
        factory,
        evaluator_handshake=handshake,
    )


async def run_and_publish_evaluation_once() -> dict[str, object]:
    """Run the sole frozen zero-injection evaluation-to-terminal path."""

    return await _run_live_evaluation_cli()


def evaluator_worker_main(
    input_stream: Any = None,
    output_stream: Any = None,
) -> None:
    """Serve bounded canonical JSONL until exact close or fail-closed error."""

    source = input_stream or sys.stdin.buffer
    sink = output_stream or sys.stdout.buffer
    state = EvaluatorProtocolState()
    while not state.closed:
        request_id = state.next_request_id
        try:
            raw = source.readline(MAXIMUM_IPC_BYTES + 1)
            if (
                not raw
                or len(raw) > MAXIMUM_IPC_BYTES
                or not raw.endswith(b"\n")
            ):
                raise EvaluatorProtocolError(
                    "worker input ended or exceeded its bound"
                )
            value = decode_canonical_json(raw[:-1])
            if type(value) is not dict:
                raise EvaluatorProtocolError("worker request is not an object")
            response = dispatch_evaluator_request(None, value, state)
        except BaseException:
            state.invalidate()
            response = _generic_error(request_id)
            encoded = canonical_json_bytes(response) + b"\n"
            sink.write(encoded)
            sink.flush()
            raise SystemExit(2) from None
        encoded = canonical_json_bytes(response) + b"\n"
        if len(encoded) > MAXIMUM_IPC_BYTES:
            raise EvaluatorProtocolError("worker response exceeds IPC ceiling")
        sink.write(encoded)
        sink.flush()


def main(argv: Sequence[str] | None = None) -> None:
    intent = RunIntent.parse(tuple(sys.argv[1:] if argv is None else argv))
    if intent.mode is RunMode.EVALUATOR_WORKER:
        try:
            _validate_evaluator_worker_environment()
            evaluator_worker_main()
        except SystemExit:
            raise
        except BaseException:
            raise SystemExit(2) from None
        return
    if intent.mode is RunMode.SEAL_EVALUATION_SEEDS:
        seal = seal_evaluation_seeds(SEALED_SEED_PATH)
        print(canonical_json_bytes(seal.to_canonical()).decode("utf-8"), flush=True)
        return
    if intent.mode is RunMode.QUALIFICATION:
        result = asyncio.run(run_and_publish_qualification_once())
        summary = {
            "classification": result["classification"],
            "result": str(QUALIFICATION_RESULT_PATH),
        }
        print(canonical_json_bytes(summary).decode("utf-8"), flush=True)
        return
    if intent.mode is RunMode.EVALUATION:
        result = asyncio.run(run_and_publish_evaluation_once())
        summary = {
            "classification": result["classification"],
            "result": str(EVALUATION_RESULT_PATH),
        }
        print(canonical_json_bytes(summary).decode("utf-8"), flush=True)
        return
    raise RunnerInvariantError(f"unsupported live mode: {intent.mode.value}")


if __name__ == "__main__":
    main()
