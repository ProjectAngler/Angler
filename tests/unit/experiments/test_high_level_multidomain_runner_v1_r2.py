"""CPU-only contract tests for the high-level multi-domain runner."""

from __future__ import annotations

from dataclasses import replace
import asyncio
from collections import Counter
import copy
import gc
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile
import unittest
from unittest import mock
import weakref


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


from experiments.evaluators import high_level_multidomain_v1 as evaluator  # noqa: E402
from experiments.runners import high_level_multidomain_v1_r2 as runner  # noqa: E402


_FROZEN_SOURCE_HASHES = {
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

_IMMUTABLE_V1_EVIDENCE_SHA256 = {
    "exact_qwen_integration": "05836ee77994f8905a8a2523135872a81b8b67bcdae77e102e730ac9ef55025d",
    "leaf": "b8678803ac2ea63f44d9f1e386dffefb907be619d208c18aba71de31e39e629a",
    "release_claim": "9002e17cc58de95796cc2dd1812509a68e3a9c81e1dcc9c0b4a2def54c6eaa61",
    "result_report": "e532ecaba5c8efa2d479531c9a9686dbe80ad7589f3ee4058f0be18cfe38ad97",
    "retained_scratch_tree": "362c077ca959b55a44fd322df25ab468a79914013119d9ecb8a1107171a04062",
    "retained_state_tree": "f23bda8dfb077450aaed47bc2ebb920d5594fef310f169ff0f76967f943a6cbb",
    "runner": "4c9dc1551d5902f4132ac8d7262a025c17d72299de79d872f0b300d0299356d7",
    "runner_tests": "5a225fdd57b8690cf33b63eed5d0d8cf25d575ee46d929a7e9e67f7329d9ae92",
    "seed_seal": "74e035d78a27c7ddcafe8702b9cb1e35e5b9c5e5816e22f4d65d5b1d46d6d5bc",
    "source_manifest": "91542cc7557495c7793f82247772e71b0f00f6995e3b4b3fbf2185ee38b470f8",
    "terminal_result": "feae381e64781d458e2cce5661e63393d165680964a2de24739cc9e24a4d34e1",
}

_IMMUTABLE_V1_FILE_PATHS = {
    "exact_qwen_integration": ROOT
    / "tests/integration/runtime/test_high_level_multidomain_qwen_v1.py",
    "leaf": ROOT
    / "docs/blueprints/branches/science/work/"
    "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-001.md",
    "release_claim": Path(
        "/opt/angler/state/project-angler/high-level-multidomain-v1/"
        "sealed/qualification-release-claim-v1.json"
    ),
    "result_report": ROOT / "docs/reports/HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_RESULT.md",
    "runner": ROOT / "experiments/runners/high_level_multidomain_v1.py",
    "runner_tests": ROOT
    / "tests/unit/experiments/test_high_level_multidomain_runner_v1.py",
    "seed_seal": Path(
        "/opt/angler/state/project-angler/high-level-multidomain-v1/"
        "sealed/seeds-v1.json"
    ),
    "source_manifest": ROOT / "experiments/manifests/high-level-multidomain-v1.json",
    "terminal_result": Path(
        "/opt/angler/results/high-level-multidomain-v1-qualification.json"
    ),
}

_IMMUTABLE_R1_EVIDENCE_SHA256 = {
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

_R1_STATE_ROOT = Path(
    "/opt/angler/state/project-angler/high-level-multidomain-v1-r1"
)
_R1_QUALIFICATION_ROOT = _R1_STATE_ROOT / "qualification-v1-r1"
_IMMUTABLE_R1_FILE_PATHS = {
    "active_leaf": ROOT
    / "docs/blueprints/branches/science/work/"
    "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R1-"
    "LIVE-EVALUATION-COMPLETION-001.md",
    "arm_factory_audit": _R1_QUALIFICATION_ROOT / "arm-factory-audit.json",
    "attempt_ledger": _R1_QUALIFICATION_ROOT / "attempt-evidence.sqlite3",
    "exact_qwen_integration": ROOT
    / "tests/integration/runtime/test_high_level_multidomain_qwen_v1_r1.py",
    "release_claim": _R1_STATE_ROOT
    / "sealed/qualification-release-claim-v1-r1.json",
    "result_report": ROOT
    / "docs/reports/HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_R1_RESULT.md",
    "runner": ROOT / "experiments/runners/high_level_multidomain_v1_r1.py",
    "runner_tests": ROOT
    / "tests/unit/experiments/test_high_level_multidomain_runner_v1_r1.py",
    "seed_seal": _R1_STATE_ROOT / "sealed/seeds-v1-r1.json",
    "source_manifest": ROOT
    / "experiments/manifests/high-level-multidomain-v1-r1.json",
    "terminal_result": Path(
        "/opt/angler/results/high-level-multidomain-v1-r1-qualification.json"
    ),
}

_R1_EVALUATION_PATHS = (
    _R1_STATE_ROOT / "evaluation-v1-r1",
    _R1_STATE_ROOT / "evaluation-cognee-scopes",
    _R1_STATE_ROOT / "sealed/evaluation-admission-claim-v1-r1.json",
    Path("/opt/angler/results/high-level-multidomain-v1-r1.json"),
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_signature(path: Path) -> object:
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if not path.exists():
        return None
    if path.is_file():
        return ("file", hashlib.sha256(path.read_bytes()).hexdigest())
    return (
        "directory",
        tuple(
            (
                str(item.relative_to(path)),
                "directory"
                if item.is_dir()
                else hashlib.sha256(item.read_bytes()).hexdigest(),
            )
            for item in sorted(path.rglob("*"))
            if not item.is_symlink()
        ),
    )


def _local_tree_digest(root: Path) -> tuple[str, int, int, int, int]:
    if root.is_symlink() or not root.is_dir():
        raise AssertionError(f"tree root is not an exact directory: {root}")
    entries = (root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()))
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    logical_bytes = 0
    for path in entries:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            size = 0
            file_digest = None
            directory_count += 1
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            kind = "file"
            size = metadata.st_size
            content = hashlib.sha256()
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NOATIME", 0),
            )
            observed_bytes = 0
            try:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    content.update(chunk)
                    observed_bytes += len(chunk)
                final_metadata = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (
                observed_bytes != size
                or final_metadata.st_size != size
                or final_metadata.st_ino != metadata.st_ino
                or final_metadata.st_dev != metadata.st_dev
            ):
                raise AssertionError("tree file changed during digest")
            file_digest = content.hexdigest()
            file_count += 1
            logical_bytes += size
        else:
            raise AssertionError(f"tree contains a link or special entry: {path}")
        record = [
            relative,
            kind,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_uid,
            metadata.st_gid,
            size,
            file_digest,
        ]
        digest.update(
            json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
    return digest.hexdigest(), file_count, directory_count, logical_bytes, len(entries)


def _fair_arm_evidence(task_id: str, arm: str) -> dict[str, object]:
    common = {
        "backend_raw_hit_count": 1,
        "backend_rejected_hit_count": 0,
        "evidence_final_ref": _digest(f"final:{task_id}"),
        "evidence_stage_ref": _digest(f"stage:{task_id}"),
        "task_id": task_id,
        "public_task_ref": _digest(f"public:{task_id}"),
        "frozen_recall_ref": _digest(f"recall:{task_id}"),
        "recalled_record_refs": [_digest(f"record:{task_id}")],
        "recalled_record_bytes_sha256": [
            hashlib.sha256(task_id.encode("ascii")).hexdigest()
        ],
        "proposal_request_bytes": task_id.encode("ascii").hex(),
        "proposal_generation_ref": _digest(f"proposal-generation:{task_id}"),
        "proposal_generation_bytes": b"PUBLIC-PROPOSALS".hex(),
        "proposal_prompt_ref": _digest(f"proposal-prompt:{task_id}"),
        "proposals": ["PUBLIC-PROCEDURE-ZERO", "PUBLIC-PROCEDURE-ONE"],
        "selection_bytes": b"PUBLIC-SELECTION".hex(),
        "selection_ref": _digest(f"selection:{task_id}"),
        "selected_trace": "PUBLIC-PROCEDURE-ZERO",
        "execution_prompt_ref": _digest(f"execution-prompt:{task_id}"),
        "execution_request_bytes": b"PUBLIC-EXECUTION-REQUEST".hex(),
        "execution_request_ref": _digest(f"execution-request:{task_id}"),
        "task_response_generation_bytes": b"PUBLIC-TASK-GENERATION".hex(),
        "task_response_generation_ref": _digest(f"task-generation:{task_id}"),
        "execution_receipt_bytes": b"PUBLIC-EXECUTION-RECEIPT".hex(),
        "execution_receipt_ref": _digest(f"execution-receipt:{task_id}"),
        "raw_response": "PUBLIC-MODEL-RESPONSE",
        "runtime_quiescence_ref": _digest(f"quiescence:{task_id}"),
        "score": 0.0,
    }
    return {
        **common,
        "backend_search_calls": (
            0 if arm in ("BACKEND_REMOVAL", "RETRIEVAL_ONLY") else 1
        ),
        "probe_integrity_ref": _digest(f"probe:{task_id}:{arm}"),
    }


def _valid_final_metrics() -> dict[str, object]:
    counts = {
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
    rows = []
    for phase in ("adaptation", "development", "final"):
        arms = (
            ("FULL", "RANDOM_FEEDBACK")
            if phase == "adaptation"
            else evaluator.EVALUATION_ARMS
        )
        for arm in arms:
            for family in evaluator.FAMILIES:
                attempts = counts[phase][family]
                rows.append(
                    {
                        "arm": arm,
                        "attempts": attempts,
                        "binary_success_total": 0,
                        "family": family,
                        "pairwise_agreement_total": (
                            0.0
                            if family == "symbolic-demonstration-transfer"
                            else None
                        ),
                        "phase": phase,
                    }
                )
    return {
        "aggregates": rows,
        "identity": evaluator.EVALUATION_IDENTITY,
        "purpose": "evaluation",
        "schema": evaluator.SUITE_SCHEMA,
    }


def _final_metrics_with_scores(
    scores: dict[str, tuple[int, int, int]],
) -> dict[str, object]:
    metrics = _valid_final_metrics()
    family_index = {
        family: index for index, family in enumerate(evaluator.FAMILIES)
    }
    for row in metrics["aggregates"]:
        if row["phase"] == "final" and row["arm"] in scores:
            row["binary_success_total"] = scores[row["arm"]][
                family_index[row["family"]]
            ]
    return metrics


class _CanonicalRecord:
    def __init__(self, value: dict[str, object], digest: str | None = None) -> None:
        self.value = value
        self.digest = digest or _digest("canonical-record")

    def to_canonical(self) -> dict[str, object]:
        return dict(self.value)


class _FakeEvaluator:
    def __init__(self, commitment: str) -> None:
        commitment_payload = {"replicate_commitments": [commitment]}
        self.commitments = _CanonicalRecord(
            commitment_payload,
            runner.evaluator_record_digest(
                "evaluator-commitments",
                commitment_payload,
            ),
        )
        self.calls: list[tuple[str, object]] = []

    @property
    def completed_phases(self) -> tuple[str, ...]:
        return ("adaptation",)

    @property
    def released_phases(self) -> tuple[str, ...]:
        return ("adaptation", "development")

    def release_phase(self, phase: str) -> tuple[_CanonicalRecord, ...]:
        self.calls.append(("release_phase", phase))
        return (_CanonicalRecord({"phase": phase, "task_id": _digest("ipc-task")}),)

    def complete_phase(self, phase: str) -> None:
        self.calls.append(("complete_phase", phase))

    def admit_final(self, admission_digest: str) -> None:
        self.calls.append(("admit_final", admission_digest))

    def judge_response(
        self,
        task_id: str,
        arm: str,
        attempt_receipt_ref: str,
        raw_response: str,
    ) -> _CanonicalRecord:
        self.calls.append(
            (
                "judge_response",
                (task_id, arm, attempt_receipt_ref, raw_response),
            )
        )
        return _CanonicalRecord(
            {
                "arm": arm,
                "attempt_receipt_ref": attempt_receipt_ref,
                "disposition": "UNSUCCESSFUL",
                "raw_response": raw_response,
                "response_commitment": _digest("ipc-response"),
                "score": 0.0,
                "task_id": task_id,
            }
        )

    def random_feedback_schedule(
        self,
        replicate_commitment: str,
    ) -> tuple[tuple[str, float], ...]:
        self.calls.append(("random_feedback", replicate_commitment))
        return ((_digest("ipc-task"), 0.0),)

    def final_metrics(self) -> _CanonicalRecord:
        self.calls.append(("final_metrics", None))
        return _CanonicalRecord(
            {"aggregates": [], "identity": "fake", "purpose": "evaluation"},
            _digest("ipc-metrics"),
        )


class _CommitmentOnlyEvaluator:
    def __init__(self, commitments: tuple[str, str]) -> None:
        payload = {"replicate_commitments": list(commitments)}
        self.commitments = _CanonicalRecord(
            payload,
            runner.evaluator_record_digest("evaluator-commitments", payload),
        )

    @property
    def completed_phases(self) -> tuple[()]:
        return ()

    @property
    def released_phases(self) -> tuple[()]:
        return ()


class _CaptureWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.closed = False

    def write(self, value: bytes) -> int:
        self.chunks.append(value)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _ScriptedReader:
    def __init__(self, rows: list[bytes]) -> None:
        self.rows = rows

    def readline(self, _: int = -1) -> bytes:
        return self.rows.pop(0) if self.rows else b""


class _ScriptedProcess:
    def __init__(self, rows: list[bytes]) -> None:
        self.stdin = _CaptureWriter()
        self.stdout = _ScriptedReader(rows)
        self.stderr = io.BytesIO()
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return -15 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        return -15 if self.terminated else 0


class _FakeOrchestrationEvaluator:
    def __init__(self, purpose: str) -> None:
        self.purpose = purpose
        self.calls: list[tuple[str, object]] = []
        self.judged_pairs: set[tuple[str, str]] = set()
        self.receipts: set[str] = set()
        if purpose == "qualification":
            self.replicates = (_digest("qualification-replicate-0"),)
        else:
            self.replicates = tuple(
                evaluator.commit_replicate_seed(seed)
                for seed in (bytes(range(32)), bytes(range(32, 64)))
            )
        replicate_count = len(self.replicates)
        counts = (
            {"adaptation": 12, "development": 10}
            if purpose == "qualification"
            else {"adaptation": 24, "development": 20, "final": 40}
        )
        self.tasks = {
            phase: self._tasks(phase, count)
            for phase, count in counts.items()
        }
        self.schedules = {
            replicate: tuple(
                (task["task_id"], float(index >= 6))
                for index, task in enumerate(
                    tuple(
                        task
                        for task in self.tasks["adaptation"]
                        if task["replicate_commitment"] == replicate
                    )
                )
            )
            for replicate in self.replicates
        }
        self.admitted: str | None = None
        self.closed = False
        self.invalidated = False

    def bind_evaluation_tasks(
        self,
        commitments: list[dict[str, object]],
    ) -> None:
        if self.purpose != "evaluation":
            raise AssertionError("only evaluation fake tasks can be rebound")
        rebound: dict[str, list[dict[str, object]]] = {
            phase: [] for phase in runner.PHASES
        }
        for task in commitments:
            replicate_index = self.replicates.index(task["replicate_commitment"])
            rebound[task["phase"]].append(
                {
                    "family": task["family"],
                    "ordinal": task["ordinal"],
                    "payload": {"opaque": task["public_commitment"]},
                    "phase": task["phase"],
                    "public_commitment": task["public_commitment"],
                    "replicate": f"replicate-{replicate_index + 1:02d}",
                    "replicate_commitment": task["replicate_commitment"],
                    "schema": evaluator.SUITE_SCHEMA,
                    "task_id": task["task_id"],
                }
            )
        self.tasks = {
            phase: tuple(rows) for phase, rows in rebound.items()
        }
        self.schedules = {
            replicate: tuple(
                (task["task_id"], float(index >= 6))
                for index, task in enumerate(
                    tuple(
                        task
                        for task in self.tasks["adaptation"]
                        if task["replicate_commitment"] == replicate
                    )
                )
            )
            for replicate in self.replicates
        }

    def _tasks(self, phase: str, count: int) -> tuple[dict[str, object], ...]:
        per_replicate = count // len(self.replicates)
        rows = []
        for index in range(count):
            replicate = self.replicates[index // per_replicate]
            rows.append(
                {
                    "family": evaluator.FAMILIES[index % len(evaluator.FAMILIES)],
                    "ordinal": index,
                    "payload": {"opaque": f"{phase}-{index}"},
                    "phase": phase,
                    "public_commitment": _digest(
                        f"{self.purpose}-{phase}-public-{index}"
                    ),
                    "replicate": (
                        "qualification-01"
                        if self.purpose == "qualification"
                        else f"replicate-{index // per_replicate + 1:02d}"
                    ),
                    "replicate_commitment": replicate,
                    "schema": evaluator.SUITE_SCHEMA,
                    "task_id": _digest(f"{self.purpose}-{phase}-task-{index}"),
                }
            )
        return tuple(rows)

    def release_phase(self, phase: str) -> tuple[dict[str, object], ...]:
        self.calls.append(("release_phase", phase))
        return self.tasks[phase]

    def complete_phase(self, phase: str) -> None:
        self.calls.append(("complete_phase", phase))

    def random_feedback(
        self,
        replicate_commitment: str,
    ) -> tuple[tuple[str, float], ...]:
        self.calls.append(("random_feedback", replicate_commitment))
        return self.schedules[replicate_commitment]

    def judge(
        self,
        spec: runner.ArmTaskSpec,
        attempt_receipt_ref: str,
        raw_response: str,
    ) -> dict[str, object] | None:
        if spec.judge_arm is None:
            return None
        key = (spec.task_id, spec.judge_arm)
        if key in self.judged_pairs or attempt_receipt_ref in self.receipts:
            raise AssertionError("fake evaluator identity reused")
        self.judged_pairs.add(key)
        self.receipts.add(attempt_receipt_ref)
        self.calls.append(("judge_response", key))
        response_payload = {
            "arm": spec.judge_arm,
            "attempt_receipt_ref": attempt_receipt_ref,
            "raw_response": raw_response,
            "task_id": spec.task_id,
        }
        return {
            "arm": spec.judge_arm,
            "attempt_receipt_ref": attempt_receipt_ref,
            "disposition": "UNSUCCESSFUL",
            "raw_response": raw_response,
            "response_commitment": runner.evaluator_record_digest(
                "response",
                response_payload,
            ),
            "score": 0.0,
            "task_id": spec.task_id,
        }

    def admit_final(self, admission_digest: str) -> None:
        self.calls.append(("admit_final", admission_digest))
        self.admitted = admission_digest

    def final_metrics(self) -> dict[str, object]:
        self.calls.append(("final_metrics", None))
        return _valid_final_metrics()

    def close(self) -> None:
        self.calls.append(("close", None))
        self.closed = True

    def invalidate(self) -> None:
        self.calls.append(("invalidate", None))
        self.invalidated = True
        self.closed = True


class _FakeProbeIntegrityFactory:
    def __init__(self, root: Path, foundation_tensor_digest: str) -> None:
        self.root = root
        self.foundation_tensor_digest = foundation_tensor_digest
        self.clone_hashes = {
            "journal": hashlib.sha256(b"fixture-journal-baseline").hexdigest(),
            "learner": hashlib.sha256(b"fixture-learner-baseline").hexdigest(),
            "store": hashlib.sha256(b"fixture-store-baseline").hexdigest(),
        }
        self.source_hashes = {
            **self.clone_hashes,
            "acquisition_sequence": hashlib.sha256(
                b"fixture-acquisition-baseline"
            ).hexdigest(),
        }
        self._lineages: dict[tuple[str, str, str], tuple[str, str, int]] = {}
        self._ordinal = 0

    @staticmethod
    def genesis_digest(purpose: str, replicate: str, arm: str) -> str:
        del replicate, arm
        return _digest(f"{purpose}:learner-genesis")

    def __call__(
        self,
        spec: runner.ArmTaskSpec,
        parser_disposition: str = "ADMITTED",
    ) -> runner.ProbeIntegrityEvidence:
        self._ordinal += 1
        disposable = spec.phase in ("development", "final") and spec.arm != "QWEN_ONLY"
        clone_audit = None
        if disposable:
            pairs = tuple(sorted(self.clone_hashes.items()))
            clone_audit = runner.CloneAudit(
                schema=runner.CLONE_SCHEMA,
                source_hashes=pairs,
                destination_hashes=pairs,
                root=str(self.root / f"probe-{self._ordinal:04d}"),
            )

        genesis = parent = child = None
        sequence = None
        if spec.arm not in runner.CONTROL_ARMS:
            genesis = self.genesis_digest(
                spec.purpose,
                spec.replicate_commitment,
                spec.arm,
            )
            key = (spec.purpose, spec.replicate_commitment, spec.arm)
            if spec.phase == "adaptation":
                _, parent, previous_sequence = self._lineages.get(
                    key,
                    (
                        genesis,
                        genesis,
                        runner.QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE,
                    ),
                )
            else:
                parent = genesis
                previous_sequence = runner.QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
            should_transition = (
                spec.arm in runner.STATEFUL_ARMS
                and parser_disposition == "ADMITTED"
                and (
                    spec.purpose == "evaluation"
                    or spec.phase == "adaptation"
                    or spec.arm == "FULL"
                )
            )
            if should_transition:
                sequence = previous_sequence + 1
                child = _digest(
                    f"{spec.purpose}:{spec.replicate_commitment}:{spec.arm}:"
                    f"{spec.task_id}:learner-child:{sequence}"
                )
                if spec.phase == "adaptation":
                    self._lineages[key] = (genesis, child, sequence)

        return runner.ProbeIntegrityEvidence.create(
            purpose=spec.purpose,
            phase=spec.phase,
            task_id=spec.task_id,
            arm=spec.arm,
            replicate_commitment=spec.replicate_commitment,
            foundation_tensor_digest=self.foundation_tensor_digest,
            source_baseline_before=self.source_hashes if disposable else None,
            source_baseline_after=self.source_hashes if disposable else None,
            clone_before=self.source_hashes if disposable else None,
            clone_after=self.source_hashes if disposable else None,
            clone_audit=clone_audit,
            learner_genesis_digest=genesis,
            learner_parent_digest=parent,
            learner_child_digest=child,
            learner_sequence=sequence,
        )


class _FakeArmExecutor:
    def __init__(
        self,
        owned_evaluator: _FakeOrchestrationEvaluator,
        clone_parent: Path,
    ) -> None:
        self.evaluator = owned_evaluator
        self.clone_parent = clone_parent
        self.specs: list[runner.ArmTaskSpec] = []
        self.clone_paths: list[Path] = []
        self.generation_attempts = 0
        self.integrity_factory = _FakeProbeIntegrityFactory(
            clone_parent / "integrity",
            _digest("fixture-foundation-tensor"),
        )

    def __call__(self, spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
        ordinal = len(self.specs)
        self.specs.append(spec)
        attempt_receipt = _digest(f"fake-attempt-{ordinal}")
        clone = self.clone_parent / f"clone-{ordinal:04d}"
        clone.mkdir(parents=True)
        (clone / "probe-evidence.bin").write_bytes(b"isolated")
        (clone / "probe-evidence.bin").unlink()
        clone.rmdir()
        self.clone_paths.append(clone)
        self.generation_attempts += 2
        raw_response = "PUBLIC-MODEL-RESPONSE"
        judgment = self.evaluator.judge(spec, attempt_receipt, raw_response)
        integrity = self.integrity_factory(spec)
        return runner.ArmTaskResult(
            task_id=spec.task_id,
            arm=spec.arm,
            attempt_receipt_ref=attempt_receipt,
            raw_response=raw_response,
            parser_disposition="ADMITTED",
            judgment=judgment,
            evidence={
                **_fair_arm_evidence(spec.task_id, spec.arm),
                "probe_integrity_ref": runner.content_ref(
                    "probe-integrity",
                    integrity.to_canonical(),
                ),
            },
            integrity=integrity,
        )


class CanonicalAndIntentTests(unittest.TestCase):
    def test_canonical_json_bytes_are_stable_strict_and_unframed(self) -> None:
        value = {"z": 3, "a": ["é", True, None]}
        encoded = runner.canonical_json_bytes(value)
        self.assertEqual(encoded, b'{"a":["\xc3\xa9",true,null],"z":3}')
        self.assertFalse(encoded.endswith(b"\n"))
        self.assertEqual(json.loads(encoded), value)

        for invalid in (float("nan"), float("inf"), b"bytes", {1: "key"}):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises((TypeError, ValueError)):
                    runner.canonical_json_bytes({"value": invalid})

    def test_exactly_four_mutually_exclusive_modes_parse(self) -> None:
        expected = {
            "--evaluator-worker": "evaluator-worker",
            "--seal-evaluation-seeds": "seal-evaluation-seeds",
            "--qualification": "qualification",
            "--evaluation": "evaluation",
        }
        for argument, mode in expected.items():
            with self.subTest(argument=argument):
                intent = runner.RunIntent.parse((argument,))
                self.assertEqual(intent.mode, mode)

        for argv in (
            (),
            ("--unknown",),
            ("--qualification", "--evaluation"),
            ("--qualification", "extra"),
        ):
            with self.subTest(argv=argv):
                with self.assertRaises((TypeError, ValueError, runner.RunnerInvariantError)):
                    runner.RunIntent.parse(argv)

    def test_r2_run_identities_are_distinct_from_frozen_evaluator_identities(
        self,
    ) -> None:
        self.assertEqual(
            runner.EVALUATOR_QUALIFICATION_IDENTITY,
            evaluator.QUALIFICATION_IDENTITY,
        )
        self.assertEqual(
            runner.EVALUATOR_EVALUATION_IDENTITY,
            evaluator.EVALUATION_IDENTITY,
        )
        self.assertEqual(
            runner.QUALIFICATION_IDENTITY,
            "angler.high-level-multidomain.v1-r2-harness-qualification",
        )
        self.assertEqual(
            runner.EVALUATION_IDENTITY,
            "angler.high-level-multidomain.v1-r2-evaluation",
        )
        self.assertNotEqual(
            runner.QUALIFICATION_IDENTITY,
            runner.EVALUATOR_QUALIFICATION_IDENTITY,
        )
        self.assertNotEqual(
            runner.EVALUATION_IDENTITY,
            runner.EVALUATOR_EVALUATION_IDENTITY,
        )


class FrozenSourceTests(unittest.TestCase):
    def test_every_frozen_predecessor_hash_matches_the_active_leaf(self) -> None:
        self.assertEqual(len(_FROZEN_SOURCE_HASHES), 35)
        for relative, expected in _FROZEN_SOURCE_HASHES.items():
            with self.subTest(relative=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_r2_runner_is_an_exact_normalized_mechanical_delta(self) -> None:
        predecessor = (
            ROOT / "experiments/runners/high_level_multidomain_v1_r1.py"
        ).read_text(encoding="utf-8")
        recovery = runner.RUNNER_PATH.read_text(encoding="utf-8")

        def remove_block(pattern: str) -> None:
            nonlocal recovery
            recovery, substitutions = re.subn(
                pattern,
                "",
                recovery,
                count=1,
                flags=re.DOTALL,
            )
            self.assertEqual(substitutions, 1)

        def replace_once(current: str, predecessor_text: str) -> None:
            nonlocal recovery
            self.assertEqual(recovery.count(current), 1)
            recovery = recovery.replace(current, predecessor_text, 1)

        # Remove only the declared baseline binding and immutable-R1 witness
        # additions before normalizing the fresh R2 identities back to R1.
        remove_block(
            r"# The qualified persistent-lineage baseline contains "
            r"transaction-store genesis\n"
            r"# at sequence 0 followed by exactly one neutral unrelated "
            r"bootstrap commit\.\n"
            r"QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE = 1\n"
        )
        remove_block(
            r'    "experiments/runners/high_level_multidomain_v1_r1\.py": \(\n'
            r'.*?'
            r'    "docs/reports/'
            r'HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_R1_RESULT\.md": \(\n'
            r'        "114fd31e35e4bb0778eaeb7b25421c41b297b1b443a3dc95d5b5d6755954be7f"\n'
            r'    \),\n'
        )
        remove_block(
            r"IMMUTABLE_R1_EVIDENCE_SHA256 = MappingProxyType\(\n"
            r".*?(?=CONSUMED_V1_SEED_COMMITMENTS)"
        )
        remove_block(
            r"CONSUMED_R1_SEED_COMMITMENTS = frozenset\(\n.*?\n\)\n"
        )
        remove_block(
            r"    if set\(values\) & CONSUMED_R1_SEED_COMMITMENTS:\n"
            r'        raise RunnerInvariantError\("R2 seed commitments '
            r'overlap consumed R1"\)\n'
        )
        replace_once(
            "        bootstrap_commit = store.commit_episode(\n"
            "            _qualified_unrelated_bootstrap_episode(state_digest, manifest),\n"
            "            snapshot,\n"
            "            expected_parent_digest=state_digest,\n"
            "        )\n"
            "        journal = SQLiteQwenExecutionJournal(paths.journal)\n"
            "        bootstrap_head = store.audit_integrity()\n"
            "        if (\n"
            "            bootstrap_commit.sequence != QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE\n"
            "            or bootstrap_commit.head.sequence\n"
            "            != QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE\n"
            "            or bootstrap_commit.head.state_digest != state_digest\n"
            "            or bootstrap_head != bootstrap_commit.head\n"
            "        ):\n"
            "            raise RunnerInvariantError(\n"
            '                "qualified neutral bootstrap transaction baseline differs"\n'
            "            )\n",
            "        store.commit_episode(\n"
            "            _qualified_unrelated_bootstrap_episode(state_digest, manifest),\n"
            "            snapshot,\n"
            "            expected_parent_digest=state_digest,\n"
            "        )\n"
            "        journal = SQLiteQwenExecutionJournal(paths.journal)\n"
            "        store.audit_integrity()\n",
        )
        self.assertEqual(
            recovery.count(
                "previous_sequence = QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE"
            ),
            2,
        )
        recovery = recovery.replace(
            "previous_sequence = QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE",
            "previous_sequence = 0",
        )
        replace_once(
            "advanced = (\n"
            "            previous_sequence > QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE\n"
            "            and previous_child != genesis\n"
            "        )",
            "advanced = previous_sequence >= 1 and previous_child != genesis",
        )
        replace_once(
            "advanced = (\n"
            "            previous_sequence > QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE\n"
            '            and previous_child != row["genesis_digest"]\n'
            "        )",
            'advanced = previous_sequence >= 1 and previous_child != row["genesis_digest"]',
        )
        replace_once(
            '"ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-"\n'
            '    "LINEAGE-SEQUENCE-RECOVERY-001.md"',
            '"ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R1-"\n'
            '    "LIVE-EVALUATION-COMPLETION-001.md"',
        )
        replace_once(
            "a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455",
            "a54aa78d48fe5435b035e3c057470c915b87c8f4a4de9d001997a1dcd57d39de",
        )
        recovery = recovery.replace("r2", "r1").replace("R2", "R1")
        replace_once(
            "len(frozen) != 35 or len(set(frozen)) != 35",
            "len(frozen) != 29 or len(set(frozen)) != 29",
        )
        replace_once(
            "len(combined) != 38 or len(set(combined)) != 38",
            "len(combined) != 32 or len(set(combined)) != 32",
        )
        self.assertNotIn("QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE", recovery)
        self.assertEqual(recovery, predecessor)

    def test_r2_paths_are_exact_and_disjoint_from_every_predecessor_identity(
        self,
    ) -> None:
        self.assertEqual(
            runner.RUNNER_PATH,
            ROOT / "experiments/runners/high_level_multidomain_v1_r2.py",
        )
        self.assertEqual(
            runner.MANIFEST_PATH,
            ROOT / "experiments/manifests/high-level-multidomain-v1-r2.json",
        )
        self.assertEqual(
            runner.STATE_ROOT,
            Path("/opt/angler/state/project-angler/high-level-multidomain-v1-r2"),
        )
        self.assertEqual(
            runner.SCRATCH_ROOT,
            Path("/opt/angler/scratch/high-level-multidomain-v1-r2"),
        )
        self.assertEqual(runner.SEALED_SEED_PATH.name, "seeds-v1-r2.json")
        self.assertEqual(
            runner.QUALIFICATION_RELEASE_CLAIM_PATH.name,
            "qualification-release-claim-v1-r2.json",
        )
        self.assertEqual(
            runner.EVALUATION_ADMISSION_CLAIM_PATH.name,
            "evaluation-admission-claim-v1-r2.json",
        )
        self.assertEqual(runner.QUALIFICATION_RUNTIME_ROOT.name, "qualification-v1-r2")
        self.assertEqual(runner.EVALUATION_RUNTIME_ROOT.name, "evaluation-v1-r2")
        self.assertEqual(
            runner.EVALUATION_COGNEE_SCOPES_ROOT.name,
            "evaluation-cognee-scopes",
        )
        self.assertEqual(runner.EVALUATION_GENESIS_ROOT.parent, runner.EVALUATION_RUNTIME_ROOT)
        self.assertEqual(
            runner.EVALUATION_ARM_FACTORY_ROOT.parent,
            runner.EVALUATION_RUNTIME_ROOT,
        )
        self.assertEqual(
            runner.EVALUATION_EVIDENCE_LEDGER_PATH.parent,
            runner.EVALUATION_RUNTIME_ROOT,
        )
        self.assertTrue(
            {
                runner.EVALUATION_RUNTIME_ROOT,
                runner.EVALUATION_COGNEE_SCOPES_ROOT,
            }.isdisjoint(
                {
                    runner.QUALIFICATION_RUNTIME_ROOT,
                    runner.QUALIFICATION_COGNEE_SCOPES_ROOT,
                }
            )
        )

        predecessor_paths = (
            ROOT / "experiments/runners/high_level_multidomain_v1.py",
            ROOT / "tests/unit/experiments/test_high_level_multidomain_runner_v1.py",
            ROOT / "tests/integration/runtime/test_high_level_multidomain_qwen_v1.py",
            ROOT / "experiments/manifests/high-level-multidomain-v1.json",
            ROOT
            / "docs/blueprints/branches/science/work/"
            "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-001.md",
            Path("/opt/angler/state/project-angler/high-level-multidomain-v1"),
            Path("/opt/angler/scratch/high-level-multidomain-v1"),
            Path("/opt/angler/results/high-level-multidomain-v1-qualification.json"),
            Path("/opt/angler/results/high-level-multidomain-v1.json"),
            ROOT / "experiments/runners/high_level_multidomain_v1_r1.py",
            ROOT
            / "tests/unit/experiments/test_high_level_multidomain_runner_v1_r1.py",
            ROOT
            / "tests/integration/runtime/test_high_level_multidomain_qwen_v1_r1.py",
            ROOT / "experiments/manifests/high-level-multidomain-v1-r1.json",
            ROOT
            / "docs/blueprints/branches/science/work/"
            "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R1-"
            "LIVE-EVALUATION-COMPLETION-001.md",
            _R1_STATE_ROOT,
            Path("/opt/angler/scratch/high-level-multidomain-v1-r1"),
            Path(
                "/opt/angler/results/high-level-multidomain-v1-r1-qualification.json"
            ),
            Path("/opt/angler/results/high-level-multidomain-v1-r1.json"),
        )
        recovery_mutable_paths = (
            runner.RUNNER_PATH,
            ROOT / "tests/unit/experiments/test_high_level_multidomain_runner_v1_r2.py",
            ROOT / "tests/integration/runtime/test_high_level_multidomain_qwen_v1_r2.py",
            runner.MANIFEST_PATH,
            runner.ACTIVE_LEAF_PATH,
            runner.STATE_ROOT,
            runner.SEALED_SEED_PATH,
            runner.QUALIFICATION_RELEASE_CLAIM_PATH,
            runner.EVALUATION_ADMISSION_CLAIM_PATH,
            runner.QUALIFICATION_RUNTIME_ROOT,
            runner.QUALIFICATION_GENESIS_ROOT,
            runner.QUALIFICATION_ARM_FACTORY_ROOT,
            runner.QUALIFICATION_EVIDENCE_LEDGER_PATH,
            runner.QUALIFICATION_COGNEE_SCOPES_ROOT,
            runner.EVALUATION_RUNTIME_ROOT,
            runner.EVALUATION_GENESIS_ROOT,
            runner.EVALUATION_ARM_FACTORY_ROOT,
            runner.EVALUATION_EVIDENCE_LEDGER_PATH,
            runner.EVALUATION_COGNEE_SCOPES_ROOT,
            runner.SCRATCH_ROOT,
            runner.SCRATCH_ROOT / "tmp",
            runner.SCRATCH_ROOT / "evaluator-cwd",
            runner.SCRATCH_ROOT / "evaluator-tmp",
            runner.QUALIFICATION_RESULT_PATH,
            runner.EVALUATION_RESULT_PATH,
        )
        for recovery_path in recovery_mutable_paths:
            for predecessor_path in predecessor_paths:
                with self.subTest(
                    recovery=recovery_path,
                    predecessor=predecessor_path,
                ):
                    self.assertNotEqual(recovery_path, predecessor_path)
                    self.assertNotIn(recovery_path, predecessor_path.parents)
                    self.assertNotIn(predecessor_path, recovery_path.parents)
        self.assertEqual(runner.REPOSITORY_ROOT, ROOT)
        self.assertEqual(runner.MODEL_ROOT, Path("/opt/angler/models/Qwen3-4B"))
        self.assertEqual(
            runner.FASTEMBED_CACHE_ROOT,
            Path("/opt/angler/models/fastembed-cache-v1"),
        )
        self.assertEqual(
            runner.ANGLER_PYTHON,
            Path("/opt/angler/venvs/angler/bin/python"),
        )

    def test_all_immutable_v1_r1_witnesses_and_source_inventories_are_exact(
        self,
    ) -> None:
        self.assertEqual(
            runner.EXPECTED_ACTIVE_LEAF_SHA256,
            "a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455",
        )
        self.assertEqual(
            hashlib.sha256(runner.ACTIVE_LEAF_PATH.read_bytes()).hexdigest(),
            runner.EXPECTED_ACTIVE_LEAF_SHA256,
        )
        for label, path in _IMMUTABLE_V1_FILE_PATHS.items():
            with self.subTest(label=label):
                metadata = path.lstat()
                self.assertTrue(stat.S_ISREG(metadata.st_mode))
                self.assertEqual(metadata.st_nlink, 1)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    _IMMUTABLE_V1_EVIDENCE_SHA256[label],
                )
        self.assertEqual(
            _local_tree_digest(
                Path("/opt/angler/state/project-angler/high-level-multidomain-v1")
            ),
            (
                _IMMUTABLE_V1_EVIDENCE_SHA256["retained_state_tree"],
                28,
                24,
                6_384_237,
                52,
            ),
        )
        self.assertEqual(
            _local_tree_digest(
                Path("/opt/angler/scratch/high-level-multidomain-v1")
            ),
            (
                _IMMUTABLE_V1_EVIDENCE_SHA256["retained_scratch_tree"],
                57,
                13,
                728_211,
                70,
            ),
        )
        v1_manifest = _read_json(
            ROOT / "experiments/manifests/high-level-multidomain-v1.json"
        )
        self.assertEqual(
            frozenset(v1_manifest["seed_commitments"]),
            runner.CONSUMED_V1_SEED_COMMITMENTS,
        )
        self.assertNotIn(
            runner.qualification_replicate_commitment(),
            runner.CONSUMED_V1_SEED_COMMITMENTS,
        )
        for label, path in _IMMUTABLE_R1_FILE_PATHS.items():
            with self.subTest(predecessor="R1", label=label):
                metadata = path.lstat()
                self.assertTrue(stat.S_ISREG(metadata.st_mode))
                self.assertEqual(metadata.st_nlink, 1)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    _IMMUTABLE_R1_EVIDENCE_SHA256[label],
                )
        self.assertEqual(
            _local_tree_digest(_R1_STATE_ROOT),
            (
                _IMMUTABLE_R1_EVIDENCE_SHA256["retained_state_tree"],
                53,
                36,
                11_487_875,
                89,
            ),
        )
        self.assertEqual(
            _local_tree_digest(
                Path("/opt/angler/scratch/high-level-multidomain-v1-r1")
            ),
            (
                _IMMUTABLE_R1_EVIDENCE_SHA256["retained_scratch_tree"],
                65,
                14,
                812_194,
                79,
            ),
        )
        for path in _R1_EVALUATION_PATHS:
            with self.subTest(predecessor="R1", absent=path):
                self.assertFalse(path.exists())
                self.assertFalse(path.is_symlink())
        r1_manifest = _read_json(
            ROOT / "experiments/manifests/high-level-multidomain-v1-r1.json"
        )
        self.assertEqual(
            frozenset(r1_manifest["seed_commitments"]),
            runner.CONSUMED_R1_SEED_COMMITMENTS,
        )
        self.assertTrue(
            runner.CONSUMED_V1_SEED_COMMITMENTS.isdisjoint(
                runner.CONSUMED_R1_SEED_COMMITMENTS
            )
        )
        self.assertNotIn(
            runner.qualification_replicate_commitment(),
            runner.CONSUMED_R1_SEED_COMMITMENTS,
        )
        r1_result = _read_json(
            Path("/opt/angler/results/high-level-multidomain-v1-r1-qualification.json")
        )
        self.assertEqual(r1_result["classification"], "QUALIFICATION_FAILURE")
        self.assertFalse(r1_result["scientific_claim"])
        self.assertEqual(runner.FROZEN_SOURCE_SHA256, _FROZEN_SOURCE_HASHES)
        self.assertEqual(len(runner.FROZEN_SOURCE_SHA256), 35)
        self.assertEqual(
            runner.REQUIRED_CONSTRUCTION_SOURCES,
            (
                "experiments/runners/high_level_multidomain_v1_r2.py",
                "tests/unit/experiments/test_high_level_multidomain_runner_v1_r2.py",
                "tests/integration/runtime/test_high_level_multidomain_qwen_v1_r2.py",
            ),
        )
        self.assertEqual(len(set(runner.REQUIRED_CONSTRUCTION_SOURCES)), 3)
        self.assertFalse(
            set(runner.FROZEN_SOURCE_SHA256).intersection(
                runner.REQUIRED_CONSTRUCTION_SOURCES
            )
        )
        self.assertEqual(
            len(set(runner.FROZEN_SOURCE_SHA256) | set(runner.REQUIRED_CONSTRUCTION_SOURCES)),
            38,
        )

    def test_runner_contains_no_task_solver_or_public_final_seed_literal(self) -> None:
        source = runner.RUNNER_PATH.read_text(encoding="utf-8")
        forbidden = (
            "score_demonstration_procedure_answer",
            "judge_glyph_procedure_attempt",
            "evaluate_committed_sequence",
            "make_heldout_operator_suite",
            "2026083191",
            "2026083192",
        )
        for sentinel in forbidden:
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, source)


class EvaluatorDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.commitment = _digest("qualification-replicate")
        self.spec = runner.EvaluatorWorkerSpec(
            purpose="qualification",
            expected_commitments=(self.commitment,),
        )
        self.evaluator = _FakeEvaluator(self.commitment)
        state_type = getattr(
            runner,
            "EvaluatorProtocolState",
            runner.EvaluatorDispatchState,
        )
        self.state = state_type()

    def _dispatch(self, op: str, payload: dict[str, object]) -> dict[str, object]:
        request_id = self.state.next_request_id
        response = runner.dispatch_evaluator_request(
            self.evaluator,
            {"id": request_id, "op": op, "payload": payload, "v": 1},
            self.state,
        )
        self.assertEqual(set(response), {"id", "ok", "result", "v"})
        self.assertEqual(response["id"], request_id)
        self.assertIs(response["ok"], True)
        self.assertEqual(response["v"], 1)
        runner.scan_for_hidden_material(response)
        return response

    def test_all_eight_operations_have_exact_canonical_result_shapes(self) -> None:
        commitments = self._dispatch(
            "commitments",
            {"spec": self.spec.to_canonical()},
        )["result"]
        self.assertEqual(
            commitments,
            {
                "commitments": {"replicate_commitments": [self.commitment]},
                "digest": runner.evaluator_record_digest(
                    "evaluator-commitments",
                    {"replicate_commitments": [self.commitment]},
                ),
            },
        )
        self.assertEqual(
            self._dispatch("commitments", {})["result"],
            commitments,
        )
        self.assertEqual(
            self._dispatch("release_phase", {"phase": "adaptation"})["result"],
            {
                "tasks": [
                    {"phase": "adaptation", "task_id": _digest("ipc-task")}
                ]
            },
        )
        self.assertEqual(
            self._dispatch("complete_phase", {"phase": "adaptation"})["result"],
            {},
        )
        admission = _digest("admission")
        self.assertEqual(
            self._dispatch(
                "admit_final",
                {"admission_digest": admission},
            )["result"],
            {},
        )
        task_id = _digest("ipc-task")
        attempt_ref = _digest("ipc-attempt")
        judgment = self._dispatch(
            "judge_response",
            {
                "arm": "FULL",
                "attempt_receipt_ref": attempt_ref,
                "raw_response": "MALFORMED",
                "task_id": task_id,
            },
        )["result"]
        self.assertEqual(
            set(judgment),
            {
                "arm",
                "attempt_receipt_ref",
                "disposition",
                "raw_response",
                "response_commitment",
                "score",
                "task_id",
            },
        )
        random_feedback = self._dispatch(
            "random_feedback",
            {"replicate_commitment": self.commitment},
        )["result"]
        self.assertEqual(
            random_feedback,
            {
                "assignments": [{"task_id": task_id, "value": 0.0}],
                "replicate_commitment": self.commitment,
            },
        )
        metrics = self._dispatch("final_metrics", {})["result"]
        self.assertEqual(metrics["identity"], "fake")
        self.assertEqual(
            self._dispatch("close", {})["result"],
            {},
        )
        self.assertTrue(self.state.closed)

    def test_malformed_duplicate_out_of_order_and_post_close_requests_invalidate(self) -> None:
        state_type = type(self.state)
        cases = (
            {"id": 2, "op": "commitments", "payload": {}, "v": 1},
            {
                "extra": True,
                "id": 1,
                "op": "commitments",
                "payload": {},
                "v": 1,
            },
            {"id": 1, "op": "unknown", "payload": {}, "v": 1},
            {"id": 1, "op": "commitments", "payload": [], "v": 1},
        )
        for request in cases:
            with self.subTest(request=request):
                state = state_type()
                with self.assertRaises(runner.EvaluatorProtocolError):
                    runner.dispatch_evaluator_request(
                        self.evaluator,
                        request,
                        state,
                    )
                self.assertFalse(getattr(state, "valid", False))

        self._dispatch("commitments", {"spec": self.spec.to_canonical()})
        self._dispatch("close", {})
        with self.assertRaises(runner.EvaluatorProtocolError):
            runner.dispatch_evaluator_request(
                self.evaluator,
                {
                    "id": self.state.next_request_id,
                    "op": "commitments",
                    "payload": {},
                    "v": 1,
                },
                self.state,
            )
        self.assertFalse(getattr(self.state, "valid", False))

    def test_worker_failure_is_one_generic_secret_free_bounded_error(self) -> None:
        hidden = "PRIVATE-SEED-AND-ROUTE-SENTINEL"
        request = {
            "id": 1,
            "op": "commitments",
            "payload": {"spec": {"hidden_answer": hidden}},
            "v": 1,
        }
        source = io.BytesIO(runner.canonical_json_bytes(request) + b"\n")
        sink = io.BytesIO()
        with self.assertRaises(SystemExit) as stopped:
            runner.evaluator_worker_main(source, sink)
        self.assertEqual(stopped.exception.code, 2)
        raw = sink.getvalue()
        self.assertLessEqual(len(raw), runner.MAXIMUM_IPC_BYTES)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(hidden.encode("ascii"), raw)
        self.assertNotIn(b"Traceback", raw)
        response = runner.decode_canonical_json(raw[:-1])
        self.assertEqual(
            response,
            {
                "error": {
                    "code": "OPERATION_FAILED",
                    "message": "EVALUATOR_OPERATION_FAILED",
                },
                "id": 1,
                "ok": False,
                "v": 1,
            },
        )

    def test_commitment_only_evaluation_has_no_circular_full_binding_or_phase_access(
        self,
    ) -> None:
        commitments = (_digest("premanifest-one"), _digest("premanifest-two"))
        spec = runner.EvaluatorWorkerSpec(
            purpose="evaluation",
            expected_commitments=commitments,
            expected_final_admission=runner.COMMITMENT_ONLY_ADMISSION_DIGEST,
            seed_path=str(runner.SEALED_SEED_PATH),
            commitment_only=True,
        )
        self.assertIsNone(spec.expected_evaluator_commitments)
        self.assertIsNone(spec.expected_evaluator_commitment_digest)
        fake = _CommitmentOnlyEvaluator(commitments)
        state = runner.EvaluatorProtocolState()
        response = runner.dispatch_evaluator_request(
            fake,
            {
                "id": 1,
                "op": "commitments",
                "payload": {"spec": spec.to_canonical()},
                "v": 1,
            },
            state,
        )
        self.assertEqual(
            response["result"],
            {
                "commitments": fake.commitments.to_canonical(),
                "digest": fake.commitments.digest,
            },
        )
        self.assertEqual(
            runner.dispatch_evaluator_request(
                fake,
                {"id": 2, "op": "close", "payload": {}, "v": 1},
                state,
            )["result"],
            {},
        )
        self.assertTrue(state.closed)

        forbidden_state = runner.EvaluatorProtocolState()
        runner.dispatch_evaluator_request(
            fake,
            {
                "id": 1,
                "op": "commitments",
                "payload": {"spec": spec.to_canonical()},
                "v": 1,
            },
            forbidden_state,
        )
        with self.assertRaises(runner.EvaluatorProtocolError):
            runner.dispatch_evaluator_request(
                fake,
                {
                    "id": 2,
                    "op": "release_phase",
                    "payload": {"phase": "adaptation"},
                    "v": 1,
                },
                forbidden_state,
            )
        self.assertFalse(forbidden_state.valid)

        with self.assertRaises(ValueError):
            runner.EvaluatorWorkerSpec(
                purpose="evaluation",
                expected_commitments=commitments,
                expected_final_admission=_digest("not-the-placeholder"),
                seed_path=str(runner.SEALED_SEED_PATH),
                commitment_only=True,
            )


class EvaluatorClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.commitment = _digest("client-replicate")
        self.spec = runner.EvaluatorWorkerSpec(
            purpose="qualification",
            expected_commitments=(self.commitment,),
        )

    @staticmethod
    def _line(value: object) -> bytes:
        return runner.canonical_json_bytes(value) + b"\n"

    def _handshake(self, request_id: int = 1) -> dict[str, object]:
        commitments = {
            "identity": runner.EVALUATOR_QUALIFICATION_IDENTITY,
            "purpose": "qualification",
            "qualification_seed": runner.QUALIFICATION_SEED,
            "random_feedback": [],
            "replicate_commitments": [self.commitment],
            "schema": runner.SUITE_SCHEMA,
            "tasks": [],
        }
        return {
            "id": request_id,
            "ok": True,
            "result": {
                "commitments": commitments,
                "digest": runner.evaluator_record_digest(
                    "evaluator-commitments",
                    commitments,
                ),
            },
            "v": 1,
        }

    def test_client_frames_monotonic_requests_and_closes_exactly(self) -> None:
        task = {"family": "glyph-machine", "task_id": _digest("client-task")}
        process = _ScriptedProcess(
            [
                self._line(self._handshake()),
                self._line(
                    {
                        "id": 2,
                        "ok": True,
                        "result": {"tasks": [task]},
                        "v": 1,
                    }
                ),
                self._line(
                    {
                        "id": 3,
                        "ok": True,
                        "result": {},
                        "v": 1,
                    }
                ),
            ]
        )
        client = runner.EvaluatorClient.start(
            self.spec,
            process_factory=lambda _: process,
        )
        self.assertTrue(client.usable)
        self.assertEqual(client.release_phase("adaptation"), (task,))
        client.close()
        self.assertFalse(client.usable)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.waited)

        requests = tuple(
            runner.decode_canonical_json(chunk[:-1])
            for chunk in process.stdin.chunks
        )
        self.assertEqual([item["id"] for item in requests], [1, 2, 3])
        self.assertEqual(
            [item["op"] for item in requests],
            ["commitments", "release_phase", "close"],
        )
        self.assertTrue(all(set(item) == {"id", "op", "payload", "v"} for item in requests))

    def test_handshake_identity_is_purpose_derived_and_revalidated(self) -> None:
        wrong = self._handshake()
        wrong_commitments = dict(wrong["result"]["commitments"])
        wrong_commitments["identity"] = runner.QUALIFICATION_IDENTITY
        wrong["result"] = {
            "commitments": wrong_commitments,
            "digest": runner.evaluator_record_digest(
                "evaluator-commitments",
                wrong_commitments,
            ),
        }
        rejected = _ScriptedProcess([self._line(wrong)])
        with self.assertRaises(runner.EvaluatorProtocolError):
            runner.EvaluatorClient.start(
                self.spec,
                process_factory=lambda _: rejected,
            )
        self.assertTrue(rejected.waited)

        commitments = (_digest("commitment-only-one"), _digest("commitment-only-two"))
        evaluation_spec = runner.EvaluatorWorkerSpec(
            purpose="evaluation",
            expected_commitments=commitments,
            expected_final_admission=runner.COMMITMENT_ONLY_ADMISSION_DIGEST,
            seed_path=str(runner.SEALED_SEED_PATH),
            commitment_only=True,
        )
        evaluation_payload = {
            "identity": runner.EVALUATOR_EVALUATION_IDENTITY,
            "purpose": "evaluation",
            "qualification_seed": None,
            "random_feedback": [],
            "replicate_commitments": list(commitments),
            "schema": runner.SUITE_SCHEMA,
            "tasks": [],
        }
        accepted = _ScriptedProcess(
            [
                self._line(
                    {
                        "id": 1,
                        "ok": True,
                        "result": {
                            "commitments": evaluation_payload,
                            "digest": runner.evaluator_record_digest(
                                "evaluator-commitments",
                                evaluation_payload,
                            ),
                        },
                        "v": 1,
                    }
                )
            ]
        )
        client = runner.EvaluatorClient.start(
            evaluation_spec,
            process_factory=lambda _: accepted,
        )
        self.assertEqual(
            client.commitment_snapshot["commitments"]["identity"],
            runner.EVALUATOR_EVALUATION_IDENTITY,
        )
        client.invalidate()

        extra = self._handshake()
        extra_payload = dict(extra["result"]["commitments"])
        extra_payload["extra"] = True
        extra["result"] = {
            "commitments": extra_payload,
            "digest": runner.evaluator_record_digest(
                "evaluator-commitments",
                extra_payload,
            ),
        }
        extra_process = _ScriptedProcess([self._line(extra)])
        with self.assertRaises(runner.EvaluatorProtocolError):
            runner.EvaluatorClient.start(
                self.spec,
                process_factory=lambda _: extra_process,
            )
        self.assertTrue(extra_process.waited)

        valid = self._handshake()
        repeated_payload = dict(valid["result"]["commitments"])
        repeated_payload["identity"] = runner.QUALIFICATION_IDENTITY
        repeated = _ScriptedProcess(
            [
                self._line(valid),
                self._line(
                    {
                        "id": 2,
                        "ok": True,
                        "result": {
                            "commitments": repeated_payload,
                            "digest": runner.evaluator_record_digest(
                                "evaluator-commitments",
                                repeated_payload,
                            ),
                        },
                        "v": 1,
                    }
                ),
            ]
        )
        client = runner.EvaluatorClient.start(
            self.spec,
            process_factory=lambda _: repeated,
        )
        with self.assertRaises(runner.EvaluatorProtocolError):
            client.commitments()
        self.assertTrue(repeated.waited)

    def test_every_malformed_transport_or_response_invalidates_the_client(self) -> None:
        hidden = "PRIVATE-HIDDEN-IPC-ERROR"
        cases = {
            "eof": b"",
            "missing-newline": self._line(self._handshake())[:-1],
            "malformed-json": b"not-json\n",
            "wrong-id": self._line(self._handshake(7)),
            "extra-field": self._line({**self._handshake(), "extra": True}),
            "secret-error": self._line(
                {
                    "error": {"code": "OPERATION_FAILED", "message": hidden},
                    "id": 1,
                    "ok": False,
                    "v": 1,
                }
            ),
            "oversized": b"x" * (runner.MAXIMUM_IPC_BYTES + 1),
        }
        for label, row in cases.items():
            with self.subTest(label=label):
                process = _ScriptedProcess([row])
                with self.assertRaises((RuntimeError, ValueError)) as caught:
                    runner.EvaluatorClient.start(
                        self.spec,
                        process_factory=lambda _, process=process: process,
                    )
                self.assertTrue(process.terminated)
                self.assertTrue(process.waited)
                self.assertNotIn(hidden, str(caught.exception))

    def test_oversized_outbound_submission_invalidates_before_a_second_write(self) -> None:
        process = _ScriptedProcess([self._line(self._handshake())])
        client = runner.EvaluatorClient.start(
            self.spec,
            process_factory=lambda _: process,
        )
        with self.assertRaises(runner.EvaluatorProtocolError):
            client.judge_response(
                _digest("oversized-task"),
                "FULL",
                _digest("oversized-attempt"),
                "x" * runner.MAXIMUM_IPC_BYTES,
            )
        self.assertFalse(client.usable)
        self.assertTrue(process.terminated)
        self.assertEqual(len(process.stdin.chunks), 1)

    def test_invalid_timeout_is_rejected_before_process_creation(self) -> None:
        calls: list[runner.EvaluatorWorkerSpec] = []

        def factory(spec: runner.EvaluatorWorkerSpec) -> _ScriptedProcess:
            calls.append(spec)
            return _ScriptedProcess([])

        with self.assertRaises(ValueError):
            runner.EvaluatorClient.start(
                self.spec,
                process_factory=factory,
                timeout_seconds=0.0,
            )
        self.assertEqual(calls, [])

    def test_unresponsive_process_is_killed_and_reaped_after_terminate(self) -> None:
        class StubbornProcess(_ScriptedProcess):
            def __init__(self) -> None:
                super().__init__([b"malformed-handshake\n"])
                self.wait_calls = 0

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                self.wait_calls += 1
                if not self.killed:
                    raise TimeoutError("worker ignored terminate")
                self.waited = True
                return -9

        process = StubbornProcess()
        with self.assertRaises((RuntimeError, ValueError)):
            runner.EvaluatorClient.start(
                self.spec,
                process_factory=lambda _: process,
            )
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertTrue(process.waited)
        self.assertGreaterEqual(process.wait_calls, 2)

    def test_failed_second_wait_never_certifies_reaping_and_remains_retryable(
        self,
    ) -> None:
        class NeverReapedProcess(_ScriptedProcess):
            def __init__(self) -> None:
                super().__init__([self_line])
                self.wait_calls = 0

            def poll(self) -> int | None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                self.wait_calls += 1
                raise TimeoutError("worker ignored terminate and kill")

        self_line = self._line(self._handshake())
        process = NeverReapedProcess()
        client = runner.EvaluatorClient.start(
            self.spec,
            process_factory=lambda _: process,
        )
        for expected_waits in (2, 4):
            with self.assertRaises(runner.EvaluatorTransportError):
                client.invalidate()
            self.assertEqual(process.wait_calls, expected_waits)
            self.assertTrue(process.terminated)
            self.assertTrue(process.killed)
            self.assertFalse(client.usable)

    def test_close_wait_failures_remain_uncertified_until_external_retry_reaps(
        self,
    ) -> None:
        close_ack = self._line(
            {"id": 2, "ok": True, "result": {}, "v": 1}
        )

        class RetryableCloseProcess(_ScriptedProcess):
            def __init__(self) -> None:
                super().__init__([handshake, close_ack])
                self.wait_calls = 0
                self.kill_calls = 0

            def kill(self) -> None:
                self.kill_calls += 1
                super().kill()

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                self.wait_calls += 1
                if self.wait_calls <= 4:
                    raise TimeoutError("worker remained unreaped")
                self.waited = True
                return -9

        handshake = self._line(self._handshake())
        process = RetryableCloseProcess()
        client = runner.EvaluatorClient.start(
            self.spec,
            process_factory=lambda _: process,
        )
        with self.assertRaises(BaseExceptionGroup) as captured:
            client.close()
        leaves = runner._exception_leaves(captured.exception)
        self.assertTrue(
            any(isinstance(item, runner.EvaluatorTransportError) for item in leaves)
        )
        self.assertTrue(
            any(isinstance(item, runner.CleanupFailure) for item in leaves)
        )
        self.assertFalse(client._reaped)
        self.assertFalse(client.usable)
        self.assertEqual(process.wait_calls, 4)
        self.assertEqual(process.kill_calls, 2)

        client.invalidate()
        self.assertTrue(client._reaped)
        self.assertTrue(process.waited)
        self.assertEqual(process.wait_calls, 5)

    def test_real_pipe_read_honors_the_bounded_response_timeout(self) -> None:
        reader_descriptor, writer_descriptor = os.pipe()
        reader = os.fdopen(reader_descriptor, "rb", buffering=0)
        try:
            with self.assertRaises(TimeoutError):
                runner._readline_with_timeout(
                    reader,
                    runner.MAXIMUM_IPC_BYTES,
                    0.1,
                )
        finally:
            reader.close()
            os.close(writer_descriptor)

    def test_semantically_invalid_judgment_response_invalidates_ipc_client(self) -> None:
        task_id = _digest("client-judgment-task")
        receipt = _digest("client-judgment-attempt")
        raw_response = "PUBLIC-RESPONSE"
        response_payload = {
            "arm": "FULL",
            "attempt_receipt_ref": receipt,
            "raw_response": raw_response,
            "task_id": task_id,
        }
        valid = {
            **response_payload,
            "disposition": "UNSUCCESSFUL",
            "response_commitment": runner.evaluator_record_digest(
                "response",
                response_payload,
            ),
            "score": 0.0,
        }
        cases = (
            {**valid, "response_commitment": _digest("bad-client-response")},
            {**valid, "score": 0.5},
            {**valid, "disposition": "SUCCESS"},
        )
        for judgment in cases:
            with self.subTest(judgment=judgment):
                process = _ScriptedProcess(
                    [
                        self._line(self._handshake()),
                        self._line(
                            {
                                "id": 2,
                                "ok": True,
                                "result": judgment,
                                "v": 1,
                            }
                        ),
                    ]
                )
                client = runner.EvaluatorClient.start(
                    self.spec,
                    process_factory=lambda _, process=process: process,
                )
                with self.assertRaises(ValueError):
                    client.judge_response(
                        task_id,
                        "FULL",
                        receipt,
                        raw_response,
                    )
                self.assertFalse(client.usable)
                self.assertTrue(process.terminated)
                self.assertTrue(process.waited)

    def test_live_handshake_rejects_same_replicates_with_different_full_binding(
        self,
    ) -> None:
        seeds = (bytes(range(32)), bytes(range(32, 64)))
        commitments = tuple(
            evaluator.commit_replicate_seed(seed) for seed in seeds
        )
        admission = _digest("manifest-bound-live-admission")
        owned = evaluator.make_evaluation_evaluator(
            seeds,
            commitments,
            admission,
        )
        expected = owned.commitments.to_canonical()
        spec = runner.EvaluatorWorkerSpec(
            purpose="evaluation",
            expected_commitments=commitments,
            expected_final_admission=admission,
            seed_path=str(runner.SEALED_SEED_PATH),
            expected_evaluator_commitments_json=(
                runner.canonical_json_bytes(expected).decode("utf-8")
            ),
            expected_evaluator_commitment_digest=owned.commitments.digest,
        )
        for label in ("task", "schedule"):
            with self.subTest(label=label):
                altered = json.loads(
                    runner.canonical_json_bytes(expected).decode("utf-8")
                )
                if label == "task":
                    altered["tasks"][0]["private_commitment"] = _digest(
                        "altered-private-task-binding"
                    )
                else:
                    altered["random_feedback"][0]["schedule_commitment"] = _digest(
                        "altered-random-feedback-binding"
                    )
                self.assertEqual(
                    altered["replicate_commitments"],
                    list(commitments),
                )
                altered_digest = runner.evaluator_record_digest(
                    "evaluator-commitments",
                    altered,
                )
                process = _ScriptedProcess(
                    [
                        self._line(
                            {
                                "id": 1,
                                "ok": True,
                                "result": {
                                    "commitments": altered,
                                    "digest": altered_digest,
                                },
                                "v": 1,
                            }
                        )
                    ]
                )
                with self.assertRaises(runner.EvaluatorProtocolError):
                    runner.EvaluatorClient.start(
                        spec,
                        process_factory=lambda _, process=process: process,
                    )
                self.assertTrue(process.terminated)
                self.assertTrue(process.waited)


class EvaluatorProcessBoundaryTests(unittest.TestCase):
    def test_clean_parent_environment_imports_runner_from_an_unrelated_cwd(self) -> None:
        expected_pythonpath = f"{runner.REPOSITORY_ROOT}:{runner.REPOSITORY_ROOT / 'src'}"
        self.assertEqual(
            runner.FROZEN_PARENT_ENVIRONMENT["PYTHONPATH"],
            expected_pythonpath,
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed = runner.subprocess.run(
                (
                    str(runner.ANGLER_PYTHON),
                    "-c",
                    (
                        "from experiments.runners import "
                        "high_level_multidomain_v1_r2; print('IMPORT_OK')"
                    ),
                ),
                cwd=temporary,
                env=dict(runner.FROZEN_PARENT_ENVIRONMENT),
                stdin=runner.subprocess.DEVNULL,
                stdout=runner.subprocess.PIPE,
                stderr=runner.subprocess.PIPE,
                timeout=10.0,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", errors="replace"),
        )
        self.assertEqual(completed.stdout, b"IMPORT_OK\n")

    def test_default_worker_argv_environment_and_private_directories_are_secret_free(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            commitments = (_digest("seed-one"), _digest("seed-two"))
            admission = _digest("final-admission")
            spec = runner.EvaluatorWorkerSpec(
                purpose="evaluation",
                expected_commitments=commitments,
                expected_final_admission=runner.COMMITMENT_ONLY_ADMISSION_DIGEST,
                seed_path=str(runner.SEALED_SEED_PATH),
                commitment_only=True,
            )
            captured: dict[str, object] = {}
            process = _ScriptedProcess([])

            def popen(argv: object, **kwargs: object) -> _ScriptedProcess:
                captured["argv"] = argv
                captured.update(kwargs)
                return process

            with mock.patch.object(runner, "SCRATCH_ROOT", scratch), mock.patch.object(
                runner.subprocess,
                "Popen",
                side_effect=popen,
            ):
                managed = runner._default_evaluator_process(spec)

            argv = tuple(captured["argv"])  # type: ignore[arg-type]
            environment = captured["env"]
            self.assertEqual(
                argv,
                (
                    str(runner.ANGLER_PYTHON),
                    str(runner.RUNNER_PATH),
                    "--evaluator-worker",
                ),
            )
            self.assertIs(type(environment), dict)
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")  # type: ignore[index]
            self.assertEqual(environment["TMPDIR"], str(managed.worker_tmpdir))  # type: ignore[index]
            self.assertEqual(captured["cwd"], managed.worker_cwd)
            self.assertIs(captured["stderr"], runner.subprocess.DEVNULL)
            self.assertNotEqual(managed.worker_cwd, managed.worker_tmpdir)
            for private in (managed.worker_cwd, managed.worker_tmpdir):
                self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)
                self.assertEqual(tuple(private.iterdir()), ())

            public_process_metadata = repr((argv, environment))
            for secret in (*commitments, admission, str(runner.SEALED_SEED_PATH)):
                self.assertNotIn(secret, public_process_metadata)
            managed.terminate()
            managed.wait(timeout=1.0)
            self.assertFalse(managed.worker_cwd.exists())
            self.assertFalse(managed.worker_tmpdir.exists())

    def test_evaluation_worker_rejects_any_noncanonical_seed_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                runner.EvaluatorWorkerSpec(
                    purpose="evaluation",
                    expected_commitments=(
                        _digest("seed-one"),
                        _digest("seed-two"),
                    ),
                    expected_final_admission=runner.COMMITMENT_ONLY_ADMISSION_DIGEST,
                    seed_path=str(Path(temporary) / "substitute-seeds.json"),
                    commitment_only=True,
                )


class SeedSealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "sealed" / "seeds-v1.json"
        self.seeds = (bytes(range(32)), bytes(range(32, 64)))
        self.commitments = tuple(
            evaluator.commit_replicate_seed(seed) for seed in self.seeds
        )

    def _entropy(self):
        pending = list(self.seeds)
        calls: list[int] = []

        def entropy(size: int) -> bytes:
            calls.append(size)
            return pending.pop(0)

        return entropy, calls

    def test_seed_artifact_is_injected_create_once_canonical_and_mode_0600(self) -> None:
        entropy, calls = self._entropy()
        seal = runner.seal_evaluation_seeds(self.path, entropy)
        self.assertEqual(calls, [32, 32])
        self.assertEqual(tuple(seal.seed_commitments), self.commitments)
        self.assertEqual(stat.S_IMODE(self.path.lstat().st_mode), 0o600)
        self.assertTrue(self.path.is_file())
        self.assertFalse(self.path.is_symlink())
        self.assertEqual(
            self.path.read_bytes(),
            runner.canonical_json_bytes(_read_json(self.path)),
        )
        self.assertEqual(runner.load_seed_seal(self.path), seal)
        self.assertNotIn(self.seeds[0].hex(), json.dumps(seal.to_canonical()))
        self.assertNotIn(self.seeds[1].hex(), json.dumps(seal.to_canonical()))

    def test_existing_valid_artifact_is_revalidated_without_entropy(self) -> None:
        entropy, _ = self._entropy()
        original = runner.seal_evaluation_seeds(self.path, entropy)

        def forbidden_entropy(_: int) -> bytes:
            raise AssertionError("existing create-once artifact read entropy")

        repeated = runner.seal_evaluation_seeds(
            self.path,
            forbidden_entropy,
            expected_commitments=self.commitments,
        )
        self.assertEqual(repeated, original)

    def test_consumed_v1_commitment_overlap_is_rejected_on_create_and_rejoin(
        self,
    ) -> None:
        entropy, calls = self._entropy()
        denied = frozenset({self.commitments[0]})
        with mock.patch.object(runner, "CONSUMED_V1_SEED_COMMITMENTS", denied):
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "overlap consumed V1",
            ):
                runner.seal_evaluation_seeds(self.path, entropy)
        self.assertEqual(calls, [32, 32])
        self.assertFalse(self.path.exists())

        entropy, _ = self._entropy()
        runner.seal_evaluation_seeds(self.path, entropy)
        original = self.path.read_bytes()
        with mock.patch.object(runner, "CONSUMED_V1_SEED_COMMITMENTS", denied):
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "overlap consumed V1",
            ):
                runner.load_seed_seal(self.path)

            def forbidden_entropy(_: int) -> bytes:
                raise AssertionError("consumed seed identity requested entropy")

            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "overlap consumed V1",
            ):
                runner.seal_evaluation_seeds(self.path, forbidden_entropy)
        self.assertEqual(self.path.read_bytes(), original)

    def test_consumed_r1_commitment_overlap_is_rejected_on_create_and_rejoin(
        self,
    ) -> None:
        entropy, calls = self._entropy()
        denied = frozenset({self.commitments[0]})
        with mock.patch.object(runner, "CONSUMED_R1_SEED_COMMITMENTS", denied):
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "overlap consumed R1",
            ):
                runner.seal_evaluation_seeds(self.path, entropy)
        self.assertEqual(calls, [32, 32])
        self.assertFalse(self.path.exists())

        entropy, _ = self._entropy()
        runner.seal_evaluation_seeds(self.path, entropy)
        original = self.path.read_bytes()
        with mock.patch.object(runner, "CONSUMED_R1_SEED_COMMITMENTS", denied):
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "overlap consumed R1",
            ):
                runner.load_seed_seal(self.path)

            def forbidden_entropy(_: int) -> bytes:
                raise AssertionError("consumed R1 seed identity requested entropy")

            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "overlap consumed R1",
            ):
                runner.seal_evaluation_seeds(self.path, forbidden_entropy)
        self.assertEqual(self.path.read_bytes(), original)

    def test_duplicate_entropy_permissions_symlinks_and_tamper_fail_closed(self) -> None:
        def duplicate_entropy(size: int) -> bytes:
            self.assertEqual(size, 32)
            return self.seeds[0]

        with self.assertRaises((RuntimeError, ValueError)):
            runner.seal_evaluation_seeds(self.path, duplicate_entropy)
        self.assertFalse(self.path.exists())

        entropy, _ = self._entropy()
        runner.seal_evaluation_seeds(self.path, entropy)
        os.chmod(self.path, 0o640)
        with self.assertRaises((RuntimeError, ValueError)):
            runner.load_seed_seal(self.path)
        os.chmod(self.path, 0o600)

        canonical = self.path.read_bytes()
        self.path.write_bytes(canonical + b" ")
        os.chmod(self.path, 0o600)
        with self.assertRaises((RuntimeError, ValueError)):
            runner.load_seed_seal(self.path)

        self.path.unlink()
        target = self.root / "target.json"
        target.write_bytes(canonical)
        os.chmod(target, 0o600)
        self.path.symlink_to(target)
        with self.assertRaises((RuntimeError, ValueError)):
            runner.load_seed_seal(self.path)

    def test_expected_commitment_mismatch_aborts_without_replacement(self) -> None:
        entropy, _ = self._entropy()
        runner.seal_evaluation_seeds(self.path, entropy)
        original = self.path.read_bytes()
        with self.assertRaises((RuntimeError, ValueError)):
            runner.seal_evaluation_seeds(
                self.path,
                lambda _: b"x" * 32,
                expected_commitments=(_digest("wrong-1"), _digest("wrong-2")),
            )
        self.assertEqual(self.path.read_bytes(), original)

    def test_parent_fsync_failure_consumes_and_preserves_loadable_seed_target(
        self,
    ) -> None:
        entropy, calls = self._entropy()
        with mock.patch.object(
            runner,
            "_fsync_directory",
            side_effect=OSError("INJECTED_SEED_PARENT_FSYNC_FAILURE"),
        ):
            with self.assertRaisesRegex(OSError, "SEED_PARENT_FSYNC_FAILURE"):
                runner.seal_evaluation_seeds(self.path, entropy)
        self.assertEqual(calls, [32, 32])
        self.assertTrue(self.path.is_file())
        self.assertFalse(self.path.is_symlink())
        self.assertEqual(stat.S_IMODE(self.path.lstat().st_mode), 0o600)
        preserved = runner.load_seed_seal(
            self.path,
            expected_commitments=self.commitments,
        )
        self.assertEqual(preserved.seed_commitments, self.commitments)

        def forbidden_entropy(_: int) -> bytes:
            raise AssertionError("consumed target attempted fresh entropy")

        self.assertEqual(
            runner.seal_evaluation_seeds(
                self.path,
                forbidden_entropy,
                expected_commitments=self.commitments,
            ),
            preserved,
        )


class ManifestAdmissionTests(unittest.TestCase):
    @staticmethod
    def _completed_sources() -> dict[str, str]:
        paths = set(_FROZEN_SOURCE_HASHES) | set(runner.REQUIRED_CONSTRUCTION_SOURCES)
        return {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in sorted(paths)
        }

    def setUp(self) -> None:
        self.seeds = (bytes(range(32)), bytes(range(32, 64)))
        self.seed_commitments = tuple(
            evaluator.commit_replicate_seed(seed) for seed in self.seeds
        )
        self.admission = _digest("construction-placeholder")
        owned = evaluator.make_evaluation_evaluator(
            self.seeds,
            self.seed_commitments,
            self.admission,
        )
        self.evaluator_commitments = owned.commitments.to_canonical()
        self.evaluator_commitment_digest = owned.commitments.digest
        self.completed_sources = self._completed_sources()
        self.accepted_leaf_sha256 = hashlib.sha256(
            runner.ACTIVE_LEAF_PATH.read_bytes()
        ).hexdigest()

    def _manifest(self) -> dict[str, object]:
        return runner.build_source_manifest(
            evaluator_commitments=self.evaluator_commitments,
            seed_commitments=self.seed_commitments,
            accepted_leaf_sha256=self.accepted_leaf_sha256,
            completed_source_hashes=self.completed_sources,
        )

    def test_completed_source_inventory_is_exact_without_missing_or_extra_paths(
        self,
    ) -> None:
        self.assertEqual(
            runner.completed_source_manifest(ROOT),
            self.completed_sources,
        )
        missing = dict(self.completed_sources)
        missing.pop(next(iter(missing)))
        extra = {**self.completed_sources, "undeclared.py": "0" * 64}
        for label, source_map in (("missing", missing), ("extra", extra)):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    runner.RunnerInvariantError,
                    label,
                ):
                    runner._validate_completed_source_map(source_map)

    def test_source_inventory_rejects_duplicate_overlap_and_noncanonical_paths(
        self,
    ) -> None:
        expected = runner.REQUIRED_CONSTRUCTION_SOURCES
        cases = (
            (expected[:2] + (expected[1],), "construction source inventory"),
            (
                (next(iter(runner.FROZEN_SOURCE_SHA256)), *expected[1:]),
                "construction source inventory",
            ),
        )
        for sources, message in cases:
            with self.subTest(sources=sources):
                with mock.patch.object(
                    runner,
                    "REQUIRED_CONSTRUCTION_SOURCES",
                    sources,
                ):
                    with self.assertRaisesRegex(
                        runner.RunnerInvariantError,
                        message,
                    ):
                        runner._source_inventory_paths()
        for value in ("./source.py", "source//file.py", "source/../file.py", "source\\file.py"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    runner._canonical_source_path(value)

    def test_completed_sources_are_single_link_and_never_alias_v1_inodes(
        self,
    ) -> None:
        r1_paths = tuple(ROOT / value for value in runner.REQUIRED_CONSTRUCTION_SOURCES)
        v1_paths = (
            ROOT / "experiments/runners/high_level_multidomain_v1.py",
            ROOT / "tests/unit/experiments/test_high_level_multidomain_runner_v1.py",
            ROOT / "tests/integration/runtime/test_high_level_multidomain_qwen_v1.py",
        )
        v1_identities = {
            (path.lstat().st_dev, path.lstat().st_ino) for path in v1_paths
        }
        for path in r1_paths:
            with self.subTest(path=path):
                metadata = path.lstat()
                self.assertTrue(stat.S_ISREG(metadata.st_mode))
                self.assertEqual(metadata.st_nlink, 1)
                self.assertNotIn((metadata.st_dev, metadata.st_ino), v1_identities)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen: dict[str, str] = {}
            all_paths = (
                tuple(runner.FROZEN_SOURCE_SHA256)
                + runner.REQUIRED_CONSTRUCTION_SOURCES
            )
            for index, relative in enumerate(all_paths):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = f"source-{index}:{relative}".encode("ascii")
                target.write_bytes(payload)
                if relative in runner.FROZEN_SOURCE_SHA256:
                    frozen[relative] = hashlib.sha256(payload).hexdigest()
            alias = root / runner.REQUIRED_CONSTRUCTION_SOURCES[0]
            target = root / runner.REQUIRED_CONSTRUCTION_SOURCES[1]
            target.unlink()
            os.link(alias, target)
            with mock.patch.object(runner, "FROZEN_SOURCE_SHA256", frozen):
                with self.assertRaisesRegex(
                    runner.RunnerInvariantError,
                    "single-link|filesystem identity",
                ):
                    runner.completed_source_manifest(root)

    def test_manifest_rejects_rehashed_malformed_task_or_feedback_commitments(
        self,
    ) -> None:
        for label, mutate in (
            ("task-row", lambda value: value.update({"tasks": [{}]})),
            ("random-feedback", lambda value: value.update({"random_feedback": []})),
        ):
            with self.subTest(label=label):
                manifest = self._manifest()
                commitments = json.loads(
                    runner.canonical_json_bytes(
                        manifest["evaluator_commitments"]
                    ).decode("utf-8")
                )
                mutate(commitments)
                manifest["evaluator_commitments"] = commitments
                manifest["evaluator_commitment_digest"] = (
                    runner.evaluator_record_digest(
                        "evaluator-commitments",
                        commitments,
                    )
                )
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_source_manifest(
                        manifest,
                        repository_root=ROOT,
                    )

    def test_manifest_publisher_is_canonical_create_once_and_fsync_consuming(
        self,
    ) -> None:
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "manifest.json"
            self.assertEqual(
                runner.publish_source_manifest(
                    target,
                    manifest,
                    repository_root=ROOT,
                ),
                target,
            )
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assertEqual(
                runner.load_source_manifest(target, repository_root=ROOT),
                manifest,
            )
            with self.assertRaises(runner.RunnerInvariantError):
                runner.publish_source_manifest(
                    target,
                    manifest,
                    repository_root=ROOT,
                )

            original = target.read_bytes()
            target.write_bytes(original + b" ")
            with self.assertRaises((ValueError, runner.RunnerInvariantError)):
                runner.load_source_manifest(target, repository_root=ROOT)
            target.write_bytes(original)
            os.chmod(target, 0o640)
            with self.assertRaises(runner.RunnerInvariantError):
                runner.load_source_manifest(target, repository_root=ROOT)

            symlink = root / "symlink-manifest.json"
            symlink.symlink_to(target)
            with self.assertRaises(runner.RunnerInvariantError):
                runner.publish_source_manifest(
                    symlink,
                    manifest,
                    repository_root=ROOT,
                )

            fsync_target = root / "fsync-manifest.json"
            with mock.patch.object(
                runner,
                "_fsync_directory",
                side_effect=OSError("INJECTED_MANIFEST_FSYNC_FAILURE"),
            ):
                with self.assertRaisesRegex(OSError, "MANIFEST_FSYNC_FAILURE"):
                    runner.publish_source_manifest(
                        fsync_target,
                        manifest,
                        repository_root=ROOT,
                    )
            self.assertTrue(fsync_target.is_file())
            self.assertEqual(stat.S_IMODE(fsync_target.stat().st_mode), 0o600)
            self.assertEqual(
                runner.load_source_manifest(
                    fsync_target,
                    repository_root=ROOT,
                ),
                manifest,
            )

            race_target = root / "race-manifest.json"
            real_link = runner.os.link

            def competing_link(
                source: object,
                destination: object,
                **kwargs: object,
            ) -> None:
                race_target.write_bytes(b"COMPETING")
                real_link(source, destination, **kwargs)

            with mock.patch.object(runner.os, "link", side_effect=competing_link):
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.publish_source_manifest(
                        race_target,
                        manifest,
                        repository_root=ROOT,
                    )
            self.assertEqual(race_target.read_bytes(), b"COMPETING")

    def test_frozen_source_helper_detects_byte_drift_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            first = root / "first.py"
            second = root / "nested/second.py"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            expected = {
                "first.py": hashlib.sha256(b"first").hexdigest(),
                "nested/second.py": hashlib.sha256(b"second").hexdigest(),
            }
            self.assertEqual(
                runner.frozen_source_manifest(root, expected),
                dict(sorted(expected.items())),
            )
            second.write_bytes(b"changed")
            with self.assertRaises(runner.RunnerInvariantError):
                runner.frozen_source_manifest(root, expected)
            second.unlink()
            second.symlink_to(first)
            with self.assertRaises(runner.RunnerInvariantError):
                runner.frozen_source_manifest(root, expected)

    def test_manifest_is_canonical_secret_free_pre_result_and_drift_closed(self) -> None:
        manifest = self._manifest()
        manifest_hash = runner.validate_source_manifest(
            manifest,
            repository_root=ROOT,
        )
        self.assertEqual(
            manifest_hash,
            hashlib.sha256(runner.canonical_json_bytes(manifest)).hexdigest(),
        )
        self.assertIsNone(manifest["qualification_result_sha256"])
        self.assertEqual(manifest["arms"], list(evaluator.EVALUATION_ARMS))
        public = runner.canonical_json_bytes(manifest)
        for seed in self.seeds:
            self.assertNotIn(seed.hex().encode("ascii"), public)
        runner.scan_for_hidden_material(manifest, forbidden_values=self.seeds)

        with self.assertRaises(runner.RunnerInvariantError):
            runner.build_source_manifest(
                evaluator_commitments=self.evaluator_commitments,
                seed_commitments=self.seed_commitments,
                accepted_leaf_sha256=self.accepted_leaf_sha256,
                completed_source_hashes=self.completed_sources,
                qualification_result_sha256=hashlib.sha256(b"qualification").hexdigest(),
            )

        mutations: list[dict[str, object]] = []
        wrong_arm = dict(manifest)
        wrong_arm["arms"] = list(reversed(evaluator.EVALUATION_ARMS))
        mutations.append(wrong_arm)
        wrong_leaf = dict(manifest)
        wrong_leaf["leaf_sha256"] = "0" * 64
        mutations.append(wrong_leaf)
        wrong_sources = dict(manifest)
        source_map = dict(self.completed_sources)
        source_map[next(iter(source_map))] = "0" * 64
        wrong_sources["frozen_sources"] = source_map
        mutations.append(wrong_sources)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_source_manifest(changed, repository_root=ROOT)

        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_source_manifest(
                manifest,
                expected_hash="0" * 64,
                repository_root=ROOT,
            )

    def test_manifest_freezes_host_paths_environment_gpu_resources_and_writes(
        self,
    ) -> None:
        manifest = self._manifest()
        self.assertEqual(manifest["host"], "angler-workstation")
        self.assertEqual(
            manifest["paths"],
            {
                "active_leaf": str(runner.ACTIVE_LEAF_PATH),
                "evaluation_arm_factory_root": str(
                    runner.EVALUATION_ARM_FACTORY_ROOT
                ),
                "evaluation_admission_claim": str(
                    runner.EVALUATION_ADMISSION_CLAIM_PATH
                ),
                "evaluation_cognee_scopes_root": str(
                    runner.EVALUATION_COGNEE_SCOPES_ROOT
                ),
                "evaluation_evidence_ledger": str(
                    runner.EVALUATION_EVIDENCE_LEDGER_PATH
                ),
                "evaluation_genesis_root": str(runner.EVALUATION_GENESIS_ROOT),
                "evaluation_runtime_root": str(runner.EVALUATION_RUNTIME_ROOT),
                "evaluation_result": str(runner.EVALUATION_RESULT_PATH),
                "manifest": str(runner.MANIFEST_PATH),
                "model_root": "/opt/angler/models/Qwen3-4B",
                "qualification_release_claim": str(
                    runner.QUALIFICATION_RELEASE_CLAIM_PATH
                ),
                "qualification_result": str(runner.QUALIFICATION_RESULT_PATH),
                "repository_root": str(runner.REPOSITORY_ROOT),
                "runner": str(runner.RUNNER_PATH),
                "scratch_root": str(runner.SCRATCH_ROOT),
                "sealed_seed": str(runner.SEALED_SEED_PATH),
                "state_root": str(runner.STATE_ROOT),
            },
        )
        self.assertEqual(manifest["paths"], runner.FROZEN_PATHS)
        self.assertEqual(
            manifest["gpu_assignment"],
            {
                "assigned_model": "Qwen3-4B BF16",
                "assigned_name": "NVIDIA GeForce RTX 5080",
                "assigned_uuid": "GPU-df4bb978-e75f-08a0-6660-2b9ed69ee8ca",
                "unassigned_name": "NVIDIA GeForce RTX 5070",
                "unassigned_uuid": "GPU-d9dd1ae0-f65d-ef22-f924-2c3e9c976c1e",
            },
        )
        self.assertEqual(manifest["gpu_assignment"], runner.FROZEN_GPU_ASSIGNMENT)
        self.assertEqual(
            manifest["resource_ceilings"],
            {
                "batch_size": 1,
                "candidate_count": 2,
                "configured_pool_sum": 8,
                "cuda_allocated_bytes": 12 * 1024**3,
                "cuda_reserved_bytes": 12 * 1024**3,
                "evaluation_admitted_execution_attempts": 512,
                "evaluation_generation_attempts": 1_024,
                "ipc_message_bytes": 4 * 1024**2,
                "max_input_tokens": 4_096,
                "max_output_tokens": 64,
                "new_state_scratch_bytes": 4 * 1024**3,
                "pending_snapshot_bytes": 16 * 1024**2,
                "process_rss_bytes": 24 * 1024**3,
                "projection_retry_limit": 64,
                "proposal_attempts_per_task": 1,
                "qualification_generation_attempts": 64,
                "recall_records": 12,
                "result_bytes": 64 * 1024**2,
                "store_page_limit": 64,
                "successful_proposal_execution_attempts_per_task": 1,
                "wall_seconds_after_model_load": 7_200.0,
            },
        )
        self.assertEqual(
            manifest["resource_ceilings"],
            runner.FROZEN_RESOURCE_CEILINGS,
        )
        self.assertEqual(
            manifest["write_scope"],
            [
                f"{runner.STATE_ROOT}/**",
                f"{runner.SCRATCH_ROOT}/**",
                str(runner.QUALIFICATION_RESULT_PATH),
                str(runner.EVALUATION_RESULT_PATH),
            ],
        )
        self.assertEqual(manifest["write_scope"], list(runner.FROZEN_WRITE_SCOPE))

        environment = manifest["environment"]
        self.assertEqual(set(environment), {"cognee_worker", "evaluator_worker", "parent"})
        self.assertEqual(
            environment["parent"],
            {
                "inheritance": "env -i",
                "network": "offline-loopback-only",
                "variables": runner.FROZEN_PARENT_ENVIRONMENT,
            },
        )
        self.assertEqual(
            environment["evaluator_worker"],
            {
                "cwd_parent": str(runner.SCRATCH_ROOT / "evaluator-cwd"),
                "directories": "fresh-empty-0700",
                "fixed_variables": runner.FROZEN_EVALUATOR_WORKER_ENVIRONMENT,
                "inheritance": "env -i",
                "network": "inherits-offline-loopback-only-parent",
                "tmpdir_parent": str(runner.SCRATCH_ROOT / "evaluator-tmp"),
            },
        )
        self.assertEqual(
            environment["cognee_worker"],
            {
                "cuda_visible_devices": "",
                "execution_providers": ["CPUExecutionProvider"],
                "fastembed_onnx_inter_op_threads": 2,
                "fastembed_onnx_intra_op_threads": 2,
                "inheritance": "env -i",
                "network_namespace": "unshare --net loopback-only",
                "scope_state_parents": {
                    "evaluation": str(runner.EVALUATION_COGNEE_SCOPES_ROOT),
                    "qualification": str(runner.QUALIFICATION_COGNEE_SCOPES_ROOT),
                },
                "source_path": "src/angler/memory/cognee_worker_protocol.py",
                "source_sha256": _FROZEN_SOURCE_HASHES[
                    "src/angler/memory/cognee_worker_protocol.py"
                ],
                "telemetry": "disabled",
                "worker_cwd": "private-empty-0700-child-of-each-scope",
                "worker_tmpdir": "private-empty-0700-child-of-each-scope",
            },
        )

        for section in (
            "host",
            "paths",
            "environment",
            "gpu_assignment",
            "resource_ceilings",
            "write_scope",
        ):
            with self.subTest(section=section):
                changed = json.loads(
                    runner.canonical_json_bytes(manifest).decode("utf-8")
                )
                if section == "host":
                    changed[section] = "different-host"
                elif section == "write_scope":
                    changed[section] = changed[section][:-1]
                else:
                    changed[section][next(iter(changed[section]))] = "DRIFT"
                changed_hash = hashlib.sha256(
                    runner.canonical_json_bytes(changed)
                ).hexdigest()
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_source_manifest(
                        changed,
                        expected_hash=changed_hash,
                        repository_root=ROOT,
                    )

    def test_admission_digest_binds_manifest_qualification_evaluator_and_sources(self) -> None:
        manifest_hash = runner.validate_source_manifest(
            self._manifest(),
            repository_root=ROOT,
        )
        qualification_hash = hashlib.sha256(b"qualification-result").hexdigest()
        baseline = runner.admission_digest(
            manifest_sha256=manifest_hash,
            qualification_result_sha256=qualification_hash,
            evaluator_commitment_digest=self.evaluator_commitment_digest,
            source_hashes=self.completed_sources,
        )
        self.assertRegex(baseline, r"^sha256:[0-9a-f]{64}$")

        cases = (
            {
                "manifest_sha256": "0" * 64,
                "qualification_result_sha256": qualification_hash,
                "evaluator_commitment_digest": self.evaluator_commitment_digest,
                "source_hashes": self.completed_sources,
            },
            {
                "manifest_sha256": manifest_hash,
                "qualification_result_sha256": "1" * 64,
                "evaluator_commitment_digest": self.evaluator_commitment_digest,
                "source_hashes": self.completed_sources,
            },
            {
                "manifest_sha256": manifest_hash,
                "qualification_result_sha256": qualification_hash,
                "evaluator_commitment_digest": _digest("different-evaluator"),
                "source_hashes": self.completed_sources,
            },
            {
                "manifest_sha256": manifest_hash,
                "qualification_result_sha256": qualification_hash,
                "evaluator_commitment_digest": self.evaluator_commitment_digest,
                "source_hashes": {
                    **self.completed_sources,
                    "experiments/runners/high_level_multidomain_v1.py": "2" * 64,
                },
            },
        )
        for changed in cases:
            with self.subTest(changed=changed):
                try:
                    observed = runner.admission_digest(**changed)
                except runner.RunnerInvariantError:
                    continue
                self.assertNotEqual(observed, baseline)

    def test_recomputed_hash_cannot_admit_changed_budgets_or_duplicate_seeds(self) -> None:
        baseline = self._manifest()
        changed_budget = json.loads(
            runner.canonical_json_bytes(baseline).decode("utf-8")
        )
        changed_budget["budgets"]["evaluation"]["arm_tasks"] = 467
        changed_budget_hash = hashlib.sha256(
            runner.canonical_json_bytes(changed_budget)
        ).hexdigest()
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_source_manifest(
                changed_budget,
                expected_hash=changed_budget_hash,
                repository_root=ROOT,
            )

        duplicate = json.loads(
            runner.canonical_json_bytes(baseline).decode("utf-8")
        )
        duplicate["seed_commitments"] = [
            self.seed_commitments[0],
            self.seed_commitments[0],
        ]
        duplicate_hash = hashlib.sha256(
            runner.canonical_json_bytes(duplicate)
        ).hexdigest()
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_source_manifest(
                duplicate,
                expected_hash=duplicate_hash,
                repository_root=ROOT,
            )

        with self.assertRaises(ValueError):
            runner.build_source_manifest(
                evaluator_commitments=self.evaluator_commitments,
                seed_commitments=(
                    self.seed_commitments[0],
                    self.seed_commitments[0],
                ),
                accepted_leaf_sha256=self.accepted_leaf_sha256,
                completed_source_hashes=self.completed_sources,
            )


class AttemptBudgetTests(unittest.TestCase):
    def _complete_exact_budget(self, arm_tasks: int) -> dict[str, int]:
        budget = runner.AttemptBudget(
            proposal_ceiling=arm_tasks,
            execution_ceiling=arm_tasks,
            arm_task_ceiling=arm_tasks,
        )
        for ordinal in range(arm_tasks):
            task_id = _digest(f"task-{ordinal}")
            arm = evaluator.EVALUATION_ARMS[ordinal % len(evaluator.EVALUATION_ARMS)]
            budget.consume_proposal(task_id, arm)
            budget.consume_execution(task_id, arm)
            budget.finish_arm_task(task_id, arm)
        return budget.snapshot()

    def test_exact_qualification_and_evaluation_global_counts(self) -> None:
        qualification = self._complete_exact_budget(31)
        self.assertEqual(qualification["arm_tasks"], 31)
        self.assertEqual(qualification["proposal_attempts"], 31)
        self.assertEqual(qualification["execution_attempts"], 31)
        self.assertEqual(qualification["generation_attempts"], 62)

        evaluation = self._complete_exact_budget(468)
        self.assertEqual(evaluation["arm_tasks"], 468)
        self.assertEqual(evaluation["proposal_attempts"], 468)
        self.assertEqual(evaluation["execution_attempts"], 468)
        self.assertEqual(evaluation["generation_attempts"], 936)
        self.assertEqual(runner.QUALIFICATION_COUNTS, (31, 62))
        self.assertEqual(runner.EVALUATION_COUNTS, (468, 936))
        self.assertEqual(
            runner.expected_run_budget("qualification"),
            runner.RunBudget(31, 62, 31),
        )
        self.assertEqual(
            runner.expected_run_budget("evaluation"),
            runner.RunBudget(468, 936, 468),
        )

    def test_malformed_proposal_consumes_one_slot_without_execution_or_retry(self) -> None:
        budget = runner.AttemptBudget(1, 1, 1)
        task_id = _digest("malformed")
        budget.consume_proposal(task_id, "FULL")
        budget.finish_arm_task(task_id, "FULL")
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["arm_tasks"], 1)
        self.assertEqual(snapshot["execution_attempts"], 0)
        self.assertEqual(snapshot["proposal_attempts"], 1)
        self.assertEqual(snapshot["generation_attempts"], 1)
        for operation in (
            budget.consume_proposal,
            budget.consume_execution,
            budget.finish_arm_task,
        ):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises((RuntimeError, ValueError)):
                    operation(task_id, "FULL")

    def test_order_duplicate_and_global_ceiling_failures_do_not_mutate_counts(self) -> None:
        budget = runner.AttemptBudget(1, 1, 1)
        first = _digest("first")
        second = _digest("second")
        with self.assertRaises((RuntimeError, ValueError)):
            budget.consume_execution(first, "FULL")
        self.assertEqual(budget.snapshot()["proposal_attempts"], 0)
        self.assertEqual(budget.snapshot()["execution_attempts"], 0)

        budget.consume_proposal(first, "FULL")
        before = budget.snapshot()
        for operation, key in (
            (budget.consume_proposal, (first, "FULL")),
            (budget.consume_proposal, (second, "FULL")),
            (budget.finish_arm_task, (second, "FULL")),
        ):
            with self.subTest(operation=operation.__name__, key=key):
                with self.assertRaises((RuntimeError, ValueError)):
                    operation(*key)
                self.assertEqual(budget.snapshot(), before)

    def test_budget_keys_admit_only_the_seven_declared_execution_arms(self) -> None:
        for arm in evaluator.EVALUATION_ARMS:
            with self.subTest(arm=arm):
                budget = runner.AttemptBudget(1, 1, 1)
                task_id = _digest(f"declared-{arm}")
                budget.consume_proposal(task_id, arm)
                budget.finish_arm_task(task_id, arm)

        for arm in (evaluator.QUALIFICATION_ARM, "UNKNOWN", "", 1):
            with self.subTest(arm=arm):
                budget = runner.AttemptBudget(1, 1, 1)
                with self.assertRaises((TypeError, ValueError)):
                    budget.consume_proposal(_digest("undeclared"), arm)
                self.assertEqual(budget.snapshot()["proposal_attempts"], 0)


class FeedbackControlAndHiddenBoundaryTests(unittest.TestCase):
    def test_qwen_only_uses_the_exact_no_history_sentinel_without_mislabeling(self) -> None:
        self.assertEqual(
            runner.evidence_for_arm("QWEN_ONLY", ()),
            (runner.NO_PERSISTENT_RECALLED_EVIDENCE_V1,),
        )
        with self.assertRaises(runner.RunnerInvariantError):
            runner.evidence_for_arm("QWEN_ONLY", ("persistent-history",))

        for arm in tuple(
            value for value in evaluator.EVALUATION_ARMS if value != "QWEN_ONLY"
        ):
            with self.subTest(arm=arm):
                self.assertEqual(
                    runner.evidence_for_arm(arm, ("canonical-record",)),
                    ("canonical-record",),
                )
                with self.assertRaises(ValueError):
                    runner.evidence_for_arm(
                        arm,
                        (runner.NO_PERSISTENT_RECALLED_EVIDENCE_V1,),
                    )

    def test_random_feedback_lookup_has_no_truth_or_judgment_input(self) -> None:
        parameters = tuple(inspect.signature(runner.random_feedback_scalar).parameters)
        self.assertEqual(parameters, ("schedule", "task_id"))
        task_ids = tuple(_digest(f"random-{index}") for index in range(12))
        schedule = tuple(
            (task_id, float(index >= 6))
            for index, task_id in enumerate(task_ids)
        )
        truth_a = {task_id: float(index % 2) for index, task_id in enumerate(task_ids)}
        truth_b = {task_id: 1.0 - value for task_id, value in truth_a.items()}
        observed_a = tuple(
            runner.random_feedback_scalar(schedule, task_id) for task_id in task_ids
        )
        observed_b = tuple(
            runner.random_feedback_scalar(schedule, task_id) for task_id in task_ids
        )
        self.assertNotEqual(truth_a, truth_b)
        self.assertEqual(observed_a, observed_b)
        self.assertEqual(observed_a.count(0.0), 6)
        self.assertEqual(observed_a.count(1.0), 6)

        malformed = (*schedule, schedule[0])
        with self.assertRaises((RuntimeError, ValueError)):
            runner.random_feedback_scalar(malformed, task_ids[0])
        with self.assertRaises((RuntimeError, ValueError)):
            runner.random_feedback_scalar(schedule, _digest("not-scheduled"))

    def test_control_receipt_is_literal_first_and_explicitly_nonauthorizing(self) -> None:
        generation_ref = _digest("proposal-generation")
        task_id = _digest("control-task")
        proposals = ("PUBLIC-PROCEDURE-ZERO", "PUBLIC-PROCEDURE-ONE")
        for arm in runner.CONTROL_ARMS:
            with self.subTest(arm=arm):
                receipt = runner.ControlSelectionReceipt.create(
                    arm=arm,
                    proposal_generation_ref=generation_ref,
                    proposals=proposals,
                    task_id=task_id,
                )
                self.assertEqual(receipt.schema, runner.CONTROL_SELECTION_SCHEMA)
                self.assertEqual(receipt.policy, runner.LITERAL_FIRST_PROPOSAL)
                self.assertEqual(receipt.selected_index, 0)
                self.assertEqual(receipt.selected_trace, proposals[0])
                self.assertEqual(receipt.nonauthorization, runner.CONTROL_NONAUTHORIZATION)
                self.assertIn("not an approved action", receipt.nonauthorization)
                self.assertIn("external-effect permission", receipt.nonauthorization)
                self.assertRegex(receipt.receipt_ref, r"^sha256:[0-9a-f]{64}$")
                self.assertEqual(receipt.receipt_ref, receipt.receipt_ref)

                with self.assertRaises(ValueError):
                    replace(receipt, selected_index=1, selected_trace=proposals[1])
                with self.assertRaises(ValueError):
                    replace(receipt, nonauthorization="AUTHORIZED")

        for arm in ("FULL", "RANDOM_FEEDBACK", "QUALIFICATION"):
            with self.subTest(arm=arm):
                with self.assertRaises(ValueError):
                    runner.ControlSelectionReceipt.create(
                        arm=arm,
                        proposal_generation_ref=generation_ref,
                        proposals=proposals,
                        task_id=task_id,
                    )

    def test_hidden_material_scan_rejects_keys_values_and_oversize_without_echo(self) -> None:
        hidden = "SENTINEL-PRIVATE-ANSWER-DO-NOT-ECHO"
        safe = {"public": {"task": "opaque"}}
        self.assertEqual(
            runner.scan_for_hidden_material(safe, forbidden_values=(hidden,)),
            len(runner.canonical_json_bytes(safe)),
        )
        cases = (
            {"raw_seed": "00" * 32},
            {"nested": {"solution": "x"}},
            {"route": ["A", "B"]},
            {"public": hidden},
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(runner.RunnerInvariantError) as caught:
                    runner.scan_for_hidden_material(
                        value,
                        forbidden_values=(hidden,),
                    )
                self.assertNotIn(hidden, str(caught.exception))

        with self.assertRaises(runner.RunnerInvariantError):
            runner.scan_for_hidden_material(
                {"public": "x" * 64},
                maximum_bytes=16,
            )


class ResourceLedgerTests(unittest.TestCase):
    def test_foundation_hashes_must_match_before_after_and_expected(self) -> None:
        expected = {
            "model-files": hashlib.sha256(b"model-files").hexdigest(),
            "tensor-state": hashlib.sha256(b"tensor-state").hexdigest(),
            "tokenizer": hashlib.sha256(b"tokenizer").hexdigest(),
        }
        self.assertEqual(
            runner.validate_foundation_hashes(expected, expected, expected),
            dict(sorted(expected.items())),
        )
        for position in range(3):
            with self.subTest(position=position):
                rows = [dict(expected), dict(expected), dict(expected)]
                rows[position]["tensor-state"] = "0" * 64
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_foundation_hashes(*rows)

        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_foundation_hashes(
                expected,
                expected,
                {"model-files": expected["model-files"]},
            )

    def test_generation_memory_wall_and_pool_ceilings_are_exact(self) -> None:
        ceilings = runner.ResourceCeilings(
            maximum_prompt_tokens=16,
            maximum_output_tokens=4,
            maximum_rss_bytes=100,
            maximum_cuda_allocated_bytes=50,
            maximum_cuda_reserved_bytes=60,
            maximum_wall_seconds=2.5,
            maximum_configured_pool_sum=3,
        )
        ledger = runner.ResourceLedger(ceilings)
        ledger.record_generation(16, 4)
        ledger.record_memory(100, 50, 60)
        ledger.record_wall(2.5)
        ledger.record_configured_pools(3)
        ledger.record_state_scratch(0)
        self.assertEqual(
            ledger.snapshot(),
            {
                "configured_pool_sum": 3,
                "configured_pool_samples": 1,
                "generation_calls": 1,
                "memory_samples": 1,
                "output_tokens": 4,
                "peak_cuda_allocated_bytes": 50,
                "peak_cuda_reserved_bytes": 60,
                "peak_rss_bytes": 100,
                "peak_state_scratch_bytes": 0,
                "prompt_tokens": 16,
                "state_scratch_samples": 1,
                "wall_seconds": 2.5,
                "wall_samples": 1,
            },
        )

        for operation in (
            lambda: runner.ResourceLedger(ceilings).record_generation(17, 1),
            lambda: runner.ResourceLedger(ceilings).record_generation(1, 5),
            lambda: runner.ResourceLedger(ceilings).record_memory(101, 0, 0),
            lambda: runner.ResourceLedger(ceilings).record_memory(0, 51, 0),
            lambda: runner.ResourceLedger(ceilings).record_memory(0, 0, 61),
            lambda: runner.ResourceLedger(ceilings).record_wall(-0.001),
            lambda: runner.ResourceLedger(ceilings).record_wall(2.500001),
            lambda: runner.ResourceLedger(ceilings).record_configured_pools(4),
            lambda: runner.ResourceLedger(ceilings).record_state_scratch(-1),
            lambda: runner.ResourceLedger(ceilings).record_state_scratch(
                4 * 1024**3 + 1
            ),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises((RuntimeError, ValueError)):
                    operation()

    def test_leaf_default_resource_limits_remain_frozen(self) -> None:
        ceilings = runner.ResourceCeilings()
        self.assertEqual(ceilings.maximum_prompt_tokens, 4_096)
        self.assertEqual(ceilings.maximum_output_tokens, 64)
        self.assertEqual(ceilings.maximum_rss_bytes, 24 * 1024**3)
        self.assertEqual(ceilings.maximum_cuda_allocated_bytes, 12 * 1024**3)
        self.assertEqual(ceilings.maximum_cuda_reserved_bytes, 12 * 1024**3)
        self.assertEqual(ceilings.maximum_wall_seconds, 7_200.0)
        self.assertEqual(ceilings.maximum_configured_pool_sum, 8)
        ledger = runner.ResourceLedger()
        ledger.record_state_scratch(4 * 1024**3)
        self.assertEqual(ledger.snapshot()["peak_state_scratch_bytes"], 4 * 1024**3)
        self.assertEqual(ledger.snapshot()["state_scratch_samples"], 1)

    def test_resource_ceiling_types_and_finiteness_are_strict(self) -> None:
        integer_fields = (
            "maximum_prompt_tokens",
            "maximum_output_tokens",
            "maximum_rss_bytes",
            "maximum_cuda_allocated_bytes",
            "maximum_cuda_reserved_bytes",
            "maximum_configured_pool_sum",
        )
        for field_name in integer_fields:
            for invalid in (0, -1, True, 1.0):
                with self.subTest(field=field_name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        runner.ResourceCeilings(**{field_name: invalid})
        for invalid in (0.0, -1.0, True, float("nan"), float("inf")):
            with self.subTest(field="maximum_wall_seconds", invalid=invalid):
                with self.assertRaises(ValueError):
                    runner.ResourceCeilings(maximum_wall_seconds=invalid)

        class DerivedCeilings(runner.ResourceCeilings):
            pass

        with self.assertRaises(TypeError):
            runner.ResourceLedger(DerivedCeilings())
        with self.assertRaises(TypeError):
            runner.ResourceLedger(ceilings=object())  # type: ignore[arg-type]

    def test_unobservable_backend_search_counter_is_never_reported_as_zero(self) -> None:
        class ProposalAdapter:
            def propose_procedure_traces(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

        class Executor:
            def execute(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

        foundation = _digest("frozen-foundation")
        runtime = runner.ArmRuntime(
            proposal_adapter=ProposalAdapter(),
            executor=Executor(),
            foundation_tensor_digest=foundation,
            integrity_context=runner.ProbeIntegrityContext(
                foundation_tensor_digest=foundation,
                foundation_identity_probe=lambda: foundation,
            ),
            backend=object(),
        )
        self.assertIsNone(runtime.backend_search_calls)


class ProbeIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.foundation = _digest("probe-test-foundation")
        self.owned = _FakeOrchestrationEvaluator("evaluation")

    def _spec(
        self,
        phase: str,
        arm: str,
        ordinal: int = 0,
    ) -> runner.ArmTaskSpec:
        task = self.owned.tasks[phase][ordinal]
        return runner.ArmTaskSpec(
            purpose="evaluation",
            phase=phase,  # type: ignore[arg-type]
            arm=arm,  # type: ignore[arg-type]
            task=task,
            judge_arm=arm,  # type: ignore[arg-type]
            feedback_value=(
                0.0
                if phase == "adaptation" and arm == "RANDOM_FEEDBACK"
                else None
            ),
        )

    @staticmethod
    def _result(
        spec: runner.ArmTaskSpec,
        integrity: runner.ProbeIntegrityEvidence,
        ordinal: int,
        parser_disposition: str = "ADMITTED",
    ) -> runner.ArmTaskResult:
        return runner.ArmTaskResult(
            task_id=spec.task_id,
            arm=spec.arm,
            attempt_receipt_ref=_digest(f"probe-integrity-attempt-{ordinal}"),
            raw_response="PUBLIC-RESPONSE",
            parser_disposition=parser_disposition,  # type: ignore[arg-type]
            judgment=None,
            evidence={
                "probe_integrity_ref": runner.content_ref(
                    "probe-integrity",
                    integrity.to_canonical(),
                )
            },
            integrity=integrity,
        )

    def test_per_result_identity_foundation_and_independent_genesis_are_exact(
        self,
    ) -> None:
        spec = self._spec("adaptation", "FULL")
        factory = _FakeProbeIntegrityFactory(self.root, self.foundation)
        integrity = factory(spec)
        expected_genesis = _FakeProbeIntegrityFactory.genesis_digest(
            "evaluation",
            spec.replicate_commitment,
            "FULL",
        )
        integrity.validate_for(
            spec,
            "ADMITTED",
            self.foundation,
            expected_genesis,
        )
        mutations = {
            "identity": replace(integrity, task_id=_digest("other-task")),
            "foundation": replace(
                integrity,
                foundation_tensor_digest=_digest("other-foundation"),
            ),
            "self-asserted-genesis": replace(
                integrity,
                learner_genesis_digest=_digest("self-asserted-genesis"),
            ),
        }
        for label, changed in mutations.items():
            with self.subTest(label=label):
                with self.assertRaises(runner.RunnerInvariantError):
                    changed.validate_for(
                        spec,
                        "ADMITTED",
                        self.foundation,
                        expected_genesis,
                    )

    def test_phase_clone_source_applicability_and_content_bound_audit_ref(
        self,
    ) -> None:
        factory = _FakeProbeIntegrityFactory(self.root, self.foundation)
        adaptation = factory(self._spec("adaptation", "FULL"))
        self.assertIsNone(adaptation.source_baseline_before)
        self.assertIsNone(adaptation.clone_audit)
        self.assertNotEqual(
            adaptation.learner_parent_digest,
            adaptation.learner_child_digest,
        )

        development = factory(self._spec("development", "FULL"))
        self.assertEqual(
            set(dict(development.source_baseline_before)),  # type: ignore[arg-type]
            runner.PROBE_BASELINE_HASH_NAMES,
        )
        self.assertEqual(
            development.source_baseline_before,
            development.source_baseline_after,
        )
        self.assertEqual(development.clone_before, development.source_baseline_before)
        audit = development.clone_audit
        self.assertIs(type(audit), runner.CloneAudit)
        self.assertEqual(
            set(dict(audit.source_hashes)),  # type: ignore[union-attr]
            runner.CLONED_BASELINE_HASH_NAMES,
        )
        canonical = development.to_canonical()
        self.assertEqual(canonical["clone_audit_ref"], audit.audit_ref)  # type: ignore[union-attr]
        self.assertEqual(
            audit.audit_ref,  # type: ignore[union-attr]
            runner.content_ref("probe-clone", audit.to_canonical()),  # type: ignore[union-attr]
        )
        alternate = replace(audit, root=str(self.root / "different-clone"))  # type: ignore[arg-type]
        self.assertNotEqual(alternate.audit_ref, audit.audit_ref)  # type: ignore[union-attr]

        changed_source = dict(development.source_baseline_after)  # type: ignore[arg-type]
        changed_source["acquisition_sequence"] = hashlib.sha256(
            b"MUTATED-ACQUISITION-SEQUENCE"
        ).hexdigest()
        with self.assertRaises(runner.RunnerInvariantError):
            replace(
                development,
                source_baseline_after=tuple(sorted(changed_source.items())),
            )
        with self.assertRaises((TypeError, ValueError)):
            replace(development, clone_audit=None)
        with self.assertRaises(ValueError):
            replace(
                adaptation,
                source_baseline_before=development.source_baseline_before,
                source_baseline_after=development.source_baseline_after,
                clone_before=development.clone_before,
                clone_after=development.clone_after,
                clone_audit=development.clone_audit,
            )

        qwen = factory(self._spec("development", "QWEN_ONLY"))
        self.assertIsNone(qwen.source_baseline_before)
        self.assertIsNone(qwen.clone_audit)
        with self.assertRaises(ValueError):
            replace(
                qwen,
                source_baseline_before=development.source_baseline_before,
                source_baseline_after=development.source_baseline_after,
                clone_before=development.clone_before,
                clone_after=development.clone_after,
                clone_audit=development.clone_audit,
            )

    def test_full_adaptation_lineages_separate_integrity_from_advance_predicate(
        self,
    ) -> None:
        factory = _FakeProbeIntegrityFactory(self.root, self.foundation)
        expected = {
            replicate: _FakeProbeIntegrityFactory.genesis_digest(
                "evaluation",
                replicate,
                "FULL",
            )
            for replicate in self.owned.replicates
        }
        attempts: list[runner.ArmTaskResult] = []
        for ordinal, task in enumerate(self.owned.tasks["adaptation"]):
            spec = runner.ArmTaskSpec(
                purpose="evaluation",
                phase="adaptation",
                arm="FULL",
                task=task,
                judge_arm="FULL",
            )
            attempts.append(self._result(spec, factory(spec), ordinal))

        aggregate, aggregate_ref = runner.validate_full_adaptation_lineages(
            attempts,
            expected,
        )
        self.assertEqual(aggregate["expected_genesis_by_replicate"], expected)
        self.assertEqual(
            aggregate_ref,
            runner.content_ref("full-adaptation-lineages", aggregate),
        )
        for replicate in aggregate["replicates"]:
            transitions = replicate["transitions"]
            self.assertEqual(len(transitions), 12)
            self.assertEqual(transitions[0]["parent_digest"], replicate["genesis_digest"])
            self.assertEqual(
                [row["sequence"] for row in transitions],
                list(range(2, 14)),
            )
            for previous, current in zip(transitions, transitions[1:]):
                self.assertEqual(
                    current["parent_digest"],
                    previous["child_digest"],
                )
            self.assertEqual(
                replicate["final_child_digest"],
                transitions[-1]["child_digest"],
            )

        reset = list(attempts)
        reset_integrity = replace(
            reset[1].integrity,
            learner_parent_digest=reset[1].integrity.learner_genesis_digest,
        )
        reset[1] = replace(reset[1], integrity=reset_integrity)
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_full_adaptation_lineages(reset, expected)

        self_asserted = {
            replicate: _digest("self-asserted-shared-genesis")
            for replicate in expected
        }
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_full_adaptation_lineages(attempts, self_asserted)

        malformed_factory = _FakeProbeIntegrityFactory(
            self.root / "malformed",
            self.foundation,
        )
        all_malformed: list[runner.ArmTaskResult] = []
        first_replicate = self.owned.replicates[0]
        for ordinal, task in enumerate(self.owned.tasks["adaptation"]):
            spec = runner.ArmTaskSpec(
                purpose="evaluation",
                phase="adaptation",
                arm="FULL",
                task=task,
                judge_arm="FULL",
            )
            disposition = (
                "MALFORMED"
                if spec.replicate_commitment == first_replicate
                else "ADMITTED"
            )
            all_malformed.append(
                self._result(
                    spec,
                    malformed_factory(spec, disposition),
                    100 + ordinal,
                    disposition,
                )
            )
        malformed_aggregate, malformed_ref = (
            runner.validate_full_adaptation_lineages(all_malformed, expected)
        )
        by_replicate = {
            row["replicate_commitment"]: row
            for row in malformed_aggregate["replicates"]
        }
        self.assertFalse(malformed_aggregate["all_replicates_advanced"])
        self.assertFalse(by_replicate[first_replicate]["advanced"])
        self.assertTrue(
            by_replicate[self.owned.replicates[1]]["advanced"]
        )
        self.assertEqual(
            malformed_ref,
            runner.content_ref(
                "full-adaptation-lineages",
                malformed_aggregate,
            ),
        )

        admitted_without_transition = list(attempts)
        admitted_without_transition[0] = replace(
            admitted_without_transition[0],
            integrity=replace(
                admitted_without_transition[0].integrity,
                learner_child_digest=None,
                learner_sequence=None,
            ),
        )
        with self.assertRaisesRegex(
            runner.RunnerInvariantError,
            "transition did not advance",
        ):
            runner.validate_full_adaptation_lineages(
                admitted_without_transition,
                expected,
            )

    def test_bootstrap_sequence_two_is_exact_for_live_and_serialized_lineages(
        self,
    ) -> None:
        self.assertEqual(runner.QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE, 1)
        factory = _FakeProbeIntegrityFactory(self.root, self.foundation)
        expected = {
            replicate: _FakeProbeIntegrityFactory.genesis_digest(
                "evaluation",
                replicate,
                "FULL",
            )
            for replicate in self.owned.replicates
        }
        attempts: list[runner.ArmTaskResult] = []
        ordinal = 0
        for task in self.owned.tasks["adaptation"]:
            for arm in ("FULL", "RANDOM_FEEDBACK"):
                spec = runner.ArmTaskSpec(
                    purpose="evaluation",
                    phase="adaptation",
                    arm=arm,
                    task=task,
                    judge_arm=arm,
                    feedback_value=(0.0 if arm == "RANDOM_FEEDBACK" else None),
                )
                attempts.append(self._result(spec, factory(spec), ordinal))
                ordinal += 1

        aggregate, _ = runner.validate_adaptation_lineages(
            attempts,
            expected,
            purpose="evaluation",
        )
        full_aggregate, _ = runner.validate_full_adaptation_lineages(
            attempts,
            expected,
        )
        self.assertTrue(
            all(row["transitions"][0]["sequence"] == 2 for row in aggregate["lineages"])
        )
        self.assertTrue(
            all(
                row["transitions"][0]["sequence"] == 2
                for row in full_aggregate["replicates"]
            )
        )
        serialized, indexed = runner._validate_serialized_adaptation_lineage(
            aggregate,
            purpose="evaluation",
        )
        self.assertEqual(serialized, aggregate)
        self.assertEqual(len(indexed), 4)

        full_attempts = [item for item in attempts if item.arm == "FULL"]

        def changed_live(
            index: int,
            *,
            parser_disposition: str | None = None,
            **integrity_changes: object,
        ) -> list[runner.ArmTaskResult]:
            changed = list(full_attempts)
            original = changed[index]
            changed[index] = replace(
                original,
                parser_disposition=(
                    original.parser_disposition
                    if parser_disposition is None
                    else parser_disposition
                ),
                integrity=replace(original.integrity, **integrity_changes),
            )
            return changed

        genesis = full_attempts[0].integrity.learner_genesis_digest
        live_failures = {
            "start-at-one": changed_live(0, learner_sequence=1),
            "start-at-three": changed_live(0, learner_sequence=3),
            "gap": changed_live(1, learner_sequence=4),
            "duplicate": changed_live(1, learner_sequence=2),
            "reset": changed_live(1, learner_parent_digest=genesis),
            "fork": changed_live(1, learner_parent_digest=_digest("live-fork")),
            "malformed-with-transition": changed_live(
                0,
                parser_disposition="MALFORMED",
            ),
            "admitted-without-transition": changed_live(
                0,
                learner_child_digest=None,
                learner_sequence=None,
            ),
        }
        unchanged_child = list(full_attempts)
        corrupted_integrity = copy.copy(unchanged_child[0].integrity)
        object.__setattr__(
            corrupted_integrity,
            "learner_child_digest",
            corrupted_integrity.learner_parent_digest,
        )
        unchanged_child[0] = replace(
            unchanged_child[0],
            integrity=corrupted_integrity,
        )
        live_failures["unchanged-child"] = unchanged_child
        for label, changed in live_failures.items():
            with self.subTest(validator="live", mutation=label):
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_full_adaptation_lineages(changed, expected)

        def serialized_change(label: str) -> dict[str, object]:
            changed = json.loads(json.dumps(aggregate))
            row = changed["lineages"][0]
            transitions = row["transitions"]
            if label == "start-at-one":
                transitions[0]["sequence"] = 1
            elif label == "start-at-three":
                transitions[0]["sequence"] = 3
            elif label == "gap":
                transitions[1]["sequence"] = 4
            elif label == "duplicate":
                transitions[1]["sequence"] = 2
            elif label == "reset":
                transitions[1]["parent_digest"] = row["genesis_digest"]
            elif label == "fork":
                transitions[1]["parent_digest"] = _digest("serialized-fork")
            elif label == "unchanged-child":
                transitions[0]["child_digest"] = transitions[0]["parent_digest"]
            elif label == "malformed-with-transition":
                transitions[0]["parser_disposition"] = "MALFORMED"
            elif label == "admitted-without-transition":
                transitions[0]["child_digest"] = None
                transitions[0]["sequence"] = None
            else:
                raise AssertionError(f"unknown serialized mutation: {label}")
            return changed

        for label in live_failures:
            with self.subTest(validator="serialized", mutation=label):
                with self.assertRaises(runner.RunnerInvariantError):
                    runner._validate_serialized_adaptation_lineage(
                        serialized_change(label),
                        purpose="evaluation",
                    )

    def test_random_feedback_lineage_is_continuous_and_clone_roots_are_one_use(
        self,
    ) -> None:
        factory = _FakeProbeIntegrityFactory(self.root, self.foundation)
        expected = {
            replicate: _FakeProbeIntegrityFactory.genesis_digest(
                "evaluation",
                replicate,
                "FULL",
            )
            for replicate in self.owned.replicates
        }
        attempts: list[runner.ArmTaskResult] = []
        ordinal = 0
        for task in self.owned.tasks["adaptation"]:
            for arm in ("FULL", "RANDOM_FEEDBACK"):
                spec = runner.ArmTaskSpec(
                    purpose="evaluation",
                    phase="adaptation",
                    arm=arm,
                    task=task,
                    judge_arm=arm,
                    feedback_value=(0.0 if arm == "RANDOM_FEEDBACK" else None),
                )
                attempts.append(self._result(spec, factory(spec), ordinal))
                ordinal += 1
        aggregate, aggregate_ref = runner.validate_adaptation_lineages(
            attempts,
            expected,
            purpose="evaluation",
        )
        self.assertEqual(len(aggregate["lineages"]), 4)
        self.assertEqual(
            aggregate_ref,
            runner.content_ref("adaptation-lineages", aggregate),
        )

        random_indices = [
            index
            for index, result in enumerate(attempts)
            if result.arm == "RANDOM_FEEDBACK"
            and result.integrity.replicate_commitment == self.owned.replicates[0]
        ]
        broken = list(attempts)
        target = random_indices[1]
        broken[target] = replace(
            broken[target],
            integrity=replace(
                broken[target].integrity,
                learner_parent_digest=(
                    broken[target].integrity.learner_genesis_digest
                ),
            ),
        )
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_adaptation_lineages(
                broken,
                expected,
                purpose="evaluation",
            )

        development_factory = _FakeProbeIntegrityFactory(
            self.root / "disposable",
            self.foundation,
        )
        first_spec = self._spec("development", "FULL")
        second_spec = self._spec("development", "FROZEN_ORIGIN")
        first = self._result(first_spec, development_factory(first_spec), 500)
        second_integrity = development_factory(second_spec)
        second_integrity = replace(
            second_integrity,
            clone_audit=first.integrity.clone_audit,
        )
        second = self._result(second_spec, second_integrity, 501)
        with self.assertRaises(runner.RunnerInvariantError):
            runner._append_unique_attempt([first], second)

    def test_foundation_identity_probe_runs_at_finalize_and_rejects_drift(
        self,
    ) -> None:
        spec = self._spec("adaptation", "QWEN_ONLY")
        observed: list[str] = []

        def exact_probe() -> str:
            observed.append("parameter-identity-and-version-checked")
            return self.foundation

        context = runner.ProbeIntegrityContext(
            foundation_tensor_digest=self.foundation,
            foundation_identity_probe=exact_probe,
        )
        integrity = context.finalize(
            spec,
            learner_parent_digest=None,
            learner_child_digest=None,
            learner_sequence=None,
        )
        self.assertEqual(observed, ["parameter-identity-and-version-checked"])
        self.assertEqual(integrity.foundation_tensor_digest, self.foundation)
        self.assertEqual(
            integrity.foundation_guard_semantics,
            runner.FOUNDATION_GUARD_SEMANTICS,
        )

        drift = runner.ProbeIntegrityContext(
            foundation_tensor_digest=self.foundation,
            foundation_identity_probe=lambda: _digest("drifted-foundation"),
        )
        with self.assertRaises(runner.RunnerInvariantError):
            drift.finalize(
                spec,
                learner_parent_digest=None,
                learner_child_digest=None,
                learner_sequence=None,
            )


class RemovalFairnessTests(unittest.TestCase):
    @staticmethod
    def _rows() -> dict[str, dict[str, object]]:
        task_id = _digest("removal-task")
        return {
            "full": _fair_arm_evidence(task_id, "FULL"),
            "frozen_origin": _fair_arm_evidence(task_id, "FROZEN_ORIGIN"),
            "prospective_removal": _fair_arm_evidence(
                task_id,
                "PROSPECTIVE_REMOVAL",
            ),
            "backend_removal": _fair_arm_evidence(task_id, "BACKEND_REMOVAL"),
            "retrieval_only": _fair_arm_evidence(task_id, "RETRIEVAL_ONLY"),
        }

    def test_exact_removal_inputs_and_backend_replay_output_are_bound(self) -> None:
        rows = self._rows()
        observed = runner.validate_removal_fairness(**rows)
        self.assertRegex(observed, r"^sha256:[0-9a-f]{64}$")

        for arm, field, value in (
            ("prospective_removal", "proposal_request_bytes", "different"),
            ("backend_removal", "backend_search_calls", 1),
            ("backend_removal", "raw_response", "different"),
        ):
            with self.subTest(arm=arm, field=field):
                changed = {name: dict(row) for name, row in rows.items()}
                changed[arm][field] = value
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_removal_fairness(**changed)
        for field_mutation in (
            lambda row: row.update({"extra_evidence": "laundered"}),
            lambda row: row.pop("evidence_stage_ref"),
        ):
            changed = {name: dict(row) for name, row in rows.items()}
            field_mutation(changed["full"])
            with self.assertRaises(runner.RunnerInvariantError):
                runner.validate_removal_fairness(**changed)


class CloneIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source_root = self.root / "baseline"
        source_root.mkdir(mode=0o700)
        self.source = runner.ClonePaths(
            source_root,
            source_root / "cognitive.sqlite3",
            source_root / "qwen-execution-journal.sqlite3",
            source_root / "learner-state.bin",
        )
        self._make_database(self.source.store, "store-baseline")
        self._make_database(self.source.journal, "journal-baseline")
        self.source.learner.write_bytes(b"learner-baseline-v1")
        self.expected = {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in (
                ("store", self.source.store),
                ("journal", self.source.journal),
                ("learner", self.source.learner),
            )
        }

    @staticmethod
    def _make_database(path: Path, value: str) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE evidence(value TEXT NOT NULL)")
            connection.execute("INSERT INTO evidence VALUES (?)", (value,))
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _database_values(path: Path) -> tuple[str, ...]:
        connection = sqlite3.connect(path)
        try:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT value FROM evidence ORDER BY rowid"
                )
            )
        finally:
            connection.close()

    def test_each_arm_task_clone_is_byte_exact_unique_and_mutation_isolated(self) -> None:
        clone_parent = self.root / "clones"
        first = runner.ClonePaths.for_task(
            clone_parent,
            _digest("probe-task"),
            "FULL",
        )
        second = runner.ClonePaths.for_task(
            clone_parent,
            _digest("probe-task"),
            "FROZEN_ORIGIN",
        )
        first_audit = runner.clone_closed_baseline(self.source, first, self.expected)
        second_audit = runner.clone_closed_baseline(self.source, second, self.expected)
        self.assertNotEqual(first.root, second.root)
        self.assertEqual(dict(first_audit.source_hashes), self.expected)
        self.assertEqual(dict(first_audit.destination_hashes), self.expected)
        self.assertEqual(dict(second_audit.destination_hashes), self.expected)

        connection = sqlite3.connect(first.store)
        try:
            connection.execute("INSERT INTO evidence VALUES ('probe-only')")
            connection.commit()
        finally:
            connection.close()
        first.learner.write_bytes(b"mutated-probe-owner")

        self.assertEqual(self._database_values(self.source.store), ("store-baseline",))
        self.assertEqual(self._database_values(second.store), ("store-baseline",))
        self.assertEqual(second.learner.read_bytes(), b"learner-baseline-v1")
        self.assertEqual(
            hashlib.sha256(self.source.journal.read_bytes()).hexdigest(),
            self.expected["journal"],
        )

    def test_clone_rejects_drift_open_sidecars_and_destination_reuse(self) -> None:
        clone_parent = self.root / "clones"
        destination = runner.ClonePaths.for_task(
            clone_parent,
            _digest("isolation-task"),
            "BACKEND_REMOVAL",
        )
        wrong = dict(self.expected)
        wrong["learner"] = "0" * 64
        with self.assertRaises(runner.RunnerInvariantError):
            runner.clone_closed_baseline(self.source, destination, wrong)
        self.assertFalse(destination.root.exists())

        sidecar = Path(str(self.source.store) + "-wal")
        sidecar.write_bytes(b"open")
        with self.assertRaises(runner.RunnerInvariantError):
            runner.clone_closed_baseline(self.source, destination, self.expected)
        self.assertFalse(destination.root.exists())
        sidecar.unlink()

        runner.clone_closed_baseline(self.source, destination, self.expected)
        with self.assertRaises(runner.RunnerInvariantError):
            runner.clone_closed_baseline(self.source, destination, self.expected)

    def test_clone_rejects_source_mutation_after_a_file_is_copied(self) -> None:
        destination = runner.ClonePaths.for_task(
            self.root / "clones",
            _digest("source-race-task"),
            "FULL",
        )
        real_copy = runner.shutil.copyfileobj
        calls = 0

        def mutate_after_copy(reader: object, writer: object, **kwargs: object) -> None:
            nonlocal calls
            real_copy(reader, writer, **kwargs)
            calls += 1
            if calls == 1:
                self.source.journal.write_bytes(b"MUTATED-AFTER-COPY")

        with mock.patch.object(
            runner.shutil,
            "copyfileobj",
            side_effect=mutate_after_copy,
        ):
            with self.assertRaises(runner.RunnerInvariantError):
                runner.clone_closed_baseline(
                    self.source,
                    destination,
                    self.expected,
                )
        self.assertFalse(destination.root.exists())


class EvidenceLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "evidence" / "attempts.sqlite3"
        self.intent_ref = _digest("run-intent")
        self.ledger = runner.EvidenceLedger(self.path, self.intent_ref)
        self.task_id = _digest("ledger-task")
        self.source_receipt = _digest("ledger-execution-receipt")
        self.receipt = runner.attempt_receipt_reference(
            runner.attempt_receipt_payload(
                purpose="evaluation",
                phase="development",
                task_id=self.task_id,
                arm="FULL",
                parser_disposition="ADMITTED",
                source_receipt_ref=self.source_receipt,
                source_kind="EXECUTION_RECEIPT",
            )
        )

    def _stage(
        self,
        *,
        task_id: str | None = None,
        arm: str = "FULL",
        receipt: str | None = None,
        purpose: str = "evaluation",
        judge_arm: str | None = None,
    ) -> dict[str, object]:
        source_receipt = receipt or self.source_receipt
        resolved_judge_arm = arm if purpose == "evaluation" and judge_arm is None else judge_arm
        resolved_task = task_id or self.task_id
        attempt_receipt = runner.attempt_receipt_payload(
            purpose=purpose,
            phase="development",
            task_id=resolved_task,
            arm=arm,
            parser_disposition="ADMITTED",
            source_receipt_ref=source_receipt,
            source_kind="EXECUTION_RECEIPT",
        )
        return {
            "arm": arm,
            "attempt_receipt": attempt_receipt,
            "attempt_receipt_ref": runner.attempt_receipt_reference(attempt_receipt),
            "execution_receipt": {"execution_receipt_ref": source_receipt},
            "execution_request": {"request_ref": _digest("execution-request")},
            "feedback_source": {"kind": "OBJECTIVE_EVALUATOR_POST_RECEIPT"},
            "foundation_tensor_digest": _digest("foundation"),
            "frozen_recall_ref": _digest("frozen-recall"),
            "learner_parent_digest": _digest("learner-parent"),
            "judge_arm": resolved_judge_arm,
            "parser_disposition": "ADMITTED",
            "phase": "development",
            "proposal_generation": {
                "generation_ref": _digest("proposal-generation"),
                "raw_response": "P0\nP1",
            },
            "proposal_request": {"candidate_count": 2},
            "proposals": ["P0", "P1"],
            "public_task_ref": _digest("public-task"),
            "purpose": purpose,
            "raw_response": "MODEL-OUTPUT",
            "resource_counters": {"generation_attempts": 2},
            "schema": runner.ATTEMPT_STAGE_SCHEMA,
            "selection": {"selected_index": 0},
            "task_id": resolved_task,
            "task_response_generation": {"raw_response": "MODEL-OUTPUT"},
        }

    def _judgment(
        self,
        *,
        task_id: str | None = None,
        arm: str = "FULL",
        receipt: str | None = None,
        purpose: str = "evaluation",
        phase: str = "development",
    ) -> dict[str, object]:
        attempt_receipt = receipt or self.receipt
        task = task_id or self.task_id
        response_payload = {
            "arm": arm,
            "attempt_receipt_ref": attempt_receipt,
            "raw_response": "MODEL-OUTPUT",
            "task_id": task,
        }
        integrity = {
            "arm": arm,
            "phase": phase,
            "purpose": purpose,
            "schema": runner.PROBE_INTEGRITY_SCHEMA,
            "task_id": task,
        }
        return {
            "arm": arm,
            "attempt_receipt_ref": attempt_receipt,
            "feedback_record": {"score": 0.0},
            "learner_transition": {"parent": _digest("p"), "child": _digest("c")},
            "objective_judgment": {
                "arm": arm,
                "attempt_receipt_ref": attempt_receipt,
                "disposition": "UNSUCCESSFUL",
                "raw_response": "MODEL-OUTPUT",
                "response_commitment": runner.evaluator_record_digest(
                    "response",
                    response_payload,
                ),
                "score": 0.0,
                "task_id": task,
            },
            "probe_integrity": integrity,
            "probe_integrity_ref": runner.content_ref(
                "probe-integrity",
                integrity,
            ),
            "resource_counters": {"generation_attempts": 2},
            "runtime_quiescence": None,
            "runtime_quiescence_ref": None,
            "schema": runner.ATTEMPT_FINAL_SCHEMA,
            "task_id": task,
            "unevaluated_resolution": None,
        }

    def test_stage_finalize_restart_and_exact_idempotence_are_canonical(self) -> None:
        stage = self._stage()
        stage_ref = self.ledger.stage_attempt(stage)
        self.assertEqual(self.ledger.stage_attempt(stage), stage_ref)
        self.assertEqual(
            self.ledger.read_all(),
            ({"attempt": stage, "judgment": None},),
        )

        key = (self.task_id, "FULL", self.receipt)
        judgment = self._judgment()
        judgment_ref = self.ledger.finalize_attempt(key, judgment)
        self.assertEqual(self.ledger.finalize_attempt(key, judgment), judgment_ref)
        self.ledger.audit()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

        restarted = runner.EvidenceLedger(self.path, self.intent_ref)
        self.assertEqual(
            restarted.read_all(),
            ({"attempt": stage, "judgment": judgment},),
        )
        with self.assertRaises(runner.RunnerInvariantError):
            runner.EvidenceLedger(self.path, _digest("different-intent"))

    def test_attempt_identity_separates_arms_without_rewriting_source_receipt(
        self,
    ) -> None:
        full = self._stage(arm="FULL", receipt=self.source_receipt)
        random_feedback = self._stage(
            arm="RANDOM_FEEDBACK",
            receipt=self.source_receipt,
        )
        self.assertEqual(
            full["execution_receipt"],
            random_feedback["execution_receipt"],
        )
        self.assertNotEqual(
            full["attempt_receipt_ref"],
            random_feedback["attempt_receipt_ref"],
        )
        for stage in (full, random_feedback):
            self.assertEqual(
                stage["attempt_receipt"]["source_receipt_ref"],
                self.source_receipt,
            )
            self.assertEqual(
                stage["attempt_receipt"]["nonauthorization"],
                runner.ATTEMPT_RECEIPT_NONAUTHORIZATION,
            )
            self.ledger.stage_attempt(stage)
        self.assertEqual(len(self.ledger.read_all()), 2)

    def test_qwen_journal_quiescence_is_an_exact_content_bound_shape(self) -> None:
        quiescence = {
            "entries": 1,
            "journal_bytes": 4096,
            "journal_sha256": hashlib.sha256(b"journal").hexdigest(),
            "kind": runner.QWEN_JOURNAL_QUIESCENCE_KIND,
        }
        quiescence_ref = runner.content_ref("runtime-quiescence", quiescence)
        self.assertEqual(
            runner._validate_qwen_journal_quiescence(
                quiescence,
                quiescence_ref,
            ),
            quiescence,
        )
        cases = (
            {**quiescence, "extra": True},
            {**quiescence, "entries": True},
            {**quiescence, "journal_bytes": 0},
            {**quiescence, "journal_sha256": "sha256:" + "0" * 64},
        )
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises((ValueError, runner.RunnerInvariantError)):
                    runner._validate_qwen_journal_quiescence(
                        malformed,
                        runner.content_ref("runtime-quiescence", malformed),
                    )
        with self.assertRaises(runner.RunnerInvariantError):
            runner._validate_qwen_journal_quiescence(
                quiescence,
                _digest("wrong-quiescence-ref"),
            )

    def test_global_receipt_and_task_arm_uniqueness_fail_before_mutation(self) -> None:
        stage = self._stage()
        self.ledger.stage_attempt(stage)
        before = self.ledger.read_all()

        with self.assertRaises(runner.RunnerInvariantError):
            self.ledger.stage_attempt(
                self._stage(receipt=_digest("different-receipt"))
            )
        self.assertEqual(self.ledger.read_all(), before)

        laundered = self._stage(task_id=_digest("different-task"))
        laundered["attempt_receipt_ref"] = self.receipt
        with self.assertRaises(runner.RunnerInvariantError):
            self.ledger.stage_attempt(laundered)
        self.assertEqual(self.ledger.read_all(), before)

        with self.assertRaises(runner.RunnerInvariantError):
            self.ledger.finalize_attempt(
                (_digest("unstaged"), "FULL", _digest("unstaged-receipt")),
                self._judgment(
                    task_id=_digest("unstaged"),
                    receipt=_digest("unstaged-receipt"),
                ),
            )
        self.assertEqual(self.ledger.read_all(), before)

    def test_finalized_rows_are_immutable_and_audit_detects_byte_tamper(self) -> None:
        self.ledger.stage_attempt(self._stage())
        key = (self.task_id, "FULL", self.receipt)
        judgment = self._judgment()
        self.ledger.finalize_attempt(key, judgment)
        conflicting = dict(judgment)
        conflicting["resource_counters"] = {"generation_attempts": 3}
        with self.assertRaises(runner.RunnerInvariantError):
            self.ledger.finalize_attempt(key, conflicting)

        connection = sqlite3.connect(self.path)
        try:
            connection.execute(
                "UPDATE attempts SET stage_bytes=? WHERE task_id=? AND arm=?",
                (b'{"tampered":true}', self.task_id, "FULL"),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises((RuntimeError, ValueError)):
            self.ledger.audit()

    def test_wrong_file_mode_rejects_mutation_before_any_ledger_write(self) -> None:
        before = self.path.read_bytes()
        os.chmod(self.path, 0o640)
        try:
            with self.assertRaises(runner.RunnerInvariantError):
                self.ledger.stage_attempt(self._stage())
        finally:
            os.chmod(self.path, 0o600)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.ledger.read_all(), ())

    def test_nested_judgment_arm_and_score_disposition_are_exactly_bound(self) -> None:
        cases = ("nested-arm", "score-disposition")
        for label in cases:
            with self.subTest(label=label):
                task_id = _digest(f"{label}-task")
                receipt = _digest(f"{label}-receipt")
                ledger = runner.EvidenceLedger(
                    self.root / label / "attempts.sqlite3",
                    self.intent_ref,
                )
                stage = self._stage(task_id=task_id, receipt=receipt)
                ledger.stage_attempt(stage)
                attempt_receipt = stage["attempt_receipt_ref"]
                final = self._judgment(
                    task_id=task_id,
                    receipt=attempt_receipt,
                )
                objective = final["objective_judgment"]
                if label == "nested-arm":
                    objective["arm"] = "QWEN_ONLY"  # type: ignore[index]
                else:
                    objective["score"] = 1.0  # type: ignore[index]
                with self.assertRaises(ValueError):
                    ledger.finalize_attempt(
                        (task_id, "FULL", attempt_receipt),
                        final,
                    )
                self.assertIsNone(ledger.read_all()[0]["judgment"])

    def test_random_feedback_transition_is_finalized_without_objective_judgment(
        self,
    ) -> None:
        task_id = _digest("random-feedback-persistence-task")
        receipt = _digest("random-feedback-persistence-receipt")
        stage = self._stage(
            task_id=task_id,
            arm="RANDOM_FEEDBACK",
            receipt=receipt,
            purpose="qualification",
            judge_arm=None,
        )
        stage["phase"] = "adaptation"
        stage["attempt_receipt"] = runner.attempt_receipt_payload(
            purpose="qualification",
            phase="adaptation",
            task_id=task_id,
            arm="RANDOM_FEEDBACK",
            parser_disposition="ADMITTED",
            source_receipt_ref=receipt,
            source_kind="EXECUTION_RECEIPT",
        )
        stage["attempt_receipt_ref"] = runner.attempt_receipt_reference(
            stage["attempt_receipt"]
        )
        stage["feedback_source"] = {
            "kind": "RANDOM_FEEDBACK_SCHEDULE",
            "source_ref": _digest("random-feedback-source"),
        }
        self.ledger.stage_attempt(stage)
        final = self._judgment(
            task_id=task_id,
            arm="RANDOM_FEEDBACK",
            receipt=stage["attempt_receipt_ref"],
            purpose="qualification",
            phase="adaptation",
        )
        final["objective_judgment"] = None
        final["feedback_record"] = {
            "feedback_source_ref": _digest("random-feedback-source"),
            "learner_visible_scalar": 1.0,
            "outcome": "success",
        }
        final["learner_transition"] = {
            "child_state_digest": _digest("random-feedback-child"),
            "episode_ref": _digest("random-feedback-episode"),
            "parent_state_digest": _digest("random-feedback-parent"),
            "projection_pending": False,
            "sequence": 1,
        }
        final_ref = self.ledger.finalize_attempt(
            (task_id, "RANDOM_FEEDBACK", stage["attempt_receipt_ref"]),
            final,
        )
        self.assertRegex(final_ref, r"^sha256:[0-9a-f]{64}$")
        persisted = self.ledger.read_all()[0]["judgment"]
        self.assertIsNone(persisted["objective_judgment"])
        self.assertEqual(persisted["learner_transition"], final["learner_transition"])

    def test_malformed_random_feedback_is_auditable_zero_not_an_update_bypass(
        self,
    ) -> None:
        def diagnostic_stage(
            *,
            label: str,
            malformed: bool,
        ) -> tuple[dict[str, object], str]:
            task_id = _digest(f"{label}-task")
            source_receipt = _digest(f"{label}-execution-receipt")
            stage = self._stage(
                task_id=task_id,
                arm="RANDOM_FEEDBACK",
                receipt=source_receipt,
                purpose="qualification",
                judge_arm=None,
            )
            stage["phase"] = "adaptation"
            stage["feedback_source"] = {
                "kind": "RANDOM_FEEDBACK_SCHEDULE",
                "source_ref": _digest(f"{label}-schedule"),
            }
            if malformed:
                stage["parser_disposition"] = "MALFORMED"
                stage["proposals"] = []
                stage["selection"] = None
                stage["execution_request"] = None
                stage["execution_receipt"] = None
                stage["task_response_generation"] = None
                stage["raw_response"] = ""
                source_receipt = stage["proposal_generation"]["generation_ref"]
                source_kind = "PROPOSAL_GENERATION"
            else:
                source_kind = "EXECUTION_RECEIPT"
            attempt_receipt = runner.attempt_receipt_payload(
                purpose="qualification",
                phase="adaptation",
                task_id=task_id,
                arm="RANDOM_FEEDBACK",
                parser_disposition=stage["parser_disposition"],
                source_receipt_ref=source_receipt,
                source_kind=source_kind,
            )
            stage["attempt_receipt"] = attempt_receipt
            stage["attempt_receipt_ref"] = runner.attempt_receipt_reference(
                attempt_receipt
            )
            return stage, task_id

        malformed, malformed_task = diagnostic_stage(
            label="malformed-random-zero",
            malformed=True,
        )
        malformed_ledger = runner.EvidenceLedger(
            self.root / "malformed-random-zero" / "attempts.sqlite3",
            self.intent_ref,
        )
        malformed_ledger.stage_attempt(malformed)
        zero = self._judgment(
            task_id=malformed_task,
            arm="RANDOM_FEEDBACK",
            receipt=malformed["attempt_receipt_ref"],
            purpose="qualification",
            phase="adaptation",
        )
        zero["objective_judgment"] = None
        zero["feedback_record"] = None
        zero["learner_transition"] = None
        final_ref = malformed_ledger.finalize_attempt(
            (
                malformed_task,
                "RANDOM_FEEDBACK",
                malformed["attempt_receipt_ref"],
            ),
            zero,
        )
        self.assertRegex(final_ref, r"^sha256:[0-9a-f]{64}$")
        malformed_ledger.audit()
        self.assertEqual(
            malformed_ledger.read_all()[0]["judgment"],
            zero,
        )

        admitted, admitted_task = diagnostic_stage(
            label="admitted-random-missing-update",
            malformed=False,
        )
        admitted_ledger = runner.EvidenceLedger(
            self.root / "admitted-random-missing-update" / "attempts.sqlite3",
            self.intent_ref,
        )
        admitted_ledger.stage_attempt(admitted)
        missing = self._judgment(
            task_id=admitted_task,
            arm="RANDOM_FEEDBACK",
            receipt=admitted["attempt_receipt_ref"],
            purpose="qualification",
            phase="adaptation",
        )
        missing["objective_judgment"] = None
        missing["feedback_record"] = None
        missing["learner_transition"] = None
        with self.assertRaisesRegex(
            ValueError,
            "requires feedback and transition|lacks its learner transition",
        ):
            admitted_ledger.finalize_attempt(
                (
                    admitted_task,
                    "RANDOM_FEEDBACK",
                    admitted["attempt_receipt_ref"],
                ),
                missing,
            )
        self.assertIsNone(admitted_ledger.read_all()[0]["judgment"])
        admitted_ledger.audit()

        laundered, laundered_task = diagnostic_stage(
            label="malformed-random-laundered-update",
            malformed=True,
        )
        laundering_ledger = runner.EvidenceLedger(
            self.root / "malformed-random-laundered-update" / "attempts.sqlite3",
            self.intent_ref,
        )
        laundering_ledger.stage_attempt(laundered)
        laundering = self._judgment(
            task_id=laundered_task,
            arm="RANDOM_FEEDBACK",
            receipt=laundered["attempt_receipt_ref"],
            purpose="qualification",
            phase="adaptation",
        )
        laundering["objective_judgment"] = None
        with self.assertRaisesRegex(ValueError, "cannot launder"):
            laundering_ledger.finalize_attempt(
                (
                    laundered_task,
                    "RANDOM_FEEDBACK",
                    laundered["attempt_receipt_ref"],
                ),
                laundering,
            )
        self.assertIsNone(laundering_ledger.read_all()[0]["judgment"])
        laundering_ledger.audit()


class ManagedRuntimeLifecycleTests(unittest.TestCase):
    def test_exact_scoped_session_mismatch_forgets_and_closes_before_rejection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_scope = runner.AcquisitionScopeSpec(
                dataset_name="source-dataset",
                tenant_name="source-tenant",
                node_set_name="source-nodes",
                state_root=str(root / "source"),
            )
            requested_scope = runner.AcquisitionScopeSpec(
                dataset_name="requested-dataset",
                tenant_name="requested-tenant",
                node_set_name="requested-nodes",
                state_root=str(root / "requested"),
            )

            class Binding:
                def __init__(self) -> None:
                    self.close_calls = 0

                async def close(self) -> None:
                    self.close_calls += 1

            class Backend:
                def __init__(self) -> None:
                    self.forget_calls = 0

                async def project(self, projection: object) -> None:
                    del projection

                async def search(
                    self,
                    query: str,
                    *,
                    limit: int,
                ) -> tuple[object, ...]:
                    del query, limit
                    return ()

                async def forget_namespace(self) -> None:
                    self.forget_calls += 1

            binding = Binding()
            backend = Backend()
            session = runner.ScopedAcquisitionSession(
                source_scope,
                binding,
                backend,
                "source-dataset-id",
            )
            opener_calls = 0

            async def opener(
                scope_spec: runner.AcquisitionScopeSpec,
                *,
                expected_dataset_id: str | None = None,
            ) -> runner.ScopedAcquisitionSession:
                nonlocal opener_calls
                del scope_spec, expected_dataset_id
                opener_calls += 1
                return session

            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "session identity differs",
            ):
                asyncio.run(
                    runner._open_scoped_session(
                        requested_scope,
                        opener=opener,
                        expected_dataset_id="requested-dataset-id",
                    )
                )
            self.assertEqual(opener_calls, 1)
            self.assertEqual(backend.forget_calls, 1)
            self.assertEqual(binding.close_calls, 1)
            self.assertTrue(session._closed)

    def test_primary_failure_cleanup_is_terminal_and_never_reenters_closed_session(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = _FakeOrchestrationEvaluator("evaluation")
            task = owned.tasks["development"][0]
            spec = runner.ArmTaskSpec(
                purpose="evaluation",
                phase="development",
                arm="FULL",
                task=task,
                judge_arm="FULL",
            )
            scope = runner.AcquisitionScopeSpec.for_clone(
                run_label="evaluation",
                replicate=spec.replicate_commitment,
                arm=spec.arm,
                task_id=spec.task_id,
                state_parent=root / "scopes",
            )

            class Session:
                scope_spec = scope
                dataset_id = "managed-lifecycle-dataset"

                def __init__(self) -> None:
                    self.calls = 0

                async def forget_then_close(self, *, opener: object) -> None:
                    del opener
                    self.calls += 1

            class Factory:
                acquisition_opener = object()

                def __init__(self) -> None:
                    self.released: list[str] = []

                def _release_scope(self, key: str) -> None:
                    self.released.append(key)

            class Backend:
                async def project(self, projection: object) -> None:
                    del projection

                async def search(
                    self,
                    query: str,
                    *,
                    limit: int,
                ) -> tuple[object, ...]:
                    del query, limit
                    return ()

                async def forget_namespace(self) -> None:
                    return None

            session = Session()
            factory = Factory()
            provider = runner.CaptureOnceRecallProvider(
                Backend(),
                expected_query="PUBLIC-TASK",
            )
            managed = runner.ManagedArmRuntime(
                factory=factory,  # type: ignore[arg-type]
                spec=spec,
                runtime=object(),  # type: ignore[arg-type]
                session=session,  # type: ignore[arg-type]
                scope_key=scope.state_root,
                recall_provider=provider,
                namespace_rebuild_calls=1,
            )
            ledger = runner.EvidenceLedger(
                root / "evidence" / "attempts.sqlite3",
                _digest("managed-lifecycle-intent"),
            )
            with mock.patch.object(
                runner.CaptureOnceRecallProvider,
                "dispose",
                side_effect=RuntimeError("provider disposal failed"),
            ):
                with self.assertRaisesRegex(
                    BaseExceptionGroup,
                    "managed arm cleanup failed",
                ):
                    asyncio.run(managed.close(result=None, ledger=ledger))
            self.assertEqual(session.calls, 1)
            self.assertEqual(factory.released, [scope.state_root])
            self.assertIsNone(managed.session)
            self.assertIsNone(managed.scope_key)
            self.assertIs(managed.recall_provider, provider)
            self.assertTrue(managed._poisoned)
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "terminally poisoned",
            ):
                asyncio.run(managed.close(result=None, ledger=ledger))
            self.assertEqual(session.calls, 1)


class InjectedOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clone_parent = self.root / "disposable-clones"
        injected_seeds = (
            bytes(range(32)),
            bytes(range(32, 64)),
        )
        self.seed_commitments = tuple(
            evaluator.commit_replicate_seed(seed) for seed in injected_seeds
        )
        committed_evaluator = evaluator.make_evaluation_evaluator(
            injected_seeds,
            self.seed_commitments,
            _digest("fixture-preconstruction-admission"),
        )
        self.evaluator_commitments = committed_evaluator.commitments.to_canonical()
        self.evaluator_commitment_digest = committed_evaluator.commitments.digest
        paths = set(_FROZEN_SOURCE_HASHES) | set(runner.REQUIRED_CONSTRUCTION_SOURCES)
        self.completed_sources = {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in sorted(paths)
        }
        leaf_sha256 = hashlib.sha256(runner.ACTIVE_LEAF_PATH.read_bytes()).hexdigest()
        self.manifest = runner.build_source_manifest(
            evaluator_commitments=self.evaluator_commitments,
            seed_commitments=self.seed_commitments,
            accepted_leaf_sha256=leaf_sha256,
            completed_source_hashes=self.completed_sources,
        )
        self.manifest_sha256 = hashlib.sha256(
            runner.canonical_json_bytes(self.manifest)
        ).hexdigest()
        self.qualification_result_sha256 = hashlib.sha256(
            b"fake-qualified-result"
        ).hexdigest()
        self.admission = runner.admission_digest(
            manifest_sha256=self.manifest_sha256,
            qualification_result_sha256=self.qualification_result_sha256,
            evaluator_commitment_digest=self.evaluator_commitment_digest,
            source_hashes=self.completed_sources,
        )
        self.foundation_tensor_digest = _digest("fixture-foundation-tensor")
        self.foundation_hashes = {
            "foundation_tensor": self.foundation_tensor_digest.removeprefix(
                "sha256:"
            ),
            "model-files": hashlib.sha256(b"fixture-model-files").hexdigest(),
            "tensor-state": hashlib.sha256(b"fixture-tensor-state").hexdigest(),
            "tokenizer": hashlib.sha256(b"fixture-tokenizer").hexdigest(),
        }
        self.foundation_snapshots = [
            dict(self.foundation_hashes),
            dict(self.foundation_hashes),
        ]
        self.foundation_probe_calls = 0
        self.accountings: list[runner.RunAccounting] = []
        self.live_paths = (
            runner.STATE_ROOT,
            runner.SCRATCH_ROOT,
            runner.QUALIFICATION_RESULT_PATH,
            runner.EVALUATION_RESULT_PATH,
        )
        self.live_before = {
            path: _path_signature(path) for path in self.live_paths
        }

    def tearDown(self) -> None:
        self.assertEqual(
            {path: _path_signature(path) for path in self.live_paths},
            self.live_before,
        )

    def _dependencies(
        self,
        owned_evaluator: _FakeOrchestrationEvaluator,
        execute_arm: object,
        *,
        manifest: dict[str, object] | None = None,
        expected_manifest_sha256: str | None = None,
        foundation_probe: object | None = None,
        expected_foundation_hashes: dict[str, str] | None = None,
        expected_learner_genesis_digests: dict[str, str] | None = None,
        repository_root: Path | None = None,
        accounting: runner.RunAccounting | None = None,
        auto_accounting: bool = True,
        auto_integrity: bool = True,
        auto_resource_samples: bool = True,
    ) -> runner.OrchestrationDependencies:
        mode = (
            runner.RunMode.QUALIFICATION
            if owned_evaluator.purpose == "qualification"
            else runner.RunMode.EVALUATION
        )
        selected_accounting = accounting or runner.RunAccounting(mode)
        self.accountings.append(selected_accounting)
        if auto_resource_samples:
            selected_accounting.resources.record_memory(0, 0, 0)
            selected_accounting.resources.record_wall(0.0)
            selected_accounting.resources.record_configured_pools(0)
            selected_accounting.resources.record_state_scratch(0)
        integrity_factory = _FakeProbeIntegrityFactory(
            self.root / f"integrity-run-{len(self.accountings):04d}",
            self.foundation_tensor_digest,
        )

        async def accounted_execute(
            spec: runner.ArmTaskSpec,
        ) -> runner.ArmTaskResult:
            result = execute_arm(spec)  # type: ignore[operator]
            if inspect.isawaitable(result):
                result = await result
            if auto_integrity:
                integrity = integrity_factory(spec, result.parser_disposition)
                result = replace(
                    result,
                    evidence={
                        **result.evidence,
                        "probe_integrity_ref": runner.content_ref(
                            "probe-integrity",
                            integrity.to_canonical(),
                        ),
                    },
                    integrity=integrity,
                )
            if auto_accounting:
                selected_accounting.budget.consume_proposal(
                    spec.task_id,
                    spec.arm,
                    spec.phase,
                )
                selected_accounting.resources.record_generation(1, 1)
                if result.parser_disposition == "ADMITTED":
                    selected_accounting.budget.consume_execution(
                        spec.task_id,
                        spec.arm,
                    )
                    selected_accounting.resources.record_generation(1, 1)
                selected_accounting.budget.finish_arm_task(
                    spec.task_id,
                    spec.arm,
                )
            return result

        if owned_evaluator.purpose == "evaluation":
            owned_evaluator.bind_evaluation_tasks(
                self.evaluator_commitments["tasks"]
            )
            owned_evaluator.commitments = _CanonicalRecord(
                self.evaluator_commitments,
                self.evaluator_commitment_digest,
            )
        return runner.OrchestrationDependencies(
            evaluator=owned_evaluator,
            execute_arm=accounted_execute,
            manifest=self.manifest if manifest is None else manifest,
            expected_manifest_sha256=(
                self.manifest_sha256
                if expected_manifest_sha256 is None
                else expected_manifest_sha256
            ),
            foundation_probe=(
                self._foundation_probe
                if foundation_probe is None
                else foundation_probe
            ),
            expected_foundation_hashes=(
                self.foundation_hashes
                if expected_foundation_hashes is None
                else expected_foundation_hashes
            ),
            expected_foundation_tensor_digest=self.foundation_tensor_digest,
            expected_learner_genesis_digests=(
                {
                    replicate: _FakeProbeIntegrityFactory.genesis_digest(
                        owned_evaluator.purpose,
                        replicate,
                        "FULL",
                    )
                    for replicate in owned_evaluator.replicates
                }
                if expected_learner_genesis_digests is None
                else expected_learner_genesis_digests
            ),
            accounting=selected_accounting,
            repository_root=ROOT if repository_root is None else repository_root,
        )

    def _foundation_probe(self) -> dict[str, str]:
        index = min(self.foundation_probe_calls, len(self.foundation_snapshots) - 1)
        self.foundation_probe_calls += 1
        return dict(self.foundation_snapshots[index])

    def _admission_artifacts(
        self,
        label: str,
    ) -> tuple[Path, Path, Path, Path, Path]:
        root = self.root / label
        root.mkdir(mode=0o700)
        manifest_path = root / "manifest.json"
        seed_path = root / "sealed" / "seeds.json"
        qualification_path = root / "qualification.json"
        result_path = root / "evaluation.json"
        claim_path = root / "sealed" / "admission.json"
        runner.publish_source_manifest(
            manifest_path,
            self.manifest,
            repository_root=ROOT,
        )
        entropy_values = iter((bytes(range(32)), bytes(range(32, 64))))
        runner.seal_evaluation_seeds(
            seed_path,
            entropy=lambda count: next(entropy_values),
            expected_commitments=self.seed_commitments,
        )
        owned = _FakeOrchestrationEvaluator("qualification")
        qualification = asyncio.run(
            runner._run_qualification_injected(
                self._dependencies(
                    owned,
                    _FakeArmExecutor(owned, root / "qualification-clones"),
                )
            )
        )
        runner.publish_qualification_result(
            qualification_path,
            qualification,
            expected_manifest_sha256=self.manifest_sha256,
        )
        return (
            manifest_path,
            seed_path,
            qualification_path,
            result_path,
            claim_path,
        )

    def _assert_integrity_result(
        self,
        result: dict[str, object],
        *,
        purpose: str,
        attempt_count: int,
        clone_count: int,
        removal_count: int,
    ) -> None:
        run_integrity = result["run_integrity"]
        self.assertEqual(
            result["run_integrity_ref"],
            runner.content_ref("run-integrity", run_integrity),
        )
        self.assertEqual(run_integrity["purpose"], purpose)
        self.assertEqual(len(run_integrity["attempts"]), attempt_count)
        self.assertEqual(run_integrity["foundation_guard_checks"], attempt_count)
        for attempt in run_integrity["attempts"]:
            self.assertEqual(
                attempt["probe_integrity_ref"],
                runner.content_ref(
                    "probe-integrity",
                    attempt["probe_integrity"],
                ),
            )
            self.assertEqual(
                attempt["probe_integrity"]["foundation_guard_semantics"],
                runner.FOUNDATION_GUARD_SEMANTICS,
            )

        clone_roots = run_integrity["clone_roots"]
        self.assertEqual(len(clone_roots), clone_count)
        self.assertEqual(len(set(clone_roots)), clone_count)
        self.assertEqual(clone_roots, sorted(clone_roots))

        foundation = result["foundation_integrity"]
        self.assertEqual(run_integrity["foundation"], foundation)
        self.assertEqual(
            result["foundation_integrity_ref"],
            runner.content_ref("foundation-integrity", foundation),
        )
        self.assertEqual(
            run_integrity["foundation_ref"],
            result["foundation_integrity_ref"],
        )
        self.assertEqual(foundation["before"], self.foundation_hashes)
        self.assertEqual(foundation["after"], self.foundation_hashes)
        self.assertEqual(foundation["expected"], self.foundation_hashes)
        self.assertEqual(foundation["tensor_digest"], self.foundation_tensor_digest)
        self.assertEqual(
            foundation["guard_semantics"],
            runner.FOUNDATION_GUARD_SEMANTICS,
        )

        adaptation = result["adaptation_lineage"]
        self.assertEqual(run_integrity["adaptation_lineage"], adaptation)
        self.assertEqual(
            result["adaptation_lineage_ref"],
            runner.content_ref("adaptation-lineages", adaptation),
        )
        expected_lineages = 2 if purpose == "qualification" else 4
        self.assertEqual(len(adaptation["lineages"]), expected_lineages)
        for lineage in adaptation["lineages"]:
            transitions = lineage["transitions"]
            self.assertEqual(len(transitions), 12)
            self.assertEqual(
                transitions[0]["parent_digest"],
                lineage["genesis_digest"],
            )
            previous_child = lineage["genesis_digest"]
            previous_sequence = runner.QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE
            for transition in transitions:
                self.assertEqual(transition["parent_digest"], previous_child)
                if transition["parser_disposition"] == "ADMITTED":
                    previous_sequence += 1
                    self.assertEqual(transition["sequence"], previous_sequence)
                    previous_child = transition["child_digest"]
                else:
                    self.assertIsNone(transition["sequence"])
                    self.assertIsNone(transition["child_digest"])
            self.assertEqual(lineage["final_child_digest"], previous_child)
            if lineage["arm"] == "FULL":
                self.assertNotEqual(previous_child, lineage["genesis_digest"])

        removal_records = run_integrity["removal_fairness"]
        self.assertEqual(len(removal_records), removal_count)
        self.assertEqual(
            result["removal_fairness_refs"],
            [record["evidence_ref"] for record in removal_records],
        )
        for record in removal_records:
            self.assertEqual(
                record["evidence_ref"],
                runner.content_ref(
                    "removal-fairness",
                    {
                        "arms": record["arms"],
                        "require_score": record["require_score"],
                    },
                ),
            )
            self.assertEqual(
                set(record["arms"]),
                {
                    "BACKEND_REMOVAL",
                    "FROZEN_ORIGIN",
                    "FULL",
                    "PROSPECTIVE_REMOVAL",
                    "RETRIEVAL_ONLY",
                },
            )
        self.assertEqual(run_integrity["resource_accounting"], result["accounting"])

    def _synthetic_result(
        self,
        spec: runner.ArmTaskSpec,
        receipt: str,
        *,
        judgment_changes: dict[str, object] | None = None,
    ) -> runner.ArmTaskResult:
        raw_response = "PUBLIC-MODEL-RESPONSE"
        judgment = None
        if spec.judge_arm is not None:
            response_payload = {
                "arm": spec.judge_arm,
                "attempt_receipt_ref": receipt,
                "raw_response": raw_response,
                "task_id": spec.task_id,
            }
            judgment = {
                **response_payload,
                "disposition": "UNSUCCESSFUL",
                "response_commitment": runner.evaluator_record_digest(
                    "response",
                    response_payload,
                ),
                "score": 0.0,
            }
            if judgment_changes is not None:
                judgment.update(judgment_changes)
        integrity = _FakeProbeIntegrityFactory(
            self.root / "synthetic-integrity",
            self.foundation_tensor_digest,
        )(spec)
        return runner.ArmTaskResult(
            task_id=spec.task_id,
            arm=spec.arm,
            attempt_receipt_ref=receipt,
            raw_response=raw_response,
            parser_disposition="ADMITTED",
            judgment=judgment,
            evidence={
                **_fair_arm_evidence(spec.task_id, spec.arm),
                "probe_integrity_ref": runner.content_ref(
                    "probe-integrity",
                    integrity.to_canonical(),
                ),
            },
            integrity=integrity,
        )

    def test_final_metric_aggregate_domain_and_pairwise_invariants_are_exact(
        self,
    ) -> None:
        cases: dict[str, dict[str, object]] = {
            "phase": {"phase": "qualification"},
            "arm": {"arm": "QUALIFICATION"},
            "family": {"family": "unknown-family"},
            "pairwise-nonfinite": {"pairwise_agreement_total": float("nan")},
            "pairwise-out-of-range": {"pairwise_agreement_total": 9.0},
            "pairwise-wrong-family": {
                "family": "glyph-machine",
                "pairwise_agreement_total": 0.0,
            },
        }
        baseline = _valid_final_metrics()
        self.assertEqual(runner._validate_final_metrics(baseline), baseline)
        for label, mutation in cases.items():
            with self.subTest(label=label):
                metrics = json.loads(
                    runner.canonical_json_bytes(baseline).decode("utf-8")
                )
                metrics["aggregates"][0].update(mutation)
                with self.assertRaises(ValueError):
                    runner._validate_final_metrics(metrics)

    def test_qualification_executes_exact_31_arm_tasks_without_final_or_live_writes(
        self,
    ) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")
        executor = _FakeArmExecutor(owned, self.clone_parent)
        result = asyncio.run(
            runner._run_qualification_injected(self._dependencies(owned, executor))
        )
        self.assertEqual(result["identity"], runner.QUALIFICATION_IDENTITY)
        self.assertEqual(result["arm_tasks"], 31)
        self.assertEqual(result["generation_attempt_ceiling"], 62)
        self.assertIs(result["scientific_claim"], False)
        self.assertEqual(executor.generation_attempts, 62)
        self.assertEqual(len(executor.specs), 31)
        self.assertEqual(len(set(result["attempt_receipt_refs"])), 31)
        self.assertTrue(all(not path.exists() for path in executor.clone_paths))

        adaptation = tuple(
            spec for spec in executor.specs if spec.phase == "adaptation"
        )
        development = tuple(
            spec for spec in executor.specs if spec.phase == "development"
        )
        self.assertEqual(len(adaptation), 24)
        self.assertEqual(len(development), 7)
        self.assertEqual(
            [spec.arm for spec in adaptation],
            [arm for _ in range(12) for arm in ("FULL", "RANDOM_FEEDBACK")],
        )
        self.assertEqual(
            [spec.arm for spec in development],
            list(evaluator.EVALUATION_ARMS),
        )
        self.assertEqual(
            sum(spec.judge_arm == "QUALIFICATION" for spec in executor.specs),
            13,
        )
        for spec in adaptation:
            if spec.arm == "RANDOM_FEEDBACK":
                self.assertEqual(
                    spec.feedback_value,
                    dict(owned.schedules[spec.replicate_commitment])[spec.task_id],
                )
        self.assertNotIn(("release_phase", "final"), owned.calls)
        self.assertNotIn(("complete_phase", "development"), owned.calls)
        self.assertEqual(owned.calls[-1], ("close", None))
        self.assertTrue(owned.closed)
        self.assertEqual(self.foundation_probe_calls, 2)
        self.assertEqual(result["accounting"], self.accountings[-1].snapshot())
        self.assertEqual(result["accounting"]["attempts"]["execution_attempts"], 31)
        self.assertEqual(result["accounting"]["resources"]["generation_calls"], 62)
        self._assert_integrity_result(
            result,
            purpose="qualification",
            attempt_count=31,
            clone_count=6,
            removal_count=1,
        )
        self.assertEqual(
            runner.validate_qualification_result(
                result,
                expected_manifest_sha256=self.manifest_sha256,
            ),
            result,
        )
        qualification_path = self.root / "qualification-result.json"
        runner.publish_qualification_result(
            qualification_path,
            result,
            expected_manifest_sha256=self.manifest_sha256,
        )
        loaded_qualification, qualification_sha256 = (
            runner.load_qualification_result(
                qualification_path,
                expected_manifest_sha256=self.manifest_sha256,
            )
        )
        self.assertEqual(loaded_qualification, result)
        self.assertEqual(
            qualification_sha256,
            hashlib.sha256(qualification_path.read_bytes()).hexdigest(),
        )
        with self.assertRaises(runner.RunnerInvariantError):
            runner.publish_qualification_result(
                qualification_path,
                result,
                expected_manifest_sha256=self.manifest_sha256,
            )

    def test_evaluation_executes_exact_468_automatically_admitted_arm_tasks(self) -> None:
        owned = _FakeOrchestrationEvaluator("evaluation")
        executor = _FakeArmExecutor(owned, self.clone_parent)

        async def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            return executor(spec)

        result = asyncio.run(
            runner._run_evaluation_injected(
                self._dependencies(owned, execute),
                qualification_result_sha256=self.qualification_result_sha256,
                evaluator_commitment_digest=self.evaluator_commitment_digest,
                expected_admission_digest=self.admission,
            )
        )
        self.assertEqual(result["identity"], runner.EVALUATION_IDENTITY)
        self.assertEqual(result["arm_tasks"], 468)
        self.assertEqual(result["generation_attempt_ceiling"], 936)
        self.assertEqual(
            result["admission_claim"]["admission_digest"],
            self.admission,
        )
        self.assertEqual(
            result["admission_claim_ref"],
            runner.content_ref(
                "evaluation-admission-claim",
                result["admission_claim"],
            ),
        )
        self.assertEqual(executor.generation_attempts, 936)
        self.assertEqual(len(executor.specs), 468)
        self.assertEqual(len(owned.judged_pairs), 468)
        self.assertEqual(len(set(result["attempt_receipt_refs"])), 468)
        self.assertTrue(all(not path.exists() for path in executor.clone_paths))
        self.assertEqual(
            Counter((spec.phase, spec.arm) for spec in executor.specs),
            Counter(
                {
                    **{("adaptation", arm): 24 for arm in ("FULL", "RANDOM_FEEDBACK")},
                    **{("development", arm): 20 for arm in evaluator.EVALUATION_ARMS},
                    **{("final", arm): 40 for arm in evaluator.EVALUATION_ARMS},
                }
            ),
        )
        call_names = tuple(item[0] for item in owned.calls)
        development_complete = owned.calls.index(("complete_phase", "development"))
        admitted = owned.calls.index(("admit_final", self.admission))
        final_release = owned.calls.index(("release_phase", "final"))
        self.assertLess(development_complete, admitted)
        self.assertLess(admitted, final_release)
        self.assertEqual(owned.calls[-1], ("close", None))
        self.assertIn("final_metrics", call_names)
        self.assertTrue(owned.closed)
        self.assertEqual(self.foundation_probe_calls, 2)
        self.assertEqual(result["accounting"], self.accountings[-1].snapshot())
        self.assertEqual(result["accounting"]["attempts"]["execution_attempts"], 468)
        self.assertEqual(result["accounting"]["resources"]["generation_calls"], 936)
        self._assert_integrity_result(
            result,
            purpose="evaluation",
            attempt_count=468,
            clone_count=360,
            removal_count=60,
        )
        canonical_integrity, advanced = runner.validate_serialized_run_integrity(
            result["run_integrity"],
            purpose="evaluation",
        )
        self.assertEqual(canonical_integrity, result["run_integrity"])
        self.assertEqual(advanced, 2)
        classification, classifier, classifier_ref = (
            runner.classify_evaluation_result(
                result["evaluation_metrics"],
                result["run_integrity"],
            )
        )
        self.assertEqual(classification, "NOT_SUPPORTED")
        self.assertEqual(
            classifier_ref,
            runner.content_ref("integer-classifier", classifier),
        )
        self.assertEqual(
            runner._validate_evaluation_result_injected(
                result,
                manifest=self.manifest,
                repository_root=ROOT,
                expected_manifest_sha256=self.manifest_sha256,
                expected_admission_claim_ref=result["admission_claim_ref"],
            ),
            result,
        )
        live_claim, live_claim_ref = runner.build_evaluation_admission_claim(
            manifest_sha256=self.manifest_sha256,
            qualification_result_sha256=self.qualification_result_sha256,
            evaluator_commitment_digest=self.evaluator_commitment_digest,
            source_hashes=self.completed_sources,
            manifest_path=runner.MANIFEST_PATH,
            seed_path=runner.SEALED_SEED_PATH,
            qualification_result_path=runner.QUALIFICATION_RESULT_PATH,
            evaluation_result_path=runner.EVALUATION_RESULT_PATH,
            claim_path=runner.EVALUATION_ADMISSION_CLAIM_PATH,
            repository_root=ROOT,
            seed_seal_ref=_digest("live-validator-seed-seal"),
            path_binding_mode="FROZEN_LIVE",
        )
        live_result = json.loads(
            runner.canonical_json_bytes(result).decode("utf-8")
        )
        live_result["admission_claim"] = live_claim
        live_result["admission_claim_ref"] = live_claim_ref
        self.assertEqual(
            runner.validate_evaluation_result(
                live_result,
                manifest=self.manifest,
                repository_root=ROOT,
                expected_manifest_sha256=self.manifest_sha256,
                expected_admission_claim_ref=live_claim_ref,
            ),
            live_result,
        )

        def mutate_evaluator_binding(claim: dict[str, object]) -> None:
            changed_digest = _digest("another-evaluator-commitment")
            claim["evaluator_commitment_digest"] = changed_digest
            claim["admission_digest"] = runner.admission_digest(
                manifest_sha256=claim["manifest_sha256"],  # type: ignore[arg-type]
                qualification_result_sha256=claim[  # type: ignore[arg-type]
                    "qualification_result_sha256"
                ],
                evaluator_commitment_digest=changed_digest,
                source_hashes=claim["source_hashes"],  # type: ignore[arg-type]
            )

        for label, mutate in (
            (
                "mode",
                lambda claim: claim.__setitem__(
                    "path_binding_mode", "INJECTED_CPU_TEST"
                ),
            ),
            (
                "path",
                lambda claim: claim["paths"].__setitem__(
                    "evaluation_result", str(self.root / "other-result.json")
                ),
            ),
            ("evaluator", mutate_evaluator_binding),
        ):
            with self.subTest(claim_binding=label):
                changed = json.loads(
                    runner.canonical_json_bytes(live_result).decode("utf-8")
                )
                mutate(changed["admission_claim"])
                changed["admission_claim_ref"] = runner.content_ref(
                    "evaluation-admission-claim",
                    changed["admission_claim"],
                )
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_evaluation_result(
                        changed,
                        manifest=self.manifest,
                        repository_root=ROOT,
                    )
        evaluation_path = self.root / "evaluation-result.json"
        with self.assertRaises(runner.RunnerInvariantError):
            runner.publish_evaluation_result(
                evaluation_path,
                result,
                manifest=self.manifest,
                repository_root=ROOT,
                expected_admission_claim_ref=result["admission_claim_ref"],
                admission_claim_path=self.root / "missing-live-claim.json",
            )
        self.assertFalse(evaluation_path.exists())
        reordered = json.loads(
            runner.canonical_json_bytes(result).decode("utf-8")
        )
        qwen_development = [
            index
            for index, row in enumerate(reordered["run_integrity"]["attempts"])
            if row["arm"] == "QWEN_ONLY"
            and row["probe_integrity"]["phase"] == "development"
        ]
        first, second = qwen_development[:2]
        attempts = reordered["run_integrity"]["attempts"]
        attempts[first], attempts[second] = attempts[second], attempts[first]
        reordered["attempt_receipt_refs"] = [
            row["attempt_receipt_ref"] for row in attempts
        ]
        reordered["run_integrity_ref"] = runner.content_ref(
            "run-integrity",
            reordered["run_integrity"],
        )
        classification, classifier, classifier_ref = (
            runner.classify_evaluation_result(
                reordered["evaluation_metrics"],
                reordered["run_integrity"],
            )
        )
        reordered["classification"] = classification
        reordered["classifier"] = classifier
        reordered["classifier_ref"] = classifier_ref
        with self.assertRaisesRegex(
            runner.RunnerInvariantError,
            "schedule differs from manifest",
        ):
            runner._validate_evaluation_result_injected(
                reordered,
                manifest=self.manifest,
                repository_root=ROOT,
            )

    def test_integer_classifier_edges_and_structural_invalid_precedence(self) -> None:
        owned = _FakeOrchestrationEvaluator("evaluation")
        executor = _FakeArmExecutor(owned, self.clone_parent)
        result = asyncio.run(
            runner._run_evaluation_injected(
                self._dependencies(owned, executor),
                qualification_result_sha256=self.qualification_result_sha256,
                evaluator_commitment_digest=self.evaluator_commitment_digest,
                expected_admission_digest=self.admission,
            )
        )
        integrity = result["run_integrity"]
        support_scores = {
            "FULL": (5, 5, 8),
            "QWEN_ONLY": (3, 3, 6),
            "RETRIEVAL_ONLY": (3, 3, 6),
            "RANDOM_FEEDBACK": (4, 4, 8),
            "PROSPECTIVE_REMOVAL": (4, 4, 8),
            "FROZEN_ORIGIN": (4, 5, 8),
            "BACKEND_REMOVAL": (5, 5, 8),
        }

        def classify(
            scores: dict[str, tuple[int, int, int]],
            evidence: dict[str, object] = integrity,
        ) -> tuple[str, dict[str, object]]:
            classification, witness, _ = runner.classify_evaluation_result(
                _final_metrics_with_scores(scores),
                evidence,
            )
            return classification, witness

        classification, witness = classify(support_scores)
        self.assertEqual(classification, "EXPERIMENTALLY_SUPPORTED")
        self.assertTrue(witness["all_predicates_passed"])

        p1_boundary = {
            **support_scores,
            "FULL": (4, 3, 7),
            "QWEN_ONLY": (2, 1, 5),
            "RETRIEVAL_ONLY": (2, 1, 5),
            "RANDOM_FEEDBACK": (3, 3, 6),
            "PROSPECTIVE_REMOVAL": (3, 3, 6),
            "FROZEN_ORIGIN": (3, 3, 7),
        }
        self.assertEqual(classify(p1_boundary)[0], "EXPERIMENTALLY_SUPPORTED")
        p1_fail = {**p1_boundary, "FULL": (4, 3, 6)}
        failed, failed_witness = classify(p1_fail)
        self.assertEqual(failed, "NOT_SUPPORTED")
        self.assertFalse(failed_witness["predicates"]["p1_full_at_least_14"])

        boundary_mutations = {
            "p2_full_minus_qwen_at_least_3": (
                {**support_scores, "QWEN_ONLY": (4, 4, 7)},
                {**support_scores, "QWEN_ONLY": (4, 4, 8)},
            ),
            "p3_full_minus_retrieval_at_least_3": (
                {**support_scores, "RETRIEVAL_ONLY": (4, 4, 7)},
                {**support_scores, "RETRIEVAL_ONLY": (4, 4, 8)},
            ),
            "p4_full_minus_random_at_least_2": (
                support_scores,
                {**support_scores, "RANDOM_FEEDBACK": (5, 4, 8)},
            ),
            "p5_full_minus_prospective_at_least_2": (
                support_scores,
                {**support_scores, "PROSPECTIVE_REMOVAL": (5, 4, 8)},
            ),
            "p6_full_minus_frozen_origin_at_least_1": (
                support_scores,
                {**support_scores, "FROZEN_ORIGIN": (5, 5, 8)},
            ),
        }
        for predicate, (passing, failing) in boundary_mutations.items():
            with self.subTest(predicate=predicate):
                self.assertTrue(classify(passing)[1]["predicates"][predicate])
                classification, edge = classify(failing)
                self.assertEqual(classification, "NOT_SUPPORTED")
                self.assertFalse(edge["predicates"][predicate])

        p7_fail = {
            **support_scores,
            "QWEN_ONLY": (5, 5, 5),
            "RETRIEVAL_ONLY": (1, 1, 4),
        }
        p7_classification, p7_witness = classify(p7_fail)
        self.assertEqual(p7_classification, "NOT_SUPPORTED")
        self.assertFalse(p7_witness["predicates"]["p7_family_control_margin"])

        rational_full = (6, 6, 12)
        p7_integer_cases = {
            "symbolic-deficit-one-allowed": ((7, 5, 10), True),
            "glyph-deficit-one-allowed": ((5, 7, 10), True),
            "causal-deficit-three-allowed": ((5, 5, 15), True),
            "symbolic-deficit-two-rejected": ((8, 5, 10), False),
            "glyph-deficit-two-rejected": ((5, 8, 10), False),
            "causal-deficit-four-rejected": ((5, 5, 16), False),
        }
        for label, (better_control, expected) in p7_integer_cases.items():
            with self.subTest(p7_integer_case=label):
                scores = {
                    **support_scores,
                    "FULL": rational_full,
                    "QWEN_ONLY": better_control,
                    "RETRIEVAL_ONLY": (1, 1, 1),
                }
                observed = classify(scores)[1]["predicates"][
                    "p7_family_control_margin"
                ]
                self.assertIs(observed, expected)
        equality_scores = {
            **support_scores,
            "FULL": rational_full,
            "QWEN_ONLY": (6, 5, 10),
            "RETRIEVAL_ONLY": (1, 1, 1),
        }
        equality_witness = classify(equality_scores)[1]
        symbolic_witness = next(
            row
            for row in equality_witness["family_witnesses"]
            if row["family"] == "symbolic-demonstration-transfer"
        )
        self.assertFalse(symbolic_witness["full_strictly_better"])
        self.assertTrue(symbolic_witness["within_one_eighth_floor"])

        def canonical_copy(value: object) -> dict[str, object]:
            return json.loads(runner.canonical_json_bytes(value).decode("utf-8"))

        phase_shift = canonical_copy(integrity)
        shifted = next(
            row
            for row in phase_shift["attempts"]
            if row["probe_integrity"]["phase"] == "development"
            and row["arm"] == "QWEN_ONLY"
        )
        shifted["probe_integrity"]["phase"] = "final"
        shifted["probe_integrity_ref"] = runner.content_ref(
            "probe-integrity",
            shifted["probe_integrity"],
        )
        with self.assertRaises(runner.RunnerInvariantError):
            classify(support_scores, phase_shift)

        disjoint_arm_task = canonical_copy(integrity)
        disjoint = next(
            row
            for row in disjoint_arm_task["attempts"]
            if row["probe_integrity"]["phase"] == "adaptation"
            and row["arm"] == "RANDOM_FEEDBACK"
        )
        disjoint["task_id"] = _digest("disjoint-random-feedback-task")
        disjoint["probe_integrity"]["task_id"] = disjoint["task_id"]
        disjoint["probe_integrity_ref"] = runner.content_ref(
            "probe-integrity",
            disjoint["probe_integrity"],
        )
        with self.assertRaises(runner.RunnerInvariantError):
            classify(support_scores, disjoint_arm_task)

        genesis_swap = canonical_copy(integrity)
        stateful_probe = next(
            row
            for row in genesis_swap["attempts"]
            if row["probe_integrity"]["phase"] == "development"
            and row["arm"] == "FULL"
        )
        stateful_probe["probe_integrity"]["learner_genesis_digest"] = _digest(
            "substituted-stateful-genesis"
        )
        stateful_probe["probe_integrity_ref"] = runner.content_ref(
            "probe-integrity",
            stateful_probe["probe_integrity"],
        )
        with self.assertRaises(runner.RunnerInvariantError):
            classify(support_scores, genesis_swap)

        tensor_swap = canonical_copy(integrity)
        tensor_attempt = tensor_swap["attempts"][0]
        tensor_attempt["probe_integrity"]["foundation_tensor_digest"] = _digest(
            "substituted-foundation"
        )
        tensor_attempt["probe_integrity_ref"] = runner.content_ref(
            "probe-integrity",
            tensor_attempt["probe_integrity"],
        )
        with self.assertRaises(runner.RunnerInvariantError):
            classify(support_scores, tensor_swap)

        removal_swap = canonical_copy(integrity)
        removal = removal_swap["removal_fairness"][0]
        substituted_task = _digest("substituted-removal-task")
        removal["task_id"] = substituted_task
        for arm in removal["arms"].values():
            arm["task_id"] = substituted_task
        removal["evidence_ref"] = runner.content_ref(
            "removal-fairness",
            {
                "arms": removal["arms"],
                "require_score": removal["require_score"],
            },
        )
        removal_swap["removal_fairness_refs"][0] = removal["evidence_ref"]
        with self.assertRaises(runner.RunnerInvariantError):
            classify(support_scores, removal_swap)

        no_advance = canonical_copy(integrity)
        full_rows = {
            row["replicate_commitment"]: row
            for row in no_advance["adaptation_lineage"]["lineages"]
            if row["arm"] == "FULL"
        }
        for replicate, row in full_rows.items():
            genesis = row["genesis_digest"]
            row["advanced"] = False
            row["final_child_digest"] = genesis
            for transition in row["transitions"]:
                transition.update(
                    {
                        "child_digest": None,
                        "parent_digest": genesis,
                        "parser_disposition": "MALFORMED",
                        "sequence": None,
                    }
                )
                attempt = next(
                    item
                    for item in no_advance["attempts"]
                    if item["arm"] == "FULL"
                    and item["task_id"] == transition["task_id"]
                )
                attempt["parser_disposition"] = "MALFORMED"
                attempt["probe_integrity"].update(
                    {
                        "learner_child_digest": None,
                        "learner_parent_digest": genesis,
                        "learner_sequence": None,
                    }
                )
                attempt["probe_integrity_ref"] = runner.content_ref(
                    "probe-integrity",
                    attempt["probe_integrity"],
                )
        no_advance["adaptation_lineage_ref"] = runner.content_ref(
            "adaptation-lineages",
            no_advance["adaptation_lineage"],
        )
        no_advance["full_adaptation_lineage"] = {
            "all_replicates_advanced": False,
            "expected_genesis_by_replicate": no_advance[
                "adaptation_lineage"
            ]["expected_genesis_by_replicate"],
            "replicates": [
                {name: value for name, value in row.items() if name != "arm"}
                for row in no_advance["adaptation_lineage"]["lineages"]
                if row["arm"] == "FULL"
            ],
            "schema": runner.LINEAGE_INTEGRITY_SCHEMA,
        }
        no_advance["full_adaptation_lineage_ref"] = runner.content_ref(
            "full-adaptation-lineages",
            no_advance["full_adaptation_lineage"],
        )
        no_advance_classification, no_advance_witness = classify(
            support_scores,
            no_advance,
        )
        self.assertEqual(no_advance_classification, "NOT_SUPPORTED")
        self.assertFalse(
            no_advance_witness["predicates"][
                "p9_two_advanced_full_lineages"
            ]
        )
        reset = canonical_copy(no_advance)
        reset["adaptation_lineage"]["lineages"][0]["transitions"][1][
            "parent_digest"
        ] = _digest("structural-reset")
        with self.assertRaises(runner.RunnerInvariantError):
            classify(support_scores, reset)

    def test_durable_admission_validates_pass_before_consuming_once(self) -> None:
        (
            manifest_path,
            seed_path,
            qualification_path,
            result_path,
            claim_path,
        ) = self._admission_artifacts("admission-success")
        handshake_calls = 0

        def handshake() -> dict[str, object]:
            nonlocal handshake_calls
            handshake_calls += 1
            self.assertFalse(claim_path.exists())
            return {
                "commitments": self.evaluator_commitments,
                "digest": self.evaluator_commitment_digest,
            }

        permit = asyncio.run(
            runner._claim_evaluation_admission(
                manifest_path=manifest_path,
                seed_path=seed_path,
                qualification_result_path=qualification_path,
                evaluation_result_path=result_path,
                claim_path=claim_path,
                repository_root=ROOT,
                evaluator_handshake=handshake,
                injected_paths=True,
            )
        )
        receipt = permit.receipt
        self.assertEqual(handshake_calls, 1)
        self.assertTrue(claim_path.is_file())
        self.assertEqual(stat.S_IMODE(claim_path.stat().st_mode), 0o600)
        self.assertEqual(
            receipt["claim_ref"],
            runner.content_ref("evaluation-admission-claim", receipt["claim"]),
        )
        self.assertEqual(
            receipt["claim"]["qualification_result_sha256"],
            hashlib.sha256(qualification_path.read_bytes()).hexdigest(),
        )
        self.assertFalse(result_path.exists())
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._claim_evaluation_admission(
                    manifest_path=manifest_path,
                    seed_path=seed_path,
                    qualification_result_path=qualification_path,
                    evaluation_result_path=result_path,
                    claim_path=claim_path,
                    repository_root=ROOT,
                    evaluator_handshake=handshake,
                    injected_paths=True,
                )
            )
        self.assertEqual(handshake_calls, 1)

        invalid_root = self.root / "invalid-qualification"
        invalid_root.mkdir(mode=0o700)
        invalid_manifest = invalid_root / "manifest.json"
        invalid_seed = invalid_root / "seeds.json"
        invalid_qualification = invalid_root / "qualification.json"
        invalid_claim = invalid_root / "claim.json"
        runner.publish_source_manifest(
            invalid_manifest,
            self.manifest,
            repository_root=ROOT,
        )
        entropy_values = iter((bytes(range(32)), bytes(range(32, 64))))
        runner.seal_evaluation_seeds(
            invalid_seed,
            entropy=lambda count: next(entropy_values),
            expected_commitments=self.seed_commitments,
        )
        invalid_qualification.write_bytes(
            runner.canonical_json_bytes(
                {"classification": "QUALIFICATION_PASS"}
            )
        )
        invalid_qualification.chmod(0o600)
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._claim_evaluation_admission(
                    manifest_path=invalid_manifest,
                    seed_path=invalid_seed,
                    qualification_result_path=invalid_qualification,
                    evaluation_result_path=invalid_root / "result.json",
                    claim_path=invalid_claim,
                    repository_root=ROOT,
                    evaluator_handshake=handshake,
                    injected_paths=True,
                )
            )
        self.assertFalse(invalid_claim.exists())

    def test_admission_fsync_uncertainty_consumes_without_release(self) -> None:
        (
            manifest_path,
            seed_path,
            qualification_path,
            result_path,
            claim_path,
        ) = self._admission_artifacts("admission-fsync")
        handshake_calls = 0

        def handshake() -> dict[str, object]:
            nonlocal handshake_calls
            handshake_calls += 1
            return {
                "commitments": self.evaluator_commitments,
                "digest": self.evaluator_commitment_digest,
            }

        real_fsync_directory = runner._fsync_directory

        def fail_claim_parent(path: Path) -> None:
            if path == claim_path.parent:
                raise OSError("INJECTED_CLAIM_DIRECTORY_FSYNC_FAILURE")
            real_fsync_directory(path)

        with mock.patch.object(
            runner,
            "_fsync_directory",
            side_effect=fail_claim_parent,
        ):
            with self.assertRaises(runner.AdmissionClaimPublicationFailure):
                asyncio.run(
                    runner._claim_evaluation_admission(
                        manifest_path=manifest_path,
                        seed_path=seed_path,
                        qualification_result_path=qualification_path,
                        evaluation_result_path=result_path,
                        claim_path=claim_path,
                        repository_root=ROOT,
                        evaluator_handshake=handshake,
                        injected_paths=True,
                    )
                )
        self.assertTrue(claim_path.exists())
        self.assertFalse(result_path.exists())
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._claim_evaluation_admission(
                    manifest_path=manifest_path,
                    seed_path=seed_path,
                    qualification_result_path=qualification_path,
                    evaluation_result_path=result_path,
                    claim_path=claim_path,
                    repository_root=ROOT,
                    evaluator_handshake=handshake,
                    injected_paths=True,
                )
            )
        self.assertEqual(handshake_calls, 1)

    def test_post_o_excl_permit_failure_is_consumed_owner_failure(self) -> None:
        (
            manifest_path,
            seed_path,
            qualification_path,
            result_path,
            claim_path,
        ) = self._admission_artifacts("admission-permit-memory")

        def handshake() -> dict[str, object]:
            return {
                "commitments": self.evaluator_commitments,
                "digest": self.evaluator_commitment_digest,
            }

        with mock.patch.object(
            runner,
            "_EvaluationAdmissionPermit",
            side_effect=MemoryError("PRIVATE_POST_CLAIM_DETAIL"),
        ):
            with self.assertRaises(runner.AdmissionClaimPublicationFailure):
                asyncio.run(
                    runner._claim_evaluation_admission(
                        manifest_path=manifest_path,
                        seed_path=seed_path,
                        qualification_result_path=qualification_path,
                        evaluation_result_path=result_path,
                        claim_path=claim_path,
                        repository_root=ROOT,
                        evaluator_handshake=handshake,
                        injected_paths=True,
                    )
                )
        self.assertTrue(claim_path.is_file())
        self.assertFalse(result_path.exists())
        with self.assertRaises(runner.EvaluationIdentityConsumed):
            asyncio.run(
                runner._claim_evaluation_admission(
                    manifest_path=manifest_path,
                    seed_path=seed_path,
                    qualification_result_path=qualification_path,
                    evaluation_result_path=result_path,
                    claim_path=claim_path,
                    repository_root=ROOT,
                    evaluator_handshake=handshake,
                    injected_paths=True,
                )
            )

    def test_qualification_coordinator_preflights_reserves_and_publishes_once(
        self,
    ) -> None:
        root = self.root / "qualification-coordinator"
        root.mkdir(mode=0o700)
        manifest_path = root / "manifest.json"
        result_path = root / "qualification.json"
        release_path = root / "sealed" / "qualification-release.json"
        runner.publish_source_manifest(
            manifest_path,
            self.manifest,
            repository_root=ROOT,
        )
        owned = _FakeOrchestrationEvaluator("qualification")
        executor = _FakeArmExecutor(owned, root / "clones")
        cleanup_calls = 0
        transferred_cleanup_calls = 0
        factory_calls = 0

        def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        def factory(payload: dict[str, object]) -> runner._LiveDependenciesOwner:
            nonlocal factory_calls, transferred_cleanup_calls
            factory_calls += 1
            self.assertEqual(payload["purpose"], "qualification")
            self.assertIsInstance(
                payload["partial_cleanup"],
                runner._PartialConstructionCleanup,
            )

            def transferred_cleanup() -> None:
                nonlocal transferred_cleanup_calls
                transferred_cleanup_calls += 1

            payload["partial_cleanup"].register(  # type: ignore[union-attr]
                "TRANSFERRED_OWNER_CLEANUP_FAILED",
                transferred_cleanup,
            )
            return runner._LiveDependenciesOwner(
                self._dependencies(owned, executor),
                cleanup,
                "QUALIFICATION_OWNER_CLEANUP_FAILED",
            )

        async def concurrent_qualification() -> dict[str, object]:
            entered = asyncio.Event()
            release = asyncio.Event()

            async def blocking_factory(
                payload: dict[str, object],
            ) -> runner._LiveDependenciesOwner:
                owner = factory(payload)
                entered.set()
                await release.wait()
                return owner

            winner = asyncio.create_task(
                runner._run_and_publish_qualification_once(
                    blocking_factory,
                    manifest_path=manifest_path,
                    result_path=result_path,
                    release_claim_path=release_path,
                    repository_root=ROOT,
                    injected_paths=True,
                )
            )
            await entered.wait()
            with self.assertRaises(runner.RunnerInvariantError):
                await runner._run_and_publish_qualification_once(
                    blocking_factory,
                    manifest_path=manifest_path,
                    result_path=result_path,
                    release_claim_path=release_path,
                    repository_root=ROOT,
                    injected_paths=True,
                )
            self.assertFalse(result_path.exists())
            release.set()
            return await winner

        result = asyncio.run(concurrent_qualification())
        self.assertEqual(result["classification"], "QUALIFICATION_PASS")
        self.assertEqual(factory_calls, 1)
        self.assertEqual(cleanup_calls, 1)
        self.assertEqual(transferred_cleanup_calls, 1)
        release, _ = runner._load_qualification_release_claim(release_path)
        self.assertEqual(release["path_binding_mode"], "INJECTED_CPU_TEST")
        self.assertEqual(release["paths"]["result"], str(result_path))
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_and_publish_qualification_once(
                    factory,
                    manifest_path=manifest_path,
                    result_path=result_path,
                    release_claim_path=release_path,
                    repository_root=ROOT,
                    injected_paths=True,
                )
            )
        self.assertEqual(factory_calls, 1)
        self.assertEqual(transferred_cleanup_calls, 1)

        occupied_root = self.root / "qualification-occupied"
        occupied_root.mkdir(mode=0o700)
        occupied_manifest = occupied_root / "manifest.json"
        occupied_result = occupied_root / "qualification.json"
        runner.publish_source_manifest(
            occupied_manifest,
            self.manifest,
            repository_root=ROOT,
        )
        occupied_result.write_bytes(b"already-consumed")
        occupied_result.chmod(0o600)
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_and_publish_qualification_once(
                    factory,
                    manifest_path=occupied_manifest,
                    result_path=occupied_result,
                    release_claim_path=occupied_root / "release.json",
                    repository_root=ROOT,
                    injected_paths=True,
                )
            )
        self.assertEqual(factory_calls, 1)

    def test_admitted_factory_infrastructure_is_typed_and_rolls_back_once(
        self,
    ) -> None:
        error_types = (
            OSError,
            MemoryError,
            type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"}),
        )
        for index, error_type in enumerate(error_types):
            with self.subTest(error_type=error_type.__name__):
                (
                    manifest_path,
                    seed_path,
                    qualification_path,
                    result_path,
                    claim_path,
                ) = self._admission_artifacts(f"typed-infrastructure-{index}")

                def handshake() -> dict[str, object]:
                    return {
                        "commitments": self.evaluator_commitments,
                        "digest": self.evaluator_commitment_digest,
                    }

                permit = asyncio.run(
                    runner._claim_evaluation_admission(
                        manifest_path=manifest_path,
                        seed_path=seed_path,
                        qualification_result_path=qualification_path,
                        evaluation_result_path=result_path,
                        claim_path=claim_path,
                        repository_root=ROOT,
                        evaluator_handshake=handshake,
                        injected_paths=True,
                    )
                )
                rollback_calls = 0

                def factory(payload: dict[str, object]) -> object:
                    nonlocal rollback_calls

                    def rollback() -> None:
                        nonlocal rollback_calls
                        rollback_calls += 1

                    payload["partial_cleanup"].register(  # type: ignore[union-attr]
                        "PARTIAL_FACTORY_CLEANUP_FAILED",
                        rollback,
                    )
                    raise error_type("PRIVATE_INFRASTRUCTURE_DETAIL")

                with self.assertRaises(runner.InfrastructureFailure) as captured:
                    asyncio.run(
                        runner._run_admitted_evaluation(
                            factory,
                            admission_permit=permit,
                            allow_injected_paths=True,
                        )
                    )
                self.assertEqual(rollback_calls, 1)
                classification, failure = runner.classify_exception_disposition(
                    captured.exception,
                    admission_validated=True,
                    evaluator_cleanup_completed=True,
                )
                self.assertEqual(classification, "INCONCLUSIVE")
                self.assertEqual(failure["category"], "INFRASTRUCTURE")
                with self.assertRaises(runner.RunnerInvariantError):
                    permit.consume()

        (
            manifest_path,
            seed_path,
            qualification_path,
            result_path,
            claim_path,
        ) = self._admission_artifacts("typed-runtime-infrastructure")

        def handshake() -> dict[str, object]:
            return {
                "commitments": self.evaluator_commitments,
                "digest": self.evaluator_commitment_digest,
            }

        permit = asyncio.run(
            runner._claim_evaluation_admission(
                manifest_path=manifest_path,
                seed_path=seed_path,
                qualification_result_path=qualification_path,
                evaluation_result_path=result_path,
                claim_path=claim_path,
                repository_root=ROOT,
                evaluator_handshake=handshake,
                injected_paths=True,
            )
        )
        owned = _FakeOrchestrationEvaluator("evaluation")
        executor = _FakeArmExecutor(owned, self.root / "typed-runtime-clones")
        dependencies = self._dependencies(owned, executor)

        def fail_release(phase: str) -> tuple[dict[str, object], ...]:
            raise OSError("PRIVATE_RUNTIME_IO_DETAIL")

        owned.release_phase = fail_release  # type: ignore[method-assign]
        cleanup_calls = 0

        def owner_factory(
            payload: dict[str, object],
        ) -> runner._LiveDependenciesOwner:
            def cleanup() -> None:
                nonlocal cleanup_calls
                cleanup_calls += 1

            return runner._LiveDependenciesOwner(
                dependencies,
                cleanup,
                "EVALUATION_OWNER_CLEANUP_FAILED",
            )

        with self.assertRaises(runner.InfrastructureFailure) as captured:
            asyncio.run(
                runner._run_admitted_evaluation(
                    owner_factory,
                    admission_permit=permit,
                    allow_injected_paths=True,
                )
            )
        self.assertEqual(cleanup_calls, 1)
        classification, failure = runner.classify_exception_disposition(
            captured.exception,
            admission_validated=True,
            evaluator_cleanup_completed=True,
        )
        self.assertEqual(classification, "INCONCLUSIVE")
        self.assertEqual(failure["category"], "INFRASTRUCTURE")

    def test_two_live_coordinators_release_exactly_one_and_loser_cannot_publish(
        self,
    ) -> None:
        root = self.root / "patched-live-coordinator"
        state_root = root / "state"
        scratch_root = root / "scratch"
        result_root = root / "results"
        manifest_path = root / "manifest.json"
        qualification_path = result_root / "qualification.json"
        evaluation_path = result_root / "evaluation.json"
        seed_path = state_root / "sealed" / "seeds.json"
        claim_path = state_root / "sealed" / "evaluation-claim.json"
        qualification_release = (
            state_root / "sealed" / "qualification-release.json"
        )
        for directory in (root, state_root, scratch_root, result_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        frozen_paths = {
            **runner.FROZEN_PATHS,
            "evaluation_admission_claim": str(claim_path),
            "evaluation_result": str(evaluation_path),
            "manifest": str(manifest_path),
            "qualification_release_claim": str(qualification_release),
            "qualification_result": str(qualification_path),
            "scratch_root": str(scratch_root),
            "sealed_seed": str(seed_path),
            "state_root": str(state_root),
        }
        with mock.patch.multiple(
            runner,
            EVALUATION_ADMISSION_CLAIM_PATH=claim_path,
            EVALUATION_RESULT_PATH=evaluation_path,
            FROZEN_PATHS=frozen_paths,
            MANIFEST_PATH=manifest_path,
            QUALIFICATION_RELEASE_CLAIM_PATH=qualification_release,
            QUALIFICATION_RESULT_PATH=qualification_path,
            SCRATCH_ROOT=scratch_root,
            SEALED_SEED_PATH=seed_path,
            STATE_ROOT=state_root,
        ):
            leaf_sha256 = hashlib.sha256(
                runner.ACTIVE_LEAF_PATH.read_bytes()
            ).hexdigest()
            manifest = runner.build_source_manifest(
                evaluator_commitments=self.evaluator_commitments,
                seed_commitments=self.seed_commitments,
                accepted_leaf_sha256=leaf_sha256,
                completed_source_hashes=runner.completed_source_manifest(ROOT),
            )
            manifest_sha256 = hashlib.sha256(
                runner.canonical_json_bytes(manifest)
            ).hexdigest()
            runner.publish_source_manifest(
                manifest_path,
                manifest,
                repository_root=ROOT,
            )
            entropy_values = iter((bytes(range(32)), bytes(range(32, 64))))
            runner.seal_evaluation_seeds(
                seed_path,
                entropy=lambda count: next(entropy_values),
                expected_commitments=self.seed_commitments,
            )
            qualification_evaluator = _FakeOrchestrationEvaluator("qualification")
            qualification = asyncio.run(
                runner._run_qualification_injected(
                    self._dependencies(
                        qualification_evaluator,
                        _FakeArmExecutor(
                            qualification_evaluator,
                            root / "qualification-clones",
                        ),
                        manifest=manifest,
                        expected_manifest_sha256=manifest_sha256,
                    )
                )
            )
            runner.publish_qualification_result(
                qualification_path,
                qualification,
                expected_manifest_sha256=manifest_sha256,
            )
            release_claim, _ = runner._build_qualification_release_claim(
                manifest_sha256=manifest_sha256,
                manifest_path=manifest_path,
                result_path=qualification_path,
                claim_path=qualification_release,
                repository_root=ROOT,
                path_binding_mode="FROZEN_LIVE",
            )
            runner._publish_qualification_release_claim(
                qualification_release,
                release_claim,
            )

            async def scenario() -> tuple[dict[str, object], int, int, int]:
                entered = asyncio.Event()
                release = asyncio.Event()
                factory_calls = 0
                cleanup_calls = 0
                handshake_calls = 0

                def handshake() -> dict[str, object]:
                    nonlocal handshake_calls
                    handshake_calls += 1
                    return {
                        "commitments": self.evaluator_commitments,
                        "digest": self.evaluator_commitment_digest,
                    }

                async def factory(
                    payload: dict[str, object],
                ) -> runner._LiveDependenciesOwner:
                    nonlocal factory_calls, cleanup_calls
                    factory_calls += 1
                    owned = _FakeOrchestrationEvaluator("evaluation")
                    executor = _FakeArmExecutor(
                        owned,
                        root / f"evaluation-clones-{factory_calls}",
                    )

                    def cleanup() -> None:
                        nonlocal cleanup_calls
                        cleanup_calls += 1

                    dependencies = self._dependencies(
                        owned,
                        executor,
                        manifest=payload["manifest"],  # type: ignore[arg-type]
                        expected_manifest_sha256=payload["manifest_sha256"],  # type: ignore[arg-type]
                    )
                    entered.set()
                    await release.wait()
                    return runner._LiveDependenciesOwner(
                        dependencies,
                        cleanup,
                        "EVALUATION_OWNER_CLEANUP_FAILED",
                    )

                winner = asyncio.create_task(
                    runner._run_and_publish_evaluation_once(
                        factory,
                        evaluator_handshake=handshake,
                    )
                )
                await entered.wait()
                with self.assertRaises(runner.EvaluationIdentityConsumed):
                    await runner._run_and_publish_evaluation_once(
                        factory,
                        evaluator_handshake=handshake,
                    )
                self.assertFalse(evaluation_path.exists())
                release.set()
                result = await winner
                return result, factory_calls, cleanup_calls, handshake_calls

            real_require_identity_absent = runner._require_identity_absent

            def force_collision_at_o_excl(
                path: Path,
                error_type: type[runner.RunnerInvariantError],
            ) -> None:
                if Path(path) == claim_path:
                    return
                real_require_identity_absent(path, error_type)

            with (
                mock.patch.object(
                    runner,
                    "_require_identity_absent",
                    side_effect=force_collision_at_o_excl,
                ),
                mock.patch.object(
                    runner,
                    "_write_consuming_admission_claim",
                    wraps=runner._write_consuming_admission_claim,
                ) as claim_writer,
                mock.patch.object(
                    runner,
                    "publish_evaluation_failure_result",
                    wraps=runner.publish_evaluation_failure_result,
                ) as failure_publisher,
            ):
                result, factory_calls, cleanup_calls, handshake_calls = asyncio.run(
                    scenario()
                )
            self.assertEqual(claim_writer.call_count, 2)
            self.assertEqual(failure_publisher.call_count, 0)
            self.assertIn(
                result["classification"],
                ("EXPERIMENTALLY_SUPPORTED", "NOT_SUPPORTED"),
            )
            self.assertEqual(factory_calls, 1)
            self.assertEqual(cleanup_calls, 1)
            self.assertEqual(handshake_calls, 2)
            self.assertTrue(claim_path.is_file())
            self.assertTrue(evaluation_path.is_file())
            self.assertFalse(
                evaluation_path.with_name(evaluation_path.name + ".tmp").exists()
            )

    def test_qualification_revalidates_sources_and_never_certifies_fsync_uncertainty(
        self,
    ) -> None:
        drift_root = self.root / "qualification-source-drift"
        drift_root.mkdir(mode=0o700)
        drift_manifest = drift_root / "manifest.json"
        drift_result = drift_root / "qualification.json"
        drift_release = drift_root / "sealed" / "release.json"
        runner.publish_source_manifest(
            drift_manifest,
            self.manifest,
            repository_root=ROOT,
        )
        drift_evaluator = _FakeOrchestrationEvaluator("qualification")
        drift_executor = _FakeArmExecutor(
            drift_evaluator,
            drift_root / "clones",
        )
        cleanup_calls = 0

        def execute_with_drift(
            spec: runner.ArmTaskSpec,
        ) -> runner.ArmTaskResult:
            result = drift_executor(spec)
            if (
                spec.phase == "development"
                and spec.arm == evaluator.EVALUATION_ARMS[-1]
            ):
                drift_manifest.write_bytes(b"{}")
                drift_manifest.chmod(0o600)
            return result

        def drift_factory(
            payload: dict[str, object],
        ) -> runner._LiveDependenciesOwner:
            nonlocal cleanup_calls

            def cleanup() -> None:
                nonlocal cleanup_calls
                cleanup_calls += 1

            return runner._LiveDependenciesOwner(
                self._dependencies(drift_evaluator, execute_with_drift),
                cleanup,
                "QUALIFICATION_OWNER_CLEANUP_FAILED",
            )

        failure = asyncio.run(
            runner._run_and_publish_qualification_once(
                drift_factory,
                manifest_path=drift_manifest,
                result_path=drift_result,
                release_claim_path=drift_release,
                repository_root=ROOT,
                injected_paths=True,
            )
        )
        self.assertEqual(failure["classification"], "QUALIFICATION_FAILURE")
        self.assertEqual(failure["failure"]["category"], "INVARIANT")
        self.assertEqual(cleanup_calls, 1)
        self.assertEqual(_read_json(drift_result), failure)

        release_root = self.root / "qualification-release-fsync-uncertain"
        release_root.mkdir(mode=0o700)
        release_manifest = release_root / "manifest.json"
        release_result = release_root / "qualification.json"
        release_claim = release_root / "sealed" / "release.json"
        runner.publish_source_manifest(
            release_manifest,
            self.manifest,
            repository_root=ROOT,
        )
        release_factory_calls = 0

        def unreleased_factory(payload: dict[str, object]) -> object:
            nonlocal release_factory_calls
            release_factory_calls += 1
            raise AssertionError("factory must remain unreleased")

        real_fsync_directory = runner._fsync_directory

        def fail_release_parent(path: Path) -> None:
            if path == release_claim.parent:
                raise OSError("PRIVATE_RELEASE_FSYNC_UNCERTAIN")
            real_fsync_directory(path)

        with mock.patch.object(
            runner,
            "_fsync_directory",
            side_effect=fail_release_parent,
        ):
            release_failure = asyncio.run(
                runner._run_and_publish_qualification_once(
                    unreleased_factory,
                    manifest_path=release_manifest,
                    result_path=release_result,
                    release_claim_path=release_claim,
                    repository_root=ROOT,
                    injected_paths=True,
                )
            )
        self.assertEqual(
            release_failure["classification"],
            "QUALIFICATION_FAILURE",
        )
        self.assertEqual(release_factory_calls, 0)
        self.assertTrue(release_claim.exists())
        self.assertEqual(_read_json(release_result), release_failure)

        fsync_root = self.root / "qualification-fsync-uncertain"
        fsync_root.mkdir(mode=0o700)
        fsync_manifest = fsync_root / "manifest.json"
        fsync_result = fsync_root / "qualification.json"
        fsync_release = fsync_root / "sealed" / "release.json"
        runner.publish_source_manifest(
            fsync_manifest,
            self.manifest,
            repository_root=ROOT,
        )
        fsync_evaluator = _FakeOrchestrationEvaluator("qualification")
        fsync_executor = _FakeArmExecutor(
            fsync_evaluator,
            fsync_root / "clones",
        )
        factory_calls = 0

        def fsync_factory(
            payload: dict[str, object],
        ) -> runner._LiveDependenciesOwner:
            nonlocal factory_calls
            factory_calls += 1
            return runner._LiveDependenciesOwner(
                self._dependencies(fsync_evaluator, fsync_executor),
                lambda: None,
                "QUALIFICATION_OWNER_CLEANUP_FAILED",
            )

        real_fsync_directory = runner._fsync_directory

        def fail_result_parent(path: Path) -> None:
            if path == fsync_result.parent:
                raise OSError("PRIVATE_RESULT_FSYNC_UNCERTAIN")
            real_fsync_directory(path)

        with mock.patch.object(
            runner,
            "_fsync_directory",
            side_effect=fail_result_parent,
        ):
            with self.assertRaises(BaseExceptionGroup):
                asyncio.run(
                    runner._run_and_publish_qualification_once(
                        fsync_factory,
                        manifest_path=fsync_manifest,
                        result_path=fsync_result,
                        release_claim_path=fsync_release,
                        repository_root=ROOT,
                        injected_paths=True,
                    )
                )
        self.assertTrue(fsync_result.exists())
        self.assertTrue(fsync_release.exists())
        self.assertEqual(factory_calls, 1)
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_and_publish_qualification_once(
                    fsync_factory,
                    manifest_path=fsync_manifest,
                    result_path=fsync_result,
                    release_claim_path=fsync_release,
                    repository_root=ROOT,
                    injected_paths=True,
                )
            )
        self.assertEqual(factory_calls, 1)

    def test_malformed_proposal_is_one_generation_without_execution(self) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")
        observed: list[runner.ArmTaskSpec] = []

        def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            observed.append(spec)
            result = self._synthetic_result(
                spec,
                _digest(f"malformed-accounting-attempt-{len(observed)}"),
            )
            return (
                replace(result, parser_disposition="MALFORMED")
                if len(observed) == 1
                else result
            )

        result = asyncio.run(
            runner._run_qualification_injected(self._dependencies(owned, execute))
        )
        attempts = result["accounting"]["attempts"]
        resources = result["accounting"]["resources"]
        self.assertEqual(attempts["arm_tasks"], 31)
        self.assertEqual(attempts["proposal_attempts"], 31)
        self.assertEqual(attempts["execution_attempts"], 30)
        self.assertEqual(attempts["generation_attempts"], 61)
        self.assertEqual(attempts["phase_proposal_attempts"]["adaptation"], 24)
        self.assertEqual(attempts["phase_execution_attempts"]["adaptation"], 23)
        self.assertEqual(resources["generation_calls"], 61)
        self.assertEqual(owned.calls[-1], ("close", None))

    def test_attempt_and_resource_accounting_mismatches_fail_closed(self) -> None:
        for label in ("unfinished-arm-task", "missing-resource-generation"):
            with self.subTest(label=label):
                owned = _FakeOrchestrationEvaluator("qualification")
                accounting = runner.RunAccounting(runner.RunMode.QUALIFICATION)
                observed: list[runner.ArmTaskSpec] = []

                def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
                    observed.append(spec)
                    result = self._synthetic_result(
                        spec,
                        _digest(f"{label}-attempt-{len(observed)}"),
                    )
                    accounting.budget.consume_proposal(
                        spec.task_id,
                        spec.arm,
                        spec.phase,
                    )
                    accounting.resources.record_generation(1, 1)
                    accounting.budget.consume_execution(spec.task_id, spec.arm)
                    if not (
                        label == "missing-resource-generation"
                        and len(observed) == 31
                    ):
                        accounting.resources.record_generation(1, 1)
                    if not (
                        label == "unfinished-arm-task" and len(observed) == 31
                    ):
                        accounting.budget.finish_arm_task(spec.task_id, spec.arm)
                    return result

                with self.assertRaises(runner.RunnerInvariantError):
                    asyncio.run(
                        runner._run_qualification_injected(
                            self._dependencies(
                                owned,
                                execute,
                                accounting=accounting,
                                auto_accounting=False,
                            )
                        )
                    )
                self.assertEqual(len(observed), 31)
                self.assertTrue(owned.invalidated)
                self.assertEqual(owned.calls[-1], ("invalidate", None))
                self.assertNotIn(("close", None), owned.calls)

    def test_each_resource_dimension_must_be_explicitly_sampled(self) -> None:
        sample_operations = {
            "memory_samples": lambda ledger: ledger.record_memory(0, 0, 0),
            "wall_samples": lambda ledger: ledger.record_wall(0.0),
            "configured_pool_samples": lambda ledger: ledger.record_configured_pools(0),
            "state_scratch_samples": lambda ledger: ledger.record_state_scratch(0),
        }
        for missing in sample_operations:
            with self.subTest(missing=missing):
                owned = _FakeOrchestrationEvaluator("qualification")
                accounting = runner.RunAccounting(runner.RunMode.QUALIFICATION)
                for name, operation in sample_operations.items():
                    if name != missing:
                        operation(accounting.resources)
                executor = _FakeArmExecutor(owned, self.clone_parent)
                with self.assertRaises(runner.RunnerInvariantError):
                    asyncio.run(
                        runner._run_qualification_injected(
                            self._dependencies(
                                owned,
                                executor,
                                accounting=accounting,
                                auto_resource_samples=False,
                            )
                        )
                    )
                snapshot = accounting.resources.snapshot()
                self.assertEqual(snapshot[missing], 0)
                self.assertTrue(
                    all(
                        snapshot[name] == 1
                        for name in sample_operations
                        if name != missing
                    )
                )
                self.assertEqual(len(executor.specs), 31)
                self.assertTrue(owned.invalidated)

    def test_independently_expected_genesis_rejects_first_self_asserted_probe(
        self,
    ) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")
        executor = _FakeArmExecutor(owned, self.clone_parent)
        wrong = {
            owned.replicates[0]: _digest("wrong-independent-genesis")
        }
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_qualification_injected(
                    self._dependencies(
                        owned,
                        executor,
                        expected_learner_genesis_digests=wrong,
                    )
                )
            )
        self.assertEqual(len(executor.specs), 1)
        self.assertTrue(owned.invalidated)

    def test_foundation_map_must_bind_the_per_probe_tensor_digest(self) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")
        executor = _FakeArmExecutor(owned, self.clone_parent)
        mismatched = {
            **self.foundation_hashes,
            "foundation_tensor": hashlib.sha256(
                b"different-foundation-tensor"
            ).hexdigest(),
        }
        with self.assertRaises(runner.RunnerInvariantError):
            self._dependencies(
                owned,
                executor,
                expected_foundation_hashes=mismatched,
            )

    def test_manifest_drift_after_development_fails_before_final_admission(self) -> None:
        mutable_manifest = json.loads(
            runner.canonical_json_bytes(self.manifest).decode("utf-8")
        )
        owned = _FakeOrchestrationEvaluator("evaluation")
        executor = _FakeArmExecutor(owned, self.clone_parent)

        def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            result = executor(spec)
            if (
                spec.phase == "development"
                and spec.task["task_id"]
                == owned.tasks["development"][-1]["task_id"]
                and spec.arm == evaluator.EVALUATION_ARMS[-1]
            ):
                mutable_manifest["thresholds"]["full_final_minimum"] = 0.99
            return result

        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_evaluation_injected(
                    self._dependencies(
                        owned,
                        execute,
                        manifest=mutable_manifest,
                    ),
                    qualification_result_sha256=self.qualification_result_sha256,
                    evaluator_commitment_digest=self.evaluator_commitment_digest,
                    expected_admission_digest=self.admission,
                )
            )
        self.assertEqual(len(executor.specs), 188)
        self.assertIn(("complete_phase", "development"), owned.calls)
        self.assertNotIn(("admit_final", self.admission), owned.calls)
        self.assertNotIn(("release_phase", "final"), owned.calls)
        self.assertEqual(owned.calls[-1], ("invalidate", None))
        self.assertEqual(self.foundation_probe_calls, 1)

    def test_source_drift_after_development_fails_before_final_admission(self) -> None:
        repository = self.root / "isolated-repository"
        relative_leaf = runner.ACTIVE_LEAF_PATH.relative_to(runner.REPOSITORY_ROOT)
        for relative in set(self.completed_sources) | {str(relative_leaf)}:
            source = ROOT / relative
            target = repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

        drift_target = repository / "experiments/evaluators/high_level_multidomain_v1.py"
        owned = _FakeOrchestrationEvaluator("evaluation")
        executor = _FakeArmExecutor(owned, self.clone_parent)

        def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            result = executor(spec)
            if (
                spec.phase == "development"
                and spec.task["task_id"]
                == owned.tasks["development"][-1]["task_id"]
                and spec.arm == evaluator.EVALUATION_ARMS[-1]
            ):
                drift_target.write_bytes(drift_target.read_bytes() + b"\nDRIFT")
            return result

        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_evaluation_injected(
                    self._dependencies(
                        owned,
                        execute,
                        repository_root=repository,
                    ),
                    qualification_result_sha256=self.qualification_result_sha256,
                    evaluator_commitment_digest=self.evaluator_commitment_digest,
                    expected_admission_digest=self.admission,
                )
            )
        self.assertEqual(len(executor.specs), 188)
        self.assertIn(("complete_phase", "development"), owned.calls)
        self.assertNotIn(("admit_final", self.admission), owned.calls)
        self.assertNotIn(("release_phase", "final"), owned.calls)
        self.assertEqual(owned.calls[-1], ("invalidate", None))
        self.assertEqual(self.foundation_probe_calls, 1)

    def test_manifest_and_admission_fail_before_any_scheduled_attempt(self) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")
        executor = _FakeArmExecutor(owned, self.clone_parent)
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_qualification_injected(
                    self._dependencies(
                        owned,
                        executor,
                        expected_manifest_sha256="0" * 64,
                    )
                )
            )
        self.assertEqual(executor.specs, [])
        self.assertEqual(owned.calls, [("invalidate", None)])
        self.assertTrue(owned.invalidated)

        evaluation_owned = _FakeOrchestrationEvaluator("evaluation")
        evaluation_executor = _FakeArmExecutor(evaluation_owned, self.clone_parent)
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_evaluation_injected(
                    self._dependencies(evaluation_owned, evaluation_executor),
                    qualification_result_sha256=self.qualification_result_sha256,
                    evaluator_commitment_digest=self.evaluator_commitment_digest,
                    expected_admission_digest=_digest("wrong-admission"),
                )
            )
        self.assertEqual(evaluation_executor.specs, [])
        self.assertEqual(evaluation_owned.calls, [("invalidate", None)])
        self.assertTrue(evaluation_owned.invalidated)

    def test_duplicate_attempt_receipt_fails_globally_at_the_second_arm_task(self) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")
        shared_receipt = _digest("globally-reused-attempt-receipt")
        observed: list[runner.ArmTaskSpec] = []

        def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            observed.append(spec)
            receipt = (
                shared_receipt
                if len(observed) <= 2
                else _digest(f"unreachable-attempt-{len(observed)}")
            )
            return self._synthetic_result(spec, receipt)

        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_qualification_injected(self._dependencies(owned, execute))
            )
        self.assertEqual(len(observed), 2)
        self.assertNotEqual(
            (observed[0].task_id, observed[0].arm),
            (observed[1].task_id, observed[1].arm),
        )
        self.assertTrue(owned.invalidated)
        self.assertEqual(owned.calls[-1], ("invalidate", None))

    def test_malformed_objective_judgments_fail_before_the_next_attempt(self) -> None:
        cases = {
            "extra-field": {"secondary_score": 1},
            "bad-response-commitment": {
                "response_commitment": _digest("wrong-response")
            },
            "nonbinary-score": {"score": 0.5},
            "score-disposition-mismatch": {"disposition": "SUCCESS"},
        }
        for label, changes in cases.items():
            with self.subTest(label=label):
                owned = _FakeOrchestrationEvaluator("qualification")
                observed: list[runner.ArmTaskSpec] = []

                def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
                    observed.append(spec)
                    return self._synthetic_result(
                        spec,
                        _digest(f"{label}-attempt-{len(observed)}"),
                        judgment_changes=changes,
                    )

                with self.assertRaises(runner.RunnerInvariantError):
                    asyncio.run(
                        runner._run_qualification_injected(
                            self._dependencies(owned, execute)
                        )
                    )
                self.assertEqual(len(observed), 1)
                self.assertTrue(owned.invalidated)
                self.assertEqual(owned.calls[-1], ("invalidate", None))

    def test_hidden_or_extra_final_metrics_invalidate_after_the_frozen_schedule(
        self,
    ) -> None:
        for label in ("hidden-top-level", "extra-aggregate-field"):
            with self.subTest(label=label):
                owned = _FakeOrchestrationEvaluator("evaluation")
                observed: list[runner.ArmTaskSpec] = []

                def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
                    observed.append(spec)
                    return self._synthetic_result(
                        spec,
                        _digest(f"{label}-attempt-{len(observed)}"),
                    )

                original_metrics = owned.final_metrics

                def invalid_metrics() -> dict[str, object]:
                    metrics = original_metrics()
                    if label == "hidden-top-level":
                        metrics["solution"] = "PRIVATE-SENTINEL"
                    else:
                        metrics["aggregates"][0]["secondary_score"] = 1  # type: ignore[index]
                    return metrics

                owned.final_metrics = invalid_metrics  # type: ignore[method-assign]
                with self.assertRaises(runner.RunnerInvariantError):
                    asyncio.run(
                        runner._run_evaluation_injected(
                            self._dependencies(owned, execute),
                            qualification_result_sha256=(
                                self.qualification_result_sha256
                            ),
                            evaluator_commitment_digest=(
                                self.evaluator_commitment_digest
                            ),
                            expected_admission_digest=self.admission,
                        )
                    )
                self.assertEqual(len(observed), 468)
                self.assertTrue(owned.invalidated)
                self.assertEqual(owned.calls[-1], ("invalidate", None))

    def test_mid_schedule_executor_exception_invalidates_and_reaps_evaluator(self) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")
        observed: list[runner.ArmTaskSpec] = []

        def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            observed.append(spec)
            if len(observed) == 5:
                raise RuntimeError("INJECTED_EXECUTOR_FAILURE")
            return self._synthetic_result(
                spec,
                _digest(f"mid-schedule-attempt-{len(observed)}"),
            )

        with self.assertRaisesRegex(RuntimeError, "INJECTED_EXECUTOR_FAILURE"):
            asyncio.run(
                runner._run_qualification_injected(self._dependencies(owned, execute))
            )
        self.assertEqual(len(observed), 5)
        self.assertTrue(owned.invalidated)
        self.assertTrue(owned.closed)
        self.assertEqual(owned.calls[-1], ("invalidate", None))
        self.assertNotIn(("close", None), owned.calls)
        self.assertEqual(self.foundation_probe_calls, 1)

    def test_primary_error_is_preserved_when_evaluator_cleanup_also_fails(self) -> None:
        owned = _FakeOrchestrationEvaluator("qualification")

        def failing_invalidate() -> None:
            owned.calls.append(("invalidate", None))
            raise RuntimeError("PRIVATE-CLEANUP-DETAIL")

        owned.invalidate = failing_invalidate  # type: ignore[method-assign]

        def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            raise RuntimeError("PRIVATE-PRIMARY-DETAIL")

        with self.assertRaises(BaseExceptionGroup) as caught:
            asyncio.run(
                runner._run_qualification_injected(self._dependencies(owned, execute))
            )
        leaves = runner._exception_leaves(caught.exception)
        self.assertIsInstance(leaves[0], RuntimeError)
        self.assertIsInstance(leaves[-1], runner.CleanupFailure)
        classification, failure = runner.classify_exception_disposition(
            caught.exception,
            admission_validated=True,
            evaluator_cleanup_completed=False,
        )
        self.assertEqual(classification, "INVALID")
        self.assertEqual(failure["category"], "CLEANUP")
        self.assertEqual(
            failure["cleanup_codes"],
            ["EVALUATOR_CLEANUP_FAILED"],
        )
        self.assertNotIn(
            "PRIVATE",
            runner.canonical_json_bytes(failure).decode("utf-8"),
        )

    def test_foundation_drift_after_schedule_invalidates_before_success(self) -> None:
        self.foundation_snapshots[1] = {
            **self.foundation_hashes,
            "foundation_tensor": hashlib.sha256(
                b"MUTATED-FOUNDATION-TENSOR"
            ).hexdigest(),
        }
        owned = _FakeOrchestrationEvaluator("qualification")
        executor = _FakeArmExecutor(owned, self.clone_parent)
        with self.assertRaises(runner.RunnerInvariantError):
            asyncio.run(
                runner._run_qualification_injected(self._dependencies(owned, executor))
            )
        self.assertEqual(len(executor.specs), 31)
        self.assertEqual(self.foundation_probe_calls, 2)
        self.assertTrue(owned.invalidated)
        self.assertEqual(owned.calls[-1], ("invalidate", None))


class _LiveCudaFake:
    def __init__(self) -> None:
        self.current = 0
        self.reset_calls: list[int] = []
        self.synchronize_calls: list[int] = []
        self.empty_cache_calls = 0
        self.allocated = 128
        self.reserved = 256

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def set_device(self, value: int) -> None:
        self.current = value

    def current_device(self) -> int:
        return self.current

    def get_device_name(self, _: int) -> str:
        return runner.FROZEN_GPU_ASSIGNMENT["assigned_name"]

    def reset_peak_memory_stats(self, value: int) -> None:
        self.reset_calls.append(value)

    def max_memory_allocated(self, _: int) -> int:
        return self.allocated

    def max_memory_reserved(self, _: int) -> int:
        return self.reserved

    def synchronize(self, value: int) -> None:
        self.synchronize_calls.append(value)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _LiveTorchFake:
    __version__ = runner.EXPECTED_TORCH_VERSION

    class _Version:
        cuda = runner.EXPECTED_CUDA_VERSION

    class _Cudnn:
        @staticmethod
        def version() -> int:
            return runner.EXPECTED_CUDNN_RUNTIME

    class _Backends:
        cudnn = None

    version = _Version()
    backends = _Backends()
    backends.cudnn = _Cudnn()

    def __init__(self) -> None:
        self.cuda = _LiveCudaFake()
        self.threads = 2
        self.interop = 1
        self.interop_set_calls = 0

    def set_num_threads(self, value: int) -> None:
        self.threads = value

    def set_num_interop_threads(self, value: int) -> None:
        self.interop_set_calls += 1
        if self.interop_set_calls > 1:
            raise RuntimeError("interop pool can be configured only once")
        self.interop = value

    def get_num_threads(self) -> int:
        return self.threads

    def get_num_interop_threads(self) -> int:
        return self.interop


class _CompletedCommand:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


class LiveQualificationOwnerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _manifest() -> tuple[dict[str, object], str]:
        seeds = (bytes(range(32)), bytes(range(32, 64)))
        commitments = tuple(evaluator.commit_replicate_seed(seed) for seed in seeds)
        owned = evaluator.make_evaluation_evaluator(
            seeds,
            commitments,
            _digest("live-owner-manifest-placeholder"),
        )
        paths = set(_FROZEN_SOURCE_HASHES) | set(
            runner.REQUIRED_CONSTRUCTION_SOURCES
        )
        completed = {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in sorted(paths)
        }
        manifest = runner.build_source_manifest(
            evaluator_commitments=owned.commitments.to_canonical(),
            seed_commitments=commitments,
            accepted_leaf_sha256=hashlib.sha256(
                runner.ACTIVE_LEAF_PATH.read_bytes()
            ).hexdigest(),
            completed_source_hashes=completed,
        )
        return manifest, hashlib.sha256(
            runner.canonical_json_bytes(manifest)
        ).hexdigest()

    @staticmethod
    def _gpu_output() -> str:
        rows = []
        free = {
            runner.ASSIGNED_GPU_UUID: runner.MINIMUM_ASSIGNED_GPU_FREE_MIB,
            runner.UNASSIGNED_GPU_UUID: 10_000,
        }
        for uuid, value in runner.EXPECTED_PHYSICAL_GPUS.items():
            rows.append(
                ", ".join(
                    (
                        uuid,
                        str(value["name"]),
                        str(value["pci_bus_id"]),
                        str(value["compute_capability"]),
                        str(value["total_mib"]),
                        str(free[uuid]),
                    )
                )
            )
        return "\n".join(rows) + "\n"

    @staticmethod
    def _boundary_text(path: Path) -> str:
        if path == Path("/proc/net/route"):
            return "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
        if path == Path("/proc/net/ipv6_route"):
            return "0 0 0 0 0 0 0 0 0 lo\n"
        if path == Path("/sys/class/net/lo/operstate"):
            return "unknown\n"
        raise AssertionError(f"unexpected boundary path: {path}")

    @staticmethod
    def _foundation_evidence() -> dict[str, object]:
        return {
            "fastembed_cache_tree": dict(runner.EXPECTED_FASTEMBED_CACHE_TREE),
            "file_count": runner.QUALIFIED_MODEL_FILE_COUNT,
            "model_cache_tree": {
                "exists": False,
                "files": 0,
                "manifest_sha256": hashlib.sha256(b"ABSENT\n").hexdigest(),
                "total_bytes": 0,
            },
            "root_manifest_sha256": runner.QUALIFIED_MODEL_ROOT_SHA256,
            "schema": runner.FOUNDATION_LOAD_SCHEMA,
            "tiktoken_cache_tree": dict(runner.EXPECTED_TIKTOKEN_CACHE_TREE),
            "tokenizer_manifest_sha256": runner.QUALIFIED_TOKENIZER_SHA256,
            "total_bytes": runner.QUALIFIED_MODEL_TOTAL_BYTES,
        }

    @staticmethod
    def _prepare_roots(root: Path) -> tuple[Path, Path]:
        state = root / "state"
        scratch = root / "scratch"
        state.mkdir(mode=0o700)
        scratch.mkdir(mode=0o700)
        sealed = state / "sealed"
        sealed.mkdir(mode=0o700)
        for name in (
            runner.SEALED_SEED_PATH.name,
            runner.QUALIFICATION_RELEASE_CLAIM_PATH.name,
        ):
            path = sealed / name
            path.write_bytes(b"{}")
            path.chmod(0o600)
        temporary = scratch / "tmp"
        temporary.mkdir(mode=0o700)
        cuda_cache = scratch / "cuda-cache"
        cuda_cache.mkdir(mode=0o700)
        return state, scratch

    def test_live_parent_configures_once_then_observes_exact_same_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "tmp"
            temporary.mkdir(mode=0o700)
            environment = dict(runner.FROZEN_PARENT_ENVIRONMENT)
            environment["TMPDIR"] = str(temporary)
            torch_fake = _LiveTorchFake()

            class TorchVersion(str):
                pass

            torch_fake.__version__ = TorchVersion(runner.EXPECTED_TORCH_VERSION)

            def observe(*, configure_pools: bool) -> dict[str, object]:
                with mock.patch.object(
                    runner,
                    "FROZEN_PARENT_ENVIRONMENT",
                    environment,
                ):
                    return runner.validate_live_parent_boundary(
                        configure_pools=configure_pools,
                        environment=environment,
                        executable=runner.ANGLER_PYTHON,
                        runner_path=runner.RUNNER_PATH,
                        working_directory=runner.REPOSITORY_ROOT,
                        interface_inventory=lambda: ((1, "lo"),),
                        text_reader=self._boundary_text,
                        namespace_reader=lambda _: "net:[4026533000]",
                        gpu_inventory_runner=lambda *args, **kwargs: _CompletedCommand(
                            self._gpu_output()
                        ),
                        hostname_reader=lambda: runner.EXPECTED_HOSTNAME,
                        python_version_reader=lambda: runner.EXPECTED_PYTHON_VERSION,
                        distribution_reader=lambda name: (
                            runner.EXPECTED_PARENT_DISTRIBUTIONS[name]
                        ),
                        torch_module=torch_fake,
                    )

            initial = observe(configure_pools=True)
            revalidated = observe(configure_pools=False)
            self.assertEqual(initial, revalidated)
            self.assertEqual(torch_fake.interop_set_calls, 1)
            self.assertEqual(initial["configured_pool_sum"], 8)
            self.assertEqual(initial["hostname"], "angler-workstation")
            self.assertIs(type(initial["software"]["torch"]), str)
            self.assertEqual(
                initial["software"]["torch"],
                runner.EXPECTED_TORCH_VERSION,
            )
            self.assertIs(type(initial["software"]["cuda"]), str)
            self.assertIs(type(initial["software"]["cudnn_runtime"]), int)

            class EqualitySpoof:
                def __eq__(self, other: object) -> bool:
                    del other
                    return True

            torch_spoof = EqualitySpoof()
            self.assertEqual(torch_spoof, runner.EXPECTED_TORCH_VERSION)
            torch_fake.__version__ = torch_spoof
            with self.assertRaises(runner.RunnerInvariantError):
                observe(configure_pools=False)
            torch_fake.__version__ = TorchVersion(runner.EXPECTED_TORCH_VERSION)

            cuda_spoof = EqualitySpoof()
            self.assertEqual(cuda_spoof, runner.EXPECTED_CUDA_VERSION)
            torch_fake.version.cuda = cuda_spoof
            with self.assertRaises(runner.RunnerInvariantError):
                observe(configure_pools=False)
            torch_fake.version.cuda = runner.EXPECTED_CUDA_VERSION

            class SpoofCudnn:
                calls = 0

                def version(self) -> EqualitySpoof:
                    self.calls += 1
                    return EqualitySpoof()

            original_cudnn = torch_fake.backends.cudnn
            spoof_cudnn = SpoofCudnn()
            torch_fake.backends.cudnn = spoof_cudnn
            with self.assertRaises(runner.RunnerInvariantError):
                observe(configure_pools=False)
            self.assertEqual(spoof_cudnn.calls, 1)
            torch_fake.backends.cudnn = original_cudnn

            torch_fake.__version__ = TorchVersion("2.13.0+cu130-mismatch")
            with self.assertRaises(runner.RunnerInvariantError):
                observe(configure_pools=False)
            torch_fake.__version__ = TorchVersion(runner.EXPECTED_TORCH_VERSION)
            changed = self._gpu_output().replace("16303", "16304")
            with mock.patch.object(
                runner,
                "FROZEN_PARENT_ENVIRONMENT",
                environment,
            ):
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.validate_live_parent_boundary(
                        configure_pools=False,
                        environment=environment,
                        executable=runner.ANGLER_PYTHON,
                        runner_path=runner.RUNNER_PATH,
                        working_directory=runner.REPOSITORY_ROOT,
                        interface_inventory=lambda: ((1, "lo"),),
                        text_reader=self._boundary_text,
                        namespace_reader=lambda _: "net:[4026533000]",
                        gpu_inventory_runner=lambda *args, **kwargs: _CompletedCommand(
                            changed
                        ),
                        hostname_reader=lambda: runner.EXPECTED_HOSTNAME,
                        python_version_reader=lambda: runner.EXPECTED_PYTHON_VERSION,
                        distribution_reader=lambda name: (
                            runner.EXPECTED_PARENT_DISTRIBUTIONS[name]
                        ),
                        torch_module=torch_fake,
                    )

    def test_preconstruction_shape_and_resource_inventory_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, scratch = self._prepare_roots(root)
            runner._validate_qualification_preconstruction_roots(state, scratch)
            release = state / "sealed" / runner.QUALIFICATION_RELEASE_CLAIM_PATH.name
            release.unlink()
            runner._validate_qualification_preconstruction_roots(
                state,
                scratch,
                release_required=False,
            )
            with self.assertRaises(runner.RunnerInvariantError):
                runner._validate_qualification_preconstruction_roots(state, scratch)
            release.write_bytes(b"{}")
            release.chmod(0o600)
            cuda_cache = scratch / "cuda-cache"
            cuda_cache.rmdir()
            with self.assertRaises(runner.RunnerInvariantError):
                runner._validate_qualification_preconstruction_roots(state, scratch)
            cuda_cache.mkdir(mode=0o700)
            cache_entry = cuda_cache / "unexpected-cache-entry"
            cache_entry.write_bytes(b"unexpected")
            with self.assertRaises(runner.RunnerInvariantError):
                runner._validate_qualification_preconstruction_roots(state, scratch)
            cache_entry.unlink()
            foreign = scratch / "foreign"
            foreign.mkdir(mode=0o700)
            with self.assertRaises(runner.RunnerInvariantError):
                runner._validate_qualification_preconstruction_roots(state, scratch)
            foreign.rmdir()

            accounting = runner.RunAccounting(runner.RunMode.QUALIFICATION)
            torch_fake = _LiveTorchFake()
            baseline = state / "baseline.bin"
            baseline.write_bytes(b"a" * 8)
            sampler = runner._ProductionResourceSampler(
                accounting,
                state,
                scratch,
                torch_fake,
                clock=iter((10.0, 11.0, 12.0, 13.0)).__next__,
                rss_reader=lambda: 1024,
            )
            self.assertEqual(torch_fake.cuda.reset_calls, [0])
            sampler.start_after_model_load()
            baseline.write_bytes(b"b" * 8)
            with self.assertRaises(runner.RunnerInvariantError):
                sampler.sample()
            baseline.write_bytes(b"a" * 8)
            original_mode = stat.S_IMODE(baseline.stat().st_mode)
            baseline.chmod(0o600 if original_mode != 0o600 else 0o640)
            with self.assertRaises(runner.RunnerInvariantError):
                sampler.sample()
            baseline.chmod(original_mode)
            baseline.write_bytes(b"a" * 4)
            (state / "replacement.bin").write_bytes(b"b" * 4)
            with self.assertRaises(runner.RunnerInvariantError):
                sampler.sample()

            tree = root / "hardlinks"
            tree.mkdir()
            source = tree / "source"
            source.write_bytes(b"x")
            os.link(source, tree / "alias")
            with self.assertRaisesRegex(runner.RunnerInvariantError, "hardlink"):
                runner._regular_tree_bytes(tree)

            owned = root / "ownership"
            owned.mkdir()
            entry = owned / "entry"
            entry.write_bytes(b"x")
            real_lstat = Path.lstat

            def foreign_lstat(path: Path) -> object:
                observed = real_lstat(path)
                if path == entry:
                    values = list(observed)
                    values[4] = os.getuid() + 1
                    return os.stat_result(values)
                return observed

            with mock.patch.object(Path, "lstat", foreign_lstat):
                with self.assertRaisesRegex(
                    runner.RunnerInvariantError,
                    "ownership",
                ):
                    runner._regular_tree_bytes(owned)

    def test_evaluation_preconstruction_retains_only_audited_qualification_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, scratch = self._prepare_roots(root)
            qualification_runtime = state / "qualification-runtime"
            qualification_scopes = state / "qualification-scopes"
            evaluation_runtime = state / "evaluation-runtime"
            evaluation_scopes = state / "evaluation-scopes"
            qualification_runtime.mkdir(mode=0o700)
            qualification_scopes.mkdir(mode=0o700)
            seed = state / "sealed" / runner.SEALED_SEED_PATH.name
            release = (
                state
                / "sealed"
                / runner.QUALIFICATION_RELEASE_CLAIM_PATH.name
            )
            admission = (
                state
                / "sealed"
                / runner.EVALUATION_ADMISSION_CLAIM_PATH.name
            )

            def validate(*, admitted: bool) -> None:
                runner._validate_evaluation_preconstruction_roots(
                    state,
                    scratch,
                    qualification_runtime_root=qualification_runtime,
                    qualification_scope_root=qualification_scopes,
                    evaluation_runtime_root=evaluation_runtime,
                    evaluation_scope_root=evaluation_scopes,
                    seed_path=seed,
                    qualification_release_path=release,
                    admission_claim_path=admission,
                    admission_required=admitted,
                )

            validate(admitted=False)
            admission.write_bytes(b"{}")
            admission.chmod(0o600)
            validate(admitted=True)
            with self.assertRaises(runner.RunnerInvariantError):
                validate(admitted=False)

            admission.unlink()
            (scratch / "tmp" / "stale").write_bytes(b"stale")
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "temporary",
            ):
                validate(admitted=False)
            (scratch / "tmp" / "stale").unlink()

            evaluation_runtime.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "retained state shape|runtime identity",
            ):
                validate(admitted=False)
            evaluation_runtime.rmdir()

            source = qualification_runtime / "source"
            alias = qualification_runtime / "alias"
            source.write_bytes(b"x")
            os.link(source, alias)
            with self.assertRaisesRegex(runner.RunnerInvariantError, "hardlink"):
                validate(admitted=False)
            alias.unlink()
            source.unlink()
            (qualification_scopes / "link").symlink_to(scratch / "tmp")
            with self.assertRaisesRegex(runner.RunnerInvariantError, "symlink"):
                validate(admitted=False)

    async def test_manifest_requires_existing_seal_and_uses_one_cached_handshake(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_path = root / "sealed" / "seeds-v1.json"
            manifest_path = root / "manifest.json"
            result_path = root / "qualification.json"
            release_path = root / "qualification-release.json"
            calls = 0

            async def forbidden_factory(_: runner.EvaluatorWorkerSpec) -> object:
                nonlocal calls
                calls += 1
                raise AssertionError("evaluator must not start without a seed seal")

            with self.assertRaises(runner.RunnerInvariantError):
                await runner._ensure_live_source_manifest(
                    manifest_path=manifest_path,
                    seed_path=seed_path,
                    qualification_result_path=result_path,
                    qualification_release_path=release_path,
                    repository_root=ROOT,
                    evaluator_client_factory=forbidden_factory,
                    injected_paths=True,
                )
            self.assertEqual(calls, 0)

            seeds = (bytes(range(32)), bytes(range(32, 64)))
            entropy = iter(seeds)
            seal = runner.seal_evaluation_seeds(
                seed_path,
                entropy=lambda size: next(entropy) if size == 32 else b"",
            )
            with mock.patch.object(
                runner,
                "completed_source_manifest",
                side_effect=runner.RunnerInvariantError(
                    "injected source validation failure"
                ),
            ):
                with self.assertRaisesRegex(
                    runner.RunnerInvariantError,
                    "source validation",
                ):
                    await runner._ensure_live_source_manifest(
                        manifest_path=manifest_path,
                        seed_path=seed_path,
                        qualification_result_path=result_path,
                        qualification_release_path=release_path,
                        repository_root=ROOT,
                        evaluator_client_factory=forbidden_factory,
                        injected_paths=True,
                    )
            self.assertEqual(calls, 0)
            owned = evaluator.make_evaluation_evaluator(
                seeds,
                seal.seed_commitments,
                runner.COMMITMENT_ONLY_ADMISSION_DIGEST,
            )
            commitments = owned.commitments.to_canonical()
            commitment_digest = owned.commitments.digest
            process = _ScriptedProcess(
                [
                    runner.canonical_json_bytes(
                        {
                            "id": 1,
                            "ok": True,
                            "result": {
                                "commitments": commitments,
                                "digest": commitment_digest,
                            },
                            "v": 1,
                        }
                    )
                    + b"\n",
                    runner.canonical_json_bytes(
                        {"id": 2, "ok": True, "result": {}, "v": 1}
                    )
                    + b"\n",
                ]
            )

            def client_factory(
                spec: runner.EvaluatorWorkerSpec,
            ) -> runner.EvaluatorClient:
                nonlocal calls
                calls += 1
                return runner.EvaluatorClient.start(
                    spec,
                    process_factory=lambda _: process,
                )

            with mock.patch.object(runner, "SEALED_SEED_PATH", seed_path):
                manifest, manifest_sha256 = await runner._ensure_live_source_manifest(
                    manifest_path=manifest_path,
                    seed_path=seed_path,
                    qualification_result_path=result_path,
                    qualification_release_path=release_path,
                    repository_root=ROOT,
                    evaluator_client_factory=client_factory,
                    injected_paths=True,
                )
                rejoined, rejoined_sha256 = (
                    await runner._ensure_live_source_manifest(
                        manifest_path=manifest_path,
                        seed_path=seed_path,
                        qualification_result_path=result_path,
                        qualification_release_path=release_path,
                        repository_root=ROOT,
                        evaluator_client_factory=lambda _: (_ for _ in ()).throw(
                            AssertionError("manifest rejoin restarted evaluator")
                        ),
                        injected_paths=True,
                    )
                )
            self.assertEqual(calls, 1)
            self.assertEqual(manifest, rejoined)
            self.assertEqual(manifest_sha256, rejoined_sha256)
            self.assertEqual(manifest["qualification_result_sha256"], None)
            requests = [
                runner.decode_canonical_json(chunk[:-1])
                for chunk in process.stdin.chunks
            ]
            self.assertEqual(
                [request["op"] for request in requests],
                ["commitments", "close"],
            )

    def test_foundation_cache_guard_and_cuda_cleanup_are_independent(self) -> None:
        class Guard:
            initial_tensor_digest = _digest("live-foundation")

            def probe(self) -> str:
                return self.initial_tensor_digest

            def verify_boundary_digest(self) -> str:
                return self.initial_tensor_digest

        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "model"
            model_root.mkdir()
            evidence = self._foundation_evidence()
            loaded = runner.LoadedQwenFoundation(
                model=object(),
                tokenizer=object(),
                io=object(),
                manifest=object(),
                guard=Guard(),
                production_file_evidence=evidence,
            )
            torch_fake = _LiveTorchFake()
            verifier_calls = 0

            def verifier(_: Path) -> dict[str, object]:
                nonlocal verifier_calls
                verifier_calls += 1
                return json.loads(json.dumps(evidence))

            owner = runner._LiveFoundationOwner(
                loaded,
                model_root,
                torch_fake,
                verifier,
            )
            self.assertEqual(owner.expected_hashes["foundation_tensor"], _digest("live-foundation").removeprefix("sha256:"))
            owner.probe()
            owner.probe()
            owner.close()
            self.assertEqual(verifier_calls, 1)
            self.assertEqual(torch_fake.cuda.synchronize_calls, [0])
            self.assertEqual(torch_fake.cuda.empty_cache_calls, 1)

            failing_loaded = runner.LoadedQwenFoundation(
                model=object(),
                tokenizer=object(),
                io=object(),
                manifest=object(),
                guard=Guard(),
                production_file_evidence=evidence,
            )
            failing_cuda = _LiveCudaFake()

            def failed_sync(_: int) -> None:
                raise RuntimeError("injected synchronize failure")

            failing_cuda.synchronize = failed_sync  # type: ignore[method-assign]
            failing_torch = _LiveTorchFake()
            failing_torch.cuda = failing_cuda
            failing_owner = runner._LiveFoundationOwner(
                failing_loaded,
                model_root,
                failing_torch,
                verifier,
            )
            with self.assertRaises(runner.CleanupFailure):
                failing_owner.close()
            self.assertEqual(failing_cuda.empty_cache_calls, 1)

            class Token:
                pass

            retained = Token()
            retained_ref = weakref.ref(retained)
            verifier_failure_cuda = _LiveCudaFake()
            original_sync = verifier_failure_cuda.synchronize

            def observe_released(value: int) -> None:
                gc.collect()
                if retained_ref() is not None:
                    raise AssertionError(
                        "file-verifier traceback retained the loaded foundation"
                    )
                original_sync(value)

            verifier_failure_cuda.synchronize = observe_released  # type: ignore[method-assign]
            verifier_failure_torch = _LiveTorchFake()
            verifier_failure_torch.cuda = verifier_failure_cuda

            def failing_verifier(_: Path) -> dict[str, object]:
                raise runner.RunnerInvariantError(
                    "injected final foundation-file verification failure"
                )

            verifier_failure_owner = runner._LiveFoundationOwner(
                runner.LoadedQwenFoundation(
                    model=retained,
                    tokenizer=object(),
                    io=object(),
                    manifest=object(),
                    guard=Guard(),
                    production_file_evidence=evidence,
                ),
                model_root,
                verifier_failure_torch,
                failing_verifier,
            )
            retained = None  # type: ignore[assignment]
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "foundation-file verification",
            ):
                verifier_failure_owner.close()
            self.assertIsNone(retained_ref())
            self.assertEqual(verifier_failure_cuda.synchronize_calls, [0])
            self.assertEqual(verifier_failure_cuda.empty_cache_calls, 1)

    def test_qwen_loader_uses_one_exact_injected_owner_and_drops_partial_state(
        self,
    ) -> None:
        import torch

        class Config:
            hidden_size = 2_560
            num_hidden_layers = 36
            max_position_embeddings = 40_960

        class Qwen3ForCausalLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.ones(1, dtype=torch.bfloat16),
                    requires_grad=False,
                )
                self.config = Config()

        class Tokenizer:
            pass

        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "fake-model"
            model_root.mkdir()
            model_calls: list[dict[str, object]] = []
            tokenizer_calls: list[dict[str, object]] = []

            def tokenizer_factory(_: Path, **kwargs: object) -> Tokenizer:
                tokenizer_calls.append(dict(kwargs))
                return Tokenizer()

            def model_factory(_: Path, **kwargs: object) -> Qwen3ForCausalLM:
                model_calls.append(dict(kwargs))
                return Qwen3ForCausalLM()

            loaded = runner.load_qualified_qwen_foundation(
                model_root,
                model_factory=model_factory,
                tokenizer_factory=tokenizer_factory,
            )
            self.assertFalse(loaded.production_file_identity_verified)
            self.assertEqual(type(loaded.model).__name__, "Qwen3ForCausalLM")
            self.assertEqual(loaded.guard.probe(), loaded.guard.initial_tensor_digest)
            self.assertEqual(model_calls[0]["device_map"], {"": "cuda:0"})
            self.assertIs(model_calls[0]["dtype"], torch.bfloat16)
            self.assertEqual(tokenizer_calls[0]["local_files_only"], True)

            tokenizer_ref: list[weakref.ReferenceType[Tokenizer]] = []

            def partial_tokenizer(_: Path, **kwargs: object) -> Tokenizer:
                del kwargs
                value = Tokenizer()
                tokenizer_ref.append(weakref.ref(value))
                return value

            with self.assertRaisesRegex(RuntimeError, "injected model load"):
                runner.load_qualified_qwen_foundation(
                    model_root,
                    model_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
                        RuntimeError("injected model load")
                    ),
                    tokenizer_factory=partial_tokenizer,
                )
            gc.collect()
            self.assertIsNone(tokenizer_ref[0]())

    async def test_repository_drift_is_rejected_before_live_construction(
        self,
    ) -> None:
        manifest, manifest_sha256 = self._manifest()
        effects: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, scratch = self._prepare_roots(root)
            repository_guard = root / "repository"
            repository_guard.mkdir()
            repository_before = runner._repository_tree_manifest(repository_guard)
            (repository_guard / "unexpected-write").write_bytes(b"mutation")

            async def factory(
                payload: dict[str, object],
            ) -> runner._LiveDependenciesOwner:
                return await runner._build_live_qualification_owner(
                    payload,
                    parent_boundary_validator=lambda: effects.append("boundary"),
                    expected_parent_boundary={"boundary": "never"},
                    evaluator_client_factory=lambda _: effects.append("evaluator"),
                    foundation_model_root=root / "model",
                    state_root=state,
                    scratch_root=scratch,
                    runtime_root=state / "qualification-v1",
                    scope_root=state / "cognee-scopes",
                    expected_repository_tree=repository_before,
                    repository_tree_probe=lambda: runner._repository_tree_manifest(
                        repository_guard
                    ),
                    injected_boundaries=True,
                )

            with self.assertRaises(BaseExceptionGroup):
                await runner._construct_live_dependencies_owner(
                    factory,
                    {
                        "manifest": manifest,
                        "manifest_sha256": manifest_sha256,
                        "purpose": "qualification",
                    },
                    purpose="qualification",
                )
            self.assertEqual(effects, [])

    async def test_live_owner_registers_effects_immediately_and_cleans_in_reverse(
        self,
    ) -> None:
        manifest, manifest_sha256 = self._manifest()
        events: list[str] = []

        class Token:
            pass

        class Guard:
            initial_tensor_digest = _digest("owned-live-foundation")

            def probe(self) -> str:
                events.append("foundation-probe")
                return self.initial_tensor_digest

            def verify_boundary_digest(self) -> str:
                events.append("foundation-boundary")
                return self.initial_tensor_digest

        class Client:
            def release_phase(self, _: str) -> tuple[()]:
                return ()

            def complete_phase(self, _: str) -> None:
                return None

            def judge_response(self, *args: object) -> None:
                del args

            def random_feedback(self, _: str) -> tuple[()]:
                return ()

            def close(self) -> None:
                events.append("evaluator-close")

            def invalidate(self) -> None:
                events.append("evaluator-invalidate")

        class ArmFactory:
            def audit_lineages(self) -> dict[str, object]:
                events.append("arm-audit")
                return {
                    "lineages": [],
                    "managed_dispositions": [],
                    "purpose": "qualification",
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, scratch = self._prepare_roots(root)
            model_root = root / "model"
            model_root.mkdir()
            repository_guard = root / "repository"
            repository_guard.mkdir()
            repository_before = runner._repository_tree_manifest(repository_guard)
            torch_fake = _LiveTorchFake()
            model_ref: list[weakref.ReferenceType[Token]] = []

            def load_foundation() -> runner.LoadedQwenFoundation:
                self.assertEqual(torch_fake.cuda.reset_calls, [0])
                model = Token()
                model_ref.append(weakref.ref(model))
                return runner.LoadedQwenFoundation(
                    model=model,
                    tokenizer=Token(),
                    io=object(),
                    manifest=object(),
                    guard=Guard(),
                    production_file_evidence=self._foundation_evidence(),
                )

            def verify_files(_: Path) -> dict[str, object]:
                events.append("file-verify")
                return self._foundation_evidence()

            boundary = {"boundary": "freshly-revalidated"}

            async def factory(
                payload: dict[str, object],
            ) -> runner._LiveDependenciesOwner:
                return await runner._build_live_qualification_owner(
                    payload,
                    parent_boundary_validator=lambda: boundary,
                    expected_parent_boundary=boundary,
                    evaluator_client_factory=lambda _: Client(),
                    genesis_factory=lambda _: object(),
                    foundation_loader=load_foundation,
                    foundation_model_root=model_root,
                    foundation_file_verifier=verify_files,
                    arm_factory_builder=lambda *args, **kwargs: ArmFactory(),
                    torch_module=torch_fake,
                    state_root=state,
                    scratch_root=scratch,
                    runtime_root=state / "qualification-v1",
                    scope_root=state / "cognee-scopes",
                    expected_repository_tree=repository_before,
                    repository_tree_probe=lambda: runner._repository_tree_manifest(
                        repository_guard
                    ),
                    injected_boundaries=True,
                )

            owner = await runner._construct_live_dependencies_owner(
                factory,
                {
                    "manifest": manifest,
                    "manifest_sha256": manifest_sha256,
                    "purpose": "qualification",
                },
                purpose="qualification",
            )
            self.assertEqual(torch_fake.cuda.reset_calls, [0])
            finalizer = owner.dependencies.accounting._finalizer
            self.assertTrue(callable(finalizer))
            finalizer()
            terminal_resources = owner.dependencies.accounting.resources.snapshot()
            await owner.close()
            self.assertEqual(
                owner.dependencies.accounting.resources.snapshot(),
                terminal_resources,
            )
            gc.collect()
            self.assertIsNone(model_ref[0]())
            self.assertLess(events.index("arm-audit"), events.index("file-verify"))
            self.assertLess(
                events.index("file-verify"),
                events.index("evaluator-invalidate"),
            )
            self.assertTrue(
                (state / "qualification-v1" / "arm-factory-audit.json").is_file()
            )
            self.assertGreaterEqual(
                terminal_resources["peak_state_scratch_bytes"],
                (state / "qualification-v1" / "arm-factory-audit.json").stat().st_size,
            )

        failure_events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, scratch = self._prepare_roots(root)
            model_root = root / "model"
            model_root.mkdir()
            repository_guard = root / "repository"
            repository_guard.mkdir()
            repository_before = runner._repository_tree_manifest(repository_guard)
            torch_fake = _LiveTorchFake()
            rejected_ref: list[weakref.ReferenceType[Token]] = []

            def rejected_loader() -> runner.LoadedQwenFoundation:
                model = Token()
                rejected_ref.append(weakref.ref(model))
                return runner.LoadedQwenFoundation(
                    model=model,
                    tokenizer=Token(),
                    io=object(),
                    manifest=object(),
                    guard=Guard(),
                    production_file_evidence=None,
                )

            rejected_client = Client()
            rejected_client.invalidate = lambda: failure_events.append(  # type: ignore[method-assign]
                "evaluator-invalidate"
            )

            async def rejected_factory(
                payload: dict[str, object],
            ) -> runner._LiveDependenciesOwner:
                return await runner._build_live_qualification_owner(
                    payload,
                    parent_boundary_validator=lambda: {"boundary": "same"},
                    expected_parent_boundary={"boundary": "same"},
                    evaluator_client_factory=lambda _: rejected_client,
                    genesis_factory=lambda _: object(),
                    foundation_loader=rejected_loader,
                    foundation_model_root=model_root,
                    foundation_file_verifier=lambda _: self._foundation_evidence(),
                    arm_factory_builder=lambda *args, **kwargs: ArmFactory(),
                    torch_module=torch_fake,
                    state_root=state,
                    scratch_root=scratch,
                    runtime_root=state / "qualification-v1",
                    scope_root=state / "cognee-scopes",
                    expected_repository_tree=repository_before,
                    repository_tree_probe=lambda: runner._repository_tree_manifest(
                        repository_guard
                    ),
                    injected_boundaries=True,
                )

            with self.assertRaises(runner.RunnerInvariantError):
                await runner._construct_live_dependencies_owner(
                    rejected_factory,
                    {
                        "manifest": manifest,
                        "manifest_sha256": manifest_sha256,
                        "purpose": "qualification",
                    },
                    purpose="qualification",
                )
            gc.collect()
            self.assertIsNone(rejected_ref[0]())
            self.assertEqual(failure_events, ["evaluator-invalidate"])
            self.assertEqual(torch_fake.cuda.synchronize_calls, [0])
            self.assertEqual(torch_fake.cuda.empty_cache_calls, 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, scratch = self._prepare_roots(root)
            model_root = root / "model"
            model_root.mkdir()
            repository_guard = root / "repository"
            repository_guard.mkdir()
            repository_before = runner._repository_tree_manifest(repository_guard)
            torch_fake = _LiveTorchFake()
            late_ref: list[weakref.ReferenceType[Token]] = []
            original_sync = torch_fake.cuda.synchronize

            def observe_release(value: int) -> None:
                if not late_ref or late_ref[0]() is not None:
                    raise AssertionError("foundation remained reachable at CUDA cleanup")
                original_sync(value)

            torch_fake.cuda.synchronize = observe_release  # type: ignore[method-assign]

            def late_loader() -> runner.LoadedQwenFoundation:
                model = Token()
                late_ref.append(weakref.ref(model))
                return runner.LoadedQwenFoundation(
                    model=model,
                    tokenizer=Token(),
                    io=Token(),
                    manifest=object(),
                    guard=Guard(),
                    production_file_evidence=self._foundation_evidence(),
                )

            def fail_after_repository_mutation(*args: object, **kwargs: object) -> object:
                del args, kwargs
                (repository_guard / "unexpected-write").write_bytes(b"mutation")
                raise BaseExceptionGroup(
                    "injected grouped late arm failure",
                    [runner.RunnerInvariantError("injected late arm failure")],
                )

            async def late_factory(
                payload: dict[str, object],
            ) -> runner._LiveDependenciesOwner:
                return await runner._build_live_qualification_owner(
                    payload,
                    parent_boundary_validator=lambda: {"boundary": "same"},
                    expected_parent_boundary={"boundary": "same"},
                    evaluator_client_factory=lambda _: Client(),
                    genesis_factory=lambda _: object(),
                    foundation_loader=late_loader,
                    foundation_model_root=model_root,
                    foundation_file_verifier=lambda _: self._foundation_evidence(),
                    arm_factory_builder=fail_after_repository_mutation,
                    torch_module=torch_fake,
                    state_root=state,
                    scratch_root=scratch,
                    runtime_root=state / "qualification-v1",
                    scope_root=state / "cognee-scopes",
                    expected_repository_tree=repository_before,
                    repository_tree_probe=lambda: runner._repository_tree_manifest(
                        repository_guard
                    ),
                    injected_boundaries=True,
                )

            with self.assertRaises(BaseExceptionGroup) as caught:
                await runner._construct_live_dependencies_owner(
                    late_factory,
                    {
                        "manifest": manifest,
                        "manifest_sha256": manifest_sha256,
                        "purpose": "qualification",
                    },
                    purpose="qualification",
                )
            leaves = runner._exception_leaves(caught.exception)
            self.assertTrue(
                any(
                    isinstance(item, runner.CleanupFailure)
                    and item.code
                    == "QUALIFICATION_REPOSITORY_REVALIDATION_FAILED"
                    for item in leaves
                )
            )
            self.assertIsNone(late_ref[0]())
            self.assertEqual(torch_fake.cuda.synchronize_calls, [0])
            self.assertEqual(torch_fake.cuda.empty_cache_calls, 1)

    async def test_run_failure_tracebacks_release_state_before_owner_cleanup(
        self,
    ) -> None:
        class Token:
            pass

        token_ref: list[weakref.ReferenceType[Token]] = []

        def failed_execution() -> BaseException:
            retained = Token()
            token_ref.append(weakref.ref(retained))
            try:
                raise runner.RunnerInvariantError("injected arm execution failure")
            except BaseException as error:
                return BaseExceptionGroup("grouped execution failure", [error])

        class Owner:
            async def close(self) -> None:
                gc.collect()
                if token_ref[0]() is not None:
                    raise AssertionError(
                        "execution traceback retained state during owner cleanup"
                    )

        with self.assertRaises(BaseExceptionGroup):
            await runner._raise_after_owner_cleanup(Owner(), failed_execution())
        self.assertIsNone(token_ref[0]())

    async def test_live_cli_rechecks_seed_only_roots_around_manifest_worker(
        self,
    ) -> None:
        events: list[object] = []

        def roots(*args: object, **kwargs: object) -> None:
            del args
            events.append(("roots", kwargs.get("release_required")))

        async def ensure() -> tuple[dict[str, object], str]:
            events.append("manifest")
            return {}, "0" * 64

        async def publish(*args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            events.append("publish")
            return {"classification": "QUALIFICATION_PASS"}

        def boundary(*, configure_pools: bool) -> dict[str, object]:
            events.append(("boundary", configure_pools))
            return {"boundary": "exact"}

        def repository(_: Path) -> dict[str, object]:
            events.append("repository")
            return {
                "entries": 1,
                "files": 0,
                "manifest_sha256": "0" * 64,
                "schema": runner.REPOSITORY_TREE_SCHEMA,
                "total_bytes": 0,
            }

        with mock.patch.object(
            runner,
            "validate_live_parent_boundary",
            side_effect=boundary,
        ), mock.patch.object(
            runner,
            "_require_identity_absent",
            side_effect=lambda *args: events.append("identity"),
        ), mock.patch.object(
            runner,
            "_validate_qualification_preconstruction_roots",
            side_effect=roots,
        ), mock.patch.object(
            runner,
            "_ensure_live_source_manifest",
            side_effect=ensure,
        ), mock.patch.object(
            runner,
            "_repository_tree_manifest",
            side_effect=repository,
        ), mock.patch.object(
            runner,
            "_run_and_publish_qualification_once",
            side_effect=publish,
        ):
            result = await runner._run_live_qualification_cli()
        self.assertEqual(result["classification"], "QUALIFICATION_PASS")
        self.assertEqual(
            events,
            [
                *("identity" for _ in range(6)),
                ("roots", False),
                ("boundary", True),
                ("roots", False),
                "manifest",
                ("roots", False),
                "repository",
                "publish",
            ],
        )

    async def test_live_cli_rejects_boundary_cache_mutation_before_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state, scratch = self._prepare_roots(Path(directory))
            release = state / "sealed" / runner.QUALIFICATION_RELEASE_CLAIM_PATH.name
            release.unlink()
            effects: list[str] = []

            def mutating_boundary(*, configure_pools: bool) -> dict[str, object]:
                self.assertTrue(configure_pools)
                (scratch / "cuda-cache" / "mutated").write_bytes(b"mutation")
                return {"boundary": "mutated"}

            with mock.patch.object(runner, "STATE_ROOT", state), mock.patch.object(
                runner,
                "SCRATCH_ROOT",
                scratch,
            ), mock.patch.object(
                runner,
                "validate_live_parent_boundary",
                side_effect=mutating_boundary,
            ), mock.patch.object(
                runner,
                "_require_identity_absent",
            ), mock.patch.object(
                runner,
                "_ensure_live_source_manifest",
                side_effect=lambda: effects.append("manifest"),
            ), mock.patch.object(
                runner,
                "_run_and_publish_qualification_once",
                side_effect=lambda *args, **kwargs: effects.append("publish"),
            ):
                with self.assertRaises(runner.RunnerInvariantError):
                    await runner._run_live_qualification_cli()
            self.assertEqual(effects, [])

    def test_run_accounting_finalizer_precedes_terminal_snapshot(self) -> None:
        accounting = runner.RunAccounting(runner.RunMode.QUALIFICATION)
        calls = 0

        def finalize() -> None:
            nonlocal calls
            calls += 1
            accounting.resources.record_state_scratch(123)

        accounting.bind_finalizer(finalize)
        with mock.patch.object(
            runner,
            "validate_serialized_run_accounting",
            side_effect=lambda value, mode: value,
        ):
            first = accounting.validate_complete()
            second = accounting.validate_complete()
        self.assertEqual(calls, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["resources"]["peak_state_scratch_bytes"], 123)

    def test_evaluator_process_removes_ephemeral_worker_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd_parent = root / "evaluator-cwd"
            tmp_parent = root / "evaluator-tmp"
            worker_cwd = cwd_parent / "worker-one"
            worker_tmp = tmp_parent / "worker-one"
            worker_cwd.mkdir(parents=True)
            worker_tmp.mkdir(parents=True)
            process = _ScriptedProcess([])
            managed = runner._ManagedEvaluatorProcess(
                process,
                worker_cwd,
                worker_tmp,
            )
            self.assertEqual(managed.wait(timeout=1.0), 0)
            self.assertFalse(cwd_parent.exists())
            self.assertFalse(tmp_parent.exists())

    def test_evaluator_process_retains_nonempty_shared_worker_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd_parent = root / "evaluator-cwd"
            tmp_parent = root / "evaluator-tmp"
            worker_cwd = cwd_parent / "worker-one"
            worker_tmp = tmp_parent / "worker-one"
            other_cwd = cwd_parent / "worker-two"
            other_tmp = tmp_parent / "worker-two"
            for path in (worker_cwd, worker_tmp, other_cwd, other_tmp):
                path.mkdir(parents=True)
            managed = runner._ManagedEvaluatorProcess(
                _ScriptedProcess([]),
                worker_cwd,
                worker_tmp,
            )
            self.assertEqual(managed.wait(timeout=1.0), 0)
            self.assertFalse(worker_cwd.exists())
            self.assertFalse(worker_tmp.exists())
            self.assertTrue(other_cwd.is_dir())
            self.assertTrue(other_tmp.is_dir())
            other_cwd.rmdir()
            other_tmp.rmdir()
            cwd_parent.rmdir()
            tmp_parent.rmdir()

    def test_public_live_entrypoints_have_no_injection_and_main_calls_exact_mode(
        self,
    ) -> None:
        self.assertEqual(
            tuple(inspect.signature(runner.run_and_publish_qualification_once).parameters),
            (),
        )
        completed = {"classification": "QUALIFICATION_PASS"}
        with mock.patch.object(
            runner,
            "run_and_publish_qualification_once",
            new=mock.AsyncMock(return_value=completed),
        ) as live, mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            runner.main(("--qualification",))
        live.assert_awaited_once_with()
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "classification": "QUALIFICATION_PASS",
                "result": str(runner.QUALIFICATION_RESULT_PATH),
            },
        )
        self.assertEqual(
            tuple(inspect.signature(runner.run_and_publish_evaluation_once).parameters),
            (),
        )
        completed = {"classification": "NOT_SUPPORTED"}
        with mock.patch.object(
            runner,
            "run_and_publish_evaluation_once",
            new=mock.AsyncMock(return_value=completed),
        ) as live:
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                runner.main(("--evaluation",))
        live.assert_awaited_once_with()
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "classification": "NOT_SUPPORTED",
                "result": str(runner.EVALUATION_RESULT_PATH),
            },
        )

    def test_cuda_oom_is_typed_inside_cleanup_exception_groups(self) -> None:
        OutOfMemoryError = type(
            "OutOfMemoryError",
            (RuntimeError,),
            {"__module__": "torch.cuda"},
        )
        typed = runner._typed_live_infrastructure(
            OutOfMemoryError("private CUDA allocation detail"),
            code="LIVE_CONSTRUCTION_FAILED",
            stage="LIVE_CONSTRUCTION",
        )
        classification, failure = runner.classify_exception_disposition(
            typed,
            admission_validated=True,
            evaluator_cleanup_completed=True,
        )
        self.assertEqual(classification, "INCONCLUSIVE")
        self.assertEqual(failure["code"], "CUDA_OUT_OF_MEMORY")
        self.assertEqual(failure["stage"], "RESOURCE_GUARD")

        grouped = runner._typed_live_infrastructure(
            BaseExceptionGroup(
                "live OOM and cleanup",
                [
                    OutOfMemoryError("private CUDA allocation detail"),
                    runner.CleanupFailure(
                        "QUALIFICATION_QWEN_CLEANUP_FAILED",
                        "LIVE_CONSTRUCTION_CLEANUP",
                    ),
                ],
            ),
            code="LIVE_CONSTRUCTION_FAILED",
            stage="LIVE_CONSTRUCTION",
        )
        classification, failure = runner.classify_exception_disposition(
            grouped,
            admission_validated=True,
            evaluator_cleanup_completed=False,
        )
        self.assertEqual(classification, "INVALID")
        self.assertEqual(failure["category"], "CLEANUP")
        self.assertEqual(failure["code"], "CUDA_OUT_OF_MEMORY")
        self.assertEqual(failure["stage"], "RESOURCE_GUARD")
        self.assertEqual(
            failure["cleanup_codes"],
            ["QUALIFICATION_QWEN_CLEANUP_FAILED"],
        )


class LiveEvaluationOwnerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _foundation_evidence() -> dict[str, object]:
        return LiveQualificationOwnerTests._foundation_evidence()

    async def _artifacts(self, root: Path) -> dict[str, object]:
        state = root / "state"
        scratch = root / "scratch"
        results = root / "results"
        sealed = state / "sealed"
        for path in (state, scratch, results, sealed):
            path.mkdir(mode=0o700)
        for name in ("tmp", "cuda-cache"):
            (scratch / name).mkdir(mode=0o700)
        qualification_runtime = state / "qualification-runtime"
        qualification_scopes = state / "qualification-scopes"
        qualification_runtime.mkdir(mode=0o700)
        qualification_scopes.mkdir(mode=0o700)
        manifest_path = root / "manifest.json"
        seed_path = sealed / "seeds.json"
        qualification_path = results / "qualification.json"
        result_path = results / "evaluation.json"
        release_path = sealed / "qualification-release.json"
        claim_path = sealed / "evaluation-claim.json"

        manifest, manifest_sha256 = LiveQualificationOwnerTests._manifest()
        runner.publish_source_manifest(
            manifest_path,
            manifest,
            repository_root=ROOT,
        )
        entropy = iter((bytes(range(32)), bytes(range(32, 64))))
        seal = runner.seal_evaluation_seeds(
            seed_path,
            entropy=lambda size: next(entropy) if size == 32 else b"",
            expected_commitments=tuple(manifest["seed_commitments"]),
        )

        owned = _FakeOrchestrationEvaluator("qualification")
        executor = _FakeArmExecutor(owned, root / "qualification-clones")
        accounting = runner.RunAccounting(runner.RunMode.QUALIFICATION)
        accounting.resources.record_memory(0, 0, 0)
        accounting.resources.record_wall(0.0)
        accounting.resources.record_configured_pools(0)
        accounting.resources.record_state_scratch(0)

        async def execute(spec: runner.ArmTaskSpec) -> runner.ArmTaskResult:
            result = executor(spec)
            accounting.budget.consume_proposal(
                spec.task_id,
                spec.arm,
                spec.phase,
            )
            accounting.resources.record_generation(1, 1)
            accounting.budget.consume_execution(spec.task_id, spec.arm)
            accounting.resources.record_generation(1, 1)
            accounting.budget.finish_arm_task(spec.task_id, spec.arm)
            return result

        foundation_digest = _digest("fixture-foundation-tensor")
        foundation_hashes = {
            "foundation_tensor": foundation_digest.removeprefix("sha256:"),
            "model-files": hashlib.sha256(b"model-files").hexdigest(),
            "tensor-state": hashlib.sha256(b"tensor-state").hexdigest(),
            "tokenizer": hashlib.sha256(b"tokenizer").hexdigest(),
        }
        dependencies = runner.OrchestrationDependencies(
            evaluator=owned,
            execute_arm=execute,
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            foundation_probe=lambda: dict(foundation_hashes),
            expected_foundation_hashes=foundation_hashes,
            expected_foundation_tensor_digest=foundation_digest,
            expected_learner_genesis_digests={
                owned.replicates[0]: _FakeProbeIntegrityFactory.genesis_digest(
                    "qualification",
                    owned.replicates[0],
                    "FULL",
                )
            },
            accounting=accounting,
            repository_root=ROOT,
        )
        qualification = await runner._run_qualification_injected(dependencies)
        runner.publish_qualification_result(
            qualification_path,
            qualification,
            expected_manifest_sha256=manifest_sha256,
        )
        release, _ = runner._build_qualification_release_claim(
            manifest_sha256=manifest_sha256,
            manifest_path=manifest_path,
            result_path=qualification_path,
            claim_path=release_path,
            repository_root=ROOT,
            path_binding_mode="INJECTED_CPU_TEST",
        )
        runner._publish_qualification_release_claim(release_path, release)
        permit = await runner._claim_evaluation_admission(
            manifest_path=manifest_path,
            seed_path=seed_path,
            qualification_result_path=qualification_path,
            evaluation_result_path=result_path,
            claim_path=claim_path,
            repository_root=ROOT,
            evaluator_handshake=lambda: {
                "commitments": manifest["evaluator_commitments"],
                "digest": manifest["evaluator_commitment_digest"],
            },
            injected_paths=True,
        )
        return {
            "claim_path": claim_path,
            "evaluation_runtime": state / "evaluation-runtime",
            "evaluation_scopes": state / "evaluation-scopes",
            "manifest": manifest,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
            "permit": permit,
            "qualification": qualification,
            "qualification_path": qualification_path,
            "qualification_runtime": qualification_runtime,
            "qualification_scopes": qualification_scopes,
            "release_path": release_path,
            "result_path": result_path,
            "scratch": scratch,
            "seal": seal,
            "seed_path": seed_path,
            "state": state,
        }

    async def test_commitment_only_handshake_closes_or_reaps_before_return(
        self,
    ) -> None:
        manifest, _ = LiveQualificationOwnerTests._manifest()
        with tempfile.TemporaryDirectory() as directory:
            seed_path = Path(directory) / "sealed" / "seeds.json"
            entropy = iter((bytes(range(32)), bytes(range(32, 64))))
            seal = runner.seal_evaluation_seeds(
                seed_path,
                entropy=lambda size: next(entropy) if size == 32 else b"",
                expected_commitments=tuple(manifest["seed_commitments"]),
            )
            events: list[str] = []
            specs: list[runner.EvaluatorWorkerSpec] = []

            class Client:
                @property
                def commitment_snapshot(self) -> dict[str, object]:
                    events.append("snapshot")
                    return {
                        "commitments": manifest["evaluator_commitments"],
                        "digest": manifest["evaluator_commitment_digest"],
                    }

                def close(self) -> None:
                    events.append("close")

                def invalidate(self) -> None:
                    events.append("reap")

            def factory(spec: runner.EvaluatorWorkerSpec) -> Client:
                events.append("start")
                specs.append(spec)
                return Client()

            with mock.patch.object(runner, "SEALED_SEED_PATH", seed_path):
                observed = await runner._run_live_evaluation_commitment_handshake(
                    manifest,
                    seal,
                    evaluator_client_factory=factory,
                    injected_boundaries=True,
                )
            self.assertEqual(events, ["start", "snapshot", "close"])
            self.assertEqual(
                observed,
                {
                    "commitments": manifest["evaluator_commitments"],
                    "digest": manifest["evaluator_commitment_digest"],
                },
            )
            self.assertTrue(specs[0].commitment_only)
            self.assertEqual(
                specs[0].expected_final_admission,
                runner.COMMITMENT_ONLY_ADMISSION_DIGEST,
            )
            self.assertEqual(
                specs[0].expected_commitments,
                tuple(manifest["seed_commitments"]),
            )

            class CloseFailureClient(Client):
                def close(self) -> None:
                    events.append("close-failed")
                    raise RuntimeError("injected close failure")

            events.clear()
            with mock.patch.object(runner, "SEALED_SEED_PATH", seed_path):
                with self.assertRaisesRegex(RuntimeError, "close failure"):
                    await runner._run_live_evaluation_commitment_handshake(
                        manifest,
                        seal,
                        evaluator_client_factory=lambda spec: (
                            events.append("start") or CloseFailureClient()
                        ),
                        injected_boundaries=True,
                    )
            self.assertEqual(
                events,
                ["start", "snapshot", "close-failed", "reap"],
            )

    async def test_live_admission_rejoins_release_after_handshake_before_write(
        self,
    ) -> None:
        manifest, manifest_sha256 = LiveQualificationOwnerTests._manifest()
        qualification_sha256 = hashlib.sha256(b"qualification-pass").hexdigest()
        qualification = {"classification": "QUALIFICATION_PASS"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_parent = root / "results"
            sealed = root / "sealed"
            result_parent.mkdir(mode=0o700)
            sealed.mkdir(mode=0o700)
            manifest_path = root / "manifest.json"
            seed_path = sealed / "seed.json"
            qualification_path = result_parent / "qualification.json"
            result_path = result_parent / "evaluation.json"
            claim_path = sealed / "admission.json"
            seal = runner.SeedSeal(
                schema=runner.SEED_SEAL_SCHEMA,
                seed_commitments=tuple(manifest["seed_commitments"]),
                artifact_sha256="0" * 64,
                artifact_bytes=1,
                path=str(seed_path),
            )
            events: list[str] = []

            async def handshake() -> dict[str, object]:
                events.append("handshake")
                return {
                    "commitments": manifest["evaluator_commitments"],
                    "digest": manifest["evaluator_commitment_digest"],
                }

            def release_loader(path: Path) -> tuple[dict[str, object], str]:
                self.assertEqual(path, runner.QUALIFICATION_RELEASE_CLAIM_PATH)
                events.append("release-rejoin")
                return {
                    "manifest_sha256": "f" * 64,
                    "path_binding_mode": "FROZEN_LIVE",
                    "paths": {},
                }, _digest("mutated-release")

            def writer(*args: object) -> None:
                del args
                events.append("claim-write")

            manifest_raw = runner.canonical_json_bytes(manifest)
            self.assertEqual(
                hashlib.sha256(manifest_raw).hexdigest(),
                manifest_sha256,
            )
            with mock.patch.object(
                runner,
                "load_source_manifest",
                return_value=manifest,
            ), mock.patch.object(
                runner,
                "_read_private_artifact",
                return_value=manifest_raw,
            ), mock.patch.object(
                runner,
                "load_seed_seal",
                return_value=seal,
            ), mock.patch.object(
                runner,
                "load_qualification_result",
                return_value=(qualification, qualification_sha256),
            ), mock.patch.object(
                runner,
                "_load_qualification_release_claim",
                side_effect=release_loader,
            ), mock.patch.object(
                runner,
                "_write_consuming_admission_claim",
                side_effect=writer,
            ) as claim_writer:
                with self.assertRaisesRegex(
                    runner.RunnerInvariantError,
                    "qualification release differs",
                ):
                    await runner._claim_evaluation_admission(
                        manifest_path=manifest_path,
                        seed_path=seed_path,
                        qualification_result_path=qualification_path,
                        evaluation_result_path=result_path,
                        claim_path=claim_path,
                        repository_root=ROOT,
                        evaluator_handshake=handshake,
                        injected_paths=False,
                    )
            self.assertEqual(events, ["handshake", "release-rejoin"])
            claim_writer.assert_not_called()
            self.assertFalse(claim_path.exists())

    async def test_live_evaluation_owner_binds_admission_two_replicates_and_cleanup(
        self,
    ) -> None:
        class Guard:
            initial_tensor_digest = _digest("evaluation-live-foundation")

            def probe(self) -> str:
                return self.initial_tensor_digest

            def verify_boundary_digest(self) -> str:
                return self.initial_tensor_digest

        class Client:
            def __init__(
                self,
                snapshot: dict[str, object],
                events: list[str],
            ) -> None:
                self.commitment_snapshot = snapshot
                self.events = events

            def commitments(self) -> dict[str, object]:
                return self.commitment_snapshot

            def release_phase(self, phase: str) -> tuple[()]:
                del phase
                return ()

            def complete_phase(self, phase: str) -> None:
                del phase

            def admit_final(self, digest: str) -> None:
                del digest

            def judge_response(self, *args: object) -> None:
                del args

            def random_feedback(self, replicate: str) -> tuple[()]:
                del replicate
                return ()

            def final_metrics(self) -> dict[str, object]:
                return _valid_final_metrics()

            def close(self) -> None:
                self.events.append("evaluator-close")

            def invalidate(self) -> None:
                self.events.append("evaluator-reap")

        class ArmFactory:
            def __init__(self, events: list[str]) -> None:
                self.events = events

            def audit_lineages(self) -> dict[str, object]:
                self.events.append("arm-audit")
                return {
                    "lineages": [],
                    "managed_dispositions": [],
                    "purpose": "evaluation",
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = await self._artifacts(root)
            receipt = artifacts["permit"].consume()
            with self.assertRaises(runner.RunnerInvariantError):
                artifacts["permit"].consume()
            manifest = artifacts["manifest"]
            payload = {
                **receipt,
                "manifest": manifest,
                "qualification_result": artifacts["qualification"],
                "seed_seal": artifacts["seal"].to_canonical(),
            }
            events: list[str] = []
            specs: list[runner.EvaluatorWorkerSpec] = []
            arm_calls: list[dict[str, object]] = []
            model_root = root / "model"
            model_root.mkdir()
            repository_guard = root / "repository-guard"
            repository_guard.mkdir()
            repository_before = runner._repository_tree_manifest(repository_guard)
            torch_fake = _LiveTorchFake()

            def evaluator_factory(spec: runner.EvaluatorWorkerSpec) -> Client:
                events.append("full-evaluator-start")
                specs.append(spec)
                return Client(
                    {
                        "commitments": manifest["evaluator_commitments"],
                        "digest": manifest["evaluator_commitment_digest"],
                    },
                    events,
                )

            def foundation_loader() -> runner.LoadedQwenFoundation:
                events.append("foundation-load")
                return runner.LoadedQwenFoundation(
                    model=object(),
                    tokenizer=object(),
                    io=object(),
                    manifest=object(),
                    guard=Guard(),
                    production_file_evidence=self._foundation_evidence(),
                )

            async def arm_builder(*args: object, **kwargs: object) -> ArmFactory:
                arm_calls.append({"args": args, **kwargs})
                return ArmFactory(events)

            async def factory(
                owner_payload: dict[str, object],
            ) -> runner._LiveDependenciesOwner:
                return await runner._build_live_evaluation_owner(
                    owner_payload,
                    parent_boundary_validator=lambda: {"boundary": "same"},
                    expected_parent_boundary={"boundary": "same"},
                    evaluator_client_factory=evaluator_factory,
                    genesis_factory=lambda path: (
                        events.append(f"genesis:{Path(path).name}") or object()
                    ),
                    foundation_loader=foundation_loader,
                    foundation_model_root=model_root,
                    foundation_file_verifier=lambda path: (
                        self._foundation_evidence()
                    ),
                    arm_factory_builder=arm_builder,
                    torch_module=torch_fake,
                    state_root=artifacts["state"],
                    scratch_root=artifacts["scratch"],
                    qualification_runtime_root=artifacts[
                        "qualification_runtime"
                    ],
                    qualification_scope_root=artifacts[
                        "qualification_scopes"
                    ],
                    qualification_release_path=artifacts["release_path"],
                    runtime_root=artifacts["evaluation_runtime"],
                    scope_root=artifacts["evaluation_scopes"],
                    expected_repository_tree=repository_before,
                    repository_tree_probe=lambda: runner._repository_tree_manifest(
                        repository_guard
                    ),
                    injected_boundaries=True,
                )

            with mock.patch.object(
                runner,
                "SEALED_SEED_PATH",
                artifacts["seed_path"],
            ):
                owner = await runner._construct_live_dependencies_owner(
                    factory,
                    payload,
                    purpose="evaluation",
                )
                self.assertEqual(
                    owner.dependencies.accounting.mode,
                    runner.RunMode.EVALUATION,
                )
                self.assertEqual(
                    owner.dependencies.expected_learner_genesis_digests,
                    {
                        commitment: runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST
                        for commitment in artifacts["seal"].seed_commitments
                    },
                )
                owner.dependencies.accounting._finalizer()
                await owner.close()
            self.assertEqual(len(specs), 1)
            self.assertFalse(specs[0].commitment_only)
            self.assertEqual(
                specs[0].expected_final_admission,
                receipt["claim"]["admission_digest"],
            )
            self.assertEqual(
                specs[0].expected_commitments,
                artifacts["seal"].seed_commitments,
            )
            self.assertEqual(len(arm_calls), 1)
            self.assertEqual(arm_calls[0]["purpose"], "evaluation")
            self.assertEqual(
                arm_calls[0]["replicate_commitments"],
                artifacts["seal"].seed_commitments,
            )
            self.assertEqual(
                Path(arm_calls[0]["args"][0]),
                artifacts["evaluation_runtime"] / "arm-runtime",
            )
            self.assertTrue(
                (artifacts["evaluation_runtime"] / "arm-factory-audit.json").is_file()
            )
            self.assertIn("arm-audit", events)
            self.assertIn("evaluator-reap", events)

            effects: list[str] = []
            invalid = dict(payload)
            invalid["extra"] = True
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "payload differs",
            ):
                await runner._build_live_evaluation_owner(
                    invalid,
                    parent_boundary_validator=lambda: effects.append("boundary"),
                    evaluator_client_factory=lambda spec: effects.append("evaluator"),
                    foundation_loader=lambda: effects.append("foundation"),
                    arm_factory_builder=lambda *args, **kwargs: effects.append("arms"),
                    injected_boundaries=True,
                )
            self.assertEqual(effects, [])

    async def test_live_evaluation_cli_reaps_handshake_before_claim(self) -> None:
        events: list[object] = []
        manifest, manifest_sha256 = LiveQualificationOwnerTests._manifest()
        seal = runner.SeedSeal(
            schema=runner.SEED_SEAL_SCHEMA,
            seed_commitments=tuple(manifest["seed_commitments"]),
            artifact_sha256="0" * 64,
            artifact_bytes=1,
            path=str(runner.SEALED_SEED_PATH),
        )
        evidence = (
            manifest,
            manifest_sha256,
            {"classification": "QUALIFICATION_PASS"},
            "1" * 64,
            seal,
            {"release": True},
            _digest("release-ref"),
        )

        async def handshake(*args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            events.extend(("handshake-start", "handshake-snapshot", "handshake-close"))
            return {
                "commitments": manifest["evaluator_commitments"],
                "digest": manifest["evaluator_commitment_digest"],
            }

        async def coordinator(
            factory: object,
            *,
            evaluator_handshake: object,
        ) -> dict[str, object]:
            del factory
            events.append("coordinator")
            await evaluator_handshake()
            events.append("claim")
            return {"classification": "NOT_SUPPORTED"}

        def boundary(*, configure_pools: bool) -> dict[str, object]:
            events.append(("boundary", configure_pools))
            return {"boundary": "same"}

        def repository(_: Path) -> dict[str, object]:
            events.append("repository")
            return {
                "entries": 1,
                "files": 0,
                "manifest_sha256": "0" * 64,
                "schema": runner.REPOSITORY_TREE_SCHEMA,
                "total_bytes": 0,
            }

        with mock.patch.object(
            runner,
            "_require_identity_absent",
            side_effect=lambda *args: events.append("identity"),
        ), mock.patch.object(
            runner,
            "_validate_evaluation_preconstruction_roots",
            side_effect=lambda *args, **kwargs: events.append(
                ("roots", kwargs["admission_required"])
            ),
        ), mock.patch.object(
            runner,
            "_load_evaluation_preconstruction_evidence",
            return_value=evidence,
        ), mock.patch.object(
            runner,
            "validate_live_parent_boundary",
            side_effect=boundary,
        ), mock.patch.object(
            runner,
            "_repository_tree_manifest",
            side_effect=repository,
        ), mock.patch.object(
            runner,
            "_run_live_evaluation_commitment_handshake",
            side_effect=handshake,
        ), mock.patch.object(
            runner,
            "_run_and_publish_evaluation_once",
            side_effect=coordinator,
        ):
            result = await runner._run_live_evaluation_cli()
        self.assertEqual(result["classification"], "NOT_SUPPORTED")
        self.assertLess(events.index("handshake-close"), events.index("claim"))
        self.assertLess(events.index(("boundary", True)), events.index("handshake-start"))
        self.assertGreaterEqual(events.count(("roots", False)), 3)
        self.assertNotIn(("roots", True), events)


class AtomicOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_success_and_failure_records_are_canonical_create_once(self) -> None:
        for classification in ("INCONCLUSIVE", "INVALID"):
            with self.subTest(classification=classification):
                path = self.root / f"{classification}.json"
                payload = {"classification": classification, "value": 1}
                written = runner.atomic_result(
                    path,
                    payload,
                    classification,
                    max_bytes=4096,
                )
                expected = {"classification": classification, "value": 1}
                encoded = runner.canonical_json_bytes(expected)
                self.assertEqual(path.read_bytes(), encoded)
                self.assertEqual(written, path)
                with self.assertRaises(runner.RunnerInvariantError):
                    runner.atomic_result(
                        path,
                        payload,
                        classification,
                        max_bytes=4096,
                    )
                self.assertEqual(_read_json(path), expected)
        success_path = self.root / "untyped-success.json"
        with self.assertRaises(runner.RunnerInvariantError):
            runner.atomic_result(
                success_path,
                {"classification": "EXPERIMENTALLY_SUPPORTED"},
                "EXPERIMENTALLY_SUPPORTED",
                max_bytes=4096,
            )
        self.assertFalse(success_path.exists())

    def test_recursive_failure_precedence_is_typed_bounded_and_secret_free(self) -> None:
        infrastructure = runner.InfrastructureFailure(
            "MODEL_DEVICE_FAILURE",
            "MODEL_EXECUTION",
        )
        self.assertEqual(
            runner.classify_exception_disposition(
                infrastructure,
                admission_validated=False,
                evaluator_cleanup_completed=True,
            ),
            (None, None),
        )
        classification, failure = runner.classify_exception_disposition(
            infrastructure,
            admission_validated=True,
            evaluator_cleanup_completed=True,
        )
        self.assertEqual(classification, "INCONCLUSIVE")
        self.assertEqual(failure["category"], "INFRASTRUCTURE")
        self.assertEqual(failure["cleanup_codes"], [])
        self.assertNotIn("message", failure)
        self.assertNotIn("traceback", failure)

        mixed = BaseExceptionGroup(
            "private group label",
            [
                runner.InfrastructureFailure("TRANSPORT_DOWN", "EVALUATOR_IO"),
                runner.RunnerInvariantError("PRIVATE-HIDDEN-MATERIAL"),
            ],
        )
        classification, invalid = runner.classify_exception_disposition(
            mixed,
            admission_validated=True,
            evaluator_cleanup_completed=True,
        )
        self.assertEqual(classification, "INVALID")
        self.assertEqual(invalid["category"], "INVARIANT")
        self.assertEqual(invalid["exception_type"], "RunnerInvariantError")
        self.assertEqual(invalid["code"], "RunnerInvariantError")
        self.assertNotIn("PRIVATE", runner.canonical_json_bytes(invalid).decode())

        grouped_cleanup = BaseExceptionGroup(
            "private cleanup label",
            [
                runner.InfrastructureFailure("TRANSPORT_DOWN", "EVALUATOR_IO"),
                runner.CleanupFailure(
                    "EVALUATOR_CLEANUP_FAILED",
                    "EVALUATOR_CLEANUP",
                ),
            ],
        )
        classification, cleanup = runner.classify_exception_disposition(
            grouped_cleanup,
            admission_validated=True,
            evaluator_cleanup_completed=False,
        )
        self.assertEqual(classification, "INVALID")
        self.assertEqual(cleanup["category"], "CLEANUP")
        self.assertEqual(
            cleanup["cleanup_codes"],
            ["EVALUATOR_CLEANUP_FAILED"],
        )
        self.assertEqual(cleanup["code"], "TRANSPORT_DOWN")

        caused = runner.RunnerInvariantError("OUTER-PRIVATE")
        caused.__cause__ = runner.InfrastructureFailure(
            "INNER_TRANSPORT",
            "EVALUATOR_IO",
        )
        caused.__context__ = caused
        classification, recursive = runner.classify_exception_disposition(
            caused,
            admission_validated=True,
            evaluator_cleanup_completed=True,
        )
        self.assertEqual(classification, "INVALID")
        self.assertEqual(recursive["exception_type"], "RunnerInvariantError")

        transport = runner.EvaluatorTransportError("PRIVATE_TRANSPORT")
        transport.__cause__ = TimeoutError("PRIVATE_TIMEOUT")
        classification, normalized_transport = (
            runner.classify_exception_disposition(
                transport,
                admission_validated=True,
                evaluator_cleanup_completed=True,
            )
        )
        self.assertEqual(classification, "INCONCLUSIVE")
        self.assertEqual(normalized_transport["category"], "INFRASTRUCTURE")
        self.assertEqual(
            normalized_transport["exception_type"],
            "EvaluatorTransportError",
        )

        claim, claim_ref = runner.build_evaluation_admission_claim(
            manifest_sha256="1" * 64,
            qualification_result_sha256="2" * 64,
            evaluator_commitment_digest=_digest("commitments"),
            source_hashes={
                relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                for relative in sorted(
                    set(_FROZEN_SOURCE_HASHES)
                    | set(runner.REQUIRED_CONSTRUCTION_SOURCES)
                )
            },
            manifest_path=runner.MANIFEST_PATH,
            seed_path=runner.SEALED_SEED_PATH,
            qualification_result_path=runner.QUALIFICATION_RESULT_PATH,
            evaluation_result_path=runner.EVALUATION_RESULT_PATH,
            claim_path=runner.EVALUATION_ADMISSION_CLAIM_PATH,
            repository_root=ROOT,
            seed_seal_ref=_digest("seed-seal"),
            path_binding_mode="FROZEN_LIVE",
        )
        failure_result = runner.build_evaluation_failure_result(
            classification="INCONCLUSIVE",
            failure=failure,
            admission_claim=claim,
            admission_claim_ref=claim_ref,
            manifest_sha256="1" * 64,
        )
        failure_path = self.root / "typed-inconclusive.json"
        with self.assertRaises(runner.RunnerInvariantError):
            runner.publish_evaluation_failure_result(
                failure_path,
                failure_result,
            )
        self.assertFalse(failure_path.exists())
        changed = json.loads(
            runner.canonical_json_bytes(failure_result).decode("utf-8")
        )
        changed["failure"]["evaluator_cleanup_completed"] = False
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_evaluation_failure_result(changed)
        contradictory = dict(failure)
        contradictory["admission_validated"] = False
        contradictory["evaluator_cleanup_completed"] = False
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_failure_disposition(contradictory)
        qualification_failure = runner.build_qualification_failure_result(
            failure={
                **failure,
                "admission_validated": False,
            },
            manifest_sha256="1" * 64,
        )
        forged_qualification = json.loads(
            runner.canonical_json_bytes(qualification_failure).decode("utf-8")
        )
        forged_qualification["failure"]["admission_validated"] = True
        with self.assertRaises(runner.RunnerInvariantError):
            runner.validate_qualification_failure_result(forged_qualification)

    def test_oversized_or_symlink_output_fails_without_partial_replacement(self) -> None:
        path = self.root / "oversized.json"
        with self.assertRaises((RuntimeError, ValueError)):
            runner.atomic_result(
                path,
                {"classification": "INVALID", "detail": "x" * 100},
                "INVALID",
                max_bytes=32,
            )
        self.assertFalse(path.exists())
        self.assertEqual(tuple(self.root.glob(".*.tmp")), ())

        target = self.root / "target.json"
        target.write_text("sentinel", encoding="utf-8")
        path.symlink_to(target)
        with self.assertRaises((FileExistsError, RuntimeError, ValueError)):
            runner.atomic_result(
                path,
                {"classification": "INVALID"},
                "INVALID",
                max_bytes=4096,
            )
        self.assertEqual(target.read_text(encoding="utf-8"), "sentinel")

    def test_competing_create_between_precheck_and_publish_is_never_overwritten(
        self,
    ) -> None:
        target = self.root / "race.json"
        temporary = target.with_name(target.name + ".tmp")
        competing = b"COMPETING-CREATE-WINS"
        real_open = runner.os.open
        injected = False

        def racing_open(path: object, flags: int, mode: int = 0o777) -> int:
            nonlocal injected
            descriptor = real_open(path, flags, mode)
            if Path(path) == temporary and not injected:
                injected = True
                target.write_bytes(competing)
            return descriptor

        with mock.patch.object(runner.os, "open", side_effect=racing_open):
            with self.assertRaises(runner.RunnerInvariantError):
                runner.atomic_result(
                    target,
                    {"classification": "INVALID", "value": 1},
                    "INVALID",
                    max_bytes=4096,
                )
        self.assertTrue(injected)
        self.assertEqual(target.read_bytes(), competing)
        self.assertFalse(temporary.exists())

    def test_directory_fsync_failure_preserves_consumed_hard_link_target(self) -> None:
        target = self.root / "fsync-failure.json"
        payload = {"classification": "INVALID", "value": 7}
        expected = runner.canonical_json_bytes(payload)
        with mock.patch.object(
            runner,
            "_fsync_directory",
            side_effect=OSError("INJECTED_DIRECTORY_FSYNC_FAILURE"),
        ):
            with self.assertRaises((OSError, runner.RunnerInvariantError)):
                runner.atomic_result(
                    target,
                    payload,
                    "INVALID",
                    max_bytes=4096,
                )
        self.assertEqual(target.read_bytes(), expected)
        self.assertFalse(target.with_name(target.name + ".tmp").exists())
        with self.assertRaises(runner.RunnerInvariantError):
            runner.atomic_result(
                target,
                payload,
                "INVALID",
                max_bytes=4096,
            )
        self.assertEqual(target.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
