"""Privacy-preserving primitives for the consumed V1-R2 trace diagnosis.

This module is deliberately a diagnostic boundary, not a task evaluator.  It
can verify immutable inputs, parse canonical evidence, compare already-recorded
values, classify text against a public literal grammar, rejoin provenance
references, and publish a redacted result exactly once.  It cannot generate,
repair, select, execute, or score a procedure.

The live R2 evidence is not opened at import time and this construction module
has no command-line entry point.  Its sole live entry point accepts no paths and
is hard-bound to the literal frozen inputs and create-once output below.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import math
import os
import re
import resource
import signal
import sqlite3
import stat
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence


DIAGNOSTIC_IDENTITY = (
    "angler.high-level-multidomain.v1-r2-posthoc-trace-diagnosis.v1"
)
DIAGNOSTIC_SCHEMA = "angler.high-level-multidomain.v1-r2-trace-diagnosis.v1"
OUTPUT_PATH = Path(
    "/opt/angler/results/high-level-multidomain-v1-r2-trace-diagnosis.json"
)

MAX_OUTPUT_BYTES = 1_048_576
MAX_JSON_INPUT_BYTES = 64 * 1_048_576
MAX_EVIDENCE_RECORD_BYTES = 4 * 1_048_576
MAX_TEXT_BYTES = 16_384
MAX_ATTEMPTS = 468
MAX_WALL_SECONDS = 120
MAX_RSS_BYTES = 1_073_741_824
MAX_RAW_MATERIAL_BYTES = 64 * 1_048_576
MAX_RAW_MATERIAL_VALUES = 16_384
MAX_ACQUISITIONS_PER_LINEAGE = 256
SQL_FETCH_BATCH = 32
EXPECTED_COHORT_TASKS = 3
EXPECTED_MATRIX_ATTEMPTS = 21
EXPECTED_ARMS = (
    "FULL",
    "QWEN_ONLY",
    "RETRIEVAL_ONLY",
    "FROZEN_ORIGIN",
    "PROSPECTIVE_REMOVAL",
    "BACKEND_REMOVAL",
    "RANDOM_FEEDBACK",
)
NO_HISTORY_SENTINEL = "NO_PERSISTENT_RECALLED_EVIDENCE_V1"

_MANIFEST_SCHEMA = "angler.high-level-multidomain.manifest.v1"
_EVALUATION_RESULT_SCHEMA = "angler.high-level-multidomain.evaluation-result.v1"
_EVALUATION_IDENTITY = "angler.high-level-multidomain.v1-r2-evaluation"
_EVIDENCE_SCHEMA = "angler.high-level-multidomain.evidence-ledger.v1"
_MANAGED_DISPOSITION_SCHEMA = (
    "angler.high-level-multidomain.managed-runtime-disposition.v1"
)
_ACQUISITION_CONTRACT = "ANG-CTR-COGNITIVE-ACQUISITION-001@0.1.0"
_MEMORY_RECORD_CONTRACT = "ANG-CTR-COGNITIVE-MEMORY-001@0.1.0"
_LEGACY_EPISODE_CONTRACT = "ANG-CTR-COGNITIVE-EPISODE-001@0.1.0"
_PROSPECTIVE_BATCH_CONTRACT = (
    "ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001@0.1.0"
)
_PROSPECTIVE_RESOLUTION_CONTRACT = (
    "ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0"
)
_GRAPH_PROJECTION_CONTRACT = (
    "ANG-CTR-COGNITIVE-GRAPH-PROJECTION-002@0.1.0"
)
_RUN_INTEGRITY_SCHEMA = "angler.high-level-multidomain.run-integrity.v1"
_REMOVAL_FAIRNESS_SCHEMA = (
    "angler.high-level-multidomain.removal-fairness.v1"
)
_LINEAGE_INTEGRITY_SCHEMA = (
    "angler.high-level-multidomain.lineage-integrity.v1"
)
_PROBE_INTEGRITY_SCHEMA = "angler.high-level-multidomain.probe-integrity.v1"
_ATTEMPT_STAGE_SCHEMA = "angler.high-level-multidomain.attempt-stage.v1"
_ATTEMPT_FINAL_SCHEMA = "angler.high-level-multidomain.attempt-final.v1"
_ATTEMPT_RECEIPT_SCHEMA = "angler.high-level-multidomain.attempt-receipt.v1"
_ATTEMPT_RECEIPT_NONAUTHORIZATION = (
    "LOCAL_NONAUTHORIZING_EVALUATION_ATTEMPT_IDENTITY; not an approved Action, "
    "authorization-bearing prospective reservation, EvaluationReceipt, "
    "PromotionDecision, or external-effect permission"
)

_SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_SAFE_ABSOLUTE_PATH = re.compile(r"/[A-Za-z0-9_./-]+\Z")
_SAFE_ENUM = re.compile(r"[A-Z][A-Z0-9_.:-]{0,127}\Z")
_SAFE_KEY = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_GLYPH_ALIAS = re.compile(r"A_[0-9a-f]{16}\Z")


class DiagnosticInvariantError(RuntimeError):
    """Raised when evidence or an integrity boundary fails closed."""


@dataclass(frozen=True, slots=True)
class FrozenInputSpec:
    """One exact, literal, immutable diagnostic input."""

    label: str
    path: Path
    sha256: str
    kind: str
    expected_mode: int

    def __post_init__(self) -> None:
        if _SAFE_ENUM.fullmatch(self.label) is None:
            raise ValueError("input label must be a bounded enum")
        if not self.path.is_absolute() or _SAFE_ABSOLUTE_PATH.fullmatch(
            os.fspath(self.path)
        ) is None:
            raise ValueError("input path must be a safe literal absolute path")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("input sha256 must be one raw lowercase digest")
        if self.kind not in {"JSON", "SQLITE", "SOURCE", "REPORT", "LEAF"}:
            raise ValueError("input kind is not declared")
        if type(self.expected_mode) is not int or self.expected_mode not in {
            0o600,
            0o664,
        }:
            raise ValueError("input expected mode is not an exact admitted mode")


@dataclass(frozen=True, slots=True)
class FileWitness:
    """Internal file identity; the path/inode fields are never serialized."""

    label: str
    path: Path
    sha256: str
    size_bytes: int
    mode: int
    uid: int
    gid: int
    nlink: int
    device: int
    inode: int
    mtime_ns: int

    def redacted(self) -> dict[str, object]:
        return {
            "gid": self.gid,
            "label": self.label,
            "mode": self.mode,
            "nlink": self.nlink,
            "sha256": "sha256:" + self.sha256,
            "size_bytes": self.size_bytes,
            "uid": self.uid,
        }


@dataclass(frozen=True, slots=True)
class LiteralGrammar:
    """A public literal-output grammar, without any semantic answer rule."""

    family: str
    separator: str
    terminator: str
    allowed_tokens: tuple[str, ...]
    maximum_items: int
    token_pattern: str = r"A_[0-9a-f]{16}"

    def __post_init__(self) -> None:
        if _SAFE_ENUM.fullmatch(self.family) is None:
            raise ValueError("grammar family must be a bounded enum")
        if (
            type(self.separator) is not str
            or not self.separator
            or len(self.separator) > 4
            or type(self.terminator) is not str
            or not self.terminator
            or len(self.terminator) > 32
        ):
            raise ValueError("literal separators are invalid")
        if (
            type(self.allowed_tokens) is not tuple
            or not self.allowed_tokens
            or len(self.allowed_tokens) > 256
            or len(set(self.allowed_tokens)) != len(self.allowed_tokens)
            or self.terminator in self.allowed_tokens
        ):
            raise ValueError("allowed literal tokens are invalid")
        if type(self.maximum_items) is not int or not 1 <= self.maximum_items <= 256:
            raise ValueError("maximum_items must be in 1..256")
        try:
            token_re = re.compile(self.token_pattern + r"\Z")
        except re.error as error:
            raise ValueError("token pattern is invalid") from error
        if any(
            type(token) is not str
            or not token
            or len(token.encode("utf-8")) > 256
            or token_re.fullmatch(token) is None
            for token in self.allowed_tokens
        ):
            raise ValueError("allowed token differs from its declared grammar")

    @classmethod
    def from_public_glyph_task(cls, value: Mapping[str, object]) -> "LiteralGrammar":
        """Read only the public glyph output grammar, never its goal/solution."""

        if type(value) is not dict or value.get("family") != "glyph-machine":
            raise DiagnosticInvariantError("target task lacks the public glyph grammar")
        actions = value.get("actions")
        maximum = value.get("maximum_steps")
        if (
            type(actions) is not list
            or not actions
            or any(type(item) is not str for item in actions)
            or type(maximum) is not int
        ):
            raise DiagnosticInvariantError("public glyph grammar is malformed")
        return cls(
            family="GLYPH_MACHINE",
            separator=",",
            terminator="STOP",
            allowed_tokens=tuple(actions),
            maximum_items=maximum,
            token_pattern=r"A_[0-9a-f]{16}",
        )


@dataclass(frozen=True, slots=True)
class AttemptTrace:
    """A redacted finalized-attempt summary; it contains no model-facing text."""

    task_id: str
    arm: str
    phase: str
    attempt_receipt_ref: str
    stage_ref: str
    judgment_ref: str
    parser_disposition: str
    history_class: str
    recall_batch_ref: str | None
    recalled_record_count: int
    candidate_classes: tuple[str, ...]
    selected_index: int | None
    selected_class: str
    selection_ref: str | None
    reservation_ref: str | None
    response_class: str
    response_length: int
    selected_response_equal: bool
    objective_success: bool
    objective_disposition: str
    execution_request_ref: str | None
    execution_receipt_ref: str | None
    learner_parent_ref: str | None
    learner_child_ref: str | None
    learner_sequence: int | None
    episode_ref: str | None

    def __post_init__(self) -> None:
        _require_ref(self.task_id, "task_id")
        _require_ref(self.attempt_receipt_ref, "attempt_receipt_ref")
        _require_ref(self.stage_ref, "stage_ref")
        _require_ref(self.judgment_ref, "judgment_ref")
        if self.arm not in EXPECTED_ARMS or self.phase not in {
            "ADAPTATION",
            "DEVELOPMENT",
            "FINAL",
        }:
            raise ValueError("attempt arm or phase differs")
        if self.parser_disposition not in {"ADMITTED", "MALFORMED"}:
            raise ValueError("parser disposition differs")
        if self.history_class not in {"NO_PERSISTENT_HISTORY", "FROZEN_RECALL"}:
            raise ValueError("history class differs")
        _optional_ref(self.recall_batch_ref, "recall_batch_ref")
        if (
            type(self.recalled_record_count) is not int
            or not 0 <= self.recalled_record_count <= 256
            or type(self.candidate_classes) is not tuple
            or any(_SAFE_ENUM.fullmatch(item) is None for item in self.candidate_classes)
        ):
            raise ValueError("redacted candidate or recall evidence differs")
        if self.selected_index is not None and (
            type(self.selected_index) is not int
            or not 0 <= self.selected_index < len(self.candidate_classes)
        ):
            raise ValueError("selected index differs")
        for value, label in (
            (self.selection_ref, "selection_ref"),
            (self.reservation_ref, "reservation_ref"),
            (self.execution_request_ref, "execution_request_ref"),
            (self.execution_receipt_ref, "execution_receipt_ref"),
            (self.learner_parent_ref, "learner_parent_ref"),
            (self.learner_child_ref, "learner_child_ref"),
            (self.episode_ref, "episode_ref"),
        ):
            _optional_ref(value, label)
        for value in (self.selected_class, self.response_class, self.objective_disposition):
            if _SAFE_ENUM.fullmatch(value) is None:
                raise ValueError("attempt enum differs")
        if (
            type(self.response_length) is not int
            or not 0 <= self.response_length <= MAX_TEXT_BYTES
            or type(self.selected_response_equal) is not bool
            or type(self.objective_success) is not bool
        ):
            raise ValueError("response summary differs")
        if self.learner_sequence is not None and (
            type(self.learner_sequence) is not int or self.learner_sequence < 1
        ):
            raise ValueError("learner sequence differs")

    def redacted(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "attempt_receipt_ref": self.attempt_receipt_ref,
            "candidate_classes": list(self.candidate_classes),
            "episode_ref": self.episode_ref,
            "execution_receipt_ref": self.execution_receipt_ref,
            "execution_request_ref": self.execution_request_ref,
            "history_class": self.history_class,
            "judgment_ref": self.judgment_ref,
            "learner_child_ref": self.learner_child_ref,
            "learner_parent_ref": self.learner_parent_ref,
            "learner_sequence": self.learner_sequence,
            "objective_disposition": self.objective_disposition,
            "objective_success": self.objective_success,
            "parser_disposition": self.parser_disposition,
            "phase": self.phase,
            "recall_batch_ref": self.recall_batch_ref,
            "recalled_record_count": self.recalled_record_count,
            "reservation_ref": self.reservation_ref,
            "response_class": self.response_class,
            "response_length": self.response_length,
            "selected_class": self.selected_class,
            "selected_index": self.selected_index,
            "selected_response_equal": self.selected_response_equal,
            "selection_ref": self.selection_ref,
            "stage_ref": self.stage_ref,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceRow:
    """Reference-only retained provenance used for an exact rejoin."""

    record_ref: str
    lineage_ref: str
    ancestry_refs: tuple[str, ...]
    provenance_class: str = "RETAINED_ACQUISITION"

    def __post_init__(self) -> None:
        _require_ref(self.record_ref, "record_ref")
        _require_ref(self.lineage_ref, "lineage_ref")
        if (
            type(self.ancestry_refs) is not tuple
            or len(set(self.ancestry_refs)) != len(self.ancestry_refs)
        ):
            raise ValueError("ancestry references differ")
        for value in self.ancestry_refs:
            _require_ref(value, "ancestry_ref")
        if _SAFE_ENUM.fullmatch(self.provenance_class) is None:
            raise ValueError("provenance class differs")


FROZEN_INPUTS = (
    FrozenInputSpec(
        "R2_RESULT",
        Path("/opt/angler/results/high-level-multidomain-v1-r2.json"),
        "c2316da22208cf0a8693f32f4ea28060c529329df1c6d7d67841b56720aca676",
        "JSON",
        0o600,
    ),
    FrozenInputSpec(
        "R2_RESULT_REPORT",
        Path(
            "/opt/angler/src/angler/docs/reports/"
            "HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_R2_RESULT.md"
        ),
        "4aae9d581333eebc468ed5c44f52c62b3784cc98ac6896afada70628c95960d1",
        "REPORT",
        0o664,
    ),
    FrozenInputSpec(
        "R2_LEDGER",
        Path(
            "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/"
            "evaluation-v1-r2/attempt-evidence.sqlite3"
        ),
        "914e1695cc4c9187a60f48a035632dbe5e849056e113bab4295ca36f5bbb4eca",
        "SQLITE",
        0o600,
    ),
    FrozenInputSpec(
        "R2_ARM_FACTORY_AUDIT",
        Path(
            "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/"
            "evaluation-v1-r2/arm-factory-audit.json"
        ),
        "eb6f4e16336f83b4cee95ecc9ec4803af99045dd1ba904373599db8a3f0b4e82",
        "JSON",
        0o600,
    ),
    FrozenInputSpec(
        "R2_SOURCE_MANIFEST",
        Path(
            "/opt/angler/src/angler/experiments/manifests/"
            "high-level-multidomain-v1-r2.json"
        ),
        "842e39e713c87352a9dbc5e04c8cd73ab74a1b32fceb9aaa72c1d1fd75a4344d",
        "JSON",
        0o600,
    ),
    FrozenInputSpec(
        "R2_RUNNER",
        Path(
            "/opt/angler/src/angler/experiments/runners/"
            "high_level_multidomain_v1_r2.py"
        ),
        "fbd10104ee566e327e812f24e6ea44d30e405d4a2dea2650ebd94dad58fee574",
        "SOURCE",
        0o664,
    ),
    FrozenInputSpec(
        "COGNITIVE_TRANSACTION_STORE_SOURCE",
        Path(
            "/opt/angler/src/angler/src/angler/runtime/"
            "cognitive_transaction_store.py"
        ),
        "4c8add9e1b49649b381b0f4dad439709f7b4691cec514c88b39b386bc8a28c0a",
        "SOURCE",
        0o664,
    ),
    FrozenInputSpec(
        "COGNITIVE_ACQUISITION_SOURCE",
        Path(
            "/opt/angler/src/angler/src/angler/memory/cognitive_acquisition.py"
        ),
        "e93c9f181fe1938a4c033a55a88827ba3d0ca5b8b9f46cb1b5569bc4987d36e7",
        "SOURCE",
        0o664,
    ),
    FrozenInputSpec(
        "COGNITIVE_CONTRACTS_SOURCE",
        Path("/opt/angler/src/angler/src/angler/cognition/contracts.py"),
        "cb2fe5d597c10550425a03951aaecf2e9b104065e6ea32adaba30d1d61b1eb77",
        "SOURCE",
        0o664,
    ),
    FrozenInputSpec(
        "R2_PREDECESSOR_LEAF",
        Path(
            "/opt/angler/src/angler/docs/blueprints/branches/science/work/"
            "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-"
            "LINEAGE-SEQUENCE-RECOVERY-001.md"
        ),
        "a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455",
        "LEAF",
        0o664,
    ),
    FrozenInputSpec(
        "DIAGNOSTIC_LEAF",
        Path(
            "/opt/angler/src/angler/docs/blueprints/branches/science/work/"
            "ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-"
            "EVIDENCE-TRACE-DIAGNOSIS-001.md"
        ),
        "eedc848394a088e347db68db23f24fabab409315a0775747597edc76abd5f960",
        "LEAF",
        0o664,
    ),
    FrozenInputSpec(
        "LINEAGE_102E3196",
        Path(
            "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/"
            "evaluation-v1-r2/arm-runtime/lineages/"
            "102e3196e05f0ce031e8585c2eef0f94b48acf1e91e272bc998d5b0cbc30a395/"
            "cognitive.sqlite3"
        ),
        "54a58c587c2c79ec27999ff8f9c48e1607f64e4f7c8d1fcca3e17af6f2d72a57",
        "SQLITE",
        0o600,
    ),
    FrozenInputSpec(
        "LINEAGE_75CADC26",
        Path(
            "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/"
            "evaluation-v1-r2/arm-runtime/lineages/"
            "75cadc260b52f873a7893ea7c18a3bb027ed25d98fabb300942e429ac08df706/"
            "cognitive.sqlite3"
        ),
        "c27f14fce291334eb785cd02ca77560021a61546c5cabbc0c0db87738d5bd022",
        "SQLITE",
        0o600,
    ),
    FrozenInputSpec(
        "LINEAGE_E4ACB76A",
        Path(
            "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/"
            "evaluation-v1-r2/arm-runtime/lineages/"
            "e4acb76aae39f554ee54ad98854eb67f83c7ffc7a0ab2961c7709d31bfb248de/"
            "cognitive.sqlite3"
        ),
        "748a7499667382c754fa389b15c76af57e20f29bacab25a30d51200f7205cfda",
        "SQLITE",
        0o600,
    ),
    FrozenInputSpec(
        "LINEAGE_F4F478B1",
        Path(
            "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/"
            "evaluation-v1-r2/arm-runtime/lineages/"
            "f4f478b15524d0d88cb6363ae2484c05acfbc0c608d8ab1153cdebc01f324312/"
            "cognitive.sqlite3"
        ),
        "534191cc6e404da2880681074906f550a91c811e5d80df45442212eb834dc9fe",
        "SQLITE",
        0o600,
    ),
)


def _require_ref(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise DiagnosticInvariantError(f"{label} must be an exact content reference")
    return value


def _optional_ref(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_ref(value, label)


def _reject_constant(value: str) -> None:
    raise DiagnosticInvariantError("JSON contains a non-finite numeric constant")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise DiagnosticInvariantError("JSON object keys are invalid or duplicated")
        result[key] = value
    return result


def _validate_exact_json(value: object, *, depth: int = 0) -> None:
    """Mirror the frozen R2 JSON-native and Unicode-NFC contract exactly."""

    if depth > 64:
        raise DiagnosticInvariantError("canonical JSON exceeds the nesting ceiling")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise DiagnosticInvariantError("canonical JSON numbers must be finite")
        return
    if type(value) is str:
        if value != unicodedata.normalize("NFC", value):
            raise DiagnosticInvariantError("canonical JSON strings must be Unicode NFC")
        return
    if type(value) is list:
        for item in value:
            _validate_exact_json(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or key != unicodedata.normalize("NFC", key):
                raise DiagnosticInvariantError(
                    "canonical JSON object keys must be exact NFC strings"
                )
            _validate_exact_json(item, depth=depth + 1)
        return
    raise DiagnosticInvariantError("canonical JSON admits only exact JSON-native values")


def canonical_json_bytes(value: object) -> bytes:
    """Mirror frozen R2 finite UTF-8/NFC JSON bytes without repair."""

    _validate_exact_json(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise DiagnosticInvariantError("value is not canonical JSON") from error


def parse_canonical_json(
    raw: bytes | str,
    *,
    label: str = "JSON",
    maximum_bytes: int = MAX_JSON_INPUT_BYTES,
) -> object:
    """Parse exact canonical JSON, rejecting duplicates and alternate bytes."""

    if isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8")
        except UnicodeError as error:
            raise DiagnosticInvariantError(f"{label} is not strict UTF-8") from error
    elif type(raw) is bytes:
        encoded = raw
    else:
        raise TypeError("canonical JSON input must be exact bytes or text")
    if not encoded or len(encoded) > maximum_bytes:
        raise DiagnosticInvariantError(f"{label} bytes are empty or over limit")
    try:
        text = encoded.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosticInvariantError(f"{label} is not strict JSON") from error
    if canonical_json_bytes(value) != encoded:
        raise DiagnosticInvariantError(f"{label} bytes are not canonical")
    return value


def content_ref(kind: str, value: object) -> str:
    """Recompute the exact frozen R2 domain-separated content reference."""

    if (
        type(kind) is not str
        or not kind
        or len(kind) > 128
        or kind != kind.strip()
        or kind != unicodedata.normalize("NFC", kind)
    ):
        raise DiagnosticInvariantError("content kind must be bounded stripped NFC text")
    try:
        domain = kind.encode("ascii")
    except UnicodeEncodeError as error:
        raise DiagnosticInvariantError("content kind must be ASCII") from error
    return "sha256:" + hashlib.sha256(
        domain + b"\x00" + canonical_json_bytes(value)
    ).hexdigest()


def _evaluator_record_digest(kind: str, payload: Mapping[str, object]) -> str:
    if type(kind) is not str or not kind or len(kind) > 128:
        raise DiagnosticInvariantError("evaluator digest kind differs")
    return "sha256:" + hashlib.sha256(
        canonical_json_bytes(
            {
                "kind": kind,
                "payload": dict(payload),
                "schema": "angler.high-level-multidomain.v1",
            }
        )
    ).hexdigest()


def _open_nofollow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    while True:
        block = os.read(fd, 1_048_576)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _sidecar_paths(path: Path) -> tuple[Path, ...]:
    return tuple(Path(os.fspath(path) + suffix) for suffix in ("-journal", "-shm", "-wal"))


def _require_sidecars_absent(path: Path) -> None:
    for sidecar in _sidecar_paths(path):
        try:
            os.lstat(sidecar)
        except FileNotFoundError:
            continue
        raise DiagnosticInvariantError("SQLite sidecar exists")


def capture_frozen_input(spec: FrozenInputSpec) -> FileWitness:
    """Capture and hash an exact non-linked input at its frozen permission mode."""

    if type(spec) is not FrozenInputSpec:
        raise TypeError("spec must be an exact FrozenInputSpec")
    try:
        before = os.lstat(spec.path)
    except FileNotFoundError as error:
        raise DiagnosticInvariantError("frozen input is absent") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_gid != os.getegid()
        or stat.S_IMODE(before.st_mode) != spec.expected_mode
    ):
        raise DiagnosticInvariantError("frozen input metadata differs")
    if spec.kind == "SQLITE":
        _require_sidecars_absent(spec.path)
    fd = _open_nofollow(spec.path)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise DiagnosticInvariantError("frozen input changed before hashing")
        observed = _sha256_fd(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        observed != spec.sha256
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise DiagnosticInvariantError("frozen input hash or identity differs")
    return FileWitness(
        label=spec.label,
        path=spec.path,
        sha256=observed,
        size_bytes=before.st_size,
        mode=stat.S_IMODE(before.st_mode),
        uid=before.st_uid,
        gid=before.st_gid,
        nlink=before.st_nlink,
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
    )


def verify_frozen_input_unchanged(
    spec: FrozenInputSpec,
    before: FileWitness,
) -> FileWitness:
    """Rejoin a complete before witness after all reads."""

    if type(before) is not FileWitness or before.path != spec.path:
        raise TypeError("before witness does not belong to its exact input")
    after = capture_frozen_input(spec)
    if after != before:
        raise DiagnosticInvariantError("frozen input metadata changed")
    return after


def immutable_sqlite_uri(spec: FrozenInputSpec) -> str:
    """Return the only accepted SQLite URI for an exact input."""

    if type(spec) is not FrozenInputSpec or spec.kind != "SQLITE":
        raise TypeError("SQLite input must be an exact SQLite FrozenInputSpec")
    path = os.fspath(spec.path)
    if _SAFE_ABSOLUTE_PATH.fullmatch(path) is None or any(c in path for c in "?%#"):
        raise DiagnosticInvariantError("SQLite path is not safe for a literal URI")
    return f"file:{path}?mode=ro&immutable=1"


@contextlib.contextmanager
def immutable_sqlite(spec: FrozenInputSpec) -> Iterator[sqlite3.Connection]:
    """Open one verified DB with immutable read-only URI and query_only enabled."""

    before = capture_frozen_input(spec)
    uri = immutable_sqlite_uri(spec)
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        observed = connection.execute("PRAGMA query_only").fetchone()
        if observed != (1,):
            raise DiagnosticInvariantError("SQLite query_only did not bind")
        yield connection
    finally:
        connection.close()
        _require_sidecars_absent(spec.path)
    verify_frozen_input_unchanged(spec, before)


def classify_declared_grammar(grammar: LiteralGrammar, text: str) -> str:
    """Classify syntax only; never compare with a goal or hidden answer."""

    if type(grammar) is not LiteralGrammar or type(text) is not str:
        raise TypeError("grammar and text must be exact declared values")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise DiagnosticInvariantError("text is over the diagnostic bound")
    if not text:
        return "EMPTY"
    if text != text.strip():
        return "MALFORMED_SHAPE"
    parts = text.split(grammar.separator)
    token_re = re.compile(grammar.token_pattern + r"\Z")
    if (
        1 <= len(parts) <= grammar.maximum_items + 1
        and parts[-1] == grammar.terminator
        and parts.count(grammar.terminator) == 1
        and all(token_re.fullmatch(item) is not None for item in parts[:-1])
    ):
        if all(item in grammar.allowed_tokens for item in parts[:-1]):
            return "CONFORMANT"
        return "FOREIGN_TOKEN"
    if (
        grammar.separator in text
        or grammar.terminator in text
        or any(token_re.fullmatch(item) is not None for item in parts)
    ):
        return "MALFORMED_SHAPE"
    return "OUTSIDE_DECLARED_GRAMMAR"


def classify_candidate(grammar: LiteralGrammar, text: str) -> str:
    """Return a redacted candidate class derived only from public syntax."""

    return {
        "CONFORMANT": "DECLARED_GRAMMAR_CONFORMANT",
        "FOREIGN_TOKEN": "DECLARED_SHAPE_FOREIGN_TOKEN",
        "MALFORMED_SHAPE": "DECLARED_SHAPE_MALFORMED",
        "OUTSIDE_DECLARED_GRAMMAR": "OUTSIDE_DECLARED_GRAMMAR",
        "EMPTY": "EMPTY",
    }[classify_declared_grammar(grammar, text)]


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise DiagnosticInvariantError(f"{label} must be an exact object")
    return value


def _phase(value: object) -> str:
    if value not in {"adaptation", "development", "final"}:
        raise DiagnosticInvariantError("attempt phase differs")
    return str(value).upper()


def extract_attempt_trace(
    stage_bytes: bytes,
    judgment_bytes: bytes,
    *,
    stage_ref: str,
    judgment_ref: str,
    recalled_record_count: int | None = None,
) -> AttemptTrace:
    """Extract one reference-only trace from canonical finalized evidence."""

    _require_ref(stage_ref, "stage_ref")
    _require_ref(judgment_ref, "judgment_ref")
    stage = _mapping(parse_canonical_json(stage_bytes, label="STAGE"), "stage")
    judgment = _mapping(
        parse_canonical_json(judgment_bytes, label="JUDGMENT"), "judgment"
    )
    if stage_ref != content_ref("attempt-stage", stage):
        raise DiagnosticInvariantError("attempt stage reference differs")
    if judgment_ref != content_ref("attempt-judgment", judgment):
        raise DiagnosticInvariantError("attempt judgment reference differs")
    task_id = _require_ref(stage.get("task_id"), "task_id")
    arm = stage.get("arm")
    if arm not in EXPECTED_ARMS or judgment.get("task_id") != task_id:
        raise DiagnosticInvariantError("stage and judgment task/arm binding differs")
    if judgment.get("arm") != arm:
        raise DiagnosticInvariantError("judgment arm differs")
    receipt_ref = _require_ref(stage.get("attempt_receipt_ref"), "attempt receipt")
    if judgment.get("attempt_receipt_ref") != receipt_ref:
        raise DiagnosticInvariantError("judgment receipt binding differs")
    parser = stage.get("parser_disposition")
    if parser not in {"ADMITTED", "MALFORMED"}:
        raise DiagnosticInvariantError("proposal parser disposition differs")

    request = _mapping(stage.get("proposal_request"), "proposal request")
    rendered_task = request.get("task")
    if type(rendered_task) is not str:
        raise DiagnosticInvariantError("public task rendering is absent")
    public_task = _mapping(
        parse_canonical_json(rendered_task, label="PUBLIC_TASK"),
        "public task",
    )
    grammar = LiteralGrammar.from_public_glyph_task(public_task)
    recalled = request.get("recalled_evidence")
    if type(recalled) is not list or any(type(item) is not str for item in recalled):
        raise DiagnosticInvariantError("recalled evidence shape differs")
    if arm == "QWEN_ONLY":
        if recalled != [NO_HISTORY_SENTINEL] or stage.get("frozen_recall_ref") is not None:
            raise DiagnosticInvariantError("QWEN_ONLY history boundary differs")
        history_class = "NO_PERSISTENT_HISTORY"
        default_recalled_count = 0
        recall_ref = None
    else:
        if not recalled or NO_HISTORY_SENTINEL in recalled:
            raise DiagnosticInvariantError("history arm lacks bounded recall")
        history_class = "FROZEN_RECALL"
        recall_ref = _require_ref(stage.get("frozen_recall_ref"), "recall batch")
        default_recalled_count = len(recalled)
    if recalled_record_count is None:
        recalled_count = default_recalled_count
    elif type(recalled_record_count) is not int or recalled_record_count < 0:
        raise TypeError("recalled_record_count must be a non-negative integer")
    else:
        recalled_count = recalled_record_count
    if recalled_count != default_recalled_count:
        raise DiagnosticInvariantError("recall text/reference cardinality differs")

    proposals = stage.get("proposals")
    if type(proposals) is not list or any(type(item) is not str for item in proposals):
        raise DiagnosticInvariantError("proposal evidence differs")
    candidate_classes = tuple(classify_candidate(grammar, item) for item in proposals)
    selection = stage.get("selection")
    raw_response = stage.get("raw_response")
    if type(raw_response) is not str:
        raise DiagnosticInvariantError("recorded response type differs")

    selected_index: int | None = None
    selected_text = ""
    selected_class = "NOT_SELECTED"
    selection_ref: str | None = None
    reservation_ref: str | None = None
    execution_request_ref: str | None = None
    execution_receipt_ref: str | None = None
    if parser == "MALFORMED":
        if proposals or selection is not None or raw_response:
            raise DiagnosticInvariantError("malformed attempt contains admitted material")
        response_class = "NOT_EXECUTED"
    else:
        selected = _mapping(selection, "selection")
        selected_index = selected.get("selected_index")  # type: ignore[assignment]
        if (
            type(selected_index) is not int
            or not 0 <= selected_index < len(proposals)
            or selected.get("selected_trace") != proposals[selected_index]
        ):
            raise DiagnosticInvariantError("selected candidate binding differs")
        selected_text = proposals[selected_index]
        selected_class = candidate_classes[selected_index]
        selection_ref = _optional_ref(
            selected.get("selection_ref", selected.get("decision_evidence_ref")),
            "selection reference",
        )
        if selection_ref is None:
            raise DiagnosticInvariantError("selection reference is absent")
        reservation_ref = _optional_ref(selected.get("reservation_ref"), "reservation")
        execution_request = _mapping(stage.get("execution_request"), "execution request")
        execution_receipt = _mapping(stage.get("execution_receipt"), "execution receipt")
        execution_request_ref = _require_ref(
            execution_request.get("execution_request_ref"), "execution request"
        )
        execution_receipt_ref = _require_ref(
            execution_receipt.get("execution_receipt_ref"), "execution receipt"
        )
        response_class = classify_declared_grammar(grammar, raw_response)

    objective = _mapping(judgment.get("objective_judgment"), "objective judgment")
    if (
        objective.get("task_id") != task_id
        or objective.get("arm") != arm
        or objective.get("attempt_receipt_ref") != receipt_ref
        or objective.get("raw_response") != raw_response
    ):
        raise DiagnosticInvariantError("objective judgment binding differs")
    score = objective.get("score")
    if type(score) is not float or score not in {0.0, 1.0}:
        raise DiagnosticInvariantError("objective score differs")
    objective_success = score == 1.0
    expected_disposition = "SUCCESS" if objective_success else "UNSUCCESSFUL"
    if objective.get("disposition") != expected_disposition:
        raise DiagnosticInvariantError("objective disposition differs")

    transition = judgment.get("learner_transition")
    if transition is None:
        learner_parent_ref = None
        learner_child_ref = None
        learner_sequence = None
        episode_ref = None
    else:
        learned = _mapping(transition, "learner transition")
        learner_parent_ref = _require_ref(
            learned.get("parent_state_digest"), "learner parent"
        )
        learner_child_ref = _require_ref(
            learned.get("child_state_digest"), "learner child"
        )
        learner_sequence = learned.get("sequence")  # type: ignore[assignment]
        if type(learner_sequence) is not int or learner_sequence < 1:
            raise DiagnosticInvariantError("learner sequence differs")
        episode_ref = _require_ref(learned.get("episode_ref"), "episode")

    return AttemptTrace(
        task_id=task_id,
        arm=arm,  # type: ignore[arg-type]
        phase=_phase(stage.get("phase")),
        attempt_receipt_ref=receipt_ref,
        stage_ref=stage_ref,
        judgment_ref=judgment_ref,
        parser_disposition=parser,  # type: ignore[arg-type]
        history_class=history_class,
        recall_batch_ref=recall_ref,
        recalled_record_count=recalled_count,
        candidate_classes=candidate_classes,
        selected_index=selected_index,
        selected_class=selected_class,
        selection_ref=selection_ref,
        reservation_ref=reservation_ref,
        response_class=response_class,
        response_length=len(raw_response.encode("utf-8")),
        selected_response_equal=parser == "ADMITTED" and raw_response == selected_text,
        objective_success=objective_success,
        objective_disposition=expected_disposition,
        execution_request_ref=execution_request_ref,
        execution_receipt_ref=execution_receipt_ref,
        learner_parent_ref=learner_parent_ref,
        learner_child_ref=learner_child_ref,
        learner_sequence=learner_sequence,
        episode_ref=episode_ref,
    )


def equality_groups(values: Mapping[str, object]) -> tuple[tuple[str, ...], ...]:
    """Group labels by exact canonical equality without exposing compared values."""

    if type(values) is not dict or not values:
        raise TypeError("values must be one nonempty exact mapping")
    buckets: dict[bytes, list[str]] = {}
    order = {arm: index for index, arm in enumerate(EXPECTED_ARMS)}
    for label, value in values.items():
        if type(label) is not str or _SAFE_ENUM.fullmatch(label) is None:
            raise ValueError("equality label must be a bounded enum")
        buckets.setdefault(canonical_json_bytes(value), []).append(label)
    groups = [
        tuple(sorted(labels, key=lambda item: (order.get(item, len(order)), item)))
        for labels in buckets.values()
    ]
    groups.sort(key=lambda group: (order.get(group[0], len(order)), group[0]))
    return tuple(groups)


TRACE_DIVERGENCE_ORDER = (
    "parser_disposition",
    "history_preimage",
    "candidate_classes",
    "selection",
    "response_conformance",
    "objective_disposition",
    "objective_score",
)


def first_divergence(
    reference: Mapping[str, object],
    observed: Mapping[str, object],
    *,
    ordered_fields: Sequence[str] = TRACE_DIVERGENCE_ORDER,
) -> str:
    """Return the first exact observational divergence, never a causal claim."""

    if type(reference) is not dict or type(observed) is not dict:
        raise TypeError("divergence inputs must be exact mappings")
    if type(ordered_fields) not in {tuple, list} or not ordered_fields:
        raise TypeError("ordered_fields must be one bounded sequence")
    for field in ordered_fields:
        if type(field) is not str or field not in reference or field not in observed:
            raise DiagnosticInvariantError("divergence field is absent")
        if canonical_json_bytes(reference[field]) != canonical_json_bytes(observed[field]):
            enum = field.upper()
            if _SAFE_ENUM.fullmatch(enum) is None:
                raise DiagnosticInvariantError("divergence enum is unsafe")
            return enum
    return "NONE"


def validate_unique_cohort(
    attempts: Sequence[AttemptTrace],
    *,
    expected_task_count: int = EXPECTED_COHORT_TASKS,
    expected_arms: Sequence[str] = EXPECTED_ARMS,
) -> tuple[str, ...]:
    """Require exactly one seven-arm matrix for three QWEN_ONLY successes."""

    values = tuple(attempts)
    arms = tuple(expected_arms)
    if (
        type(expected_task_count) is not int
        or expected_task_count < 1
        or type(expected_arms) not in {tuple, list}
        or len(arms) != len(set(arms))
        or set(arms) != set(EXPECTED_ARMS)
    ):
        raise ValueError("cohort expectation differs from the frozen design")
    if len(values) != expected_task_count * len(arms):
        raise DiagnosticInvariantError("cohort attempt cardinality differs")
    by_task: dict[str, dict[str, AttemptTrace]] = {}
    receipts: set[str] = set()
    for item in values:
        if type(item) is not AttemptTrace or item.phase != "FINAL":
            raise DiagnosticInvariantError("cohort contains a non-final attempt")
        if item.attempt_receipt_ref in receipts:
            raise DiagnosticInvariantError("cohort receipt is duplicated")
        receipts.add(item.attempt_receipt_ref)
        task_arms = by_task.setdefault(item.task_id, {})
        if item.arm in task_arms:
            raise DiagnosticInvariantError("cohort task-arm is duplicated")
        task_arms[item.arm] = item
    if len(by_task) != expected_task_count:
        raise DiagnosticInvariantError("cohort task cardinality differs")
    for task_arms in by_task.values():
        if set(task_arms) != set(arms):
            raise DiagnosticInvariantError("cohort lacks its exact seven arms")
        qwen = task_arms["QWEN_ONLY"]
        if not qwen.objective_success:
            raise DiagnosticInvariantError("cohort task lacks its QWEN_ONLY success")
    return tuple(sorted(by_task))


def rejoin_provenance(
    target_record_keys: Sequence[tuple[str, str]],
    rows: Sequence[ProvenanceRow],
    *,
    required_ancestry_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Rejoin repeated occurrences by exact ``(lineage_ref, record_ref)`` key."""

    targets = tuple(target_record_keys)
    ancestry = tuple(required_ancestry_refs)
    evidence = tuple(rows)
    if not targets or any(
        type(key) is not tuple or len(key) != 2 for key in targets
    ):
        raise DiagnosticInvariantError("target provenance identities differ")
    for value in (*[item for key in targets for item in key], *ancestry):
        _require_ref(value, "provenance reference")
    by_key: dict[tuple[str, str], ProvenanceRow] = {}
    for row in evidence:
        if type(row) is not ProvenanceRow:
            raise TypeError("provenance rows must be exact ProvenanceRow values")
        key = (row.lineage_ref, row.record_ref)
        if key in by_key:
            raise DiagnosticInvariantError("retained provenance identity is duplicated")
        by_key[key] = row
    missing = tuple(value for value in targets if value not in by_key)
    if missing:
        raise DiagnosticInvariantError("required retained provenance is absent")
    matched = tuple(by_key[value] for value in targets)
    observed_ancestry = {value for row in matched for value in row.ancestry_refs}
    absent_ancestry = tuple(value for value in ancestry if value not in observed_ancestry)
    if absent_ancestry:
        raise DiagnosticInvariantError("required adaptation ancestry is absent")
    return {
        "all_rejoined": True,
        "ancestry_ref_count": len(ancestry),
        "lineage_count": len({row.lineage_ref for row in matched}),
        "occurrence_count": len(targets),
        "provenance_classes": sorted({row.provenance_class for row in matched}),
        "rejoined_ancestry_ref_count": len(ancestry),
        "unique_record_count": len(set(targets)),
    }


_FORBIDDEN_OUTPUT_KEYS = {
    "answer",
    "answers",
    "content",
    "database_row",
    "generated_token_ids",
    "judgment_bytes",
    "mechanism",
    "mechanisms",
    "prompt",
    "prompt_text",
    "proposal",
    "proposals",
    "raw_response",
    "recalled_evidence",
    "recalled_text",
    "request",
    "response",
    "response_text",
    "route",
    "routes",
    "seed",
    "seeds",
    "selected_trace",
    "solution",
    "solutions",
    "stage_bytes",
    "task",
    "task_payload",
    "token_ids",
}


def validate_redacted_output(
    value: object,
    *,
    forbidden_values: Sequence[str] = (),
) -> None:
    """Fail closed unless every durable value is non-raw and non-reversible."""

    if type(value) is not dict:
        raise DiagnosticInvariantError("diagnostic output must be one object")
    _validate_closed_diagnostic_output(value)
    if value.get("schema") != DIAGNOSTIC_SCHEMA or value.get("identity") != DIAGNOSTIC_IDENTITY:
        raise DiagnosticInvariantError("diagnostic output identity differs")
    forbidden = tuple(forbidden_values)
    if type(forbidden_values) not in {tuple, list} or any(
        type(item) is not str or not item for item in forbidden
    ):
        raise TypeError("forbidden_values must be one exact nonempty-text sequence")
    forbidden_encodings: set[str] = set()
    for raw in forbidden:
        encoded = raw.encode("utf-8")
        forbidden_encodings.update(
            {
                raw,
                encoded.hex(),
                base64.b32encode(encoded).decode("ascii"),
                base64.b32encode(encoded).decode("ascii").rstrip("="),
                base64.b64encode(encoded).decode("ascii"),
                base64.urlsafe_b64encode(encoded).decode("ascii"),
                base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("="),
            }
        )

    def walk(item: object, key: str | None = None) -> None:
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if item < 0:
                raise DiagnosticInvariantError("diagnostic integer is negative")
            return
        if type(item) is float:
            raise DiagnosticInvariantError("diagnostic output cannot contain floats")
        if type(item) is str:
            if item in forbidden_encodings or any(
                len(raw) >= 8 and raw in item for raw in forbidden
            ):
                raise DiagnosticInvariantError("diagnostic output contains forbidden raw text")
            if key == "schema":
                if item != DIAGNOSTIC_SCHEMA:
                    raise DiagnosticInvariantError("diagnostic schema differs")
                return
            if key == "identity":
                if item != DIAGNOSTIC_IDENTITY:
                    raise DiagnosticInvariantError("diagnostic identity differs")
                return
            if key == "source_contract" and item in {
                _LEGACY_EPISODE_CONTRACT,
                _PROSPECTIVE_BATCH_CONTRACT,
                _PROSPECTIVE_RESOLUTION_CONTRACT,
            }:
                return
            if _SHA256.fullmatch(item) is not None or _SAFE_ENUM.fullmatch(item) is not None:
                return
            raise DiagnosticInvariantError("diagnostic output contains unsafe text")
        if type(item) is list:
            for child in item:
                walk(child, key)
            return
        if type(item) is dict:
            for child_key, child in item.items():
                if (
                    type(child_key) is not str
                    or _SAFE_KEY.fullmatch(child_key) is None
                    or child_key in _FORBIDDEN_OUTPUT_KEYS
                    or child_key.endswith(("_payload", "_text", "_tokens"))
                    or "prompt" in child_key
                    or "seed" in child_key
                ):
                    raise DiagnosticInvariantError("diagnostic output key is forbidden")
                walk(child, child_key)
            return
        raise DiagnosticInvariantError("diagnostic output contains a forbidden type")

    walk(value)


def publish_create_once(
    path: str | os.PathLike[str],
    value: object,
    *,
    expected_path: str | os.PathLike[str],
    maximum_bytes: int = MAX_OUTPUT_BYTES,
    forbidden_values: Sequence[str] = (),
) -> FileWitness:
    """Publish canonical redacted bytes directly with exclusive no-follow create."""

    target = Path(path)
    expected = Path(expected_path)
    if (
        target != expected
        or not target.is_absolute()
        or _SAFE_ABSOLUTE_PATH.fullmatch(os.fspath(target)) is None
    ):
        raise DiagnosticInvariantError("output path differs from its literal scope")
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= MAX_OUTPUT_BYTES:
        raise ValueError("output ceiling differs")
    validate_redacted_output(value, forbidden_values=forbidden_values)
    raw = canonical_json_bytes(value)
    if not raw or len(raw) > maximum_bytes:
        raise DiagnosticInvariantError("canonical output is empty or over limit")
    # Round-trip before the irreversible exclusive creation.
    if parse_canonical_json(raw, label="DIAGNOSTIC_OUTPUT", maximum_bytes=maximum_bytes) != value:
        raise DiagnosticInvariantError("canonical output round-trip differs")

    parent = target.parent
    try:
        parent_stat = os.lstat(parent)
    except FileNotFoundError as error:
        raise DiagnosticInvariantError("output parent is absent") from error
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_ISLNK(parent_stat.st_mode)
        or os.path.realpath(parent) != os.fspath(parent)
        or parent_stat.st_uid != os.geteuid()
        or parent_stat.st_gid != os.getegid()
    ):
        raise DiagnosticInvariantError("output parent metadata differs")
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        raise DiagnosticInvariantError("create-once output already exists")

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created = False
    fd = -1
    created_identity: tuple[int, int] | None = None
    try:
        fd = os.open(target, flags, 0o600)
        created = True
        opened = os.fstat(fd)
        created_identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or opened.st_gid != os.getegid()
        ):
            raise DiagnosticInvariantError("exclusive output metadata differs")
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise DiagnosticInvariantError("exclusive output write made no progress")
            offset += written
        os.fsync(fd)
    except FileExistsError as error:
        raise DiagnosticInvariantError("create-once output already exists") from error
    finally:
        if fd >= 0:
            os.close(fd)
    if not created:
        raise DiagnosticInvariantError("exclusive output was not created")

    final_stat = os.lstat(target)
    if (
        not stat.S_ISREG(final_stat.st_mode)
        or stat.S_ISLNK(final_stat.st_mode)
        or stat.S_IMODE(final_stat.st_mode) != 0o600
        or final_stat.st_nlink != 1
        or final_stat.st_uid != os.geteuid()
        or final_stat.st_gid != os.getegid()
        or final_stat.st_size != len(raw)
        or final_stat.st_dev != parent_stat.st_dev
        or created_identity != (final_stat.st_dev, final_stat.st_ino)
    ):
        raise DiagnosticInvariantError("closed output metadata differs")
    read_fd = _open_nofollow(target)
    try:
        opened = os.fstat(read_fd)
        observed_raw = bytearray()
        while True:
            block = os.read(read_fd, 1_048_576)
            if not block:
                break
            observed_raw.extend(block)
    finally:
        os.close(read_fd)
    final_recheck = os.lstat(target)
    if (
        created_identity != (opened.st_dev, opened.st_ino)
        or (opened.st_dev, opened.st_ino)
        != (final_stat.st_dev, final_stat.st_ino)
        or not stat.S_ISREG(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
        or opened.st_uid != os.geteuid()
        or opened.st_gid != os.getegid()
        or opened.st_size != len(raw)
        or not stat.S_ISREG(final_recheck.st_mode)
        or stat.S_ISLNK(final_recheck.st_mode)
        or stat.S_IMODE(final_recheck.st_mode) != 0o600
        or final_recheck.st_nlink != 1
        or final_recheck.st_uid != os.geteuid()
        or final_recheck.st_gid != os.getegid()
        or final_recheck.st_size != len(raw)
        or created_identity != (final_recheck.st_dev, final_recheck.st_ino)
        or bytes(observed_raw) != raw
        or hashlib.sha256(observed_raw).digest() != hashlib.sha256(raw).digest()
    ):
        raise DiagnosticInvariantError("closed output bytes differ")
    parse_canonical_json(bytes(observed_raw), label="PUBLISHED_OUTPUT", maximum_bytes=maximum_bytes)
    return FileWitness(
        label="DIAGNOSTIC_OUTPUT",
        path=target,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        mode=stat.S_IMODE(final_recheck.st_mode),
        uid=final_recheck.st_uid,
        gid=final_recheck.st_gid,
        nlink=final_recheck.st_nlink,
        device=final_recheck.st_dev,
        inode=final_recheck.st_ino,
        mtime_ns=final_recheck.st_mtime_ns,
    )


_RAW_MATERIAL_KEYS = frozenset(
    {
        "answer",
        "answers",
        "content",
        "canonical_batch",
        "canonical_episode_v2",
        "canonical_execution_receipt",
        "canonical_execution_request",
        "canonical_feedback",
        "canonical_payload",
        "canonical_reservation",
        "canonical_resolution",
        "execution_receipt_bytes",
        "execution_request_bytes",
        "feedback_text",
        "generated_token_ids",
        "input_observations",
        "mechanism",
        "mechanisms",
        "observations",
        "prompt",
        "prompt_text",
        "proposal",
        "proposal_generation_bytes",
        "proposals",
        "proposal_request_bytes",
        "raw_response",
        "recalled_evidence",
        "recalled_text",
        "request",
        "response",
        "response_text",
        "route",
        "routes",
        "seed",
        "seeds",
        "selected_trace",
        "selection_bytes",
        "solution",
        "solutions",
        "task",
        "task_payload",
        "task_response_generation_bytes",
        "token_ids",
    }
)


@dataclass(slots=True)
class RawMaterialWitness:
    """Ephemeral completeness witness for every parsed payload-bearing field."""

    values: set[str] = field(default_factory=set, repr=False)
    attempt_receipts: set[str] = field(default_factory=set, repr=False)
    cognitive_records: set[tuple[str, str]] = field(default_factory=set, repr=False)
    category_counts: dict[str, int] = field(default_factory=dict, repr=False)
    total_bytes: int = 0

    def _add_scalar(self, value: object) -> None:
        if type(value) is str:
            if not value:
                return
            encoded = value.encode("utf-8")
            if value not in self.values:
                self.values.add(value)
                self.total_bytes += len(encoded)
        elif type(value) is bytes:
            if not value:
                return
            try:
                self._add_scalar(value.decode("utf-8"))
            except UnicodeDecodeError:
                self._add_scalar(value.hex())
        elif type(value) is list:
            for item in value:
                self._add_scalar(item)
        elif type(value) is dict:
            for item in value.values():
                self._add_scalar(item)
        elif value is None or type(value) in {bool, int, float}:
            return
        else:
            raise DiagnosticInvariantError("raw material contains an unsupported type")
        if (
            len(self.values) > MAX_RAW_MATERIAL_VALUES
            or self.total_bytes > MAX_RAW_MATERIAL_BYTES
        ):
            raise DiagnosticInvariantError("raw-material witness exceeds its ceiling")

    def observe_document(
        self,
        value: Mapping[str, object],
        *,
        category: str = "DOCUMENT",
    ) -> None:
        """Collect only payload-bearing leaves, never ordinary refs or enums."""

        if type(value) is not dict or _SAFE_ENUM.fullmatch(category) is None:
            raise TypeError("raw-material document must be an exact object")
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

        def walk(item: object) -> None:
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise DiagnosticInvariantError("raw-material key is not text")
                    if key in _RAW_MATERIAL_KEYS:
                        self._add_scalar(child)
                    else:
                        walk(child)
            elif type(item) is list:
                for child in item:
                    walk(child)

        walk(value)

    def observe_attempt(
        self,
        receipt_ref: str,
        stage: Mapping[str, object],
        judgment: Mapping[str, object],
    ) -> None:
        _require_ref(receipt_ref, "raw witness attempt receipt")
        if receipt_ref in self.attempt_receipts:
            raise DiagnosticInvariantError("raw witness attempt is duplicated")
        self.attempt_receipts.add(receipt_ref)
        self.observe_document(stage, category="ATTEMPT_STAGE")
        self.observe_document(judgment, category="ATTEMPT_JUDGMENT")

    def observe_cognitive_record(
        self,
        lineage_ref: str,
        record_ref: str,
        record: Mapping[str, object],
    ) -> None:
        _require_ref(lineage_ref, "raw witness lineage")
        _require_ref(record_ref, "raw witness record")
        key = (lineage_ref, record_ref)
        if key in self.cognitive_records:
            raise DiagnosticInvariantError("raw witness cognitive record is duplicated")
        self.cognitive_records.add(key)
        self.observe_document(record, category="COGNITIVE_RECORD")

    def assert_complete(
        self,
        *,
        expected_attempts: int,
        expected_cognitive_records: int,
        required_categories: Mapping[str, int],
    ) -> None:
        if (
            len(self.attempt_receipts) != expected_attempts
            or len(self.cognitive_records) != expected_cognitive_records
            or not self.values
            or any(
                self.category_counts.get(category) != count
                for category, count in required_categories.items()
            )
            or any(count < 1 for count in self.category_counts.values())
        ):
            raise DiagnosticInvariantError("raw-material witness is incomplete")

    def forbidden_values(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))


@dataclass(frozen=True, slots=True)
class DiagnosticResourceBudget:
    """One-process wall/RSS witness checked before every irreversible boundary."""

    started: float = field(default_factory=time.monotonic)
    maximum_wall_seconds: int = field(default=MAX_WALL_SECONDS, init=False)
    maximum_rss_bytes: int = field(default=MAX_RSS_BYTES, init=False)

    def __post_init__(self) -> None:
        if type(self.started) is not float or not math.isfinite(self.started):
            raise ValueError("resource-budget start must be a finite monotonic value")

    @staticmethod
    def peak_rss_bytes() -> int:
        # Linux ru_maxrss is KiB.  The active workstation is fixed Ubuntu.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    def checkpoint(self) -> dict[str, int]:
        elapsed_ms = int((time.monotonic() - self.started) * 1000)
        peak = self.peak_rss_bytes()
        if (
            elapsed_ms < 0
            or elapsed_ms >= self.maximum_wall_seconds * 1000
            or peak > self.maximum_rss_bytes
        ):
            raise DiagnosticInvariantError("diagnostic resource ceiling was exceeded")
        return {"elapsed_milliseconds": elapsed_ms, "peak_rss_bytes": peak}


@contextlib.contextmanager
def _live_resource_guard(budget: DiagnosticResourceBudget) -> Iterator[None]:
    """Enforce a wall alarm and a stricter 1-GiB address-space ceiling."""

    if type(budget) is not DiagnosticResourceBudget:
        raise TypeError("live guard requires an exact resource budget")
    if signal.getsignal(signal.SIGALRM) not in {signal.SIG_DFL, signal.SIG_IGN}:
        raise DiagnosticInvariantError("SIGALRM already has an active owner")
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer != (0.0, 0.0):
        raise DiagnosticInvariantError("diagnostic refuses an active wall timer")
    old_limit = resource.getrlimit(resource.RLIMIT_AS)
    infinity = resource.RLIM_INFINITY
    old_soft, old_hard = old_limit
    new_soft = budget.maximum_rss_bytes
    if old_soft != infinity:
        new_soft = min(new_soft, old_soft)
    if old_hard != infinity:
        new_soft = min(new_soft, old_hard)

    def expired(_signum: int, _frame: object) -> None:
        raise DiagnosticInvariantError("diagnostic wall deadline expired")

    handler_changed = False
    limit_changed = False
    timer_changed = False
    elapsed = time.monotonic() - budget.started
    remaining = budget.maximum_wall_seconds - elapsed
    if remaining <= 0.0:
        raise DiagnosticInvariantError("diagnostic wall deadline already expired")
    try:
        signal.signal(signal.SIGALRM, expired)
        handler_changed = True
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, old_hard))
        limit_changed = True
        signal.setitimer(signal.ITIMER_REAL, remaining)
        timer_changed = True
        budget.checkpoint()
        yield
        budget.checkpoint()
    finally:
        try:
            if timer_changed:
                signal.setitimer(signal.ITIMER_REAL, *old_timer)
        finally:
            try:
                if limit_changed:
                    resource.setrlimit(resource.RLIMIT_AS, old_limit)
            finally:
                if handler_changed:
                    signal.signal(signal.SIGALRM, old_handler)


def _spec_map(specs: Sequence[FrozenInputSpec]) -> dict[str, FrozenInputSpec]:
    values = tuple(specs)
    if any(type(spec) is not FrozenInputSpec for spec in values):
        raise TypeError("composer inputs must be exact FrozenInputSpec values")
    if values != FROZEN_INPUTS:
        raise DiagnosticInvariantError("composer inputs differ from FROZEN_INPUTS")
    result = {spec.label: spec for spec in values}
    if (
        len(result) != len(values)
        or len({spec.path for spec in values}) != len(values)
    ):
        raise DiagnosticInvariantError("composer frozen-input inventory differs")
    return result


def _read_frozen_bytes(
    spec: FrozenInputSpec,
    *,
    maximum_bytes: int = MAX_JSON_INPUT_BYTES,
) -> tuple[bytes, FileWitness]:
    witness = capture_frozen_input(spec)
    if not 1 <= witness.size_bytes <= maximum_bytes:
        raise DiagnosticInvariantError("frozen input size exceeds its read ceiling")
    fd = _open_nofollow(spec.path)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (witness.device, witness.inode):
            raise DiagnosticInvariantError("frozen input changed before bounded read")
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            block = os.read(fd, min(1_048_576, maximum_bytes + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        len(raw) > maximum_bytes
        or hashlib.sha256(raw).hexdigest() != spec.sha256
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (witness.device, witness.inode, witness.size_bytes, witness.mtime_ns)
    ):
        raise DiagnosticInvariantError("bounded frozen-input read differs")
    verify_frozen_input_unchanged(spec, witness)
    return bytes(raw), witness


def _read_frozen_json(spec: FrozenInputSpec) -> tuple[dict[str, object], FileWitness]:
    if spec.kind != "JSON":
        raise TypeError("JSON reader requires one frozen JSON input")
    raw, witness = _read_frozen_bytes(spec)
    value = parse_canonical_json(raw, label=spec.label)
    return _mapping(value, spec.label), witness


def _raw_sha256(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DiagnosticInvariantError(f"{label} must be one raw SHA-256")
    return value


def _manifest_task_map(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
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
        "resource_ceilings",
        "schema",
        "seed_commitments",
        "thresholds",
        "write_scope",
    }
    if type(manifest) is not dict or set(manifest) != required or (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("evaluation_identity") != _EVALUATION_IDENTITY
        or manifest.get("host") != "angler-workstation"
        or manifest.get("arms") != list(EXPECTED_ARMS)
        or manifest.get("leaf_sha256")
        != "a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455"
        or manifest.get("qualification_result_sha256") is not None
    ):
        raise DiagnosticInvariantError("source manifest frozen identity differs")
    seeds = manifest.get("seed_commitments")
    if (
        type(seeds) is not list
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or any(_SHA256.fullmatch(item) is None for item in seeds)
    ):
        raise DiagnosticInvariantError("manifest seed commitments differ")
    commitments = _mapping(
        manifest.get("evaluator_commitments"), "evaluator commitments"
    )
    commitment_fields = {
        "identity",
        "purpose",
        "qualification_seed",
        "random_feedback",
        "replicate_commitments",
        "schema",
        "tasks",
    }
    if set(commitments) != commitment_fields or (
        commitments.get("identity") != "angler.high-level-multidomain.v1-evaluation"
        or commitments.get("purpose") != "evaluation"
        or commitments.get("qualification_seed") is not None
        or commitments.get("replicate_commitments") != seeds
        or commitments.get("schema") != "angler.high-level-multidomain.v1"
    ):
        raise DiagnosticInvariantError("evaluator commitment identity differs")
    expected_commitment = "sha256:" + hashlib.sha256(
        canonical_json_bytes(
            {
                "kind": "evaluator-commitments",
                "payload": commitments,
                "schema": "angler.high-level-multidomain.v1",
            }
        )
    ).hexdigest()
    if manifest.get("evaluator_commitment_digest") != expected_commitment:
        raise DiagnosticInvariantError("evaluator commitment digest differs")
    random_feedback = commitments.get("random_feedback")
    if type(random_feedback) is not list or len(random_feedback) != 2:
        raise DiagnosticInvariantError("random-feedback commitment count differs")
    for index, row in enumerate(random_feedback):
        if type(row) is not dict or set(row) != {
            "replicate_commitment",
            "schedule_commitment",
        } or row.get("replicate_commitment") != seeds[index]:
            raise DiagnosticInvariantError("random-feedback commitment differs")
        _require_ref(row.get("schedule_commitment"), "random-feedback schedule")

    tasks = commitments.get("tasks")
    if type(tasks) is not list or len(tasks) != 84:
        raise DiagnosticInvariantError("manifest task count differs")
    task_fields = {
        "family",
        "ordinal",
        "phase",
        "private_commitment",
        "public_commitment",
        "replicate_commitment",
        "task_id",
    }
    families = (
        "symbolic-demonstration-transfer",
        "glyph-machine",
        "causal-operator",
    )
    counts = {
        "symbolic-demonstration-transfer": {"adaptation": 4, "development": 2, "final": 4},
        "glyph-machine": {"adaptation": 2, "development": 2, "final": 4},
        "causal-operator": {"adaptation": 6, "development": 6, "final": 12},
    }
    expected_order: list[tuple[str, str, str, int]] = []
    for replicate in seeds:
        for phase in ("adaptation", "development", "final"):
            maximum = max(counts[family][phase] for family in families)
            for ordinal in range(maximum):
                expected_order.extend(
                    (replicate, family, phase, ordinal)
                    for family in families
                    if ordinal < counts[family][phase]
                )
    observed_order: list[tuple[str, str, str, int]] = []
    result: dict[str, dict[str, object]] = {}
    public_refs: set[str] = set()
    private_refs: set[str] = set()
    for task in tasks:
        if type(task) is not dict or set(task) != task_fields:
            raise DiagnosticInvariantError("manifest task fields differ")
        task_id = _require_ref(task.get("task_id"), "manifest task")
        public_ref = _require_ref(task.get("public_commitment"), "public task")
        private_ref = _require_ref(task.get("private_commitment"), "private task")
        replicate = _require_ref(task.get("replicate_commitment"), "task replicate")
        family = task.get("family")
        phase = task.get("phase")
        ordinal = task.get("ordinal")
        if (
            family not in families
            or phase not in {"adaptation", "development", "final"}
            or type(ordinal) is not int
            or ordinal < 0
            or replicate not in seeds
            or task_id in result
            or public_ref in public_refs
            or private_ref in private_refs
        ):
            raise DiagnosticInvariantError("manifest task identity differs")
        result[task_id] = dict(task)
        public_refs.add(public_ref)
        private_refs.add(private_ref)
        observed_order.append((replicate, family, phase, ordinal))
    if observed_order != expected_order:
        raise DiagnosticInvariantError("manifest task order/coverage differs")
    return result


def _expected_schedule(
    tasks: Mapping[str, Mapping[str, object]],
) -> set[tuple[str, str]]:
    result = set(_expected_schedule_order(tasks))
    if len(result) != MAX_ATTEMPTS:
        raise DiagnosticInvariantError("manifest schedule does not contain 468 attempts")
    return result


def _expected_schedule_order(
    tasks: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for phase in ("adaptation", "development", "final"):
        for task_id, task in tasks.items():
            if task.get("phase") != phase:
                continue
            arms = (
                ("FULL", "RANDOM_FEEDBACK")
                if phase == "adaptation"
                else EXPECTED_ARMS
            )
            result.extend((task_id, arm) for arm in arms)
    if len(result) != MAX_ATTEMPTS or len(set(result)) != MAX_ATTEMPTS:
        raise DiagnosticInvariantError("manifest schedule order differs")
    return tuple(result)


def _result_attempt_bindings(
    result: Mapping[str, object],
    *,
    manifest_sha256: str,
    tasks: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    set[str],
    dict[tuple[str, str, str], tuple[int, int]],
]:
    required = {
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
    if type(result) is not dict or set(result) != required or (
        result.get("schema") != _EVALUATION_RESULT_SCHEMA
        or result.get("identity") != _EVALUATION_IDENTITY
        or result.get("purpose") != "evaluation"
        or result.get("classification") != "NOT_SUPPORTED"
        or result.get("scientific_claim") != "BOUNDED_SYNTHETIC_EXPERIMENT_ONLY"
        or result.get("arm_tasks") != MAX_ATTEMPTS
        or result.get("manifest_sha256") != manifest_sha256
    ):
        raise DiagnosticInvariantError("terminal result identity differs")
    for key, domain in (
        ("adaptation_lineage", "adaptation-lineages"),
        ("full_adaptation_lineage", "full-adaptation-lineages"),
        ("foundation_integrity", "foundation-integrity"),
        ("run_integrity", "run-integrity"),
        ("classifier", "integer-classifier"),
        ("evaluation_metrics", "final-metrics"),
    ):
        if result.get(f"{key}_ref") != content_ref(domain, result.get(key)):
            raise DiagnosticInvariantError(f"terminal result {key} reference differs")
    admission = _mapping(result.get("admission_claim"), "admission claim")
    if result.get("admission_claim_ref") != content_ref(
        "evaluation-admission-claim", admission
    ):
        raise DiagnosticInvariantError("terminal admission claim reference differs")
    run_integrity = _mapping(result.get("run_integrity"), "run integrity")
    if (
        set(run_integrity)
        != {
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
        or run_integrity.get("schema") != _RUN_INTEGRITY_SCHEMA
        or run_integrity.get("purpose") != "evaluation"
        or result.get("run_integrity_ref")
        != content_ref("run-integrity", run_integrity)
        or run_integrity.get("foundation_guard_checks") != MAX_ATTEMPTS
    ):
        raise DiagnosticInvariantError("run-integrity envelope differs")
    if (
        result.get("accounting") != run_integrity.get("resource_accounting")
        or result.get("adaptation_lineage")
        != run_integrity.get("adaptation_lineage")
        or result.get("adaptation_lineage_ref")
        != run_integrity.get("adaptation_lineage_ref")
        or result.get("full_adaptation_lineage")
        != run_integrity.get("full_adaptation_lineage")
        or result.get("full_adaptation_lineage_ref")
        != run_integrity.get("full_adaptation_lineage_ref")
        or result.get("foundation_integrity") != run_integrity.get("foundation")
        or result.get("foundation_integrity_ref")
        != run_integrity.get("foundation_ref")
    ):
        raise DiagnosticInvariantError("terminal duplicated run evidence differs")
    raw_attempts = run_integrity.get("attempts")
    if type(raw_attempts) is not list or len(raw_attempts) != MAX_ATTEMPTS:
        raise DiagnosticInvariantError("result attempt bindings differ")
    bindings: dict[tuple[str, str], dict[str, object]] = {}
    observed_order: list[tuple[str, str]] = []
    for row in raw_attempts:
        if type(row) is not dict or set(row) != {
            "arm",
            "attempt_receipt_ref",
            "parser_disposition",
            "probe_integrity",
            "probe_integrity_ref",
            "task_id",
        }:
            raise DiagnosticInvariantError("result attempt binding fields differ")
        task_id = _require_ref(row.get("task_id"), "result task")
        arm = row.get("arm")
        receipt = _require_ref(row.get("attempt_receipt_ref"), "result receipt")
        probe = _mapping(row.get("probe_integrity"), "result probe")
        if (
            arm not in EXPECTED_ARMS
            or row.get("probe_integrity_ref") != content_ref("probe-integrity", probe)
            or (task_id, arm) in bindings
        ):
            raise DiagnosticInvariantError("result attempt binding differs")
        task = tasks.get(task_id)
        if (
            task is None
            or set(probe)
            != {
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
            }
            or probe.get("schema") != _PROBE_INTEGRITY_SCHEMA
            or (
                probe.get("purpose"),
                probe.get("phase"),
                probe.get("task_id"),
                probe.get("arm"),
                probe.get("replicate_commitment"),
            )
            != (
                "evaluation",
                task.get("phase"),
                task_id,
                arm,
                task.get("replicate_commitment"),
            )
        ):
            raise DiagnosticInvariantError("result probe schedule binding differs")
        bindings[(task_id, arm)] = row
        observed_order.append((task_id, str(arm)))
    if tuple(observed_order) != _expected_schedule_order(tasks):
        raise DiagnosticInvariantError("result attempt order differs from manifest")
    receipt_refs = result.get("attempt_receipt_refs")
    if (
        type(receipt_refs) is not list
        or len(receipt_refs) != MAX_ATTEMPTS
        or len(set(receipt_refs)) != MAX_ATTEMPTS
        or receipt_refs
        != [row["attempt_receipt_ref"] for row in raw_attempts]
    ):
        raise DiagnosticInvariantError("result receipt inventory differs")
    metrics = _mapping(result.get("evaluation_metrics"), "evaluation metrics")
    if (
        set(metrics) != {"aggregates", "identity", "purpose", "schema"}
        or metrics.get("schema") != "angler.high-level-multidomain.v1"
        or metrics.get("identity")
        != "angler.high-level-multidomain.v1-evaluation"
        or metrics.get("purpose") != "evaluation"
    ):
        raise DiagnosticInvariantError("evaluation metric envelope differs")
    aggregates = metrics.get("aggregates")
    if type(aggregates) is not list:
        raise DiagnosticInvariantError("evaluation metric rows differ")
    metric_map: dict[tuple[str, str, str], tuple[int, int]] = {}
    for row in aggregates:
        if type(row) is not dict or set(row) != {
            "arm",
            "attempts",
            "binary_success_total",
            "family",
            "pairwise_agreement_total",
            "phase",
        }:
            raise DiagnosticInvariantError("evaluation metric fields differ")
        key = (str(row["phase"]), str(row["arm"]), str(row["family"]))
        attempts = row["attempts"]
        successes = row["binary_success_total"]
        if (
            key in metric_map
            or type(attempts) is not int
            or type(successes) is not int
            or not 0 <= successes <= attempts
        ):
            raise DiagnosticInvariantError("evaluation metric count differs")
        metric_map[key] = (attempts, successes)
    return bindings, set(receipt_refs), metric_map


@dataclass(frozen=True, slots=True)
class _LedgerRow:
    task_id: str
    arm: str
    phase: str
    family: str
    replicate_ref: str
    receipt_ref: str
    stage_ref: str
    judgment_ref: str
    stage_bytes: bytes = field(repr=False)
    judgment_bytes: bytes = field(repr=False)
    stage: dict[str, object] = field(repr=False)
    judgment: dict[str, object] = field(repr=False)
    success: bool


@dataclass(frozen=True, slots=True)
class _LedgerDataset:
    rows: dict[tuple[str, str], _LedgerRow]
    receipt_refs: frozenset[str]
    run_intent_ref: str
    metric_counts: dict[tuple[str, str, str], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class _RemovalDataset:
    records: dict[str, dict[str, object]]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LineageBinding:
    replicate_ref: str
    arm: str
    scope_ref: str
    spec: FrozenInputSpec
    hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class _FactoryDataset:
    lineages: dict[tuple[str, str], _LineageBinding]
    disposition_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RetainedRecord:
    lineage_scope_ref: str
    acquisition_ref: str
    record_ref: str
    record_bytes_sha256: str
    source_contract: str
    source_ref: str
    content: str = field(repr=False)
    ancestry_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProspectiveJoin:
    legacy_reservation_ref: str
    batch_ref: str
    resolution_ref: str | None
    legacy_episode_ref: str | None
    batch_acquisition_ref: str
    resolution_acquisition_ref: str | None


@dataclass(frozen=True, slots=True)
class _CognitiveLineage:
    binding: _LineageBinding
    records: dict[str, _RetainedRecord]
    prospective_by_episode: dict[str, _ProspectiveJoin]
    prospective_by_reservation: dict[str, _ProspectiveJoin]
    acquisition_count: int


def _table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[str, ...]:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", table) is None:
        raise DiagnosticInvariantError("SQLite table name is unsafe")
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return tuple(str(row[1]) for row in rows)


_STAGE_FIELDS = {
    "arm",
    "attempt_receipt",
    "attempt_receipt_ref",
    "execution_receipt",
    "execution_request",
    "feedback_source",
    "foundation_tensor_digest",
    "frozen_recall_ref",
    "judge_arm",
    "learner_parent_digest",
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
    "task_id",
    "task_response_generation",
}
_FINAL_FIELDS = {
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


def _validate_attempt_receipt(stage: Mapping[str, object]) -> str:
    parser = stage.get("parser_disposition")
    proposal_generation = _mapping(
        stage.get("proposal_generation"), "proposal generation"
    )
    if parser == "MALFORMED":
        source_kind = "PROPOSAL_GENERATION"
        source_ref = _require_ref(
            proposal_generation.get("generation_ref"), "proposal generation"
        )
    elif parser == "ADMITTED":
        source_kind = "EXECUTION_RECEIPT"
        receipt = _mapping(stage.get("execution_receipt"), "execution receipt")
        source_ref = _require_ref(
            receipt.get("execution_receipt_ref"), "execution receipt"
        )
    else:
        raise DiagnosticInvariantError("attempt parser disposition differs")
    expected = {
        "arm": stage.get("arm"),
        "nonauthorization": _ATTEMPT_RECEIPT_NONAUTHORIZATION,
        "parser_disposition": parser,
        "phase": stage.get("phase"),
        "purpose": stage.get("purpose"),
        "schema": _ATTEMPT_RECEIPT_SCHEMA,
        "source_kind": source_kind,
        "source_receipt_ref": source_ref,
        "task_id": stage.get("task_id"),
    }
    if stage.get("attempt_receipt") != expected:
        raise DiagnosticInvariantError("attempt receipt payload differs")
    expected_ref = content_ref("arm-attempt-receipt", expected)
    if stage.get("attempt_receipt_ref") != expected_ref:
        raise DiagnosticInvariantError("attempt receipt reference differs")
    return expected_ref


def _manifest_replicate_labels(
    tasks: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    ordered: list[str] = []
    for task in tasks.values():
        replicate = _require_ref(
            task.get("replicate_commitment"), "manifest replicate"
        )
        if replicate not in ordered:
            ordered.append(replicate)
    if len(ordered) != 2:
        raise DiagnosticInvariantError("manifest replicate inventory differs")
    return {
        replicate: f"replicate-{index:02d}"
        for index, replicate in enumerate(ordered, start=1)
    }


def _validate_public_task_binding(
    stage: Mapping[str, object],
    task: Mapping[str, object],
    *,
    replicate_labels: Mapping[str, str],
) -> dict[str, object]:
    request = _mapping(stage.get("proposal_request"), "proposal request")
    rendered = request.get("task")
    if type(rendered) is not str:
        raise DiagnosticInvariantError("rendered public task is absent")
    payload = _mapping(
        parse_canonical_json(
            rendered,
            label="PUBLIC_TASK",
            maximum_bytes=MAX_EVIDENCE_RECORD_BYTES,
        ),
        "public task",
    )
    replicate_ref = _require_ref(
        task.get("replicate_commitment"), "public task replicate"
    )
    replicate = replicate_labels.get(replicate_ref)
    expected = _evaluator_record_digest(
        "public-task",
        {
            "family": task.get("family"),
            "ordinal": task.get("ordinal"),
            "payload_json": rendered,
            "phase": task.get("phase"),
            "replicate": replicate,
            "replicate_commitment": replicate_ref,
        },
    )
    if (
        replicate is None
        or payload.get("family") != task.get("family")
        or stage.get("public_task_ref") != task.get("public_commitment")
        or task.get("public_commitment") != expected
    ):
        raise DiagnosticInvariantError("public task/manifest commitment differs")
    return payload


def _stream_ledger(
    spec: FrozenInputSpec,
    *,
    tasks: Mapping[str, Mapping[str, object]],
    result_bindings: Mapping[tuple[str, str], Mapping[str, object]],
    result_receipts: set[str],
    result_metrics: Mapping[tuple[str, str, str], tuple[int, int]],
    raw_witness: RawMaterialWitness,
    budget: DiagnosticResourceBudget,
) -> _LedgerDataset:
    schedule = _expected_schedule(tasks)
    replicate_labels = _manifest_replicate_labels(tasks)
    expected_columns = (
        "task_id",
        "arm",
        "attempt_receipt_ref",
        "stage_bytes",
        "stage_ref",
        "judgment_bytes",
        "judgment_ref",
    )
    rows: dict[tuple[str, str], _LedgerRow] = {}
    receipts: set[str] = set()
    stage_refs: set[str] = set()
    judgment_refs: set[str] = set()
    metric_accumulator: dict[tuple[str, str, str], list[int]] = {}
    with immutable_sqlite(spec) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise DiagnosticInvariantError("evaluation ledger integrity check failed")
        if _table_columns(connection, "attempts") != expected_columns:
            raise DiagnosticInvariantError("evaluation ledger columns differ")
        identity_columns = _table_columns(connection, "ledger_identity")
        if identity_columns != ("singleton", "schema", "run_intent_ref"):
            raise DiagnosticInvariantError("evaluation ledger identity columns differ")
        identity = connection.execute(
            "SELECT schema, run_intent_ref FROM ledger_identity WHERE singleton=1"
        ).fetchone()
        if identity is None or identity[0] != _EVIDENCE_SCHEMA:
            raise DiagnosticInvariantError("evaluation ledger identity differs")
        run_intent_ref = _require_ref(identity[1], "ledger run intent")
        if connection.execute("PRAGMA user_version").fetchone() != (1,):
            raise DiagnosticInvariantError("evaluation ledger schema version differs")
        cursor = connection.execute(
            "SELECT task_id, arm, attempt_receipt_ref, stage_bytes, stage_ref, "
            "judgment_bytes, judgment_ref FROM attempts ORDER BY task_id, arm"
        )
        while True:
            batch = cursor.fetchmany(SQL_FETCH_BATCH)
            if not batch:
                break
            for raw_row in batch:
                if len(rows) >= MAX_ATTEMPTS:
                    raise DiagnosticInvariantError("evaluation ledger exceeds 468 rows")
                (
                    indexed_task,
                    indexed_arm,
                    indexed_receipt,
                    raw_stage,
                    indexed_stage_ref,
                    raw_judgment,
                    indexed_judgment_ref,
                ) = raw_row
                if type(raw_stage) is not bytes or type(raw_judgment) is not bytes:
                    raise DiagnosticInvariantError("evaluation ledger has an unfinalized row")
                stage = _mapping(
                    parse_canonical_json(
                        raw_stage,
                        label="LEDGER_STAGE",
                        maximum_bytes=MAX_EVIDENCE_RECORD_BYTES,
                    ),
                    "ledger stage",
                )
                judgment = _mapping(
                    parse_canonical_json(
                        raw_judgment,
                        label="LEDGER_JUDGMENT",
                        maximum_bytes=MAX_EVIDENCE_RECORD_BYTES,
                    ),
                    "ledger judgment",
                )
                task_id = _require_ref(stage.get("task_id"), "ledger task")
                arm = stage.get("arm")
                receipt = _require_ref(stage.get("attempt_receipt_ref"), "ledger receipt")
                if set(stage) != _STAGE_FIELDS or set(judgment) != _FINAL_FIELDS:
                    raise DiagnosticInvariantError("ledger stage/final fields differ")
                if _validate_attempt_receipt(stage) != receipt:
                    raise DiagnosticInvariantError("ledger attempt receipt differs")
                stage_ref = _require_ref(indexed_stage_ref, "ledger stage ref")
                judgment_ref = _require_ref(indexed_judgment_ref, "ledger judgment ref")
                if (
                    (indexed_task, indexed_arm, indexed_receipt)
                    != (task_id, arm, receipt)
                    or (task_id, arm) not in schedule
                    or stage_ref != content_ref("attempt-stage", stage)
                    or judgment_ref != content_ref("attempt-judgment", judgment)
                    or judgment.get("task_id") != task_id
                    or judgment.get("arm") != arm
                    or judgment.get("attempt_receipt_ref") != receipt
                    or stage.get("schema") != _ATTEMPT_STAGE_SCHEMA
                    or judgment.get("schema") != _ATTEMPT_FINAL_SCHEMA
                    or stage.get("purpose") != "evaluation"
                    or stage.get("judge_arm") != arm
                    or receipt in receipts
                    or stage_ref in stage_refs
                    or judgment_ref in judgment_refs
                    or (task_id, arm) in rows
                ):
                    raise DiagnosticInvariantError("evaluation ledger binding differs")
                task = tasks[task_id]
                phase = stage.get("phase")
                if phase != task.get("phase"):
                    raise DiagnosticInvariantError("ledger phase differs from manifest")
                _validate_public_task_binding(
                    stage,
                    task,
                    replicate_labels=replicate_labels,
                )
                objective = _mapping(
                    judgment.get("objective_judgment"), "ledger objective judgment"
                )
                score = objective.get("score")
                expected_response_commitment = _evaluator_record_digest(
                    "response",
                    {
                        "arm": arm,
                        "attempt_receipt_ref": receipt,
                        "raw_response": stage.get("raw_response"),
                        "task_id": task_id,
                    },
                )
                if (
                    set(objective)
                    != {
                        "arm",
                        "attempt_receipt_ref",
                        "disposition",
                        "raw_response",
                        "response_commitment",
                        "score",
                        "task_id",
                    }
                    or objective.get("task_id") != task_id
                    or objective.get("arm") != arm
                    or objective.get("attempt_receipt_ref") != receipt
                    or objective.get("raw_response") != stage.get("raw_response")
                    or objective.get("response_commitment")
                    != expected_response_commitment
                    or type(score) is not float
                    or score not in {0.0, 1.0}
                    or objective.get("disposition")
                    != ("SUCCESS" if score == 1.0 else "UNSUCCESSFUL")
                ):
                    raise DiagnosticInvariantError("ledger objective binding differs")
                raw_witness.observe_attempt(receipt, stage, judgment)
                success = score == 1.0
                family = str(task["family"])
                replicate = _require_ref(task.get("replicate_commitment"), "task replicate")
                row = _LedgerRow(
                    task_id=task_id,
                    arm=str(arm),
                    phase=str(phase),
                    family=family,
                    replicate_ref=replicate,
                    receipt_ref=receipt,
                    stage_ref=stage_ref,
                    judgment_ref=judgment_ref,
                    stage_bytes=raw_stage,
                    judgment_bytes=raw_judgment,
                    stage=stage,
                    judgment=judgment,
                    success=success,
                )
                rows[(task_id, str(arm))] = row
                receipts.add(receipt)
                stage_refs.add(stage_ref)
                judgment_refs.add(judgment_ref)
                metric_key = (str(phase), str(arm), family)
                aggregate = metric_accumulator.setdefault(metric_key, [0, 0])
                aggregate[0] += 1
                aggregate[1] += int(success)
            budget.checkpoint()
    if (
        set(rows) != schedule
        or receipts != result_receipts
        or len(stage_refs) != MAX_ATTEMPTS
        or len(judgment_refs) != MAX_ATTEMPTS
    ):
        raise DiagnosticInvariantError("ledger/result attempt inventory differs")
    for key, result_row in result_bindings.items():
        ledger = rows.get(key)
        if ledger is None:
            raise DiagnosticInvariantError("result attempt is absent from ledger")
        probe = _mapping(result_row.get("probe_integrity"), "result probe")
        if (
            result_row.get("attempt_receipt_ref") != ledger.receipt_ref
            or result_row.get("parser_disposition")
            != ledger.stage.get("parser_disposition")
            or (probe.get("task_id"), probe.get("arm"), probe.get("phase"), probe.get("purpose"))
            != (ledger.task_id, ledger.arm, ledger.phase, "evaluation")
            or ledger.judgment.get("probe_integrity") != probe
            or ledger.judgment.get("probe_integrity_ref")
            != result_row.get("probe_integrity_ref")
        ):
            raise DiagnosticInvariantError("result/ledger attempt linkage differs")
    metric_counts = {key: tuple(value) for key, value in metric_accumulator.items()}
    if metric_counts != dict(result_metrics):
        raise DiagnosticInvariantError("ledger/result aggregate metrics differ")
    return _LedgerDataset(
        rows=rows,
        receipt_refs=frozenset(receipts),
        run_intent_ref=run_intent_ref,
        metric_counts=metric_counts,  # type: ignore[arg-type]
    )


_REMOVAL_EVIDENCE_FIELDS = frozenset(
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


def _decode_hex_bytes(value: object, label: str) -> bytes:
    if type(value) is not str or re.fullmatch(r"(?:[0-9a-f]{2})*", value) is None:
        raise DiagnosticInvariantError(f"{label} is not canonical lowercase hex")
    return bytes.fromhex(value)


def _validate_removal_arm(
    value: object,
    *,
    arm: str,
    ledger: _LedgerRow,
) -> dict[str, object]:
    row = _mapping(value, f"{arm} removal evidence")
    if set(row) != _REMOVAL_EVIDENCE_FIELDS or row.get("task_id") != ledger.task_id:
        raise DiagnosticInvariantError("removal arm fields or task binding differ")
    for name in (
        "backend_raw_hit_count",
        "backend_rejected_hit_count",
        "backend_search_calls",
    ):
        if type(row.get(name)) is not int or int(row[name]) < 0:
            raise DiagnosticInvariantError("removal arm counter differs")
    for name in (
        "evidence_final_ref",
        "evidence_stage_ref",
        "frozen_recall_ref",
        "proposal_generation_ref",
        "proposal_prompt_ref",
        "probe_integrity_ref",
        "public_task_ref",
    ):
        _require_ref(row.get(name), f"removal {name}")
    for name in (
        "execution_prompt_ref",
        "execution_receipt_ref",
        "execution_request_ref",
        "runtime_quiescence_ref",
        "selection_ref",
        "task_response_generation_ref",
    ):
        _optional_ref(row.get(name), f"removal {name}")
    refs = row.get("recalled_record_refs")
    hashes = row.get("recalled_record_bytes_sha256")
    proposals = row.get("proposals")
    if (
        type(refs) is not list
        or type(hashes) is not list
        or len(refs) != len(hashes)
        or len(set(refs)) != len(refs)
        or any(
            type(item) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None
            for item in refs
        )
        or any(
            type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in hashes
        )
        or type(proposals) is not list
        or any(type(item) is not str for item in proposals)
        or type(row.get("raw_response")) is not str
        or row.get("selected_trace") is not None
        and type(row.get("selected_trace")) is not str
        or type(row.get("score")) is not float
        or row.get("score") not in {0.0, 1.0}
    ):
        raise DiagnosticInvariantError("removal arm public evidence differs")
    decoded = {
        name: _decode_hex_bytes(row.get(name), f"removal {name}")
        for name in (
            "execution_receipt_bytes",
            "execution_request_bytes",
            "proposal_generation_bytes",
            "proposal_request_bytes",
            "selection_bytes",
            "task_response_generation_bytes",
        )
    }
    expected_bytes = {
        "execution_receipt_bytes": (
            b""
            if ledger.stage.get("execution_receipt") is None
            else canonical_json_bytes(
                {
                    key: item
                    for key, item in _mapping(
                        ledger.stage.get("execution_receipt"),
                        "removal execution receipt",
                    ).items()
                    if key != "execution_receipt_ref"
                }
            )
        ),
        "execution_request_bytes": (
            b""
            if ledger.stage.get("execution_request") is None
            else canonical_json_bytes(
                {
                    key: item
                    for key, item in _mapping(
                        ledger.stage.get("execution_request"),
                        "removal execution request",
                    ).items()
                    if key != "execution_request_ref"
                }
            )
        ),
        "proposal_generation_bytes": canonical_json_bytes(
            ledger.stage.get("proposal_generation")
        ),
        "proposal_request_bytes": canonical_json_bytes(
            ledger.stage.get("proposal_request")
        ),
        "selection_bytes": (
            b""
            if ledger.stage.get("selection") is None
            else canonical_json_bytes(ledger.stage.get("selection"))
        ),
        "task_response_generation_bytes": (
            b""
            if ledger.stage.get("task_response_generation") is None
            else canonical_json_bytes(ledger.stage.get("task_response_generation"))
        ),
    }
    objective = _mapping(
        ledger.judgment.get("objective_judgment"), "removal objective judgment"
    )
    selection = ledger.stage.get("selection")
    selected_trace = (
        None if selection is None else _mapping(selection, "removal selection").get("selected_trace")
    )
    selection_ref = None
    if selection is not None:
        selected = _mapping(selection, "removal selection")
        selection_ref = selected.get(
            "selection_ref", selected.get("reservation_ref")
        )
    proposal_generation = _mapping(
        ledger.stage.get("proposal_generation"), "removal proposal generation"
    )
    execution_request = ledger.stage.get("execution_request")
    execution_receipt = ledger.stage.get("execution_receipt")
    task_generation = ledger.stage.get("task_response_generation")
    task_generation_map = (
        None
        if task_generation is None
        else _mapping(task_generation, "removal task generation")
    )
    for bytes_name, ref_name in (
        ("execution_request_bytes", "execution_request_ref"),
        ("execution_receipt_bytes", "execution_receipt_ref"),
    ):
        reference = row.get(ref_name)
        if reference is not None and (
            not decoded[bytes_name]
            or "sha256:" + hashlib.sha256(decoded[bytes_name]).hexdigest()
            != reference
        ):
            raise DiagnosticInvariantError("removal execution reference differs")
    quiescence = ledger.judgment.get("runtime_quiescence")
    if row.get("runtime_quiescence_ref") != (
        None if quiescence is None else content_ref("runtime-quiescence", quiescence)
    ):
        raise DiagnosticInvariantError("removal quiescence reference differs")
    if (
        decoded != expected_bytes
        or row.get("evidence_stage_ref") != ledger.stage_ref
        or row.get("evidence_final_ref") != ledger.judgment_ref
        or row.get("probe_integrity_ref")
        != ledger.judgment.get("probe_integrity_ref")
        or row.get("frozen_recall_ref") != ledger.stage.get("frozen_recall_ref")
        or row.get("proposals") != ledger.stage.get("proposals")
        or row.get("raw_response") != ledger.stage.get("raw_response")
        or row.get("selected_trace") != selected_trace
        or row.get("selection_ref") != selection_ref
        or row.get("score") != objective.get("score")
        or row.get("public_task_ref") != ledger.stage.get("public_task_ref")
        or row.get("proposal_generation_ref")
        != proposal_generation.get("generation_ref")
        or row.get("proposal_prompt_ref") != proposal_generation.get("prompt_ref")
        or row.get("execution_request_ref")
        != (
            None
            if execution_request is None
            else _mapping(execution_request, "removal execution request").get(
                "execution_request_ref"
            )
        )
        or row.get("execution_receipt_ref")
        != (
            None
            if execution_receipt is None
            else _mapping(execution_receipt, "removal execution receipt").get(
                "execution_receipt_ref"
            )
        )
        or row.get("task_response_generation_ref")
        != (None if task_generation_map is None else task_generation_map.get("generation_ref"))
        or row.get("execution_prompt_ref")
        != (None if task_generation_map is None else task_generation_map.get("prompt_ref"))
        or row.get("runtime_quiescence_ref")
        != ledger.judgment.get("runtime_quiescence_ref")
    ):
        raise DiagnosticInvariantError("removal arm does not rejoin its ledger row")
    return row


def _validate_removal_dataset(
    result: Mapping[str, object],
    *,
    tasks: Mapping[str, Mapping[str, object]],
    ledger: _LedgerDataset,
) -> _RemovalDataset:
    run_integrity = _mapping(result.get("run_integrity"), "run integrity")
    values = run_integrity.get("removal_fairness")
    refs = run_integrity.get("removal_fairness_refs")
    expected_order = tuple(
        task_id
        for phase in ("development", "final")
        for task_id, task in tasks.items()
        if task.get("phase") == phase
    )
    if (
        type(values) is not list
        or len(values) != 60
        or type(refs) is not list
        or len(refs) != 60
        or result.get("removal_fairness_refs") != refs
    ):
        raise DiagnosticInvariantError("removal-fairness inventory differs")
    records: dict[str, dict[str, object]] = {}
    observed_refs: list[str] = []
    observed_order: list[str] = []
    removal_arms = (
        "FULL",
        "FROZEN_ORIGIN",
        "PROSPECTIVE_REMOVAL",
        "BACKEND_REMOVAL",
        "RETRIEVAL_ONLY",
    )
    for value in values:
        record = _mapping(value, "removal-fairness record")
        if (
            set(record)
            != {
                "arms",
                "evidence_ref",
                "phase",
                "purpose",
                "require_score",
                "schema",
                "task_id",
            }
            or record.get("schema") != _REMOVAL_FAIRNESS_SCHEMA
            or record.get("purpose") != "evaluation"
            or record.get("require_score") is not True
            or record.get("phase") not in {"development", "final"}
        ):
            raise DiagnosticInvariantError("removal-fairness identity differs")
        task_id = _require_ref(record.get("task_id"), "removal task")
        if task_id in records or tasks.get(task_id, {}).get("phase") != record.get("phase"):
            raise DiagnosticInvariantError("removal task identity is reused or mismatched")
        arms = _mapping(record.get("arms"), "removal arms")
        if set(arms) != set(removal_arms):
            raise DiagnosticInvariantError("removal arm inventory differs")
        validated = {
            arm: _validate_removal_arm(
                arms.get(arm),
                arm=arm,
                ledger=ledger.rows[(task_id, arm)],
            )
            for arm in removal_arms
        }
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
        common = tuple(validated["FULL"].get(name) for name in common_fields)
        if any(
            tuple(validated[arm].get(name) for name in common_fields) != common
            for arm in removal_arms
        ):
            raise DiagnosticInvariantError("removal arms differ before intervention")
        expected_search = {
            "FULL": 1,
            "FROZEN_ORIGIN": 1,
            "PROSPECTIVE_REMOVAL": 1,
            "BACKEND_REMOVAL": 0,
            "RETRIEVAL_ONLY": 0,
        }
        if any(
            validated[arm].get("backend_search_calls") != expected
            for arm, expected in expected_search.items()
        ):
            raise DiagnosticInvariantError("removal backend-search accounting differs")
        for name in (
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
            "score",
        ):
            if validated["BACKEND_REMOVAL"].get(name) != validated["FULL"].get(name):
                raise DiagnosticInvariantError("BACKEND_REMOVAL differs from FULL")
        expected_ref = content_ref(
            "removal-fairness", {"arms": arms, "require_score": True}
        )
        if record.get("evidence_ref") != expected_ref:
            raise DiagnosticInvariantError("removal-fairness reference differs")
        records[task_id] = record
        observed_refs.append(expected_ref)
        observed_order.append(task_id)
    if tuple(observed_order) != expected_order or observed_refs != refs:
        raise DiagnosticInvariantError("removal-fairness order differs from manifest")
    return _RemovalDataset(records=records, evidence_refs=tuple(observed_refs))


def _validate_factory_audit(
    audit_envelope: Mapping[str, object],
    *,
    ledger_spec: FrozenInputSpec,
    ledger_witness: FileWitness,
    ledger: _LedgerDataset,
    tasks: Mapping[str, Mapping[str, object]],
    specs: Mapping[str, FrozenInputSpec],
) -> _FactoryDataset:
    if set(audit_envelope) != {
        "factory_audit",
        "factory_audit_ref",
        "ledger_bytes",
        "ledger_sha256",
        "run_intent_ref",
    }:
        raise DiagnosticInvariantError("arm-factory audit envelope differs")
    factory = _mapping(audit_envelope.get("factory_audit"), "factory audit")
    if (
        audit_envelope.get("factory_audit_ref")
        != content_ref("live-arm-factory-audit", factory)
        or audit_envelope.get("ledger_sha256") != ledger_spec.sha256
        or audit_envelope.get("ledger_bytes") != ledger_witness.size_bytes
        or audit_envelope.get("run_intent_ref") != ledger.run_intent_ref
        or set(factory) != {"lineages", "managed_dispositions", "purpose"}
        or factory.get("purpose") != "evaluation"
    ):
        raise DiagnosticInvariantError("arm-factory audit binding differs")
    raw_lineages = factory.get("lineages")
    lineage_specs = {
        spec.sha256: spec
        for spec in specs.values()
        if spec.label.startswith("LINEAGE_")
    }
    if (
        type(raw_lineages) is not list
        or len(raw_lineages) != 4
        or len(lineage_specs) != 4
    ):
        raise DiagnosticInvariantError("arm-factory lineage inventory differs")
    lineages: dict[tuple[str, str], _LineageBinding] = {}
    consumed_specs: set[Path] = set()
    for raw in raw_lineages:
        row = _mapping(raw, "factory lineage")
        if set(row) != {"arm", "audit", "replicate_commitment"}:
            raise DiagnosticInvariantError("factory lineage fields differ")
        arm = row.get("arm")
        replicate = _require_ref(row.get("replicate_commitment"), "lineage replicate")
        binding = _mapping(row.get("audit"), "lineage audit")
        if arm not in {"FULL", "RANDOM_FEEDBACK"} or set(binding) != {
            "hashes",
            "scope",
            "scope_ref",
        }:
            raise DiagnosticInvariantError("factory lineage identity differs")
        hashes = _mapping(binding.get("hashes"), "lineage hashes")
        if set(hashes) != {"acquisition_sequence", "journal", "learner", "store"}:
            raise DiagnosticInvariantError("lineage hash inventory differs")
        validated_hashes = {
            name: _raw_sha256(value, f"lineage {name}")
            for name, value in hashes.items()
        }
        spec = lineage_specs.get(validated_hashes["store"])
        scope = _mapping(binding.get("scope"), "lineage scope")
        if (
            spec is None
            or spec.path in consumed_specs
            or set(scope)
            != {
                "arm",
                "dataset_id",
                "dataset_name",
                "node_set_name",
                "purpose",
                "replicate_commitment",
                "state_root",
                "tenant_name",
            }
            or (scope.get("arm"), scope.get("purpose"), scope.get("replicate_commitment"))
            != (arm, "evaluation", replicate)
            or binding.get("scope_ref")
            != content_ref("lineage-acquisition-scope", scope)
            or (replicate, str(arm)) in lineages
        ):
            raise DiagnosticInvariantError("lineage scope/store binding differs")
        scope_ref = _require_ref(binding.get("scope_ref"), "lineage scope")
        consumed_specs.add(spec.path)
        lineages[(replicate, str(arm))] = _LineageBinding(
            replicate_ref=replicate,
            arm=str(arm),
            scope_ref=scope_ref,
            spec=spec,
            hashes=validated_hashes,
        )
    replicates = {
        _require_ref(task.get("replicate_commitment"), "manifest replicate")
        for task in tasks.values()
    }
    if (
        set(lineages)
        != {(replicate, arm) for replicate in replicates for arm in ("FULL", "RANDOM_FEEDBACK")}
        or consumed_specs != {spec.path for spec in lineage_specs.values()}
    ):
        raise DiagnosticInvariantError("four-lineage bijection differs")
    raw_dispositions = factory.get("managed_dispositions")
    if type(raw_dispositions) is not list or len(raw_dispositions) != MAX_ATTEMPTS:
        raise DiagnosticInvariantError("factory disposition count differs")
    expected_schedule = _expected_schedule(tasks)
    observed: dict[tuple[str, str], str] = {}
    refs: list[str] = []
    for raw in raw_dispositions:
        envelope = _mapping(raw, "managed disposition envelope")
        if set(envelope) != {"disposition", "disposition_ref"}:
            raise DiagnosticInvariantError("managed disposition envelope differs")
        disposition = _mapping(envelope.get("disposition"), "managed disposition")
        disposition_ref = _require_ref(
            envelope.get("disposition_ref"), "managed disposition"
        )
        task_id = _require_ref(disposition.get("task_id"), "disposition task")
        arm = disposition.get("arm")
        key = (task_id, str(arm))
        ledger_row = ledger.rows.get(key)
        if (
            disposition_ref
            != content_ref("managed-runtime-disposition", disposition)
            or disposition.get("schema") != _MANAGED_DISPOSITION_SCHEMA
            or disposition.get("purpose") != "evaluation"
            or arm not in EXPECTED_ARMS
            or key not in expected_schedule
            or key in observed
            or ledger_row is None
            or disposition.get("phase") != ledger_row.phase
            or disposition.get("evidence_final_ref") != ledger_row.judgment_ref
            or disposition.get("probe_integrity_ref")
            != ledger_row.judgment.get("probe_integrity_ref")
        ):
            raise DiagnosticInvariantError("managed disposition binding differs")
        observed[key] = disposition_ref
        refs.append(disposition_ref)
    if set(observed) != expected_schedule or len(set(refs)) != MAX_ATTEMPTS:
        raise DiagnosticInvariantError("managed disposition coverage differs")
    return _FactoryDataset(lineages=lineages, disposition_refs=tuple(refs))


_MEMORY_RECORD_FIELDS = {
    "acquired_ordinal",
    "competence_ref",
    "confidence_ppm",
    "content",
    "contract",
    "epistemic_status",
    "kind",
    "producer_checkpoint_ref",
    "producer_id",
    "provenance_refs",
    "relations",
    "supersedes_refs",
    "tombstone_refs",
    "visibility",
    "world_valid_from",
    "world_valid_until",
}
_VISIBILITIES = {
    "LEARNER_VISIBLE",
    "CONTROL_PLANE",
    "SEALED_EVALUATION",
    "HUMAN_AUTHORITY",
    "RESTRICTED_PERSONAL",
}
_KINDS = {
    "EPISODIC",
    "SEMANTIC",
    "PROCEDURAL",
    "CAUSAL",
    "SELF",
    "COUNTERFACTUAL",
}
_STATUSES = {"OBSERVED", "PROPOSED", "VALIDATED", "RETRACTED"}
_RELATIONS = {
    "DERIVED_FROM",
    "SUPPORTS",
    "OPPOSES",
    "VERIFIES",
    "CAUSES",
    "PREDICTS",
    "ACTUAL_OF",
    "ALTERNATIVE_TO",
    "ABOUT",
    "IMPLEMENTS",
}


def _ref_list(
    value: object,
    label: str,
    *,
    nonempty: bool = False,
) -> tuple[str, ...]:
    if (
        type(value) is not list
        or (nonempty and not value)
        or value != sorted(value)
        or len(set(value)) != len(value)
        or any(
            type(item) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None
            for item in value
        )
    ):
        raise DiagnosticInvariantError(f"{label} is not a canonical ref list")
    return tuple(value)


def _validate_memory_record(
    value: object,
    *,
    ordinal: int,
    source_contract: str,
    source_ref: str,
) -> tuple[dict[str, object], bytes]:
    record = _mapping(value, "cognitive memory record")
    if (
        set(record) != _MEMORY_RECORD_FIELDS
        or record.get("contract") != _MEMORY_RECORD_CONTRACT
        or record.get("acquired_ordinal") != ordinal
        or record.get("kind") not in _KINDS
        or record.get("epistemic_status") not in _STATUSES
        or record.get("visibility") not in _VISIBILITIES
        or type(record.get("content")) is not str
        or not record.get("content")
        or len(str(record.get("content")).encode("utf-8")) > MAX_TEXT_BYTES
        or record.get("content") != str(record.get("content")).strip()
        or record.get("content") != unicodedata.normalize("NFC", str(record.get("content")))
        or type(record.get("producer_id")) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}", str(record.get("producer_id")))
        is None
    ):
        raise DiagnosticInvariantError("cognitive memory record identity differs")
    for name in ("producer_checkpoint_ref", "competence_ref"):
        _require_ref(record.get(name), f"record {name}")
    provenance = _ref_list(
        record.get("provenance_refs"), "record provenance", nonempty=True
    )
    supersedes = _ref_list(record.get("supersedes_refs"), "record supersedes")
    tombstones = _ref_list(record.get("tombstone_refs"), "record tombstones")
    if (
        source_ref not in provenance
        or set(supersedes) & set(tombstones)
        or (record.get("epistemic_status") == "RETRACTED") != bool(tombstones)
    ):
        raise DiagnosticInvariantError("cognitive record provenance/status differs")
    relations = record.get("relations")
    relation_keys: list[tuple[str, str]] = []
    if type(relations) is not list:
        raise DiagnosticInvariantError("cognitive record relations differ")
    for relation in relations:
        row = _mapping(relation, "cognitive relation")
        if set(row) != {"relation_type", "target_ref"} or row.get(
            "relation_type"
        ) not in _RELATIONS:
            raise DiagnosticInvariantError("cognitive relation fields differ")
        relation_keys.append(
            (str(row["relation_type"]), _require_ref(row.get("target_ref"), "relation target"))
        )
    if relation_keys != sorted(relation_keys) or len(set(relation_keys)) != len(
        relation_keys
    ):
        raise DiagnosticInvariantError("cognitive relations are not canonical")
    for name in ("world_valid_from", "world_valid_until"):
        item = record.get(name)
        if item is not None and (type(item) is not int or item < 0):
            raise DiagnosticInvariantError("cognitive world validity differs")
    valid_from = record.get("world_valid_from")
    valid_until = record.get("world_valid_until")
    if valid_from is not None and valid_until is not None and valid_until < valid_from:
        raise DiagnosticInvariantError("cognitive world validity is reversed")
    confidence = record.get("confidence_ppm")
    if confidence is not None and (
        type(confidence) is not int or not 0 <= confidence <= 1_000_000
    ):
        raise DiagnosticInvariantError("cognitive confidence differs")
    expected_semantics = {
        _LEGACY_EPISODE_CONTRACT: ("EPISODIC", "OBSERVED"),
        _PROSPECTIVE_BATCH_CONTRACT: ("COUNTERFACTUAL", "PROPOSED"),
        _PROSPECTIVE_RESOLUTION_CONTRACT: ("EPISODIC", "OBSERVED"),
    }
    if (record.get("kind"), record.get("epistemic_status")) != expected_semantics[
        source_contract
    ]:
        raise DiagnosticInvariantError("record semantics differ from acquisition source")
    return record, canonical_json_bytes(record)


def _canonical_contract_from_blob(
    raw: object,
    *,
    label: str,
    expected_ref: str,
) -> dict[str, object]:
    if type(raw) is not bytes:
        raise DiagnosticInvariantError(f"{label} is not stored as bytes")
    value = _mapping(
        parse_canonical_json(
            raw,
            label=label,
            maximum_bytes=MAX_EVIDENCE_RECORD_BYTES,
        ),
        label,
    )
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected_ref:
        raise DiagnosticInvariantError(f"{label} reference does not identify its bytes")
    return value


def _scan_cognitive_lineage(
    binding: _LineageBinding,
    *,
    raw_witness: RawMaterialWitness,
    budget: DiagnosticResourceBudget,
) -> _CognitiveLineage:
    acquisition_columns = (
        "ordinal",
        "acquisition_ref",
        "predecessor_acquisition_ref",
        "source_contract",
        "source_ref",
        "record_ref",
        "payload_sha256",
        "canonical_payload",
    )
    projection_columns = (
        "ordinal",
        "acquisition_ref",
        "record_ref",
        "projection_ref",
        "payload_sha256",
        "canonical_payload",
        "acknowledged",
    )
    prospective_columns = (
        "reservation_ref",
        "legacy_reservation_ref",
        "batch_ref",
        "parent_lineage_ref",
        "selected_branch_ref",
        "status",
        "parent_sequence",
        "parent_event_ref",
        "parent_state_digest",
        "parent_snapshot_sha256",
        "model_ref",
        "encoder_ref",
        "reservation_payload_sha256",
        "canonical_reservation",
        "canonical_batch",
        "pending_blob_sha256",
        "pending_blob_size",
        "pending_blob",
        "execution_request_ref",
        "canonical_execution_request",
        "execution_receipt_ref",
        "canonical_execution_receipt",
        "feedback_ref",
        "canonical_feedback",
        "resolution_ref",
        "resolution_disposition",
        "canonical_resolution",
        "legacy_episode_ref",
        "episode_v2_ref",
        "canonical_episode_v2",
        "batch_acquisition_ref",
        "resolution_acquisition_ref",
    )
    records: dict[str, _RetainedRecord] = {}
    with immutable_sqlite(binding.spec) as connection:
        if (
            connection.execute("PRAGMA integrity_check").fetchone() != ("ok",)
            or connection.execute("PRAGMA foreign_key_check").fetchone() is not None
            or connection.execute("PRAGMA user_version").fetchone() != (3,)
            or _table_columns(connection, "cognitive_acquisitions")
            != acquisition_columns
            or _table_columns(connection, "acquisition_projection_outbox")
            != projection_columns
            or _table_columns(connection, "acquisition_clock")
            != ("singleton", "next_ordinal", "acquisition_ref", "record_ref")
            or _table_columns(connection, "prospective_turns_v3")
            != prospective_columns
            or _table_columns(connection, "episodes")
            != (
                "episode_ref",
                "sequence",
                "parent_state_digest",
                "child_state_digest",
                "child_snapshot_sha256",
                "payload_sha256",
                "model_ref",
                "encoder_ref",
                "canonical_payload",
            )
        ):
            raise DiagnosticInvariantError("cognitive SQLite schema/integrity differs")

        episode_sources: dict[str, dict[str, object]] = {}
        episode_cursor = connection.execute(
            "SELECT episode_ref, payload_sha256, canonical_payload FROM episodes "
            "ORDER BY sequence"
        )
        episode_count = 0
        while True:
            batch_rows = episode_cursor.fetchmany(SQL_FETCH_BATCH)
            if not batch_rows:
                break
            for episode_ref, payload_sha, raw_payload in batch_rows:
                episode_count += 1
                if episode_count > MAX_ACQUISITIONS_PER_LINEAGE:
                    raise DiagnosticInvariantError("cognitive episode count exceeds bound")
                reference = _require_ref(episode_ref, "stored episode")
                if payload_sha != reference or reference in episode_sources:
                    raise DiagnosticInvariantError("stored episode identity differs")
                episode = _canonical_contract_from_blob(
                    raw_payload,
                    label="COGNITIVE_EPISODE",
                    expected_ref=reference,
                )
                if (
                    set(episode)
                    != {
                        "child_state_digest",
                        "commitment",
                        "contract",
                        "encoder_ref",
                        "feedback_source_ref",
                        "feedback_text",
                        "model_ref",
                        "observations",
                        "outcome",
                        "parent_state_digest",
                        "proposals",
                        "recalled_refs",
                        "request",
                        "response",
                        "selected_index",
                        "supporting_evidence_refs",
                        "task_id",
                        "visibility",
                    }
                    or episode.get("contract") != _LEGACY_EPISODE_CONTRACT
                ):
                    raise DiagnosticInvariantError("legacy episode contract differs")
                raw_witness.observe_document(episode, category="COGNITIVE_EPISODE")
                episode_sources[reference] = episode
            budget.checkpoint()

        prospective_by_episode: dict[str, _ProspectiveJoin] = {}
        prospective_by_reservation: dict[str, _ProspectiveJoin] = {}
        batch_sources: dict[str, dict[str, object]] = {}
        resolution_sources: dict[str, dict[str, object]] = {}
        prospective_cursor = connection.execute(
            "SELECT reservation_ref, legacy_reservation_ref, batch_ref, "
            "parent_lineage_ref, selected_branch_ref, status, "
            "reservation_payload_sha256, canonical_reservation, canonical_batch, "
            "execution_request_ref, canonical_execution_request, "
            "execution_receipt_ref, canonical_execution_receipt, feedback_ref, "
            "canonical_feedback, resolution_ref, resolution_disposition, "
            "canonical_resolution, legacy_episode_ref, episode_v2_ref, "
            "canonical_episode_v2, batch_acquisition_ref, resolution_acquisition_ref "
            "FROM prospective_turns_v3 ORDER BY rowid"
        )
        prospective_count = 0
        while True:
            batch_rows = prospective_cursor.fetchmany(SQL_FETCH_BATCH)
            if not batch_rows:
                break
            for raw_row in batch_rows:
                prospective_count += 1
                if prospective_count > MAX_ACQUISITIONS_PER_LINEAGE:
                    raise DiagnosticInvariantError("prospective turn count exceeds bound")
                (
                    reservation_ref,
                    legacy_reservation_ref,
                    batch_ref,
                    parent_lineage_ref,
                    selected_branch_ref,
                    status_value,
                    reservation_payload_sha256,
                    canonical_reservation,
                    canonical_batch,
                    execution_request_ref,
                    canonical_execution_request,
                    execution_receipt_ref,
                    canonical_execution_receipt,
                    feedback_ref,
                    canonical_feedback,
                    resolution_ref,
                    resolution_disposition,
                    canonical_resolution,
                    legacy_episode_ref,
                    episode_v2_ref,
                    canonical_episode_v2,
                    batch_acquisition_ref,
                    resolution_acquisition_ref,
                ) = raw_row
                reservation_reference = _require_ref(
                    reservation_ref, "prospective reservation"
                )
                legacy_reservation = _require_ref(
                    legacy_reservation_ref, "legacy reservation"
                )
                batch_reference = _require_ref(batch_ref, "prospective batch")
                if status_value != "RESOLVED":
                    raise DiagnosticInvariantError("retained prospective turn is not resolved")
                batch_value = _canonical_contract_from_blob(
                    canonical_batch,
                    label="PROSPECTIVE_BATCH",
                    expected_ref=batch_reference,
                )
                raw_witness.observe_document(batch_value, category="PROSPECTIVE_BATCH")
                reservation_value = _canonical_contract_from_blob(
                    canonical_reservation,
                    label="PROSPECTIVE_RESERVATION",
                    expected_ref=reservation_reference,
                )
                raw_witness.observe_document(
                    reservation_value, category="PROSPECTIVE_RESERVATION"
                )
                if reservation_payload_sha256 != reservation_reference:
                    raise DiagnosticInvariantError(
                        "prospective reservation bytes/reference differ"
                    )
                if batch_reference in batch_sources:
                    raise DiagnosticInvariantError("prospective batch is duplicated")
                batch_sources[batch_reference] = batch_value
                resolved_ref = _optional_ref(resolution_ref, "prospective resolution")
                resolved_episode = _optional_ref(legacy_episode_ref, "legacy episode")
                resolution_value: dict[str, object] | None = None
                request_value: dict[str, object] | None = None
                receipt_value: dict[str, object] | None = None
                feedback_value: dict[str, object] | None = None
                for source_label, reference_value, raw_value in (
                    (
                        "EXECUTION_REQUEST",
                        execution_request_ref,
                        canonical_execution_request,
                    ),
                    (
                        "EXECUTION_RECEIPT",
                        execution_receipt_ref,
                        canonical_execution_receipt,
                    ),
                    ("OBJECTIVE_FEEDBACK", feedback_ref, canonical_feedback),
                ):
                    optional_reference = _optional_ref(
                        reference_value, source_label.lower()
                    )
                    parsed_value = None
                    if optional_reference is not None:
                        parsed_value = _canonical_contract_from_blob(
                            raw_value,
                            label=source_label,
                            expected_ref=optional_reference,
                        )
                        raw_witness.observe_document(parsed_value, category=source_label)
                    elif raw_value is not None:
                        raise DiagnosticInvariantError(
                            "prospective lifecycle bytes have no reference"
                        )
                    if source_label == "EXECUTION_REQUEST":
                        request_value = parsed_value
                    elif source_label == "EXECUTION_RECEIPT":
                        receipt_value = parsed_value
                    else:
                        feedback_value = parsed_value
                if resolved_ref is not None:
                    resolution_value = _canonical_contract_from_blob(
                        canonical_resolution,
                        label="PROSPECTIVE_RESOLUTION",
                        expected_ref=resolved_ref,
                    )
                    raw_witness.observe_document(
                        resolution_value, category="PROSPECTIVE_RESOLUTION"
                    )
                    if resolved_ref in resolution_sources:
                        raise DiagnosticInvariantError("prospective resolution is duplicated")
                    resolution_sources[resolved_ref] = resolution_value
                elif canonical_resolution is not None:
                    raise DiagnosticInvariantError("orphan prospective resolution bytes")
                if resolved_episode is not None and resolved_episode not in episode_sources:
                    raise DiagnosticInvariantError("prospective legacy episode is absent")
                episode_v2_reference = _optional_ref(
                    episode_v2_ref, "prospective episode v2"
                )
                if episode_v2_reference is not None:
                    episode_v2_value = _canonical_contract_from_blob(
                        canonical_episode_v2,
                        label="COGNITIVE_EPISODE_V2",
                        expected_ref=episode_v2_reference,
                    )
                    raw_witness.observe_document(
                        episode_v2_value, category="COGNITIVE_EPISODE_V2"
                    )
                elif canonical_episode_v2 is not None:
                    raise DiagnosticInvariantError("orphan prospective episode-v2 bytes")
                if (
                    resolved_ref is None
                    or resolution_disposition
                    not in {
                        "OBSERVED",
                        "COMPLETED_UNEVALUATED",
                        "CANCELLED",
                        "CLARIFICATION_REQUIRED",
                        "ERROR",
                    }
                    or (resolution_disposition == "OBSERVED")
                    != (resolved_episode is not None and episode_v2_reference is not None)
                ):
                    raise DiagnosticInvariantError(
                        "prospective resolved-row reference state differs"
                    )
                join = _ProspectiveJoin(
                    legacy_reservation_ref=legacy_reservation,
                    batch_ref=batch_reference,
                    resolution_ref=resolved_ref,
                    legacy_episode_ref=resolved_episode,
                    batch_acquisition_ref=_require_ref(
                        batch_acquisition_ref, "batch acquisition"
                    ),
                    resolution_acquisition_ref=_optional_ref(
                        resolution_acquisition_ref, "resolution acquisition"
                    ),
                )
                if legacy_reservation in prospective_by_reservation:
                    raise DiagnosticInvariantError("legacy reservation is duplicated")
                prospective_by_reservation[legacy_reservation] = join
                if resolved_episode is not None:
                    if resolved_episode in prospective_by_episode:
                        raise DiagnosticInvariantError("legacy episode join is duplicated")
                    prospective_by_episode[resolved_episode] = join
            budget.checkpoint()

        rows = connection.execute(
            "SELECT a.ordinal, a.acquisition_ref, a.predecessor_acquisition_ref, "
            "a.source_contract, a.source_ref, a.record_ref, a.payload_sha256, "
            "a.canonical_payload, o.acquisition_ref, o.record_ref, o.projection_ref, "
            "o.payload_sha256, o.canonical_payload, o.acknowledged "
            "FROM cognitive_acquisitions AS a JOIN acquisition_projection_outbox AS o "
            "ON o.ordinal=a.ordinal ORDER BY a.ordinal"
        )
        previous: str | None = None
        acquisition_refs: set[str] = set()
        acquisition_sources: dict[str, tuple[str, str]] = {}
        acquisition_count = 0
        while True:
            batch_rows = rows.fetchmany(SQL_FETCH_BATCH)
            if not batch_rows:
                break
            for raw_row in batch_rows:
                acquisition_count += 1
                if acquisition_count > MAX_ACQUISITIONS_PER_LINEAGE:
                    raise DiagnosticInvariantError("cognitive acquisition count exceeds bound")
                (
                    ordinal,
                    acquisition_ref,
                    predecessor_ref,
                    source_contract,
                    source_ref,
                    record_ref,
                    payload_sha,
                    raw_acquisition,
                    out_acquisition_ref,
                    out_record_ref,
                    projection_ref,
                    projection_sha,
                    raw_projection,
                    acknowledged,
                ) = raw_row
                if type(ordinal) is not int or ordinal != acquisition_count - 1:
                    raise DiagnosticInvariantError("acquisition ordinal is not contiguous")
                acquisition_reference = _require_ref(
                    acquisition_ref, "cognitive acquisition"
                )
                record_reference = _require_ref(record_ref, "cognitive record")
                source_reference = _require_ref(source_ref, "acquisition source")
                if (
                    source_contract
                    not in {
                        _LEGACY_EPISODE_CONTRACT,
                        _PROSPECTIVE_BATCH_CONTRACT,
                        _PROSPECTIVE_RESOLUTION_CONTRACT,
                    }
                    or predecessor_ref != previous
                    or payload_sha != acquisition_reference
                    or out_acquisition_ref != acquisition_reference
                    or out_record_ref != record_reference
                    or acknowledged not in {0, 1}
                    or acquisition_reference in acquisition_refs
                    or record_reference in records
                ):
                    raise DiagnosticInvariantError("cognitive acquisition row differs")
                acquisition = _canonical_contract_from_blob(
                    raw_acquisition,
                    label="COGNITIVE_ACQUISITION",
                    expected_ref=acquisition_reference,
                )
                if (
                    set(acquisition)
                    != {
                        "contract",
                        "ordinal",
                        "predecessor_acquisition_ref",
                        "record",
                        "source_contract",
                        "source_ref",
                    }
                    or acquisition.get("contract") != _ACQUISITION_CONTRACT
                    or (
                        acquisition.get("ordinal"),
                        acquisition.get("predecessor_acquisition_ref"),
                        acquisition.get("source_contract"),
                        acquisition.get("source_ref"),
                    )
                    != (ordinal, predecessor_ref, source_contract, source_reference)
                ):
                    raise DiagnosticInvariantError("canonical acquisition fields differ")
                record, record_bytes = _validate_memory_record(
                    acquisition.get("record"),
                    ordinal=ordinal,
                    source_contract=str(source_contract),
                    source_ref=source_reference,
                )
                if "sha256:" + hashlib.sha256(record_bytes).hexdigest() != record_reference:
                    raise DiagnosticInvariantError("record reference does not identify bytes")
                projection_reference = _require_ref(
                    projection_ref, "cognitive projection"
                )
                if projection_sha != projection_reference:
                    raise DiagnosticInvariantError("projection payload hash differs")
                projection = _canonical_contract_from_blob(
                    raw_projection,
                    label="COGNITIVE_PROJECTION",
                    expected_ref=projection_reference,
                )
                if (
                    set(projection)
                    != {
                        "acquisition_ref",
                        "contract",
                        "record",
                        "source_contract",
                        "source_ref",
                    }
                    or projection.get("contract") != _GRAPH_PROJECTION_CONTRACT
                    or projection.get("acquisition_ref") != acquisition_reference
                    or projection.get("source_contract") != source_contract
                    or projection.get("source_ref") != source_reference
                    or projection.get("record") != record
                ):
                    raise DiagnosticInvariantError("cognitive projection binding differs")
                source_value = {
                    _LEGACY_EPISODE_CONTRACT: episode_sources,
                    _PROSPECTIVE_BATCH_CONTRACT: batch_sources,
                    _PROSPECTIVE_RESOLUTION_CONTRACT: resolution_sources,
                }[str(source_contract)].get(source_reference)
                if source_value is None:
                    raise DiagnosticInvariantError("cognitive acquisition source is absent")
                ancestry = {source_reference, acquisition_reference, projection_reference}
                for join in prospective_by_reservation.values():
                    if source_reference in {
                        join.batch_ref,
                        join.resolution_ref,
                        join.legacy_episode_ref,
                    }:
                        ancestry.add(join.legacy_reservation_ref)
                        ancestry.add(join.batch_ref)
                        if join.resolution_ref is not None:
                            ancestry.add(join.resolution_ref)
                        if join.legacy_episode_ref is not None:
                            ancestry.add(join.legacy_episode_ref)
                raw_witness.observe_cognitive_record(
                    binding.scope_ref, record_reference, record
                )
                records[record_reference] = _RetainedRecord(
                    lineage_scope_ref=binding.scope_ref,
                    acquisition_ref=acquisition_reference,
                    record_ref=record_reference,
                    record_bytes_sha256=hashlib.sha256(record_bytes).hexdigest(),
                    source_contract=str(source_contract),
                    source_ref=source_reference,
                    content=str(record["content"]),
                    ancestry_refs=tuple(sorted(ancestry)),
                )
                acquisition_refs.add(acquisition_reference)
                acquisition_sources[acquisition_reference] = (
                    str(source_contract),
                    source_reference,
                )
                previous = acquisition_reference
            budget.checkpoint()
        clock = connection.execute(
            "SELECT next_ordinal, acquisition_ref, record_ref FROM acquisition_clock "
            "WHERE singleton=1"
        ).fetchone()
        if (
            clock is None
            or clock[0] != acquisition_count
            or (acquisition_count == 0 and clock[1:] != (None, None))
            or acquisition_count > 0
            and (clock[1] != previous or clock[2] not in records)
        ):
            raise DiagnosticInvariantError("cognitive acquisition clock differs")
        for join in prospective_by_reservation.values():
            if (
                acquisition_sources.get(join.batch_acquisition_ref)
                != (_PROSPECTIVE_BATCH_CONTRACT, join.batch_ref)
                or join.resolution_acquisition_ref is not None
                and acquisition_sources.get(join.resolution_acquisition_ref)
                != (_PROSPECTIVE_RESOLUTION_CONTRACT, join.resolution_ref)
            ):
                raise DiagnosticInvariantError("prospective acquisition join is absent")
    return _CognitiveLineage(
        binding=binding,
        records=records,
        prospective_by_episode=prospective_by_episode,
        prospective_by_reservation=prospective_by_reservation,
        acquisition_count=acquisition_count,
    )


def _target_task_ids(
    ledger: _LedgerDataset,
) -> tuple[str, ...]:
    targets = tuple(
        sorted(
            row.task_id
            for row in ledger.rows.values()
            if row.phase == "final" and row.arm == "QWEN_ONLY" and row.success
        )
    )
    qwen_final_metrics = {
        key[2]: value
        for key, value in ledger.metric_counts.items()
        if key[:2] == ("final", "QWEN_ONLY")
    }
    if (
        len(targets) != EXPECTED_COHORT_TASKS
        or any(ledger.rows[(task_id, "QWEN_ONLY")].family != "glyph-machine" for task_id in targets)
        or sum(value[0] for value in qwen_final_metrics.values()) != 40
        or sum(value[1] for value in qwen_final_metrics.values()) != 3
        or qwen_final_metrics.get("glyph-machine") != (8, 3)
    ):
        raise DiagnosticInvariantError(
            "full population does not contain the exact three glyph QWEN_ONLY successes"
        )
    return targets


def _selection_preimage(stage: Mapping[str, object]) -> dict[str, object] | None:
    value = stage.get("selection")
    if value is None:
        return None
    selection = _mapping(value, "selection preimage")
    return {
        "selected_index": selection.get("selected_index"),
        "selected_trace": selection.get("selected_trace"),
    }


def _build_trace_matrix(
    target_ids: Sequence[str],
    *,
    ledger: _LedgerDataset,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], AttemptTrace]]:
    traces: dict[tuple[str, str], AttemptTrace] = {}
    for task_id in target_ids:
        for arm in EXPECTED_ARMS:
            row = ledger.rows.get((task_id, arm))
            if row is None:
                raise DiagnosticInvariantError("target matrix row is absent")
            traces[(task_id, arm)] = extract_attempt_trace(
                row.stage_bytes,
                row.judgment_bytes,
                stage_ref=row.stage_ref,
                judgment_ref=row.judgment_ref,
            )
    validate_unique_cohort(tuple(traces.values()))
    matrix: list[dict[str, object]] = []
    candidate_to_response = {
        "DECLARED_GRAMMAR_CONFORMANT": "CONFORMANT",
        "DECLARED_SHAPE_FOREIGN_TOKEN": "FOREIGN_TOKEN",
        "DECLARED_SHAPE_MALFORMED": "MALFORMED_SHAPE",
        "OUTSIDE_DECLARED_GRAMMAR": "OUTSIDE_DECLARED_GRAMMAR",
        "EMPTY": "EMPTY",
    }
    for task_id in sorted(target_ids):
        rows = {arm: ledger.rows[(task_id, arm)] for arm in EXPECTED_ARMS}
        task_traces = {arm: traces[(task_id, arm)] for arm in EXPECTED_ARMS}

        def comparison(arm: str) -> dict[str, object]:
            row = rows[arm]
            trace = task_traces[arm]
            request = _mapping(row.stage.get("proposal_request"), "proposal request")
            objective = _mapping(
                row.judgment.get("objective_judgment"), "objective judgment"
            )
            return {
                "parser_disposition": trace.parser_disposition,
                "history_preimage": {
                    "frozen_recall_ref": row.stage.get("frozen_recall_ref"),
                    "recalled_evidence": request.get("recalled_evidence"),
                },
                "candidate_classes": list(trace.candidate_classes),
                "selection": _selection_preimage(row.stage),
                "response_conformance": trace.response_class,
                "objective_disposition": trace.objective_disposition,
                "objective_score": objective.get("score"),
            }

        comparisons = {arm: comparison(arm) for arm in EXPECTED_ARMS}
        reference = comparisons["QWEN_ONLY"]
        equalities = {
            "recall_preimage_equality": [
                list(group)
                for group in equality_groups(
                    {
                        arm: comparisons[arm]["history_preimage"]
                        for arm in EXPECTED_ARMS
                    }
                )
            ],
            "proposal_request_equality": [
                list(group)
                for group in equality_groups(
                    {
                        arm: rows[arm].stage.get("proposal_request")
                        for arm in EXPECTED_ARMS
                    }
                )
            ],
            "proposal_equality": [
                list(group)
                for group in equality_groups(
                    {arm: rows[arm].stage.get("proposals") for arm in EXPECTED_ARMS}
                )
            ],
            "selected_trace_equality": [
                list(group)
                for group in equality_groups(
                    {arm: _selection_preimage(rows[arm].stage) for arm in EXPECTED_ARMS}
                )
            ],
            "response_equality": [
                list(group)
                for group in equality_groups(
                    {arm: rows[arm].stage.get("raw_response") for arm in EXPECTED_ARMS}
                )
            ],
            "objective_score_equality": [
                list(group)
                for group in equality_groups(
                    {
                        arm: _mapping(
                            rows[arm].judgment.get("objective_judgment"),
                            "objective judgment",
                        ).get("score")
                        for arm in EXPECTED_ARMS
                    }
                )
            ],
        }
        divergence = [
            {
                "arm": arm,
                "field": first_divergence(reference, comparisons[arm]),
            }
            for arm in EXPECTED_ARMS
        ]
        mismatches = 0
        for trace in task_traces.values():
            if trace.parser_disposition == "ADMITTED" and candidate_to_response.get(
                trace.selected_class
            ) != trace.response_class:
                mismatches += 1
        matrix.append(
            {
                "attempts": [task_traces[arm].redacted() for arm in EXPECTED_ARMS],
                "equality_groups": equalities,
                "first_divergences": divergence,
                "replicate_ref": rows["FULL"].replicate_ref,
                "selected_execution_class_mismatch_count": mismatches,
                "task_ref": task_id,
            }
        )
    return matrix, traces


def _rejoin_target_provenance(
    target_ids: Sequence[str],
    *,
    tasks: Mapping[str, Mapping[str, object]],
    ledger: _LedgerDataset,
    removals: _RemovalDataset,
    factory: _FactoryDataset,
    cognitive: Mapping[tuple[str, str], _CognitiveLineage],
) -> dict[str, object]:
    occurrences: list[dict[str, object]] = []
    occurrence_keys: set[tuple[str, str, str, int, str]] = set()
    target_replicates: set[str] = set()
    for task_id in sorted(target_ids):
        row = ledger.rows[(task_id, "FULL")]
        replicate = row.replicate_ref
        target_replicates.add(replicate)
        binding = factory.lineages.get((replicate, "FULL"))
        lineage = cognitive.get((replicate, "FULL"))
        if binding is None or lineage is None or lineage.binding != binding:
            raise DiagnosticInvariantError("target FULL lineage mapping is absent")
        probe = _mapping(row.judgment.get("probe_integrity"), "target FULL probe")
        before = probe.get("source_baseline_before")
        after = probe.get("source_baseline_after")
        if (
            type(before) is not dict
            or type(after) is not dict
            or before != after
            or set(before) != {"acquisition_sequence", "journal", "learner", "store"}
            or before.get("store") != binding.hashes["store"]
            or before.get("store") != binding.spec.sha256
            or probe.get("replicate_commitment") != replicate
        ):
            raise DiagnosticInvariantError("target FULL baseline does not map to retained store")
        removal = removals.records.get(task_id)
        if removal is None or removal.get("phase") != "final":
            raise DiagnosticInvariantError("target final removal evidence is absent")
        full = _mapping(_mapping(removal.get("arms"), "target removal arms").get("FULL"), "target FULL removal")
        refs = full.get("recalled_record_refs")
        hashes = full.get("recalled_record_bytes_sha256")
        request = _mapping(row.stage.get("proposal_request"), "target FULL request")
        recalled = request.get("recalled_evidence")
        if (
            type(refs) is not list
            or type(hashes) is not list
            or type(recalled) is not list
            or not 1 <= len(refs) <= 12
            or len(refs) != len(hashes)
            or len(refs) != len(recalled)
            or full.get("frozen_recall_ref") != row.stage.get("frozen_recall_ref")
        ):
            raise DiagnosticInvariantError("target FULL recall cardinality differs")
        for recall_index, (record_ref, record_hash, recalled_text) in enumerate(
            zip(refs, hashes, recalled, strict=True)
        ):
            retained = lineage.records.get(str(record_ref))
            if (
                retained is None
                or retained.lineage_scope_ref != binding.scope_ref
                or retained.record_bytes_sha256 != record_hash
                or retained.record_ref.removeprefix("sha256:") != record_hash
                or retained.content != recalled_text
            ):
                raise DiagnosticInvariantError("target recall does not rejoin retained record")
            key = (task_id, replicate, binding.scope_ref, recall_index, retained.record_ref)
            if key in occurrence_keys:
                raise DiagnosticInvariantError("target recall occurrence is duplicated")
            occurrence_keys.add(key)
            occurrences.append(
                {
                    "acquisition_ref": retained.acquisition_ref,
                    "ancestry_refs": list(retained.ancestry_refs),
                    "recall_index": recall_index,
                    "record_bytes_ref": "sha256:" + retained.record_bytes_sha256,
                    "record_ref": retained.record_ref,
                    "source_contract": retained.source_contract,
                    "source_ref": retained.source_ref,
                    "target_full_replicate_ref": replicate,
                    "target_lineage_scope_ref": binding.scope_ref,
                    "target_task_id": task_id,
                }
            )

    lineage_summaries: list[dict[str, object]] = []
    for replicate in sorted(target_replicates):
        lineage = cognitive[(replicate, "FULL")]
        adaptation_task_ids = [
            task_id
            for task_id, task in tasks.items()
            if task.get("phase") == "adaptation"
            and task.get("replicate_commitment") == replicate
        ]
        if len(adaptation_task_ids) != 12:
            raise DiagnosticInvariantError("target replicate adaptation schedule differs")
        admitted = 0
        malformed = 0
        episode_refs: list[str] = []
        for task_id in adaptation_task_ids:
            row = ledger.rows[(task_id, "FULL")]
            parser = row.stage.get("parser_disposition")
            transition_value = row.judgment.get("learner_transition")
            if parser == "MALFORMED":
                if transition_value is not None or row.stage.get("selection") is not None:
                    raise DiagnosticInvariantError("malformed adaptation row advanced")
                malformed += 1
                continue
            if parser != "ADMITTED":
                raise DiagnosticInvariantError("adaptation parser disposition differs")
            transition = _mapping(transition_value, "adaptation transition")
            selection = _mapping(row.stage.get("selection"), "adaptation selection")
            episode_ref = _require_ref(transition.get("episode_ref"), "adaptation episode")
            reservation_ref = _require_ref(
                selection.get("reservation_ref"), "adaptation reservation"
            )
            by_episode = lineage.prospective_by_episode.get(episode_ref)
            by_reservation = lineage.prospective_by_reservation.get(reservation_ref)
            probe = _mapping(row.judgment.get("probe_integrity"), "adaptation probe")
            if (
                by_episode is None
                or by_reservation is None
                or by_episode != by_reservation
                or by_episode.legacy_episode_ref != episode_ref
                or by_episode.legacy_reservation_ref != reservation_ref
                or by_episode.resolution_ref is None
                or probe.get("learner_parent_digest")
                != transition.get("parent_state_digest")
                or probe.get("learner_child_digest")
                != transition.get("child_state_digest")
                or probe.get("learner_sequence") != transition.get("sequence")
            ):
                raise DiagnosticInvariantError("adaptation episode/reservation join differs")
            admitted += 1
            episode_refs.append(episode_ref)
        if admitted + malformed != 12 or admitted == 0 or len(set(episode_refs)) != admitted:
            raise DiagnosticInvariantError("adaptation ancestry cardinality differs")
        lineage_summaries.append(
            {
                "adaptation_attempt_count": 12,
                "admitted_transition_count": admitted,
                "joined_episode_refs": episode_refs,
                "lineage_scope_ref": lineage.binding.scope_ref,
                "malformed_transition_count": malformed,
                "replicate_ref": replicate,
            }
        )
    unique_records = {
        (str(row["target_lineage_scope_ref"]), str(row["record_ref"]))
        for row in occurrences
    }
    scopes_by_raw_ref: dict[str, set[str]] = {}
    for scope_ref, record_ref in unique_records:
        scopes_by_raw_ref.setdefault(record_ref, set()).add(scope_ref)
    collision_count = sum(len(scopes) > 1 for scopes in scopes_by_raw_ref.values())
    return {
        "adaptation_episode_joined": True,
        "all_rejoined": True,
        "frozen_batch_recomputed": False,
        "join_method": "PERSISTED_REF_HASH_CONTENT_TO_RETAINED_RECORD",
        "prospective_contract_recomputed": False,
        "prospective_contract_recomputation_disposition": "INDETERMINATE",
        "lineages": lineage_summaries,
        "occurrence_count": len(occurrences),
        "occurrences": occurrences,
        "raw_record_ref_collision_count": collision_count,
        "unique_record_count": len(unique_records),
    }


_ATTEMPT_OUTPUT_FIELDS = {
    "arm",
    "attempt_receipt_ref",
    "candidate_classes",
    "episode_ref",
    "execution_receipt_ref",
    "execution_request_ref",
    "history_class",
    "judgment_ref",
    "learner_child_ref",
    "learner_parent_ref",
    "learner_sequence",
    "objective_disposition",
    "objective_success",
    "parser_disposition",
    "phase",
    "recall_batch_ref",
    "recalled_record_count",
    "reservation_ref",
    "response_class",
    "response_length",
    "selected_class",
    "selected_index",
    "selected_response_equal",
    "selection_ref",
    "stage_ref",
    "task_id",
}

_CANDIDATE_CLASSES = {
    "DECLARED_GRAMMAR_CONFORMANT",
    "DECLARED_SHAPE_FOREIGN_TOKEN",
    "DECLARED_SHAPE_MALFORMED",
    "EMPTY",
    "OUTSIDE_DECLARED_GRAMMAR",
}
_RESPONSE_CLASSES = {
    "CONFORMANT",
    "EMPTY",
    "FOREIGN_TOKEN",
    "MALFORMED_SHAPE",
    "NOT_EXECUTED",
    "OUTSIDE_DECLARED_GRAMMAR",
}


def _validate_closed_attempt(value: object, *, arm: str, task_id: str) -> AttemptTrace:
    attempt = _require_exact_keys(value, _ATTEMPT_OUTPUT_FIELDS, "matrix attempt")
    candidates = attempt.get("candidate_classes")
    if (
        type(candidates) is not list
        or any(type(item) is not str or item not in _CANDIDATE_CLASSES for item in candidates)
        or attempt.get("arm") != arm
        or attempt.get("task_id") != task_id
        or attempt.get("phase") != "FINAL"
        or attempt.get("parser_disposition") not in {"ADMITTED", "MALFORMED"}
        or attempt.get("history_class")
        not in {"NO_PERSISTENT_HISTORY", "FROZEN_RECALL"}
        or attempt.get("selected_class")
        not in {*_CANDIDATE_CLASSES, "NOT_SELECTED"}
        or attempt.get("response_class") not in _RESPONSE_CLASSES
        or attempt.get("objective_disposition") not in {"SUCCESS", "UNSUCCESSFUL"}
        or type(attempt.get("objective_success")) is not bool
        or type(attempt.get("response_length")) is not int
        or not 0 <= int(attempt["response_length"]) <= MAX_TEXT_BYTES
        or type(attempt.get("selected_response_equal")) is not bool
        or type(attempt.get("recalled_record_count")) is not int
        or not 0 <= int(attempt["recalled_record_count"]) <= 12
    ):
        raise DiagnosticInvariantError("matrix attempt typed fields differ")
    for name in ("attempt_receipt_ref", "judgment_ref", "stage_ref"):
        _require_ref(attempt.get(name), f"matrix {name}")
    for name in (
        "episode_ref",
        "execution_receipt_ref",
        "execution_request_ref",
        "learner_child_ref",
        "learner_parent_ref",
        "recall_batch_ref",
        "reservation_ref",
        "selection_ref",
    ):
        _optional_ref(attempt.get(name), f"matrix {name}")
    selected_index = attempt.get("selected_index")
    learner_sequence = attempt.get("learner_sequence")
    if selected_index is not None and (
        type(selected_index) is not int or not 0 <= selected_index < len(candidates)
    ):
        raise DiagnosticInvariantError("matrix selected index differs")
    if learner_sequence is not None and (
        type(learner_sequence) is not int or learner_sequence < 1
    ):
        raise DiagnosticInvariantError("matrix learner sequence differs")
    if (
        (attempt["objective_disposition"] == "SUCCESS")
        != attempt["objective_success"]
        or (
            learner_sequence is None
            and any(
                attempt[name] is not None
                for name in (
                    "episode_ref",
                    "learner_child_ref",
                    "learner_parent_ref",
                )
            )
        )
        or (
            learner_sequence is not None
            and any(
                attempt[name] is None
                for name in (
                    "episode_ref",
                    "learner_child_ref",
                    "learner_parent_ref",
                )
            )
        )
    ):
        raise DiagnosticInvariantError("matrix outcome/learner correlation differs")
    if arm == "QWEN_ONLY":
        if (
            attempt["history_class"] != "NO_PERSISTENT_HISTORY"
            or attempt["recall_batch_ref"] is not None
            or attempt["recalled_record_count"] != 0
            or attempt["objective_success"] is not True
        ):
            raise DiagnosticInvariantError("matrix QWEN_ONLY boundary differs")
    elif (
        attempt["history_class"] != "FROZEN_RECALL"
        or attempt["recall_batch_ref"] is None
        or not 1 <= int(attempt["recalled_record_count"]) <= 12
    ):
        raise DiagnosticInvariantError("matrix frozen-recall boundary differs")
    if attempt["parser_disposition"] == "MALFORMED":
        if (
            candidates
            or selected_index is not None
            or attempt["selected_class"] != "NOT_SELECTED"
            or attempt["selection_ref"] is not None
            or attempt["reservation_ref"] is not None
            or attempt["execution_request_ref"] is not None
            or attempt["execution_receipt_ref"] is not None
            or attempt["response_class"] != "NOT_EXECUTED"
            or attempt["response_length"] != 0
            or attempt["selected_response_equal"] is not False
            or attempt["objective_success"] is not False
        ):
            raise DiagnosticInvariantError("matrix malformed-attempt boundary differs")
    elif (
        not candidates
        or selected_index is None
        or attempt["selected_class"] != candidates[selected_index]
        or attempt["selection_ref"] is None
        or attempt["execution_request_ref"] is None
        or attempt["execution_receipt_ref"] is None
        or attempt["response_class"] == "NOT_EXECUTED"
    ):
        raise DiagnosticInvariantError("matrix admitted-attempt boundary differs")
    try:
        return AttemptTrace(
            task_id=task_id,
            arm=arm,
            phase="FINAL",
            attempt_receipt_ref=str(attempt["attempt_receipt_ref"]),
            stage_ref=str(attempt["stage_ref"]),
            judgment_ref=str(attempt["judgment_ref"]),
            parser_disposition=str(attempt["parser_disposition"]),
            history_class=str(attempt["history_class"]),
            recall_batch_ref=attempt["recall_batch_ref"],  # type: ignore[arg-type]
            recalled_record_count=int(attempt["recalled_record_count"]),
            candidate_classes=tuple(candidates),
            selected_index=selected_index,  # type: ignore[arg-type]
            selected_class=str(attempt["selected_class"]),
            selection_ref=attempt["selection_ref"],  # type: ignore[arg-type]
            reservation_ref=attempt["reservation_ref"],  # type: ignore[arg-type]
            response_class=str(attempt["response_class"]),
            response_length=int(attempt["response_length"]),
            selected_response_equal=bool(attempt["selected_response_equal"]),
            objective_success=bool(attempt["objective_success"]),
            objective_disposition=str(attempt["objective_disposition"]),
            execution_request_ref=attempt["execution_request_ref"],  # type: ignore[arg-type]
            execution_receipt_ref=attempt["execution_receipt_ref"],  # type: ignore[arg-type]
            learner_parent_ref=attempt["learner_parent_ref"],  # type: ignore[arg-type]
            learner_child_ref=attempt["learner_child_ref"],  # type: ignore[arg-type]
            learner_sequence=learner_sequence,  # type: ignore[arg-type]
            episode_ref=attempt["episode_ref"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise DiagnosticInvariantError("matrix attempt contract differs") from error


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, object]:
    row = _mapping(value, label)
    if set(row) != keys:
        raise DiagnosticInvariantError(f"{label} fields differ")
    return row


def _validate_partition(value: object) -> None:
    if type(value) is not list or not value or any(type(group) is not list for group in value):
        raise DiagnosticInvariantError("matrix equality partition differs")
    flattened = [arm for group in value for arm in group]
    if sorted(flattened) != sorted(EXPECTED_ARMS) or len(flattened) != len(
        set(flattened)
    ):
        raise DiagnosticInvariantError("matrix equality partition lacks exact arms")


def _validate_closed_diagnostic_output(value: object) -> None:
    document = _require_exact_keys(
        value,
        {
            "schema",
            "identity",
            "predecessor",
            "inputs",
            "population",
            "matrix",
            "provenance",
            "questions",
            "resources",
        },
        "diagnostic output",
    )
    if (
        document.get("schema") != DIAGNOSTIC_SCHEMA
        or document.get("identity") != DIAGNOSTIC_IDENTITY
    ):
        raise DiagnosticInvariantError("diagnostic output identity differs")
    predecessor = _require_exact_keys(
        document.get("predecessor"),
        {
            "arm_factory_audit_ref",
            "classification",
            "diagnostic_leaf_ref",
            "ledger_ref",
            "manifest_ref",
            "predecessor_leaf_ref",
            "result_ref",
            "result_report_ref",
            "runner_ref",
        },
        "diagnostic predecessor",
    )
    if predecessor.get("classification") != "NOT_SUPPORTED":
        raise DiagnosticInvariantError("diagnostic predecessor classification differs")
    for name, item in predecessor.items():
        if name != "classification":
            _require_ref(item, f"predecessor {name}")
    inputs = _require_exact_keys(
        document.get("inputs"), {"after", "before", "unchanged"}, "diagnostic inputs"
    )
    if inputs.get("unchanged") is not True:
        raise DiagnosticInvariantError("diagnostic input unchanged flag differs")
    expected_labels = sorted(spec.label for spec in FROZEN_INPUTS)
    expected_specs = {spec.label: spec for spec in FROZEN_INPUTS}
    input_witnesses: dict[str, dict[str, dict[str, object]]] = {}
    for name in ("before", "after"):
        rows = inputs.get(name)
        if type(rows) is not list or len(rows) != len(FROZEN_INPUTS):
            raise DiagnosticInvariantError("diagnostic input witness count differs")
        labels: list[str] = []
        section: dict[str, dict[str, object]] = {}
        for raw in rows:
            witness = _require_exact_keys(
                raw,
                {"gid", "label", "mode", "nlink", "sha256", "size_bytes", "uid"},
                "redacted input witness",
            )
            label = witness.get("label")
            if type(label) is not str or label not in expected_labels:
                raise DiagnosticInvariantError("redacted input label differs")
            labels.append(label)
            if witness.get("sha256") != "sha256:" + expected_specs[label].sha256:
                raise DiagnosticInvariantError("redacted input frozen hash differs")
            for field_name in ("gid", "mode", "nlink", "size_bytes", "uid"):
                if type(witness.get(field_name)) is not int or int(witness[field_name]) < 0:
                    raise DiagnosticInvariantError("redacted input metadata differs")
            if witness.get("mode") != expected_specs[label].expected_mode:
                raise DiagnosticInvariantError("redacted input frozen mode differs")
            section[label] = witness
        if labels != expected_labels:
            raise DiagnosticInvariantError("redacted input order differs")
        input_witnesses[name] = section
    if inputs.get("before") != inputs.get("after"):
        raise DiagnosticInvariantError("redacted input witnesses changed")
    predecessor_inputs = {
        "arm_factory_audit_ref": "R2_ARM_FACTORY_AUDIT",
        "diagnostic_leaf_ref": "DIAGNOSTIC_LEAF",
        "ledger_ref": "R2_LEDGER",
        "manifest_ref": "R2_SOURCE_MANIFEST",
        "predecessor_leaf_ref": "R2_PREDECESSOR_LEAF",
        "result_ref": "R2_RESULT",
        "result_report_ref": "R2_RESULT_REPORT",
        "runner_ref": "R2_RUNNER",
    }
    if any(
        predecessor.get(field_name)
        != input_witnesses["before"][label]["sha256"]
        for field_name, label in predecessor_inputs.items()
    ):
        raise DiagnosticInvariantError("predecessor/input hash linkage differs")

    population = _require_exact_keys(
        document.get("population"),
        {
            "admitted_count",
            "arms_per_task",
            "attempt_count",
            "factory_disposition_count",
            "finalized_receipt_count",
            "malformed_count",
            "matrix_attempt_count",
            "persistent_lineage_count",
            "raw_record_ref_collision_count",
            "recall_occurrence_count",
            "removal_row_count",
            "target_lineage_count",
            "target_task_count",
            "unique_recalled_record_count",
        },
        "diagnostic population",
    )
    fixed_counts = {
        "admitted_count": 433,
        "arms_per_task": 7,
        "attempt_count": 468,
        "factory_disposition_count": 468,
        "finalized_receipt_count": 468,
        "malformed_count": 35,
        "matrix_attempt_count": 21,
        "persistent_lineage_count": 4,
        "removal_row_count": 60,
        "target_task_count": 3,
    }
    if any(population.get(name) != expected for name, expected in fixed_counts.items()):
        raise DiagnosticInvariantError("diagnostic fixed population count differs")
    for name in (
        "raw_record_ref_collision_count",
        "recall_occurrence_count",
        "target_lineage_count",
        "unique_recalled_record_count",
    ):
        if type(population.get(name)) is not int or int(population[name]) < 0:
            raise DiagnosticInvariantError("diagnostic observed population count differs")
    if (
        not 1 <= int(population["target_lineage_count"]) <= 2
        or not 3 <= int(population["recall_occurrence_count"]) <= 36
        or not 1
        <= int(population["unique_recalled_record_count"])
        <= int(population["recall_occurrence_count"])
    ):
        raise DiagnosticInvariantError("diagnostic observed population bounds differ")

    matrix = document.get("matrix")
    if type(matrix) is not list or len(matrix) != 3:
        raise DiagnosticInvariantError("diagnostic matrix cardinality differs")
    matrix_task_refs: list[str] = []
    matrix_task_replicates: set[tuple[str, str]] = set()
    matrix_attempts: list[AttemptTrace] = []
    matrix_stage_refs: set[str] = set()
    matrix_judgment_refs: set[str] = set()
    qwen_association_evidence: set[str] = set()
    qwen_mismatch_associated = False
    declared_to_executed = {
        "DECLARED_GRAMMAR_CONFORMANT": "CONFORMANT",
        "DECLARED_SHAPE_FOREIGN_TOKEN": "FOREIGN_TOKEN",
        "DECLARED_SHAPE_MALFORMED": "MALFORMED_SHAPE",
        "EMPTY": "EMPTY",
        "OUTSIDE_DECLARED_GRAMMAR": "OUTSIDE_DECLARED_GRAMMAR",
    }
    for raw in matrix:
        row = _require_exact_keys(
            raw,
            {
                "attempts",
                "equality_groups",
                "first_divergences",
                "replicate_ref",
                "selected_execution_class_mismatch_count",
                "task_ref",
            },
            "diagnostic matrix row",
        )
        task_ref = _require_ref(row.get("task_ref"), "matrix task")
        replicate_ref = _require_ref(row.get("replicate_ref"), "matrix replicate")
        matrix_task_refs.append(task_ref)
        matrix_task_replicates.add((task_ref, replicate_ref))
        attempts = row.get("attempts")
        if type(attempts) is not list or len(attempts) != 7:
            raise DiagnosticInvariantError("matrix attempt count differs")
        task_attempts: list[AttemptTrace] = []
        for arm, raw_attempt in zip(EXPECTED_ARMS, attempts, strict=True):
            trace = _validate_closed_attempt(raw_attempt, arm=arm, task_id=task_ref)
            matrix_attempts.append(trace)
            task_attempts.append(trace)
            if trace.stage_ref in matrix_stage_refs or trace.judgment_ref in matrix_judgment_refs:
                raise DiagnosticInvariantError("matrix evidence reference is duplicated")
            matrix_stage_refs.add(trace.stage_ref)
            matrix_judgment_refs.add(trace.judgment_ref)
            if arm == "QWEN_ONLY":
                qwen_association_evidence.update({trace.stage_ref, trace.judgment_ref})
                qwen_mismatch_associated = qwen_mismatch_associated or (
                    trace.objective_success
                    and declared_to_executed.get(trace.selected_class)
                    != trace.response_class
                )
        equality = _require_exact_keys(
            row.get("equality_groups"),
            {
                "objective_score_equality",
                "proposal_equality",
                "proposal_request_equality",
                "recall_preimage_equality",
                "response_equality",
                "selected_trace_equality",
            },
            "matrix equality groups",
        )
        for partition in equality.values():
            _validate_partition(partition)
        recomputed_objective_partition = [
            list(group)
            for group in equality_groups(
                {trace.arm: trace.objective_success for trace in task_attempts}
            )
        ]
        if equality.get("objective_score_equality") != recomputed_objective_partition:
            raise DiagnosticInvariantError("matrix objective-score partition differs")
        divergences = row.get("first_divergences")
        allowed_fields = {"NONE", *(name.upper() for name in TRACE_DIVERGENCE_ORDER)}
        if type(divergences) is not list or len(divergences) != 7:
            raise DiagnosticInvariantError("matrix divergence cardinality differs")
        for arm, raw_divergence in zip(EXPECTED_ARMS, divergences, strict=True):
            divergence = _require_exact_keys(
                raw_divergence, {"arm", "field"}, "matrix divergence"
            )
            if divergence.get("arm") != arm or divergence.get("field") not in allowed_fields:
                raise DiagnosticInvariantError("matrix divergence value differs")
        mismatch_count = row.get("selected_execution_class_mismatch_count")
        recomputed_mismatch_count = sum(
            trace.parser_disposition == "ADMITTED"
            and declared_to_executed.get(trace.selected_class) != trace.response_class
            for trace in task_attempts
        )
        if (
            type(mismatch_count) is not int
            or mismatch_count != recomputed_mismatch_count
        ):
            raise DiagnosticInvariantError("matrix mismatch count differs")
    if matrix_task_refs != sorted(matrix_task_refs) or len(set(matrix_task_refs)) != 3:
        raise DiagnosticInvariantError("matrix task order/uniqueness differs")
    validate_unique_cohort(matrix_attempts)

    provenance = _require_exact_keys(
        document.get("provenance"),
        {
            "adaptation_episode_joined",
            "all_rejoined",
            "frozen_batch_recomputed",
            "join_method",
            "prospective_contract_recomputed",
            "prospective_contract_recomputation_disposition",
            "lineages",
            "occurrence_count",
            "occurrences",
            "raw_record_ref_collision_count",
            "unique_record_count",
        },
        "diagnostic provenance",
    )
    if (
        provenance.get("adaptation_episode_joined") is not True
        or provenance.get("all_rejoined") is not True
        or provenance.get("frozen_batch_recomputed") is not False
        or provenance.get("join_method")
        != "PERSISTED_REF_HASH_CONTENT_TO_RETAINED_RECORD"
        or provenance.get("prospective_contract_recomputed") is not False
        or provenance.get("prospective_contract_recomputation_disposition")
        != "INDETERMINATE"
    ):
        raise DiagnosticInvariantError("diagnostic provenance claim differs")
    occurrences = provenance.get("occurrences")
    if type(occurrences) is not list or len(occurrences) != provenance.get(
        "occurrence_count"
    ):
        raise DiagnosticInvariantError("diagnostic occurrence count differs")
    composite_keys: set[tuple[object, ...]] = set()
    unique_records: set[tuple[object, object]] = set()
    scopes_by_record: dict[object, set[object]] = {}
    recall_indices_by_task: dict[str, list[int]] = {}
    required_q5_provenance_refs: set[str] = set()
    for raw in occurrences:
        row = _require_exact_keys(
            raw,
            {
                "acquisition_ref",
                "ancestry_refs",
                "recall_index",
                "record_bytes_ref",
                "record_ref",
                "source_contract",
                "source_ref",
                "target_full_replicate_ref",
                "target_lineage_scope_ref",
                "target_task_id",
            },
            "diagnostic occurrence",
        )
        for ref_name in (
            "acquisition_ref",
            "record_bytes_ref",
            "record_ref",
            "source_ref",
            "target_full_replicate_ref",
            "target_lineage_scope_ref",
            "target_task_id",
        ):
            _require_ref(row.get(ref_name), f"occurrence {ref_name}")
        if row.get("source_contract") not in {
            _LEGACY_EPISODE_CONTRACT,
            _PROSPECTIVE_BATCH_CONTRACT,
            _PROSPECTIVE_RESOLUTION_CONTRACT,
        }:
            raise DiagnosticInvariantError("occurrence source contract differs")
        ancestry = row.get("ancestry_refs")
        if type(ancestry) is not list or not ancestry or ancestry != sorted(set(ancestry)):
            raise DiagnosticInvariantError("occurrence ancestry differs")
        for item in ancestry:
            _require_ref(item, "occurrence ancestry")
        if (
            row.get("record_bytes_ref") != row.get("record_ref")
            or row.get("acquisition_ref") not in ancestry
            or row.get("source_ref") not in ancestry
        ):
            raise DiagnosticInvariantError("occurrence retained-record binding differs")
        recall_index = row.get("recall_index")
        if type(recall_index) is not int or not 0 <= recall_index < 12:
            raise DiagnosticInvariantError("occurrence recall index differs")
        composite = (
            row.get("target_task_id"),
            row.get("target_full_replicate_ref"),
            row.get("target_lineage_scope_ref"),
            recall_index,
            row.get("record_ref"),
        )
        if composite in composite_keys:
            raise DiagnosticInvariantError("occurrence composite identity is duplicated")
        composite_keys.add(composite)
        recall_indices_by_task.setdefault(str(row["target_task_id"]), []).append(
            recall_index
        )
        required_q5_provenance_refs.update(
            {
                str(row["acquisition_ref"]),
                str(row["record_ref"]),
                str(row["source_ref"]),
            }
        )
        identity = (row.get("target_lineage_scope_ref"), row.get("record_ref"))
        unique_records.add(identity)
        scopes_by_record.setdefault(row.get("record_ref"), set()).add(
            row.get("target_lineage_scope_ref")
        )
    collision_count = sum(len(scopes) > 1 for scopes in scopes_by_record.values())
    occurrence_task_ids = {str(row["target_task_id"]) for row in occurrences}
    if (
        provenance.get("unique_record_count") != len(unique_records)
        or provenance.get("raw_record_ref_collision_count") != collision_count
        or occurrence_task_ids != set(matrix_task_refs)
        or any(
            sorted(indices) != list(range(len(indices)))
            for indices in recall_indices_by_task.values()
        )
        or any(
            (
                str(row["target_task_id"]),
                str(row["target_full_replicate_ref"]),
            )
            not in matrix_task_replicates
            for row in occurrences
        )
    ):
        raise DiagnosticInvariantError("provenance aggregate counts differ")
    lineage_rows = provenance.get("lineages")
    if type(lineage_rows) is not list or not 1 <= len(lineage_rows) <= 2:
        raise DiagnosticInvariantError("provenance lineage count differs")
    lineage_identities: set[tuple[str, str]] = set()
    for raw in lineage_rows:
        row = _require_exact_keys(
            raw,
            {
                "adaptation_attempt_count",
                "admitted_transition_count",
                "joined_episode_refs",
                "lineage_scope_ref",
                "malformed_transition_count",
                "replicate_ref",
            },
            "provenance lineage",
        )
        if (
            row.get("adaptation_attempt_count") != 12
            or type(row.get("admitted_transition_count")) is not int
            or type(row.get("malformed_transition_count")) is not int
            or row.get("admitted_transition_count")
            + row.get("malformed_transition_count")
            != 12
        ):
            raise DiagnosticInvariantError("provenance adaptation count differs")
        _require_ref(row.get("lineage_scope_ref"), "provenance lineage scope")
        _require_ref(row.get("replicate_ref"), "provenance replicate")
        lineage_identity = (
            str(row.get("replicate_ref")),
            str(row.get("lineage_scope_ref")),
        )
        if lineage_identity in lineage_identities:
            raise DiagnosticInvariantError("provenance lineage identity is duplicated")
        lineage_identities.add(lineage_identity)
        joined = row.get("joined_episode_refs")
        if (
            type(joined) is not list
            or len(joined) != row.get("admitted_transition_count")
            or len(set(joined)) != len(joined)
        ):
            raise DiagnosticInvariantError("joined adaptation episode count differs")
        for item in joined:
            _require_ref(item, "joined adaptation episode")
            required_q5_provenance_refs.add(item)
    if any(
        (
            str(row["target_full_replicate_ref"]),
            str(row["target_lineage_scope_ref"]),
        )
        not in lineage_identities
        for row in occurrences
    ):
        raise DiagnosticInvariantError("occurrence does not belong to target lineage")
    if (
        population.get("recall_occurrence_count") != len(occurrences)
        or population.get("unique_recalled_record_count") != len(unique_records)
        or population.get("raw_record_ref_collision_count") != collision_count
        or population.get("target_lineage_count") != len(lineage_rows)
    ):
        raise DiagnosticInvariantError("population/provenance counts differ")

    questions = document.get("questions")
    if type(questions) is not list or len(questions) != 6:
        raise DiagnosticInvariantError("diagnostic question count differs")
    expected_dispositions = (
        "ESTABLISHED",
        "ESTABLISHED",
        "RULED_OUT",
        "CONSISTENT_WITH_EVIDENCE",
        "ESTABLISHED" if qwen_mismatch_associated else "RULED_OUT",
        "INDETERMINATE",
    )
    for index, raw in enumerate(questions, start=1):
        row = _require_exact_keys(
            raw, {"disposition", "evidence_refs", "question"}, "diagnostic question"
        )
        if (
            row.get("question") != f"Q{index}"
            or row.get("disposition") != expected_dispositions[index - 1]
        ):
            raise DiagnosticInvariantError("diagnostic question disposition differs")
        evidence_refs = row.get("evidence_refs")
        if type(evidence_refs) is not list or not evidence_refs or evidence_refs != sorted(
            set(evidence_refs)
        ):
            raise DiagnosticInvariantError("diagnostic question evidence differs")
        for item in evidence_refs:
            _require_ref(item, "question evidence")
        if index == 5 and not (
            qwen_association_evidence | required_q5_provenance_refs
        ).issubset(set(evidence_refs)):
            raise DiagnosticInvariantError("Q5 association/provenance evidence is absent")

    resources = _require_exact_keys(
        document.get("resources"),
        {
            "cognitive_record_count",
            "input_bytes",
            "no_cognee",
            "no_gpu",
            "no_model",
            "no_network",
            "no_subprocess",
            "output_size_bytes",
            "peak_rss_bytes",
            "process_count",
            "wall_milliseconds",
        },
        "diagnostic resources",
    )
    for name in (
        "cognitive_record_count",
        "input_bytes",
        "output_size_bytes",
        "peak_rss_bytes",
        "process_count",
        "wall_milliseconds",
    ):
        if type(resources.get(name)) is not int or int(resources[name]) < 0:
            raise DiagnosticInvariantError("diagnostic resource count differs")
    if (
        resources.get("process_count") != 1
        or resources.get("input_bytes")
        != sum(
            int(witness["size_bytes"])
            for witness in input_witnesses["before"].values()
        )
        or not 0 < resources.get("output_size_bytes", 0) <= MAX_OUTPUT_BYTES
        or not 0 <= resources.get("wall_milliseconds", MAX_WALL_SECONDS * 1000)
        < MAX_WALL_SECONDS * 1000
        or resources.get("peak_rss_bytes", MAX_RSS_BYTES + 1) > MAX_RSS_BYTES
        or any(
            resources.get(name) is not True
            for name in ("no_cognee", "no_gpu", "no_model", "no_network", "no_subprocess")
        )
    ):
        raise DiagnosticInvariantError("diagnostic resource boundary differs")


def _question_rows(
    target_ids: Sequence[str],
    *,
    ledger: _LedgerDataset,
    removals: _RemovalDataset,
    provenance: Mapping[str, object],
    result: Mapping[str, object],
) -> list[dict[str, object]]:
    target_rows = [
        ledger.rows[(task_id, arm)]
        for task_id in target_ids
        for arm in EXPECTED_ARMS
    ]
    admitted_stage_refs = sorted(
        row.stage_ref
        for row in target_rows
        if row.stage.get("parser_disposition") == "ADMITTED"
    )
    removal_refs = sorted(removals.records[task_id]["evidence_ref"] for task_id in target_ids)
    provenance_refs: set[str] = set()
    for raw in provenance.get("occurrences", []):
        occurrence = _mapping(raw, "question provenance occurrence")
        provenance_refs.update(
            {
                str(occurrence["acquisition_ref"]),
                str(occurrence["record_ref"]),
                str(occurrence["source_ref"]),
            }
        )
    for raw in provenance.get("lineages", []):
        lineage = _mapping(raw, "question provenance lineage")
        provenance_refs.update(str(item) for item in lineage["joined_episode_refs"])
    qwen_rows = [ledger.rows[(task_id, "QWEN_ONLY")] for task_id in target_ids]
    qwen_traces = [
        extract_attempt_trace(
            row.stage_bytes,
            row.judgment_bytes,
            stage_ref=row.stage_ref,
            judgment_ref=row.judgment_ref,
        )
        for row in qwen_rows
    ]
    declared_to_executed = {
        "DECLARED_GRAMMAR_CONFORMANT": "CONFORMANT",
        "DECLARED_SHAPE_FOREIGN_TOKEN": "FOREIGN_TOKEN",
        "DECLARED_SHAPE_MALFORMED": "MALFORMED_SHAPE",
        "EMPTY": "EMPTY",
        "OUTSIDE_DECLARED_GRAMMAR": "OUTSIDE_DECLARED_GRAMMAR",
    }
    mismatch_associated = any(
        trace.objective_success
        and declared_to_executed.get(trace.selected_class) != trace.response_class
        for trace in qwen_traces
    )
    provenance_refs.update(
        ref for row in qwen_rows for ref in (row.stage_ref, row.judgment_ref)
    )
    return [
        {
            "disposition": "ESTABLISHED",
            "evidence_refs": sorted(
                {
                    _require_ref(result.get("run_integrity_ref"), "run integrity"),
                    *(ledger.rows[(task_id, "QWEN_ONLY")].judgment_ref for task_id in target_ids),
                }
            ),
            "question": "Q1",
        },
        {
            "disposition": "ESTABLISHED",
            "evidence_refs": sorted(
                {ref for row in target_rows for ref in (row.stage_ref, row.judgment_ref)}
            ),
            "question": "Q2",
        },
        {
            "disposition": "RULED_OUT",
            "evidence_refs": admitted_stage_refs,
            "question": "Q3",
        },
        {
            "disposition": "CONSISTENT_WITH_EVIDENCE",
            "evidence_refs": removal_refs,
            "question": "Q4",
        },
        {
            "disposition": "ESTABLISHED" if mismatch_associated else "RULED_OUT",
            "evidence_refs": sorted(provenance_refs),
            "question": "Q5",
        },
        {
            "disposition": "INDETERMINATE",
            "evidence_refs": sorted(
                {
                    _require_ref(result.get("classifier_ref"), "classifier"),
                    _require_ref(result.get("run_integrity_ref"), "run integrity"),
                }
            ),
            "question": "Q6",
        },
    ]


def _stabilize_output_size(document: dict[str, object]) -> None:
    resources = _mapping(document.get("resources"), "diagnostic resources")
    for _ in range(16):
        observed = len(canonical_json_bytes(document))
        if resources.get("output_size_bytes") == observed:
            return
        resources["output_size_bytes"] = observed
    raise DiagnosticInvariantError("diagnostic output size did not reach a fixed point")


def _compose_diagnostic(budget: DiagnosticResourceBudget) -> FileWitness:
    """Read only the frozen R2 inputs and publish the sole closed result."""

    if type(budget) is not DiagnosticResourceBudget:
        raise TypeError("diagnostic composition requires its exact resource budget")
    try:
        os.lstat(OUTPUT_PATH)
    except FileNotFoundError:
        pass
    else:
        raise DiagnosticInvariantError("create-once output already exists")
    specs = _spec_map(FROZEN_INPUTS)
    before = {
        label: capture_frozen_input(specs[label]) for label in sorted(specs)
    }
    budget.checkpoint()
    manifest, _ = _read_frozen_json(specs["R2_SOURCE_MANIFEST"])
    result, _ = _read_frozen_json(specs["R2_RESULT"])
    audit, _ = _read_frozen_json(specs["R2_ARM_FACTORY_AUDIT"])
    raw_witness = RawMaterialWitness()
    raw_witness.observe_document(manifest, category="MANIFEST")
    raw_witness.observe_document(result, category="RESULT")
    raw_witness.observe_document(audit, category="FACTORY_AUDIT")
    tasks = _manifest_task_map(manifest)
    result_bindings, result_receipts, result_metrics = _result_attempt_bindings(
        result,
        manifest_sha256=specs["R2_SOURCE_MANIFEST"].sha256,
        tasks=tasks,
    )
    ledger = _stream_ledger(
        specs["R2_LEDGER"],
        tasks=tasks,
        result_bindings=result_bindings,
        result_receipts=result_receipts,
        result_metrics=result_metrics,
        raw_witness=raw_witness,
        budget=budget,
    )
    removals = _validate_removal_dataset(result, tasks=tasks, ledger=ledger)
    factory = _validate_factory_audit(
        audit,
        ledger_spec=specs["R2_LEDGER"],
        ledger_witness=before["R2_LEDGER"],
        ledger=ledger,
        tasks=tasks,
        specs=specs,
    )
    cognitive = {
        key: _scan_cognitive_lineage(
            binding,
            raw_witness=raw_witness,
            budget=budget,
        )
        for key, binding in sorted(factory.lineages.items())
    }
    target_ids = _target_task_ids(ledger)
    matrix, _traces = _build_trace_matrix(target_ids, ledger=ledger)
    provenance = _rejoin_target_provenance(
        target_ids,
        tasks=tasks,
        ledger=ledger,
        removals=removals,
        factory=factory,
        cognitive=cognitive,
    )
    admitted_count = sum(
        row.stage.get("parser_disposition") == "ADMITTED" for row in ledger.rows.values()
    )
    malformed_count = sum(
        row.stage.get("parser_disposition") == "MALFORMED" for row in ledger.rows.values()
    )
    if (admitted_count, malformed_count) != (433, 35):
        raise DiagnosticInvariantError("full-population parser counts differ")
    cognitive_record_count = sum(len(lineage.records) for lineage in cognitive.values())
    raw_witness.assert_complete(
        expected_attempts=MAX_ATTEMPTS,
        expected_cognitive_records=cognitive_record_count,
        required_categories={
            "ATTEMPT_JUDGMENT": MAX_ATTEMPTS,
            "ATTEMPT_STAGE": MAX_ATTEMPTS,
            "COGNITIVE_RECORD": cognitive_record_count,
            "FACTORY_AUDIT": 1,
            "MANIFEST": 1,
            "RESULT": 1,
        },
    )
    after = {
        label: verify_frozen_input_unchanged(specs[label], before[label])
        for label in sorted(specs)
    }
    resource_values = budget.checkpoint()
    document: dict[str, object] = {
        "identity": DIAGNOSTIC_IDENTITY,
        "inputs": {
            "after": [after[label].redacted() for label in sorted(after)],
            "before": [before[label].redacted() for label in sorted(before)],
            "unchanged": True,
        },
        "matrix": matrix,
        "population": {
            "admitted_count": admitted_count,
            "arms_per_task": len(EXPECTED_ARMS),
            "attempt_count": len(ledger.rows),
            "factory_disposition_count": len(factory.disposition_refs),
            "finalized_receipt_count": len(ledger.receipt_refs),
            "malformed_count": malformed_count,
            "matrix_attempt_count": len(matrix) * len(EXPECTED_ARMS),
            "persistent_lineage_count": len(factory.lineages),
            "raw_record_ref_collision_count": provenance[
                "raw_record_ref_collision_count"
            ],
            "recall_occurrence_count": provenance["occurrence_count"],
            "removal_row_count": len(removals.records),
            "target_lineage_count": len(provenance["lineages"]),
            "target_task_count": len(target_ids),
            "unique_recalled_record_count": provenance["unique_record_count"],
        },
        "predecessor": {
            "arm_factory_audit_ref": "sha256:" + specs["R2_ARM_FACTORY_AUDIT"].sha256,
            "classification": "NOT_SUPPORTED",
            "diagnostic_leaf_ref": "sha256:" + specs["DIAGNOSTIC_LEAF"].sha256,
            "ledger_ref": "sha256:" + specs["R2_LEDGER"].sha256,
            "manifest_ref": "sha256:" + specs["R2_SOURCE_MANIFEST"].sha256,
            "predecessor_leaf_ref": "sha256:" + specs["R2_PREDECESSOR_LEAF"].sha256,
            "result_ref": "sha256:" + specs["R2_RESULT"].sha256,
            "result_report_ref": "sha256:" + specs["R2_RESULT_REPORT"].sha256,
            "runner_ref": "sha256:" + specs["R2_RUNNER"].sha256,
        },
        "provenance": provenance,
        "questions": _question_rows(
            target_ids,
            ledger=ledger,
            removals=removals,
            provenance=provenance,
            result=result,
        ),
        "resources": {
            "cognitive_record_count": cognitive_record_count,
            "input_bytes": sum(witness.size_bytes for witness in before.values()),
            "no_cognee": True,
            "no_gpu": True,
            "no_model": True,
            "no_network": True,
            "no_subprocess": True,
            "output_size_bytes": 0,
            "peak_rss_bytes": resource_values["peak_rss_bytes"],
            "process_count": 1,
            "wall_milliseconds": resource_values["elapsed_milliseconds"],
        },
        "schema": DIAGNOSTIC_SCHEMA,
    }
    _stabilize_output_size(document)
    validate_redacted_output(
        document,
        forbidden_values=raw_witness.forbidden_values(),
    )
    budget.checkpoint()
    for label in sorted(specs):
        verify_frozen_input_unchanged(specs[label], before[label])
    return publish_create_once(
        OUTPUT_PATH,
        document,
        expected_path=OUTPUT_PATH,
        maximum_bytes=MAX_OUTPUT_BYTES,
        forbidden_values=raw_witness.forbidden_values(),
    )


def run_live_diagnostic() -> FileWitness:
    """Execute the audited no-path live composition under exact resource bounds."""

    budget = DiagnosticResourceBudget()
    with _live_resource_guard(budget):
        return _compose_diagnostic(budget)


__all__ = (
    "DIAGNOSTIC_IDENTITY",
    "DIAGNOSTIC_SCHEMA",
    "EXPECTED_ARMS",
    "FROZEN_INPUTS",
    "MAX_OUTPUT_BYTES",
    "OUTPUT_PATH",
    "AttemptTrace",
    "DiagnosticInvariantError",
    "DiagnosticResourceBudget",
    "FileWitness",
    "FrozenInputSpec",
    "LiteralGrammar",
    "ProvenanceRow",
    "canonical_json_bytes",
    "capture_frozen_input",
    "classify_candidate",
    "classify_declared_grammar",
    "content_ref",
    "equality_groups",
    "extract_attempt_trace",
    "first_divergence",
    "immutable_sqlite",
    "immutable_sqlite_uri",
    "parse_canonical_json",
    "publish_create_once",
    "rejoin_provenance",
    "run_live_diagnostic",
    "validate_redacted_output",
    "validate_unique_cohort",
    "verify_frozen_input_unchanged",
)
