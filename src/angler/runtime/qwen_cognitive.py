"""Frozen-Qwen adapters for the transactional successor cognitive cycle.

This module contains representation and integrity plumbing only.  Qwen emits
public candidate procedures and the selected response; deterministic code
frames those requests, validates strict structure, compresses label-free
representations, tensorizes Moving-Origin coordinates, and journals the exact
execution receipt.  It never sees an objective label or chooses a candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import struct
from typing import ClassVar
import unicodedata

import torch

from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
)
from angler.episodes.canonical import canonical_bytes, parse_json
from angler.reasoning import SituatedFeatureSpec, encode_situated_features
from angler.runtime.cognitive_cycle import (
    CognitiveExecution,
    CognitiveExecutionRecovery,
)
from angler.runtime.durable_ability_bridge import EncodedProcedureCandidate
from angler.runtime.situated_qwen import (
    FrozenQwenGenerationRecord,
    LocalQwenIO,
    local_qwen_generation_config_ref,
)


QWEN_PROCEDURE_PROPOSAL_SCHEMA = "angler.qwen-procedure-proposal-prompt.v1"
QWEN_SELECTED_PROCEDURE_EXECUTION_SCHEMA = (
    "angler.qwen-selected-procedure-execution-prompt.v1"
)
QWEN_KNOWLEDGE_ENCODER_SCHEMA = "angler.qwen-detached-last-nonpadding.v1"
QWEN_LABEL_FREE_PROJECTION_SCHEMA = "angler.qwen-label-free-projection.v1"
QWEN_MOVING_ORIGIN_SCHEMA = "angler.qwen-moving-origin-mean.v1"
QWEN_RECEIPT_OBSERVATION_SCHEMA = "angler.qwen-receipt-observed-state.v1"
QWEN_CYCLE_MANIFEST_SCHEMA = "angler.frozen-qwen-cycle-manifest.v1"
QWEN_EXECUTION_JOURNAL_SCHEMA = "angler.qwen-execution-journal.v1"

MAX_PROPOSAL_RESPONSE_CHARS = 131_072
MAX_JOURNAL_ENTRY_BYTES = 4 * 1024 * 1024
MAX_RECALLED_ITEMS = 12
MAX_PROCEDURE_CHARS = 1_024
MAX_TASK_CHARS = 16_384
MAX_OBSERVATION_CHARS = 8_192
MAX_OBSERVATIONS = 64
MAX_QWEN_NEW_TOKENS = 256

_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROJECTION_DOMAIN = b"angler.qwen-label-free-projection.v1\0"


def _sha256_ref(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _raw_sha256(value: object, label: str) -> str:
    if type(value) is not str or _RAW_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        qualifier = f" from {minimum} through {maximum}" if maximum else ""
        raise ValueError(f"{label} must be an exact integer{qualifier}")
    return value


def _text(value: object, label: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty text of at most {maximum} characters")
    if value != value.strip():
        raise ValueError(f"{label} must not require whitespace repair")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{label} must be Unicode NFC-normalized")
    return value


def _content_ref(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenQwenCycleManifestV1:
    """All frozen identities that may influence one Qwen successor cycle."""

    model_ref: str
    tokenizer_ref: str
    genesis_config_ref: str
    prospective_config_ref: str
    learner_checkpoint_ref: str
    initial_competence_state_digest: str
    genesis_snapshot_sha256: str
    genesis_snapshot_bytes: int
    genesis_seed: int
    prospective_lesion: bool
    input_width: int = 2_560
    relation_width: int = 64
    temporal_width: int = 8
    latent_width: int = 16
    max_input_tokens: int = 4_096
    max_new_tokens: int = 64
    embedding_batch_size: int = 1

    SCHEMA: ClassVar[str] = QWEN_CYCLE_MANIFEST_SCHEMA
    PROPOSAL_SCHEMA: ClassVar[str] = QWEN_PROCEDURE_PROPOSAL_SCHEMA
    EXECUTION_SCHEMA: ClassVar[str] = QWEN_SELECTED_PROCEDURE_EXECUTION_SCHEMA
    KNOWLEDGE_SCHEMA: ClassVar[str] = QWEN_KNOWLEDGE_ENCODER_SCHEMA
    PROJECTION_SCHEMA: ClassVar[str] = QWEN_LABEL_FREE_PROJECTION_SCHEMA
    TEMPORAL_SCHEMA: ClassVar[str] = QWEN_MOVING_ORIGIN_SCHEMA
    OBSERVATION_SCHEMA: ClassVar[str] = QWEN_RECEIPT_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        for label, value in (
            ("model_ref", self.model_ref),
            ("tokenizer_ref", self.tokenizer_ref),
            ("genesis_config_ref", self.genesis_config_ref),
            ("prospective_config_ref", self.prospective_config_ref),
            ("learner_checkpoint_ref", self.learner_checkpoint_ref),
            ("initial_competence_state_digest", self.initial_competence_state_digest),
        ):
            _sha256_ref(value, label)
        _raw_sha256(self.genesis_snapshot_sha256, "genesis_snapshot_sha256")
        _integer(self.genesis_snapshot_bytes, "genesis_snapshot_bytes")
        _integer(self.genesis_seed, "genesis_seed", minimum=0)
        if type(self.prospective_lesion) is not bool:
            raise TypeError("prospective_lesion must be bool")
        _integer(self.input_width, "input_width", maximum=65_536)
        _integer(self.relation_width, "relation_width", maximum=4_096)
        _integer(self.temporal_width, "temporal_width", maximum=1_024)
        _integer(self.latent_width, "latent_width", maximum=1_024)
        _integer(self.max_input_tokens, "max_input_tokens", maximum=4_096)
        _integer(
            self.max_new_tokens,
            "max_new_tokens",
            maximum=MAX_QWEN_NEW_TOKENS,
        )
        if self.embedding_batch_size != 1:
            raise ValueError("the frozen qualification requires embedding batch size one")
        expected_temporal_width = SituatedFeatureSpec(
            landmarks=(), include_acquired_ordinal=True
        ).width
        if self.temporal_width != expected_temporal_width:
            raise ValueError(
                "temporal_width must match the frozen Moving-Origin feature layout"
            )

    @property
    def generation_config_ref(self) -> str:
        return local_qwen_generation_config_ref(
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=False,
        )

    @property
    def knowledge_encoder_ref(self) -> str:
        return _content_ref(
            {
                "consumer_dtype": "float32",
                "hidden_state_index": -1,
                "input_width": self.input_width,
                "model_ref": self.model_ref,
                "pooling": "last_nonpadding",
                "schema": self.KNOWLEDGE_SCHEMA,
                "storage_dtype": "bfloat16",
                "tokenizer_ref": self.tokenizer_ref,
            }
        )

    @property
    def projection_seed_ref(self) -> str:
        return _content_ref(
            {
                "input_width": self.input_width,
                "knowledge_encoder_ref": self.knowledge_encoder_ref,
                "relation_width": self.relation_width,
                "schema": self.PROJECTION_SCHEMA,
            }
        )

    @property
    def encoder_ref(self) -> str:
        return _content_ref(
            {
                "knowledge_encoder_ref": self.knowledge_encoder_ref,
                "latent_width": self.latent_width,
                "observation_schema": self.OBSERVATION_SCHEMA,
                "projection_seed_ref": self.projection_seed_ref,
                "relation_width": self.relation_width,
                "temporal_schema": self.TEMPORAL_SCHEMA,
                "temporal_spec": {
                    "include_acquired_ordinal": True,
                    "landmarks": [],
                },
                "temporal_width": self.temporal_width,
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "embedding_batch_size": self.embedding_batch_size,
            "encoder_ref": self.encoder_ref,
            "execution_schema": self.EXECUTION_SCHEMA,
            "generation_config_ref": self.generation_config_ref,
            "initial_competence_state_digest": self.initial_competence_state_digest,
            "genesis_config_ref": self.genesis_config_ref,
            "genesis_seed": self.genesis_seed,
            "genesis_snapshot_bytes": self.genesis_snapshot_bytes,
            "genesis_snapshot_sha256": self.genesis_snapshot_sha256,
            "input_width": self.input_width,
            "knowledge_encoder_ref": self.knowledge_encoder_ref,
            "latent_width": self.latent_width,
            "learner_checkpoint_ref": self.learner_checkpoint_ref,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "model_ref": self.model_ref,
            "observation_schema": self.OBSERVATION_SCHEMA,
            "projection_schema": self.PROJECTION_SCHEMA,
            "projection_seed_ref": self.projection_seed_ref,
            "proposal_schema": self.PROPOSAL_SCHEMA,
            "prospective_config_ref": self.prospective_config_ref,
            "prospective_lesion": self.prospective_lesion,
            "relation_width": self.relation_width,
            "schema": self.SCHEMA,
            "temporal_schema": self.TEMPORAL_SCHEMA,
            "temporal_width": self.temporal_width,
            "tokenizer_ref": self.tokenizer_ref,
        }

    @property
    def manifest_ref(self) -> str:
        return _content_ref(self.to_payload())

    def assert_io(self, io: LocalQwenIO) -> None:
        if type(io) is not LocalQwenIO:
            raise TypeError("exact Qwen components require an exact LocalQwenIO")
        io.assert_exact_identity(
            model_ref=self.model_ref,
            tokenizer_ref=self.tokenizer_ref,
            generation_config_ref=self.generation_config_ref,
        )
        if (
            io.embedding_batch_size != self.embedding_batch_size
            or io.max_input_tokens != self.max_input_tokens
            or io.max_new_tokens != self.max_new_tokens
            or io.enable_thinking is not False
        ):
            raise ValueError("LocalQwenIO resource settings differ from the manifest")


def build_qwen_procedure_proposal_prompt(
    task: str,
    recalled_evidence: tuple[str, ...],
    *,
    count: int,
) -> str:
    """Frame a task-agnostic request for public candidate procedures."""

    task = _text(task, "task", MAX_TASK_CHARS)
    _integer(count, "count", minimum=2, maximum=64)
    if type(recalled_evidence) is not tuple or not recalled_evidence:
        raise ValueError("recalled_evidence must be one nonempty immutable tuple")
    if len(recalled_evidence) > MAX_RECALLED_ITEMS:
        raise ValueError("recalled_evidence exceeds the fixed recall ceiling")
    evidence = tuple(
        _text(value, "recalled evidence", MAX_TASK_CHARS)
        for value in recalled_evidence
    )
    sections = [
        f"Schema: {QWEN_PROCEDURE_PROPOSAL_SCHEMA}",
        "Propose general procedures for the current task. Do not answer the task.",
        (
            "Return exactly one JSON object with exactly one key named "
            f"procedures whose value is a list of exactly {count} distinct "
            "nonempty procedure strings. Return JSON only."
        ),
        "Current task:\n" + task,
        "Ordered recalled public evidence:",
    ]
    sections.extend(f"[{index}] {value}" for index, value in enumerate(evidence))
    return "\n\n".join(sections)


def parse_qwen_procedure_proposals(
    response: str,
    *,
    count: int,
) -> tuple[str, ...]:
    """Accept one exact proposal object; never repair model output."""

    _integer(count, "count", minimum=2, maximum=64)
    if type(response) is not str or not response or len(response) > MAX_PROPOSAL_RESPONSE_CHARS:
        raise ValueError("proposal response must be bounded nonempty text")
    parsed = parse_json(response)
    if type(parsed) is not dict or set(parsed) != {"procedures"}:
        raise ValueError("proposal response must contain only the procedures field")
    values = parsed["procedures"]
    if type(values) is not list or len(values) != count:
        raise ValueError("proposal response has the wrong candidate cardinality")
    proposals = tuple(
        _text(value, "procedure", MAX_PROCEDURE_CHARS) for value in values
    )
    if len(set(proposals)) != len(proposals):
        raise ValueError("proposal response contains duplicate procedures")
    return proposals


def _execution_prompt_parts(
    task: str,
    selected_trace: str,
    observations: tuple[str, ...],
) -> str:
    task = _text(task, "task", MAX_TASK_CHARS)
    selected_trace = _text(selected_trace, "selected_trace", MAX_PROCEDURE_CHARS)
    if type(observations) is not tuple or len(observations) > MAX_OBSERVATIONS:
        raise ValueError("observations must be a bounded immutable tuple")
    bounded = tuple(
        _text(value, "observation", MAX_OBSERVATION_CHARS)
        for value in observations
    )
    sections = [
        f"Schema: {QWEN_SELECTED_PROCEDURE_EXECUTION_SCHEMA}",
        (
            "Execute the externally selected procedure for the current task. "
            "Do not choose or invent an alternative procedure. Return only the "
            "public task response."
        ),
        "Current task:\n" + task,
        "Selected procedure:\n" + selected_trace,
        "Ordered public observations:",
    ]
    if bounded:
        sections.extend(f"[{index}] {value}" for index, value in enumerate(bounded))
    else:
        sections.append("(none)")
    return "\n\n".join(sections)


def build_qwen_execution_prompt(request: CognitiveExecutionRequest) -> str:
    if type(request) is not CognitiveExecutionRequest:
        raise TypeError("request must be an exact CognitiveExecutionRequest")
    return _execution_prompt_parts(
        request.request,
        request.selected_trace,
        request.input_observations,
    )


class FrozenQwenProcedureAdapterV1:
    """Strict frozen-Qwen public procedure proposal boundary."""

    def __init__(
        self,
        io: LocalQwenIO,
        manifest: FrozenQwenCycleManifestV1,
    ) -> None:
        if type(manifest) is not FrozenQwenCycleManifestV1:
            raise TypeError("manifest must be an exact FrozenQwenCycleManifestV1")
        manifest.assert_io(io)
        self.io = io
        self.manifest = manifest
        self._last_proposal_generation: FrozenQwenGenerationRecord | None = None

    @property
    def model_ref(self) -> str:
        return self.manifest.model_ref

    @property
    def manifest_ref(self) -> str:
        return self.manifest.manifest_ref

    @property
    def last_proposal_generation(self) -> FrozenQwenGenerationRecord | None:
        return self._last_proposal_generation

    def _assert_generation(
        self,
        generation: FrozenQwenGenerationRecord,
        *,
        prompt: str,
    ) -> None:
        if (
            generation.model_ref != self.manifest.model_ref
            or generation.tokenizer_ref != self.manifest.tokenizer_ref
            or generation.generation_config_ref
            != self.manifest.generation_config_ref
            or generation.prompt_ref != _content_ref({"prompt": prompt})
            or not 1 <= generation.prompt_tokens <= self.manifest.max_input_tokens
            or not 1
            <= len(generation.generated_token_ids)
            <= self.manifest.max_new_tokens
        ):
            raise ValueError("Qwen generation record differs from the frozen manifest")

    def propose_procedure_traces(
        self,
        task: str,
        recalled_evidence: tuple[str, ...],
        *,
        count: int = 4,
    ) -> tuple[str, ...]:
        self.manifest.assert_io(self.io)
        prompt = build_qwen_procedure_proposal_prompt(
            task,
            recalled_evidence,
            count=count,
        )
        generation = self.io.generate_record(prompt)
        self._assert_generation(generation, prompt=prompt)
        self._last_proposal_generation = generation
        proposals = parse_qwen_procedure_proposals(
            generation.response,
            count=count,
        )
        return proposals

    def execute_with_procedure(
        self,
        task: str,
        selected_trace: str,
        observations: tuple[str, ...],
    ) -> str:
        self.manifest.assert_io(self.io)
        prompt = _execution_prompt_parts(task, selected_trace, observations)
        generation = self.io.generate_record(prompt)
        self._assert_generation(generation, prompt=prompt)
        if not generation.response.strip():
            raise RuntimeError("Qwen returned an empty response")
        return generation.response


def build_label_free_projection(
    manifest: FrozenQwenCycleManifestV1,
) -> torch.Tensor:
    """Build the exact manifest-seeded signed random compression matrix."""

    if type(manifest) is not FrozenQwenCycleManifestV1:
        raise TypeError("manifest must be an exact FrozenQwenCycleManifestV1")
    seed = bytes.fromhex(manifest.projection_seed_ref.removeprefix("sha256:"))
    scale = 1.0 / math.sqrt(float(manifest.input_width))
    values: list[float] = []
    for output_index in range(manifest.relation_width):
        for input_index in range(manifest.input_width):
            digest = hashlib.sha256(
                _PROJECTION_DOMAIN
                + seed
                + struct.pack(">II", output_index, input_index)
            ).digest()
            values.append(scale if digest[0] & 1 else -scale)
    matrix = torch.tensor(values, dtype=torch.float32).reshape(
        manifest.relation_width,
        manifest.input_width,
    )
    if matrix.requires_grad or not bool(torch.isfinite(matrix).all().item()):
        raise RuntimeError("label-free projection is not a finite frozen tensor")
    return matrix


class FrozenQwenProcedureRelationAdapterV1:
    """Map detached Qwen rows and Moving-Origin coordinates to candidates."""

    def __init__(
        self,
        io: LocalQwenIO,
        manifest: FrozenQwenCycleManifestV1,
    ) -> None:
        if type(manifest) is not FrozenQwenCycleManifestV1:
            raise TypeError("manifest must be an exact FrozenQwenCycleManifestV1")
        manifest.assert_io(io)
        self.io = io
        self.manifest = manifest
        self._projection = build_label_free_projection(manifest)
        self._feature_spec = SituatedFeatureSpec(
            landmarks=(), include_acquired_ordinal=True
        )

    @property
    def model_ref(self) -> str:
        return self.manifest.model_ref

    @property
    def encoder_ref(self) -> str:
        return self.manifest.encoder_ref

    @property
    def manifest_ref(self) -> str:
        return self.manifest.manifest_ref

    @staticmethod
    def _recall_items(recall: object) -> tuple[object, ...]:
        items = getattr(recall, "items", None)
        if type(items) is not tuple or not 1 <= len(items) <= MAX_RECALLED_ITEMS:
            raise ValueError("recall must contain one bounded immutable item tuple")
        return items

    def encode_candidates(
        self,
        task: str,
        proposals: tuple[str, ...],
        recall: object,
    ) -> tuple[EncodedProcedureCandidate, ...]:
        self.manifest.assert_io(self.io)
        task = _text(task, "task", MAX_TASK_CHARS)
        if type(proposals) is not tuple or not 2 <= len(proposals) <= 64:
            raise ValueError("proposals must contain 2 through 64 exact candidates")
        validated_proposals = tuple(
            _text(value, "procedure", MAX_PROCEDURE_CHARS) for value in proposals
        )
        if len(set(validated_proposals)) != len(validated_proposals):
            raise ValueError("proposals must be distinct")
        items = self._recall_items(recall)
        texts = tuple(
            _text(getattr(item, "text", None), "recalled text", MAX_TASK_CHARS)
            for item in items
        )
        refs = tuple(
            _sha256_ref(getattr(item, "artifact_ref", None), "artifact_ref")
            for item in items
        )
        supporting_refs = tuple(sorted(refs))
        if len(set(supporting_refs)) != len(supporting_refs):
            raise ValueError("recalled evidence references must be unique")

        origins = set()
        for item in items:
            acquired = _integer(
                getattr(item, "acquired_ordinal", None),
                "acquired_ordinal",
                minimum=0,
            )
            age = _integer(getattr(item, "age", None), "age", minimum=0)
            origins.add(acquired + age)
        # Normal Moving-Origin positions all imply the same current ordinal.
        # The frozen-origin removal deliberately sets every age to zero, so it
        # need not.  The maximum represented coordinate supplies only the
        # scale horizon; each exact age/ordinal field remains untouched.
        origin = max(origins)
        temporal, mask = encode_situated_features(
            (recall,),  # AcquisitionRecallBatch is intentionally field-compatible.
            now=(origin,),
            spec=self._feature_spec,
            device="cpu",
            dtype=torch.float32,
        )
        if (
            temporal.shape != (1, len(items), self.manifest.temporal_width)
            or mask.shape != (1, len(items))
            or not bool(mask.all().item())
            or not bool(torch.isfinite(temporal).all().item())
        ):
            raise RuntimeError("Moving-Origin tensorization returned incompatible rows")
        temporal_mean = temporal[0].mean(dim=0).detach()

        segments = (task, *validated_proposals, *texts)
        represented = self.io.embed(segments)
        expected_rows = 1 + len(validated_proposals) + len(items)
        if represented.shape != (expected_rows, self.manifest.input_width):
            raise ValueError("Qwen representations do not match the frozen manifest")
        if represented.requires_grad or represented.grad_fn is not None:
            raise RuntimeError("candidate representations unexpectedly retain gradients")
        task_row = represented[0]
        proposal_rows = represented[1 : 1 + len(validated_proposals)]
        evidence_mean = represented[1 + len(validated_proposals) :].mean(dim=0)
        combined = (
            task_row.unsqueeze(0)
            + proposal_rows
            + evidence_mean.unsqueeze(0)
        ) / 3.0
        relations = combined @ self._projection.t()
        if (
            relations.shape
            != (len(validated_proposals), self.manifest.relation_width)
            or not bool(torch.isfinite(relations).all().item())
        ):
            raise RuntimeError("label-free relation compression returned invalid rows")
        return tuple(
            EncodedProcedureCandidate(
                trace=trace,
                relation_features=relations[index].detach().clone(),
                temporal_features=temporal_mean.clone(),
                base_logit=torch.zeros((), dtype=torch.float32),
                supporting_evidence_refs=supporting_refs,
            )
            for index, trace in enumerate(validated_proposals)
        )


@dataclass(frozen=True, slots=True)
class QwenExecutionJournalEntry:
    """One exact Qwen generation and receipt durable before executor return."""

    manifest_ref: str
    request: CognitiveExecutionRequest
    generation: FrozenQwenGenerationRecord
    receipt: CognitiveExecutionReceipt

    SCHEMA: ClassVar[str] = QWEN_EXECUTION_JOURNAL_SCHEMA

    def __post_init__(self) -> None:
        _sha256_ref(self.manifest_ref, "manifest_ref")
        if type(self.request) is not CognitiveExecutionRequest:
            raise TypeError("request must be an exact CognitiveExecutionRequest")
        if type(self.generation) is not FrozenQwenGenerationRecord:
            raise TypeError("generation must be an exact FrozenQwenGenerationRecord")
        if type(self.receipt) is not CognitiveExecutionReceipt:
            raise TypeError("receipt must be an exact CognitiveExecutionReceipt")
        self.receipt.assert_request(self.request)
        if self.receipt.status != "COMPLETED":
            raise ValueError("Qwen journal records only completed generations")
        if self.receipt.response != self.generation.response:
            raise ValueError("journaled generation and receipt responses differ")
        if self.receipt.output_observations:
            raise ValueError("Qwen executor does not invent output observations")
        if len(self.canonical_bytes()) > MAX_JOURNAL_ENTRY_BYTES:
            raise ValueError("Qwen journal entry exceeds the fixed byte ceiling")

    @property
    def idempotency_key(self) -> str:
        return self.request.idempotency_key

    @property
    def execution_request_ref(self) -> str:
        return self.request.execution_request_ref

    @property
    def execution_receipt_ref(self) -> str:
        return self.receipt.execution_receipt_ref

    def to_payload(self) -> dict[str, object]:
        return {
            "generation": self.generation.to_payload(),
            "manifest_ref": self.manifest_ref,
            "receipt": self.receipt.to_payload(),
            "request": self.request.to_payload(),
            "schema": self.SCHEMA,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Self:
        if type(payload) is not dict or set(payload) != {
            "generation",
            "manifest_ref",
            "receipt",
            "request",
            "schema",
        }:
            raise ValueError("Qwen journal payload fields are not exact")
        if payload["schema"] != cls.SCHEMA:
            raise ValueError("Qwen journal payload schema is unsupported")
        for label in ("generation", "receipt", "request"):
            if type(payload[label]) is not dict:
                raise TypeError(f"journal {label} payload must be an object")
        return cls(
            manifest_ref=payload["manifest_ref"],  # type: ignore[arg-type]
            request=CognitiveExecutionRequest.from_payload(  # type: ignore[arg-type]
                payload["request"]
            ),
            generation=FrozenQwenGenerationRecord.from_payload(  # type: ignore[arg-type]
                payload["generation"]
            ),
            receipt=CognitiveExecutionReceipt.from_payload(  # type: ignore[arg-type]
                payload["receipt"]
            ),
        )


class SQLiteQwenExecutionJournal:
    """Small durable exactly-once receipt journal for local frozen inference."""

    SCHEMA_VERSION = 1
    _TABLE_NAME = "qwen_execution_journal"
    _CREATE_TABLE_SQL = """
        CREATE TABLE qwen_execution_journal (
            idempotency_key TEXT PRIMARY KEY NOT NULL,
            manifest_ref TEXT NOT NULL,
            execution_request_ref TEXT UNIQUE NOT NULL,
            execution_receipt_ref TEXT UNIQUE NOT NULL,
            entry_bytes BLOB NOT NULL
        ) WITHOUT ROWID
    """
    _EXPECTED_TABLE_SQL = " ".join(_CREATE_TABLE_SQL.split())
    _EXPECTED_SCHEMA_OBJECTS = (
        (
            "index",
            "sqlite_autoindex_qwen_execution_journal_2",
            _TABLE_NAME,
            None,
        ),
        (
            "index",
            "sqlite_autoindex_qwen_execution_journal_3",
            _TABLE_NAME,
            None,
        ),
        ("table", _TABLE_NAME, _TABLE_NAME, _EXPECTED_TABLE_SQL),
    )
    _EXPECTED_COLUMNS = (
        (0, "idempotency_key", "TEXT", 1, None, 1, 0),
        (1, "manifest_ref", "TEXT", 1, None, 0, 0),
        (2, "execution_request_ref", "TEXT", 1, None, 0, 0),
        (3, "execution_receipt_ref", "TEXT", 1, None, 0, 0),
        (4, "entry_bytes", "BLOB", 1, None, 0, 0),
    )
    _EXPECTED_INDEXES = frozenset(
        {
            ("sqlite_autoindex_qwen_execution_journal_1", 1, "pk", 0),
            ("sqlite_autoindex_qwen_execution_journal_2", 1, "u", 0),
            ("sqlite_autoindex_qwen_execution_journal_3", 1, "u", 0),
        }
    )

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        if not isinstance(path, Path):
            raise TypeError("journal path must be a str or Path")
        if not path.is_absolute():
            raise ValueError("journal path must be absolute")
        if not path.parent.is_dir():
            raise ValueError("journal parent directory must already exist")
        existed = path.exists()
        if existed and not path.is_file():
            raise ValueError("journal path must identify a regular file")
        self.path = path
        self._initialize(existed=existed)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _connect_validation_only(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path.as_uri() + "?mode=ro",
            timeout=5.0,
            isolation_level=None,
            uri=True,
        )
        connection.execute("PRAGMA query_only=ON")
        return connection

    @classmethod
    def _schema_objects(
        cls,
        connection: sqlite3.Connection,
    ) -> tuple[tuple[object, ...], ...]:
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            ORDER BY type, name
            """
        ).fetchall()
        return tuple(
            (
                kind,
                name,
                table_name,
                None if sql is None else " ".join(sql.split()),
            )
            for kind, name, table_name, sql in rows
        )

    @classmethod
    def _assert_exact_schema(cls, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != cls.SCHEMA_VERSION:
            raise ValueError("Qwen journal schema version is unsupported")
        if cls._schema_objects(connection) != cls._EXPECTED_SCHEMA_OBJECTS:
            raise ValueError("Qwen journal schema objects are not exact")
        columns = tuple(
            connection.execute(
                "PRAGMA table_xinfo(qwen_execution_journal)"
            ).fetchall()
        )
        if columns != cls._EXPECTED_COLUMNS:
            raise ValueError("Qwen journal table shape is not exact")
        indexes = frozenset(
            (name, unique, origin, partial)
            for _sequence, name, unique, origin, partial in connection.execute(
                "PRAGMA index_list(qwen_execution_journal)"
            ).fetchall()
        )
        if indexes != cls._EXPECTED_INDEXES:
            raise ValueError("Qwen journal index shape is not exact")

    def _initialize(self, *, existed: bool) -> None:
        if existed:
            validation = self._connect_validation_only()
            try:
                version = int(
                    validation.execute("PRAGMA user_version").fetchone()[0]
                )
                if version == self.SCHEMA_VERSION:
                    self._assert_exact_schema(validation)
                    return
                if version != 0:
                    raise ValueError("Qwen journal schema version is unsupported")
                if self._schema_objects(validation):
                    raise ValueError(
                        "Qwen journal version-zero schema must be empty"
                    )
            finally:
                validation.close()

        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != 0:
                raise ValueError("Qwen journal schema version is unsupported")
            if self._schema_objects(connection):
                raise ValueError("Qwen journal version-zero schema must be empty")
            connection.execute("BEGIN IMMEDIATE")
            if (
                int(connection.execute("PRAGMA user_version").fetchone()[0]) != 0
                or self._schema_objects(connection)
            ):
                raise ValueError("Qwen journal version-zero schema changed")
            connection.execute(self._CREATE_TABLE_SQL)
            connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            connection.commit()
            self._assert_exact_schema(connection)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _decode(value: object) -> QwenExecutionJournalEntry:
        if type(value) is not bytes:
            raise ValueError("Qwen journal entry storage is not exact bytes")
        if not 1 <= len(value) <= MAX_JOURNAL_ENTRY_BYTES:
            raise ValueError("Qwen journal entry byte length is invalid")
        payload = parse_json(value)
        if type(payload) is not dict:
            raise ValueError("Qwen journal entry is not an object")
        entry = QwenExecutionJournalEntry.from_payload(payload)
        if entry.canonical_bytes() != value:
            raise ValueError("Qwen journal entry bytes are not canonical")
        return entry

    @staticmethod
    def _assert_row(
        row: tuple[object, ...],
        entry: QwenExecutionJournalEntry,
    ) -> None:
        if row[:3] != (
            entry.manifest_ref,
            entry.execution_request_ref,
            entry.execution_receipt_ref,
        ):
            raise ValueError("Qwen journal index fields differ from canonical bytes")

    def get(self, idempotency_key: str) -> QwenExecutionJournalEntry | None:
        key = _sha256_ref(idempotency_key, "idempotency_key")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT manifest_ref, execution_request_ref,
                       execution_receipt_ref, entry_bytes
                FROM qwen_execution_journal
                WHERE idempotency_key = ?
                """,
                (key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        entry = self._decode(row[3])
        if entry.idempotency_key != key:
            raise ValueError("Qwen journal key differs from canonical bytes")
        self._assert_row(row, entry)
        return entry

    def put_if_absent(
        self,
        entry: QwenExecutionJournalEntry,
    ) -> QwenExecutionJournalEntry:
        if type(entry) is not QwenExecutionJournalEntry:
            raise TypeError("entry must be an exact QwenExecutionJournalEntry")
        encoded = entry.canonical_bytes()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT manifest_ref, execution_request_ref,
                       execution_receipt_ref, entry_bytes
                FROM qwen_execution_journal
                WHERE idempotency_key = ?
                """,
                (entry.idempotency_key,),
            ).fetchone()
            if row is not None:
                existing = self._decode(row[3])
                self._assert_row(row, existing)
                if existing != entry or row[3] != encoded:
                    raise ValueError("Qwen journal idempotency identity conflict")
                connection.commit()
                return existing
            try:
                connection.execute(
                    """
                    INSERT INTO qwen_execution_journal (
                        idempotency_key, manifest_ref, execution_request_ref,
                        execution_receipt_ref, entry_bytes
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entry.idempotency_key,
                        entry.manifest_ref,
                        entry.execution_request_ref,
                        entry.execution_receipt_ref,
                        encoded,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Qwen journal unique identity conflict") from exc
            connection.commit()
            return entry
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def audit_integrity(self) -> None:
        connection = self._connect()
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise ValueError("Qwen journal SQLite integrity check failed")
            rows = connection.execute(
                """
                SELECT idempotency_key, manifest_ref, execution_request_ref,
                       execution_receipt_ref, entry_bytes
                FROM qwen_execution_journal
                ORDER BY idempotency_key
                """
            ).fetchall()
        finally:
            connection.close()
        for key, *row in rows:
            entry = self._decode(row[3])
            if entry.idempotency_key != key:
                raise ValueError("Qwen journal audit found a key mismatch")
            self._assert_row(tuple(row), entry)


class FrozenQwenCognitiveExecutorV1:
    """Execute only the learned-selected public trace and journal its receipt."""

    def __init__(
        self,
        io: LocalQwenIO,
        manifest: FrozenQwenCycleManifestV1,
        journal: SQLiteQwenExecutionJournal,
    ) -> None:
        if type(manifest) is not FrozenQwenCycleManifestV1:
            raise TypeError("manifest must be an exact FrozenQwenCycleManifestV1")
        if type(journal) is not SQLiteQwenExecutionJournal:
            raise TypeError("journal must be an exact SQLiteQwenExecutionJournal")
        manifest.assert_io(io)
        self.io = io
        self.manifest = manifest
        self.journal = journal

    @property
    def model_ref(self) -> str:
        return self.manifest.model_ref

    @property
    def manifest_ref(self) -> str:
        return self.manifest.manifest_ref

    def _assert_entry(
        self,
        entry: QwenExecutionJournalEntry,
        request: CognitiveExecutionRequest,
    ) -> None:
        expected_prompt_ref = _content_ref(
            {"prompt": build_qwen_execution_prompt(request)}
        )
        if (
            entry.manifest_ref != self.manifest_ref
            or entry.request != request
            or entry.generation.model_ref != self.manifest.model_ref
            or entry.generation.tokenizer_ref != self.manifest.tokenizer_ref
            or entry.generation.generation_config_ref
            != self.manifest.generation_config_ref
            or entry.generation.prompt_ref != expected_prompt_ref
            or not 1
            <= entry.generation.prompt_tokens
            <= self.manifest.max_input_tokens
            or not 1
            <= len(entry.generation.generated_token_ids)
            <= self.manifest.max_new_tokens
        ):
            raise ValueError("journaled Qwen execution differs from the exact context")
        entry.receipt.assert_request(request)

    @staticmethod
    def _execution(entry: QwenExecutionJournalEntry) -> CognitiveExecution:
        return CognitiveExecution(
            entry.receipt.status,
            entry.receipt.executed_trace,
            entry.receipt.response,
            entry.receipt.output_observations,
        )

    def execute(self, request: CognitiveExecutionRequest) -> CognitiveExecution:
        if type(request) is not CognitiveExecutionRequest:
            raise TypeError("request must be an exact CognitiveExecutionRequest")
        self.manifest.assert_io(self.io)
        existing = self.journal.get(request.idempotency_key)
        if existing is not None:
            self._assert_entry(existing, request)
            return self._execution(existing)
        prompt = build_qwen_execution_prompt(request)
        generation = self.io.generate_record(prompt)
        if (
            generation.model_ref != self.manifest.model_ref
            or generation.tokenizer_ref != self.manifest.tokenizer_ref
            or generation.generation_config_ref
            != self.manifest.generation_config_ref
            or generation.prompt_ref != _content_ref({"prompt": prompt})
            or not 1 <= generation.prompt_tokens <= self.manifest.max_input_tokens
            or not 1
            <= len(generation.generated_token_ids)
            <= self.manifest.max_new_tokens
        ):
            raise ValueError("Qwen execution generation differs from the manifest")
        if not generation.response.strip():
            raise RuntimeError("Qwen returned an empty execution response")
        receipt = CognitiveExecutionReceipt.from_request(
            request,
            status="COMPLETED",
            executed_trace=request.selected_trace,
            response=generation.response,
            output_observations=(),
        )
        entry = QwenExecutionJournalEntry(
            manifest_ref=self.manifest_ref,
            request=request,
            generation=generation,
            receipt=receipt,
        )
        durable = self.journal.put_if_absent(entry)
        self._assert_entry(durable, request)
        return self._execution(durable)

    def recover(
        self,
        request: CognitiveExecutionRequest,
    ) -> CognitiveExecutionRecovery:
        if type(request) is not CognitiveExecutionRequest:
            raise TypeError("request must be an exact CognitiveExecutionRequest")
        self.manifest.assert_io(self.io)
        entry = self.journal.get(request.idempotency_key)
        if entry is None:
            return CognitiveExecutionRecovery("NOT_STARTED")
        self._assert_entry(entry, request)
        return CognitiveExecutionRecovery("RECORDED", entry.receipt)


__all__ = [
    "FrozenQwenCognitiveExecutorV1",
    "FrozenQwenCycleManifestV1",
    "FrozenQwenProcedureAdapterV1",
    "FrozenQwenProcedureRelationAdapterV1",
    "MAX_JOURNAL_ENTRY_BYTES",
    "QWEN_CYCLE_MANIFEST_SCHEMA",
    "QWEN_EXECUTION_JOURNAL_SCHEMA",
    "QWEN_KNOWLEDGE_ENCODER_SCHEMA",
    "QWEN_LABEL_FREE_PROJECTION_SCHEMA",
    "QWEN_MOVING_ORIGIN_SCHEMA",
    "QWEN_PROCEDURE_PROPOSAL_SCHEMA",
    "QWEN_RECEIPT_OBSERVATION_SCHEMA",
    "QWEN_SELECTED_PROCEDURE_EXECUTION_SCHEMA",
    "QwenExecutionJournalEntry",
    "SQLiteQwenExecutionJournal",
    "build_label_free_projection",
    "build_qwen_execution_prompt",
    "build_qwen_procedure_proposal_prompt",
    "parse_qwen_procedure_proposals",
]
