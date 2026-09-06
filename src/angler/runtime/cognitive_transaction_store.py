"""Atomic canonical episode, competence-state, and projection persistence.

The SQLite database is the authority.  Cognee and Moving Origin consume the
outbox later; acknowledging those projections never changes the canonical
episode or competence head.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tempfile
from typing import Final

from angler.cognition.contracts import CognitiveEpisode
from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
    ObjectiveFeedbackRecord,
    ProspectiveTurnReservation,
)
from angler.cognition.prospective_origin import (
    CognitiveEpisodeV2,
    ProspectiveDynamicsBatch,
    ProspectiveResolution,
    ProspectiveTurnReservationV2,
    ResolutionDisposition,
)
from angler.memory.cognitive_acquisition import (
    CognitiveAcquisition,
    CognitiveGraphProjectionV2,
    LEGACY_COGNITIVE_EPISODE_CONTRACT,
    PROSPECTIVE_DYNAMICS_BATCH_CONTRACT,
    PROSPECTIVE_RESOLUTION_CONTRACT,
)
from angler.memory.cognitive_graph import episode_record


_SCHEMA_VERSION: Final[int] = 3
_V2_SCHEMA_VERSION: Final[int] = 2
_SHA256_REF: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_V1_SCHEMA_FINGERPRINT: Final[str] = (
    "1cacf8a5fec80e9819a1527e0644d6087bb028b7e9b2967f27f5e3cabcf1cff1"
)
_V2_SCHEMA_FINGERPRINT: Final[str] = (
    "e8f33be7d9f8dd03d5e1b6242b71a81ca9599507159e6b97029635660d54b089"
)
_SCHEMA_FINGERPRINT: Final[str] = (
    "7ee40c612af03f4619ba9a017c654ceefb45e513229fe811a3965b42b6501f5f"
)
_MAX_PENDING_BLOB_BYTES: Final[int] = 16 * 1024 * 1024
_ACTIVE_RESERVATION_STATUSES: Final[frozenset[str]] = frozenset(
    {"RESERVED", "CLAIMED", "EXECUTION_RECORDED"}
)
_NONCOMPLETION_STATUSES: Final[frozenset[str]] = frozenset(
    {"CLARIFICATION_REQUIRED", "ERROR"}
)
_PROSPECTIVE_RESOLUTION_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "OBSERVED",
        "COMPLETED_UNEVALUATED",
        "CANCELLED",
        "CLARIFICATION_REQUIRED",
        "ERROR",
    }
)
_SCHEMA_V1: Final[str] = """
CREATE TABLE IF NOT EXISTS store_identity (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    model_ref TEXT NOT NULL,
    encoder_ref TEXT NOT NULL,
    UNIQUE(model_ref, encoder_ref)
);
CREATE TABLE IF NOT EXISTS state_history (
    sequence INTEGER PRIMARY KEY,
    episode_ref TEXT UNIQUE,
    state_digest TEXT NOT NULL,
    snapshot_sha256 TEXT NOT NULL,
    snapshot BLOB NOT NULL,
    UNIQUE(sequence, episode_ref, state_digest, snapshot_sha256)
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_ref TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL UNIQUE,
    parent_state_digest TEXT NOT NULL,
    child_state_digest TEXT NOT NULL,
    child_snapshot_sha256 TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    encoder_ref TEXT NOT NULL,
    canonical_payload BLOB NOT NULL UNIQUE,
    UNIQUE(sequence, episode_ref, canonical_payload, payload_sha256),
    FOREIGN KEY(sequence, episode_ref, child_state_digest, child_snapshot_sha256)
        REFERENCES state_history(sequence, episode_ref, state_digest, snapshot_sha256),
    FOREIGN KEY(model_ref, encoder_ref)
        REFERENCES store_identity(model_ref, encoder_ref)
);
CREATE TABLE IF NOT EXISTS canonical_head (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    sequence INTEGER NOT NULL,
    episode_ref TEXT,
    state_digest TEXT NOT NULL,
    snapshot_sha256 TEXT NOT NULL,
    FOREIGN KEY(sequence, episode_ref, state_digest, snapshot_sha256)
        REFERENCES state_history(sequence, episode_ref, state_digest, snapshot_sha256)
);
CREATE TABLE IF NOT EXISTS projection_outbox (
    sequence INTEGER PRIMARY KEY,
    episode_ref TEXT NOT NULL UNIQUE,
    payload_sha256 TEXT NOT NULL,
    canonical_payload BLOB NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
    FOREIGN KEY(sequence, episode_ref, canonical_payload, payload_sha256)
        REFERENCES episodes(sequence, episode_ref, canonical_payload, payload_sha256)
);
"""

_SCHEMA_V2_ADDITION: Final[str] = """
CREATE TABLE IF NOT EXISTS turn_reservations (
    reservation_ref TEXT PRIMARY KEY,
    commitment_ref TEXT NOT NULL,
    episode_context_ref TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('RESERVED', 'CLAIMED', 'EXECUTION_RECORDED', 'RESOLVED')
    ),
    parent_sequence INTEGER NOT NULL CHECK (parent_sequence >= 0),
    parent_event_ref TEXT,
    parent_state_digest TEXT NOT NULL,
    parent_snapshot_sha256 TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    encoder_ref TEXT NOT NULL,
    canonical_reservation BLOB NOT NULL UNIQUE,
    pending_blob_sha256 TEXT NOT NULL,
    pending_blob_size INTEGER NOT NULL CHECK (
        pending_blob_size >= 0 AND pending_blob_size <= 16777216
    ),
    pending_blob BLOB,
    execution_request_ref TEXT UNIQUE,
    canonical_execution_request BLOB UNIQUE,
    execution_receipt_ref TEXT UNIQUE,
    canonical_execution_receipt BLOB UNIQUE,
    feedback_ref TEXT UNIQUE,
    canonical_feedback BLOB UNIQUE,
    resolution_disposition TEXT CHECK (
        resolution_disposition IS NULL OR resolution_disposition IN (
            'EPISODE_COMMITTED', 'CANCELLED', 'CLARIFICATION_REQUIRED', 'ERROR'
        )
    ),
    resolved_episode_ref TEXT UNIQUE,
    FOREIGN KEY(parent_sequence) REFERENCES state_history(sequence),
    FOREIGN KEY(model_ref, encoder_ref)
        REFERENCES store_identity(model_ref, encoder_ref),
    FOREIGN KEY(resolved_episode_ref) REFERENCES episodes(episode_ref),
    CHECK (
        (status = 'RESERVED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NULL
            AND canonical_execution_request IS NULL
            AND execution_receipt_ref IS NULL
            AND canonical_execution_receipt IS NULL
            AND feedback_ref IS NULL
            AND canonical_feedback IS NULL
            AND resolution_disposition IS NULL
            AND resolved_episode_ref IS NULL)
        OR
        (status = 'CLAIMED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NOT NULL
            AND canonical_execution_request IS NOT NULL
            AND execution_receipt_ref IS NULL
            AND canonical_execution_receipt IS NULL
            AND feedback_ref IS NULL
            AND canonical_feedback IS NULL
            AND resolution_disposition IS NULL
            AND resolved_episode_ref IS NULL)
        OR
        (status = 'EXECUTION_RECORDED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NOT NULL
            AND canonical_execution_request IS NOT NULL
            AND execution_receipt_ref IS NOT NULL
            AND canonical_execution_receipt IS NOT NULL
            AND resolution_disposition IS NULL
            AND resolved_episode_ref IS NULL)
        OR
        (status = 'RESOLVED'
            AND pending_blob IS NULL
            AND resolution_disposition IS NOT NULL
            AND (
                (resolution_disposition = 'EPISODE_COMMITTED'
                    AND resolved_episode_ref IS NOT NULL
                    AND execution_request_ref IS NOT NULL
                    AND canonical_execution_request IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND canonical_execution_receipt IS NOT NULL
                    AND feedback_ref IS NOT NULL
                    AND canonical_feedback IS NOT NULL)
                OR
                (resolution_disposition IN ('CLARIFICATION_REQUIRED', 'ERROR')
                    AND resolved_episode_ref IS NULL
                    AND execution_request_ref IS NOT NULL
                    AND canonical_execution_request IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND canonical_execution_receipt IS NOT NULL
                    AND feedback_ref IS NULL
                    AND canonical_feedback IS NULL)
                OR
                (resolution_disposition = 'CANCELLED'
                    AND resolved_episode_ref IS NULL
                    AND execution_request_ref IS NULL
                    AND canonical_execution_request IS NULL
                    AND execution_receipt_ref IS NULL
                    AND canonical_execution_receipt IS NULL
                    AND feedback_ref IS NULL
                    AND canonical_feedback IS NULL)
            ))
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_turn_reservation
    ON turn_reservations ((1)) WHERE status != 'RESOLVED';
CREATE INDEX IF NOT EXISTS turn_reservation_episode_context
    ON turn_reservations (episode_context_ref);
"""

_SCHEMA_V3_ADDITION: Final[str] = """
CREATE TABLE IF NOT EXISTS cognitive_acquisitions (
    ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0),
    acquisition_ref TEXT NOT NULL UNIQUE,
    predecessor_acquisition_ref TEXT UNIQUE,
    source_contract TEXT NOT NULL CHECK (source_contract IN (
        'ANG-CTR-COGNITIVE-EPISODE-001@0.1.0',
        'ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001@0.1.0',
        'ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0'
    )),
    source_ref TEXT NOT NULL UNIQUE,
    record_ref TEXT NOT NULL UNIQUE,
    payload_sha256 TEXT NOT NULL UNIQUE,
    canonical_payload BLOB NOT NULL UNIQUE,
    UNIQUE(ordinal, acquisition_ref, record_ref),
    UNIQUE(acquisition_ref, record_ref),
    FOREIGN KEY(predecessor_acquisition_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref),
    CHECK (
        (ordinal = 0 AND predecessor_acquisition_ref IS NULL)
        OR (ordinal > 0 AND predecessor_acquisition_ref IS NOT NULL)
    )
);
CREATE TABLE IF NOT EXISTS acquisition_clock (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_ordinal INTEGER NOT NULL CHECK (next_ordinal >= 0),
    acquisition_ref TEXT,
    record_ref TEXT,
    FOREIGN KEY(acquisition_ref, record_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref, record_ref),
    CHECK (
        (next_ordinal = 0 AND acquisition_ref IS NULL AND record_ref IS NULL)
        OR (next_ordinal > 0 AND acquisition_ref IS NOT NULL AND record_ref IS NOT NULL)
    )
);
CREATE TABLE IF NOT EXISTS acquisition_projection_outbox (
    ordinal INTEGER PRIMARY KEY,
    acquisition_ref TEXT NOT NULL UNIQUE,
    record_ref TEXT NOT NULL UNIQUE,
    projection_ref TEXT NOT NULL UNIQUE,
    payload_sha256 TEXT NOT NULL UNIQUE,
    canonical_payload BLOB NOT NULL UNIQUE,
    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
    FOREIGN KEY(ordinal, acquisition_ref, record_ref)
        REFERENCES cognitive_acquisitions(ordinal, acquisition_ref, record_ref)
);
CREATE TABLE IF NOT EXISTS prospective_turns_v3 (
    reservation_ref TEXT PRIMARY KEY,
    legacy_reservation_ref TEXT NOT NULL UNIQUE,
    batch_ref TEXT NOT NULL UNIQUE,
    parent_lineage_ref TEXT NOT NULL,
    selected_branch_ref TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('RESERVED', 'CLAIMED', 'EXECUTION_RECORDED', 'RESOLVED')
    ),
    parent_sequence INTEGER NOT NULL CHECK (parent_sequence >= 0),
    parent_event_ref TEXT,
    parent_state_digest TEXT NOT NULL,
    parent_snapshot_sha256 TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    encoder_ref TEXT NOT NULL,
    reservation_payload_sha256 TEXT NOT NULL UNIQUE,
    canonical_reservation BLOB NOT NULL UNIQUE,
    canonical_batch BLOB NOT NULL UNIQUE,
    pending_blob_sha256 TEXT NOT NULL,
    pending_blob_size INTEGER NOT NULL CHECK (
        pending_blob_size >= 0 AND pending_blob_size <= 16777216
    ),
    pending_blob BLOB,
    execution_request_ref TEXT UNIQUE,
    canonical_execution_request BLOB UNIQUE,
    execution_receipt_ref TEXT UNIQUE,
    canonical_execution_receipt BLOB UNIQUE,
    feedback_ref TEXT UNIQUE,
    canonical_feedback BLOB UNIQUE,
    resolution_ref TEXT UNIQUE,
    resolution_disposition TEXT CHECK (
        resolution_disposition IS NULL OR resolution_disposition IN (
            'OBSERVED', 'COMPLETED_UNEVALUATED', 'CANCELLED',
            'CLARIFICATION_REQUIRED', 'ERROR'
        )
    ),
    canonical_resolution BLOB UNIQUE,
    legacy_episode_ref TEXT UNIQUE,
    episode_v2_ref TEXT UNIQUE,
    canonical_episode_v2 BLOB UNIQUE,
    batch_acquisition_ref TEXT NOT NULL UNIQUE,
    resolution_acquisition_ref TEXT UNIQUE,
    FOREIGN KEY(parent_sequence) REFERENCES state_history(sequence),
    FOREIGN KEY(model_ref, encoder_ref)
        REFERENCES store_identity(model_ref, encoder_ref),
    FOREIGN KEY(legacy_episode_ref) REFERENCES episodes(episode_ref),
    FOREIGN KEY(batch_acquisition_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref),
    FOREIGN KEY(resolution_acquisition_ref)
        REFERENCES cognitive_acquisitions(acquisition_ref),
    CHECK ((execution_request_ref IS NULL) = (canonical_execution_request IS NULL)),
    CHECK ((execution_receipt_ref IS NULL) = (canonical_execution_receipt IS NULL)),
    CHECK ((feedback_ref IS NULL) = (canonical_feedback IS NULL)),
    CHECK ((resolution_ref IS NULL) = (canonical_resolution IS NULL)),
    CHECK ((episode_v2_ref IS NULL) = (canonical_episode_v2 IS NULL)),
    CHECK (
        (status = 'RESERVED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NULL
            AND execution_receipt_ref IS NULL
            AND feedback_ref IS NULL
            AND resolution_ref IS NULL
            AND resolution_disposition IS NULL
            AND legacy_episode_ref IS NULL
            AND episode_v2_ref IS NULL
            AND resolution_acquisition_ref IS NULL)
        OR
        (status = 'CLAIMED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NOT NULL
            AND execution_receipt_ref IS NULL
            AND feedback_ref IS NULL
            AND resolution_ref IS NULL
            AND resolution_disposition IS NULL
            AND legacy_episode_ref IS NULL
            AND episode_v2_ref IS NULL
            AND resolution_acquisition_ref IS NULL)
        OR
        (status = 'EXECUTION_RECORDED'
            AND pending_blob IS NOT NULL
            AND length(pending_blob) = pending_blob_size
            AND execution_request_ref IS NOT NULL
            AND execution_receipt_ref IS NOT NULL
            AND resolution_ref IS NULL
            AND resolution_disposition IS NULL
            AND legacy_episode_ref IS NULL
            AND episode_v2_ref IS NULL
            AND resolution_acquisition_ref IS NULL)
        OR
        (status = 'RESOLVED'
            AND pending_blob IS NULL
            AND resolution_ref IS NOT NULL
            AND resolution_disposition IS NOT NULL
            AND resolution_acquisition_ref IS NOT NULL
            AND (
                (resolution_disposition = 'OBSERVED'
                    AND execution_request_ref IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND feedback_ref IS NOT NULL
                    AND legacy_episode_ref IS NOT NULL
                    AND episode_v2_ref IS NOT NULL)
                OR
                (resolution_disposition = 'COMPLETED_UNEVALUATED'
                    AND execution_request_ref IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND feedback_ref IS NULL
                    AND legacy_episode_ref IS NULL
                    AND episode_v2_ref IS NULL)
                OR
                (resolution_disposition IN ('CLARIFICATION_REQUIRED', 'ERROR')
                    AND execution_request_ref IS NOT NULL
                    AND execution_receipt_ref IS NOT NULL
                    AND feedback_ref IS NULL
                    AND legacy_episode_ref IS NULL
                    AND episode_v2_ref IS NULL)
                OR
                (resolution_disposition = 'CANCELLED'
                    AND execution_request_ref IS NULL
                    AND execution_receipt_ref IS NULL
                    AND feedback_ref IS NULL
                    AND legacy_episode_ref IS NULL
                    AND episode_v2_ref IS NULL)
            ))
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_prospective_turn_v3
    ON prospective_turns_v3 ((1)) WHERE status != 'RESOLVED';
CREATE INDEX IF NOT EXISTS prospective_turn_v3_legacy_ref
    ON prospective_turns_v3 (legacy_reservation_ref);
"""

_SCHEMA: Final[str] = _SCHEMA_V1 + _SCHEMA_V2_ADDITION + _SCHEMA_V3_ADDITION
_RESERVATION_COLUMNS: Final[str] = (
    "reservation_ref, commitment_ref, episode_context_ref, status, parent_sequence, parent_event_ref, "
    "parent_state_digest, parent_snapshot_sha256, model_ref, encoder_ref, "
    "canonical_reservation, pending_blob_sha256, pending_blob_size, pending_blob, "
    "execution_request_ref, canonical_execution_request, execution_receipt_ref, "
    "canonical_execution_receipt, feedback_ref, canonical_feedback, "
    "resolution_disposition, resolved_episode_ref"
)
_PROSPECTIVE_RESERVATION_COLUMNS: Final[str] = (
    "reservation_ref, legacy_reservation_ref, batch_ref, parent_lineage_ref, "
    "selected_branch_ref, status, parent_sequence, parent_event_ref, "
    "parent_state_digest, parent_snapshot_sha256, model_ref, encoder_ref, "
    "reservation_payload_sha256, canonical_reservation, canonical_batch, "
    "pending_blob_sha256, pending_blob_size, pending_blob, execution_request_ref, "
    "canonical_execution_request, execution_receipt_ref, canonical_execution_receipt, "
    "feedback_ref, canonical_feedback, resolution_ref, resolution_disposition, "
    "canonical_resolution, legacy_episode_ref, episode_v2_ref, canonical_episode_v2, "
    "batch_acquisition_ref, resolution_acquisition_ref"
)


def _database_schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    material = "\n".join(
        "\x1f".join("" if item is None else str(item) for item in row) for row in rows
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class StateHead:
    sequence: int
    episode_ref: str | None
    state_digest: str
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class TransactionCommit:
    episode_ref: str
    sequence: int
    head: StateHead
    projection_pending: bool


@dataclass(frozen=True, slots=True)
class ProjectionOutboxItem:
    sequence: int
    episode_ref: str
    episode: CognitiveEpisode


@dataclass(frozen=True, slots=True)
class CognitiveEpisodeItem:
    sequence: int
    episode_ref: str
    episode: CognitiveEpisode


@dataclass(frozen=True, slots=True)
class AcquisitionHead:
    next_ordinal: int
    acquisition_ref: str | None
    record_ref: str | None


@dataclass(frozen=True, slots=True)
class CognitiveAcquisitionItem:
    ordinal: int
    acquisition_ref: str
    record_ref: str
    acquisition: CognitiveAcquisition


@dataclass(frozen=True, slots=True)
class AcquisitionProjectionOutboxItem:
    ordinal: int
    acquisition_ref: str
    record_ref: str
    projection_ref: str
    projection: CognitiveGraphProjectionV2


@dataclass(frozen=True, slots=True)
class ReservationRecord:
    """One verified durable prospective-turn aggregate."""

    status: str
    reservation: ProspectiveTurnReservation
    pending_blob: bytes | None
    execution_request: CognitiveExecutionRequest | None
    execution_receipt: CognitiveExecutionReceipt | None
    feedback: ObjectiveFeedbackRecord | None
    resolution_disposition: str | None
    resolved_episode_ref: str | None

    @property
    def commitment_ref(self) -> str:
        return self.reservation.commitment_ref

    @property
    def reservation_ref(self) -> str:
        return self.reservation.reservation_ref


@dataclass(frozen=True, slots=True)
class ReservationTransition:
    """A lifecycle result that distinguishes an actual write from a replay."""

    record: ReservationRecord
    transitioned: bool


@dataclass(frozen=True, slots=True)
class ProspectiveReservationRecord:
    """One verified durable successor prospective-turn aggregate."""

    status: str
    reservation: ProspectiveTurnReservationV2
    pending_blob: bytes | None
    execution_request: CognitiveExecutionRequest | None
    execution_receipt: CognitiveExecutionReceipt | None
    feedback: ObjectiveFeedbackRecord | None
    resolution: ProspectiveResolution | None
    episode_v2: CognitiveEpisodeV2 | None
    legacy_episode_ref: str | None
    batch_acquisition_ref: str
    resolution_acquisition_ref: str | None

    @property
    def reservation_ref(self) -> str:
        return self.reservation.reservation_ref

    @property
    def legacy_reservation_ref(self) -> str:
        return self.reservation.legacy_reservation_ref

    @property
    def resolution_disposition(self) -> str | None:
        return None if self.resolution is None else self.resolution.disposition.value


@dataclass(frozen=True, slots=True)
class ProspectiveReservationTransition:
    record: ProspectiveReservationRecord
    transitioned: bool


FaultInjector = Callable[[str], None]


@contextmanager
def _read_snapshot(connection: sqlite3.Connection):
    """Hold one consistent SQLite snapshot across a multi-SELECT rejoin."""

    connection.execute("BEGIN")
    try:
        yield
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _nonempty_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _sha256_ref(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _snapshot_bytes(value: object) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError("competence snapshot must be non-empty bytes")
    return value


def _snapshot_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_episode_bytes(episode: CognitiveEpisode) -> bytes:
    if type(episode) is not CognitiveEpisode:
        raise TypeError("episode must be a CognitiveEpisode")
    # The contract owns canonicalization.  Keeping the conversion in one
    # narrow helper makes the transaction layer unable to reinterpret fields.
    payload = episode.canonical_bytes()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("canonical episode payload must be non-empty bytes")
    # Frozen dataclasses can still be forged through object.__setattr__.  A
    # canonical store must re-run every contract invariant at its trust edge.
    try:
        validated = CognitiveEpisode.from_json(payload)
    except Exception as exc:
        raise ValueError("episode does not satisfy the canonical contract") from exc
    if validated != episode or validated.episode_ref != episode.episode_ref:
        raise ValueError("episode does not round-trip through its canonical contract")
    return payload


def _episode_from_bytes(payload: bytes) -> CognitiveEpisode:
    return CognitiveEpisode.from_json(payload)


def _canonical_contract_bytes(value: object, expected_type: type, label: str) -> bytes:
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    payload = value.canonical_bytes()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"canonical {label} payload must be non-empty bytes")
    try:
        validated = expected_type.from_json(payload)
    except Exception as exc:
        raise ValueError(f"{label} does not satisfy its canonical contract") from exc
    if validated != value:
        raise ValueError(f"{label} does not round-trip through its canonical contract")
    return payload


def _pending_blob_bytes(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError("pending_blob must be bytes")
    if len(value) > _MAX_PENDING_BLOB_BYTES:
        raise ValueError("pending_blob exceeds the 16 MiB ceiling")
    return value


def _episode_context_ref(
    *,
    task_id: str,
    request: str,
    recalled_refs: tuple[str, ...],
    proposals: tuple[str, ...],
    selected_index: int,
    commitment_payload: dict[str, object],
    parent_state_digest: str,
    model_ref: str,
    encoder_ref: str,
    supporting_evidence_refs: tuple[str, ...],
) -> str:
    """Identify only the reservation context representable by an episode."""

    payload = json.dumps(
        {
            "commitment": commitment_payload,
            "encoder_ref": encoder_ref,
            "model_ref": model_ref,
            "parent_state_digest": parent_state_digest,
            "proposals": list(proposals),
            "recalled_refs": list(recalled_refs),
            "request": request,
            "selected_index": selected_index,
            "supporting_evidence_refs": list(supporting_evidence_refs),
            "task_id": task_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _snapshot_digest(payload)


def _reservation_episode_context_ref(
    reservation: ProspectiveTurnReservation,
) -> str:
    return _episode_context_ref(
        task_id=reservation.task_id,
        request=reservation.request,
        recalled_refs=reservation.recalled_refs,
        proposals=reservation.proposals,
        selected_index=reservation.selected_index,
        commitment_payload=reservation.commitment.to_payload(),
        parent_state_digest=reservation.parent_competence_digest,
        model_ref=reservation.model_ref,
        encoder_ref=reservation.encoder_ref,
        supporting_evidence_refs=reservation.supporting_evidence_refs,
    )


def _cognitive_episode_context_ref(episode: CognitiveEpisode) -> str:
    return _episode_context_ref(
        task_id=episode.task_id,
        request=episode.request,
        recalled_refs=episode.recalled_refs,
        proposals=episode.proposals,
        selected_index=episode.selected_index,
        commitment_payload=episode.commitment.to_payload(),
        parent_state_digest=episode.parent_state_digest,
        model_ref=episode.model_ref,
        encoder_ref=episode.encoder_ref,
        supporting_evidence_refs=episode.supporting_evidence_refs,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _verified_v2_dump(connection: sqlite3.Connection) -> tuple[str, ...]:
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) != _V2_SCHEMA_VERSION:
        raise RuntimeError("cognitive store backup is not schema version 2")
    if _database_schema_fingerprint(connection) != _V2_SCHEMA_FINGERPRINT:
        raise RuntimeError("cognitive store backup schema fingerprint mismatch")
    quick = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check"))
    if quick != ("ok",):
        raise RuntimeError("cognitive store backup SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("cognitive store backup foreign-key integrity check failed")
    return tuple(connection.iterdump())


def _backup_paths(path: Path) -> tuple[Path, Path]:
    backup = Path(str(path) + ".pre-v3.sqlite")
    return backup, Path(str(backup) + ".sha256")


def _verify_backup_pair(
    store_path: Path,
    *,
    expected_dump: tuple[str, ...] | None = None,
) -> tuple[Path, tuple[str, ...]]:
    backup, sidecar = _backup_paths(store_path)
    if not backup.exists() or not sidecar.exists():
        raise RuntimeError("cognitive store v3 backup collision is incomplete")
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if backup.stat().st_mode & write_bits or sidecar.stat().st_mode & write_bits:
        raise RuntimeError("cognitive store v3 backup collision is writable")
    try:
        sidecar_bytes = sidecar.read_bytes()
    except OSError as exc:
        raise RuntimeError("cognitive store v3 backup digest cannot be read") from exc
    if re.fullmatch(rb"[0-9a-f]{64}\n", sidecar_bytes) is None:
        raise RuntimeError("cognitive store v3 backup digest sidecar is not canonical")
    digest = sidecar_bytes[:-1].decode("ascii")
    if _file_sha256(backup) != digest:
        raise RuntimeError("cognitive store v3 backup digest mismatch")
    with closing(_readonly_connection(backup)) as connection:
        actual_dump = _verified_v2_dump(connection)
    if expected_dump is not None and actual_dump != expected_dump:
        raise RuntimeError("cognitive store v3 backup logical content drifted")
    return backup, actual_dump


def _ensure_v2_backup(store_path: Path, locked: sqlite3.Connection) -> None:
    """Create or verify the immutable pre-v3 rollback pair."""

    if int(locked.execute("PRAGMA user_version").fetchone()[0]) != _V2_SCHEMA_VERSION:
        raise RuntimeError("cognitive store backup source is not schema version 2")
    if _database_schema_fingerprint(locked) != _V2_SCHEMA_FINGERPRINT:
        raise RuntimeError("cognitive store backup source fingerprint mismatch")
    locked_dump = tuple(locked.iterdump())
    backup, sidecar = _backup_paths(store_path)
    if backup.exists() or sidecar.exists():
        _verify_backup_pair(store_path, expected_dump=locked_dump)
        return

    backup_fd, raw_backup_temp = tempfile.mkstemp(
        prefix=f".{backup.name}.", dir=store_path.parent
    )
    os.close(backup_fd)
    sidecar_fd, raw_sidecar_temp = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", dir=store_path.parent
    )
    os.close(sidecar_fd)
    backup_temp = Path(raw_backup_temp)
    sidecar_temp = Path(raw_sidecar_temp)
    try:
        with closing(_readonly_connection(store_path)) as source:
            source_dump = _verified_v2_dump(source)
            if source_dump != locked_dump:
                raise RuntimeError("locked v2 source differs from read-only backup source")
            with closing(sqlite3.connect(backup_temp)) as destination:
                source.backup(destination)
                destination.commit()
        with closing(_readonly_connection(backup_temp)) as copied:
            if _verified_v2_dump(copied) != locked_dump:
                raise RuntimeError("temporary v3 rollback copy is not exact")
        _fsync_path(backup_temp)
        os.chmod(backup_temp, 0o444)
        _fsync_path(backup_temp)
        digest_line = (_file_sha256(backup_temp) + "\n").encode("ascii")
        with sidecar_temp.open("wb") as stream:
            stream.write(digest_line)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(sidecar_temp, 0o444)
        _fsync_path(sidecar_temp)
        os.link(backup_temp, backup)
        os.link(sidecar_temp, sidecar)
        _fsync_directory(store_path.parent)
    finally:
        for temporary in (backup_temp, sidecar_temp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    _verify_backup_pair(store_path, expected_dump=locked_dump)


def restore_pre_v3_backup(path: str | Path) -> None:
    """Offline restore of the verified immutable pre-v3 copy."""

    store_path = Path(path)
    backup, backup_dump = _verify_backup_pair(store_path)
    if not store_path.is_file():
        raise RuntimeError("offline restore target is absent")
    try:
        with closing(_readonly_connection(store_path)) as current:
            current_version = int(current.execute("PRAGMA user_version").fetchone()[0])
            if current_version == _V2_SCHEMA_VERSION:
                if _verified_v2_dump(current) != backup_dump:
                    raise RuntimeError("offline restore target is unrelated schema-v2 data")
                return
            if current_version != _SCHEMA_VERSION:
                raise RuntimeError("offline restore target has an unknown schema version")
            if _database_schema_fingerprint(current) != _SCHEMA_FINGERPRINT:
                raise RuntimeError("offline restore target is not the exact schema-v3 store")
            CognitiveTransactionStore._verify_database(current)
    except sqlite3.DatabaseError as exc:
        raise RuntimeError("offline restore target is not a verified SQLite store") from exc
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{store_path.name}.restore.", dir=store_path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        shutil.copyfile(backup, temporary)
        os.chmod(temporary, 0o600)
        _fsync_path(temporary)
        with closing(_readonly_connection(temporary)) as restored:
            if _verified_v2_dump(restored) != backup_dump:
                raise RuntimeError("temporary restored cognitive store is not exact")
        os.replace(temporary, store_path)
        _fsync_directory(store_path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    with closing(_readonly_connection(store_path)) as restored:
        if _verified_v2_dump(restored) != backup_dump:
            raise RuntimeError("restored cognitive store differs from its backup")


class CognitiveTransactionStore:
    """SQLite-backed event/state aggregate with a rebuildable outbox."""

    def __init__(
        self,
        path: str | Path,
        *,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("a durable filesystem path is required")
        if fault_injector is not None and not callable(fault_injector):
            raise TypeError("fault_injector must be callable")
        self._fault_injector = fault_injector
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            existing_tables = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            )
            if version == 0 and existing_tables:
                raise RuntimeError("unversioned cognitive store schema is not supported")
            if version not in (0, 1, _V2_SCHEMA_VERSION, _SCHEMA_VERSION):
                raise RuntimeError(
                    f"unsupported cognitive store schema version {version}"
                )
            if version == 0:
                connection.executescript(_SCHEMA)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                connection.commit()
            elif version == 1:
                self._migrate_v1(connection)
                self._migrate_v2(connection)
            elif version == _V2_SCHEMA_VERSION:
                self._migrate_v2(connection)
            if _database_schema_fingerprint(connection) != _SCHEMA_FINGERPRINT:
                raise RuntimeError("cognitive store schema fingerprint mismatch")
            with _read_snapshot(connection):
                self._verify_local_head(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        """Atomically add reservations to the one accepted v1 schema."""

        connection.execute("BEGIN IMMEDIATE")
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == _SCHEMA_VERSION:
                if _database_schema_fingerprint(connection) != _SCHEMA_FINGERPRINT:
                    raise RuntimeError(
                        "concurrently migrated cognitive store fingerprint mismatch"
                    )
                connection.commit()
                return
            if version == _V2_SCHEMA_VERSION:
                if _database_schema_fingerprint(connection) != _V2_SCHEMA_FINGERPRINT:
                    raise RuntimeError(
                        "concurrently migrated cognitive store fingerprint mismatch"
                    )
                connection.commit()
                return
            if version != 1:
                raise RuntimeError("cognitive store schema changed during v1 migration")
            if _database_schema_fingerprint(connection) != _V1_SCHEMA_FINGERPRINT:
                raise RuntimeError(
                    "cognitive store v1 schema fingerprint changed during migration"
                )
            for statement in (
                segment.strip()
                for segment in _SCHEMA_V2_ADDITION.split(";")
                if segment.strip()
            ):
                connection.execute(statement)
            if self._fault_injector is not None:
                self._fault_injector("before_migration_commit")
            connection.execute(f"PRAGMA user_version = {_V2_SCHEMA_VERSION}")
            if _database_schema_fingerprint(connection) != _V2_SCHEMA_FINGERPRINT:
                raise RuntimeError("migrated cognitive store schema fingerprint mismatch")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def _migrate_v2(self, connection: sqlite3.Connection) -> None:
        """Atomically add the generic acquisition and successor lane."""

        connection.execute("BEGIN IMMEDIATE")
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == _SCHEMA_VERSION:
                if _database_schema_fingerprint(connection) != _SCHEMA_FINGERPRINT:
                    raise RuntimeError(
                        "concurrently migrated cognitive store fingerprint mismatch"
                    )
                connection.commit()
                return
            if version != _V2_SCHEMA_VERSION:
                raise RuntimeError("cognitive store schema changed during v2 migration")
            if _database_schema_fingerprint(connection) != _V2_SCHEMA_FINGERPRINT:
                raise RuntimeError(
                    "cognitive store v2 schema fingerprint changed during migration"
                )
            self._verify_database(
                connection,
                expected_version=_V2_SCHEMA_VERSION,
                expected_fingerprint=_V2_SCHEMA_FINGERPRINT,
                verify_v3=False,
            )
            if connection.execute(
                "SELECT 1 FROM turn_reservations WHERE status != 'RESOLVED' LIMIT 1"
            ).fetchone() is not None:
                raise RuntimeError(
                    "active v2 reservation blocks schema-v3 migration"
                )
            _ensure_v2_backup(self.path, connection)
            if self._fault_injector is not None:
                self._fault_injector("after_v3_backup")
            for statement in (
                segment.strip()
                for segment in _SCHEMA_V3_ADDITION.split(";")
                if segment.strip()
            ):
                connection.execute(statement)
            if self._fault_injector is not None:
                self._fault_injector("after_v3_schema")
            head = self._read_head(connection)
            if head is not None:
                connection.execute(
                    "INSERT INTO acquisition_clock "
                    "(singleton, next_ordinal, acquisition_ref, record_ref) "
                    "VALUES (1, 0, NULL, NULL)"
                )
            if self._fault_injector is not None:
                self._fault_injector("after_v3_clock_initialization")
            if head is not None:
                rows = connection.execute(
                    "SELECT e.sequence, o.acknowledged FROM episodes AS e "
                    "JOIN projection_outbox AS o ON o.sequence = e.sequence "
                    "ORDER BY e.sequence"
                ).fetchall()
                for sequence, acknowledged in rows:
                    stored_ref, episode = self._verified_episode_at(
                        connection, int(sequence)
                    )
                    item = CognitiveEpisodeItem(int(sequence), stored_ref, episode)
                    acquisition_head = self._read_acquisition_head(connection)
                    if acquisition_head is None:
                        raise RuntimeError("v3 migration acquisition clock disappeared")
                    acquisition = CognitiveAcquisition.from_source(
                        episode,
                        ordinal=int(sequence) - 1,
                        predecessor_acquisition_ref=acquisition_head.acquisition_ref,
                        record=episode_record(item),
                    )
                    self._append_acquisition_locked(
                        connection,
                        acquisition,
                        episode,
                        acknowledged=int(acknowledged),
                    )
                    if self._fault_injector is not None:
                        self._fault_injector("during_v3_backfill")
            self._verify_complete_acquisition_chain(connection, head is not None)
            if self._fault_injector is not None:
                self._fault_injector("before_v3_version")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            if _database_schema_fingerprint(connection) != _SCHEMA_FINGERPRINT:
                raise RuntimeError("migrated cognitive store schema fingerprint mismatch")
            if self._fault_injector is not None:
                self._fault_injector("before_v3_migration_commit")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    @staticmethod
    def _head_from_row(row: tuple[object, ...] | None) -> StateHead | None:
        if row is None:
            return None
        return StateHead(
            sequence=int(row[0]),
            episode_ref=None if row[1] is None else str(row[1]),
            state_digest=str(row[2]),
            snapshot_sha256=str(row[3]),
        )

    @staticmethod
    def _read_head(connection: sqlite3.Connection) -> StateHead | None:
        row = connection.execute(
            "SELECT sequence, episode_ref, state_digest, snapshot_sha256 "
            "FROM canonical_head WHERE singleton = 1"
        ).fetchone()
        return CognitiveTransactionStore._head_from_row(row)

    @staticmethod
    def _read_identity(connection: sqlite3.Connection) -> tuple[str, str] | None:
        rows = connection.execute(
            "SELECT model_ref, encoder_ref FROM store_identity ORDER BY singleton LIMIT 2"
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError("cognitive store has multiple identity rows")
        model_ref, encoder_ref = str(rows[0][0]), str(rows[0][1])
        _sha256_ref(model_ref, "stored model_ref")
        _sha256_ref(encoder_ref, "stored encoder_ref")
        return model_ref, encoder_ref

    @staticmethod
    def _read_acquisition_head(
        connection: sqlite3.Connection,
    ) -> AcquisitionHead | None:
        rows = connection.execute(
            "SELECT next_ordinal, acquisition_ref, record_ref "
            "FROM acquisition_clock WHERE singleton = 1 LIMIT 2"
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RuntimeError("cognitive store has multiple acquisition clocks")
        next_ordinal = int(rows[0][0])
        acquisition_ref = None if rows[0][1] is None else str(rows[0][1])
        record_ref = None if rows[0][2] is None else str(rows[0][2])
        if next_ordinal < 0 or (next_ordinal == 0) != (acquisition_ref is None):
            raise RuntimeError("cognitive acquisition clock is malformed")
        if (acquisition_ref is None) != (record_ref is None):
            raise RuntimeError("cognitive acquisition clock references are incomplete")
        if acquisition_ref is not None:
            _sha256_ref(acquisition_ref, "stored acquisition_ref")
            _sha256_ref(record_ref, "stored acquisition record_ref")
        return AcquisitionHead(next_ordinal, acquisition_ref, record_ref)

    @staticmethod
    def _acquisition_source(
        connection: sqlite3.Connection, source_contract: str, source_ref: str
    ) -> object:
        if source_contract == LEGACY_COGNITIVE_EPISODE_CONTRACT:
            row = connection.execute(
                "SELECT canonical_payload FROM episodes WHERE episode_ref = ?",
                (source_ref,),
            ).fetchone()
            parser = CognitiveEpisode.from_json
        elif source_contract == PROSPECTIVE_DYNAMICS_BATCH_CONTRACT:
            row = connection.execute(
                f"SELECT {_PROSPECTIVE_RESERVATION_COLUMNS} FROM prospective_turns_v3 "
                "WHERE batch_ref = ?",
                (source_ref,),
            ).fetchone()
            if row is None:
                raise RuntimeError("stored acquisition source is absent")
            return CognitiveTransactionStore._prospective_reservation_from_row(
                row
            ).reservation.batch
        elif source_contract == PROSPECTIVE_RESOLUTION_CONTRACT:
            row = connection.execute(
                f"SELECT {_PROSPECTIVE_RESERVATION_COLUMNS} FROM prospective_turns_v3 "
                "WHERE resolution_ref = ?",
                (source_ref,),
            ).fetchone()
            if row is None:
                raise RuntimeError("stored acquisition source is absent")
            resolution = CognitiveTransactionStore._prospective_reservation_from_row(
                row
            ).resolution
            if resolution is None:
                raise RuntimeError("stored resolution acquisition source is absent")
            return resolution
        else:
            raise RuntimeError("stored acquisition has an unsupported source contract")
        if row is None or row[0] is None:
            raise RuntimeError("stored acquisition source is absent")
        try:
            return parser(bytes(row[0]))
        except Exception as exc:
            raise RuntimeError("stored acquisition source is not canonical") from exc

    @staticmethod
    def _verified_acquisition_at(
        connection: sqlite3.Connection, ordinal: int
    ) -> tuple[
        CognitiveAcquisitionItem,
        AcquisitionProjectionOutboxItem,
        int,
    ]:
        row = connection.execute(
            "SELECT a.acquisition_ref, a.predecessor_acquisition_ref, "
            "a.source_contract, a.source_ref, a.record_ref, a.payload_sha256, "
            "a.canonical_payload, o.acquisition_ref, o.record_ref, "
            "o.projection_ref, o.payload_sha256, o.canonical_payload, o.acknowledged "
            "FROM cognitive_acquisitions AS a "
            "JOIN acquisition_projection_outbox AS o ON o.ordinal = a.ordinal "
            "WHERE a.ordinal = ?",
            (ordinal,),
        ).fetchone()
        if row is None:
            raise RuntimeError("cognitive acquisition has incomplete local projection")
        (
            acquisition_ref,
            predecessor_ref,
            source_contract,
            source_ref,
            record_ref,
            acquisition_sha256,
            raw_acquisition,
            outbox_acquisition_ref,
            outbox_record_ref,
            projection_ref,
            projection_sha256,
            raw_projection,
            acknowledged,
        ) = row
        acquisition_payload = bytes(raw_acquisition)
        projection_payload = bytes(raw_projection)
        try:
            acquisition = CognitiveAcquisition.from_json(acquisition_payload)
            projection = CognitiveGraphProjectionV2.from_json(projection_payload)
        except Exception as exc:
            raise RuntimeError("stored cognitive acquisition is not canonical") from exc
        expected_predecessor = None
        if ordinal > 0:
            predecessor = connection.execute(
                "SELECT acquisition_ref FROM cognitive_acquisitions WHERE ordinal = ?",
                (ordinal - 1,),
            ).fetchone()
            if predecessor is None:
                raise RuntimeError("cognitive acquisition predecessor is absent")
            expected_predecessor = str(predecessor[0])
        if (
            acquisition.ordinal != ordinal
            or acquisition.acquisition_ref != str(acquisition_ref)
            or _snapshot_digest(acquisition_payload) != str(acquisition_sha256)
            or str(acquisition_ref) != str(acquisition_sha256)
            or acquisition.predecessor_acquisition_ref != predecessor_ref
            or acquisition.predecessor_acquisition_ref != expected_predecessor
            or acquisition.source_contract != str(source_contract)
            or acquisition.source_ref != str(source_ref)
            or acquisition.record.record_ref != str(record_ref)
            or str(outbox_acquisition_ref) != str(acquisition_ref)
            or str(outbox_record_ref) != str(record_ref)
            or projection.projection_ref != str(projection_ref)
            or _snapshot_digest(projection_payload) != str(projection_sha256)
            or str(projection_ref) != str(projection_sha256)
            or int(acknowledged) not in (0, 1)
        ):
            raise RuntimeError("cognitive acquisition row binding mismatch")
        try:
            source = CognitiveTransactionStore._acquisition_source(
                connection, acquisition.source_contract, acquisition.source_ref
            )
            acquisition.assert_source(source)
            projection.assert_acquisition(acquisition)
        except Exception as exc:
            raise RuntimeError(
                "cognitive acquisition source is not canonical or does not bind"
            ) from exc
        item = CognitiveAcquisitionItem(
            ordinal,
            acquisition.acquisition_ref,
            acquisition.record.record_ref,
            acquisition,
        )
        outbox_item = AcquisitionProjectionOutboxItem(
            ordinal,
            acquisition.acquisition_ref,
            acquisition.record.record_ref,
            projection.projection_ref,
            projection,
        )
        return item, outbox_item, int(acknowledged)

    @staticmethod
    def _append_acquisition_locked(
        connection: sqlite3.Connection,
        acquisition: CognitiveAcquisition,
        source: object,
        *,
        acknowledged: int = 0,
    ) -> tuple[CognitiveAcquisitionItem, AcquisitionProjectionOutboxItem]:
        payload = _canonical_contract_bytes(
            acquisition, CognitiveAcquisition, "cognitive acquisition"
        )
        acquisition.assert_source(source)
        if type(acknowledged) is not int or acknowledged not in (0, 1):
            raise ValueError("acquisition projection acknowledgement must be zero or one")
        head = CognitiveTransactionStore._read_acquisition_head(connection)
        if head is None:
            raise RuntimeError("initialized cognitive store has no acquisition clock")
        if (
            acquisition.ordinal != head.next_ordinal
            or acquisition.predecessor_acquisition_ref != head.acquisition_ref
        ):
            raise ValueError("acquisition does not extend the exact acquisition head")
        payload_sha256 = _snapshot_digest(payload)
        if acquisition.acquisition_ref != payload_sha256:
            raise ValueError("acquisition_ref does not identify its canonical bytes")
        projection = CognitiveGraphProjectionV2.from_acquisition(acquisition)
        projection_payload = _canonical_contract_bytes(
            projection, CognitiveGraphProjectionV2, "cognitive graph projection v2"
        )
        projection_sha256 = _snapshot_digest(projection_payload)
        if projection.projection_ref != projection_sha256:
            raise ValueError("projection_ref does not identify its canonical bytes")
        record_ref = acquisition.record.record_ref
        connection.execute(
            "INSERT INTO cognitive_acquisitions "
            "(ordinal, acquisition_ref, predecessor_acquisition_ref, source_contract, "
            "source_ref, record_ref, payload_sha256, canonical_payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                acquisition.ordinal,
                acquisition.acquisition_ref,
                acquisition.predecessor_acquisition_ref,
                acquisition.source_contract,
                acquisition.source_ref,
                record_ref,
                payload_sha256,
                payload,
            ),
        )
        connection.execute(
            "INSERT INTO acquisition_projection_outbox "
            "(ordinal, acquisition_ref, record_ref, projection_ref, payload_sha256, "
            "canonical_payload, acknowledged) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                acquisition.ordinal,
                acquisition.acquisition_ref,
                record_ref,
                projection.projection_ref,
                projection_sha256,
                projection_payload,
                acknowledged,
            ),
        )
        cursor = connection.execute(
            "UPDATE acquisition_clock SET next_ordinal = ?, acquisition_ref = ?, "
            "record_ref = ? WHERE singleton = 1 AND next_ordinal = ?",
            (
                acquisition.ordinal + 1,
                acquisition.acquisition_ref,
                record_ref,
                acquisition.ordinal,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("acquisition clock lost its transactional state")
        return (
            CognitiveAcquisitionItem(
                acquisition.ordinal,
                acquisition.acquisition_ref,
                record_ref,
                acquisition,
            ),
            AcquisitionProjectionOutboxItem(
                acquisition.ordinal,
                acquisition.acquisition_ref,
                record_ref,
                projection.projection_ref,
                projection,
            ),
        )

    @staticmethod
    def _verify_complete_acquisition_chain(
        connection: sqlite3.Connection, initialized: bool
    ) -> AcquisitionHead | None:
        head = CognitiveTransactionStore._read_acquisition_head(connection)
        acquisition_count = int(
            connection.execute("SELECT count(*) FROM cognitive_acquisitions").fetchone()[0]
        )
        outbox_count = int(
            connection.execute(
                "SELECT count(*) FROM acquisition_projection_outbox"
            ).fetchone()[0]
        )
        if not initialized:
            if head is not None or acquisition_count or outbox_count:
                raise RuntimeError("headless store contains acquisition aggregate rows")
            return None
        if head is None:
            raise RuntimeError("initialized cognitive store has no acquisition clock")
        if acquisition_count != outbox_count or head.next_ordinal != acquisition_count:
            raise RuntimeError("cognitive acquisition count/head invariant failed")
        last_item = None
        for ordinal in range(acquisition_count):
            last_item, _projection, _acknowledged = (
                CognitiveTransactionStore._verified_acquisition_at(
                    connection, ordinal
                )
            )
        if last_item is None:
            if head.acquisition_ref is not None or head.record_ref is not None:
                raise RuntimeError("empty acquisition clock names a tail")
        elif (
            head.acquisition_ref != last_item.acquisition_ref
            or head.record_ref != last_item.record_ref
        ):
            raise RuntimeError("acquisition clock does not name the canonical tail")
        return head

    @staticmethod
    def _reservation_from_row(row: tuple[object, ...]) -> ReservationRecord:
        (
            reservation_ref,
            commitment_ref,
            episode_context_ref,
            status,
            parent_sequence,
            parent_event_ref,
            parent_state_digest,
            parent_snapshot_sha256,
            model_ref,
            encoder_ref,
            raw_reservation,
            pending_blob_sha256,
            pending_blob_size,
            raw_pending_blob,
            execution_request_ref,
            raw_execution_request,
            execution_receipt_ref,
            raw_execution_receipt,
            feedback_ref,
            raw_feedback,
            resolution_disposition,
            resolved_episode_ref,
        ) = row
        try:
            reservation_payload = bytes(raw_reservation)
            reservation = ProspectiveTurnReservation.from_json(reservation_payload)
            if (
                reservation.reservation_ref != str(reservation_ref)
                or _snapshot_digest(reservation_payload) != str(reservation_ref)
                or reservation.commitment_ref != str(commitment_ref)
                or _reservation_episode_context_ref(reservation)
                != str(episode_context_ref)
                or reservation.parent_sequence != int(parent_sequence)
                or reservation.parent_event_ref != parent_event_ref
                or reservation.parent_competence_digest != str(parent_state_digest)
                or reservation.parent_snapshot_digest != str(parent_snapshot_sha256)
                or reservation.model_ref != str(model_ref)
                or reservation.encoder_ref != str(encoder_ref)
                or reservation.pending_blob_sha256 != str(pending_blob_sha256)
                or reservation.pending_blob_size != int(pending_blob_size)
            ):
                raise ValueError("reservation row does not bind its canonical payload")

            pending_blob = None if raw_pending_blob is None else bytes(raw_pending_blob)
            if pending_blob is not None:
                reservation.assert_pending_blob(pending_blob)

            request = None
            if raw_execution_request is not None:
                request_payload = bytes(raw_execution_request)
                request = CognitiveExecutionRequest.from_json(request_payload)
                if (
                    request.execution_request_ref != str(execution_request_ref)
                    or _snapshot_digest(request_payload) != str(execution_request_ref)
                ):
                    raise ValueError("execution request row does not bind its payload")
                request.assert_reservation(reservation)
            elif execution_request_ref is not None:
                raise ValueError("execution request reference has no payload")

            receipt = None
            if raw_execution_receipt is not None:
                if request is None:
                    raise ValueError("execution receipt has no request")
                receipt_payload = bytes(raw_execution_receipt)
                receipt = CognitiveExecutionReceipt.from_json(receipt_payload)
                if (
                    receipt.execution_receipt_ref != str(execution_receipt_ref)
                    or _snapshot_digest(receipt_payload) != str(execution_receipt_ref)
                ):
                    raise ValueError("execution receipt row does not bind its payload")
                receipt.assert_request(request)
            elif execution_receipt_ref is not None:
                raise ValueError("execution receipt reference has no payload")

            feedback = None
            if raw_feedback is not None:
                if request is None or receipt is None:
                    raise ValueError("objective feedback has incomplete execution context")
                feedback_payload = bytes(raw_feedback)
                feedback = ObjectiveFeedbackRecord.from_json(feedback_payload)
                if (
                    feedback.feedback_ref != str(feedback_ref)
                    or _snapshot_digest(feedback_payload) != str(feedback_ref)
                ):
                    raise ValueError("objective feedback row does not bind its payload")
                feedback.assert_context(reservation, request, receipt)
            elif feedback_ref is not None:
                raise ValueError("objective feedback reference has no payload")

            status_text = str(status)
            disposition = (
                None if resolution_disposition is None else str(resolution_disposition)
            )
            resolved_ref = None if resolved_episode_ref is None else str(resolved_episode_ref)
            if status_text in _ACTIVE_RESERVATION_STATUSES:
                if pending_blob is None or disposition is not None or resolved_ref is not None:
                    raise ValueError("active reservation row has invalid resolution state")
            elif status_text == "RESOLVED":
                if pending_blob is not None or disposition is None:
                    raise ValueError("resolved reservation row retains pending state")
            else:
                raise ValueError("reservation row has an unknown lifecycle status")
            if status_text == "RESERVED" and any(
                value is not None for value in (request, receipt, feedback)
            ):
                raise ValueError("RESERVED row contains later lifecycle evidence")
            if status_text == "CLAIMED" and (
                request is None or receipt is not None or feedback is not None
            ):
                raise ValueError("CLAIMED row has invalid execution evidence")
            if status_text == "EXECUTION_RECORDED" and (
                request is None or receipt is None
            ):
                raise ValueError("EXECUTION_RECORDED row lacks execution evidence")
            if disposition == "CANCELLED" and any(
                value is not None
                for value in (request, receipt, feedback, resolved_ref)
            ):
                raise ValueError("cancelled reservation contains execution evidence")
            if disposition in _NONCOMPLETION_STATUSES:
                if (
                    request is None
                    or receipt is None
                    or receipt.status != disposition
                    or feedback is not None
                    or resolved_ref is not None
                ):
                    raise ValueError("non-completion disposition does not match receipt")
            if disposition == "EPISODE_COMMITTED" and (
                request is None
                or receipt is None
                or receipt.status != "COMPLETED"
                or feedback is None
                or resolved_ref is None
            ):
                raise ValueError("committed resolution lacks completed execution feedback")
            return ReservationRecord(
                status=status_text,
                reservation=reservation,
                pending_blob=pending_blob,
                execution_request=request,
                execution_receipt=receipt,
                feedback=feedback,
                resolution_disposition=disposition,
                resolved_episode_ref=resolved_ref,
            )
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("stored prospective reservation is not canonical") from exc

    @staticmethod
    def _read_reservation(
        connection: sqlite3.Connection, reservation_ref: str
    ) -> ReservationRecord | None:
        row = connection.execute(
            f"SELECT {_RESERVATION_COLUMNS} FROM turn_reservations "
            "WHERE reservation_ref = ?",
            (reservation_ref,),
        ).fetchone()
        if row is None:
            return None
        return CognitiveTransactionStore._reservation_from_row(row)

    @staticmethod
    def _read_active_reservation(
        connection: sqlite3.Connection, head: StateHead | None
    ) -> ReservationRecord | None:
        rows = connection.execute(
            f"SELECT {_RESERVATION_COLUMNS} FROM turn_reservations "
            "WHERE status != 'RESOLVED' LIMIT 2"
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("cognitive store has multiple active reservations")
        if not rows:
            return None
        record = CognitiveTransactionStore._reservation_from_row(rows[0])
        if head is None:
            raise RuntimeError("headless cognitive store contains an active reservation")
        reservation = record.reservation
        if CognitiveTransactionStore._read_identity(connection) != (
            reservation.model_ref,
            reservation.encoder_ref,
        ):
            raise RuntimeError("active reservation belongs to another store identity")
        if (
            reservation.parent_sequence != head.sequence
            or reservation.parent_event_ref != head.episode_ref
            or reservation.parent_competence_digest != head.state_digest
            or reservation.parent_snapshot_digest != head.snapshot_sha256
        ):
            raise RuntimeError("active reservation does not bind the canonical head")
        return record

    @staticmethod
    def _prospective_reservation_from_row(
        row: tuple[object, ...],
    ) -> ProspectiveReservationRecord:
        (
            reservation_ref,
            legacy_reservation_ref,
            batch_ref,
            parent_lineage_ref,
            selected_branch_ref,
            status,
            parent_sequence,
            parent_event_ref,
            parent_state_digest,
            parent_snapshot_sha256,
            model_ref,
            encoder_ref,
            reservation_payload_sha256,
            raw_reservation,
            raw_batch,
            pending_blob_sha256,
            pending_blob_size,
            raw_pending_blob,
            execution_request_ref,
            raw_execution_request,
            execution_receipt_ref,
            raw_execution_receipt,
            feedback_ref,
            raw_feedback,
            resolution_ref,
            resolution_disposition,
            raw_resolution,
            legacy_episode_ref,
            episode_v2_ref,
            raw_episode_v2,
            batch_acquisition_ref,
            resolution_acquisition_ref,
        ) = row
        try:
            reservation_payload = bytes(raw_reservation)
            reservation = ProspectiveTurnReservationV2.from_json(reservation_payload)
            batch_payload = bytes(raw_batch)
            batch = ProspectiveDynamicsBatch.from_json(batch_payload)
            legacy = reservation.legacy_reservation
            if (
                reservation.reservation_ref != str(reservation_ref)
                or _snapshot_digest(reservation_payload) != str(reservation_ref)
                or str(reservation_payload_sha256) != str(reservation_ref)
                or reservation.legacy_reservation_ref != str(legacy_reservation_ref)
                or reservation.batch_ref != str(batch_ref)
                or batch != reservation.batch
                or batch.canonical_bytes() != batch_payload
                or _snapshot_digest(batch_payload) != str(batch_ref)
                or reservation.parent_lineage_ref != str(parent_lineage_ref)
                or reservation.selected_branch_ref != str(selected_branch_ref)
                or legacy.parent_sequence != int(parent_sequence)
                or legacy.parent_event_ref != parent_event_ref
                or legacy.parent_competence_digest != str(parent_state_digest)
                or legacy.parent_snapshot_digest != str(parent_snapshot_sha256)
                or legacy.model_ref != str(model_ref)
                or legacy.encoder_ref != str(encoder_ref)
                or legacy.pending_blob_sha256 != str(pending_blob_sha256)
                or legacy.pending_blob_size != int(pending_blob_size)
            ):
                raise ValueError("successor reservation row does not bind canonical bytes")

            pending_blob = None if raw_pending_blob is None else bytes(raw_pending_blob)
            if pending_blob is not None:
                legacy.assert_pending_blob(pending_blob)

            request = None
            if raw_execution_request is not None:
                request_payload = bytes(raw_execution_request)
                request = CognitiveExecutionRequest.from_json(request_payload)
                if (
                    request.execution_request_ref != str(execution_request_ref)
                    or _snapshot_digest(request_payload) != str(execution_request_ref)
                ):
                    raise ValueError("successor request row does not bind canonical bytes")
                request.assert_reservation(legacy)
            elif execution_request_ref is not None:
                raise ValueError("successor request reference has no payload")

            receipt = None
            if raw_execution_receipt is not None:
                if request is None:
                    raise ValueError("successor receipt has no request")
                receipt_payload = bytes(raw_execution_receipt)
                receipt = CognitiveExecutionReceipt.from_json(receipt_payload)
                if (
                    receipt.execution_receipt_ref != str(execution_receipt_ref)
                    or _snapshot_digest(receipt_payload) != str(execution_receipt_ref)
                ):
                    raise ValueError("successor receipt row does not bind canonical bytes")
                receipt.assert_request(request)
            elif execution_receipt_ref is not None:
                raise ValueError("successor receipt reference has no payload")

            feedback = None
            if raw_feedback is not None:
                if request is None or receipt is None:
                    raise ValueError("successor feedback lacks execution context")
                feedback_payload = bytes(raw_feedback)
                feedback = ObjectiveFeedbackRecord.from_json(feedback_payload)
                if (
                    feedback.feedback_ref != str(feedback_ref)
                    or _snapshot_digest(feedback_payload) != str(feedback_ref)
                ):
                    raise ValueError("successor feedback row does not bind canonical bytes")
                feedback.assert_context(legacy, request, receipt)
            elif feedback_ref is not None:
                raise ValueError("successor feedback reference has no payload")

            resolution = None
            if raw_resolution is not None:
                resolution_payload = bytes(raw_resolution)
                resolution = ProspectiveResolution.from_json(resolution_payload)
                if (
                    resolution.resolution_ref != str(resolution_ref)
                    or _snapshot_digest(resolution_payload) != str(resolution_ref)
                    or resolution.reservation != reservation
                    or resolution.execution_request != request
                    or resolution.execution_receipt != receipt
                    or resolution.objective_feedback != feedback
                    or resolution.disposition.value != str(resolution_disposition)
                ):
                    raise ValueError("successor resolution row does not bind lifecycle bytes")
            elif resolution_ref is not None or resolution_disposition is not None:
                raise ValueError("successor resolution metadata has no payload")

            episode_v2 = None
            resolved_legacy_ref = (
                None if legacy_episode_ref is None else str(legacy_episode_ref)
            )
            if raw_episode_v2 is not None:
                if resolution is None:
                    raise ValueError("successor Episode 002 has no resolution")
                episode_payload = bytes(raw_episode_v2)
                episode_v2 = CognitiveEpisodeV2.from_json(episode_payload)
                if (
                    episode_v2.episode_ref != str(episode_v2_ref)
                    or _snapshot_digest(episode_payload) != str(episode_v2_ref)
                    or episode_v2.resolution != resolution
                    or episode_v2.legacy_episode.episode_ref != resolved_legacy_ref
                ):
                    raise ValueError("successor Episode 002 row does not bind canonical bytes")
            elif episode_v2_ref is not None:
                raise ValueError("successor Episode 002 reference has no payload")

            status_text = str(status)
            batch_acquisition = _sha256_ref(
                str(batch_acquisition_ref), "stored batch_acquisition_ref"
            )
            resolution_acquisition = (
                None
                if resolution_acquisition_ref is None
                else _sha256_ref(
                    str(resolution_acquisition_ref),
                    "stored resolution_acquisition_ref",
                )
            )
            if status_text in _ACTIVE_RESERVATION_STATUSES:
                if pending_blob is None or resolution is not None:
                    raise ValueError("active successor row has invalid resolution state")
            elif status_text == "RESOLVED":
                if pending_blob is not None or resolution is None or resolution_acquisition is None:
                    raise ValueError("resolved successor row has invalid lifecycle state")
            else:
                raise ValueError("successor row has unknown lifecycle status")
            if status_text == "RESERVED" and any(
                value is not None for value in (request, receipt, feedback)
            ):
                raise ValueError("RESERVED successor row contains later evidence")
            if status_text == "CLAIMED" and (
                request is None or receipt is not None or feedback is not None
            ):
                raise ValueError("CLAIMED successor row has invalid execution evidence")
            if status_text == "EXECUTION_RECORDED" and (
                request is None or receipt is None
            ):
                raise ValueError("EXECUTION_RECORDED successor row lacks receipt evidence")
            disposition = None if resolution is None else resolution.disposition.value
            if disposition is not None and disposition not in _PROSPECTIVE_RESOLUTION_DISPOSITIONS:
                raise ValueError("successor row has unknown resolution disposition")
            if disposition == "OBSERVED":
                if episode_v2 is None or resolved_legacy_ref is None:
                    raise ValueError("observed successor row lacks Episode 002")
            elif episode_v2 is not None or resolved_legacy_ref is not None:
                raise ValueError("non-observed successor row contains an episode")
            return ProspectiveReservationRecord(
                status=status_text,
                reservation=reservation,
                pending_blob=pending_blob,
                execution_request=request,
                execution_receipt=receipt,
                feedback=feedback,
                resolution=resolution,
                episode_v2=episode_v2,
                legacy_episode_ref=resolved_legacy_ref,
                batch_acquisition_ref=batch_acquisition,
                resolution_acquisition_ref=resolution_acquisition,
            )
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("stored successor reservation is not canonical") from exc

    @staticmethod
    def _read_prospective_reservation(
        connection: sqlite3.Connection, reservation_ref: str
    ) -> ProspectiveReservationRecord | None:
        row = connection.execute(
            f"SELECT {_PROSPECTIVE_RESERVATION_COLUMNS} FROM prospective_turns_v3 "
            "WHERE reservation_ref = ?",
            (reservation_ref,),
        ).fetchone()
        return (
            None
            if row is None
            else CognitiveTransactionStore._prospective_reservation_from_row(row)
        )

    @staticmethod
    def _read_prospective_by_legacy_ref(
        connection: sqlite3.Connection, legacy_reservation_ref: str
    ) -> ProspectiveReservationRecord | None:
        row = connection.execute(
            f"SELECT {_PROSPECTIVE_RESERVATION_COLUMNS} FROM prospective_turns_v3 "
            "WHERE legacy_reservation_ref = ?",
            (legacy_reservation_ref,),
        ).fetchone()
        return (
            None
            if row is None
            else CognitiveTransactionStore._prospective_reservation_from_row(row)
        )

    @staticmethod
    def _read_active_prospective_reservation(
        connection: sqlite3.Connection,
        head: StateHead | None,
        acquisition_head: AcquisitionHead | None,
    ) -> ProspectiveReservationRecord | None:
        rows = connection.execute(
            f"SELECT {_PROSPECTIVE_RESERVATION_COLUMNS} FROM prospective_turns_v3 "
            "WHERE status != 'RESOLVED' LIMIT 2"
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("cognitive store has multiple active successor turns")
        if not rows:
            return None
        if head is None or acquisition_head is None:
            raise RuntimeError("headless cognitive store contains an active successor turn")
        record = CognitiveTransactionStore._prospective_reservation_from_row(rows[0])
        legacy = record.reservation.legacy_reservation
        if CognitiveTransactionStore._read_identity(connection) != (
            legacy.model_ref,
            legacy.encoder_ref,
        ):
            raise RuntimeError("active successor turn belongs to another store identity")
        if (
            legacy.parent_sequence != head.sequence
            or legacy.parent_event_ref != head.episode_ref
            or legacy.parent_competence_digest != head.state_digest
            or legacy.parent_snapshot_digest != head.snapshot_sha256
        ):
            raise RuntimeError("active successor turn does not bind the canonical head")
        if acquisition_head.acquisition_ref != record.batch_acquisition_ref:
            raise RuntimeError("active successor turn is not the acquisition tail")
        ordinal_row = connection.execute(
            "SELECT ordinal FROM cognitive_acquisitions WHERE acquisition_ref = ?",
            (record.batch_acquisition_ref,),
        ).fetchone()
        if ordinal_row is None:
            raise RuntimeError("active successor turn has no batch acquisition")
        item, _projection, _acknowledged = (
            CognitiveTransactionStore._verified_acquisition_at(
                connection, int(ordinal_row[0])
            )
        )
        if (
            item.acquisition.source_ref != record.reservation.batch_ref
            or item.acquisition.predecessor_acquisition_ref
            != record.reservation.batch.parent_acquisition_ref
        ):
            raise RuntimeError("active successor batch acquisition binding mismatch")
        return record

    @staticmethod
    def _verify_database(
        connection: sqlite3.Connection,
        *,
        expected_version: int = _SCHEMA_VERSION,
        expected_fingerprint: str = _SCHEMA_FINGERPRINT,
        verify_v3: bool = True,
    ) -> StateHead | None:
        """Fail closed unless the complete canonical aggregate is self-consistent."""

        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != expected_version:
            raise RuntimeError(
                f"unsupported cognitive store schema version {version}"
            )
        if _database_schema_fingerprint(connection) != expected_fingerprint:
            raise RuntimeError("cognitive store schema fingerprint mismatch")
        quick = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check"))
        if quick != ("ok",):
            raise RuntimeError("cognitive store SQLite integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("cognitive store foreign-key integrity check failed")

        head = CognitiveTransactionStore._read_head(connection)
        identity = CognitiveTransactionStore._read_identity(connection)
        states = connection.execute(
            "SELECT sequence, episode_ref, state_digest, snapshot_sha256, snapshot "
            "FROM state_history ORDER BY sequence"
        ).fetchall()
        events = connection.execute(
            "SELECT sequence, episode_ref, parent_state_digest, child_state_digest, "
            "child_snapshot_sha256, payload_sha256, model_ref, encoder_ref, canonical_payload "
            "FROM episodes ORDER BY sequence"
        ).fetchall()
        outbox = connection.execute(
            "SELECT sequence, episode_ref, payload_sha256, canonical_payload, acknowledged "
            "FROM projection_outbox ORDER BY sequence"
        ).fetchall()
        reservation_rows = connection.execute(
            f"SELECT {_RESERVATION_COLUMNS} FROM turn_reservations "
            "ORDER BY rowid"
        ).fetchall()
        prospective_rows = (
            connection.execute(
                f"SELECT {_PROSPECTIVE_RESERVATION_COLUMNS} FROM prospective_turns_v3 "
                "ORDER BY rowid"
            ).fetchall()
            if verify_v3
            else []
        )
        if verify_v3 and connection.execute(
            "SELECT 1 FROM turn_reservations AS legacy "
            "JOIN prospective_turns_v3 AS successor "
            "ON successor.legacy_reservation_ref = legacy.reservation_ref LIMIT 1"
        ).fetchone() is not None:
            raise RuntimeError("one execution idempotency key is consumed in both lanes")
        if head is None:
            if (
                identity is not None
                or states
                or events
                or outbox
                or reservation_rows
                or prospective_rows
            ):
                raise RuntimeError("headless cognitive store contains aggregate rows")
            if verify_v3:
                CognitiveTransactionStore._verify_complete_acquisition_chain(
                    connection, False
                )
            return None
        if identity is None:
            raise RuntimeError("initialized cognitive store has no identity metadata")
        if len(states) != head.sequence + 1 or len(events) != head.sequence:
            raise RuntimeError("cognitive store sequence/count invariant failed")
        if len(outbox) != len(events):
            raise RuntimeError("cognitive store outbox/event count invariant failed")

        parsed_events: dict[int, tuple[str, bytes, CognitiveEpisode]] = {}
        for expected_sequence, row in enumerate(states):
            sequence, episode_ref, state_digest, snapshot_sha256, raw_snapshot = row
            if int(sequence) != expected_sequence:
                raise RuntimeError("cognitive store state sequence is not contiguous")
            snapshot = bytes(raw_snapshot)
            if _snapshot_digest(snapshot) != str(snapshot_sha256):
                raise RuntimeError("cognitive store snapshot digest mismatch")
            if expected_sequence == 0:
                if episode_ref is not None:
                    raise RuntimeError("initial cognitive state cannot name an episode")
                continue
            event = events[expected_sequence - 1]
            (
                event_sequence,
                event_ref,
                parent_digest,
                child_digest,
                child_snapshot_sha256,
                payload_sha256,
                event_model_ref,
                event_encoder_ref,
                raw_payload,
            ) = event
            payload = bytes(raw_payload)
            computed_payload_sha256 = _snapshot_digest(payload)
            if (
                int(event_sequence) != expected_sequence
                or str(event_ref) != str(episode_ref)
                or str(child_digest) != str(state_digest)
                or str(child_snapshot_sha256) != str(snapshot_sha256)
                or str(payload_sha256) != computed_payload_sha256
                or str(event_ref) != computed_payload_sha256
                or (str(event_model_ref), str(event_encoder_ref)) != identity
            ):
                raise RuntimeError("cognitive event/state binding mismatch")
            try:
                parsed = _episode_from_bytes(payload)
            except Exception as exc:
                raise RuntimeError("stored cognitive episode is not canonical") from exc
            previous_state = states[expected_sequence - 1]
            previous_episode_ref = previous_state[1]
            if (
                parsed.episode_ref != str(event_ref)
                or parsed.parent_state_digest != str(parent_digest)
                or parsed.parent_state_digest != str(previous_state[2])
                or parsed.child_state_digest != str(child_digest)
                or parsed.commitment.parent_event_ref != previous_episode_ref
                or parsed.commitment.competence_state_digest
                != parsed.parent_state_digest
                or (parsed.model_ref, parsed.encoder_ref) != identity
            ):
                raise RuntimeError("cognitive episode lineage binding mismatch")
            parsed_events[expected_sequence] = (str(event_ref), payload, parsed)

        last = states[-1]
        if (
            head.sequence != int(last[0])
            or head.episode_ref != last[1]
            or head.state_digest != str(last[2])
            or head.snapshot_sha256 != str(last[3])
        ):
            raise RuntimeError("canonical cognitive head does not match state history")
        for row in outbox:
            sequence, episode_ref, payload_sha256, raw_payload, acknowledged = row
            event = parsed_events.get(int(sequence))
            payload = bytes(raw_payload)
            if (
                event is None
                or str(episode_ref) != event[0]
                or payload != event[1]
                or str(payload_sha256) != _snapshot_digest(payload)
                or int(acknowledged) not in (0, 1)
            ):
                raise RuntimeError("cognitive projection outbox binding mismatch")
        active_count = 0
        for row in reservation_rows:
            record = CognitiveTransactionStore._reservation_from_row(row)
            reservation = record.reservation
            if (reservation.model_ref, reservation.encoder_ref) != identity:
                raise RuntimeError("reservation belongs to another store identity")
            if reservation.parent_sequence >= len(states):
                raise RuntimeError("reservation parent sequence is outside state history")
            parent = states[reservation.parent_sequence]
            if (
                reservation.parent_event_ref != parent[1]
                or reservation.parent_competence_digest != str(parent[2])
                or reservation.parent_snapshot_digest != str(parent[3])
            ):
                raise RuntimeError("reservation parent does not bind state history")
            if record.status in _ACTIVE_RESERVATION_STATUSES:
                active_count += 1
                if (
                    reservation.parent_sequence != head.sequence
                    or reservation.parent_event_ref != head.episode_ref
                    or reservation.parent_competence_digest != head.state_digest
                    or reservation.parent_snapshot_digest != head.snapshot_sha256
                ):
                    raise RuntimeError("active reservation does not bind canonical head")
            if record.resolution_disposition == "EPISODE_COMMITTED":
                resolved = connection.execute(
                    "SELECT sequence, canonical_payload FROM episodes "
                    "WHERE episode_ref = ?",
                    (record.resolved_episode_ref,),
                ).fetchone()
                if resolved is None or int(resolved[0]) != reservation.parent_sequence + 1:
                    raise RuntimeError("resolved reservation does not bind its child episode")
                episode = _episode_from_bytes(bytes(resolved[1]))
                try:
                    CognitiveTransactionStore._assert_episode_reservation_binding(
                        episode, record
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "resolved reservation does not bind its child episode"
                    ) from exc
        if verify_v3:
            acquisition_head = CognitiveTransactionStore._verify_complete_acquisition_chain(
                connection, True
            )
            prospective_records: list[ProspectiveReservationRecord] = []
            for row in prospective_rows:
                record = CognitiveTransactionStore._prospective_reservation_from_row(row)
                prospective_records.append(record)
                legacy = record.reservation.legacy_reservation
                if (legacy.model_ref, legacy.encoder_ref) != identity:
                    raise RuntimeError("successor turn belongs to another store identity")
                if legacy.parent_sequence >= len(states):
                    raise RuntimeError("successor parent sequence is outside state history")
                parent = states[legacy.parent_sequence]
                if (
                    legacy.parent_event_ref != parent[1]
                    or legacy.parent_competence_digest != str(parent[2])
                    or legacy.parent_snapshot_digest != str(parent[3])
                ):
                    raise RuntimeError("successor parent does not bind state history")
                batch_row = connection.execute(
                    "SELECT ordinal FROM cognitive_acquisitions WHERE acquisition_ref = ?",
                    (record.batch_acquisition_ref,),
                ).fetchone()
                if batch_row is None:
                    raise RuntimeError("successor turn lacks its batch acquisition")
                batch_item, _batch_projection, _batch_ack = (
                    CognitiveTransactionStore._verified_acquisition_at(
                        connection, int(batch_row[0])
                    )
                )
                if (
                    batch_item.acquisition.source_ref != record.reservation.batch_ref
                    or batch_item.acquisition.predecessor_acquisition_ref
                    != record.reservation.batch.parent_acquisition_ref
                ):
                    raise RuntimeError("successor batch acquisition does not rejoin")
                if record.status in _ACTIVE_RESERVATION_STATUSES:
                    active_count += 1
                    if (
                        legacy.parent_sequence != head.sequence
                        or legacy.parent_event_ref != head.episode_ref
                        or legacy.parent_competence_digest != head.state_digest
                        or legacy.parent_snapshot_digest != head.snapshot_sha256
                        or acquisition_head is None
                        or acquisition_head.acquisition_ref
                        != record.batch_acquisition_ref
                    ):
                        raise RuntimeError("active successor turn does not bind both heads")
                else:
                    resolution_row = connection.execute(
                        "SELECT ordinal FROM cognitive_acquisitions "
                        "WHERE acquisition_ref = ?",
                        (record.resolution_acquisition_ref,),
                    ).fetchone()
                    if resolution_row is None or record.resolution is None:
                        raise RuntimeError("resolved successor turn lacks acquisition")
                    resolution_item, _resolution_projection, _resolution_ack = (
                        CognitiveTransactionStore._verified_acquisition_at(
                            connection, int(resolution_row[0])
                        )
                    )
                    if resolution_item.acquisition.source_ref != record.resolution.resolution_ref:
                        raise RuntimeError("successor resolution acquisition does not rejoin")
                    if (
                        resolution_item.acquisition.predecessor_acquisition_ref
                        != record.batch_acquisition_ref
                    ):
                        raise RuntimeError(
                            "successor resolution does not immediately follow its batch"
                        )
                    if record.episode_v2 is not None:
                        resolved = connection.execute(
                            "SELECT e.canonical_payload, s.snapshot_sha256 "
                            "FROM episodes AS e JOIN state_history AS s "
                            "ON s.sequence = e.sequence WHERE e.episode_ref = ?",
                            (record.legacy_episode_ref,),
                        ).fetchone()
                        if (
                            resolved is None
                            or bytes(resolved[0])
                            != record.episode_v2.legacy_episode.canonical_bytes()
                            or record.resolution.child_lineage is None
                            or str(resolved[1])
                            != record.resolution.child_lineage.snapshot_digest
                        ):
                            raise RuntimeError(
                                "successor Episode 002 state/snapshot view drifted"
                            )
            observed_episode_refs = {
                record.legacy_episode_ref
                for record in prospective_records
                if record.resolution_disposition == "OBSERVED"
            }
            for event_ref, _payload, _episode in parsed_events.values():
                mirrored = int(
                    connection.execute(
                        "SELECT count(*) FROM cognitive_acquisitions "
                        "WHERE source_contract = ? AND source_ref = ?",
                        (LEGACY_COGNITIVE_EPISODE_CONTRACT, event_ref),
                    ).fetchone()[0]
                )
                expected_mirrors = 0 if event_ref in observed_episode_refs else 1
                if mirrored != expected_mirrors:
                    raise RuntimeError(
                        "legacy episode acquisition mirror invariant failed"
                    )
        if active_count > 1:
            raise RuntimeError("cognitive store has simultaneous active turn lanes")
        return head

    @staticmethod
    def _verify_local_head(connection: sqlite3.Connection) -> StateHead | None:
        """Verify only the indexed aggregate tail used by an ordinary operation."""

        head = CognitiveTransactionStore._read_head(connection)
        identity = CognitiveTransactionStore._read_identity(connection)
        if connection.execute(
            "SELECT 1 FROM turn_reservations AS legacy "
            "JOIN prospective_turns_v3 AS successor "
            "ON successor.legacy_reservation_ref = legacy.reservation_ref LIMIT 1"
        ).fetchone() is not None:
            raise RuntimeError("one execution idempotency key is consumed in both lanes")
        if head is None:
            for table in (
                "store_identity",
                "state_history",
                "episodes",
                "projection_outbox",
                "turn_reservations",
                "cognitive_acquisitions",
                "acquisition_clock",
                "acquisition_projection_outbox",
                "prospective_turns_v3",
            ):
                if connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                    raise RuntimeError("headless cognitive store contains aggregate rows")
            return None
        if identity is None:
            raise RuntimeError("initialized cognitive store has no identity metadata")
        acquisition_head = CognitiveTransactionStore._read_acquisition_head(connection)
        if acquisition_head is None:
            raise RuntimeError("initialized cognitive store has no acquisition clock")
        if acquisition_head.next_ordinal == 0:
            if acquisition_head.acquisition_ref is not None:
                raise RuntimeError("empty acquisition clock names a tail")
        else:
            tail, _projection, _acknowledged = (
                CognitiveTransactionStore._verified_acquisition_at(
                    connection, acquisition_head.next_ordinal - 1
                )
            )
            if (
                tail.acquisition_ref != acquisition_head.acquisition_ref
                or tail.record_ref != acquisition_head.record_ref
            ):
                raise RuntimeError("acquisition clock does not bind its local tail")
        state = connection.execute(
            "SELECT episode_ref, state_digest, snapshot_sha256, snapshot "
            "FROM state_history WHERE sequence = ?",
            (head.sequence,),
        ).fetchone()
        if state is None or (
            state[0] != head.episode_ref
            or str(state[1]) != head.state_digest
            or str(state[2]) != head.snapshot_sha256
            or _snapshot_digest(bytes(state[3])) != head.snapshot_sha256
        ):
            raise RuntimeError("canonical cognitive head does not match its state row")
        if head.sequence == 0:
            if head.episode_ref is not None:
                raise RuntimeError("initial cognitive state cannot name an episode")
        else:
            event_ref, _episode = CognitiveTransactionStore._verified_episode_at(
                connection, head.sequence
            )
            if event_ref != head.episode_ref:
                raise RuntimeError("canonical cognitive tail binding mismatch")
        active_legacy = CognitiveTransactionStore._read_active_reservation(
            connection, head
        )
        active_successor = (
            CognitiveTransactionStore._read_active_prospective_reservation(
                connection, head, acquisition_head
            )
        )
        if active_legacy is not None and active_successor is not None:
            raise RuntimeError("cognitive store has simultaneous active turn lanes")
        return head

    @staticmethod
    def _verified_episode_at(
        connection: sqlite3.Connection, sequence: int
    ) -> tuple[str, CognitiveEpisode]:
        """Verify and parse one indexed episode plus its immediate lineage."""

        row = connection.execute(
            "SELECT e.episode_ref, e.parent_state_digest, e.child_state_digest, "
            "e.child_snapshot_sha256, e.payload_sha256, e.model_ref, e.encoder_ref, "
            "e.canonical_payload, "
            "s.episode_ref, s.state_digest, s.snapshot_sha256, "
            "o.episode_ref, o.payload_sha256, o.canonical_payload "
            "FROM episodes AS e "
            "JOIN state_history AS s ON s.sequence = e.sequence "
            "JOIN projection_outbox AS o ON o.sequence = e.sequence "
            "WHERE e.sequence = ?",
            (sequence,),
        ).fetchone()
        previous = connection.execute(
            "SELECT episode_ref, state_digest FROM state_history WHERE sequence = ?",
            (sequence - 1,),
        ).fetchone()
        if row is None or previous is None:
            raise RuntimeError("cognitive episode has incomplete local lineage")
        identity = CognitiveTransactionStore._read_identity(connection)
        if identity is None:
            raise RuntimeError("initialized cognitive store has no identity metadata")
        (
            episode_ref,
            parent_digest,
            child_digest,
            child_snapshot_sha256,
            payload_sha256,
            event_model_ref,
            event_encoder_ref,
            raw_payload,
            state_episode_ref,
            state_digest,
            state_snapshot_sha256,
            outbox_episode_ref,
            outbox_payload_sha256,
            outbox_payload,
        ) = row
        payload = bytes(raw_payload)
        computed = _snapshot_digest(payload)
        if (
            str(episode_ref) != computed
            or str(payload_sha256) != computed
            or (str(event_model_ref), str(event_encoder_ref)) != identity
            or str(state_episode_ref) != str(episode_ref)
            or str(state_digest) != str(child_digest)
            or str(state_snapshot_sha256) != str(child_snapshot_sha256)
            or str(outbox_episode_ref) != str(episode_ref)
            or str(outbox_payload_sha256) != computed
            or bytes(outbox_payload) != payload
        ):
            raise RuntimeError("cognitive episode local binding mismatch")
        try:
            episode = _episode_from_bytes(payload)
        except Exception as exc:
            raise RuntimeError("stored cognitive episode is not canonical") from exc
        if (
            episode.episode_ref != str(episode_ref)
            or episode.parent_state_digest != str(parent_digest)
            or episode.parent_state_digest != str(previous[1])
            or episode.child_state_digest != str(child_digest)
            or episode.commitment.parent_event_ref != previous[0]
            or episode.commitment.competence_state_digest
            != episode.parent_state_digest
            or (episode.model_ref, episode.encoder_ref) != identity
        ):
            raise RuntimeError("cognitive episode local lineage mismatch")
        return str(episode_ref), episode

    @staticmethod
    def _assert_episode_reservation_binding(
        episode: CognitiveEpisode, record: ReservationRecord
    ) -> None:
        request = record.execution_request
        receipt = record.execution_receipt
        feedback = record.feedback
        reservation = record.reservation
        if request is None or receipt is None or feedback is None:
            raise ValueError("reserved episode requires request, receipt, and feedback")
        if receipt.status != "COMPLETED":
            raise ValueError("only a completed execution can commit an episode")
        expected = (
            episode.task_id == reservation.task_id
            and episode.request == reservation.request
            and episode.recalled_refs == reservation.recalled_refs
            and episode.proposals == reservation.proposals
            and episode.selected_index == reservation.selected_index
            and episode.commitment == reservation.commitment
            and episode.commitment.commitment_ref == reservation.commitment_ref
            and episode.response == receipt.response
            and episode.observations
            == (*request.input_observations, *receipt.output_observations)
            and episode.outcome == feedback.outcome
            and episode.feedback_text == feedback.feedback_text
            and episode.feedback_source_ref == feedback.feedback_source_ref
            and episode.parent_state_digest == reservation.parent_competence_digest
            and episode.model_ref == reservation.model_ref
            and episode.encoder_ref == reservation.encoder_ref
            and episode.supporting_evidence_refs
            == reservation.supporting_evidence_refs
        )
        if not expected:
            raise ValueError("episode does not exactly bind the reserved execution")

    @staticmethod
    def _append_episode_locked(
        connection: sqlite3.Connection,
        episode: CognitiveEpisode,
        state_snapshot: bytes,
        *,
        expected_parent_digest: str,
        current: StateHead,
        mirror_acquisition: bool = True,
    ) -> TransactionCommit:
        snapshot_sha256 = _snapshot_digest(state_snapshot)
        payload = _canonical_episode_bytes(episode)
        episode_ref = _nonempty_text(episode.episode_ref, "episode_ref")
        payload_sha256 = _snapshot_digest(payload)
        if episode_ref != payload_sha256:
            raise ValueError("episode_ref does not identify the canonical episode payload")
        parent_digest = _nonempty_text(
            episode.parent_state_digest, "parent_state_digest"
        )
        child_digest = _nonempty_text(episode.child_state_digest, "child_state_digest")
        if episode.commitment.competence_state_digest != parent_digest:
            raise ValueError("commitment competence state does not match episode parent state")
        identity = CognitiveTransactionStore._read_identity(connection)
        if identity is None:
            raise RuntimeError("initialized cognitive store has no identity metadata")
        if connection.execute(
            "SELECT 1 FROM episodes WHERE episode_ref = ?", (episode_ref,)
        ).fetchone() is not None:
            raise ValueError("duplicate episode_ref")
        if parent_digest != expected_parent_digest:
            raise ValueError("episode parent_state_digest does not match expected parent")
        if current.state_digest != expected_parent_digest:
            raise ValueError("optimistic parent state digest mismatch")
        if episode.commitment.parent_event_ref != current.episode_ref:
            raise ValueError("prospective commitment does not bind the current event head")
        if (episode.model_ref, episode.encoder_ref) != identity:
            raise ValueError(
                "episode belongs to another model or encoder than the store identity"
            )
        sequence = current.sequence + 1
        connection.execute(
            "INSERT INTO state_history "
            "(sequence, episode_ref, state_digest, snapshot_sha256, snapshot) "
            "VALUES (?, ?, ?, ?, ?)",
            (sequence, episode_ref, child_digest, snapshot_sha256, state_snapshot),
        )
        connection.execute(
            "INSERT INTO episodes "
            "(episode_ref, sequence, parent_state_digest, child_state_digest, "
            "child_snapshot_sha256, payload_sha256, model_ref, encoder_ref, "
            "canonical_payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_ref,
                sequence,
                parent_digest,
                child_digest,
                snapshot_sha256,
                payload_sha256,
                episode.model_ref,
                episode.encoder_ref,
                payload,
            ),
        )
        connection.execute(
            "UPDATE canonical_head SET sequence = ?, episode_ref = ?, "
            "state_digest = ?, snapshot_sha256 = ? WHERE singleton = 1",
            (sequence, episode_ref, child_digest, snapshot_sha256),
        )
        connection.execute(
            "INSERT INTO projection_outbox "
            "(sequence, episode_ref, payload_sha256, canonical_payload, acknowledged) "
            "VALUES (?, ?, ?, ?, 0)",
            (sequence, episode_ref, payload_sha256, payload),
        )
        if mirror_acquisition:
            acquisition_head = CognitiveTransactionStore._read_acquisition_head(
                connection
            )
            if acquisition_head is None:
                raise RuntimeError("initialized cognitive store has no acquisition clock")
            record = replace(
                episode_record(CognitiveEpisodeItem(sequence, episode_ref, episode)),
                acquired_ordinal=acquisition_head.next_ordinal,
            )
            acquisition = CognitiveAcquisition.from_source(
                episode,
                ordinal=acquisition_head.next_ordinal,
                predecessor_acquisition_ref=acquisition_head.acquisition_ref,
                record=record,
            )
            CognitiveTransactionStore._append_acquisition_locked(
                connection, acquisition, episode
            )
        head = StateHead(sequence, episode_ref, child_digest, snapshot_sha256)
        return TransactionCommit(episode_ref, sequence, head, True)

    def audit_integrity(self) -> StateHead | None:
        """Explicitly run the unbounded full-history restart audit."""

        with closing(self._connect()) as connection, _read_snapshot(connection):
            return self._verify_database(connection)

    def head(self) -> StateHead | None:
        with closing(self._connect()) as connection, _read_snapshot(connection):
            return self._verify_local_head(connection)

    def initialize(
        self,
        state_digest: str,
        state_snapshot: bytes,
        *,
        model_ref: str,
        encoder_ref: str,
    ) -> StateHead:
        """Create sequence zero, or verify an exact idempotent restart."""

        state_digest = _nonempty_text(state_digest, "state_digest")
        model_ref = _sha256_ref(model_ref, "model_ref")
        encoder_ref = _sha256_ref(encoder_ref, "encoder_ref")
        snapshot = _snapshot_bytes(state_snapshot)
        snapshot_sha256 = _snapshot_digest(snapshot)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._verify_local_head(connection)
                if current is not None:
                    if self._read_identity(connection) != (model_ref, encoder_ref):
                        raise ValueError("cognitive store belongs to another model or encoder")
                    stored = connection.execute(
                        "SELECT state_digest, snapshot_sha256, snapshot "
                        "FROM state_history WHERE sequence = 0",
                    ).fetchone()
                    if (
                        stored is None
                        or str(stored[0]) != state_digest
                        or str(stored[1]) != snapshot_sha256
                        or bytes(stored[2]) != snapshot
                    ):
                        raise ValueError("store is already initialized with different state")
                    connection.commit()
                    return current
                connection.execute(
                    "INSERT INTO store_identity (singleton, model_ref, encoder_ref) "
                    "VALUES (1, ?, ?)",
                    (model_ref, encoder_ref),
                )
                connection.execute(
                    "INSERT INTO state_history "
                    "(sequence, episode_ref, state_digest, snapshot_sha256, snapshot) "
                    "VALUES (0, NULL, ?, ?, ?)",
                    (state_digest, snapshot_sha256, snapshot),
                )
                connection.execute(
                    "INSERT INTO canonical_head "
                    "(singleton, sequence, episode_ref, state_digest, snapshot_sha256) "
                    "VALUES (1, 0, NULL, ?, ?)",
                    (state_digest, snapshot_sha256),
                )
                connection.execute(
                    "INSERT INTO acquisition_clock "
                    "(singleton, next_ordinal, acquisition_ref, record_ref) "
                    "VALUES (1, 0, NULL, NULL)"
                )
                head = StateHead(0, None, state_digest, snapshot_sha256)
                connection.commit()
                return head
            except BaseException:
                connection.rollback()
                raise

    def assert_compatibility(self, model_ref: str, encoder_ref: str) -> None:
        """Reject a restart under a different frozen model or encoder."""

        expected = (
            _sha256_ref(model_ref, "model_ref"),
            _sha256_ref(encoder_ref, "encoder_ref"),
        )
        with closing(self._connect()) as connection, _read_snapshot(connection):
            head = self._verify_local_head(connection)
            if head is None:
                raise RuntimeError("cognitive store is not initialized")
            if self._read_identity(connection) != expected:
                raise ValueError("cognitive store belongs to another model or encoder")

    def load_head_state(self) -> bytes | None:
        with closing(self._connect()) as connection, _read_snapshot(connection):
            head = self._verify_local_head(connection)
            if head is None:
                return None
            row = connection.execute(
                "SELECT snapshot FROM state_history WHERE sequence = ?",
                (head.sequence,),
            ).fetchone()
            if row is None:
                raise RuntimeError("canonical head has no matching state snapshot")
            snapshot = bytes(row[0])
            if _snapshot_digest(snapshot) != head.snapshot_sha256:
                raise RuntimeError("canonical head snapshot digest mismatch")
            return snapshot

    def active_reservation(self) -> ReservationRecord | None:
        """Return the sole verified active reservation, if one exists."""

        with closing(self._connect()) as connection, _read_snapshot(connection):
            head = self._verify_local_head(connection)
            return self._read_active_reservation(connection, head)

    def get_reservation(self, reservation_ref: str) -> ReservationRecord:
        reservation_ref = _sha256_ref(reservation_ref, "reservation_ref")
        with closing(self._connect()) as connection, _read_snapshot(connection):
            self._verify_local_head(connection)
            record = self._read_reservation(connection, reservation_ref)
            if record is None:
                raise KeyError(reservation_ref)
            return record

    def active_prospective_turn(self) -> ProspectiveReservationRecord | None:
        """Return the sole verified active successor turn, if present."""

        with closing(self._connect()) as connection, _read_snapshot(connection):
            head = self._verify_local_head(connection)
            acquisition_head = self._read_acquisition_head(connection)
            return self._read_active_prospective_reservation(
                connection, head, acquisition_head
            )

    def get_prospective_turn(
        self, reservation_ref: str
    ) -> ProspectiveReservationRecord:
        reservation_ref = _sha256_ref(reservation_ref, "reservation_ref")
        with closing(self._connect()) as connection, _read_snapshot(connection):
            self._verify_local_head(connection)
            record = self._read_prospective_reservation(connection, reservation_ref)
            if record is None:
                record = self._read_prospective_by_legacy_ref(
                    connection, reservation_ref
                )
            if record is None:
                raise KeyError(reservation_ref)
            return record

    def prospective_turn_for_episode(
        self, legacy_episode_ref: str
    ) -> ProspectiveReservationRecord | None:
        """Return the verified resolved successor aggregate, if one exists."""

        legacy_episode_ref = _sha256_ref(legacy_episode_ref, "legacy_episode_ref")
        with closing(self._connect()) as connection, _read_snapshot(connection):
            self._verify_local_head(connection)
            row = connection.execute(
                "SELECT reservation_ref FROM prospective_turns_v3 "
                "WHERE legacy_episode_ref = ?",
                (legacy_episode_ref,),
            ).fetchone()
            if row is None:
                return None
            record = self._read_prospective_reservation(connection, str(row[0]))
            if record is None or record.legacy_episode_ref != legacy_episode_ref:
                raise RuntimeError("successor episode lookup did not rejoin exactly")
            return record

    def reserve_prospective_turn(
        self,
        reservation_v2: ProspectiveTurnReservationV2,
        batch_acquisition: CognitiveAcquisition,
        pending_blob: bytes,
    ) -> ProspectiveReservationTransition:
        """Atomically persist a successor batch and proposed acquisition."""

        reservation_payload = _canonical_contract_bytes(
            reservation_v2,
            ProspectiveTurnReservationV2,
            "prospective reservation v2",
        )
        batch_payload = _canonical_contract_bytes(
            reservation_v2.batch,
            ProspectiveDynamicsBatch,
            "prospective dynamics batch",
        )
        acquisition_payload = _canonical_contract_bytes(
            batch_acquisition,
            CognitiveAcquisition,
            "batch acquisition",
        )
        blob = _pending_blob_bytes(pending_blob)
        legacy = reservation_v2.legacy_reservation
        legacy.assert_pending_blob(blob)
        batch_acquisition.assert_source(reservation_v2.batch)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                if head is None:
                    raise RuntimeError("store must be initialized before reserving a turn")
                acquisition_head = self._read_acquisition_head(connection)
                if acquisition_head is None:
                    raise RuntimeError("initialized cognitive store has no acquisition clock")
                if connection.execute(
                    "SELECT 1 FROM turn_reservations WHERE reservation_ref = ?",
                    (reservation_v2.legacy_reservation_ref,),
                ).fetchone() is not None:
                    raise ValueError(
                        "legacy reservation idempotency key was already consumed"
                    )
                existing = self._read_prospective_by_legacy_ref(
                    connection, reservation_v2.legacy_reservation_ref
                )
                if existing is not None:
                    if (
                        existing.reservation.canonical_bytes() != reservation_payload
                        or existing.batch_acquisition_ref
                        != batch_acquisition.acquisition_ref
                        or (
                            existing.pending_blob is not None
                            and existing.pending_blob != blob
                        )
                    ):
                        raise ValueError(
                            "legacy reservation ref already identifies different successor bytes"
                        )
                    ordinal = connection.execute(
                        "SELECT ordinal FROM cognitive_acquisitions "
                        "WHERE acquisition_ref = ?",
                        (existing.batch_acquisition_ref,),
                    ).fetchone()
                    if ordinal is None:
                        raise RuntimeError("successor reservation lost its batch acquisition")
                    item, _projection, _acknowledged = self._verified_acquisition_at(
                        connection, int(ordinal[0])
                    )
                    if item.acquisition.canonical_bytes() != acquisition_payload:
                        raise ValueError("batch acquisition replay differs from stored bytes")
                    connection.commit()
                    return ProspectiveReservationTransition(existing, False)
                if self._read_active_reservation(connection, head) is not None:
                    raise ValueError("another prospective turn is already active")
                if self._read_active_prospective_reservation(
                    connection, head, acquisition_head
                ) is not None:
                    raise ValueError("another prospective turn is already active")
                if self._read_identity(connection) != (
                    legacy.model_ref,
                    legacy.encoder_ref,
                ):
                    raise ValueError("successor reservation belongs to another store identity")
                if (
                    legacy.parent_sequence != head.sequence
                    or legacy.parent_event_ref != head.episode_ref
                    or legacy.parent_competence_digest != head.state_digest
                    or legacy.parent_snapshot_digest != head.snapshot_sha256
                ):
                    raise ValueError("successor reservation does not bind the canonical head")
                if reservation_v2.batch.parent_acquisition_ref != acquisition_head.acquisition_ref:
                    raise ValueError("prospective batch does not bind the acquisition head")
                if (
                    batch_acquisition.ordinal != acquisition_head.next_ordinal
                    or batch_acquisition.predecessor_acquisition_ref
                    != acquisition_head.acquisition_ref
                ):
                    raise ValueError("batch acquisition does not extend the acquisition head")
                self._append_acquisition_locked(
                    connection,
                    batch_acquisition,
                    reservation_v2.batch,
                )
                connection.execute(
                    "INSERT INTO prospective_turns_v3 ("
                    "reservation_ref, legacy_reservation_ref, batch_ref, "
                    "parent_lineage_ref, selected_branch_ref, status, parent_sequence, "
                    "parent_event_ref, parent_state_digest, parent_snapshot_sha256, "
                    "model_ref, encoder_ref, reservation_payload_sha256, "
                    "canonical_reservation, canonical_batch, pending_blob_sha256, "
                    "pending_blob_size, pending_blob, batch_acquisition_ref) "
                    "VALUES (?, ?, ?, ?, ?, 'RESERVED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        reservation_v2.reservation_ref,
                        reservation_v2.legacy_reservation_ref,
                        reservation_v2.batch_ref,
                        reservation_v2.parent_lineage_ref,
                        reservation_v2.selected_branch_ref,
                        legacy.parent_sequence,
                        legacy.parent_event_ref,
                        legacy.parent_competence_digest,
                        legacy.parent_snapshot_digest,
                        legacy.model_ref,
                        legacy.encoder_ref,
                        _snapshot_digest(reservation_payload),
                        reservation_payload,
                        batch_payload,
                        legacy.pending_blob_sha256,
                        legacy.pending_blob_size,
                        blob,
                        batch_acquisition.acquisition_ref,
                    ),
                )
                record = self._read_prospective_reservation(
                    connection, reservation_v2.reservation_ref
                )
                if record is None:
                    raise RuntimeError("successor reservation insert was not readable")
                if self._fault_injector is not None:
                    self._fault_injector("before_prospective_reservation_commit")
                connection.commit()
                return ProspectiveReservationTransition(record, True)
            except BaseException:
                connection.rollback()
                raise

    def claim_prospective_turn(
        self, request: CognitiveExecutionRequest
    ) -> ProspectiveReservationTransition:
        payload = _canonical_contract_bytes(
            request, CognitiveExecutionRequest, "execution request"
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                acquisition_head = self._read_acquisition_head(connection)
                record = self._read_prospective_by_legacy_ref(
                    connection, request.reservation_ref
                )
                if record is None:
                    raise KeyError(request.reservation_ref)
                request.assert_reservation(record.reservation.legacy_reservation)
                if record.execution_request is not None:
                    if record.execution_request.canonical_bytes() != payload:
                        raise ValueError("successor turn carries different request bytes")
                    connection.commit()
                    return ProspectiveReservationTransition(record, False)
                if record.status != "RESERVED":
                    raise ValueError("only a RESERVED successor turn can be claimed")
                active = self._read_active_prospective_reservation(
                    connection, head, acquisition_head
                )
                if active is None or active.reservation_ref != record.reservation_ref:
                    raise RuntimeError("claim target is not the active successor turn")
                cursor = connection.execute(
                    "UPDATE prospective_turns_v3 SET status = 'CLAIMED', "
                    "execution_request_ref = ?, canonical_execution_request = ? "
                    "WHERE reservation_ref = ? AND status = 'RESERVED'",
                    (request.execution_request_ref, payload, record.reservation_ref),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("successor claim lost its transactional state")
                updated = self._read_prospective_reservation(
                    connection, record.reservation_ref
                )
                if updated is None:
                    raise RuntimeError("claimed successor turn disappeared")
                if self._fault_injector is not None:
                    self._fault_injector("before_prospective_claim_commit")
                connection.commit()
                return ProspectiveReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def record_prospective_execution(
        self, receipt: CognitiveExecutionReceipt
    ) -> ProspectiveReservationTransition:
        payload = _canonical_contract_bytes(
            receipt, CognitiveExecutionReceipt, "execution receipt"
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                acquisition_head = self._read_acquisition_head(connection)
                row = connection.execute(
                    "SELECT reservation_ref FROM prospective_turns_v3 "
                    "WHERE execution_request_ref = ?",
                    (receipt.execution_request_ref,),
                ).fetchone()
                if row is None:
                    raise KeyError(receipt.execution_request_ref)
                record = self._read_prospective_reservation(connection, str(row[0]))
                if record is None or record.execution_request is None:
                    raise RuntimeError("successor receipt has no claimed turn")
                receipt.assert_request(record.execution_request)
                if record.execution_receipt is not None:
                    if record.execution_receipt.canonical_bytes() != payload:
                        raise ValueError("successor request carries different receipt bytes")
                    connection.commit()
                    return ProspectiveReservationTransition(record, False)
                if record.status != "CLAIMED":
                    raise ValueError("only a CLAIMED successor turn can record execution")
                active = self._read_active_prospective_reservation(
                    connection, head, acquisition_head
                )
                if active is None or active.reservation_ref != record.reservation_ref:
                    raise RuntimeError("receipt target is not the active successor turn")
                cursor = connection.execute(
                    "UPDATE prospective_turns_v3 SET status = 'EXECUTION_RECORDED', "
                    "execution_receipt_ref = ?, canonical_execution_receipt = ? "
                    "WHERE reservation_ref = ? AND status = 'CLAIMED'",
                    (
                        receipt.execution_receipt_ref,
                        payload,
                        record.reservation_ref,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("successor execution lost its transactional state")
                updated = self._read_prospective_reservation(
                    connection, record.reservation_ref
                )
                if updated is None:
                    raise RuntimeError("recorded successor turn disappeared")
                if self._fault_injector is not None:
                    self._fault_injector("before_prospective_execution_commit")
                connection.commit()
                return ProspectiveReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def stage_prospective_feedback(
        self, feedback: ObjectiveFeedbackRecord
    ) -> ProspectiveReservationTransition:
        payload = _canonical_contract_bytes(
            feedback, ObjectiveFeedbackRecord, "objective feedback"
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                acquisition_head = self._read_acquisition_head(connection)
                record = self._read_prospective_by_legacy_ref(
                    connection, feedback.reservation_ref
                )
                if record is None:
                    raise KeyError(feedback.reservation_ref)
                if record.execution_request is None or record.execution_receipt is None:
                    raise ValueError("successor feedback requires recorded execution")
                feedback.assert_context(
                    record.reservation.legacy_reservation,
                    record.execution_request,
                    record.execution_receipt,
                )
                if record.execution_receipt.status != "COMPLETED":
                    raise ValueError("non-completion cannot stage successor feedback")
                if record.feedback is not None:
                    if record.feedback.canonical_bytes() != payload:
                        raise ValueError("successor turn carries different feedback bytes")
                    connection.commit()
                    return ProspectiveReservationTransition(record, False)
                if record.status != "EXECUTION_RECORDED":
                    raise ValueError("only a recorded successor execution can stage feedback")
                active = self._read_active_prospective_reservation(
                    connection, head, acquisition_head
                )
                if active is None or active.reservation_ref != record.reservation_ref:
                    raise RuntimeError("feedback target is not the active successor turn")
                cursor = connection.execute(
                    "UPDATE prospective_turns_v3 SET feedback_ref = ?, "
                    "canonical_feedback = ? WHERE reservation_ref = ? "
                    "AND status = 'EXECUTION_RECORDED'",
                    (feedback.feedback_ref, payload, record.reservation_ref),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("successor feedback lost its transactional state")
                updated = self._read_prospective_reservation(
                    connection, record.reservation_ref
                )
                if updated is None:
                    raise RuntimeError("feedback successor turn disappeared")
                if self._fault_injector is not None:
                    self._fault_injector("before_prospective_feedback_commit")
                connection.commit()
                return ProspectiveReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def resolve_prospective_turn(
        self,
        resolution: ProspectiveResolution,
        resolution_acquisition: CognitiveAcquisition,
    ) -> ProspectiveReservationTransition:
        """Atomically append one non-observed lifecycle resolution acquisition."""

        resolution_payload = _canonical_contract_bytes(
            resolution, ProspectiveResolution, "prospective resolution"
        )
        acquisition_payload = _canonical_contract_bytes(
            resolution_acquisition,
            CognitiveAcquisition,
            "resolution acquisition",
        )
        if resolution.disposition is ResolutionDisposition.OBSERVED:
            raise ValueError("OBSERVED resolution requires the observed commit path")
        resolution_acquisition.assert_source(resolution)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                acquisition_head = self._read_acquisition_head(connection)
                if head is None or acquisition_head is None:
                    raise RuntimeError("store must be initialized before resolving a turn")
                record = self._read_prospective_by_legacy_ref(
                    connection, resolution.reservation.legacy_reservation_ref
                )
                if record is None:
                    raise KeyError(resolution.reservation.legacy_reservation_ref)
                if record.reservation != resolution.reservation:
                    raise ValueError("resolution names different successor reservation bytes")
                if record.status == "RESOLVED":
                    if (
                        record.resolution is None
                        or record.resolution.canonical_bytes() != resolution_payload
                        or record.resolution_acquisition_ref
                        != resolution_acquisition.acquisition_ref
                    ):
                        raise ValueError("successor turn already resolved another way")
                    row = connection.execute(
                        "SELECT ordinal FROM cognitive_acquisitions "
                        "WHERE acquisition_ref = ?",
                        (record.resolution_acquisition_ref,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("resolved successor turn lost its acquisition")
                    item, _projection, _acknowledged = self._verified_acquisition_at(
                        connection, int(row[0])
                    )
                    if item.acquisition.canonical_bytes() != acquisition_payload:
                        raise ValueError("resolution acquisition replay differs")
                    connection.commit()
                    return ProspectiveReservationTransition(record, False)
                if (
                    resolution.execution_request != record.execution_request
                    or resolution.execution_receipt != record.execution_receipt
                    or resolution.objective_feedback != record.feedback
                ):
                    raise ValueError("resolution does not exactly bind stored lifecycle bytes")
                if resolution.disposition is ResolutionDisposition.CANCELLED:
                    if record.status != "RESERVED":
                        raise ValueError("only an unclaimed successor turn can be cancelled")
                elif record.status != "EXECUTION_RECORDED":
                    raise ValueError("recorded lifecycle resolution requires execution")
                active = self._read_active_prospective_reservation(
                    connection, head, acquisition_head
                )
                if active is None or active.reservation_ref != record.reservation_ref:
                    raise RuntimeError("resolution target is not the active successor turn")
                if (
                    resolution_acquisition.ordinal != acquisition_head.next_ordinal
                    or resolution_acquisition.predecessor_acquisition_ref
                    != acquisition_head.acquisition_ref
                ):
                    raise ValueError("resolution acquisition does not extend the clock")
                self._append_acquisition_locked(
                    connection, resolution_acquisition, resolution
                )
                cursor = connection.execute(
                    "UPDATE prospective_turns_v3 SET status = 'RESOLVED', "
                    "pending_blob = NULL, resolution_ref = ?, "
                    "resolution_disposition = ?, canonical_resolution = ?, "
                    "resolution_acquisition_ref = ? WHERE reservation_ref = ? "
                    "AND status != 'RESOLVED'",
                    (
                        resolution.resolution_ref,
                        resolution.disposition.value,
                        resolution_payload,
                        resolution_acquisition.acquisition_ref,
                        record.reservation_ref,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("successor resolution lost its transactional state")
                updated = self._read_prospective_reservation(
                    connection, record.reservation_ref
                )
                if updated is None:
                    raise RuntimeError("resolved successor turn disappeared")
                if self._fault_injector is not None:
                    self._fault_injector("before_prospective_resolution_commit")
                connection.commit()
                return ProspectiveReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def commit_observed_prospective_turn(
        self,
        resolution: ProspectiveResolution,
        episode_v2: CognitiveEpisodeV2,
        resolution_acquisition: CognitiveAcquisition,
        child_snapshot: bytes,
        expected_parent_digest: str,
    ) -> TransactionCommit:
        """Atomically commit observed legacy/state and successor records."""

        resolution_payload = _canonical_contract_bytes(
            resolution, ProspectiveResolution, "prospective resolution"
        )
        episode_v2_payload = _canonical_contract_bytes(
            episode_v2, CognitiveEpisodeV2, "cognitive episode v2"
        )
        acquisition_payload = _canonical_contract_bytes(
            resolution_acquisition,
            CognitiveAcquisition,
            "resolution acquisition",
        )
        snapshot = _snapshot_bytes(child_snapshot)
        expected = _nonempty_text(expected_parent_digest, "expected_parent_digest")
        if resolution.disposition is not ResolutionDisposition.OBSERVED:
            raise ValueError("observed commit requires an OBSERVED resolution")
        if episode_v2.resolution != resolution:
            raise ValueError("Episode 002 does not contain the exact resolution")
        resolution_acquisition.assert_source(resolution)
        child_lineage = resolution.child_lineage
        if child_lineage is None or _snapshot_digest(snapshot) != child_lineage.snapshot_digest:
            raise ValueError("child snapshot does not bind the observed situated lineage")
        legacy_episode = episode_v2.legacy_episode
        legacy_episode_payload = _canonical_episode_bytes(legacy_episode)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._verify_local_head(connection)
                acquisition_head = self._read_acquisition_head(connection)
                if current is None or acquisition_head is None:
                    raise RuntimeError("store must be initialized before observed commit")
                record = self._read_prospective_by_legacy_ref(
                    connection, resolution.reservation.legacy_reservation_ref
                )
                if record is None:
                    raise KeyError(resolution.reservation.legacy_reservation_ref)
                if record.reservation != resolution.reservation:
                    raise ValueError("observed resolution names different reservation bytes")
                if record.status == "RESOLVED":
                    if (
                        record.resolution is None
                        or record.resolution.canonical_bytes() != resolution_payload
                        or record.episode_v2 is None
                        or record.episode_v2.canonical_bytes() != episode_v2_payload
                        or record.resolution_acquisition_ref
                        != resolution_acquisition.acquisition_ref
                    ):
                        raise ValueError("successor turn already resolved another way")
                    stored = connection.execute(
                        "SELECT e.sequence, e.canonical_payload, s.state_digest, "
                        "s.snapshot_sha256, s.snapshot, o.acknowledged "
                        "FROM episodes AS e JOIN state_history AS s "
                        "ON s.sequence = e.sequence JOIN projection_outbox AS o "
                        "ON o.sequence = e.sequence WHERE e.episode_ref = ?",
                        (legacy_episode.episode_ref,),
                    ).fetchone()
                    ordinal = connection.execute(
                        "SELECT ordinal FROM cognitive_acquisitions "
                        "WHERE acquisition_ref = ?",
                        (record.resolution_acquisition_ref,),
                    ).fetchone()
                    if (
                        stored is None
                        or ordinal is None
                        or bytes(stored[1]) != legacy_episode_payload
                        or bytes(stored[4]) != snapshot
                        or legacy_episode.parent_state_digest != expected
                    ):
                        raise ValueError("observed successor replay differs from stored bytes")
                    item, _projection, _acknowledged = self._verified_acquisition_at(
                        connection, int(ordinal[0])
                    )
                    if item.acquisition.canonical_bytes() != acquisition_payload:
                        raise ValueError("observed acquisition replay differs")
                    committed_head = StateHead(
                        int(stored[0]),
                        legacy_episode.episode_ref,
                        str(stored[2]),
                        str(stored[3]),
                    )
                    connection.commit()
                    return TransactionCommit(
                        legacy_episode.episode_ref,
                        int(stored[0]),
                        committed_head,
                        int(stored[5]) == 0,
                    )
                if (
                    record.status != "EXECUTION_RECORDED"
                    or record.feedback is None
                    or resolution.execution_request != record.execution_request
                    or resolution.execution_receipt != record.execution_receipt
                    or resolution.objective_feedback != record.feedback
                ):
                    raise ValueError("observed commit lacks exact staged lifecycle evidence")
                active = self._read_active_prospective_reservation(
                    connection, current, acquisition_head
                )
                if active is None or active.reservation_ref != record.reservation_ref:
                    raise RuntimeError("observed commit target is not active")
                if expected != record.reservation.legacy_reservation.parent_competence_digest:
                    raise ValueError("expected parent does not match successor reservation")
                if (
                    resolution_acquisition.ordinal != acquisition_head.next_ordinal
                    or resolution_acquisition.predecessor_acquisition_ref
                    != acquisition_head.acquisition_ref
                ):
                    raise ValueError("observed resolution acquisition does not extend clock")
                result = self._append_episode_locked(
                    connection,
                    legacy_episode,
                    snapshot,
                    expected_parent_digest=expected,
                    current=current,
                    mirror_acquisition=False,
                )
                self._append_acquisition_locked(
                    connection, resolution_acquisition, resolution
                )
                cursor = connection.execute(
                    "UPDATE prospective_turns_v3 SET status = 'RESOLVED', "
                    "pending_blob = NULL, resolution_ref = ?, "
                    "resolution_disposition = 'OBSERVED', canonical_resolution = ?, "
                    "legacy_episode_ref = ?, episode_v2_ref = ?, "
                    "canonical_episode_v2 = ?, resolution_acquisition_ref = ? "
                    "WHERE reservation_ref = ? AND status = 'EXECUTION_RECORDED'",
                    (
                        resolution.resolution_ref,
                        resolution_payload,
                        legacy_episode.episode_ref,
                        episode_v2.episode_ref,
                        episode_v2_payload,
                        resolution_acquisition.acquisition_ref,
                        record.reservation_ref,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("observed successor commit lost its state")
                updated = self._read_prospective_reservation(
                    connection, record.reservation_ref
                )
                if updated is None or updated.episode_v2 != episode_v2:
                    raise RuntimeError("observed successor commit did not rejoin")
                if self._fault_injector is not None:
                    self._fault_injector("before_observed_prospective_commit")
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def reserve_turn(
        self,
        reservation: ProspectiveTurnReservation,
        pending_blob: bytes,
    ) -> ReservationTransition:
        """Durably reserve one exact decision before it can be executed."""

        payload = _canonical_contract_bytes(
            reservation, ProspectiveTurnReservation, "reservation"
        )
        blob = _pending_blob_bytes(pending_blob)
        reservation.assert_pending_blob(blob)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                if head is None:
                    raise RuntimeError("store must be initialized before reserving a turn")
                if connection.execute(
                    "SELECT 1 FROM prospective_turns_v3 "
                    "WHERE legacy_reservation_ref = ?",
                    (reservation.reservation_ref,),
                ).fetchone() is not None:
                    raise ValueError(
                        "reservation idempotency key was already consumed by successor lane"
                    )
                existing = self._read_reservation(
                    connection, reservation.reservation_ref
                )
                if existing is not None:
                    if (
                        existing.reservation.canonical_bytes() != payload
                        or (
                            existing.pending_blob is not None
                            and existing.pending_blob != blob
                        )
                    ):
                        raise ValueError(
                            "reservation_ref already identifies different reservation bytes"
                        )
                    connection.commit()
                    return ReservationTransition(existing, False)
                if self._read_active_reservation(connection, head) is not None:
                    raise ValueError("another prospective turn is already active")
                acquisition_head = self._read_acquisition_head(connection)
                if self._read_active_prospective_reservation(
                    connection, head, acquisition_head
                ) is not None:
                    raise ValueError("another prospective turn is already active")
                identity = self._read_identity(connection)
                if identity != (reservation.model_ref, reservation.encoder_ref):
                    raise ValueError("reservation belongs to another model or encoder")
                if (
                    reservation.parent_sequence != head.sequence
                    or reservation.parent_event_ref != head.episode_ref
                    or reservation.parent_competence_digest != head.state_digest
                    or reservation.parent_snapshot_digest != head.snapshot_sha256
                ):
                    raise ValueError("reservation does not bind the canonical head")
                connection.execute(
                    "INSERT INTO turn_reservations ("
                    "reservation_ref, commitment_ref, episode_context_ref, status, parent_sequence, "
                    "parent_event_ref, parent_state_digest, parent_snapshot_sha256, "
                    "model_ref, encoder_ref, canonical_reservation, pending_blob_sha256, "
                    "pending_blob_size, pending_blob) "
                    "VALUES (?, ?, ?, 'RESERVED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        reservation.reservation_ref,
                        reservation.commitment_ref,
                        _reservation_episode_context_ref(reservation),
                        reservation.parent_sequence,
                        reservation.parent_event_ref,
                        reservation.parent_competence_digest,
                        reservation.parent_snapshot_digest,
                        reservation.model_ref,
                        reservation.encoder_ref,
                        payload,
                        reservation.pending_blob_sha256,
                        reservation.pending_blob_size,
                        blob,
                    ),
                )
                if self._fault_injector is not None:
                    self._fault_injector("before_reservation_commit")
                record = self._read_reservation(
                    connection, reservation.reservation_ref
                )
                if record is None:
                    raise RuntimeError("reservation insert was not readable")
                connection.commit()
                return ReservationTransition(record, True)
            except BaseException:
                connection.rollback()
                raise

    def claim_turn(
        self, request: CognitiveExecutionRequest
    ) -> ReservationTransition:
        """Persist the exact request and report whether this call won the claim."""

        payload = _canonical_contract_bytes(
            request, CognitiveExecutionRequest, "execution request"
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                record = self._read_reservation(connection, request.reservation_ref)
                if record is None:
                    raise KeyError(request.reservation_ref)
                request.assert_reservation(record.reservation)
                if record.execution_request is not None:
                    if record.execution_request.canonical_bytes() != payload:
                        raise ValueError(
                            "reservation already carries different execution request bytes"
                        )
                    connection.commit()
                    return ReservationTransition(record, False)
                if record.status != "RESERVED":
                    raise ValueError("only a RESERVED turn can be claimed")
                active = self._read_active_reservation(connection, head)
                if active is None or active.reservation_ref != record.reservation_ref:
                    raise RuntimeError("claim target is not the active reservation")
                connection.execute(
                    "UPDATE turn_reservations SET status = 'CLAIMED', "
                    "execution_request_ref = ?, canonical_execution_request = ? "
                    "WHERE reservation_ref = ? AND status = 'RESERVED'",
                    (
                        request.execution_request_ref,
                        payload,
                        request.reservation_ref,
                    ),
                )
                if connection.total_changes != 1:
                    raise RuntimeError("reservation claim lost its transactional state")
                if self._fault_injector is not None:
                    self._fault_injector("before_claim_commit")
                updated = self._read_reservation(connection, request.reservation_ref)
                if updated is None:
                    raise RuntimeError("claimed reservation disappeared")
                connection.commit()
                return ReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def record_execution(
        self, receipt: CognitiveExecutionReceipt
    ) -> ReservationTransition:
        """Persist an executor receipt before exposing its result to learning."""

        payload = _canonical_contract_bytes(
            receipt, CognitiveExecutionReceipt, "execution receipt"
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                row = connection.execute(
                    "SELECT reservation_ref FROM turn_reservations "
                    "WHERE execution_request_ref = ?",
                    (receipt.execution_request_ref,),
                ).fetchone()
                if row is None:
                    raise KeyError(receipt.execution_request_ref)
                reservation_ref = str(row[0])
                record = self._read_reservation(connection, reservation_ref)
                if record is None or record.execution_request is None:
                    raise RuntimeError("execution receipt has no claimed reservation")
                receipt.assert_request(record.execution_request)
                if record.execution_receipt is not None:
                    if record.execution_receipt.canonical_bytes() != payload:
                        raise ValueError(
                            "execution request already carries different receipt bytes"
                        )
                    connection.commit()
                    return ReservationTransition(record, False)
                if record.status != "CLAIMED":
                    raise ValueError("only a CLAIMED turn can record execution")
                active = self._read_active_reservation(connection, head)
                if active is None or active.reservation_ref != reservation_ref:
                    raise RuntimeError("receipt target is not the active reservation")
                connection.execute(
                    "UPDATE turn_reservations SET status = 'EXECUTION_RECORDED', "
                    "execution_receipt_ref = ?, canonical_execution_receipt = ? "
                    "WHERE reservation_ref = ? AND status = 'CLAIMED'",
                    (receipt.execution_receipt_ref, payload, reservation_ref),
                )
                if connection.total_changes != 1:
                    raise RuntimeError("execution recording lost its transactional state")
                if self._fault_injector is not None:
                    self._fault_injector("before_execution_record_commit")
                updated = self._read_reservation(connection, reservation_ref)
                if updated is None:
                    raise RuntimeError("recorded reservation disappeared")
                connection.commit()
                return ReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def stage_feedback(
        self, feedback: ObjectiveFeedbackRecord
    ) -> ReservationTransition:
        """Persist objective feedback without advancing competence state."""

        payload = _canonical_contract_bytes(
            feedback, ObjectiveFeedbackRecord, "objective feedback"
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = self._verify_local_head(connection)
                record = self._read_reservation(connection, feedback.reservation_ref)
                if record is None:
                    raise KeyError(feedback.reservation_ref)
                if record.execution_request is None or record.execution_receipt is None:
                    raise ValueError("feedback requires a recorded execution")
                feedback.assert_context(
                    record.reservation,
                    record.execution_request,
                    record.execution_receipt,
                )
                if record.execution_receipt.status != "COMPLETED":
                    raise ValueError("non-completion receipts cannot stage learned feedback")
                if record.feedback is not None:
                    if record.feedback.canonical_bytes() != payload:
                        raise ValueError(
                            "reservation already carries different feedback bytes"
                        )
                    connection.commit()
                    return ReservationTransition(record, False)
                if record.status != "EXECUTION_RECORDED":
                    raise ValueError(
                        "only an EXECUTION_RECORDED turn can stage feedback"
                    )
                active = self._read_active_reservation(connection, head)
                if active is None or active.reservation_ref != record.reservation_ref:
                    raise RuntimeError("feedback target is not the active reservation")
                connection.execute(
                    "UPDATE turn_reservations SET feedback_ref = ?, "
                    "canonical_feedback = ? WHERE reservation_ref = ? "
                    "AND status = 'EXECUTION_RECORDED'",
                    (feedback.feedback_ref, payload, feedback.reservation_ref),
                )
                if connection.total_changes != 1:
                    raise RuntimeError("feedback staging lost its transactional state")
                if self._fault_injector is not None:
                    self._fault_injector("before_feedback_commit")
                updated = self._read_reservation(connection, feedback.reservation_ref)
                if updated is None:
                    raise RuntimeError("feedback reservation disappeared")
                connection.commit()
                return ReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def cancel_reservation(self, reservation_ref: str) -> ReservationTransition:
        """Cancel only a never-claimed reservation."""

        reservation_ref = _sha256_ref(reservation_ref, "reservation_ref")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._verify_local_head(connection)
                record = self._read_reservation(connection, reservation_ref)
                if record is None:
                    raise KeyError(reservation_ref)
                if record.status == "RESOLVED":
                    if record.resolution_disposition != "CANCELLED":
                        raise ValueError("reservation already resolved another way")
                    connection.commit()
                    return ReservationTransition(record, False)
                if record.status != "RESERVED":
                    raise ValueError("a claimed reservation cannot be cancelled")
                connection.execute(
                    "UPDATE turn_reservations SET status = 'RESOLVED', "
                    "pending_blob = NULL, resolution_disposition = 'CANCELLED' "
                    "WHERE reservation_ref = ? AND status = 'RESERVED'",
                    (reservation_ref,),
                )
                if connection.total_changes != 1:
                    raise RuntimeError("reservation cancellation lost its state")
                if self._fault_injector is not None:
                    self._fault_injector("before_cancellation_commit")
                updated = self._read_reservation(connection, reservation_ref)
                if updated is None:
                    raise RuntimeError("cancelled reservation disappeared")
                connection.commit()
                return ReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def resolve_noncompletion(self, reservation_ref: str) -> ReservationTransition:
        """Resolve a recorded clarification/error without an episode or update."""

        reservation_ref = _sha256_ref(reservation_ref, "reservation_ref")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._verify_local_head(connection)
                record = self._read_reservation(connection, reservation_ref)
                if record is None:
                    raise KeyError(reservation_ref)
                if record.status == "RESOLVED":
                    if record.resolution_disposition not in _NONCOMPLETION_STATUSES:
                        raise ValueError("reservation already resolved another way")
                    connection.commit()
                    return ReservationTransition(record, False)
                if record.status != "EXECUTION_RECORDED" or record.execution_receipt is None:
                    raise ValueError("non-completion requires a recorded execution")
                disposition = record.execution_receipt.status
                if disposition not in _NONCOMPLETION_STATUSES:
                    raise ValueError("completed execution cannot resolve as non-completion")
                if record.feedback is not None:
                    raise ValueError("non-completion cannot discard staged feedback")
                connection.execute(
                    "UPDATE turn_reservations SET status = 'RESOLVED', "
                    "pending_blob = NULL, resolution_disposition = ? "
                    "WHERE reservation_ref = ? AND status = 'EXECUTION_RECORDED'",
                    (disposition, reservation_ref),
                )
                if connection.total_changes != 1:
                    raise RuntimeError("non-completion resolution lost its state")
                if self._fault_injector is not None:
                    self._fault_injector("before_noncompletion_commit")
                updated = self._read_reservation(connection, reservation_ref)
                if updated is None:
                    raise RuntimeError("resolved reservation disappeared")
                connection.commit()
                return ReservationTransition(updated, True)
            except BaseException:
                connection.rollback()
                raise

    def commit_reserved_episode(
        self,
        reservation_ref: str,
        episode: CognitiveEpisode,
        state_snapshot: bytes,
        *,
        expected_parent_digest: str,
    ) -> TransactionCommit:
        """Atomically append episode/state/outbox and resolve its reservation."""

        reservation_ref = _sha256_ref(reservation_ref, "reservation_ref")
        expected = _nonempty_text(expected_parent_digest, "expected_parent_digest")
        snapshot = _snapshot_bytes(state_snapshot)
        episode_payload = _canonical_episode_bytes(episode)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._verify_local_head(connection)
                if current is None:
                    raise RuntimeError("store must be initialized before committing an episode")
                record = self._read_reservation(connection, reservation_ref)
                if record is None:
                    raise KeyError(reservation_ref)
                self._assert_episode_reservation_binding(episode, record)
                if record.status == "RESOLVED":
                    if (
                        record.resolution_disposition != "EPISODE_COMMITTED"
                        or record.resolved_episode_ref != episode.episode_ref
                    ):
                        raise ValueError("reservation already resolved another way")
                    stored = connection.execute(
                        "SELECT e.sequence, e.canonical_payload, s.state_digest, "
                        "s.snapshot_sha256, s.snapshot, o.acknowledged "
                        "FROM episodes AS e JOIN state_history AS s "
                        "ON s.sequence = e.sequence JOIN projection_outbox AS o "
                        "ON o.sequence = e.sequence WHERE e.episode_ref = ?",
                        (episode.episode_ref,),
                    ).fetchone()
                    if (
                        stored is None
                        or bytes(stored[1]) != episode_payload
                        or bytes(stored[4]) != snapshot
                        or episode.parent_state_digest != expected
                    ):
                        raise ValueError(
                            "resolved reservation replay differs from committed bytes"
                        )
                    head = StateHead(
                        int(stored[0]),
                        episode.episode_ref,
                        str(stored[2]),
                        str(stored[3]),
                    )
                    connection.commit()
                    return TransactionCommit(
                        episode.episode_ref,
                        int(stored[0]),
                        head,
                        int(stored[5]) == 0,
                    )
                if record.status != "EXECUTION_RECORDED" or record.feedback is None:
                    raise ValueError(
                        "reserved episode requires recorded execution and staged feedback"
                    )
                active = self._read_active_reservation(connection, current)
                if active is None or active.reservation_ref != reservation_ref:
                    raise RuntimeError("commit target is not the active reservation")
                if expected != record.reservation.parent_competence_digest:
                    raise ValueError("expected parent does not match reservation parent")
                result = self._append_episode_locked(
                    connection,
                    episode,
                    snapshot,
                    expected_parent_digest=expected,
                    current=current,
                )
                connection.execute(
                    "UPDATE turn_reservations SET status = 'RESOLVED', "
                    "pending_blob = NULL, resolution_disposition = 'EPISODE_COMMITTED', "
                    "resolved_episode_ref = ? WHERE reservation_ref = ? "
                    "AND status = 'EXECUTION_RECORDED'",
                    (episode.episode_ref, reservation_ref),
                )
                if connection.total_changes != 8:
                    # Four legacy writes, three acquisition writes, and resolution.
                    raise RuntimeError("reserved episode transaction lost its state")
                if self._fault_injector is not None:
                    self._fault_injector("before_reserved_commit")
                    # Preserve the accepted v1 transaction fault boundary for
                    # callers that test every episode commit uniformly.
                    self._fault_injector("before_commit")
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def commit_episode(
        self,
        episode: CognitiveEpisode,
        state_snapshot: bytes,
        *,
        expected_parent_digest: str,
    ) -> TransactionCommit:
        """Append one episode and advance state in one SQLite transaction."""

        expected = _nonempty_text(expected_parent_digest, "expected_parent_digest")
        snapshot = _snapshot_bytes(state_snapshot)
        _canonical_episode_bytes(episode)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._verify_local_head(connection)
                if current is None:
                    raise RuntimeError("store must be initialized before committing an episode")
                if self._read_active_reservation(connection, current) is not None:
                    raise ValueError(
                        "an active reservation must resolve through its atomic commit path"
                    )
                acquisition_head = self._read_acquisition_head(connection)
                if self._read_active_prospective_reservation(
                    connection, current, acquisition_head
                ) is not None:
                    raise ValueError(
                        "an active successor turn must resolve through its atomic commit path"
                    )
                if connection.execute(
                    "SELECT 1 FROM turn_reservations "
                    "WHERE episode_context_ref = ? LIMIT 1",
                    (_cognitive_episode_context_ref(episode),),
                ).fetchone() is not None:
                    raise ValueError(
                        "an exact reserved context cannot use the legacy episode commit path"
                    )
                result = self._append_episode_locked(
                    connection,
                    episode,
                    snapshot,
                    expected_parent_digest=expected,
                    current=current,
                )
                if self._fault_injector is not None:
                    self._fault_injector("before_commit")
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def get_episode(self, episode_ref: str) -> CognitiveEpisode:
        return self.get_episode_item(episode_ref).episode

    def get_episode_item(self, episode_ref: str) -> CognitiveEpisodeItem:
        """Return one exact canonical episode together with its store sequence."""

        episode_ref = _sha256_ref(episode_ref, "episode_ref")
        with closing(self._connect()) as connection, _read_snapshot(connection):
            row = connection.execute(
                "SELECT sequence FROM episodes WHERE episode_ref = ?",
                (episode_ref,),
            ).fetchone()
            if row is None:
                raise KeyError(episode_ref)
            stored_ref, episode = self._verified_episode_at(connection, int(row[0]))
            if stored_ref != episode_ref:
                raise RuntimeError("stored episode payload does not match its reference")
            return CognitiveEpisodeItem(int(row[0]), stored_ref, episode)

    def episode_items(
        self, after_sequence: int = 0, limit: int = 64
    ) -> tuple[CognitiveEpisodeItem, ...]:
        """Return an ordered, bounded canonical page after ``after_sequence``."""

        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        with closing(self._connect()) as connection, _read_snapshot(connection):
            rows = connection.execute(
                "SELECT sequence FROM episodes WHERE sequence > ? "
                "ORDER BY sequence LIMIT ?",
                (after_sequence, limit),
            ).fetchall()
            items = []
            for (sequence,) in rows:
                stored_ref, episode = self._verified_episode_at(
                    connection, int(sequence)
                )
                items.append(CognitiveEpisodeItem(int(sequence), stored_ref, episode))
            return tuple(items)

    def acquisition_head(self) -> AcquisitionHead:
        """Return the verified canonical store-side acquisition clock."""

        with closing(self._connect()) as connection, _read_snapshot(connection):
            competence_head = self._verify_local_head(connection)
            if competence_head is None:
                raise RuntimeError("cognitive store is not initialized")
            head = self._read_acquisition_head(connection)
            if head is None:
                raise RuntimeError("initialized cognitive store has no acquisition clock")
            return head

    def get_acquisition_item(self, record_ref: str) -> CognitiveAcquisitionItem:
        """Return one exact canonical acquisition by its record reference."""

        record_ref = _sha256_ref(record_ref, "record_ref")
        with closing(self._connect()) as connection, _read_snapshot(connection):
            self._verify_local_head(connection)
            row = connection.execute(
                "SELECT ordinal FROM cognitive_acquisitions WHERE record_ref = ?",
                (record_ref,),
            ).fetchone()
            if row is None:
                raise KeyError(record_ref)
            item, _projection, _acknowledged = self._verified_acquisition_at(
                connection, int(row[0])
            )
            if item.record_ref != record_ref:
                raise RuntimeError("acquisition lookup record reference drifted")
            return item

    def acquisition_items(
        self, after_ordinal: int = -1, limit: int = 64
    ) -> tuple[CognitiveAcquisitionItem, ...]:
        """Return an ordered bounded page after an exclusive ordinal."""

        if type(after_ordinal) is not int or after_ordinal < -1:
            raise ValueError("after_ordinal must be an integer of at least -1")
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        with closing(self._connect()) as connection, _read_snapshot(connection):
            self._verify_local_head(connection)
            rows = connection.execute(
                "SELECT ordinal FROM cognitive_acquisitions WHERE ordinal > ? "
                "ORDER BY ordinal LIMIT ?",
                (after_ordinal, limit),
            ).fetchall()
            items = []
            for (ordinal,) in rows:
                item, _projection, _acknowledged = self._verified_acquisition_at(
                    connection, int(ordinal)
                )
                items.append(item)
            return tuple(items)

    def pending_acquisition_projections(
        self, limit: int = 64
    ) -> tuple[AcquisitionProjectionOutboxItem, ...]:
        """Return the oldest bounded page of unacknowledged generic projections."""

        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        query = (
            "SELECT ordinal FROM acquisition_projection_outbox "
            "WHERE acknowledged = 0 ORDER BY ordinal LIMIT ?"
        )
        with closing(self._connect()) as connection, _read_snapshot(connection):
            self._verify_local_head(connection)
            rows = connection.execute(query, (limit,)).fetchall()
            items = []
            for (ordinal,) in rows:
                _item, projection, acknowledged = self._verified_acquisition_at(
                    connection, int(ordinal)
                )
                if acknowledged != 0:
                    raise RuntimeError("pending acquisition projection is acknowledged")
                items.append(projection)
            return tuple(items)

    def ack_acquisition_projection(self, projection_ref: str) -> None:
        """Idempotently acknowledge one exact generic projection reference."""

        projection_ref = _sha256_ref(projection_ref, "projection_ref")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._verify_local_head(connection)
                row = connection.execute(
                    "SELECT ordinal, acknowledged FROM acquisition_projection_outbox "
                    "WHERE projection_ref = ?",
                    (projection_ref,),
                ).fetchone()
                if row is None:
                    raise KeyError(projection_ref)
                _item, projection, acknowledged = self._verified_acquisition_at(
                    connection, int(row[0])
                )
                if projection.projection_ref != projection_ref:
                    raise RuntimeError("projection acknowledgement reference drifted")
                if int(row[1]) != acknowledged:
                    raise RuntimeError("projection acknowledgement state drifted")
                if acknowledged == 0:
                    cursor = connection.execute(
                        "UPDATE acquisition_projection_outbox SET acknowledged = 1 "
                        "WHERE projection_ref = ? AND acknowledged = 0",
                        (projection_ref,),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("projection acknowledgement lost its state")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def pending_projections(self, limit: int = 64) -> tuple[ProjectionOutboxItem, ...]:
        """Return one ordered, bounded page of unacknowledged projections."""

        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        query = (
            "SELECT sequence, episode_ref FROM projection_outbox "
            "WHERE acknowledged = 0 ORDER BY sequence LIMIT ?"
        )
        with closing(self._connect()) as connection, _read_snapshot(connection):
            rows = connection.execute(query, (limit,)).fetchall()
            items = []
            for sequence, episode_ref in rows:
                stored_ref, episode = self._verified_episode_at(connection, int(sequence))
                if stored_ref != str(episode_ref):
                    raise RuntimeError("outbox episode does not match its reference")
                items.append(ProjectionOutboxItem(int(sequence), stored_ref, episode))
            return tuple(items)

    def ack_projection(self, episode_ref: str) -> None:
        """Idempotently mark a rebuildable projection as complete."""

        episode_ref = _nonempty_text(episode_ref, "episode_ref")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT sequence, acknowledged FROM projection_outbox WHERE episode_ref = ?",
                    (episode_ref,),
                ).fetchone()
                if row is None:
                    raise KeyError(episode_ref)
                stored_ref, _episode = self._verified_episode_at(connection, int(row[0]))
                if stored_ref != episode_ref:
                    raise RuntimeError("outbox episode does not match acknowledgement reference")
                if int(row[1]) == 0:
                    connection.execute(
                        "UPDATE projection_outbox SET acknowledged = 1 WHERE episode_ref = ?",
                        (episode_ref,),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise


__all__ = [
    "AcquisitionHead",
    "AcquisitionProjectionOutboxItem",
    "CognitiveAcquisitionItem",
    "CognitiveEpisodeItem",
    "CognitiveTransactionStore",
    "ProjectionOutboxItem",
    "ProspectiveReservationRecord",
    "ProspectiveReservationTransition",
    "ReservationRecord",
    "ReservationTransition",
    "StateHead",
    "TransactionCommit",
    "restore_pre_v3_backup",
]
