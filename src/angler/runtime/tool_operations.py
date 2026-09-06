"""Durable two-phase tool-operation records for Jenny 2.0.

This module is intentionally only a transaction and provenance boundary.  It
does not execute tools, score outcomes, update learned state, project Cognee
records, or advance Moving Origin.  Execution belongs to a separately
permissioned adapter which may append observations to this store.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
from typing import Iterator, Literal


TOOL_DEFINITION_CONTRACT = "ANG-CTR-JENNY-TOOL-DEFINITION-001@0.1.0"
TOOL_MANIFEST_CONTRACT = "ANG-CTR-JENNY-TOOL-MANIFEST-001@0.1.0"
TOOL_CALL_RESERVATION_CONTRACT = "ANG-CTR-JENNY-TOOL-CALL-001@0.1.0"
TOOL_EVENT_CONTRACT = "ANG-CTR-JENNY-TOOL-EVENT-001@0.1.0"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._:/-][a-z0-9]+)*$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$")
_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TERMINAL_STATUSES = frozenset(("COMPLETED", "ERROR", "DENIED"))
_PRIVATE_SQLITE_MODE = 0o600


def _open_private_regular(
    path: Path, *, create: bool, allow_missing: bool = False
) -> int | None:
    """Open one current-user regular file without following its final path.

    The returned descriptor pins the checked inode. Owned files are narrowed
    to mode 0600 through that descriptor; foreign, linked, and non-regular
    targets fail closed.
    """

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None or not Path("/proc/self/fd").is_dir():
        raise RuntimeError("private SQLite requires Linux no-follow descriptor support")
    flags |= nofollow
    attempts = 0
    while True:
        attempts += 1
        try:
            descriptor = os.open(
                path,
                flags | (os.O_CREAT | os.O_EXCL if create else 0),
                _PRIVATE_SQLITE_MODE,
            )
        except FileExistsError:
            if attempts >= 3:
                raise RuntimeError("private SQLite path changed repeatedly during open")
            create = False
            continue
        except FileNotFoundError:
            if allow_missing:
                return None
            if not create and attempts < 3:
                create = True
                continue
            raise RuntimeError("private SQLite path disappeared during open")
        except OSError as exc:
            raise RuntimeError("private SQLite path cannot be opened safely") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("private SQLite path is not a regular file")
            if metadata.st_uid != os.getuid():
                raise RuntimeError("private SQLite path ownership differs")
            if stat.S_IMODE(metadata.st_mode) != _PRIVATE_SQLITE_MODE:
                os.fchmod(descriptor, _PRIVATE_SQLITE_MODE)
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != _PRIVATE_SQLITE_MODE:
                    raise RuntimeError("private SQLite mode could not be narrowed to 0600")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise


def _ensure_private_sqlite_sidecars(database_path: Path) -> None:
    """Validate or narrow SQLite's live WAL/SHM files when present."""

    for suffix in ("-wal", "-shm"):
        descriptor = _open_private_regular(
            Path(f"{database_path}{suffix}"), create=False, allow_missing=True
        )
        if descriptor is not None:
            os.close(descriptor)


def _connect_private_sqlite(
    database_path: Path, *, timeout: float = 30.0
) -> sqlite3.Connection:
    """Connect through a checked inode rather than reopening a mutable path."""

    _ensure_private_sqlite_sidecars(database_path)
    descriptor = _open_private_regular(database_path, create=True)
    if descriptor is None:  # pragma: no cover - create=True cannot return None
        raise RuntimeError("private SQLite descriptor is absent")
    try:
        connection = sqlite3.connect(
            f"file:/proc/self/fd/{descriptor}?mode=rw",
            timeout=timeout,
            uri=True,
        )
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)
    try:
        # An established WAL database may create its shared files while opening.
        # SQLite derives new sidecar permissions from the now-0600 main file;
        # this second check also narrows any already-owned legacy sidecars.
        connection.execute("PRAGMA schema_version").fetchone()
        _ensure_private_sqlite_sidecars(database_path)
    except BaseException:
        connection.close()
        raise
    return connection


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _bounded_text(
    value: str,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be text")
    if len(value.encode("utf-8")) > maximum or (
        not allow_empty and not value.strip()
    ):
        raise ValueError(f"{label} must be bounded text")
    return value


def _identifier(value: str, label: str) -> str:
    if type(value) is not str or len(value) > 256 or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a bounded lowercase identifier")
    return value


def _version(value: str, label: str) -> str:
    if type(value) is not str or len(value) > 64 or not _VERSION.fullmatch(value):
        raise ValueError(f"{label} must be a bounded semantic version")
    return value


def _reference(value: str, label: str) -> str:
    if type(value) is not str or not _REFERENCE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _json_object(value: str, label: str, maximum: int) -> str:
    _bounded_text(value, label, maximum)

    def reject_constant(token: str) -> object:
        raise ValueError(f"{label} contains a non-finite constant: {token}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be JSON") from exc
    if type(decoded) is not dict:
        raise ValueError(f"{label} must encode an object")
    if _canonical(decoded).decode("utf-8") != value:
        raise ValueError(f"{label} must be canonical JSON")
    return value


def _utc_timestamp(value: str) -> str:
    _bounded_text(value, "recorded_at_utc", 32)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("recorded_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("recorded_at_utc must identify UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    if value != canonical:
        raise ValueError("recorded_at_utc must be canonical UTC with microseconds")
    return value


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One immutable, versioned tool interface; never an executable binding."""

    tool_id: str
    tool_version: str
    description: str
    permission_scope: str
    input_schema_json: str
    output_schema_json: str
    external_effect: bool = False

    def __post_init__(self) -> None:
        _identifier(self.tool_id, "tool_id")
        _version(self.tool_version, "tool_version")
        _bounded_text(self.description, "description", 4_096)
        _identifier(self.permission_scope, "permission_scope")
        _json_object(self.input_schema_json, "input_schema_json", 65_536)
        _json_object(self.output_schema_json, "output_schema_json", 65_536)
        if type(self.external_effect) is not bool:
            raise TypeError("external_effect must be boolean")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contract": TOOL_DEFINITION_CONTRACT,
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
            "description": self.description,
            "permission_scope": self.permission_scope,
            "input_schema_json": self.input_schema_json,
            "output_schema_json": self.output_schema_json,
            "external_effect": self.external_effect,
        }

    @property
    def definition_ref(self) -> str:
        return _digest(_canonical(self.canonical_payload()))


@dataclass(frozen=True, slots=True)
class ToolManifest:
    """A content-addressed set of exact tool-definition versions."""

    manifest_id: str
    manifest_version: str
    tools: tuple[ToolDefinition, ...]

    def __post_init__(self) -> None:
        _identifier(self.manifest_id, "manifest_id")
        _version(self.manifest_version, "manifest_version")
        if type(self.tools) is not tuple or not 1 <= len(self.tools) <= 256:
            raise ValueError("tools must contain 1 through 256 definitions")
        if any(not isinstance(item, ToolDefinition) for item in self.tools):
            raise TypeError("tools must contain ToolDefinition values")
        order = tuple((item.tool_id, item.tool_version) for item in self.tools)
        if order != tuple(sorted(order)):
            raise ValueError("tools must be sorted by tool_id and tool_version")
        if len({item.tool_id for item in self.tools}) != len(self.tools):
            raise ValueError("a manifest cannot contain multiple definitions for one tool_id")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contract": TOOL_MANIFEST_CONTRACT,
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "tool_refs": tuple(item.definition_ref for item in self.tools),
        }

    @property
    def manifest_ref(self) -> str:
        return _digest(_canonical(self.canonical_payload()))


@dataclass(frozen=True, slots=True)
class ToolCallReservation:
    """An immutable intent to invoke one exact tool definition."""

    call_id: str
    manifest_ref: str
    tool_ref: str
    requester_ref: str
    arguments_json: str

    def __post_init__(self) -> None:
        _identifier(self.call_id, "call_id")
        _reference(self.manifest_ref, "manifest_ref")
        _reference(self.tool_ref, "tool_ref")
        _reference(self.requester_ref, "requester_ref")
        _json_object(self.arguments_json, "arguments_json", 65_536)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contract": TOOL_CALL_RESERVATION_CONTRACT,
            "call_id": self.call_id,
            "manifest_ref": self.manifest_ref,
            "tool_ref": self.tool_ref,
            "requester_ref": self.requester_ref,
            "arguments_json": self.arguments_json,
        }

    @property
    def operation_ref(self) -> str:
        return _digest(_canonical(self.canonical_payload()))


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """A content-addressed observation about a reserved operation."""

    operation_ref: str
    sequence: int
    status: Literal["RUNNING", "COMPLETED", "ERROR", "DENIED"]
    result_json: str
    recorded_at_utc: str

    def __post_init__(self) -> None:
        _reference(self.operation_ref, "operation_ref")
        if type(self.sequence) is not int or self.sequence not in (1, 2):
            raise ValueError("event sequence must be 1 or 2")
        if self.status not in ("RUNNING", "COMPLETED", "ERROR", "DENIED"):
            raise ValueError("event status is unsupported")
        _json_object(self.result_json, "result_json", 1_048_576)
        _utc_timestamp(self.recorded_at_utc)
        if self.status == "RUNNING" and self.result_json != "{}":
            raise ValueError("RUNNING events cannot claim a result")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contract": TOOL_EVENT_CONTRACT,
            "operation_ref": self.operation_ref,
            "sequence": self.sequence,
            "status": self.status,
            "result_json": self.result_json,
            "recorded_at_utc": self.recorded_at_utc,
        }

    @property
    def event_ref(self) -> str:
        return _digest(_canonical(self.canonical_payload()))


class ToolOperationStore:
    """SQLite-backed crash-safe idempotency for asynchronous tool operations.

    The store deliberately has no executor registration or invocation API.
    External-effect definitions can be recorded for provenance, but this V1
    boundary refuses to reserve them.
    """

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a Path")
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = _connect_private_sqlite(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tool_definitions (
                    definition_ref TEXT PRIMARY KEY,
                    tool_id TEXT NOT NULL,
                    tool_version TEXT NOT NULL,
                    external_effect INTEGER NOT NULL CHECK (external_effect IN (0, 1)),
                    payload_json TEXT NOT NULL,
                    UNIQUE (tool_id, tool_version)
                );
                CREATE TABLE IF NOT EXISTS tool_manifests (
                    manifest_ref TEXT PRIMARY KEY,
                    manifest_id TEXT NOT NULL,
                    manifest_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (manifest_id, manifest_version)
                );
                CREATE TABLE IF NOT EXISTS tool_manifest_members (
                    manifest_ref TEXT NOT NULL REFERENCES tool_manifests(manifest_ref),
                    definition_ref TEXT NOT NULL REFERENCES tool_definitions(definition_ref),
                    position INTEGER NOT NULL,
                    PRIMARY KEY (manifest_ref, definition_ref),
                    UNIQUE (manifest_ref, position)
                );
                CREATE TABLE IF NOT EXISTS tool_operations (
                    call_id TEXT PRIMARY KEY,
                    operation_ref TEXT NOT NULL UNIQUE,
                    manifest_ref TEXT NOT NULL REFERENCES tool_manifests(manifest_ref),
                    definition_ref TEXT NOT NULL REFERENCES tool_definitions(definition_ref),
                    reservation_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_events (
                    event_ref TEXT PRIMARY KEY,
                    operation_ref TEXT NOT NULL REFERENCES tool_operations(operation_ref),
                    sequence INTEGER NOT NULL CHECK (sequence IN (1, 2)),
                    status TEXT NOT NULL CHECK (
                        status IN ('RUNNING', 'COMPLETED', 'ERROR', 'DENIED')
                    ),
                    recorded_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (operation_ref, sequence)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_terminal_tool_event
                    ON tool_events(operation_ref)
                    WHERE status IN ('COMPLETED', 'ERROR', 'DENIED');
                """
            )
            connection.commit()
            _ensure_private_sqlite_sidecars(self.database_path)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def register_manifest(self, manifest: ToolManifest) -> str:
        if not isinstance(manifest, ToolManifest):
            raise TypeError("manifest must be a ToolManifest")
        manifest_payload = _canonical(manifest.canonical_payload()).decode("utf-8")
        with self._lock, self._transaction() as connection:
            for tool in manifest.tools:
                payload = _canonical(tool.canonical_payload()).decode("utf-8")
                existing = connection.execute(
                    """SELECT definition_ref, payload_json FROM tool_definitions
                       WHERE tool_id = ? AND tool_version = ?""",
                    (tool.tool_id, tool.tool_version),
                ).fetchone()
                if existing is not None and (
                    existing["definition_ref"] != tool.definition_ref
                    or existing["payload_json"] != payload
                ):
                    raise ValueError("tool definition version conflicts with stored content")
                connection.execute(
                    """INSERT OR IGNORE INTO tool_definitions
                       (definition_ref, tool_id, tool_version, external_effect, payload_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        tool.definition_ref,
                        tool.tool_id,
                        tool.tool_version,
                        int(tool.external_effect),
                        payload,
                    ),
                )

            existing_manifest = connection.execute(
                """SELECT manifest_ref, payload_json FROM tool_manifests
                   WHERE manifest_id = ? AND manifest_version = ?""",
                (manifest.manifest_id, manifest.manifest_version),
            ).fetchone()
            if existing_manifest is not None and (
                existing_manifest["manifest_ref"] != manifest.manifest_ref
                or existing_manifest["payload_json"] != manifest_payload
            ):
                raise ValueError("tool manifest version conflicts with stored content")
            connection.execute(
                """INSERT OR IGNORE INTO tool_manifests
                   (manifest_ref, manifest_id, manifest_version, payload_json)
                   VALUES (?, ?, ?, ?)""",
                (
                    manifest.manifest_ref,
                    manifest.manifest_id,
                    manifest.manifest_version,
                    manifest_payload,
                ),
            )
            for position, tool in enumerate(manifest.tools):
                connection.execute(
                    """INSERT OR IGNORE INTO tool_manifest_members
                       (manifest_ref, definition_ref, position) VALUES (?, ?, ?)""",
                    (manifest.manifest_ref, tool.definition_ref, position),
                )
        return manifest.manifest_ref

    def reserve(self, reservation: ToolCallReservation) -> ToolCallReservation:
        """Persist a call intent without running it.

        Retrying an identical ``call_id`` returns the original reservation.
        Any different content under that identity is rejected.
        """

        if not isinstance(reservation, ToolCallReservation):
            raise TypeError("reservation must be a ToolCallReservation")
        payload = _canonical(reservation.canonical_payload()).decode("utf-8")
        with self._lock, self._transaction() as connection:
            existing = connection.execute(
                "SELECT reservation_json FROM tool_operations WHERE call_id = ?",
                (reservation.call_id,),
            ).fetchone()
            if existing is not None:
                stored = _reservation_from_json(existing["reservation_json"])
                if stored != reservation:
                    raise ValueError("call_id conflicts with the stored reservation")
                return stored

            membership = connection.execute(
                """SELECT d.external_effect
                   FROM tool_manifest_members AS m
                   JOIN tool_definitions AS d
                     ON d.definition_ref = m.definition_ref
                   WHERE m.manifest_ref = ? AND m.definition_ref = ?""",
                (reservation.manifest_ref, reservation.tool_ref),
            ).fetchone()
            if membership is None:
                raise ValueError("reservation tool is not a member of the exact manifest")
            if bool(membership["external_effect"]):
                raise PermissionError("external-effect tool reservations are disabled")
            connection.execute(
                """INSERT INTO tool_operations
                   (call_id, operation_ref, manifest_ref, definition_ref, reservation_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    reservation.call_id,
                    reservation.operation_ref,
                    reservation.manifest_ref,
                    reservation.tool_ref,
                    payload,
                ),
            )
        return reservation

    def append_event(self, event: ToolEvent) -> ToolEvent:
        """Append RUNNING and then exactly one terminal observation."""

        if not isinstance(event, ToolEvent):
            raise TypeError("event must be a ToolEvent")
        payload = _canonical(event.canonical_payload()).decode("utf-8")
        with self._lock, self._transaction() as connection:
            operation = connection.execute(
                "SELECT 1 FROM tool_operations WHERE operation_ref = ?",
                (event.operation_ref,),
            ).fetchone()
            if operation is None:
                raise ValueError("event refers to an unknown operation")

            same_sequence = connection.execute(
                """SELECT event_ref, payload_json FROM tool_events
                   WHERE operation_ref = ? AND sequence = ?""",
                (event.operation_ref, event.sequence),
            ).fetchone()
            if same_sequence is not None:
                stored = _event_from_json(same_sequence["payload_json"])
                if same_sequence["event_ref"] == event.event_ref and stored == event:
                    return stored
                raise ValueError("event sequence conflicts with the stored event")

            prior = connection.execute(
                """SELECT payload_json FROM tool_events
                   WHERE operation_ref = ? ORDER BY sequence""",
                (event.operation_ref,),
            ).fetchall()
            if not prior:
                if event.sequence != 1 or event.status != "RUNNING":
                    raise ValueError("the first event must be RUNNING at sequence 1")
            else:
                first = _event_from_json(prior[0]["payload_json"])
                if len(prior) != 1 or first.sequence != 1 or first.status != "RUNNING":
                    raise RuntimeError("stored tool-event sequence is invalid")
                if event.sequence != 2 or event.status not in _TERMINAL_STATUSES:
                    raise ValueError("RUNNING must be followed by one terminal event")
                first_time = datetime.fromisoformat(
                    first.recorded_at_utc.replace("Z", "+00:00")
                )
                event_time = datetime.fromisoformat(
                    event.recorded_at_utc.replace("Z", "+00:00")
                )
                if event_time < first_time:
                    raise ValueError("terminal event time cannot precede RUNNING")

            connection.execute(
                """INSERT INTO tool_events
                   (event_ref, operation_ref, sequence, status, recorded_at_utc, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.event_ref,
                    event.operation_ref,
                    event.sequence,
                    event.status,
                    event.recorded_at_utc,
                    payload,
                ),
            )
        return event

    def reservation_for_call(self, call_id: str) -> ToolCallReservation | None:
        _identifier(call_id, "call_id")
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT operation_ref, reservation_json FROM tool_operations
                   WHERE call_id = ?""",
                (call_id,),
            ).fetchone()
        if row is None:
            return None
        reservation = _reservation_from_json(row["reservation_json"])
        if reservation.operation_ref != row["operation_ref"]:
            raise RuntimeError("stored tool reservation failed content validation")
        return reservation

    def reservation_for_operation(
        self, operation_ref: str
    ) -> ToolCallReservation | None:
        """Resolve the opaque operation reference with exact hash validation."""

        _reference(operation_ref, "operation_ref")
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT operation_ref, reservation_json FROM tool_operations
                   WHERE operation_ref = ?""",
                (operation_ref,),
            ).fetchone()
        if row is None:
            return None
        reservation = _reservation_from_json(row["reservation_json"])
        if reservation.operation_ref != row["operation_ref"]:
            raise RuntimeError("stored tool reservation failed content validation")
        return reservation

    def events_for_operation(self, operation_ref: str) -> tuple[ToolEvent, ...]:
        _reference(operation_ref, "operation_ref")
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT payload_json FROM tool_events
                   WHERE operation_ref = ? ORDER BY sequence""",
                (operation_ref,),
            ).fetchall()
        return tuple(_event_from_json(row["payload_json"]) for row in rows)


def _reservation_from_json(payload_json: str) -> ToolCallReservation:
    payload = json.loads(payload_json)
    if set(payload) != {
        "contract",
        "call_id",
        "manifest_ref",
        "tool_ref",
        "requester_ref",
        "arguments_json",
    } or payload["contract"] != TOOL_CALL_RESERVATION_CONTRACT:
        raise RuntimeError("stored tool reservation has an unsupported contract")
    return ToolCallReservation(
        call_id=payload["call_id"],
        manifest_ref=payload["manifest_ref"],
        tool_ref=payload["tool_ref"],
        requester_ref=payload["requester_ref"],
        arguments_json=payload["arguments_json"],
    )


def _event_from_json(payload_json: str) -> ToolEvent:
    payload = json.loads(payload_json)
    if set(payload) != {
        "contract",
        "operation_ref",
        "sequence",
        "status",
        "result_json",
        "recorded_at_utc",
    } or payload["contract"] != TOOL_EVENT_CONTRACT:
        raise RuntimeError("stored tool event has an unsupported contract")
    return ToolEvent(
        operation_ref=payload["operation_ref"],
        sequence=payload["sequence"],
        status=payload["status"],
        result_json=payload["result_json"],
        recorded_at_utc=payload["recorded_at_utc"],
    )
