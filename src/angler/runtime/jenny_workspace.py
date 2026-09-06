"""Bounded, content-addressed project workspace access for Jenny.

The executor in this module is intentionally only a filesystem mechanism.  An
``AffordanceRequest`` must name the exact operation, path, range, write mode,
and (for writes) content.  This module does not select files, synthesize
content, run programs, delete paths, or rename user-visible paths.

The workspace root is opened once without following symlinks and is rebound to
its original device/inode before every request.  Descendant traversal uses
directory file descriptors plus ``O_NOFOLLOW`` so a path never escapes through
lexical components or symlinks.  Writes are staged in the destination
directory, fsynced, atomically installed, and followed by a directory fsync.

This is a standalone component boundary, not a live-runtime binding.  Its
bounded replay cache is process-local.  Before activation, read/list must be
split from a separately authorized, durably journaled deferred-write path so a
process crash between filesystem commit and supervisor receipt cannot cause an
ambiguous retry.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import threading
from typing import Literal
import unicodedata

from .persistent_autonomy import (
    Affordance,
    AffordanceReceipt,
    AffordanceRequest,
    ObservableConsequence,
)


PROJECT_WORKSPACE_AFFORDANCE_ID = "internal.project-workspace"
PROJECT_WORKSPACE_PERMISSION_SCOPE = "internal.project-workspace"
PROJECT_WORKSPACE_SOURCE_KIND = "TOOL"
PROJECT_WORKSPACE_ACTION_CONTRACT = "jenny.project-workspace.action.v1"
PROJECT_WORKSPACE_OBSERVATION_CONTRACT = (
    "jenny.project-workspace.observation.v1"
)
PROJECT_WORKSPACE_SOURCE_CONTRACT = "jenny.project-workspace.source.v1"

PROJECT_WORKSPACE_AFFORDANCE_DESCRIPTION = (
    "Access one explicitly bound local project workspace with canonical JSON. "
    "Supported operations are list, read, write, and single-level mkdir. Every "
    "action must select an exact relative path; reads select an exact cursor "
    "and maximum character count; writes select create or replace and carry "
    "the exact UTF-8 content plus its SHA-256. Replace also requires the exact "
    "current content SHA-256. There is no shell, execution, delete, or rename."
)
PROJECT_WORKSPACE_AFFORDANCE = Affordance(
    affordance_id=PROJECT_WORKSPACE_AFFORDANCE_ID,
    disposition="ACT",
    description=PROJECT_WORKSPACE_AFFORDANCE_DESCRIPTION,
    permission_scope=PROJECT_WORKSPACE_PERMISSION_SCOPE,
    external_effect=False,
)

_REFERENCE_PREFIX = "sha256:"
_REFERENCE_LENGTH = len(_REFERENCE_PREFIX) + 64
_MAX_ACTION_BYTES = 16_384
_MAX_OBSERVATION_BYTES = 15_500
_MAX_PATH_BYTES = 1_024
_MAX_COMPONENT_BYTES = 255
_MAX_FILE_BYTES_HARD = 16 * 1024 * 1024
_MAX_READ_CHARS_HARD = 8_192
_MAX_LIST_ENTRIES_HARD = 128
_MAX_DIRECTORY_SCAN_ENTRIES = 4_096
_MAX_REPLAY_ENTRIES_HARD = 4_096
_TEMP_PREFIX = ".__jenny_workspace_staged__"


class WorkspaceSecurityError(PermissionError):
    """A request crossed an ownership, root, link, or object-type boundary."""


class WorkspaceConflictError(RuntimeError):
    """A create, compare-and-swap, or idempotency precondition did not hold."""


class WorkspaceIntegrityError(RuntimeError):
    """A bound object changed while an operation was inspecting it."""


class WorkspaceBoundsError(ValueError):
    """A request, response, directory, or file exceeded an explicit ceiling."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("workspace value is not canonical JSON data") from exc


def _sha256_ref(value: bytes) -> str:
    return _REFERENCE_PREFIX + hashlib.sha256(value).hexdigest()


def workspace_content_ref(value: str) -> str:
    """Return the content reference used for exact UTF-8 workspace text."""

    if type(value) is not str:
        raise TypeError("workspace content must be exact text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("workspace content must be valid UTF-8 text") from exc
    return _sha256_ref(encoded)


def _content_ref(value: object) -> str:
    return _sha256_ref(_canonical_json(value).encode("utf-8"))


def _reference(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != _REFERENCE_LENGTH
        or not value.startswith(_REFERENCE_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _parse_canonical_object(value: object) -> dict[str, object]:
    if type(value) is not str:
        raise TypeError("workspace action must be exact canonical JSON text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("workspace action must be valid UTF-8") from exc
    if len(encoded) > _MAX_ACTION_BYTES:
        raise WorkspaceBoundsError("workspace action exceeds its byte ceiling")

    def reject_constant(token: str) -> object:
        raise ValueError(f"workspace action contains non-finite constant {token}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("workspace action contains a duplicate key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("workspace action must be JSON") from exc
    if type(decoded) is not dict or _canonical_json(decoded) != value:
        raise ValueError("workspace action must be an exact canonical JSON object")
    return decoded


def _exact_fields(
    payload: dict[str, object], required: frozenset[str], operation: str
) -> None:
    expected = required | {"contract", "operation"}
    if set(payload) != expected:
        raise ValueError(f"workspace {operation} action schema differs")
    if payload["contract"] != PROJECT_WORKSPACE_ACTION_CONTRACT:
        raise ValueError("workspace action contract is unsupported")
    if payload["operation"] != operation:
        raise ValueError("workspace action operation differs")


def _canonical_relative_path(
    value: object, label: str, *, allow_root: bool
) -> tuple[str, tuple[str, ...]]:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise WorkspaceSecurityError(
            f"{label} must be a canonical POSIX relative path"
        )
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise WorkspaceSecurityError(f"{label} must be valid UTF-8") from exc
    if len(encoded) > _MAX_PATH_BYTES:
        raise WorkspaceBoundsError(f"{label} exceeds its byte ceiling")
    if unicodedata.normalize("NFC", value) != value:
        raise WorkspaceSecurityError(f"{label} must use NFC Unicode")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise WorkspaceSecurityError(f"{label} may not contain control characters")
    if value == ".":
        if allow_root:
            return value, ()
        raise WorkspaceSecurityError(f"{label} may not name the workspace root")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise WorkspaceSecurityError(
            f"{label} must be a canonical POSIX relative path"
        )
    for part in path.parts:
        if len(part.encode("utf-8")) > _MAX_COMPONENT_BYTES:
            raise WorkspaceBoundsError(f"{label} contains an overlong component")
        if part.startswith(_TEMP_PREFIX):
            raise WorkspaceSecurityError(f"{label} uses a reserved workspace name")
    return value, path.parts


def _positive_int(value: object, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise WorkspaceBoundsError(f"{label} must be in 1 through {maximum}")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise WorkspaceBoundsError(f"{label} must be a non-negative integer")
    return value


def _open_absolute_directory(path: str) -> int:
    """Open an absolute directory component-by-component without symlinks."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("safe workspace traversal is unsupported on this platform")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.path.sep, flags)
    try:
        for part in Path(path).parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_uid)


def _file_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@dataclass(frozen=True, slots=True)
class _ReplayRecord:
    request: AffordanceRequest
    receipt: AffordanceReceipt
    operation: Literal["list", "read", "write", "mkdir"]
    path: str
    verification_ref: str


class JennyWorkspaceExecutor:
    """Execute exact UTF-8 operations inside one same-owner canonical root."""

    AFFORDANCE_ID = PROJECT_WORKSPACE_AFFORDANCE_ID
    SOURCE_KIND = PROJECT_WORKSPACE_SOURCE_KIND

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        max_file_bytes: int = 1024 * 1024,
        max_read_chars: int = 8_192,
        max_list_entries: int = 64,
        max_replay_entries: int = 1_024,
    ) -> None:
        if not isinstance(workspace_root, (str, Path)):
            raise TypeError("workspace_root must be an explicit path")
        raw_root = os.fspath(workspace_root)
        if type(raw_root) is not str or not os.path.isabs(raw_root):
            raise ValueError("workspace_root must be an absolute path")
        canonical_root = os.path.abspath(raw_root)
        if raw_root != canonical_root:
            raise ValueError("workspace_root must be a canonical absolute path")
        if os.path.realpath(canonical_root) != canonical_root:
            raise WorkspaceSecurityError("workspace_root may not traverse a symlink")
        self._max_file_bytes = _positive_int(
            max_file_bytes, "max_file_bytes", _MAX_FILE_BYTES_HARD
        )
        self._max_read_chars = _positive_int(
            max_read_chars, "max_read_chars", _MAX_READ_CHARS_HARD
        )
        self._max_list_entries = _positive_int(
            max_list_entries, "max_list_entries", _MAX_LIST_ENTRIES_HARD
        )
        self._max_replay_entries = _positive_int(
            max_replay_entries,
            "max_replay_entries",
            _MAX_REPLAY_ENTRIES_HARD,
        )
        try:
            root_descriptor = _open_absolute_directory(canonical_root)
        except OSError as exc:
            raise WorkspaceSecurityError(
                "workspace_root must be a real directory without symlink traversal"
            ) from exc
        try:
            metadata = os.fstat(root_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise WorkspaceSecurityError("workspace_root must be a real directory")
            owner_uid = os.geteuid()
            if metadata.st_uid != owner_uid:
                raise WorkspaceSecurityError(
                    "workspace_root must be owned by the executing user"
                )
        except BaseException:
            os.close(root_descriptor)
            raise
        self._root_path = canonical_root
        self._root_descriptor = root_descriptor
        self._root_identity = _directory_identity(metadata)
        self._root_device = metadata.st_dev
        self._owner_uid = owner_uid
        self.workspace_ref = _content_ref(
            {
                "contract": PROJECT_WORKSPACE_SOURCE_CONTRACT,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "owner_uid": metadata.st_uid,
                "root": canonical_root,
            }
        )
        self._closed = False
        self._lock = threading.RLock()
        self._replays: dict[str, _ReplayRecord] = {}

    @property
    def root(self) -> Path:
        """Return the canonical path bound at construction (never user-selected)."""

        return Path(self._root_path)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                os.close(self._root_descriptor)
                self._closed = True

    def __enter__(self) -> "JennyWorkspaceExecutor":
        if self._closed:
            raise RuntimeError("workspace executor is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        descriptor = getattr(self, "_root_descriptor", None)
        if descriptor is not None and not getattr(self, "_closed", True):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _assert_root_binding(self) -> None:
        if self._closed:
            raise RuntimeError("workspace executor is closed")
        try:
            retained = os.fstat(self._root_descriptor)
            rebound_descriptor = _open_absolute_directory(self._root_path)
        except OSError as exc:
            raise WorkspaceIntegrityError("workspace root binding is unavailable") from exc
        try:
            rebound = os.fstat(rebound_descriptor)
        finally:
            os.close(rebound_descriptor)
        if (
            not stat.S_ISDIR(retained.st_mode)
            or not stat.S_ISDIR(rebound.st_mode)
            or _directory_identity(retained) != self._root_identity
            or _directory_identity(rebound) != self._root_identity
            or retained.st_uid != self._owner_uid
        ):
            raise WorkspaceIntegrityError("workspace root binding changed")

    def _validate_directory(self, metadata: os.stat_result) -> None:
        if not stat.S_ISDIR(metadata.st_mode):
            raise WorkspaceSecurityError("workspace path component is not a directory")
        if metadata.st_uid != self._owner_uid:
            raise WorkspaceSecurityError(
                "workspace path component has a different owner"
            )
        if metadata.st_dev != self._root_device:
            raise WorkspaceSecurityError("workspace path crosses a filesystem boundary")

    def _open_directory(self, parts: tuple[str, ...]) -> int:
        self._assert_root_binding()
        descriptor = os.dup(self._root_descriptor)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            self._validate_directory(os.fstat(descriptor))
            for part in parts:
                try:
                    next_descriptor = os.open(part, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise WorkspaceSecurityError(
                        "workspace directory path is unavailable without symlinks"
                    ) from exc
                os.close(descriptor)
                descriptor = next_descriptor
                self._validate_directory(os.fstat(descriptor))
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _parent_descriptor(self, parts: tuple[str, ...]) -> tuple[int, str]:
        if not parts:
            raise WorkspaceSecurityError("operation may not target the workspace root")
        return self._open_directory(parts[:-1]), parts[-1]

    def _validate_regular_file(self, metadata: os.stat_result) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkspaceSecurityError("workspace source must be a regular file")
        if metadata.st_uid != self._owner_uid:
            raise WorkspaceSecurityError("workspace source has a different owner")
        if metadata.st_dev != self._root_device:
            raise WorkspaceSecurityError("workspace source crosses a filesystem boundary")
        if metadata.st_nlink != 1:
            raise WorkspaceSecurityError("workspace source may not be hard-linked")
        if not 0 <= metadata.st_size <= self._max_file_bytes:
            raise WorkspaceBoundsError("workspace file exceeds its byte ceiling")

    def _read_file_state_at(
        self, parent_descriptor: int, name: str
    ) -> tuple[bytes, tuple[int, ...]]:
        try:
            path_metadata = os.stat(
                name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except OSError as exc:
            raise WorkspaceSecurityError("workspace source is unavailable") from exc
        self._validate_regular_file(path_metadata)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise WorkspaceSecurityError(
                "workspace source could not be opened without following links"
            ) from exc
        try:
            before = os.fstat(descriptor)
            self._validate_regular_file(before)
            if (before.st_dev, before.st_ino) != (
                path_metadata.st_dev,
                path_metadata.st_ino,
            ):
                raise WorkspaceIntegrityError("workspace source changed while opening")
            chunks: list[bytes] = []
            total = 0
            while True:
                remaining = self._max_file_bytes + 1 - total
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self._max_file_bytes:
                    raise WorkspaceBoundsError(
                        "workspace file exceeds its byte ceiling"
                    )
            after = os.fstat(descriptor)
            if _file_snapshot(before) != _file_snapshot(after):
                raise WorkspaceIntegrityError("workspace source changed while reading")
            try:
                rebound = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except OSError as exc:
                raise WorkspaceIntegrityError(
                    "workspace source changed while reading"
                ) from exc
            if (rebound.st_dev, rebound.st_ino) != (before.st_dev, before.st_ino):
                raise WorkspaceIntegrityError("workspace source changed while reading")
            return b"".join(chunks), _file_snapshot(after)
        finally:
            os.close(descriptor)

    def _read_file_at(self, parent_descriptor: int, name: str) -> bytes:
        return self._read_file_state_at(parent_descriptor, name)[0]

    def _read_path(self, parts: tuple[str, ...]) -> bytes:
        parent_descriptor, name = self._parent_descriptor(parts)
        try:
            return self._read_file_at(parent_descriptor, name)
        finally:
            os.close(parent_descriptor)

    def _entry_payload(
        self, directory_path: str, descriptor: int, name: str
    ) -> dict[str, object]:
        _canonical_relative_path(name, "workspace entry name", allow_root=False)
        if "/" in name:
            raise WorkspaceIntegrityError("workspace entry name contains a separator")
        if name.startswith(_TEMP_PREFIX):
            raise WorkspaceIntegrityError(
                "workspace contains an unfinished staging artifact"
            )
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise WorkspaceIntegrityError(
                "workspace directory changed while listing"
            ) from exc
        if metadata.st_uid != self._owner_uid:
            raise WorkspaceSecurityError("workspace entry has a different owner")
        if metadata.st_dev != self._root_device:
            raise WorkspaceSecurityError("workspace entry crosses a filesystem boundary")
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            kind = "file"
            size: int | None = metadata.st_size
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            size = None
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink-unavailable"
            size = None
        else:
            kind = "unsupported"
            size = None
        path = name if directory_path == "." else f"{directory_path}/{name}"
        if len(path.encode("utf-8")) > _MAX_PATH_BYTES:
            raise WorkspaceBoundsError("listed workspace path exceeds its byte ceiling")
        return {"kind": kind, "name": name, "path": path, "size_bytes": size}

    def _list_observation(
        self, payload: dict[str, object], request: AffordanceRequest
    ) -> tuple[AffordanceReceipt, str, str]:
        _exact_fields(
            payload,
            frozenset(("after", "max_entries", "path")),
            "list",
        )
        directory_path, parts = _canonical_relative_path(
            payload["path"], "workspace list path", allow_root=True
        )
        maximum = _positive_int(
            payload["max_entries"], "workspace max_entries", self._max_list_entries
        )
        raw_after = payload["after"]
        if raw_after is None:
            after: str | None = None
        else:
            after, after_parts = _canonical_relative_path(
                raw_after, "workspace list cursor", allow_root=False
            )
            if len(after_parts) != 1:
                raise ValueError("workspace list cursor must be one entry name")
        descriptor = self._open_directory(parts)
        try:
            before = os.fstat(descriptor)
            names: list[str] = []
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > _MAX_DIRECTORY_SCAN_ENTRIES:
                        raise WorkspaceBoundsError(
                            "workspace directory exceeds its scan ceiling"
                        )
            names.sort()
            page_names = [name for name in names if after is None or name > after]
            page_names = page_names[:maximum]
            entries_payload = [
                self._entry_payload(directory_path, descriptor, name)
                for name in page_names
            ]
            after_scan = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after_scan.st_dev,
                after_scan.st_ino,
                after_scan.st_mtime_ns,
                after_scan.st_ctime_ns,
            ):
                raise WorkspaceIntegrityError(
                    "workspace directory changed while listing"
                )
        finally:
            os.close(descriptor)

        remaining = sum(1 for name in names if after is None or name > after)
        while True:
            eof = len(entries_payload) == remaining
            next_after = None if eof else entries_payload[-1]["name"]
            listing_material = {
                "after": after,
                "entries": entries_payload,
                "eof": eof,
                "next_after": next_after,
                "path": directory_path,
            }
            listing_ref = _content_ref(listing_material)
            observation_payload = {
                "after": after,
                "contract": PROJECT_WORKSPACE_OBSERVATION_CONTRACT,
                "entries": entries_payload,
                "eof": eof,
                "listing_ref": listing_ref,
                "next_after": next_after,
                "operation": "list",
                "path": directory_path,
                "requested_max_entries": maximum,
                "workspace_ref": self.workspace_ref,
            }
            observation_json = _canonical_json(observation_payload)
            if len(observation_json.encode("utf-8")) <= _MAX_OBSERVATION_BYTES:
                break
            if not entries_payload:
                raise WorkspaceBoundsError(
                    "workspace list metadata exceeds the response ceiling"
                )
            entries_payload.pop()
        receipt = self._receipt(
            request=request,
            operation="list",
            observation_json=observation_json,
            artifact_refs=(listing_ref,),
            output=f"Listed {len(entries_payload)} workspace entrie(s) at {directory_path}.",
        )
        return receipt, directory_path, listing_ref

    def _read_observation(
        self, payload: dict[str, object], request: AffordanceRequest
    ) -> tuple[AffordanceReceipt, str, str]:
        _exact_fields(
            payload,
            frozenset(
                ("cursor", "expected_content_ref", "max_chars", "path")
            ),
            "read",
        )
        path, parts = _canonical_relative_path(
            payload["path"], "workspace read path", allow_root=False
        )
        cursor = _nonnegative_int(payload["cursor"], "workspace read cursor")
        maximum = _positive_int(
            payload["max_chars"], "workspace max_chars", self._max_read_chars
        )
        expected = payload["expected_content_ref"]
        if expected is not None:
            expected = _reference(expected, "workspace expected_content_ref")
        raw_content = self._read_path(parts)
        try:
            text = raw_content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise WorkspaceIntegrityError(
                "workspace source is not valid UTF-8 text"
            ) from exc
        content_ref = _sha256_ref(raw_content)
        if expected is not None and expected != content_ref:
            raise WorkspaceConflictError(
                "workspace read content differs from expected_content_ref"
            )
        if cursor > len(text):
            raise WorkspaceBoundsError("workspace read cursor is beyond end of file")
        desired_end = min(len(text), cursor + maximum)

        def build(end: int) -> tuple[str, str]:
            span = text[cursor:end]
            span_ref = workspace_content_ref(span)
            return _canonical_json(
                {
                    "content": span,
                    "content_ref": content_ref,
                    "contract": PROJECT_WORKSPACE_OBSERVATION_CONTRACT,
                    "cursor_unit": "unicode_code_point",
                    "eof": end == len(text),
                    "operation": "read",
                    "path": path,
                    "requested_max_chars": maximum,
                    "span": {"end": end, "start": cursor},
                    "span_ref": span_ref,
                    "total_bytes": len(raw_content),
                    "total_chars": len(text),
                    "workspace_ref": self.workspace_ref,
                }
            ), span_ref

        low, high = cursor, desired_end
        while low < high:
            middle = (low + high + 1) // 2
            candidate_json, _ = build(middle)
            if len(candidate_json.encode("utf-8")) <= _MAX_OBSERVATION_BYTES:
                low = middle
            else:
                high = middle - 1
        end = low
        observation_json, span_ref = build(end)
        if len(observation_json.encode("utf-8")) > _MAX_OBSERVATION_BYTES:
            raise WorkspaceBoundsError("workspace read metadata exceeds response ceiling")
        if cursor < len(text) and end == cursor:
            raise WorkspaceBoundsError("workspace response has no room for source text")
        receipt = self._receipt(
            request=request,
            operation="read",
            observation_json=observation_json,
            artifact_refs=tuple(sorted(set((content_ref, span_ref)))),
            output=f"Read workspace text {cursor}:{end} from {path}.",
        )
        return receipt, path, content_ref

    def _stage_content(self, parent_descriptor: int, content: bytes) -> str:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor: int | None = None
        name = ""
        for _ in range(32):
            name = _TEMP_PREFIX + secrets.token_hex(16)
            try:
                descriptor = os.open(
                    name, flags, 0o600, dir_fd=parent_descriptor
                )
                break
            except FileExistsError:
                continue
        if descriptor is None:
            raise WorkspaceIntegrityError("workspace staging name space is exhausted")
        try:
            view = memoryview(content)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError(errno.EIO, "workspace staging write made no progress")
                written += count
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            self._validate_regular_file(metadata)
            if metadata.st_size != len(content):
                raise WorkspaceIntegrityError("workspace staged content size differs")
        except BaseException:
            os.close(descriptor)
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
        os.close(descriptor)
        return name

    def _target_absent(self, parent_descriptor: int, name: str) -> bool:
        try:
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise WorkspaceSecurityError("workspace target cannot be inspected") from exc
        return False

    def _write_observation(
        self, payload: dict[str, object], request: AffordanceRequest
    ) -> tuple[AffordanceReceipt, str, str]:
        _exact_fields(
            payload,
            frozenset(
                ("content", "content_ref", "expected_content_ref", "mode", "path")
            ),
            "write",
        )
        path, parts = _canonical_relative_path(
            payload["path"], "workspace write path", allow_root=False
        )
        mode = payload["mode"]
        if mode not in ("create", "replace"):
            raise ValueError("workspace write mode must be create or replace")
        content = payload["content"]
        if type(content) is not str:
            raise TypeError("workspace write content must be exact text")
        try:
            encoded = content.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("workspace write content must be valid UTF-8") from exc
        if len(encoded) > self._max_file_bytes:
            raise WorkspaceBoundsError("workspace write exceeds the file byte ceiling")
        declared_content_ref = _reference(
            payload["content_ref"], "workspace content_ref"
        )
        actual_content_ref = _sha256_ref(encoded)
        if declared_content_ref != actual_content_ref:
            raise WorkspaceIntegrityError(
                "workspace write content differs from its declared content_ref"
            )
        raw_expected = payload["expected_content_ref"]
        if mode == "create":
            if raw_expected is not None:
                raise ValueError("workspace create requires null expected_content_ref")
            expected: str | None = None
        else:
            expected = _reference(
                raw_expected, "workspace replace expected_content_ref"
            )

        parent_descriptor, name = self._parent_descriptor(parts)
        staged_name: str | None = None
        previous_ref: str | None = None
        installed = False
        try:
            if mode == "create":
                if not self._target_absent(parent_descriptor, name):
                    raise WorkspaceConflictError(
                        "workspace create target already exists"
                    )
            else:
                previous = self._read_file_at(parent_descriptor, name)
                previous_ref = _sha256_ref(previous)
                if previous_ref != expected:
                    raise WorkspaceConflictError(
                        "workspace replace expected_content_ref is stale"
                    )
            staged_name = self._stage_content(parent_descriptor, encoded)
            self._assert_root_binding()
            if mode == "create":
                try:
                    os.link(
                        staged_name,
                        name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise WorkspaceConflictError(
                        "workspace create lost an existence race"
                    ) from exc
                installed = True
                os.unlink(staged_name, dir_fd=parent_descriptor)
                staged_name = None
            else:
                current, current_snapshot = self._read_file_state_at(
                    parent_descriptor, name
                )
                if _sha256_ref(current) != expected:
                    raise WorkspaceConflictError(
                        "workspace replace target changed before commit"
                    )
                try:
                    rebound = os.stat(
                        name, dir_fd=parent_descriptor, follow_symlinks=False
                    )
                except OSError as exc:
                    raise WorkspaceConflictError(
                        "workspace replace target changed before commit"
                    ) from exc
                self._validate_regular_file(rebound)
                if _file_snapshot(rebound) != current_snapshot:
                    raise WorkspaceConflictError(
                        "workspace replace target changed before commit"
                    )
                os.replace(
                    staged_name,
                    name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                staged_name = None
                installed = True
            os.fsync(parent_descriptor)
            written = self._read_file_at(parent_descriptor, name)
            if _sha256_ref(written) != actual_content_ref:
                raise WorkspaceIntegrityError(
                    "workspace target differs after atomic installation"
                )
        except WorkspaceConflictError:
            raise
        except (WorkspaceBoundsError, WorkspaceIntegrityError, WorkspaceSecurityError):
            raise
        except OSError as exc:
            qualifier = "after installation" if installed else "before installation"
            raise WorkspaceIntegrityError(
                f"workspace write failed {qualifier}"
            ) from exc
        finally:
            if staged_name is not None:
                try:
                    os.unlink(staged_name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                except OSError:
                    pass
            os.close(parent_descriptor)

        observation_json = _canonical_json(
            {
                "content_ref": actual_content_ref,
                "contract": PROJECT_WORKSPACE_OBSERVATION_CONTRACT,
                "mode": mode,
                "operation": "write",
                "path": path,
                "previous_content_ref": previous_ref,
                "size_bytes": len(encoded),
                "workspace_ref": self.workspace_ref,
            }
        )
        refs = tuple(
            sorted(
                set(
                    (actual_content_ref,)
                    if previous_ref is None
                    else (actual_content_ref, previous_ref)
                )
            )
        )
        receipt = self._receipt(
            request=request,
            operation="write",
            observation_json=observation_json,
            artifact_refs=refs,
            output=f"Atomically {mode}d workspace text at {path}.",
        )
        return receipt, path, actual_content_ref

    def _directory_ref(self, path: str, metadata: os.stat_result) -> str:
        return _content_ref(
            {
                "contract": "jenny.project-workspace.directory.v1",
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "path": path,
                "workspace_ref": self.workspace_ref,
            }
        )

    def _mkdir_observation(
        self, payload: dict[str, object], request: AffordanceRequest
    ) -> tuple[AffordanceReceipt, str, str]:
        # One create-only level is the minimum needed to express a nested
        # project tree.  Recursive parent creation would let mechanics choose
        # additional mutations that were not individually requested.
        _exact_fields(payload, frozenset(("mode", "path")), "mkdir")
        if payload["mode"] != "create":
            raise ValueError("workspace mkdir supports create mode only")
        path, parts = _canonical_relative_path(
            payload["path"], "workspace mkdir path", allow_root=False
        )
        parent_descriptor, name = self._parent_descriptor(parts)
        try:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError as exc:
                raise WorkspaceConflictError(
                    "workspace mkdir target already exists"
                ) from exc
            except OSError as exc:
                raise WorkspaceSecurityError("workspace directory could not be created") from exc
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        descriptor = self._open_directory(parts)
        try:
            metadata = os.fstat(descriptor)
            directory_ref = self._directory_ref(path, metadata)
        finally:
            os.close(descriptor)
        observation_json = _canonical_json(
            {
                "contract": PROJECT_WORKSPACE_OBSERVATION_CONTRACT,
                "directory_ref": directory_ref,
                "mode": "create",
                "operation": "mkdir",
                "path": path,
                "workspace_ref": self.workspace_ref,
            }
        )
        receipt = self._receipt(
            request=request,
            operation="mkdir",
            observation_json=observation_json,
            artifact_refs=(directory_ref,),
            output=f"Created one workspace directory at {path}.",
        )
        return receipt, path, directory_ref

    def _receipt(
        self,
        *,
        request: AffordanceRequest,
        operation: str,
        observation_json: str,
        artifact_refs: tuple[str, ...],
        output: str,
    ) -> AffordanceReceipt:
        if len(observation_json.encode("utf-8")) > _MAX_OBSERVATION_BYTES:
            raise WorkspaceBoundsError("workspace observation exceeds its byte ceiling")
        refs = tuple(sorted(set(artifact_refs)))
        evidence_refs = tuple(sorted(set((self.workspace_ref, *refs))))
        observation = ObservableConsequence(
            request_ref=request.idempotency_key,
            source_kind=self.SOURCE_KIND,
            source_ref=self.workspace_ref,
            observation_json=observation_json,
            artifact_refs=refs,
            evidence_refs=evidence_refs,
        )
        return AffordanceReceipt(
            status="COMPLETED",
            output=output,
            consequence=(),
            observable_consequence=observation,
        )

    def _execute(
        self, payload: dict[str, object], request: AffordanceRequest
    ) -> tuple[AffordanceReceipt, str, str, Literal["list", "read", "write", "mkdir"]]:
        operation = payload.get("operation")
        if operation == "list":
            receipt, path, marker = self._list_observation(payload, request)
            return receipt, path, marker, "list"
        if operation == "read":
            receipt, path, marker = self._read_observation(payload, request)
            return receipt, path, marker, "read"
        if operation == "write":
            receipt, path, marker = self._write_observation(payload, request)
            return receipt, path, marker, "write"
        if operation == "mkdir":
            receipt, path, marker = self._mkdir_observation(payload, request)
            return receipt, path, marker, "mkdir"
        raise ValueError(
            "workspace operation must be list, read, write, or mkdir; "
            "shell, execution, delete, and rename are unavailable"
        )

    def _verify_replay(
        self, record: _ReplayRecord, payload: dict[str, object]
    ) -> None:
        if record.operation == "write":
            _, parts = _canonical_relative_path(
                record.path, "workspace replay path", allow_root=False
            )
            current_ref = _sha256_ref(self._read_path(parts))
            if current_ref != record.verification_ref:
                raise WorkspaceIntegrityError(
                    "workspace write result changed before reservation replay"
                )
            return
        if record.operation == "mkdir":
            _, parts = _canonical_relative_path(
                record.path, "workspace replay path", allow_root=False
            )
            descriptor = self._open_directory(parts)
            try:
                current_ref = self._directory_ref(record.path, os.fstat(descriptor))
            finally:
                os.close(descriptor)
            if current_ref != record.verification_ref:
                raise WorkspaceIntegrityError(
                    "workspace directory changed before reservation replay"
                )
            return
        if record.operation == "read":
            _, parts = _canonical_relative_path(
                record.path, "workspace replay path", allow_root=False
            )
            current_ref = _sha256_ref(self._read_path(parts))
            if current_ref != record.verification_ref:
                raise WorkspaceIntegrityError(
                    "workspace read source changed before reservation replay"
                )
            return
        # A listing replay is recomputed, but never performs a mutation.  The
        # content-addressed listing marker must remain byte-identical.
        _, _, current_ref = self._list_observation(payload, record.request)
        if current_ref != record.verification_ref:
            raise WorkspaceIntegrityError(
                "workspace listing changed before reservation replay"
            )

    def __call__(self, request: AffordanceRequest) -> AffordanceReceipt:
        if type(request) is not AffordanceRequest:
            raise TypeError("request must be an exact AffordanceRequest")
        if request.affordance_id != self.AFFORDANCE_ID:
            raise ValueError("request selected a different affordance")
        payload = _parse_canonical_object(request.action_payload)
        with self._lock:
            self._assert_root_binding()
            replay = self._replays.get(request.idempotency_key)
            if replay is not None:
                if replay.request != request:
                    raise WorkspaceConflictError(
                        "idempotency key was reused for a different workspace request"
                    )
                self._verify_replay(replay, payload)
                return replay.receipt
            if len(self._replays) >= self._max_replay_entries:
                raise WorkspaceBoundsError("workspace replay cache is full")
            receipt, path, marker, operation = self._execute(payload, request)
            self._replays[request.idempotency_key] = _ReplayRecord(
                request=request,
                receipt=receipt,
                operation=operation,
                path=path,
                verification_ref=marker,
            )
            return receipt


# The longer alias keeps integration call sites explicit without creating a
# second implementation or semantic variant.
JennyProjectWorkspaceExecutor = JennyWorkspaceExecutor


__all__ = [
    "JennyProjectWorkspaceExecutor",
    "JennyWorkspaceExecutor",
    "PROJECT_WORKSPACE_ACTION_CONTRACT",
    "PROJECT_WORKSPACE_AFFORDANCE",
    "PROJECT_WORKSPACE_AFFORDANCE_DESCRIPTION",
    "PROJECT_WORKSPACE_AFFORDANCE_ID",
    "PROJECT_WORKSPACE_OBSERVATION_CONTRACT",
    "PROJECT_WORKSPACE_PERMISSION_SCOPE",
    "PROJECT_WORKSPACE_SOURCE_CONTRACT",
    "PROJECT_WORKSPACE_SOURCE_KIND",
    "WorkspaceBoundsError",
    "WorkspaceConflictError",
    "WorkspaceIntegrityError",
    "WorkspaceSecurityError",
    "workspace_content_ref",
]
