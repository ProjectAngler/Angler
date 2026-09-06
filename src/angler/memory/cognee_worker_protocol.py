"""Bounded JSON-lines protocol for the isolated local Cognee worker.

This module is deliberately standard-library only.  The Angler runtime imports
it without acquiring Cognee's dependency stack, while the worker imports it
before any Cognee module so configuration and framing can fail closed first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
from typing import Any, Mapping
import unicodedata
from uuid import NAMESPACE_OID, UUID, uuid5


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_ERROR_MESSAGE_CHARACTERS = 1_024
MAX_QUERY_CHARACTERS = 16_384
MAX_SEARCH_LIMIT = 256
MAX_POINT_NEIGHBORS = 384

COGNEE_VERSION = "1.5.3"
COGNEE_PYTHON_VERSION = "3.12.3"
SOFTWARE_VERSIONS = {
    "cognee": COGNEE_VERSION,
    "fastembed": "0.8.0",
    "ladybug": "0.19.0",
    "lancedb": "0.37.1",
    "litellm": "1.98.0",
    "onnxruntime": "1.23.2",
    "pydantic": "2.13.5",
    "sqlalchemy": "2.0.52",
}


class CogneeWorkerProtocolError(RuntimeError):
    """A local frame or message violated the frozen worker protocol."""


COGNEE_SCOPE_ROOT = "/opt/angler/state/project-angler"
SCOPE_MARKER_FILENAME = ".angler-cognee-worker-scope-v1.json"
_LEGACY_DATASET_NAME = "angler-acquisition-live-v1"
_LEGACY_TENANT_NAME = "angler-acquisition-live-v1-tenant"
_LEGACY_NODE_SET_NAME = "angler-acquisition-live-v1-records"
_LEGACY_STATE_ROOT = "/opt/angler/state/project-angler/cognee-acquisition-live-v1"
_SCOPE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SCOPE_ARGUMENTS = (
    "--dataset-name",
    "--tenant-name",
    "--node-set-name",
    "--state-root",
)


def _scope_name(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 128
        or not value.isascii()
        or value != unicodedata.normalize("NFC", value)
        or _SCOPE_NAME.fullmatch(value) is None
    ):
        raise CogneeWorkerProtocolError(
            f"{label} must match [a-z0-9]+(?:-[a-z0-9]+)* within 128 characters"
        )
    return value


def _scope_state_root(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 1_024
        or "\x00" in value
        or not value.isascii()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        or value != unicodedata.normalize("NFC", value)
    ):
        raise CogneeWorkerProtocolError(
            "state_root must be bounded, non-empty, NFC-normalized text"
        )
    path = Path(value)
    if not path.is_absolute() or str(path) != value or any(
        component in (".", "..") for component in path.parts
    ):
        raise CogneeWorkerProtocolError(
            "state_root must be an absolute lexically normalized path"
        )
    allowlist_root = Path(COGNEE_SCOPE_ROOT)
    try:
        relative = path.relative_to(allowlist_root)
    except ValueError as exc:
        raise CogneeWorkerProtocolError(
            "state_root must be below the Cognee scope allowlist root"
        ) from exc
    if not relative.parts:
        raise CogneeWorkerProtocolError(
            "state_root must be strictly below the Cognee scope allowlist root"
        )
    return value


def _native_node_set_id(node_set_name: str) -> str:
    return str(
        uuid5(
            NAMESPACE_OID,
            ("NodeSet:" + node_set_name)
            .lower()
            .replace(" ", "_")
            .replace("'", ""),
        )
    )


@dataclass(frozen=True, slots=True)
class CogneeWorkerScope:
    """One immutable dataset, tenant, NodeSet, and local state boundary."""

    dataset_name: str
    tenant_name: str
    node_set_name: str
    state_root: str

    def __post_init__(self) -> None:
        dataset_name = _scope_name(self.dataset_name, "dataset_name")
        tenant_name = _scope_name(self.tenant_name, "tenant_name")
        node_set_name = _scope_name(self.node_set_name, "node_set_name")
        state_root = _scope_state_root(self.state_root)
        if tenant_name != f"{dataset_name}-tenant":
            raise CogneeWorkerProtocolError(
                "tenant_name must be the dataset name followed by -tenant"
            )
        if node_set_name != f"{dataset_name}-records":
            raise CogneeWorkerProtocolError(
                "node_set_name must be the dataset name followed by -records"
            )
        legacy_root = Path(_LEGACY_STATE_ROOT)
        candidate_root = Path(state_root)
        if candidate_root == legacy_root:
            if (
                dataset_name,
                tenant_name,
                node_set_name,
            ) != (
                _LEGACY_DATASET_NAME,
                _LEGACY_TENANT_NAME,
                _LEGACY_NODE_SET_NAME,
            ):
                raise CogneeWorkerProtocolError(
                    "only the exact legacy scope may target the legacy state root"
                )
        elif candidate_root.is_relative_to(legacy_root):
            raise CogneeWorkerProtocolError(
                "a custom scope may not target the legacy state tree"
            )

    @property
    def node_set_id(self) -> str:
        return _native_node_set_id(self.node_set_name)

    @property
    def worker_cwd(self) -> str:
        return str(Path(self.state_root) / "worker-cwd")

    @property
    def worker_tmpdir(self) -> str:
        return str(Path(self.state_root) / "worker-tmp")


DEFAULT_COGNEE_WORKER_SCOPE = CogneeWorkerScope(
    dataset_name=_LEGACY_DATASET_NAME,
    tenant_name=_LEGACY_TENANT_NAME,
    node_set_name=_LEGACY_NODE_SET_NAME,
    state_root=_LEGACY_STATE_ROOT,
)
DATASET_NAME = DEFAULT_COGNEE_WORKER_SCOPE.dataset_name
TENANT_NAME = DEFAULT_COGNEE_WORKER_SCOPE.tenant_name
NODE_SET_NAME = DEFAULT_COGNEE_WORKER_SCOPE.node_set_name
COLLECTION_NAME = "AnglerAcquisitionReferencePoint_search_text"
NODE_SET_ID = DEFAULT_COGNEE_WORKER_SCOPE.node_set_id
SCORE_SEMANTICS = "cosine_distance_lower_is_better"
STATE_ROOT = DEFAULT_COGNEE_WORKER_SCOPE.state_root
WORKER_CWD = DEFAULT_COGNEE_WORKER_SCOPE.worker_cwd
WORKER_TMPDIR = DEFAULT_COGNEE_WORKER_SCOPE.worker_tmpdir
MODEL_CACHE_ROOT = "/opt/angler/models/fastembed-cache-v1"
TIKTOKEN_CACHE_ROOT = (
    "/opt/angler/venvs/cognee/lib/python3.12/site-packages/"
    "litellm/litellm_core_utils/tokenizers"
)
REPOSITORY_SOURCE_ROOT = "/opt/angler/src/angler/src"
COGNEE_PYTHON = "/opt/angler/venvs/cognee/bin/python"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIMENSIONS = 384
MODEL_MANIFEST = (
    "sha256:950932f40bdea47546ccc71dfb87585d1c2c74bf8f1e7d920ac0fddf53b1c148"
)
FASTEMBED_SESSION_THREADS = 2
FASTEMBED_EXECUTION_PROVIDERS = ("CPUExecutionProvider",)
INHERITED_NETWORK_NAMESPACE_ENV = (
    "ANGLER_COGNEE_INHERITED_NETWORK_NAMESPACE"
)

_NETWORK_NAMESPACE = re.compile(r"^net:\[[1-9][0-9]*\]$")
_ZERO_CAPABILITY_SET = "0000000000000000"
_CAPABILITY_STATUS_FIELDS = (
    "CapInh",
    "CapPrm",
    "CapEff",
    "CapBnd",
    "CapAmb",
)

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_POINT_FIELDS = frozenset(
    {
        "id",
        "acquisition_ref",
        "source_contract",
        "source_ref",
        "record_ref",
        "projection_ref",
        "acquired_ordinal",
        "search_text",
        "adjacent_record_refs",
        "references",
        "visibility",
        "memory_kind",
        "epistemic_status",
        "belongs_to_set",
    }
)


def exact_worker_scope(value: object) -> CogneeWorkerScope:
    if type(value) is not CogneeWorkerScope:
        raise CogneeWorkerProtocolError(
            "scope must be an exact CogneeWorkerScope"
        )
    return value


def canonical_network_namespace(value: object, label: str) -> str:
    """Return one canonical Linux network-namespace procfs identity."""

    if (
        type(value) is not str
        or _NETWORK_NAMESPACE.fullmatch(value) is None
    ):
        raise CogneeWorkerProtocolError(
            f"{label} must be an exact net:[positive-inode] identity"
        )
    return value


def _process_status_fields(value: object, label: str) -> dict[str, str]:
    if type(value) is not str or not value:
        raise CogneeWorkerProtocolError(f"{label} is unavailable")
    fields: dict[str, str] = {}
    for line in value.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        if name in fields:
            raise CogneeWorkerProtocolError(f"{label} repeats a field")
        fields[name] = raw.strip()
    return fields


def _privilege_status_receipt(value: object, label: str) -> dict[str, object]:
    fields = _process_status_fields(value, label)
    try:
        uids = tuple(int(item) for item in fields["Uid"].split())
        gids = tuple(int(item) for item in fields["Gid"].split())
        groups = tuple(int(item) for item in fields["Groups"].split())
    except (KeyError, ValueError) as exc:
        raise CogneeWorkerProtocolError(
            f"{label} identity fields differ"
        ) from exc
    capabilities = {
        field: fields.get(field) for field in _CAPABILITY_STATUS_FIELDS
    }
    if (
        uids != (1000, 1000, 1000, 1000)
        or gids != (1000, 1000, 1000, 1000)
        or groups
        or fields.get("NoNewPrivs") != "1"
        or any(
            capability != _ZERO_CAPABILITY_SET
            for capability in capabilities.values()
        )
    ):
        raise CogneeWorkerProtocolError(f"{label} privilege boundary differs")
    return {
        "capabilities": capabilities,
        "gids": list(gids),
        "groups": list(groups),
        "no_new_privileges": True,
        "uids": list(uids),
    }


def _default_process_identity() -> tuple[int, int, int, int, tuple[int, ...]]:
    return (
        os.getuid(),
        os.geteuid(),
        os.getgid(),
        os.getegid(),
        tuple(os.getgroups()),
    )


def attest_inherited_worker_parent_boundary(
    expected_network_namespace: str | None = None,
    *,
    identity_reader: Callable[
        [], tuple[int, int, int, int, tuple[int, ...]]
    ] | None = None,
    interface_inventory: Callable[[], Sequence[tuple[int, str]]] | None = None,
    text_reader: Callable[[Path], str] | None = None,
    namespace_reader: Callable[[Path], str] | None = None,
) -> dict[str, object]:
    """Attest a UID-1000, no-privilege, loopback-only launch parent.

    The returned fields contain no host address or route material.  A caller
    may bind the first receipt and require exact equality at later seams.
    """

    expected = (
        None
        if expected_network_namespace is None
        else canonical_network_namespace(
            expected_network_namespace,
            "expected inherited network namespace",
        )
    )
    identity = tuple((identity_reader or _default_process_identity)())
    if (
        len(identity) != 5
        or identity[:4] != (1000, 1000, 1000, 1000)
        or type(identity[4]) is not tuple
        or identity[4]
    ):
        raise CogneeWorkerProtocolError(
            "inherited worker parent process identity differs"
        )
    read_text = text_reader or (
        lambda path: path.read_text(encoding="ascii")
    )
    read_namespace = namespace_reader or os.readlink
    namespace = canonical_network_namespace(
        read_namespace(Path("/proc/self/ns/net")),
        "inherited worker parent network namespace",
    )
    if expected is not None and namespace != expected:
        raise CogneeWorkerProtocolError(
            "inherited worker parent network namespace differs"
        )
    privilege = _privilege_status_receipt(
        read_text(Path("/proc/self/status")),
        "inherited worker parent status",
    )
    interfaces = tuple((interface_inventory or socket.if_nameindex)())
    ipv4_lines = tuple(
        read_text(Path("/proc/net/route")).splitlines()
    )
    ipv4_header = (
        "Iface",
        "Destination",
        "Gateway",
        "Flags",
        "RefCnt",
        "Use",
        "Metric",
        "Mask",
        "MTU",
        "Window",
        "IRTT",
    )
    ipv4_routes = tuple(line for line in ipv4_lines[1:] if line.strip())
    ipv6_routes = tuple(
        tuple(line.split())
        for line in read_text(Path("/proc/net/ipv6_route")).splitlines()
        if line.strip()
    )
    loopback_operstate = read_text(
        Path("/sys/class/net/lo/operstate")
    ).strip()
    if (
        interfaces != ((1, "lo"),)
        or not ipv4_lines
        or tuple(ipv4_lines[0].split()) != ipv4_header
        or ipv4_routes
        or not ipv6_routes
        or any(
            len(route) != 10 or route[-1] != "lo"
            for route in ipv6_routes
        )
        or loopback_operstate != "unknown"
    ):
        raise CogneeWorkerProtocolError(
            "inherited worker parent is not loopback-only"
        )
    return {
        **privilege,
        "interfaces": [[1, "lo"]],
        "ipv4_route_count": 0,
        "ipv6_route_count": len(ipv6_routes),
        "loopback_operstate": loopback_operstate,
        "network_namespace": namespace,
    }


def validate_inherited_worker_boundary(
    expected_network_namespace: str,
    *,
    identity_reader: Callable[
        [], tuple[int, int, int, int, tuple[int, ...]]
    ] | None = None,
    parent_pid_reader: Callable[[], int] | None = None,
    interface_inventory: Callable[[], Sequence[tuple[int, str]]] | None = None,
    text_reader: Callable[[Path], str] | None = None,
    namespace_reader: Callable[[Path], str] | None = None,
) -> dict[str, object]:
    """Attest an inherited worker and the exact direct launch parent."""

    expected = canonical_network_namespace(
        expected_network_namespace,
        "expected inherited network namespace",
    )
    read_text = text_reader or (
        lambda path: path.read_text(encoding="ascii")
    )
    read_namespace = namespace_reader or os.readlink
    receipt = attest_inherited_worker_parent_boundary(
        expected,
        identity_reader=identity_reader,
        interface_inventory=interface_inventory,
        text_reader=read_text,
        namespace_reader=read_namespace,
    )
    parent_pid = (parent_pid_reader or os.getppid)()
    if type(parent_pid) is not int or parent_pid <= 1:
        raise CogneeWorkerProtocolError(
            "inherited worker direct parent identity differs"
        )
    parent_privilege = _privilege_status_receipt(
        read_text(Path("/proc") / str(parent_pid) / "status"),
        "inherited worker direct parent status",
    )
    parent_namespace = canonical_network_namespace(
        read_namespace(Path("/proc") / str(parent_pid) / "ns/net"),
        "inherited worker direct parent network namespace",
    )
    if parent_namespace != expected:
        raise CogneeWorkerProtocolError(
            "inherited worker direct parent network namespace differs"
        )
    if parent_privilege != {
        key: receipt[key]
        for key in (
            "capabilities",
            "gids",
            "groups",
            "no_new_privileges",
            "uids",
        )
    }:
        raise CogneeWorkerProtocolError(
            "inherited worker direct parent privilege receipt differs"
        )
    return {
        **receipt,
        "parent_network_namespace": parent_namespace,
        "parent_pid": parent_pid,
    }


def worker_scope_launch_arguments(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> tuple[str, ...]:
    """Return no legacy arguments or one exact custom launch configuration."""

    scope = exact_worker_scope(scope)
    if scope == DEFAULT_COGNEE_WORKER_SCOPE:
        return ()
    return (
        _SCOPE_ARGUMENTS[0],
        scope.dataset_name,
        _SCOPE_ARGUMENTS[1],
        scope.tenant_name,
        _SCOPE_ARGUMENTS[2],
        scope.node_set_name,
        _SCOPE_ARGUMENTS[3],
        scope.state_root,
    )


def worker_scope_from_launch_arguments(value: object) -> CogneeWorkerScope:
    """Parse only the exact ordered custom argv emitted by the parent."""

    if type(value) not in (tuple, list) or any(type(item) is not str for item in value):
        raise CogneeWorkerProtocolError(
            "worker scope launch arguments must be exact strings"
        )
    arguments = tuple(value)
    if not arguments:
        return DEFAULT_COGNEE_WORKER_SCOPE
    if len(arguments) != 8 or arguments[0::2] != _SCOPE_ARGUMENTS:
        raise CogneeWorkerProtocolError(
            "custom worker scope launch arguments differ"
        )
    scope = CogneeWorkerScope(
        dataset_name=arguments[1],
        tenant_name=arguments[3],
        node_set_name=arguments[5],
        state_root=arguments[7],
    )
    if worker_scope_launch_arguments(scope) != arguments:
        raise CogneeWorkerProtocolError(
            "custom worker scope launch arguments are not canonical"
        )
    return scope


def scope_marker_bytes(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> bytes:
    scope = exact_worker_scope(scope)
    marker = {
        "dataset_name": scope.dataset_name,
        "node_set_id": scope.node_set_id,
        "node_set_name": scope.node_set_name,
        "schema": "angler.cognee-worker-scope.v1",
        "state_root": scope.state_root,
        "tenant_name": scope.tenant_name,
        "worker_cwd": scope.worker_cwd,
        "worker_tmpdir": scope.worker_tmpdir,
    }
    encoded = json.dumps(
        marker,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"
    if len(encoded) > 4_096:
        raise CogneeWorkerProtocolError("worker scope marker exceeds 4096 bytes")
    return encoded


def _scope_directory_metadata(path: Path, label: str, *, private: bool) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CogneeWorkerProtocolError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CogneeWorkerProtocolError(f"{label} is not a real directory")
    if metadata.st_uid != 1000 or metadata.st_gid != 1000:
        raise CogneeWorkerProtocolError(f"{label} ownership differs")
    if private and metadata.st_mode & 0o7777 != 0o700:
        raise CogneeWorkerProtocolError(f"{label} permissions differ")
    return metadata


def validate_scope_path(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    *,
    require_root: bool,
) -> None:
    """Reject symlinked or cross-owner custom state ancestry."""

    scope = exact_worker_scope(scope)
    if scope == DEFAULT_COGNEE_WORKER_SCOPE:
        return
    root = Path(scope.state_root)
    allowlist_root = Path(COGNEE_SCOPE_ROOT)
    _scope_directory_metadata(allowlist_root, "Cognee scope allowlist root", private=False)
    current = allowlist_root
    relative = root.relative_to(allowlist_root)
    for component_index, component in enumerate(relative.parts):
        current = current / component
        if not current.exists() and not current.is_symlink():
            if require_root:
                raise CogneeWorkerProtocolError(
                    "custom worker scope root ancestry is unavailable"
                )
            return
        _scope_directory_metadata(
            current,
            "custom worker scope path",
            private=component_index == len(relative.parts) - 1,
        )
        if (
            component_index != len(relative.parts) - 1
            and (
                (current / SCOPE_MARKER_FILENAME).exists()
                or (current / SCOPE_MARKER_FILENAME).is_symlink()
            )
        ):
            raise CogneeWorkerProtocolError(
                "custom worker scope is nested below another scope"
            )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise CogneeWorkerProtocolError("custom worker scope root is unavailable") from exc
    if resolved != root:
        raise CogneeWorkerProtocolError("custom worker scope root differs after resolution")


def validate_scope_marker(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> None:
    """Validate a custom root's immutable name/path binding without mutation."""

    scope = exact_worker_scope(scope)
    if scope == DEFAULT_COGNEE_WORKER_SCOPE:
        return
    validate_scope_path(scope, require_root=True)
    marker = Path(scope.state_root) / SCOPE_MARKER_FILENAME
    try:
        metadata = marker.lstat()
    except OSError as exc:
        raise CogneeWorkerProtocolError("custom worker scope marker is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CogneeWorkerProtocolError("custom worker scope marker is not a regular file")
    if metadata.st_uid != 1000 or metadata.st_gid != 1000:
        raise CogneeWorkerProtocolError("custom worker scope marker ownership differs")
    if metadata.st_mode & 0o7777 != 0o600:
        raise CogneeWorkerProtocolError("custom worker scope marker permissions differ")
    expected = scope_marker_bytes(scope)
    if metadata.st_size != len(expected):
        raise CogneeWorkerProtocolError("custom worker scope marker differs")
    try:
        observed = marker.read_bytes()
    except OSError as exc:
        raise CogneeWorkerProtocolError("custom worker scope marker is unreadable") from exc
    if observed != expected:
        raise CogneeWorkerProtocolError("custom worker scope marker differs")


class CogneeWorkerRemoteError(RuntimeError):
    """A worker returned one bounded, stable operation failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = _error_code(code)
        self.remote_message = _error_message(message)
        super().__init__(f"{self.code}: {self.remote_message}")


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CogneeWorkerProtocolError(f"{label} must be an exact JSON object")
    for key in value:
        if type(key) is not str:
            raise CogneeWorkerProtocolError(f"{label} keys must be strings")
    return value


def _exact_fields(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    fields = frozenset(value)
    if fields != expected:
        missing = sorted(expected - fields)
        excess = sorted(fields - expected)
        raise CogneeWorkerProtocolError(
            f"{label} fields differ (missing={missing}, excess={excess})"
        )


def bounded_text(
    value: object,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
    allow_layout_controls: bool = False,
) -> str:
    if type(value) is not str or (not allow_empty and not value) or len(value) > maximum:
        qualifier = "text" if allow_empty else "non-empty text"
        raise CogneeWorkerProtocolError(
            f"{label} must be {qualifier} of at most {maximum} characters"
        )
    if value != unicodedata.normalize("NFC", value):
        raise CogneeWorkerProtocolError(f"{label} must be Unicode NFC-normalized")
    inspected = value.translate({9: None, 10: None, 13: None}) if allow_layout_controls else value
    if _CONTROL.search(inspected):
        raise CogneeWorkerProtocolError(f"{label} must not contain control characters")
    return value


def digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise CogneeWorkerProtocolError(f"{label} must be a lowercase sha256 reference")
    return value


def canonical_uuid(value: object, label: str) -> str:
    if type(value) is not str:
        raise CogneeWorkerProtocolError(f"{label} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise CogneeWorkerProtocolError(
            f"{label} must be a canonical UUID string"
        ) from exc
    if str(parsed) != value:
        raise CogneeWorkerProtocolError(f"{label} must be a canonical UUID string")
    return value


def modern_dataset_id(
    user_id: object,
    tenant_id: object,
    *,
    dataset_name: str = DATASET_NAME,
) -> str:
    user = canonical_uuid(user_id, "user_id")
    tenant = canonical_uuid(tenant_id, "tenant_id")
    dataset = _scope_name(dataset_name, "dataset_name")
    return str(uuid5(NAMESPACE_OID, f"{dataset}{user}{tenant}"))


def _request_id(value: object) -> int:
    if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
        raise CogneeWorkerProtocolError(
            "request id must be an integer from 1 through 2^63-1"
        )
    return value


def _error_code(value: object) -> str:
    if type(value) is not str or _ERROR_CODE.fullmatch(value) is None:
        raise CogneeWorkerProtocolError("worker error code is invalid")
    return value


def _error_message(value: object) -> str:
    return bounded_text(
        value,
        "worker error message",
        MAX_ERROR_MESSAGE_CHARACTERS,
    )


def stable_error_message(value: object) -> str:
    """Return bounded single-line text without leaking an exception traceback."""

    text = "Worker operation failed" if type(value) is not str else value
    text = " ".join(text.split())
    text = unicodedata.normalize("NFC", text)
    if not text:
        text = "Worker operation failed"
    return text[:MAX_ERROR_MESSAGE_CHARACTERS]


def encode_json_line(message: Mapping[str, Any]) -> bytes:
    if type(message) is not dict:
        raise CogneeWorkerProtocolError("message must be an exact JSON object")
    try:
        encoded = json.dumps(
            message,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise CogneeWorkerProtocolError("message is not canonical JSON data") from exc
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise CogneeWorkerProtocolError(
            f"JSON-lines message exceeds {MAX_MESSAGE_BYTES} bytes"
        )
    return encoded


def decode_json_line(line: bytes) -> dict[str, Any]:
    if type(line) is not bytes:
        raise CogneeWorkerProtocolError("JSON-lines frame must be exact bytes")
    if not line:
        raise CogneeWorkerProtocolError("worker stream ended before a response frame")
    if len(line) > MAX_MESSAGE_BYTES:
        raise CogneeWorkerProtocolError(
            f"JSON-lines message exceeds {MAX_MESSAGE_BYTES} bytes"
        )
    if not line.endswith(b"\n") or b"\n" in line[:-1] or b"\r" in line[:-1]:
        raise CogneeWorkerProtocolError("worker response is not one complete JSON line")
    try:
        decoded = json.loads(line[:-1].decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise CogneeWorkerProtocolError("worker response is not valid UTF-8 JSON") from exc
    decoded = _exact_dict(decoded, "message")
    try:
        canonical = encode_json_line(decoded)
    except CogneeWorkerProtocolError as exc:
        raise CogneeWorkerProtocolError(
            "worker response is not canonical JSON"
        ) from exc
    if canonical != line:
        raise CogneeWorkerProtocolError("worker response is not canonical JSON")
    return decoded


def validate_point_payload(
    value: object,
    *,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> dict[str, Any]:
    scope = exact_worker_scope(scope)
    point = _exact_dict(value, "point")
    _exact_fields(point, _POINT_FIELDS, "point")
    canonical_uuid(point["id"], "point.id")
    for field in (
        "acquisition_ref",
        "source_ref",
        "record_ref",
        "projection_ref",
    ):
        digest(point[field], f"point.{field}")
    bounded_text(point["source_contract"], "point.source_contract", 256)
    bounded_text(
        point["search_text"],
        "point.search_text",
        16_384,
        allow_layout_controls=True,
    )
    for field in ("visibility", "memory_kind", "epistemic_status"):
        bounded_text(point[field], f"point.{field}", 128)
    ordinal = point["acquired_ordinal"]
    if type(ordinal) is not int or ordinal < 0:
        raise CogneeWorkerProtocolError(
            "point.acquired_ordinal must be a non-negative integer"
        )

    adjacent = point["adjacent_record_refs"]
    if type(adjacent) is not list or len(adjacent) > MAX_POINT_NEIGHBORS:
        raise CogneeWorkerProtocolError(
            "point.adjacent_record_refs must be a list within the 384-reference ceiling"
        )
    for item in adjacent:
        digest(item, "point.adjacent_record_ref")
    if adjacent != sorted(set(adjacent)):
        raise CogneeWorkerProtocolError(
            "point.adjacent_record_refs must be unique and canonically sorted"
        )

    node_sets = point["belongs_to_set"]
    if type(node_sets) is not list or len(node_sets) != 1:
        raise CogneeWorkerProtocolError("point must belong to exactly one NodeSet")
    node_set = _exact_dict(node_sets[0], "point NodeSet")
    _exact_fields(node_set, frozenset(("id", "name")), "point NodeSet")
    canonical_uuid(node_set["id"], "point NodeSet id")
    if node_set["id"] != scope.node_set_id:
        raise CogneeWorkerProtocolError("point NodeSet id differs from Cognee identity")
    if (
        bounded_text(node_set["name"], "point NodeSet name", 256)
        != scope.node_set_name
    ):
        raise CogneeWorkerProtocolError("point NodeSet differs from the frozen scope")

    references = point["references"]
    if type(references) is not list or len(references) > MAX_POINT_NEIGHBORS:
        raise CogneeWorkerProtocolError(
            "point.references must be a list within the 384-reference ceiling"
        )
    reference_targets: set[str] = set()
    reference_keys: list[tuple[str, str]] = []
    for index, raw_reference in enumerate(references):
        reference = _exact_dict(raw_reference, f"point reference {index}")
        _exact_fields(
            reference,
            frozenset(("relationship_type", "properties", "target")),
            f"point reference {index}",
        )
        relationship_type = bounded_text(
            reference["relationship_type"],
            f"point reference {index} relationship_type",
            128,
        )
        properties = _exact_dict(
            reference["properties"], f"point reference {index} properties"
        )
        _exact_fields(
            properties,
            frozenset(("target_ref",)),
            f"point reference {index} properties",
        )
        target_ref = digest(
            properties["target_ref"], f"point reference {index} target_ref"
        )
        target = _exact_dict(reference["target"], f"point reference {index} target")
        _exact_fields(
            target,
            frozenset(("id", "record_ref")),
            f"point reference {index} target",
        )
        canonical_uuid(target["id"], f"point reference {index} target id")
        if digest(
            target["record_ref"], f"point reference {index} target record_ref"
        ) != target_ref:
            raise CogneeWorkerProtocolError(
                f"point reference {index} target identity differs from target_ref"
            )
        reference_targets.add(target_ref)
        reference_keys.append((relationship_type, target_ref))
    if reference_keys != sorted(set(reference_keys)):
        raise CogneeWorkerProtocolError(
            "point references must be unique and canonically sorted"
        )
    if reference_targets != set(adjacent):
        raise CogneeWorkerProtocolError(
            "point reference targets differ from adjacent_record_refs"
        )
    return point


def validate_request(
    message: object,
    *,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> dict[str, Any]:
    scope = exact_worker_scope(scope)
    request = _exact_dict(message, "request")
    common = frozenset(("v", "id", "op"))
    if request.get("v") != PROTOCOL_VERSION:
        raise CogneeWorkerProtocolError("worker protocol version differs")
    _request_id(request.get("id"))
    op = request.get("op")
    if op == "hello":
        _exact_fields(
            request,
            common
            | frozenset(
                (
                    "dataset_name",
                    "node_set_name",
                    "expected_model_manifest",
                    "expected_dataset_id",
                )
            ),
            "hello request",
        )
        if (
            bounded_text(request["dataset_name"], "dataset_name", 256)
            != scope.dataset_name
        ):
            raise CogneeWorkerProtocolError("dataset_name differs from frozen scope")
        if (
            bounded_text(request["node_set_name"], "node_set_name", 256)
            != scope.node_set_name
        ):
            raise CogneeWorkerProtocolError("node_set_name differs from frozen scope")
        if digest(request["expected_model_manifest"], "expected_model_manifest") != MODEL_MANIFEST:
            raise CogneeWorkerProtocolError("embedding model manifest differs")
        expected_dataset_id = request["expected_dataset_id"]
        if expected_dataset_id is not None:
            canonical_uuid(expected_dataset_id, "expected_dataset_id")
    elif op == "project":
        _exact_fields(request, common | frozenset(("point",)), "project request")
        validate_point_payload(request["point"], scope=scope)
    elif op == "search":
        _exact_fields(
            request, common | frozenset(("query", "limit")), "search request"
        )
        bounded_text(
            request["query"],
            "query",
            MAX_QUERY_CHARACTERS,
            allow_layout_controls=True,
        )
        limit = request["limit"]
        if type(limit) is not int or not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise CogneeWorkerProtocolError(
                "limit must be an integer from 1 through 256"
            )
    elif op in ("forget_namespace", "close"):
        _exact_fields(request, common, f"{op} request")
    else:
        raise CogneeWorkerProtocolError("worker operation is not supported")
    return request


def success_response(request_id: int, result: Any) -> dict[str, Any]:
    _request_id(request_id)
    response = {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "ok": True,
        "result": result,
    }
    encode_json_line(response)
    return response


def error_response(request_id: int, code: str, message: str) -> dict[str, Any]:
    _request_id(request_id)
    response = {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "ok": False,
        "error": {
            "code": _error_code(code),
            "message": _error_message(stable_error_message(message)),
        },
    }
    encode_json_line(response)
    return response


def validate_response(message: object, expected_id: int) -> Any:
    response = _exact_dict(message, "response")
    if response.get("v") != PROTOCOL_VERSION:
        raise CogneeWorkerProtocolError("worker protocol version differs")
    if _request_id(response.get("id")) != _request_id(expected_id):
        raise CogneeWorkerProtocolError("worker response id differs from request id")
    ok = response.get("ok")
    if type(ok) is not bool:
        raise CogneeWorkerProtocolError("worker response ok flag must be a bool")
    if ok:
        _exact_fields(
            response,
            frozenset(("v", "id", "ok", "result")),
            "success response",
        )
        return response["result"]
    _exact_fields(
        response,
        frozenset(("v", "id", "ok", "error")),
        "error response",
    )
    error = _exact_dict(response["error"], "worker error")
    _exact_fields(error, frozenset(("code", "message")), "worker error")
    raise CogneeWorkerRemoteError(error["code"], error["message"])


def build_worker_environment(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    *,
    inherited_network_namespace: str | None = None,
) -> dict[str, str]:
    """Return the complete clean environment admitted to the Cognee process."""

    scope = exact_worker_scope(scope)
    state = Path(
        STATE_ROOT if scope == DEFAULT_COGNEE_WORKER_SCOPE else scope.state_root
    )
    worker_tmpdir = (
        WORKER_TMPDIR
        if scope == DEFAULT_COGNEE_WORKER_SCOPE
        else scope.worker_tmpdir
    )
    values = {
        "LANG": "C.UTF-8",
        "PATH": "/opt/angler/venvs/cognee/bin:/usr/bin:/bin",
        "PYTHONPATH": REPOSITORY_SOURCE_ROOT,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUTF8": "1",
        "TMPDIR": worker_tmpdir,
        "TELEMETRY_DISABLED": "1",
        "COGNEE_TRACING_ENABLED": "false",
        "COGNEE_LOG_FILE": "false",
        "COGNEE_LOG_SEARCH_HISTORY": "false",
        "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DO_NOT_TRACK": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_LOG": "ERROR",
        "LITELLM_SET_VERBOSE": "False",
        "FASTEMBED_CACHE_PATH": MODEL_CACHE_ROOT,
        "TIKTOKEN_CACHE_DIR": TIKTOKEN_CACHE_ROOT,
        "CUSTOM_TIKTOKEN_CACHE_DIR": TIKTOKEN_CACHE_ROOT,
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": MODEL_NAME,
        "EMBEDDING_DIMENSIONS": str(MODEL_DIMENSIONS),
        "EMBEDDING_MAX_COMPLETION_TOKENS": "512",
        "EMBEDDING_BATCH_SIZE": "8",
        "EMBEDDING_MAX_CONCURRENT_DATA_POINTS": "8",
        "VECTOR_DB_PROVIDER": "lancedb",
        "VECTOR_DATASET_DATABASE_HANDLER": "lancedb",
        "VECTOR_DB_SUBPROCESS_ENABLED": "false",
        "VECTOR_DB_URL": str(state / "system/databases/global.lance.db"),
        "GRAPH_DATABASE_PROVIDER": "ladybug",
        "GRAPH_DATASET_DATABASE_HANDLER": "ladybug",
        "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
        "GRAPH_FILE_PATH": str(state / "system/databases"),
        "GRAPH_FILENAME": "global.lbug",
        "DB_PROVIDER": "sqlite",
        "DB_PATH": str(state / "system/databases"),
        "DB_NAME": "angler_cognee.sqlite",
        "DATA_ROOT_DIRECTORY": str(state / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(state / "system"),
        "CACHE_ROOT_DIRECTORY": str(state / "cache"),
        "LOGS_ROOT_DIRECTORY": str(state / "logs"),
        "COGNEE_LOGS_DIR": str(state / "logs"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "DEFAULT_FEEDBACK_INFLUENCE": "0.0",
        "PERSONALIZATION_ENABLED": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(FASTEMBED_SESSION_THREADS),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if inherited_network_namespace is not None:
        values[INHERITED_NETWORK_NAMESPACE_ENV] = canonical_network_namespace(
            inherited_network_namespace,
            "inherited worker network namespace",
        )
    for key, value in values.items():
        if "\x00" in key or "\x00" in value or "=" in key:
            raise CogneeWorkerProtocolError("worker environment is not executable")
    return values


def validate_exact_worker_environment(
    environment: Mapping[str, str] | None = None,
    *,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    inherited_network_namespace: str | None = None,
) -> None:
    current = dict(os.environ if environment is None else environment)
    expected = build_worker_environment(
        scope,
        inherited_network_namespace=inherited_network_namespace,
    )
    if current != expected:
        missing = sorted(set(expected) - set(current))
        excess = sorted(set(current) - set(expected))
        drifted = sorted(
            key
            for key in set(current) & set(expected)
            if current[key] != expected[key]
        )
        raise CogneeWorkerProtocolError(
            "worker environment differs from the clean allowlist "
            f"(missing={missing}, excess={excess}, drifted={drifted})"
        )


def validate_worker_cwd(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> None:
    scope = exact_worker_scope(scope)
    if os.getuid() != 1000 or os.getgid() != 1000:
        raise CogneeWorkerProtocolError("worker process identity differs")
    expected = Path(
        WORKER_CWD if scope == DEFAULT_COGNEE_WORKER_SCOPE else scope.worker_cwd
    )
    try:
        current = Path.cwd()
        metadata = current.stat()
    except OSError as exc:
        raise CogneeWorkerProtocolError("worker cwd is unavailable") from exc
    if current != expected or current.is_symlink():
        raise CogneeWorkerProtocolError("worker cwd differs from the controlled path")
    if metadata.st_uid != 1000 or metadata.st_gid != 1000:
        raise CogneeWorkerProtocolError("worker cwd ownership differs")
    if metadata.st_mode & 0o7777 != 0o700:
        raise CogneeWorkerProtocolError("worker cwd permissions differ")
    try:
        if next(current.iterdir(), None) is not None:
            raise CogneeWorkerProtocolError("worker cwd is not empty and .env-free")
    except OSError as exc:
        raise CogneeWorkerProtocolError("worker cwd cannot be inspected") from exc


def validate_worker_tmpdir(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> None:
    scope = exact_worker_scope(scope)
    path = Path(
        WORKER_TMPDIR
        if scope == DEFAULT_COGNEE_WORKER_SCOPE
        else scope.worker_tmpdir
    )
    if path.is_symlink():
        raise CogneeWorkerProtocolError("worker temp directory is a symlink")
    try:
        metadata = path.stat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CogneeWorkerProtocolError(
            "worker temp directory is unavailable"
        ) from exc
    if not path.is_dir() or resolved != path:
        raise CogneeWorkerProtocolError("worker temp directory differs")
    if metadata.st_uid != 1000 or metadata.st_gid != 1000:
        raise CogneeWorkerProtocolError("worker temp directory ownership differs")
    if metadata.st_mode & 0o7777 != 0o700:
        raise CogneeWorkerProtocolError("worker temp directory permissions differ")


def finite_score(value: object, label: str = "score") -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CogneeWorkerProtocolError(f"{label} must be finite or absent")
    return float(value)


__all__ = [
    "COGNEE_SCOPE_ROOT",
    "COGNEE_PYTHON",
    "COGNEE_PYTHON_VERSION",
    "COGNEE_VERSION",
    "COLLECTION_NAME",
    "CogneeWorkerScope",
    "CogneeWorkerProtocolError",
    "CogneeWorkerRemoteError",
    "DATASET_NAME",
    "DEFAULT_COGNEE_WORKER_SCOPE",
    "FASTEMBED_EXECUTION_PROVIDERS",
    "FASTEMBED_SESSION_THREADS",
    "INHERITED_NETWORK_NAMESPACE_ENV",
    "MAX_ERROR_MESSAGE_CHARACTERS",
    "MAX_MESSAGE_BYTES",
    "MAX_POINT_NEIGHBORS",
    "MAX_QUERY_CHARACTERS",
    "MAX_SEARCH_LIMIT",
    "MODEL_CACHE_ROOT",
    "MODEL_DIMENSIONS",
    "MODEL_MANIFEST",
    "MODEL_NAME",
    "NODE_SET_ID",
    "NODE_SET_NAME",
    "PROTOCOL_VERSION",
    "SCORE_SEMANTICS",
    "SCOPE_MARKER_FILENAME",
    "REPOSITORY_SOURCE_ROOT",
    "SOFTWARE_VERSIONS",
    "STATE_ROOT",
    "TENANT_NAME",
    "TIKTOKEN_CACHE_ROOT",
    "WORKER_CWD",
    "WORKER_TMPDIR",
    "attest_inherited_worker_parent_boundary",
    "bounded_text",
    "build_worker_environment",
    "canonical_network_namespace",
    "canonical_uuid",
    "decode_json_line",
    "digest",
    "encode_json_line",
    "error_response",
    "exact_worker_scope",
    "finite_score",
    "modern_dataset_id",
    "scope_marker_bytes",
    "stable_error_message",
    "success_response",
    "validate_exact_worker_environment",
    "validate_inherited_worker_boundary",
    "validate_point_payload",
    "validate_request",
    "validate_response",
    "validate_scope_marker",
    "validate_scope_path",
    "validate_worker_cwd",
    "validate_worker_tmpdir",
    "worker_scope_from_launch_arguments",
    "worker_scope_launch_arguments",
]
