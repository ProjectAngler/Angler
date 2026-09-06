"""Fail-closed bindings for the isolated, local Cognee acquisition worker.

The Angler environment intentionally does not import Cognee.  These bindings
therefore expose small structural shims to ``CogneeAcquisitionAdapter`` and
move only bounded, reference-oriented JSON across a networkless subprocess.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import os
from pathlib import Path
from typing import Any, Protocol

from .cognee_worker_protocol import (
    COGNEE_PYTHON,
    COGNEE_PYTHON_VERSION,
    COGNEE_VERSION,
    COLLECTION_NAME,
    DEFAULT_COGNEE_WORKER_SCOPE,
    FASTEMBED_EXECUTION_PROVIDERS,
    FASTEMBED_SESSION_THREADS,
    MAX_MESSAGE_BYTES,
    MAX_POINT_NEIGHBORS,
    MODEL_DIMENSIONS,
    MODEL_MANIFEST,
    MODEL_NAME,
    PROTOCOL_VERSION,
    SCORE_SEMANTICS,
    SCOPE_MARKER_FILENAME,
    SOFTWARE_VERSIONS,
    WORKER_CWD,
    WORKER_TMPDIR,
    CogneeWorkerScope,
    CogneeWorkerProtocolError,
    attest_inherited_worker_parent_boundary,
    bounded_text,
    build_worker_environment,
    canonical_uuid,
    decode_json_line,
    digest,
    encode_json_line,
    exact_worker_scope,
    finite_score,
    modern_dataset_id,
    scope_marker_bytes,
    validate_point_payload,
    validate_request,
    validate_response,
    validate_scope_marker,
    validate_scope_path,
    worker_scope_launch_arguments,
)


_WORKER_MODULE = "angler.memory.cognee_acquisition_worker"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_CONSTRUCTION_TOKEN = object()


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CogneeWorkerProtocolError(f"{label} must be an exact JSON object")
    return value


def _exact_fields(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected:
        raise CogneeWorkerProtocolError(f"{label} fields differ from the protocol")


class _WireDataPoint:
    """Pure-Python construction shim used by the existing adapter."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        super().__init_subclass__()

    def __init__(self, **values: Any) -> None:
        for key, value in values.items():
            if type(key) is not str or key.startswith("_"):
                raise TypeError("wire point field names must be public strings")
            setattr(self, key, value)


class _WireEdge(_WireDataPoint):
    pass


class _WireNodeSet(_WireDataPoint):
    pass


class _Transport(Protocol):
    async def exchange(self, request: dict[str, Any]) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class _SubprocessTransport:
    """One-request-at-a-time JSON-lines transport over a private subprocess."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._closed = False
        self._stderr_task = asyncio.create_task(self._discard_bounded_stderr())

    @staticmethod
    def launch_argv(
        scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
        *,
        inherited_network_namespace: str | None = None,
    ) -> tuple[str, ...]:
        scope = exact_worker_scope(scope)
        worker_environment = build_worker_environment(
            scope,
            inherited_network_namespace=inherited_network_namespace,
        )
        if inherited_network_namespace is not None:
            return (
                COGNEE_PYTHON,
                "-m",
                _WORKER_MODULE,
                *worker_scope_launch_arguments(scope),
            )
        assignments = tuple(
            f"{key}={worker_environment[key]}" for key in sorted(worker_environment)
        )
        return (
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/unshare",
            "--net",
            "--setgid=1000",
            "--setuid=1000",
            "--",
            "/usr/bin/env",
            "-i",
            *assignments,
            COGNEE_PYTHON,
            "-m",
            _WORKER_MODULE,
            *worker_scope_launch_arguments(scope),
        )

    @staticmethod
    def _prepare_custom_scope_root(scope: CogneeWorkerScope) -> None:
        scope = exact_worker_scope(scope)
        if scope == DEFAULT_COGNEE_WORKER_SCOPE:
            return
        validate_scope_path(scope, require_root=False)
        root = Path(scope.state_root)
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "custom worker scope root cannot be prepared"
            ) from exc
        validate_scope_path(scope, require_root=True)
        marker = root / SCOPE_MARKER_FILENAME
        expected = scope_marker_bytes(scope)
        try:
            marker.lstat()
        except FileNotFoundError:
            try:
                if next(root.iterdir(), None) is not None:
                    raise CogneeWorkerProtocolError(
                        "unbound custom worker scope root is not empty"
                    )
            except OSError as exc:
                raise CogneeWorkerProtocolError(
                    "custom worker scope root cannot be inspected"
                ) from exc
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "custom worker scope marker cannot be inspected"
            ) from exc
        else:
            validate_scope_marker(scope)
            return
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(marker, flags, 0o600)
        except FileExistsError:
            validate_scope_marker(scope)
            return
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "custom worker scope marker cannot be created"
            ) from exc
        try:
            os.fchmod(descriptor, 0o600)
            written = 0
            while written < len(expected):
                count = os.write(descriptor, expected[written:])
                if count <= 0:
                    raise OSError("scope marker write did not advance")
                written += count
            os.fsync(descriptor)
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "custom worker scope marker cannot be written"
            ) from exc
        finally:
            os.close(descriptor)
        validate_scope_marker(scope)

    @staticmethod
    def _prepare_worker_cwd(
        scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    ) -> str:
        scope = exact_worker_scope(scope)
        _SubprocessTransport._prepare_custom_scope_root(scope)
        path = Path(
            WORKER_CWD
            if scope == DEFAULT_COGNEE_WORKER_SCOPE
            else scope.worker_cwd
        )
        if path.is_symlink():
            raise CogneeWorkerProtocolError("controlled worker cwd is a symlink")
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = path.stat()
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "controlled worker cwd cannot be prepared"
            ) from exc
        if metadata.st_uid != 1000 or metadata.st_gid != 1000:
            raise CogneeWorkerProtocolError("controlled worker cwd ownership differs")
        if metadata.st_mode & 0o7777 != 0o700:
            raise CogneeWorkerProtocolError("controlled worker cwd permissions differ")
        try:
            if next(path.iterdir(), None) is not None:
                raise CogneeWorkerProtocolError(
                    "controlled worker cwd is not empty and .env-free"
                )
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "controlled worker cwd cannot be inspected"
            ) from exc
        return str(path)

    @staticmethod
    def _prepare_worker_tmpdir(
        scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    ) -> str:
        scope = exact_worker_scope(scope)
        _SubprocessTransport._prepare_custom_scope_root(scope)
        path = Path(
            WORKER_TMPDIR
            if scope == DEFAULT_COGNEE_WORKER_SCOPE
            else scope.worker_tmpdir
        )
        if path.is_symlink():
            raise CogneeWorkerProtocolError("controlled worker temp is a symlink")
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = path.stat()
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "controlled worker temp cannot be prepared"
            ) from exc
        if metadata.st_uid != 1000 or metadata.st_gid != 1000:
            raise CogneeWorkerProtocolError(
                "controlled worker temp ownership differs"
            )
        if metadata.st_mode & 0o7777 != 0o700:
            raise CogneeWorkerProtocolError(
                "controlled worker temp permissions differ"
            )
        try:
            if next(path.iterdir(), None) is not None:
                raise CogneeWorkerProtocolError(
                    "controlled worker temp is not empty"
                )
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "controlled worker temp cannot be inspected"
            ) from exc
        return str(path)

    @classmethod
    async def launch(
        cls,
        scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
        *,
        inherited_network_namespace: str | None = None,
    ) -> "_SubprocessTransport":
        scope = exact_worker_scope(scope)
        argv = cls.launch_argv(
            scope,
            inherited_network_namespace=inherited_network_namespace,
        )
        parent_boundary: dict[str, object] | None = None
        if inherited_network_namespace is not None:
            parent_boundary = attest_inherited_worker_parent_boundary(
                inherited_network_namespace
            )
        if scope == DEFAULT_COGNEE_WORKER_SCOPE:
            worker_cwd = cls._prepare_worker_cwd()
            worker_tmpdir = cls._prepare_worker_tmpdir()
        else:
            worker_cwd = cls._prepare_worker_cwd(scope)
            worker_tmpdir = cls._prepare_worker_tmpdir(scope)
        worker_environment = build_worker_environment(
            scope,
            inherited_network_namespace=inherited_network_namespace,
        )
        if worker_tmpdir != worker_environment["TMPDIR"]:
            raise CogneeWorkerProtocolError("controlled worker temp binding differs")
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=(
                {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"}
                if inherited_network_namespace is None
                else worker_environment
            ),
            cwd=worker_cwd,
            limit=MAX_MESSAGE_BYTES + 1,
        )
        transport = cls(process)
        if inherited_network_namespace is not None:
            try:
                rejoined = attest_inherited_worker_parent_boundary(
                    inherited_network_namespace
                )
                if rejoined != parent_boundary:
                    raise CogneeWorkerProtocolError(
                        "inherited worker parent boundary changed during launch"
                    )
            except BaseException as primary:
                try:
                    await transport.close()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "inherited worker launch boundary and cleanup failed",
                        [primary, cleanup],
                    ) from None
                raise
        return transport

    async def _discard_bounded_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        retained = 0
        while retained <= MAX_MESSAGE_BYTES:
            chunk = await stream.read(min(65_536, MAX_MESSAGE_BYTES + 1 - retained))
            if not chunk:
                return
            retained += len(chunk)
        while await stream.read(65_536):
            pass

    async def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise CogneeWorkerProtocolError("Cognee worker transport is closed")
        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise CogneeWorkerProtocolError("Cognee worker pipes are unavailable")
        if self._process.returncode is not None:
            raise CogneeWorkerProtocolError("Cognee worker exited unexpectedly")
        frame = encode_json_line(request)
        try:
            stdin.write(frame)
            await stdin.drain()
            response_frame = await stdout.readline()
        except (
            OSError,
            ConnectionError,
            asyncio.IncompleteReadError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise CogneeWorkerProtocolError(
                "Cognee worker communication failed"
            ) from exc
        return decode_json_line(response_frame)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.5)
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
        try:
            await asyncio.wait_for(self._stderr_task, timeout=1.0)
        except asyncio.TimeoutError:
            self._stderr_task.cancel()
        except Exception:
            pass


TransportFactory = Callable[[], Awaitable[_Transport]]


def _wire_uuid(value: object, label: str) -> str:
    return canonical_uuid(str(value), label)


def _wire_point(
    point: object,
    *,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> dict[str, Any]:
    scope = exact_worker_scope(scope)
    fields = (
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
    )
    if any(not hasattr(point, field) for field in fields):
        raise CogneeWorkerProtocolError("structured point is missing a wire field")
    instance_fields = getattr(point, "__dict__", None)
    if type(instance_fields) is not dict or frozenset(instance_fields) != frozenset(fields):
        raise CogneeWorkerProtocolError("structured point fields differ from the wire contract")

    node_sets = getattr(point, "belongs_to_set")
    if type(node_sets) is not list or len(node_sets) != 1:
        raise CogneeWorkerProtocolError("structured point must have one NodeSet")
    node_set = node_sets[0]
    if frozenset(getattr(node_set, "__dict__", {})) != frozenset(("id", "name")):
        raise CogneeWorkerProtocolError("structured NodeSet fields differ")
    serialized_node_set = {
        # The adapter owns an Angler-scoped local object identity.  The native
        # Cognee graph requires generate_node_id("NodeSet:" + name), so the IPC
        # boundary deliberately translates to that frozen identity.
        "id": scope.node_set_id,
        "name": getattr(node_set, "name", None),
    }
    _wire_uuid(getattr(node_set, "id", None), "adapter NodeSet id")

    serialized_references: list[dict[str, Any]] = []
    references = getattr(point, "references")
    if type(references) is not list:
        raise CogneeWorkerProtocolError("structured references must be a list")
    for reference in references:
        if type(reference) not in (tuple, list) or len(reference) != 2:
            raise CogneeWorkerProtocolError("structured reference must be an edge pair")
        edge, target = reference
        if frozenset(getattr(edge, "__dict__", {})) != frozenset(
            ("relationship_type", "properties")
        ):
            raise CogneeWorkerProtocolError("structured edge fields differ")
        if frozenset(getattr(target, "__dict__", {})) != frozenset(("id", "record_ref")):
            raise CogneeWorkerProtocolError("structured target fields differ")
        properties = getattr(edge, "properties", None)
        if type(properties) is not dict:
            raise CogneeWorkerProtocolError("structured edge properties must be an object")
        serialized_references.append(
            {
                "relationship_type": getattr(edge, "relationship_type", None),
                "properties": dict(properties),
                "target": {
                    "id": _wire_uuid(getattr(target, "id", None), "target id"),
                    "record_ref": getattr(target, "record_ref", None),
                },
            }
        )

    payload = {
        "id": _wire_uuid(getattr(point, "id"), "point id"),
        "acquisition_ref": getattr(point, "acquisition_ref"),
        "source_contract": getattr(point, "source_contract"),
        "source_ref": getattr(point, "source_ref"),
        "record_ref": getattr(point, "record_ref"),
        "projection_ref": getattr(point, "projection_ref"),
        "acquired_ordinal": getattr(point, "acquired_ordinal"),
        "search_text": getattr(point, "search_text"),
        "adjacent_record_refs": list(getattr(point, "adjacent_record_refs")),
        "references": serialized_references,
        "visibility": getattr(point, "visibility"),
        "memory_kind": getattr(point, "memory_kind"),
        "epistemic_status": getattr(point, "epistemic_status"),
        "belongs_to_set": [serialized_node_set],
    }
    validate_point_payload(payload, scope=scope)
    return payload


class CogneeSubprocessBindings:
    """Verified bindings pinned to one worker-proved Cognee namespace."""

    DataPoint = _WireDataPoint
    Edge = _WireEdge
    NodeSet = _WireNodeSet

    def __init__(
        self,
        *,
        token: object,
        transport: _Transport,
        tenant_id: str,
        dataset_id: str,
        user_id: str,
        embedding_execution_providers: tuple[str, ...],
        embedding_intra_op_threads: int,
        embedding_inter_op_threads: int,
        timeout_seconds: float,
        scope: CogneeWorkerScope,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use CogneeSubprocessBindings.start()")
        self._transport = transport
        self.tenant_id = tenant_id
        self.dataset_id = dataset_id
        self.user_id = user_id
        self._scope = exact_worker_scope(scope)
        self.score_semantics = SCORE_SEMANTICS
        self.backend_access_control = True
        self.embedding_execution_providers = embedding_execution_providers
        self.embedding_intra_op_threads = embedding_intra_op_threads
        self.embedding_inter_op_threads = embedding_inter_op_threads
        self._timeout_seconds = timeout_seconds
        self._lock = asyncio.Lock()
        self._next_request_id = 2
        self._usable = True
        self._closed = False

    @property
    def scope(self) -> CogneeWorkerScope:
        return self._scope

    @property
    def dataset_name(self) -> str:
        return self._scope.dataset_name

    @property
    def tenant_name(self) -> str:
        return self._scope.tenant_name

    @property
    def node_set_name(self) -> str:
        return self._scope.node_set_name

    @property
    def node_set_id(self) -> str:
        return self._scope.node_set_id

    @classmethod
    async def start(
        cls,
        *,
        expected_dataset_id: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
        inherited_network_namespace: str | None = None,
        _transport_factory: TransportFactory | None = None,
    ) -> "CogneeSubprocessBindings":
        scope = exact_worker_scope(scope)
        build_worker_environment(
            scope,
            inherited_network_namespace=inherited_network_namespace,
        )
        if expected_dataset_id is not None:
            canonical_uuid(expected_dataset_id, "expected_dataset_id")
        if type(timeout_seconds) not in (int, float) or not 0 < float(timeout_seconds) <= 300:
            raise ValueError("timeout_seconds must be greater than zero and at most 300")
        if (
            inherited_network_namespace is not None
            and _transport_factory is not None
        ):
            raise CogneeWorkerProtocolError(
                "inherited worker launch cannot use an injected transport"
            )
        if _transport_factory is None:
            transport = await _SubprocessTransport.launch(
                scope,
                inherited_network_namespace=inherited_network_namespace,
            )
        else:
            # Test transports deliberately remain zero-argument factories.
            transport = await _transport_factory()
        request = {
            "v": PROTOCOL_VERSION,
            "id": 1,
            "op": "hello",
            "dataset_name": scope.dataset_name,
            "node_set_name": scope.node_set_name,
            "expected_model_manifest": MODEL_MANIFEST,
            "expected_dataset_id": expected_dataset_id,
        }
        validate_request(request, scope=scope)
        try:
            raw_response = await asyncio.wait_for(
                transport.exchange(request), timeout=float(timeout_seconds)
            )
            result = validate_response(raw_response, 1)
            (
                tenant_id,
                dataset_id,
                user_id,
                embedding_execution_providers,
                embedding_intra_op_threads,
                embedding_inter_op_threads,
            ) = cls._validate_hello(
                result,
                expected_dataset_id=expected_dataset_id,
                scope=scope,
            )
        except BaseException:
            try:
                await transport.close()
            except Exception:
                pass
            raise
        return cls(
            token=_CONSTRUCTION_TOKEN,
            transport=transport,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            user_id=user_id,
            embedding_execution_providers=embedding_execution_providers,
            embedding_intra_op_threads=embedding_intra_op_threads,
            embedding_inter_op_threads=embedding_inter_op_threads,
            timeout_seconds=float(timeout_seconds),
            scope=scope,
        )

    @staticmethod
    def _validate_hello(
        value: object,
        *,
        expected_dataset_id: str | None,
        scope: CogneeWorkerScope,
    ) -> tuple[str, str, str, tuple[str, ...], int, int]:
        scope = exact_worker_scope(scope)
        result = _exact_dict(value, "hello result")
        _exact_fields(
            result,
            frozenset(
                (
                    "protocol_version",
                    "cognee_version",
                    "python_version",
                    "software_versions",
                    "tenant_id",
                    "user_id",
                    "dataset_id",
                    "dataset_name",
                    "node_set_name",
                    "node_set_id",
                    "collection_name",
                    "model_name",
                    "model_dimensions",
                    "model_manifest",
                    "backend_access_control",
                    "score_semantics",
                    "embedding_execution_providers",
                    "embedding_intra_op_threads",
                    "embedding_inter_op_threads",
                )
            ),
            "hello result",
        )
        if result["protocol_version"] != PROTOCOL_VERSION:
            raise CogneeWorkerProtocolError("hello protocol version differs")
        if result["cognee_version"] != COGNEE_VERSION:
            raise CogneeWorkerProtocolError("hello Cognee version differs")
        if result["python_version"] != COGNEE_PYTHON_VERSION:
            raise CogneeWorkerProtocolError("hello Python version differs")
        if result["software_versions"] != SOFTWARE_VERSIONS:
            raise CogneeWorkerProtocolError("hello software versions differ")
        tenant_id = canonical_uuid(result["tenant_id"], "hello tenant_id")
        user_id = canonical_uuid(result["user_id"], "hello user_id")
        dataset_id = canonical_uuid(result["dataset_id"], "hello dataset_id")
        expected = {
            "dataset_name": scope.dataset_name,
            "node_set_name": scope.node_set_name,
            "node_set_id": scope.node_set_id,
            "collection_name": COLLECTION_NAME,
            "model_name": MODEL_NAME,
            "model_dimensions": MODEL_DIMENSIONS,
            "model_manifest": MODEL_MANIFEST,
            "backend_access_control": True,
            "score_semantics": SCORE_SEMANTICS,
            "embedding_execution_providers": list(
                FASTEMBED_EXECUTION_PROVIDERS
            ),
            "embedding_intra_op_threads": FASTEMBED_SESSION_THREADS,
            "embedding_inter_op_threads": FASTEMBED_SESSION_THREADS,
        }
        for field, expected_value in expected.items():
            if result[field] != expected_value:
                raise CogneeWorkerProtocolError(f"hello {field} differs")
        if expected_dataset_id is not None and dataset_id != expected_dataset_id:
            raise CogneeWorkerProtocolError("hello dataset_id differs")
        if dataset_id != modern_dataset_id(
            user_id,
            tenant_id,
            dataset_name=scope.dataset_name,
        ):
            raise CogneeWorkerProtocolError(
                "hello dataset_id is not the modern user/tenant UUID5 identity"
            )
        providers = result["embedding_execution_providers"]
        intra_op_threads = result["embedding_intra_op_threads"]
        inter_op_threads = result["embedding_inter_op_threads"]
        if (
            type(providers) is not list
            or any(type(provider) is not str for provider in providers)
            or type(intra_op_threads) is not int
            or type(inter_op_threads) is not int
        ):
            raise CogneeWorkerProtocolError(
                "hello embedding runtime evidence is malformed"
            )
        return (
            tenant_id,
            dataset_id,
            user_id,
            tuple(providers),
            intra_op_threads,
            inter_op_threads,
        )

    def _require_usable(self) -> None:
        if not self._usable or self._closed:
            raise CogneeWorkerProtocolError("Cognee worker binding is unusable")

    async def _invalidate(self) -> None:
        self._usable = False
        try:
            await self._transport.close()
        except Exception:
            pass

    async def _request(self, operation: str, **parameters: Any) -> Any:
        self._require_usable()
        async with self._lock:
            self._require_usable()
            request_id = self._next_request_id
            self._next_request_id += 1
            request = {
                "v": PROTOCOL_VERSION,
                "id": request_id,
                "op": operation,
                **parameters,
            }
            validate_request(request, scope=self._scope)
            try:
                raw_response = await asyncio.wait_for(
                    self._transport.exchange(request),
                    timeout=self._timeout_seconds,
                )
                return validate_response(raw_response, request_id)
            except asyncio.CancelledError:
                await self._invalidate()
                raise
            except Exception:
                await self._invalidate()
                raise

    async def add_data_points(self, data_points: list[Any]) -> str:
        if type(data_points) is not list or len(data_points) != 1:
            raise CogneeWorkerProtocolError("exactly one data point is required")
        point = _wire_point(data_points[0], scope=self._scope)
        try:
            result = _exact_dict(
                await self._request("project", point=point), "project result"
            )
            _exact_fields(result, frozenset(("backend_ref",)), "project result")
            backend_ref = canonical_uuid(result["backend_ref"], "project backend_ref")
            if backend_ref != point["id"]:
                raise CogneeWorkerProtocolError("project backend_ref differs from point id")
            return backend_ref
        except Exception:
            if self._usable:
                await self._invalidate()
            raise

    def _validate_search_result(self, value: object, *, limit: int) -> list[dict[str, Any]]:
        if type(value) is not list or len(value) != 1:
            raise CogneeWorkerProtocolError("search must return one scope envelope")
        envelope = _exact_dict(value[0], "search envelope")
        _exact_fields(
            envelope,
            frozenset(
                (
                    "tenant_id",
                    "dataset_id",
                    "dataset_name",
                    "node_set_name",
                    "score_semantics",
                    "search_result",
                )
            ),
            "search envelope",
        )
        scope = {
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "node_set_name": self.node_set_name,
            "score_semantics": self.score_semantics,
        }
        for field, expected in scope.items():
            if envelope[field] != expected:
                raise CogneeWorkerProtocolError(f"search {field} differs from frozen scope")
        entries = envelope["search_result"]
        if type(entries) is not list or len(entries) > limit:
            raise CogneeWorkerProtocolError("search result exceeds the requested limit")
        previous_score: float | None = None
        seen_ids: set[str] = set()
        for index, raw_entry in enumerate(entries):
            entry = _exact_dict(raw_entry, f"search entry {index}")
            _exact_fields(entry, frozenset(("id", "score", "payload")), f"search entry {index}")
            entry_id = canonical_uuid(entry["id"], f"search entry {index} id")
            if entry_id in seen_ids:
                raise CogneeWorkerProtocolError("search result contains duplicate ids")
            seen_ids.add(entry_id)
            score = finite_score(entry["score"], f"search entry {index} score")
            if score is None:
                raise CogneeWorkerProtocolError("search scores must be finite")
            if previous_score is not None and score < previous_score:
                raise CogneeWorkerProtocolError(
                    "search scores are not in lower-is-better order"
                )
            previous_score = score
            payload = _exact_dict(entry["payload"], f"search entry {index} payload")
            _exact_fields(
                payload,
                frozenset(("record_ref", "adjacent_record_refs", "belongs_to_set")),
                f"search entry {index} payload",
            )
            digest(payload["record_ref"], f"search entry {index} record_ref")
            adjacent = payload["adjacent_record_refs"]
            if (
                type(adjacent) is not list
                or len(adjacent) > MAX_POINT_NEIGHBORS
                or adjacent != sorted(set(adjacent))
            ):
                raise CogneeWorkerProtocolError("search adjacency is not canonically sorted")
            for target_ref in adjacent:
                digest(target_ref, f"search entry {index} adjacent_record_ref")
            if payload["belongs_to_set"] != [self.node_set_name]:
                raise CogneeWorkerProtocolError("search NodeSet differs from frozen scope")
        return value

    async def search_references(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        bounded_text(query, "query", 16_384, allow_layout_controls=True)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        try:
            result = await self._request("search", query=query, limit=limit)
            return self._validate_search_result(result, limit=limit)
        except Exception:
            if self._usable:
                await self._invalidate()
            raise

    async def forget_namespace(self) -> None:
        try:
            result = _exact_dict(
                await self._request("forget_namespace"), "forget result"
            )
            _exact_fields(
                result,
                frozenset(
                    (
                        "tenant_id",
                        "dataset_id",
                        "dataset_name",
                        "node_set_name",
                        "recreated",
                    )
                ),
                "forget result",
            )
            expected = {
                "tenant_id": self.tenant_id,
                "dataset_id": self.dataset_id,
                "dataset_name": self.dataset_name,
                "node_set_name": self.node_set_name,
                "recreated": True,
            }
            if result != expected:
                raise CogneeWorkerProtocolError("forget result differs from frozen namespace")
        except Exception:
            if self._usable:
                await self._invalidate()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._usable:
            try:
                async with self._lock:
                    request_id = self._next_request_id
                    request = {"v": PROTOCOL_VERSION, "id": request_id, "op": "close"}
                    validate_request(request, scope=self._scope)
                    raw = await asyncio.wait_for(
                        self._transport.exchange(request), timeout=self._timeout_seconds
                    )
                    validate_response(raw, request_id)
            except Exception:
                pass
        self._usable = False
        await self._transport.close()

    async def __aenter__(self) -> "CogneeSubprocessBindings":
        self._require_usable()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def cognee_worker_launch_argv(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    *,
    inherited_network_namespace: str | None = None,
) -> tuple[str, ...]:
    """Expose the immutable argv for static qualification without launching it."""

    return _SubprocessTransport.launch_argv(
        scope,
        inherited_network_namespace=inherited_network_namespace,
    )


def cognee_worker_launch_cwd(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> str:
    """Return the controlled empty cwd pinned for every worker launch."""

    scope = exact_worker_scope(scope)
    return (
        WORKER_CWD
        if scope == DEFAULT_COGNEE_WORKER_SCOPE
        else scope.worker_cwd
    )


__all__ = [
    "CogneeSubprocessBindings",
    "cognee_worker_launch_argv",
    "cognee_worker_launch_cwd",
]
