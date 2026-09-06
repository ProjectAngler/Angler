from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import NAMESPACE_OID, UUID, uuid5

from angler.memory.cognitive_acquisition import CognitiveGraphProjectionV2
from angler.memory.cognee_acquisition_adapter import CogneeAcquisitionAdapter
from angler.memory.cognee_acquisition_worker import (
    _CogneeBackend,
    _ensure_qualification_tenant,
    _fastembed_runtime_evidence,
    _install_bounded_fastembed_constructor,
    _validate_persisted_dataset_storage,
    _validate_runtime_boundary,
)
from angler.memory.cognee_subprocess_bindings import (
    CogneeSubprocessBindings,
    _SubprocessTransport,
    cognee_worker_launch_argv,
    cognee_worker_launch_cwd,
)
from angler.memory.cognee_worker_protocol import (
    COGNEE_PYTHON,
    COGNEE_PYTHON_VERSION,
    COLLECTION_NAME,
    COGNEE_VERSION,
    CogneeWorkerScope,
    DATASET_NAME,
    DEFAULT_COGNEE_WORKER_SCOPE,
    FASTEMBED_EXECUTION_PROVIDERS,
    FASTEMBED_SESSION_THREADS,
    INHERITED_NETWORK_NAMESPACE_ENV,
    MAX_MESSAGE_BYTES,
    MODEL_DIMENSIONS,
    MODEL_MANIFEST,
    MODEL_NAME,
    NODE_SET_ID,
    NODE_SET_NAME,
    PROTOCOL_VERSION,
    SCORE_SEMANTICS,
    SCOPE_MARKER_FILENAME,
    SOFTWARE_VERSIONS,
    TENANT_NAME,
    WORKER_CWD,
    WORKER_TMPDIR,
    CogneeWorkerProtocolError,
    CogneeWorkerRemoteError,
    attest_inherited_worker_parent_boundary,
    build_worker_environment,
    decode_json_line,
    encode_json_line,
    error_response,
    modern_dataset_id,
    scope_marker_bytes,
    success_response,
    validate_exact_worker_environment,
    validate_inherited_worker_boundary,
    validate_scope_marker,
    worker_scope_from_launch_arguments,
    worker_scope_launch_arguments,
)
from tests.unit.memory.test_cognitive_acquisition_graph import _chain


_USER_ID = "11111111-1111-5111-8111-111111111111"
_TENANT_ID = "22222222-2222-5222-8222-222222222222"
_DATASET_ID = modern_dataset_id(_USER_ID, _TENANT_ID)
_BACKEND_ID = "93ff9bff-a7c9-5c21-b40d-6255d17b4a67"
_TARGET_ID = "5f2f5c35-b02f-5ab9-a559-f7bef2d6e025"
_NODE_SET_ID = "e2b3fc21-8156-51dc-b9b4-dacf1021d110"
_RECORD_REF = "sha256:" + "a" * 64
_TARGET_REF = "sha256:" + "b" * 64
_INHERITED_NETWORK_NAMESPACE = "net:[4026533001]"
_ZERO_CAPABILITY_STATUS = "\n".join(
    (
        "Uid:\t1000\t1000\t1000\t1000",
        "Gid:\t1000\t1000\t1000\t1000",
        "Groups:\t",
        "CapInh:\t0000000000000000",
        "CapPrm:\t0000000000000000",
        "CapEff:\t0000000000000000",
        "CapBnd:\t0000000000000000",
        "CapAmb:\t0000000000000000",
        "NoNewPrivs:\t1",
    )
) + "\n"
_LOOPBACK_IPV6_ROUTE = (
    "00000000000000000000000000000001 80 "
    "00000000000000000000000000000000 00 "
    "00000000000000000000000000000000 "
    "00000000 00000000 00000000 80200001 lo\n"
)


def _inherited_boundary_text_reader(
    changes: dict[str, str] | None = None,
):
    values = {
        "/proc/self/status": _ZERO_CAPABILITY_STATUS,
        "/proc/4242/status": _ZERO_CAPABILITY_STATUS,
        "/proc/net/route": (
            "Iface Destination Gateway Flags RefCnt Use Metric Mask "
            "MTU Window IRTT\n"
        ),
        "/proc/net/ipv6_route": _LOOPBACK_IPV6_ROUTE,
        "/sys/class/net/lo/operstate": "unknown\n",
    }
    values.update(changes or {})

    def read(path: Path) -> str:
        try:
            return values[str(path)]
        except KeyError as exc:
            raise AssertionError(f"unexpected attestation path: {path}") from exc

    return read


def _inherited_boundary_namespace_reader(
    changes: dict[str, str] | None = None,
):
    values = {
        "/proc/self/ns/net": _INHERITED_NETWORK_NAMESPACE,
        "/proc/4242/ns/net": _INHERITED_NETWORK_NAMESPACE,
    }
    values.update(changes or {})

    def read(path: Path) -> str:
        try:
            return values[str(path)]
        except KeyError as exc:
            raise AssertionError(f"unexpected namespace path: {path}") from exc

    return read


def _custom_scope(
    *,
    dataset_name: str = "angler-high-level-multidomain-v1-full-r1",
    state_root: str = (
        "/opt/angler/state/project-angler/high-level-multidomain-v1/"
        "cognee/full-r1"
    ),
) -> CogneeWorkerScope:
    return CogneeWorkerScope(
        dataset_name=dataset_name,
        tenant_name=f"{dataset_name}-tenant",
        node_set_name=f"{dataset_name}-records",
        state_root=state_root,
    )


def _write_dataset_storage_metadata(
    scope: CogneeWorkerScope,
    *,
    owner_id: str,
    dataset_id: str,
    vector_database_url: Path,
    graph_database_url: Path,
) -> Path:
    database_root = Path(scope.state_root) / "system" / "databases"
    database_root.mkdir(parents=True)
    (database_root / owner_id).mkdir()
    metadata = database_root / "angler_cognee.sqlite"
    connection = sqlite3.connect(metadata)
    try:
        connection.execute(
            """
            CREATE TABLE dataset_database (
                owner_id TEXT,
                dataset_id TEXT,
                vector_database_name TEXT,
                graph_database_name TEXT,
                vector_database_provider TEXT,
                graph_database_provider TEXT,
                graph_dataset_database_handler TEXT,
                vector_dataset_database_handler TEXT,
                vector_database_url TEXT,
                graph_database_url TEXT,
                graph_database_key TEXT,
                vector_database_key TEXT,
                graph_database_connection_info TEXT,
                vector_database_connection_info TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO dataset_database VALUES (
                ?, ?, ?, ?, 'lancedb', 'ladybug', 'ladybug', 'lancedb',
                ?, ?, NULL, NULL, '{}', '{}'
            )
            """,
            (
                owner_id,
                dataset_id,
                f"{dataset_id}.lance.db",
                f"{dataset_id}.lbug",
                str(vector_database_url),
                str(graph_database_url),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return metadata


def _hello_result(
    *,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    **changes: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "cognee_version": COGNEE_VERSION,
        "python_version": COGNEE_PYTHON_VERSION,
        "software_versions": dict(SOFTWARE_VERSIONS),
        "tenant_id": _TENANT_ID,
        "user_id": _USER_ID,
        "dataset_id": modern_dataset_id(
            _USER_ID,
            _TENANT_ID,
            dataset_name=scope.dataset_name,
        ),
        "dataset_name": scope.dataset_name,
        "node_set_name": scope.node_set_name,
        "node_set_id": scope.node_set_id,
        "collection_name": COLLECTION_NAME,
        "model_name": MODEL_NAME,
        "model_dimensions": MODEL_DIMENSIONS,
        "model_manifest": MODEL_MANIFEST,
        "backend_access_control": True,
        "score_semantics": SCORE_SEMANTICS,
        "embedding_execution_providers": list(FASTEMBED_EXECUTION_PROVIDERS),
        "embedding_intra_op_threads": FASTEMBED_SESSION_THREADS,
        "embedding_inter_op_threads": FASTEMBED_SESSION_THREADS,
    }
    result.update(changes)
    return result


def _empty_search_result(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> list[dict[str, object]]:
    return [
        {
            "tenant_id": _TENANT_ID,
            "dataset_id": modern_dataset_id(
                _USER_ID,
                _TENANT_ID,
                dataset_name=scope.dataset_name,
            ),
            "dataset_name": scope.dataset_name,
            "node_set_name": scope.node_set_name,
            "score_semantics": SCORE_SEMANTICS,
            "search_result": [],
        }
    ]


class FakeTransport:
    def __init__(self, responder=None) -> None:
        self.responder = responder or self._default_response
        self.requests: list[dict[str, object]] = []
        self.closed = False

    @staticmethod
    def _default_response(request: dict[str, object]) -> dict[str, object]:
        operation = request["op"]
        if operation == "hello":
            result: object = _hello_result()
        elif operation == "search":
            result = _empty_search_result()
        elif operation == "forget_namespace":
            result = {
                "tenant_id": _TENANT_ID,
                "dataset_id": _DATASET_ID,
                "dataset_name": DATASET_NAME,
                "node_set_name": NODE_SET_NAME,
                "recreated": True,
            }
        elif operation == "project":
            result = {"backend_ref": request["point"]["id"]}
        elif operation == "close":
            result = {"closed": True}
        else:
            raise AssertionError(f"unexpected fake operation: {operation}")
        return success_response(request["id"], result)

    async def exchange(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        response = self.responder(request)
        if inspect.isawaitable(response):
            response = await response
        return response

    async def close(self) -> None:
        self.closed = True


async def _start(
    fake: FakeTransport,
    *,
    timeout_seconds: float = 1.0,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> CogneeSubprocessBindings:
    async def factory() -> FakeTransport:
        return fake

    return await CogneeSubprocessBindings.start(
        expected_dataset_id=modern_dataset_id(
            _USER_ID,
            _TENANT_ID,
            dataset_name=scope.dataset_name,
        ),
        timeout_seconds=timeout_seconds,
        scope=scope,
        _transport_factory=factory,
    )


def _wire_point(bindings: CogneeSubprocessBindings):
    target = bindings.DataPoint(id=UUID(_TARGET_ID), record_ref=_TARGET_REF)
    edge = bindings.Edge(
        relationship_type="RELATES_TO", properties={"target_ref": _TARGET_REF}
    )
    node_set = bindings.NodeSet(
        id=UUID(_NODE_SET_ID), name=bindings.node_set_name
    )
    return bindings.DataPoint(
        id=UUID(_BACKEND_ID),
        acquisition_ref="sha256:" + "c" * 64,
        source_contract="ANG-CTR-COGNITIVE-ACQUISITION-V2",
        source_ref="sha256:" + "d" * 64,
        record_ref=_RECORD_REF,
        projection_ref="sha256:" + "e" * 64,
        acquired_ordinal=3,
        search_text="bounded synthetic reference point",
        adjacent_record_refs=[_TARGET_REF],
        references=[(edge, target)],
        visibility="PUBLIC",
        memory_kind="EPISODE",
        epistemic_status="VALIDATED",
        belongs_to_set=[node_set],
    )


def _point_payload(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> dict[str, object]:
    return {
        "id": _BACKEND_ID,
        "acquisition_ref": "sha256:" + "c" * 64,
        "source_contract": "ANG-CTR-COGNITIVE-ACQUISITION-V2",
        "source_ref": "sha256:" + "d" * 64,
        "record_ref": _RECORD_REF,
        "projection_ref": "sha256:" + "e" * 64,
        "acquired_ordinal": 3,
        "search_text": "bounded synthetic reference point",
        "adjacent_record_refs": [_TARGET_REF],
        "references": [
            {
                "relationship_type": "RELATES_TO",
                "properties": {"target_ref": _TARGET_REF},
                "target": {"id": _TARGET_ID, "record_ref": _TARGET_REF},
            }
        ],
        "visibility": "PUBLIC",
        "memory_kind": "EPISODE",
        "epistemic_status": "VALIDATED",
        "belongs_to_set": [
            {"id": scope.node_set_id, "name": scope.node_set_name}
        ],
    }


class _NativeValue:
    def __init__(self, **values) -> None:
        for key, value in values.items():
            setattr(self, key, value)

    def model_copy(self, *, deep: bool, update: dict[str, object]):
        if deep is not True:
            raise AssertionError("worker did not request a deep vector clone")
        values = vars(self).copy()
        values.update(update)
        return type(self)(**values)


class _NoopContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


def _backend(
    *,
    unified,
    add_data_points,
    index_data_points,
    forget,
    create_dataset,
    get_dataset,
    context_calls: list[tuple[object, object, object]],
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> _CogneeBackend:
    dataset_id = modern_dataset_id(
        _USER_ID,
        _TENANT_ID,
        dataset_name=scope.dataset_name,
    )
    user = SimpleNamespace(id=UUID(_USER_ID), tenant_id=UUID(_TENANT_ID))
    dataset = SimpleNamespace(
        id=UUID(dataset_id),
        owner_id=UUID(_USER_ID),
        tenant_id=UUID(_TENANT_ID),
    )

    def database_context(dataset_id, user_id, *, embedding_config):
        context_calls.append((dataset_id, user_id, embedding_config))
        return _NoopContext()

    async def get_unified():
        return unified

    bounded_constructor = object()
    fastembed_engine_module = SimpleNamespace(TextEmbedding=bounded_constructor)
    runtime_evidence = {
        "embedding_execution_providers": list(FASTEMBED_EXECUTION_PROVIDERS),
        "embedding_intra_op_threads": FASTEMBED_SESSION_THREADS,
        "embedding_inter_op_threads": FASTEMBED_SESSION_THREADS,
    }

    return _CogneeBackend(
        user=user,
        dataset=dataset,
        embedding_config="frozen-local-embedding-config",
        Point=_NativeValue,
        Target=_NativeValue,
        Edge=_NativeValue,
        NodeSet=_NativeValue,
        PipelineContext=_NativeValue,
        set_database_context=database_context,
        create_authorized_dataset=create_dataset,
        get_authorized_dataset_by_name=get_dataset,
        add_data_points=add_data_points,
        index_data_points=index_data_points,
        get_unified_engine=get_unified,
        backend_access_control_enabled=lambda: True,
        validate_runtime_boundary=lambda: None,
        fastembed_engine_module=fastembed_engine_module,
        bounded_fastembed_constructor=bounded_constructor,
        inspect_fastembed_runtime=lambda _vector: dict(runtime_evidence),
        embedding_runtime_evidence=runtime_evidence,
        forget=forget,
        scope=scope,
    )


class CogneeTenantBootstrapFakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_scope_tenant_name_reaches_worker_bootstrap(self) -> None:
        scope = _custom_scope()
        initial = SimpleNamespace(id=UUID(_USER_ID), tenant_id=None)
        reloaded = SimpleNamespace(
            id=UUID(_USER_ID), tenant_id=UUID(_TENANT_ID)
        )
        tenant = SimpleNamespace(
            id=UUID(_TENANT_ID),
            name=scope.tenant_name,
            owner_id=UUID(_USER_ID),
        )
        created: list[str] = []

        async def get_default_user():
            return reloaded

        async def get_user_tenants(user):
            if user is initial:
                return []
            return [{"id": _TENANT_ID, "name": scope.tenant_name}]

        async def get_tenant(tenant_id):
            return tenant

        async def create_tenant(name, user_id, *, set_as_active_tenant):
            self.assertEqual(user_id, UUID(_USER_ID))
            self.assertTrue(set_as_active_tenant)
            created.append(name)
            return UUID(_TENANT_ID)

        async def unused(*args, **kwargs):
            raise AssertionError("fresh custom scope must not select a tenant")

        user, verified = await _ensure_qualification_tenant(
            user=initial,
            get_default_user=get_default_user,
            get_user_tenants=get_user_tenants,
            get_tenant=get_tenant,
            create_tenant=create_tenant,
            select_tenant=unused,
            scope=scope,
        )
        self.assertIs(user, reloaded)
        self.assertIs(verified, tenant)
        self.assertEqual(created, [scope.tenant_name])

    async def test_fresh_scope_creates_one_owned_active_tenant_then_reloads(self) -> None:
        initial = SimpleNamespace(id=UUID(_USER_ID), tenant_id=None)
        reloaded = SimpleNamespace(
            id=UUID(_USER_ID), tenant_id=UUID(_TENANT_ID)
        )
        tenant = SimpleNamespace(
            id=UUID(_TENANT_ID), name=TENANT_NAME, owner_id=UUID(_USER_ID)
        )
        calls: list[tuple[object, ...]] = []

        async def get_default_user():
            calls.append(("reload",))
            return reloaded

        async def get_user_tenants(user):
            calls.append(("memberships", user.tenant_id))
            if user is initial:
                return []
            return [{"id": _TENANT_ID, "name": TENANT_NAME}]

        async def get_tenant(tenant_id):
            calls.append(("tenant", tenant_id))
            return tenant

        async def create_tenant(name, user_id, *, set_as_active_tenant):
            calls.append(("create", name, user_id, set_as_active_tenant))
            return UUID(_TENANT_ID)

        async def select_tenant(*args):
            raise AssertionError("fresh bootstrap must not select an old tenant")

        user, verified_tenant = await _ensure_qualification_tenant(
            user=initial,
            get_default_user=get_default_user,
            get_user_tenants=get_user_tenants,
            get_tenant=get_tenant,
            create_tenant=create_tenant,
            select_tenant=select_tenant,
        )

        self.assertIs(user, reloaded)
        self.assertIs(verified_tenant, tenant)
        self.assertEqual(
            calls,
            [
                ("memberships", None),
                ("create", TENANT_NAME, UUID(_USER_ID), True),
                ("tenant", UUID(_TENANT_ID)),
                ("reload",),
                ("memberships", UUID(_TENANT_ID)),
                ("tenant", UUID(_TENANT_ID)),
            ],
        )

    async def test_partial_scope_selects_sole_exact_owned_membership(self) -> None:
        initial = SimpleNamespace(id=UUID(_USER_ID), tenant_id=None)
        reloaded = SimpleNamespace(
            id=UUID(_USER_ID), tenant_id=UUID(_TENANT_ID)
        )
        tenant = SimpleNamespace(
            id=UUID(_TENANT_ID), name=TENANT_NAME, owner_id=UUID(_USER_ID)
        )
        selected: list[tuple[UUID, UUID]] = []

        async def get_default_user():
            return reloaded

        async def get_user_tenants(user):
            return [{"id": _TENANT_ID, "name": TENANT_NAME}]

        async def get_tenant(tenant_id):
            self.assertEqual(tenant_id, UUID(_TENANT_ID))
            return tenant

        async def create_tenant(*args, **kwargs):
            raise AssertionError("partial bootstrap must not create another tenant")

        async def select_tenant(user_id, tenant_id):
            selected.append((user_id, tenant_id))
            return reloaded

        user, verified_tenant = await _ensure_qualification_tenant(
            user=initial,
            get_default_user=get_default_user,
            get_user_tenants=get_user_tenants,
            get_tenant=get_tenant,
            create_tenant=create_tenant,
            select_tenant=select_tenant,
        )

        self.assertIs(user, reloaded)
        self.assertIs(verified_tenant, tenant)
        self.assertEqual(selected, [(UUID(_USER_ID), UUID(_TENANT_ID))])

    async def test_existing_active_scope_is_read_only_and_reverified(self) -> None:
        initial = SimpleNamespace(
            id=UUID(_USER_ID), tenant_id=UUID(_TENANT_ID)
        )
        reloaded = SimpleNamespace(
            id=UUID(_USER_ID), tenant_id=UUID(_TENANT_ID)
        )
        tenant = SimpleNamespace(
            id=UUID(_TENANT_ID), name=TENANT_NAME, owner_id=UUID(_USER_ID)
        )

        async def get_default_user():
            return reloaded

        async def get_user_tenants(user):
            return [{"id": _TENANT_ID, "name": TENANT_NAME}]

        async def get_tenant(tenant_id):
            return tenant

        async def unused(*args, **kwargs):
            raise AssertionError("verified active scope must not be mutated")

        user, verified_tenant = await _ensure_qualification_tenant(
            user=initial,
            get_default_user=get_default_user,
            get_user_tenants=get_user_tenants,
            get_tenant=get_tenant,
            create_tenant=unused,
            select_tenant=unused,
        )

        self.assertIs(user, reloaded)
        self.assertIs(verified_tenant, tenant)

    async def test_wrong_or_ambiguous_tenant_state_fails_closed(self) -> None:
        initial = SimpleNamespace(id=UUID(_USER_ID), tenant_id=None)
        wrong_id = "33333333-3333-5333-8333-333333333333"

        async def unused(*args, **kwargs):
            raise AssertionError("invalid scope must fail before mutation or reload")

        async def wrong_memberships(user):
            return [{"id": _TENANT_ID, "name": "foreign-tenant"}]

        with self.assertRaisesRegex(CogneeWorkerProtocolError, "qualification scope"):
            await _ensure_qualification_tenant(
                user=initial,
                get_default_user=unused,
                get_user_tenants=wrong_memberships,
                get_tenant=unused,
                create_tenant=unused,
                select_tenant=unused,
            )

        async def multiple_memberships(user):
            return [
                {"id": _TENANT_ID, "name": TENANT_NAME},
                {"id": wrong_id, "name": TENANT_NAME},
            ]

        with self.assertRaisesRegex(CogneeWorkerProtocolError, "qualification scope"):
            await _ensure_qualification_tenant(
                user=initial,
                get_default_user=unused,
                get_user_tenants=multiple_memberships,
                get_tenant=unused,
                create_tenant=unused,
                select_tenant=unused,
            )

    async def test_wrong_tenant_owner_is_rejected_before_selection(self) -> None:
        initial = SimpleNamespace(id=UUID(_USER_ID), tenant_id=None)

        async def get_user_tenants(user):
            return [{"id": _TENANT_ID, "name": TENANT_NAME}]

        async def get_tenant(tenant_id):
            return SimpleNamespace(
                id=tenant_id,
                name=TENANT_NAME,
                owner_id=UUID("33333333-3333-5333-8333-333333333333"),
            )

        async def unused(*args, **kwargs):
            raise AssertionError("wrong owner must fail before mutation or reload")

        with self.assertRaisesRegex(CogneeWorkerProtocolError, "tenant row"):
            await _ensure_qualification_tenant(
                user=initial,
                get_default_user=unused,
                get_user_tenants=get_user_tenants,
                get_tenant=get_tenant,
                create_tenant=unused,
                select_tenant=unused,
            )

    async def test_inactive_reloaded_user_is_rejected_after_creation(self) -> None:
        initial = SimpleNamespace(id=UUID(_USER_ID), tenant_id=None)
        inactive_reload = SimpleNamespace(id=UUID(_USER_ID), tenant_id=None)

        async def get_default_user():
            return inactive_reload

        async def get_user_tenants(user):
            return []

        async def get_tenant(tenant_id):
            return SimpleNamespace(
                id=tenant_id, name=TENANT_NAME, owner_id=UUID(_USER_ID)
            )

        async def create_tenant(name, user_id, *, set_as_active_tenant):
            return UUID(_TENANT_ID)

        async def unused(*args, **kwargs):
            raise AssertionError("fresh bootstrap must not select a tenant")

        with self.assertRaisesRegex(CogneeWorkerProtocolError, "active persisted"):
            await _ensure_qualification_tenant(
                user=initial,
                get_default_user=get_default_user,
                get_user_tenants=get_user_tenants,
                get_tenant=get_tenant,
                create_tenant=create_tenant,
                select_tenant=unused,
            )


class CogneeWorkerProtocolTests(unittest.TestCase):
    def test_scope_is_exported_from_memory_package(self) -> None:
        import angler.memory as memory

        self.assertIs(memory.CogneeWorkerScope, CogneeWorkerScope)
        self.assertIs(
            memory.DEFAULT_COGNEE_WORKER_SCOPE,
            DEFAULT_COGNEE_WORKER_SCOPE,
        )

    def test_legacy_scope_preserves_exact_environment_and_argv_bytes(self) -> None:
        self.assertEqual(
            DEFAULT_COGNEE_WORKER_SCOPE,
            CogneeWorkerScope(
                dataset_name=DATASET_NAME,
                tenant_name=TENANT_NAME,
                node_set_name=NODE_SET_NAME,
                state_root="/opt/angler/state/project-angler/cognee-acquisition-live-v1",
            ),
        )
        self.assertEqual(worker_scope_launch_arguments(), ())
        self.assertIs(
            worker_scope_from_launch_arguments([]),
            DEFAULT_COGNEE_WORKER_SCOPE,
        )
        environment_bytes = json.dumps(
            build_worker_environment(), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        argv_bytes = json.dumps(
            cognee_worker_launch_argv(), separators=(",", ":")
        ).encode("ascii")
        self.assertEqual(
            hashlib.sha256(environment_bytes).hexdigest(),
            "6aa4530c61e5844c4f142acf64b476b29bb5a633961c49db2f58e202adfef501",
        )
        self.assertEqual(
            hashlib.sha256(argv_bytes).hexdigest(),
            "1c28a429fa1281170c782ce380180f9004b4356cfbc4a9458d827c3da1e4bcca",
        )
        self.assertNotIn(
            INHERITED_NETWORK_NAMESPACE_ENV,
            build_worker_environment(),
        )

    def test_inherited_launch_is_direct_and_binds_one_exact_clean_environment(self) -> None:
        scope = _custom_scope()
        environment = build_worker_environment(
            scope,
            inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE,
        )
        self.assertEqual(
            environment[INHERITED_NETWORK_NAMESPACE_ENV],
            _INHERITED_NETWORK_NAMESPACE,
        )
        validate_exact_worker_environment(
            environment,
            scope=scope,
            inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE,
        )
        argv = cognee_worker_launch_argv(
            scope,
            inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE,
        )
        self.assertEqual(
            argv,
            (
                COGNEE_PYTHON,
                "-m",
                "angler.memory.cognee_acquisition_worker",
                *worker_scope_launch_arguments(scope),
            ),
        )
        self.assertFalse(
            any(
                item in argv
                for item in ("/usr/bin/sudo", "/usr/bin/unshare", "/usr/bin/env")
            )
        )
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "allowlist"):
            validate_exact_worker_environment(environment, scope=scope)
        for malformed in (
            "",
            "net:[0]",
            "net:[01]",
            "mnt:[4026533001]",
            "net:4026533001",
            4026533001,
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(CogneeWorkerProtocolError):
                    build_worker_environment(  # type: ignore[arg-type]
                        scope,
                        inherited_network_namespace=malformed,
                    )

    def test_inherited_parent_attestation_is_exact_and_reference_oriented(self) -> None:
        receipt = attest_inherited_worker_parent_boundary(
            _INHERITED_NETWORK_NAMESPACE,
            identity_reader=lambda: (1000, 1000, 1000, 1000, ()),
            interface_inventory=lambda: ((1, "lo"),),
            text_reader=_inherited_boundary_text_reader(),
            namespace_reader=_inherited_boundary_namespace_reader(),
        )
        self.assertEqual(
            receipt,
            {
                "capabilities": {
                    "CapInh": "0000000000000000",
                    "CapPrm": "0000000000000000",
                    "CapEff": "0000000000000000",
                    "CapBnd": "0000000000000000",
                    "CapAmb": "0000000000000000",
                },
                "gids": [1000, 1000, 1000, 1000],
                "groups": [],
                "interfaces": [[1, "lo"]],
                "ipv4_route_count": 0,
                "ipv6_route_count": 1,
                "loopback_operstate": "unknown",
                "network_namespace": _INHERITED_NETWORK_NAMESPACE,
                "no_new_privileges": True,
                "uids": [1000, 1000, 1000, 1000],
            },
        )
        canonical = json.dumps(
            receipt,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertNotIn("127.0.0.1", canonical)
        self.assertNotIn("00000000000000000000000000000001", canonical)

    def test_inherited_parent_attestation_rejects_every_privilege_and_network_drift(self) -> None:
        status_drifts = {
            "uid": _ZERO_CAPABILITY_STATUS.replace(
                "Uid:\t1000\t1000\t1000\t1000",
                "Uid:\t1000\t1000\t1000\t0",
            ),
            "gid": _ZERO_CAPABILITY_STATUS.replace(
                "Gid:\t1000\t1000\t1000\t1000",
                "Gid:\t1000\t1000\t1000\t0",
            ),
            "groups": _ZERO_CAPABILITY_STATUS.replace(
                "Groups:\t\n", "Groups:\t4\n"
            ),
            "no-new-privileges": _ZERO_CAPABILITY_STATUS.replace(
                "NoNewPrivs:\t1", "NoNewPrivs:\t0"
            ),
            "capability": _ZERO_CAPABILITY_STATUS.replace(
                "CapBnd:\t0000000000000000",
                "CapBnd:\t0000000000000001",
            ),
        }
        cases = [
            (
                label,
                {
                    "text_reader": _inherited_boundary_text_reader(
                        {"/proc/self/status": status}
                    )
                },
            )
            for label, status in status_drifts.items()
        ]
        cases.extend(
            (
                (
                    "os-identity",
                    {
                        "identity_reader": lambda: (
                            1000,
                            1000,
                            1000,
                            1000,
                            (4,),
                        )
                    },
                ),
                (
                    "network-namespace",
                    {
                        "namespace_reader": _inherited_boundary_namespace_reader(
                            {"/proc/self/ns/net": "net:[4026533002]"}
                        )
                    },
                ),
                (
                    "interfaces",
                    {"interface_inventory": lambda: ((1, "lo"), (2, "eth0"))},
                ),
                (
                    "ipv4-header",
                    {
                        "text_reader": _inherited_boundary_text_reader(
                            {"/proc/net/route": ""}
                        )
                    },
                ),
                (
                    "ipv4-route",
                    {
                        "text_reader": _inherited_boundary_text_reader(
                            {
                                "/proc/net/route": (
                                    "Iface Destination Gateway Flags RefCnt Use "
                                    "Metric Mask MTU Window IRTT\n"
                                    "lo 00000000 00000000 0001\n"
                                )
                            }
                        )
                    },
                ),
                (
                    "ipv6-route",
                    {
                        "text_reader": _inherited_boundary_text_reader(
                            {
                                "/proc/net/ipv6_route": (
                                    _LOOPBACK_IPV6_ROUTE.removesuffix("lo\n")
                                    + "eth0\n"
                                )
                            }
                        )
                    },
                ),
                (
                    "loopback-operstate",
                    {
                        "text_reader": _inherited_boundary_text_reader(
                            {"/sys/class/net/lo/operstate": "down\n"}
                        )
                    },
                ),
            )
        )
        for label, changes in cases:
            kwargs = {
                "identity_reader": lambda: (1000, 1000, 1000, 1000, ()),
                "interface_inventory": lambda: ((1, "lo"),),
                "text_reader": _inherited_boundary_text_reader(),
                "namespace_reader": _inherited_boundary_namespace_reader(),
                **changes,
            }
            with self.subTest(drift=label):
                with self.assertRaises(CogneeWorkerProtocolError):
                    attest_inherited_worker_parent_boundary(
                        _INHERITED_NETWORK_NAMESPACE,
                        **kwargs,
                    )

    def test_inherited_worker_rejoins_exact_direct_parent(self) -> None:
        receipt = validate_inherited_worker_boundary(
            _INHERITED_NETWORK_NAMESPACE,
            identity_reader=lambda: (1000, 1000, 1000, 1000, ()),
            parent_pid_reader=lambda: 4242,
            interface_inventory=lambda: ((1, "lo"),),
            text_reader=_inherited_boundary_text_reader(),
            namespace_reader=_inherited_boundary_namespace_reader(),
        )
        self.assertEqual(receipt["parent_pid"], 4242)
        self.assertEqual(
            receipt["parent_network_namespace"],
            _INHERITED_NETWORK_NAMESPACE,
        )
        worker_cases = (
            ("reparented", lambda: 1, {}, {}),
            (
                "parent-privilege",
                lambda: 4242,
                {
                    "/proc/4242/status": _ZERO_CAPABILITY_STATUS.replace(
                        "CapEff:\t0000000000000000",
                        "CapEff:\t0000000000000001",
                    )
                },
                {},
            ),
            (
                "parent-namespace",
                lambda: 4242,
                {},
                {"/proc/4242/ns/net": "net:[4026533002]"},
            ),
        )
        for label, parent_pid, text_changes, namespace_changes in worker_cases:
            with self.subTest(drift=label):
                with self.assertRaises(CogneeWorkerProtocolError):
                    validate_inherited_worker_boundary(
                        _INHERITED_NETWORK_NAMESPACE,
                        identity_reader=lambda: (1000, 1000, 1000, 1000, ()),
                        parent_pid_reader=parent_pid,
                        interface_inventory=lambda: ((1, "lo"),),
                        text_reader=_inherited_boundary_text_reader(text_changes),
                        namespace_reader=_inherited_boundary_namespace_reader(
                            namespace_changes
                        ),
                    )

    def test_worker_runtime_boundary_activates_inherited_attestation_only_when_bound(self) -> None:
        for inherited in (None, _INHERITED_NETWORK_NAMESPACE):
            environment = build_worker_environment(
                inherited_network_namespace=inherited
            )
            with (
                self.subTest(inherited=inherited),
                patch.dict(os.environ, environment, clear=True),
                patch(
                    "angler.memory.cognee_acquisition_worker.validate_scope_marker"
                ),
                patch(
                    "angler.memory.cognee_acquisition_worker.validate_worker_cwd"
                ),
                patch(
                    "angler.memory.cognee_acquisition_worker.validate_worker_tmpdir"
                ),
                patch(
                    "angler.memory.cognee_acquisition_worker."
                    "validate_inherited_worker_boundary"
                ) as inherited_attestation,
            ):
                _validate_runtime_boundary()
            if inherited is None:
                inherited_attestation.assert_not_called()
            else:
                inherited_attestation.assert_called_once_with(inherited)

    def test_custom_scope_argv_is_exact_ordered_and_round_trips(self) -> None:
        scope = _custom_scope()
        with self.assertRaises(AttributeError):
            scope.dataset_name = "cannot-switch"  # type: ignore[misc]
        arguments = (
            "--dataset-name",
            scope.dataset_name,
            "--tenant-name",
            scope.tenant_name,
            "--node-set-name",
            scope.node_set_name,
            "--state-root",
            scope.state_root,
        )
        self.assertEqual(worker_scope_launch_arguments(scope), arguments)
        self.assertEqual(worker_scope_from_launch_arguments(arguments), scope)
        argv = cognee_worker_launch_argv(scope)
        self.assertEqual(argv[-8:], arguments)
        python_index = argv.index(COGNEE_PYTHON)
        assignments = dict(
            argument.split("=", 1) for argument in argv[9:python_index]
        )
        self.assertEqual(assignments, build_worker_environment(scope))
        self.assertEqual(assignments["TMPDIR"], scope.worker_tmpdir)
        self.assertTrue(assignments["DB_PATH"].startswith(scope.state_root + "/"))
        validate_exact_worker_environment(assignments, scope=scope)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "drifted"):
            validate_exact_worker_environment(assignments)
        for malformed in (
            arguments[:-1],
            arguments + ("extra",),
            ("--tenant-name", scope.tenant_name, *arguments[2:]),
            (*arguments[:6], "--different-root", scope.state_root),
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(CogneeWorkerProtocolError):
                    worker_scope_from_launch_arguments(malformed)

    def test_scope_rejects_malformed_names_and_paths(self) -> None:
        invalid_names = (
            ("Upper", "Upper-tenant", "Upper-records"),
            ("under_score", "under_score-tenant", "under_score-records"),
            ("double--dash", "double--dash-tenant", "double--dash-records"),
            ("a" * 121, "a" * 121 + "-tenant", "a" * 121 + "-records"),
            ("valid", "different-tenant", "valid-records"),
            ("valid", "valid-tenant", "different-records"),
        )
        for dataset, tenant, node_set in invalid_names:
            with self.subTest(dataset=dataset, tenant=tenant, node_set=node_set):
                with self.assertRaises(CogneeWorkerProtocolError):
                    CogneeWorkerScope(
                        dataset_name=dataset,
                        tenant_name=tenant,
                        node_set_name=node_set,
                        state_root=(
                            "/opt/angler/state/project-angler/"
                            "high-level-multidomain-v1/cognee/invalid"
                        ),
                    )
        for state_root in (
            "relative/state",
            "/opt/angler/state/project-angler",
            "/opt/angler/state/project-angler/../escaped",
            "/opt/angler/state/project-angler/double//slash",
            "/tmp/outside",
        ):
            with self.subTest(state_root=state_root):
                with self.assertRaises(CogneeWorkerProtocolError):
                    _custom_scope(state_root=state_root)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "exact legacy"):
            _custom_scope(
                state_root=(
                    "/opt/angler/state/project-angler/"
                    "cognee-acquisition-live-v1"
                )
            )
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "legacy state tree"):
            _custom_scope(
                state_root=(
                    "/opt/angler/state/project-angler/"
                    "cognee-acquisition-live-v1/nested"
                )
            )

    def test_custom_scope_root_marker_is_private_exact_and_cross_scope_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope_root = Path(temporary_root) / "evaluation" / "cognee-full-r1"
                scope = _custom_scope(state_root=str(scope_root))
                cwd = _SubprocessTransport._prepare_worker_cwd(scope)
                tmpdir = _SubprocessTransport._prepare_worker_tmpdir(scope)
                marker = scope_root / SCOPE_MARKER_FILENAME
                self.assertEqual(cwd, scope.worker_cwd)
                self.assertEqual(tmpdir, scope.worker_tmpdir)
                self.assertEqual(scope_root.stat().st_mode & 0o7777, 0o700)
                self.assertEqual(Path(cwd).stat().st_mode & 0o7777, 0o700)
                self.assertEqual(Path(tmpdir).stat().st_mode & 0o7777, 0o700)
                self.assertEqual(marker.stat().st_mode & 0o7777, 0o600)
                self.assertEqual(marker.read_bytes(), scope_marker_bytes(scope))
                validate_scope_marker(scope)

                other = _custom_scope(
                    dataset_name="angler-high-level-multidomain-v1-other-r1",
                    state_root=str(scope_root),
                )
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "marker differs"
                ):
                    _SubprocessTransport._prepare_custom_scope_root(other)

    def test_persisted_dataset_storage_accepts_scope_local_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope = _custom_scope(
                    state_root=str(Path(temporary_root) / "evaluation" / "local")
                )
            dataset_id = modern_dataset_id(
                _USER_ID,
                _TENANT_ID,
                dataset_name=scope.dataset_name,
            )
            owner_root = (
                Path(scope.state_root) / "system" / "databases" / _USER_ID
            )
            _write_dataset_storage_metadata(
                scope,
                owner_id=_USER_ID,
                dataset_id=dataset_id,
                vector_database_url=owner_root / f"{dataset_id}.lance.db",
                graph_database_url=owner_root / f"{dataset_id}.lbug",
            )
            (owner_root / f"{dataset_id}.lance.db").mkdir()
            (owner_root / f"{dataset_id}.lbug").write_bytes(b"ladybug")

            with patch.dict(
                os.environ, {"DB_NAME": "angler_cognee.sqlite"}, clear=False
            ):
                _validate_persisted_dataset_storage(
                    scope,
                    expected_user_id=_USER_ID,
                    expected_dataset_id=dataset_id,
                )

    def test_persisted_dataset_storage_rejects_foreign_vector_without_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope = _custom_scope(
                    state_root=str(Path(temporary_root) / "evaluation" / "clone")
                )
            dataset_id = modern_dataset_id(
                _USER_ID,
                _TENANT_ID,
                dataset_name=scope.dataset_name,
            )
            owner_root = (
                Path(scope.state_root) / "system" / "databases" / _USER_ID
            )
            foreign_vector = (
                Path(temporary_root)
                / "source"
                / "system"
                / "databases"
                / _USER_ID
                / f"{dataset_id}.lance.db"
            )
            foreign_vector.mkdir(parents=True)
            sentinel = foreign_vector / "sentinel.bin"
            sentinel.write_bytes(b"source-vector-must-remain-unchanged")
            before_bytes = sentinel.read_bytes()
            before_stat = sentinel.stat()
            _write_dataset_storage_metadata(
                scope,
                owner_id=_USER_ID,
                dataset_id=dataset_id,
                vector_database_url=foreign_vector,
                graph_database_url=owner_root / f"{dataset_id}.lbug",
            )

            with (
                patch.dict(
                    os.environ, {"DB_NAME": "angler_cognee.sqlite"}, clear=False
                ),
                self.assertRaisesRegex(
                    CogneeWorkerProtocolError,
                    "vector storage escapes the frozen scope",
                ),
            ):
                _validate_persisted_dataset_storage(
                    scope,
                    expected_user_id=_USER_ID,
                    expected_dataset_id=dataset_id,
                )

            self.assertEqual(sentinel.read_bytes(), before_bytes)
            self.assertEqual(sentinel.stat().st_mtime_ns, before_stat.st_mtime_ns)
            self.assertEqual(tuple(foreign_vector.iterdir()), (sentinel,))

    def test_persisted_dataset_storage_rejects_foreign_graph_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope = _custom_scope(
                    state_root=str(Path(temporary_root) / "evaluation" / "clone")
                )
            dataset_id = modern_dataset_id(
                _USER_ID,
                _TENANT_ID,
                dataset_name=scope.dataset_name,
            )
            owner_root = (
                Path(scope.state_root) / "system" / "databases" / _USER_ID
            )
            foreign_graph = (
                Path(temporary_root)
                / "source"
                / "system"
                / "databases"
                / _USER_ID
                / f"{dataset_id}.lbug"
            )
            _write_dataset_storage_metadata(
                scope,
                owner_id=_USER_ID,
                dataset_id=dataset_id,
                vector_database_url=owner_root / f"{dataset_id}.lance.db",
                graph_database_url=foreign_graph,
            )
            (owner_root / f"{dataset_id}.lance.db").mkdir()

            with (
                patch.dict(
                    os.environ, {"DB_NAME": "angler_cognee.sqlite"}, clear=False
                ),
                self.assertRaisesRegex(
                    CogneeWorkerProtocolError,
                    "graph storage escapes the frozen scope",
                ),
            ):
                _validate_persisted_dataset_storage(
                    scope,
                    expected_user_id=_USER_ID,
                    expected_dataset_id=dataset_id,
                )

    def test_persisted_dataset_storage_rejects_scope_local_symlink_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope = _custom_scope(
                    state_root=str(Path(temporary_root) / "evaluation" / "clone")
                )
            dataset_id = modern_dataset_id(
                _USER_ID,
                _TENANT_ID,
                dataset_name=scope.dataset_name,
            )
            owner_root = (
                Path(scope.state_root) / "system" / "databases" / _USER_ID
            )
            vector_path = owner_root / f"{dataset_id}.lance.db"
            graph_path = owner_root / f"{dataset_id}.lbug"
            _write_dataset_storage_metadata(
                scope,
                owner_id=_USER_ID,
                dataset_id=dataset_id,
                vector_database_url=vector_path,
                graph_database_url=graph_path,
            )
            foreign_vector = Path(temporary_root) / "source-vector"
            foreign_vector.mkdir()
            vector_path.symlink_to(foreign_vector, target_is_directory=True)
            graph_path.write_bytes(b"ladybug")

            with self.assertRaisesRegex(
                CogneeWorkerProtocolError, "vector storage differs"
            ):
                _validate_persisted_dataset_storage(
                    scope,
                    expected_user_id=_USER_ID,
                    expected_dataset_id=dataset_id,
                )

    def test_persisted_dataset_storage_rejects_hardlinked_graph_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope = _custom_scope(
                    state_root=str(Path(temporary_root) / "evaluation" / "clone")
                )
            dataset_id = modern_dataset_id(
                _USER_ID,
                _TENANT_ID,
                dataset_name=scope.dataset_name,
            )
            owner_root = (
                Path(scope.state_root) / "system" / "databases" / _USER_ID
            )
            vector_path = owner_root / f"{dataset_id}.lance.db"
            graph_path = owner_root / f"{dataset_id}.lbug"
            _write_dataset_storage_metadata(
                scope,
                owner_id=_USER_ID,
                dataset_id=dataset_id,
                vector_database_url=vector_path,
                graph_database_url=graph_path,
            )
            vector_path.mkdir()
            foreign_graph = Path(temporary_root) / "source-graph.lbug"
            foreign_graph.write_bytes(b"ladybug")
            os.link(foreign_graph, graph_path)

            with self.assertRaisesRegex(
                CogneeWorkerProtocolError, "graph storage is aliased"
            ):
                _validate_persisted_dataset_storage(
                    scope,
                    expected_user_id=_USER_ID,
                    expected_dataset_id=dataset_id,
                )

    def test_custom_scope_trees_cannot_overlap_but_siblings_are_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                common = Path(temporary_root) / "evaluation"
                first = _custom_scope(state_root=str(common / "first"))
                second = _custom_scope(
                    dataset_name="angler-high-level-multidomain-v1-second-r1",
                    state_root=str(common / "second"),
                )
                _SubprocessTransport._prepare_custom_scope_root(first)
                _SubprocessTransport._prepare_custom_scope_root(second)
                validate_scope_marker(first)
                validate_scope_marker(second)

                nested = _custom_scope(
                    dataset_name="angler-high-level-multidomain-v1-nested-r1",
                    state_root=str(Path(first.state_root) / "nested"),
                )
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "nested below another scope"
                ):
                    _SubprocessTransport._prepare_custom_scope_root(nested)

                parent = Path(temporary_root) / "parent"
                child = _custom_scope(
                    dataset_name="angler-high-level-multidomain-v1-child-r1",
                    state_root=str(parent / "child"),
                )
                _SubprocessTransport._prepare_custom_scope_root(child)
                parent.chmod(0o700)
                parent_scope = _custom_scope(
                    dataset_name="angler-high-level-multidomain-v1-parent-r1",
                    state_root=str(parent),
                )
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "not empty"
                ):
                    _SubprocessTransport._prepare_custom_scope_root(parent_scope)

    def test_first_scope_marker_requires_an_empty_target_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope_root = Path(temporary_root) / "preexisting"
                scope_root.mkdir(mode=0o700)
                (scope_root / "foreign").write_text("not admitted", encoding="utf-8")
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "not empty"
                ):
                    _SubprocessTransport._prepare_custom_scope_root(
                        _custom_scope(state_root=str(scope_root))
                    )

    def test_custom_scope_rejects_symlinked_or_wrong_mode_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                target = Path(temporary_root) / "target"
                target.mkdir(mode=0o700)
                symlink = Path(temporary_root) / "symlink"
                symlink.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "real directory"
                ):
                    _SubprocessTransport._prepare_custom_scope_root(
                        _custom_scope(state_root=str(symlink))
                    )

                wrong_mode = Path(temporary_root) / "wrong-mode"
                wrong_mode.mkdir(mode=0o700)
                wrong_mode.chmod(0o750)
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "permissions differ"
                ):
                    _SubprocessTransport._prepare_custom_scope_root(
                        _custom_scope(state_root=str(wrong_mode))
                    )

                wrong_owner = Path(temporary_root) / "wrong-owner"
                wrong_owner.mkdir(mode=0o700)
                real_lstat = Path.lstat

                def lstat_with_wrong_owner(path):
                    metadata = real_lstat(path)
                    if path == wrong_owner:
                        return SimpleNamespace(
                            st_mode=metadata.st_mode,
                            st_uid=1001,
                            st_gid=metadata.st_gid,
                        )
                    return metadata

                with (
                    patch.object(
                        Path,
                        "lstat",
                        autospec=True,
                        side_effect=lstat_with_wrong_owner,
                    ),
                    self.assertRaisesRegex(
                        CogneeWorkerProtocolError, "ownership differs"
                    ),
                ):
                    _SubprocessTransport._prepare_custom_scope_root(
                        _custom_scope(state_root=str(wrong_owner))
                    )

    def test_json_lines_are_canonical_and_bounded_to_four_mib(self) -> None:
        exactly_full = {"x": "a" * (MAX_MESSAGE_BYTES - len(b'{"x":""}\n'))}
        frame = encode_json_line(exactly_full)
        self.assertEqual(len(frame), MAX_MESSAGE_BYTES)
        self.assertEqual(decode_json_line(frame), exactly_full)

        with self.assertRaisesRegex(CogneeWorkerProtocolError, "exceeds"):
            encode_json_line({"x": exactly_full["x"] + "a"})
        for malformed in (
            b'{"b":1, "a":2}\n',
            b'{"x":NaN}\n',
            b'{"x":1}\n{"x":2}\n',
            b'{"x":1,"x":1}\n',
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(CogneeWorkerProtocolError):
                    decode_json_line(malformed)

    def test_fastembed_constructor_binds_offline_cpu_session_before_use(self) -> None:
        calls: list[dict[str, object]] = []
        constructed = object()

        def native_text_embedding(**kwargs):
            calls.append(kwargs)
            return constructed

        engine_module = SimpleNamespace(TextEmbedding=native_text_embedding)
        bounded = _install_bounded_fastembed_constructor(
            engine_module=engine_module,
            native_text_embedding=native_text_embedding,
        )

        self.assertIs(engine_module.TextEmbedding, bounded)
        self.assertIs(bounded(model_name=MODEL_NAME), constructed)
        self.assertEqual(
            calls,
            [
                {
                    "model_name": MODEL_NAME,
                    "cache_dir": "/opt/angler/models/fastembed-cache-v1",
                    "threads": FASTEMBED_SESSION_THREADS,
                    "providers": FASTEMBED_EXECUTION_PROVIDERS,
                    "cuda": False,
                    "device_ids": None,
                    "lazy_load": False,
                    "local_files_only": True,
                }
            ],
        )
        for args, kwargs in (
            ((MODEL_NAME,), {}),
            ((), {"model_name": "different/model"}),
            ((), {"model_name": MODEL_NAME, "threads": 99}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(CogneeWorkerProtocolError):
                    bounded(*args, **kwargs)

    def test_fastembed_runtime_evidence_reads_actual_session_controls(self) -> None:
        class Session:
            def __init__(self) -> None:
                self.options = SimpleNamespace(
                    intra_op_num_threads=FASTEMBED_SESSION_THREADS,
                    inter_op_num_threads=FASTEMBED_SESSION_THREADS,
                )

            def get_providers(self):
                return list(FASTEMBED_EXECUTION_PROVIDERS)

            def get_session_options(self):
                return self.options

        OnnxTextEmbedding = type(
            "OnnxTextEmbedding",
            (),
            {"__module__": "fastembed.text.onnx_embedding"},
        )
        onnx_model = OnnxTextEmbedding()
        onnx_model.model_name = MODEL_NAME
        onnx_model.cache_dir = "/opt/angler/models/fastembed-cache-v1"
        onnx_model.threads = FASTEMBED_SESSION_THREADS
        onnx_model.providers = FASTEMBED_EXECUTION_PROVIDERS
        onnx_model.cuda = False
        onnx_model.device_ids = None
        onnx_model.device_id = None
        onnx_model.lazy_load = False
        onnx_model._local_files_only = True
        onnx_model.model = Session()

        TextEmbedding = type(
            "TextEmbedding",
            (),
            {"__module__": "fastembed.text.text_embedding"},
        )
        facade = TextEmbedding()
        facade.model_name = MODEL_NAME
        facade.cache_dir = "/opt/angler/models/fastembed-cache-v1"
        facade.threads = FASTEMBED_SESSION_THREADS
        facade._local_files_only = True
        facade.model = onnx_model

        FastembedEmbeddingEngine = type(
            "FastembedEmbeddingEngine",
            (),
            {
                "__module__": (
                    "cognee.infrastructure.databases.vector.embeddings."
                    "FastembedEmbeddingEngine"
                )
            },
        )
        engine = FastembedEmbeddingEngine()
        engine.model = MODEL_NAME
        engine.dimensions = MODEL_DIMENSIONS
        engine.embedding_model = facade
        vector = SimpleNamespace(embedding_engine=engine)

        self.assertEqual(
            _fastembed_runtime_evidence(vector),
            {
                "embedding_execution_providers": list(
                    FASTEMBED_EXECUTION_PROVIDERS
                ),
                "embedding_intra_op_threads": FASTEMBED_SESSION_THREADS,
                "embedding_inter_op_threads": FASTEMBED_SESSION_THREADS,
            },
        )
        onnx_model.model.options.inter_op_num_threads += 1
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "thread boundary"):
            _fastembed_runtime_evidence(vector)

    def test_worker_environment_is_exact_offline_allowlist(self) -> None:
        environment = build_worker_environment()
        validate_exact_worker_environment(environment)
        self.assertEqual(environment["TELEMETRY_DISABLED"], "1")
        self.assertEqual(environment["COGNEE_TRACING_ENABLED"], "false")
        self.assertEqual(environment["COGNEE_LOG_FILE"], "false")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["PYTHON_DOTENV_DISABLED"], "1")
        self.assertEqual(environment["LITELLM_LOG"], "ERROR")
        self.assertEqual(environment["LITELLM_SET_VERBOSE"], "False")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["EMBEDDING_PROVIDER"], "fastembed")
        self.assertEqual(environment["TMPDIR"], WORKER_TMPDIR)
        self.assertEqual(
            environment["OMP_NUM_THREADS"], str(FASTEMBED_SESSION_THREADS)
        )
        self.assertFalse(
            any(
                marker in key
                for key in environment
                for marker in ("OPENAI", "ANTHROPIC", "OLLAMA", "API_KEY", "LLM_PROVIDER")
            )
        )
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "allowlist"):
            validate_exact_worker_environment({**environment, "OPENAI_API_KEY": "forbidden"})

    def test_launch_uses_sudo_unshare_and_env_i_without_a_shell(self) -> None:
        argv = cognee_worker_launch_argv()
        self.assertEqual(
            argv[:9],
            (
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/unshare",
                "--net",
                "--setgid=1000",
                "--setuid=1000",
                "--",
                "/usr/bin/env",
                "-i",
            ),
        )
        python_index = argv.index(COGNEE_PYTHON)
        assignments = dict(
            argument.split("=", 1) for argument in argv[9:python_index]
        )
        self.assertEqual(assignments, build_worker_environment())
        self.assertEqual(
            argv[python_index + 1 :],
            ("-m", "angler.memory.cognee_acquisition_worker"),
        )

    def test_controlled_worker_cwd_rejects_any_dotenv_or_other_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            controlled = Path(temporary_root) / "worker-cwd"
            with patch(
                "angler.memory.cognee_subprocess_bindings.WORKER_CWD", str(controlled)
            ):
                self.assertEqual(
                    _SubprocessTransport._prepare_worker_cwd(), str(controlled)
                )
                (controlled / ".env").write_text(
                    "OPENAI_API_KEY=forbidden\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "empty and .env-free"
                ):
                    _SubprocessTransport._prepare_worker_cwd()

    def test_controlled_worker_temp_is_private_and_rejects_stale_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            controlled = Path(temporary_root) / "worker-tmp"
            with patch(
                "angler.memory.cognee_subprocess_bindings.WORKER_TMPDIR",
                str(controlled),
            ):
                self.assertEqual(
                    _SubprocessTransport._prepare_worker_tmpdir(), str(controlled)
                )
                self.assertEqual(controlled.stat().st_mode & 0o7777, 0o700)
                (controlled / "stale").write_text("inspect before reuse", encoding="utf-8")
                with self.assertRaisesRegex(
                    CogneeWorkerProtocolError, "not empty"
                ):
                    _SubprocessTransport._prepare_worker_tmpdir()


class CogneeAcquisitionWorkerFakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_scope_controls_worker_hello_and_native_nodeset(self) -> None:
        scope = _custom_scope()
        added = []

        async def add_data_points(points, **kwargs):
            added.extend(points)

        async def unused(*args, **kwargs):
            raise AssertionError("unexpected fake call")

        backend = _backend(
            unified=SimpleNamespace(vector=object()),
            add_data_points=add_data_points,
            index_data_points=unused,
            forget=unused,
            create_dataset=unused,
            get_dataset=unused,
            context_calls=[],
            scope=scope,
        )
        hello = backend.hello_result()
        self.assertEqual(hello["dataset_name"], scope.dataset_name)
        self.assertEqual(hello["node_set_name"], scope.node_set_name)
        self.assertEqual(hello["node_set_id"], scope.node_set_id)
        point = backend._point(_point_payload(scope))
        self.assertEqual(str(point.belongs_to_set[0].id), scope.node_set_id)
        self.assertEqual(point.belongs_to_set[0].name, scope.node_set_name)

    async def test_project_graph_writes_native_nodeset_then_indexes_string_clone(
        self,
    ) -> None:
        add_calls = []
        index_calls = []
        context_calls = []
        vector = object()

        async def add_data_points(points, **kwargs):
            add_calls.append((points, kwargs))

        async def index_data_points(points, **kwargs):
            index_calls.append((points, kwargs))

        async def unused(*args, **kwargs):
            raise AssertionError("unexpected fake call")

        backend = _backend(
            unified=SimpleNamespace(vector=vector),
            add_data_points=add_data_points,
            index_data_points=index_data_points,
            forget=unused,
            create_dataset=unused,
            get_dataset=unused,
            context_calls=context_calls,
        )
        result = await backend.project(_point_payload())

        self.assertEqual(result, {"backend_ref": _BACKEND_ID})
        self.assertEqual(len(add_calls), 1)
        native_point = add_calls[0][0][0]
        self.assertTrue(add_calls[0][1]["graph_only"])
        self.assertEqual(str(native_point.belongs_to_set[0].id), NODE_SET_ID)
        self.assertEqual(native_point.belongs_to_set[0].name, NODE_SET_NAME)
        self.assertEqual(len(index_calls), 1)
        vector_clone = index_calls[0][0][0]
        self.assertIsNot(vector_clone, native_point)
        self.assertEqual(vector_clone.belongs_to_set, [NODE_SET_NAME])
        self.assertIs(index_calls[0][1]["vector_engine"], vector)
        self.assertNotEqual(native_point.belongs_to_set, [NODE_SET_NAME])
        self.assertEqual(
            context_calls,
            [
                (
                    UUID(_DATASET_ID),
                    UUID(_USER_ID),
                    "frozen-local-embedding-config",
                )
            ],
        )

    async def test_search_reorders_unordered_join_and_proves_nodeset_edges(
        self,
    ) -> None:
        second_id = "6969d152-3591-5be3-a1e4-9b03f7b9a346"
        context_calls = []

        def node(node_id: str, record_ref: str):
            return {
                "id": node_id,
                "type": "AnglerAcquisitionReferencePoint",
                "acquisition_ref": "sha256:" + "c" * 64,
                "source_contract": "ANG-CTR-COGNITIVE-ACQUISITION-V2",
                "source_ref": "sha256:" + "d" * 64,
                "record_ref": record_ref,
                "projection_ref": "sha256:" + "e" * 64,
                "acquired_ordinal": 3,
                "search_text": "discarded Cognee text",
                "adjacent_record_refs": [],
                "visibility": "PUBLIC",
                "memory_kind": "EPISODE",
                "epistemic_status": "VALIDATED",
                "belongs_to_set": [NODE_SET_NAME],
            }

        class Vector:
            async def search(self, **kwargs):
                self.kwargs = kwargs
                return [
                    SimpleNamespace(id=UUID(_BACKEND_ID), score=0.1),
                    SimpleNamespace(id=UUID(second_id), score=0.2),
                ]

        class Graph:
            fail_membership = False

            async def get_nodes(self, node_ids):
                self.node_ids = node_ids
                return [node(second_id, _TARGET_REF), node(_BACKEND_ID, _RECORD_REF)]

            async def get_edges(self, node_id):
                if self.fail_membership and node_id == second_id:
                    return []
                return [
                    (
                        {"id": node_id},
                        "belongs_to_set",
                        {
                            "id": NODE_SET_ID,
                            "type": "NodeSet",
                            "name": NODE_SET_NAME,
                        },
                    )
                ]

        vector = Vector()
        graph = Graph()

        async def unused(*args, **kwargs):
            raise AssertionError("unexpected fake call")

        backend = _backend(
            unified=SimpleNamespace(vector=vector, graph=graph),
            add_data_points=unused,
            index_data_points=unused,
            forget=unused,
            create_dataset=unused,
            get_dataset=unused,
            context_calls=context_calls,
        )
        result = await backend.search("synthetic query", limit=2)

        self.assertEqual(graph.node_ids, [_BACKEND_ID, second_id])
        self.assertEqual(
            [entry["id"] for entry in result[0]["search_result"]],
            [_BACKEND_ID, second_id],
        )
        self.assertEqual(result[0]["score_semantics"], SCORE_SEMANTICS)
        self.assertEqual(
            frozenset(result[0]["search_result"][0]["payload"]),
            frozenset(("record_ref", "adjacent_record_refs", "belongs_to_set")),
        )
        self.assertNotIn("search_text", result[0]["search_result"][0]["payload"])
        self.assertEqual(vector.kwargs["node_name"], [NODE_SET_NAME])
        self.assertFalse(vector.kwargs["include_payload"])

        graph.fail_membership = True
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "adjacency"):
            await backend.search("synthetic query", limit=2)

    async def test_forget_converts_bound_string_to_uuid_and_recreates_same_scope(
        self,
    ) -> None:
        forget_calls = []
        context_calls = []
        recreated = SimpleNamespace(
            id=UUID(_DATASET_ID),
            owner_id=UUID(_USER_ID),
            tenant_id=UUID(_TENANT_ID),
        )

        async def forget(**kwargs):
            forget_calls.append(kwargs)

        async def create_dataset(name, user):
            self.assertEqual(name, DATASET_NAME)
            return recreated

        async def get_dataset(name, user, permission):
            self.assertEqual((name, permission), (DATASET_NAME, "delete"))
            return recreated

        async def unused(*args, **kwargs):
            raise AssertionError("unexpected fake call")

        backend = _backend(
            unified=SimpleNamespace(),
            add_data_points=unused,
            index_data_points=unused,
            forget=forget,
            create_dataset=create_dataset,
            get_dataset=get_dataset,
            context_calls=context_calls,
        )
        result = await backend.forget_namespace()

        self.assertEqual(len(forget_calls), 1)
        self.assertEqual(forget_calls[0]["dataset_id"], UUID(_DATASET_ID))
        self.assertIsInstance(forget_calls[0]["dataset_id"], UUID)
        self.assertEqual(frozenset(forget_calls[0]), frozenset(("dataset_id", "user")))
        self.assertEqual(result["dataset_id"], _DATASET_ID)
        self.assertTrue(result["recreated"])


class CogneeSubprocessBindingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_inherited_transport_launch_uses_direct_python_and_exact_environment(
        self,
    ) -> None:
        expected_argv = cognee_worker_launch_argv(
            inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE
        )
        expected_environment = build_worker_environment(
            inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE
        )
        with (
            patch(
                "angler.memory.cognee_subprocess_bindings."
                "attest_inherited_worker_parent_boundary",
                side_effect=(
                    {"network_namespace": _INHERITED_NETWORK_NAMESPACE},
                    {"network_namespace": _INHERITED_NETWORK_NAMESPACE},
                ),
            ) as attest_parent,
            patch.object(
                _SubprocessTransport,
                "_prepare_worker_cwd",
                return_value=WORKER_CWD,
            ),
            patch.object(
                _SubprocessTransport,
                "_prepare_worker_tmpdir",
                return_value=WORKER_TMPDIR,
            ),
            patch.object(_SubprocessTransport, "__init__", return_value=None),
            patch(
                "angler.memory.cognee_subprocess_bindings."
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=object(),
            ) as create_process,
        ):
            transport = await _SubprocessTransport.launch(
                inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE
            )
        self.assertIsInstance(transport, _SubprocessTransport)
        create_process.assert_awaited_once()
        call = create_process.await_args
        self.assertEqual(call.args, expected_argv)
        self.assertEqual(call.kwargs["env"], expected_environment)
        self.assertEqual(call.kwargs["cwd"], WORKER_CWD)
        self.assertEqual(call.kwargs["limit"], MAX_MESSAGE_BYTES + 1)
        self.assertEqual(attest_parent.call_count, 2)

    async def test_inherited_transport_attests_before_creation_and_closes_on_rejoin_drift(
        self,
    ) -> None:
        with (
            patch(
                "angler.memory.cognee_subprocess_bindings."
                "attest_inherited_worker_parent_boundary",
                side_effect=CogneeWorkerProtocolError("synthetic preflight stop"),
            ),
            patch.object(
                _SubprocessTransport, "_prepare_worker_cwd"
            ) as prepare_cwd,
            patch.object(
                _SubprocessTransport, "_prepare_worker_tmpdir"
            ) as prepare_tmpdir,
            patch(
                "angler.memory.cognee_subprocess_bindings."
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
            ) as create_process,
            self.assertRaisesRegex(
                CogneeWorkerProtocolError, "synthetic preflight stop"
            ),
        ):
            await _SubprocessTransport.launch(
                inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE
            )
        prepare_cwd.assert_not_called()
        prepare_tmpdir.assert_not_called()
        create_process.assert_not_awaited()

        class EmptyStream:
            async def read(self, _maximum: int) -> bytes:
                return b""

        class DirectProcess:
            def __init__(self) -> None:
                self.stdin = None
                self.stderr = EmptyStream()
                self.returncode = None
                self.terminated = False

            async def wait(self) -> int:
                self.returncode = 0
                return 0

            def terminate(self) -> None:
                self.terminated = True
                self.returncode = 1

            def kill(self) -> None:
                self.returncode = 9

        process = DirectProcess()
        with (
            patch(
                "angler.memory.cognee_subprocess_bindings."
                "attest_inherited_worker_parent_boundary",
                side_effect=(
                    {"network_namespace": _INHERITED_NETWORK_NAMESPACE},
                    {
                        "network_namespace": _INHERITED_NETWORK_NAMESPACE,
                        "drift": True,
                    },
                ),
            ),
            patch.object(
                _SubprocessTransport,
                "_prepare_worker_cwd",
                return_value=WORKER_CWD,
            ),
            patch.object(
                _SubprocessTransport,
                "_prepare_worker_tmpdir",
                return_value=WORKER_TMPDIR,
            ),
            patch(
                "angler.memory.cognee_subprocess_bindings."
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=process,
            ),
            self.assertRaisesRegex(
                CogneeWorkerProtocolError,
                "changed during launch",
            ),
        ):
            await _SubprocessTransport.launch(
                inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE
            )
        self.assertIsNotNone(process.returncode)

    async def test_inherited_start_rejects_transport_injection_before_launch(self) -> None:
        factory = AsyncMock()
        with self.assertRaisesRegex(
            CogneeWorkerProtocolError,
            "cannot use an injected transport",
        ):
            await CogneeSubprocessBindings.start(
                inherited_network_namespace=_INHERITED_NETWORK_NAMESPACE,
                _transport_factory=factory,
            )
        factory.assert_not_awaited()

    async def test_invalid_custom_root_fails_before_subprocess_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with patch(
                "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                temporary_root,
            ):
                scope_root = Path(temporary_root) / "occupied"
                scope_root.mkdir(mode=0o700)
                (scope_root / "foreign").write_bytes(b"not admitted")
                scope = _custom_scope(state_root=str(scope_root))
                with patch(
                    "angler.memory.cognee_subprocess_bindings.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                ) as create_process:
                    with self.assertRaisesRegex(
                        CogneeWorkerProtocolError, "not empty"
                    ):
                        await CogneeSubprocessBindings.start(scope=scope)
                create_process.assert_not_awaited()

    async def test_verified_hello_freezes_all_binding_scope(self) -> None:
        fake = FakeTransport()
        bindings = await _start(fake)
        self.assertEqual(bindings.tenant_id, _TENANT_ID)
        self.assertEqual(bindings.user_id, _USER_ID)
        self.assertEqual(bindings.dataset_id, _DATASET_ID)
        self.assertEqual(bindings.dataset_name, DATASET_NAME)
        self.assertEqual(bindings.node_set_name, NODE_SET_NAME)
        self.assertEqual(bindings.node_set_id, NODE_SET_ID)
        self.assertEqual(bindings.score_semantics, SCORE_SEMANTICS)
        self.assertTrue(bindings.backend_access_control)
        self.assertEqual(
            bindings.embedding_execution_providers,
            FASTEMBED_EXECUTION_PROVIDERS,
        )
        self.assertEqual(
            bindings.embedding_intra_op_threads, FASTEMBED_SESSION_THREADS
        )
        self.assertEqual(
            bindings.embedding_inter_op_threads, FASTEMBED_SESSION_THREADS
        )
        self.assertEqual(
            fake.requests,
            [
                {
                    "v": PROTOCOL_VERSION,
                    "id": 1,
                    "op": "hello",
                    "dataset_name": DATASET_NAME,
                    "node_set_name": NODE_SET_NAME,
                    "expected_model_manifest": MODEL_MANIFEST,
                    "expected_dataset_id": _DATASET_ID,
                }
            ],
        )
        await bindings.close()

    async def test_custom_scope_binds_hello_point_search_and_forget(self) -> None:
        scope = _custom_scope()
        dataset_id = modern_dataset_id(
            _USER_ID,
            _TENANT_ID,
            dataset_name=scope.dataset_name,
        )

        def responder(request):
            operation = request["op"]
            if operation == "hello":
                result: object = _hello_result(scope=scope)
            elif operation == "project":
                result = {"backend_ref": request["point"]["id"]}
            elif operation == "search":
                result = _empty_search_result(scope)
            elif operation == "forget_namespace":
                result = {
                    "tenant_id": _TENANT_ID,
                    "dataset_id": dataset_id,
                    "dataset_name": scope.dataset_name,
                    "node_set_name": scope.node_set_name,
                    "recreated": True,
                }
            elif operation == "close":
                result = {"closed": True}
            else:
                raise AssertionError(f"unexpected operation: {operation}")
            return success_response(request["id"], result)

        fake = FakeTransport(responder)
        bindings = await _start(fake, scope=scope)
        self.assertEqual(bindings.dataset_id, dataset_id)
        self.assertEqual(bindings.dataset_name, scope.dataset_name)
        self.assertEqual(bindings.tenant_name, scope.tenant_name)
        self.assertEqual(bindings.node_set_name, scope.node_set_name)
        self.assertEqual(bindings.node_set_id, scope.node_set_id)
        self.assertIs(bindings.scope, scope)
        with self.assertRaises(AttributeError):
            bindings.dataset_name = DATASET_NAME  # type: ignore[misc]
        self.assertEqual(
            fake.requests[0],
            {
                "v": PROTOCOL_VERSION,
                "id": 1,
                "op": "hello",
                "dataset_name": scope.dataset_name,
                "node_set_name": scope.node_set_name,
                "expected_model_manifest": MODEL_MANIFEST,
                "expected_dataset_id": dataset_id,
            },
        )
        await bindings.add_data_points([_wire_point(bindings)])
        self.assertEqual(
            fake.requests[-1]["point"]["belongs_to_set"],
            [{"id": scope.node_set_id, "name": scope.node_set_name}],
        )
        self.assertEqual(
            await bindings.search_references("query", limit=1),
            _empty_search_result(scope),
        )
        await bindings.forget_namespace()
        self.assertEqual(
            fake.requests[-1],
            {"v": PROTOCOL_VERSION, "id": 4, "op": "forget_namespace"},
        )
        await bindings.close()

    async def test_custom_scope_rejects_default_hello_and_nodeset_injection(self) -> None:
        scope = _custom_scope()

        def wrong_hello(request):
            return success_response(request["id"], _hello_result())

        fake = FakeTransport(wrong_hello)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "dataset_name"):
            await _start(fake, scope=scope)
        self.assertTrue(fake.closed)

        def responder(request):
            if request["op"] == "hello":
                return success_response(
                    request["id"], _hello_result(scope=scope)
                )
            return FakeTransport._default_response(request)

        point_fake = FakeTransport(responder)
        bindings = await _start(point_fake, scope=scope)
        point = _wire_point(bindings)
        point.belongs_to_set[0].name = NODE_SET_NAME
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "frozen scope"):
            await bindings.add_data_points([point])
        await bindings.close()

        def wrong_search(request):
            if request["op"] == "hello":
                return success_response(
                    request["id"], _hello_result(scope=scope)
                )
            if request["op"] == "search":
                return success_response(request["id"], _empty_search_result())
            return FakeTransport._default_response(request)

        search_fake = FakeTransport(wrong_search)
        search_bindings = await _start(search_fake, scope=scope)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "dataset_id"):
            await search_bindings.search_references("query", limit=1)
        self.assertTrue(search_fake.closed)

    async def test_two_custom_bindings_do_not_share_mutable_scope_state(self) -> None:
        first_scope = _custom_scope()
        second_scope = _custom_scope(
            dataset_name="angler-high-level-multidomain-v1-full-r2",
            state_root=(
                "/opt/angler/state/project-angler/high-level-multidomain-v1/"
                "cognee/full-r2"
            ),
        )

        def transport_for(scope):
            def responder(request):
                if request["op"] == "hello":
                    return success_response(
                        request["id"], _hello_result(scope=scope)
                    )
                if request["op"] == "search":
                    return success_response(
                        request["id"], _empty_search_result(scope)
                    )
                return FakeTransport._default_response(request)

            return FakeTransport(responder)

        first_fake = transport_for(first_scope)
        second_fake = transport_for(second_scope)
        first = await _start(first_fake, scope=first_scope)
        second = await _start(second_fake, scope=second_scope)
        self.assertNotEqual(first.dataset_name, second.dataset_name)
        self.assertNotEqual(first.node_set_id, second.node_set_id)
        await first.search_references("first", limit=1)
        await second.search_references("second", limit=1)
        self.assertEqual(
            first_fake.requests[-1]["query"], "first"
        )
        self.assertEqual(
            second_fake.requests[-1]["query"], "second"
        )
        await first.close()
        await second.close()

    async def test_process_launch_passes_the_verified_empty_cwd(self) -> None:
        process = SimpleNamespace(
            stderr=None,
            stdin=None,
            returncode=0,
            wait=AsyncMock(return_value=0),
        )
        with (
            patch.object(
                _SubprocessTransport,
                "_prepare_worker_cwd",
                return_value=WORKER_CWD,
            ) as prepare,
            patch.object(
                _SubprocessTransport,
                "_prepare_worker_tmpdir",
                return_value=WORKER_TMPDIR,
            ) as prepare_tmpdir,
            patch(
                "angler.memory.cognee_subprocess_bindings.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=process,
            ) as create_process,
        ):
            transport = await _SubprocessTransport.launch()

        prepare.assert_called_once_with()
        prepare_tmpdir.assert_called_once_with()
        self.assertEqual(cognee_worker_launch_cwd(), WORKER_CWD)
        self.assertEqual(create_process.await_args.kwargs["cwd"], WORKER_CWD)
        self.assertEqual(
            create_process.await_args.kwargs["env"],
            {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        )
        await transport.close()

    async def test_custom_process_launch_binds_argv_environment_and_paths(self) -> None:
        process = SimpleNamespace(
            stderr=None,
            stdin=None,
            returncode=0,
            wait=AsyncMock(return_value=0),
        )
        with tempfile.TemporaryDirectory() as temporary_root:
            with (
                patch(
                    "angler.memory.cognee_worker_protocol.COGNEE_SCOPE_ROOT",
                    temporary_root,
                ),
                patch(
                    "angler.memory.cognee_subprocess_bindings.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                    return_value=process,
                ) as create_process,
            ):
                scope = _custom_scope(
                    state_root=str(Path(temporary_root) / "custom")
                )
                transport = await _SubprocessTransport.launch(scope)
                validate_scope_marker(scope)

            self.assertEqual(
                create_process.await_args.args,
                cognee_worker_launch_argv(scope),
            )
            self.assertEqual(
                create_process.await_args.kwargs["cwd"], scope.worker_cwd
            )
            self.assertEqual(
                create_process.await_args.kwargs["env"],
                {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
            )
            await transport.close()

    async def test_hello_identity_mismatch_never_yields_usable_bindings(self) -> None:
        def responder(request):
            return success_response(
                request["id"], _hello_result(model_dimensions=MODEL_DIMENSIONS + 1)
            )

        fake = FakeTransport(responder)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "dimensions"):
            await _start(fake)
        self.assertTrue(fake.closed)
        self.assertEqual(len(fake.requests), 1)

        def cuda_responder(request):
            return success_response(
                request["id"],
                _hello_result(
                    embedding_execution_providers=["CUDAExecutionProvider"]
                ),
            )

        cuda = FakeTransport(cuda_responder)
        with self.assertRaisesRegex(
            CogneeWorkerProtocolError, "execution_providers"
        ):
            await _start(cuda)
        self.assertTrue(cuda.closed)

    async def test_hello_rejects_legacy_dataset_identity_and_disabled_access_control(
        self,
    ) -> None:
        legacy_id = str(uuid5(NAMESPACE_OID, f"{DATASET_NAME}{_USER_ID}"))

        async def start_without_expected(fake: FakeTransport):
            async def factory():
                return fake

            return await CogneeSubprocessBindings.start(
                _transport_factory=factory, timeout_seconds=1.0
            )

        def legacy_response(request):
            return success_response(
                request["id"], _hello_result(dataset_id=legacy_id)
            )

        legacy = FakeTransport(legacy_response)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "modern"):
            await start_without_expected(legacy)
        self.assertTrue(legacy.closed)

        def access_response(request):
            return success_response(
                request["id"], _hello_result(backend_access_control=False)
            )

        access = FakeTransport(access_response)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "access_control"):
            await start_without_expected(access)
        self.assertTrue(access.closed)

    async def test_shims_serialize_one_native_graph_point_without_content_paths(self) -> None:
        fake = FakeTransport()
        bindings = await _start(fake)
        backend_ref = await bindings.add_data_points([_wire_point(bindings)])
        self.assertEqual(backend_ref, _BACKEND_ID)
        request = fake.requests[-1]
        self.assertEqual(frozenset(request), frozenset(("v", "id", "op", "point")))
        self.assertEqual(request["op"], "project")
        point = request["point"]
        self.assertNotIn("metadata", point)
        self.assertEqual(point["id"], _BACKEND_ID)
        self.assertEqual(point["belongs_to_set"][0]["id"], NODE_SET_ID)
        self.assertEqual(
            point["references"],
            [
                {
                    "relationship_type": "RELATES_TO",
                    "properties": {"target_ref": _TARGET_REF},
                    "target": {"id": _TARGET_ID, "record_ref": _TARGET_REF},
                }
            ],
        )
        await bindings.close()

    async def test_existing_adapter_constructs_points_against_pure_python_shims(self) -> None:
        fake = FakeTransport()
        bindings = await _start(fake)
        projection = CognitiveGraphProjectionV2.from_acquisition(
            _chain()[1].acquisition
        )
        adapter = CogneeAcquisitionAdapter(
            bindings=bindings,
            tenant_id=bindings.tenant_id,
            dataset_id=bindings.dataset_id,
            dataset_name=bindings.dataset_name,
            node_set_name=bindings.node_set_name,
            local_embeddings_configured=True,
        )
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            backend_ref = await adapter.project(projection)
        self.assertEqual(backend_ref, fake.requests[-1]["point"]["id"])
        self.assertEqual(fake.requests[-1]["op"], "project")
        await bindings.close()

    async def test_search_returns_one_reference_only_graph_join_envelope(self) -> None:
        entry = {
            "id": _BACKEND_ID,
            "score": 0.125,
            "payload": {
                "record_ref": _RECORD_REF,
                "adjacent_record_refs": [_TARGET_REF],
                "belongs_to_set": [NODE_SET_NAME],
            },
        }

        def responder(request):
            if request["op"] == "search":
                envelope = _empty_search_result()
                envelope[0]["search_result"] = [entry]
                return success_response(request["id"], envelope)
            return FakeTransport._default_response(request)

        fake = FakeTransport(responder)
        bindings = await _start(fake)
        results = await bindings.search_references("synthetic query", limit=1)
        self.assertEqual(results[0]["search_result"], [entry])
        self.assertEqual(
            frozenset(entry["payload"]),
            frozenset(("record_ref", "adjacent_record_refs", "belongs_to_set")),
        )
        self.assertEqual(fake.requests[-1]["limit"], 1)
        await bindings.close()

    async def test_search_rejects_content_or_scope_injection_and_fails_closed(self) -> None:
        def responder(request):
            if request["op"] == "search":
                envelope = _empty_search_result()
                envelope[0]["search_result"] = [
                    {
                        "id": _BACKEND_ID,
                        "score": 0.2,
                        "payload": {
                            "record_ref": _RECORD_REF,
                            "adjacent_record_refs": [],
                            "belongs_to_set": [NODE_SET_NAME],
                            "content": "must never cross this boundary",
                        },
                    }
                ]
                return success_response(request["id"], envelope)
            return FakeTransport._default_response(request)

        fake = FakeTransport(responder)
        bindings = await _start(fake)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "fields"):
            await bindings.search_references("query", limit=1)
        self.assertTrue(fake.closed)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "unusable"):
            await bindings.search_references("query", limit=1)

    async def test_search_requires_unique_lower_is_better_distance_order(self) -> None:
        second_id = "6969d152-3591-5be3-a1e4-9b03f7b9a346"

        def responder(request):
            if request["op"] == "search":
                envelope = _empty_search_result()
                envelope[0]["search_result"] = [
                    {
                        "id": _BACKEND_ID,
                        "score": 0.3,
                        "payload": {
                            "record_ref": _RECORD_REF,
                            "adjacent_record_refs": [],
                            "belongs_to_set": [NODE_SET_NAME],
                        },
                    },
                    {
                        "id": second_id,
                        "score": 0.2,
                        "payload": {
                            "record_ref": _TARGET_REF,
                            "adjacent_record_refs": [],
                            "belongs_to_set": [NODE_SET_NAME],
                        },
                    },
                ]
                return success_response(request["id"], envelope)
            return FakeTransport._default_response(request)

        fake = FakeTransport(responder)
        bindings = await _start(fake)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "lower-is-better"):
            await bindings.search_references("query", limit=2)
        self.assertTrue(fake.closed)

    async def test_requests_are_strictly_ordered_under_concurrency(self) -> None:
        active = 0
        maximum_active = 0

        async def responder(request):
            nonlocal active, maximum_active
            if request["op"] != "search":
                return FakeTransport._default_response(request)
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return success_response(request["id"], _empty_search_result())

        fake = FakeTransport(responder)
        bindings = await _start(fake)
        await asyncio.gather(
            bindings.search_references("first", limit=1),
            bindings.search_references("second", limit=1),
        )
        self.assertEqual(maximum_active, 1)
        self.assertEqual(
            [request["id"] for request in fake.requests], [1, 2, 3]
        )
        await bindings.close()

    async def test_response_id_mismatch_closes_binding(self) -> None:
        def responder(request):
            if request["op"] == "search":
                return success_response(request["id"] + 1, _empty_search_result())
            return FakeTransport._default_response(request)

        fake = FakeTransport(responder)
        bindings = await _start(fake)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "id differs"):
            await bindings.search_references("query", limit=1)
        self.assertTrue(fake.closed)

    async def test_timeout_cancels_operation_and_closes_binding(self) -> None:
        async def responder(request):
            if request["op"] == "search":
                await asyncio.sleep(10)
            return FakeTransport._default_response(request)

        fake = FakeTransport(responder)
        bindings = await _start(fake, timeout_seconds=0.01)
        with self.assertRaises(asyncio.TimeoutError):
            await bindings.search_references("query", limit=1)
        self.assertTrue(fake.closed)

    async def test_worker_crash_or_eof_invalidates_binding_and_future_use(self) -> None:
        for failure in (
            ConnectionError("worker pipe closed after crash"),
            CogneeWorkerProtocolError("worker stream ended before a response frame"),
        ):
            with self.subTest(failure=type(failure).__name__):
                def responder(request, *, failure=failure):
                    if request["op"] == "search":
                        raise failure
                    return FakeTransport._default_response(request)

                fake = FakeTransport(responder)
                bindings = await _start(fake)
                with self.assertRaises(type(failure)):
                    await bindings.search_references("query", limit=1)
                self.assertTrue(fake.closed)
                with self.assertRaisesRegex(CogneeWorkerProtocolError, "unusable"):
                    await bindings.search_references("query", limit=1)

    async def test_remote_errors_are_bounded_stable_and_fail_closed(self) -> None:
        def responder(request):
            if request["op"] == "search":
                return error_response(
                    request["id"], "SEARCH_FAILED", "bounded stable failure"
                )
            return FakeTransport._default_response(request)

        fake = FakeTransport(responder)
        bindings = await _start(fake)
        with self.assertRaises(CogneeWorkerRemoteError) as caught:
            await bindings.search_references("query", limit=1)
        self.assertEqual(caught.exception.code, "SEARCH_FAILED")
        self.assertEqual(caught.exception.remote_message, "bounded stable failure")
        self.assertTrue(fake.closed)

    async def test_forget_has_no_caller_selected_scope_and_requires_same_id(self) -> None:
        fake = FakeTransport()
        bindings = await _start(fake)
        await bindings.forget_namespace()
        self.assertEqual(
            fake.requests[-1],
            {
                "v": PROTOCOL_VERSION,
                "id": 2,
                "op": "forget_namespace",
            },
        )
        await bindings.close()

        def changed_id(request):
            if request["op"] == "forget_namespace":
                result = FakeTransport._default_response(request)["result"]
                result["dataset_id"] = "9305654e-8d0d-5ec0-8bf7-cc8abfb19e99"
                return success_response(request["id"], result)
            return FakeTransport._default_response(request)

        fake_changed = FakeTransport(changed_id)
        changed = await _start(fake_changed)
        with self.assertRaisesRegex(CogneeWorkerProtocolError, "frozen namespace"):
            await changed.forget_namespace()
        self.assertTrue(fake_changed.closed)


if __name__ == "__main__":
    unittest.main()
