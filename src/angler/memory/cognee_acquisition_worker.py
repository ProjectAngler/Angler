"""Networkless Cognee 1.5.3 worker for acquisition reference projection.

Nothing from Cognee is imported until the clean environment and the frozen
local embedding/tokenizer artifacts have been verified.  The worker exposes
only graph projection, reference search, exact namespace recreation, and close.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256
from importlib.metadata import version as distribution_version
from importlib import import_module
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any
from uuid import UUID

from .cognee_worker_protocol import (
    COGNEE_VERSION,
    COGNEE_PYTHON_VERSION,
    COLLECTION_NAME,
    DEFAULT_COGNEE_WORKER_SCOPE,
    FASTEMBED_EXECUTION_PROVIDERS,
    FASTEMBED_SESSION_THREADS,
    INHERITED_NETWORK_NAMESPACE_ENV,
    MAX_MESSAGE_BYTES,
    MAX_POINT_NEIGHBORS,
    MODEL_CACHE_ROOT,
    MODEL_DIMENSIONS,
    MODEL_MANIFEST,
    MODEL_NAME,
    PROTOCOL_VERSION,
    SCORE_SEMANTICS,
    SOFTWARE_VERSIONS,
    TIKTOKEN_CACHE_ROOT,
    CogneeWorkerScope,
    CogneeWorkerProtocolError,
    bounded_text,
    canonical_uuid,
    decode_json_line,
    digest,
    encode_json_line,
    error_response,
    exact_worker_scope,
    finite_score,
    modern_dataset_id,
    success_response,
    validate_exact_worker_environment,
    validate_inherited_worker_boundary,
    validate_point_payload,
    validate_request,
    validate_scope_marker,
    validate_worker_cwd,
    validate_worker_tmpdir,
    worker_scope_from_launch_arguments,
)


_SNAPSHOT_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
_FASTEMBED_REPOSITORY = "models--qdrant--bge-small-en-v1.5-onnx-q"
_SNAPSHOT_FILES = (
    "config.json",
    "model_optimized.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SNAPSHOT_BYTES = 67_179_163
_ONNX_BYTES = 66_465_124
_ONNX_SHA256 = "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
_TIKTOKEN_FILE = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
_TIKTOKEN_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


def _validate_persisted_dataset_storage(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    *,
    expected_user_id: str | None = None,
    expected_dataset_id: str | None = None,
) -> None:
    """Reject persisted Cognee routes that escape the worker state root.

    Cognee stores an absolute vector URL in its metadata database. A copied
    state root would otherwise keep writing vectors into its source database
    while creating graph nodes in the clone. Every persisted route must be
    scope-local before Cognee is allowed to obtain an engine.
    """

    scope = exact_worker_scope(scope)
    database_root = Path(scope.state_root) / "system" / "databases"
    metadata = database_root / os.environ.get("DB_NAME", "angler_cognee.sqlite")
    if not metadata.exists():
        return
    if metadata.is_symlink() or not metadata.is_file():
        raise CogneeWorkerProtocolError("Cognee metadata store boundary differs")
    try:
        connection = sqlite3.connect(
            metadata.as_uri() + "?mode=ro",
            uri=True,
            timeout=5.0,
        )
        try:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='dataset_database'"
            ).fetchone()
            if exists is None:
                return
            rows = connection.execute(
                "SELECT owner_id,dataset_id,vector_database_name,"
                "graph_database_name,vector_database_provider,"
                "graph_database_provider,graph_dataset_database_handler,"
                "vector_dataset_database_handler,vector_database_url,"
                "graph_database_url,graph_database_key,vector_database_key,"
                "graph_database_connection_info,vector_database_connection_info "
                "FROM dataset_database"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise CogneeWorkerProtocolError(
            "Cognee metadata store cannot be inspected"
        ) from exc

    def persisted_uuid(value: object, label: str) -> str:
        if type(value) is not str:
            raise CogneeWorkerProtocolError(f"{label} differs")
        try:
            return str(UUID(value))
        except (ValueError, AttributeError) as exc:
            raise CogneeWorkerProtocolError(f"{label} differs") from exc

    normalized_expected_user = (
        None
        if expected_user_id is None
        else canonical_uuid(expected_user_id, "expected Cognee storage user id")
    )
    normalized_expected_dataset = (
        None
        if expected_dataset_id is None
        else canonical_uuid(
            expected_dataset_id, "expected Cognee storage dataset id"
        )
    )
    if normalized_expected_dataset is not None and len(rows) != 1:
        raise CogneeWorkerProtocolError("Cognee dataset storage binding differs")

    def validate_storage_tree(path: Path, *, directory: bool, label: str) -> None:
        try:
            root_stat = path.lstat()
        except OSError as exc:
            raise CogneeWorkerProtocolError(f"{label} is unavailable") from exc
        if path.is_symlink() or (directory and not path.is_dir()) or (
            not directory and not path.is_file()
        ):
            raise CogneeWorkerProtocolError(f"{label} differs")
        if not directory and root_stat.st_nlink != 1:
            raise CogneeWorkerProtocolError(f"{label} is aliased")
        if not directory:
            return
        for current_root, directories, filenames in os.walk(
            path, topdown=True, followlinks=False
        ):
            current = Path(current_root)
            for name in (*directories, *filenames):
                item = current / name
                try:
                    item_stat = item.lstat()
                except OSError as exc:
                    raise CogneeWorkerProtocolError(
                        f"{label} cannot be inspected"
                    ) from exc
                if item.is_symlink():
                    raise CogneeWorkerProtocolError(f"{label} contains a symlink")
                if item.is_file() and item_stat.st_nlink != 1:
                    raise CogneeWorkerProtocolError(f"{label} contains an alias")

    try:
        resolved_root = database_root.resolve(strict=True)
    except OSError as exc:
        raise CogneeWorkerProtocolError(
            "Cognee dataset storage root is unavailable"
        ) from exc
    for row in rows:
        # SQLAlchemy's UUID type persists compact hexadecimal text in SQLite;
        # normalize that representation without weakening the public protocol's
        # canonical UUID requirement.
        owner_id = persisted_uuid(row[0], "persisted storage owner id")
        dataset_id = persisted_uuid(row[1], "persisted storage dataset id")
        if (
            normalized_expected_user is not None
            and owner_id != normalized_expected_user
        ) or (
            normalized_expected_dataset is not None
            and dataset_id != normalized_expected_dataset
        ):
            raise CogneeWorkerProtocolError("Cognee dataset storage identity differs")

        vector_name = f"{dataset_id}.lance.db"
        graph_name = f"{dataset_id}.lbug"
        if not (
            row[2] == vector_name
            and row[3] == graph_name
            and row[4] == "lancedb"
            and row[5] == "ladybug"
            and row[6] == "ladybug"
            and row[7] == "lancedb"
            and row[10] in (None, "")
            and row[11] in (None, "")
            and row[12] == "{}"
            and row[13] == "{}"
        ):
            raise CogneeWorkerProtocolError("Cognee dataset storage metadata differs")

        owner_root = resolved_root / owner_id
        try:
            resolved_owner_root = owner_root.resolve(strict=True)
        except OSError as exc:
            raise CogneeWorkerProtocolError(
                "Cognee dataset storage owner root is unavailable"
            ) from exc
        if (
            owner_root.is_symlink()
            or not owner_root.is_dir()
            or resolved_owner_root.parent != resolved_root
        ):
            raise CogneeWorkerProtocolError(
                "Cognee dataset storage owner root escapes the frozen scope"
            )
        expected_vector = resolved_owner_root / vector_name
        vector_url = row[8]
        if (
            type(vector_url) is not str
            or not vector_url
            or not Path(vector_url).is_absolute()
            or Path(vector_url) != expected_vector
        ):
            raise CogneeWorkerProtocolError(
                "Cognee vector storage escapes the frozen scope"
            )
        validate_storage_tree(
            expected_vector,
            directory=True,
            label="Cognee vector storage",
        )
        graph_url = row[9]
        expected_graph = resolved_owner_root / graph_name
        if graph_url not in (None, "") and (
            type(graph_url) is not str
            or not Path(graph_url).is_absolute()
            or Path(graph_url) != expected_graph
        ):
            raise CogneeWorkerProtocolError(
                "Cognee graph storage escapes the frozen scope"
            )
        validate_storage_tree(
            expected_graph,
            directory=False,
            label="Cognee graph storage",
        )


def _validate_runtime_boundary(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> None:
    inherited_network_namespace = os.environ.get(
        INHERITED_NETWORK_NAMESPACE_ENV
    )
    validate_exact_worker_environment(
        scope=scope,
        inherited_network_namespace=inherited_network_namespace,
    )
    if inherited_network_namespace is not None:
        validate_inherited_worker_boundary(inherited_network_namespace)
    validate_scope_marker(scope)
    validate_worker_cwd(scope)
    validate_worker_tmpdir(scope)


def _file_sha256(path: Path) -> tuple[str, int]:
    hasher = sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1_048_576)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
    return hasher.hexdigest(), size


def _verify_local_artifacts() -> None:
    repository = Path(MODEL_CACHE_ROOT) / _FASTEMBED_REPOSITORY
    for directory in ("blobs", "refs", "snapshots", "trees"):
        if not (repository / directory).is_dir():
            raise CogneeWorkerProtocolError("frozen embedding cache tree is incomplete")
    if (repository / "refs" / "main").read_text(encoding="ascii") != _SNAPSHOT_REVISION:
        raise CogneeWorkerProtocolError("frozen embedding revision differs")
    snapshot = repository / "snapshots" / _SNAPSHOT_REVISION
    if not snapshot.is_dir():
        raise CogneeWorkerProtocolError("frozen embedding snapshot is absent")
    if sorted(path.name for path in snapshot.iterdir()) != list(_SNAPSHOT_FILES):
        raise CogneeWorkerProtocolError("frozen embedding snapshot entries differ")
    manifest = sha256()
    total = 0
    observed: dict[str, tuple[str, int]] = {}
    for filename in _SNAPSHOT_FILES:
        path = snapshot / filename
        if not path.is_symlink() or not path.is_file():
            raise CogneeWorkerProtocolError("frozen embedding snapshot is incomplete")
        try:
            path.resolve(strict=True).relative_to((repository / "blobs").resolve())
        except ValueError as exc:
            raise CogneeWorkerProtocolError(
                "frozen embedding snapshot link escapes its blob root"
            ) from exc
        file_digest, byte_count = _file_sha256(path)
        observed[filename] = (file_digest, byte_count)
        total += byte_count
        manifest.update(f"{file_digest}  {filename}\n".encode("ascii"))
    if total != _SNAPSHOT_BYTES:
        raise CogneeWorkerProtocolError("frozen embedding snapshot size differs")
    if f"sha256:{manifest.hexdigest()}" != MODEL_MANIFEST:
        raise CogneeWorkerProtocolError("frozen embedding snapshot manifest differs")
    if observed["model_optimized.onnx"] != (_ONNX_SHA256, _ONNX_BYTES):
        raise CogneeWorkerProtocolError("frozen embedding model identity differs")

    tokenizer = Path(TIKTOKEN_CACHE_ROOT) / _TIKTOKEN_FILE
    if not tokenizer.is_file() or _file_sha256(tokenizer)[0] != _TIKTOKEN_SHA256:
        raise CogneeWorkerProtocolError("frozen tokenizer cache identity differs")


def _install_bounded_fastembed_constructor(
    *, engine_module: Any, native_text_embedding: type
) -> Any:
    """Bind Cognee's constructor to one offline CPU-only FastEmbed session."""

    if getattr(engine_module, "TextEmbedding", None) is not native_text_embedding:
        raise CogneeWorkerProtocolError(
            "Cognee FastEmbed constructor differs before qualification binding"
        )

    def bounded_text_embedding(*args: Any, **kwargs: Any) -> Any:
        if args or frozenset(kwargs) != frozenset(("model_name",)):
            raise CogneeWorkerProtocolError(
                "Cognee FastEmbed constructor arguments differ"
            )
        if kwargs["model_name"] != MODEL_NAME:
            raise CogneeWorkerProtocolError("Cognee FastEmbed model differs")
        return native_text_embedding(
            model_name=MODEL_NAME,
            cache_dir=MODEL_CACHE_ROOT,
            threads=FASTEMBED_SESSION_THREADS,
            providers=FASTEMBED_EXECUTION_PROVIDERS,
            cuda=False,
            device_ids=None,
            lazy_load=False,
            local_files_only=True,
        )

    engine_module.TextEmbedding = bounded_text_embedding
    if engine_module.TextEmbedding is not bounded_text_embedding:
        raise CogneeWorkerProtocolError("Cognee FastEmbed constructor binding failed")
    return bounded_text_embedding


def _fastembed_runtime_evidence(vector_engine: Any) -> dict[str, Any]:
    """Verify the actual FastEmbed/ONNX session and return bounded evidence."""

    embedding_engine = getattr(vector_engine, "embedding_engine", None)
    if (
        type(embedding_engine).__module__
        != "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine"
        or type(embedding_engine).__name__ != "FastembedEmbeddingEngine"
        or getattr(embedding_engine, "model", None) != MODEL_NAME
        or getattr(embedding_engine, "dimensions", None) != MODEL_DIMENSIONS
    ):
        raise CogneeWorkerProtocolError("Cognee FastEmbed engine identity differs")

    facade = getattr(embedding_engine, "embedding_model", None)
    if (
        type(facade).__module__ != "fastembed.text.text_embedding"
        or type(facade).__name__ != "TextEmbedding"
        or getattr(facade, "model_name", None) != MODEL_NAME
        or getattr(facade, "cache_dir", None) != MODEL_CACHE_ROOT
        or getattr(facade, "threads", None) != FASTEMBED_SESSION_THREADS
        or getattr(facade, "_local_files_only", None) is not True
    ):
        raise CogneeWorkerProtocolError("FastEmbed facade boundary differs")

    onnx_model = getattr(facade, "model", None)
    if (
        type(onnx_model).__module__ != "fastembed.text.onnx_embedding"
        or type(onnx_model).__name__ != "OnnxTextEmbedding"
        or getattr(onnx_model, "model_name", None) != MODEL_NAME
        or getattr(onnx_model, "cache_dir", None) != MODEL_CACHE_ROOT
        or getattr(onnx_model, "threads", None) != FASTEMBED_SESSION_THREADS
        or tuple(getattr(onnx_model, "providers", ()))
        != FASTEMBED_EXECUTION_PROVIDERS
        or getattr(onnx_model, "cuda", None) is not False
        or getattr(onnx_model, "device_ids", None) is not None
        or getattr(onnx_model, "device_id", None) is not None
        or getattr(onnx_model, "lazy_load", None) is not False
        or getattr(onnx_model, "_local_files_only", None) is not True
    ):
        raise CogneeWorkerProtocolError("FastEmbed ONNX model boundary differs")

    session = getattr(onnx_model, "model", None)
    get_providers = getattr(session, "get_providers", None)
    get_session_options = getattr(session, "get_session_options", None)
    if not callable(get_providers) or not callable(get_session_options):
        raise CogneeWorkerProtocolError("FastEmbed ONNX session is absent")
    providers = get_providers()
    options = get_session_options()
    intra_op_threads = getattr(options, "intra_op_num_threads", None)
    inter_op_threads = getattr(options, "inter_op_num_threads", None)
    if (
        type(providers) is not list
        or tuple(providers) != FASTEMBED_EXECUTION_PROVIDERS
        or type(intra_op_threads) is not int
        or intra_op_threads != FASTEMBED_SESSION_THREADS
        or type(inter_op_threads) is not int
        or inter_op_threads != FASTEMBED_SESSION_THREADS
    ):
        raise CogneeWorkerProtocolError(
            "FastEmbed ONNX provider or thread boundary differs"
        )
    return {
        "embedding_execution_providers": list(providers),
        "embedding_intra_op_threads": intra_op_threads,
        "embedding_inter_op_threads": inter_op_threads,
    }


def _native_types(DataPoint: type, Edge: type, NodeSet: type) -> tuple[type, type]:
    target = type(
        "AnglerAcquisitionReferenceTarget",
        (DataPoint,),
        {
            "__module__": __name__,
            "__annotations__": {"record_ref": str, "metadata": dict},
            "metadata": {"index_fields": [], "identity_fields": ["record_ref"]},
        },
    )
    point = type(
        "AnglerAcquisitionReferencePoint",
        (DataPoint,),
        {
            "__module__": __name__,
            "__annotations__": {
                "acquisition_ref": str,
                "source_contract": str,
                "source_ref": str,
                "record_ref": str,
                "projection_ref": str,
                "acquired_ordinal": int,
                "search_text": str,
                "adjacent_record_refs": list[str],
                "references": list[Any],
                "visibility": str,
                "memory_kind": str,
                "epistemic_status": str,
                "metadata": dict,
            },
            "metadata": {
                "index_fields": ["search_text"],
                "identity_fields": ["record_ref"],
            },
        },
    )
    # Keep these parameters explicit: they are the only native construction
    # types admitted by this worker.
    if Edge.__name__ != "Edge" or NodeSet.__name__ != "NodeSet":
        raise CogneeWorkerProtocolError("Cognee native construction types differ")
    return point, target


def _exact_tenant_memberships(value: object) -> list[tuple[str, str]]:
    if type(value) is not list:
        raise CogneeWorkerProtocolError(
            "persisted Cognee tenant memberships are malformed"
        )
    memberships: list[tuple[str, str]] = []
    for entry in value:
        if type(entry) is not dict or frozenset(entry) != frozenset(("id", "name")):
            raise CogneeWorkerProtocolError(
                "persisted Cognee tenant memberships are malformed"
            )
        tenant_id = canonical_uuid(entry["id"], "persisted Cognee tenant membership id")
        tenant_name = bounded_text(
            entry["name"], "persisted Cognee tenant membership name", 256
        )
        memberships.append((tenant_id, tenant_name))
    if memberships != sorted(set(memberships)):
        raise CogneeWorkerProtocolError(
            "persisted Cognee tenant memberships are not unique and canonical"
        )
    return memberships


async def _verified_qualification_tenant(
    *,
    tenant_id: str,
    user_id: str,
    get_tenant: Any,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> Any:
    scope = exact_worker_scope(scope)
    tenant = await get_tenant(UUID(tenant_id))
    if (
        canonical_uuid(str(tenant.id), "persisted Cognee tenant row id") != tenant_id
        or tenant.name != scope.tenant_name
        or str(tenant.owner_id) != user_id
    ):
        raise CogneeWorkerProtocolError(
            "persisted Cognee tenant row differs from the qualification scope"
        )
    return tenant


async def _ensure_qualification_tenant(
    *,
    user: Any,
    get_default_user: Any,
    get_user_tenants: Any,
    get_tenant: Any,
    create_tenant: Any,
    select_tenant: Any,
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> tuple[Any, Any]:
    """Bind the default user to the one qualification-owned tenant.

    A previous interrupted bootstrap may have committed the membership before
    the active-tenant selection became durable.  That sole exact membership is
    recoverable; every ambiguous or foreign state remains a hard failure.
    """

    scope = exact_worker_scope(scope)
    user_id = canonical_uuid(str(user.id), "persisted Cognee user id")
    active_id = (
        None
        if user.tenant_id is None
        else canonical_uuid(str(user.tenant_id), "persisted Cognee tenant id")
    )
    memberships = _exact_tenant_memberships(await get_user_tenants(user))

    if not memberships:
        if active_id is not None:
            raise CogneeWorkerProtocolError(
                "active Cognee tenant has no exact persisted membership"
            )
        created_id = canonical_uuid(
            str(
                await create_tenant(
                    scope.tenant_name,
                    UUID(user_id),
                    set_as_active_tenant=True,
                )
            ),
            "created Cognee tenant id",
        )
        await _verified_qualification_tenant(
            tenant_id=created_id,
            user_id=user_id,
            get_tenant=get_tenant,
            scope=scope,
        )
        expected_tenant_id = created_id
    elif len(memberships) == 1 and memberships[0][1] == scope.tenant_name:
        expected_tenant_id = memberships[0][0]
        tenant = await _verified_qualification_tenant(
            tenant_id=expected_tenant_id,
            user_id=user_id,
            get_tenant=get_tenant,
            scope=scope,
        )
        if active_id is None:
            selected = await select_tenant(UUID(user_id), UUID(expected_tenant_id))
            if (
                canonical_uuid(str(selected.id), "selected Cognee user id") != user_id
                or canonical_uuid(
                    str(selected.tenant_id), "selected Cognee active tenant id"
                )
                != expected_tenant_id
            ):
                raise CogneeWorkerProtocolError(
                    "Cognee tenant selection returned a different persisted scope"
                )
        elif active_id != expected_tenant_id:
            raise CogneeWorkerProtocolError(
                "active Cognee tenant differs from the exact membership"
            )
    else:
        raise CogneeWorkerProtocolError(
            "persisted Cognee tenant memberships differ from the qualification scope"
        )

    reloaded = await get_default_user()
    if canonical_uuid(str(reloaded.id), "reloaded Cognee user id") != user_id:
        raise CogneeWorkerProtocolError("reloaded Cognee user identity differs")
    if reloaded.tenant_id is None or canonical_uuid(
        str(reloaded.tenant_id), "reloaded Cognee tenant id"
    ) != expected_tenant_id:
        raise CogneeWorkerProtocolError(
            "qualification tenant is not the active persisted Cognee tenant"
        )
    if _exact_tenant_memberships(await get_user_tenants(reloaded)) != [
        (expected_tenant_id, scope.tenant_name)
    ]:
        raise CogneeWorkerProtocolError(
            "reloaded Cognee tenant memberships differ from the qualification scope"
        )
    tenant = await _verified_qualification_tenant(
        tenant_id=expected_tenant_id,
        user_id=user_id,
        get_tenant=get_tenant,
        scope=scope,
    )
    return reloaded, tenant


class _CogneeBackend:
    def __init__(
        self,
        *,
        user: Any,
        dataset: Any,
        embedding_config: Any,
        Point: type,
        Target: type,
        Edge: type,
        NodeSet: type,
        PipelineContext: type,
        set_database_context: Any,
        create_authorized_dataset: Any,
        get_authorized_dataset_by_name: Any,
        add_data_points: Any,
        index_data_points: Any,
        get_unified_engine: Any,
        backend_access_control_enabled: Any,
        validate_runtime_boundary: Any,
        fastembed_engine_module: Any,
        bounded_fastembed_constructor: Any,
        inspect_fastembed_runtime: Any,
        embedding_runtime_evidence: dict[str, Any],
        forget: Any,
        scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    ) -> None:
        self._scope = exact_worker_scope(scope)
        self.user = user
        self.dataset = dataset
        self.embedding_config = embedding_config
        self.Point = Point
        self.Target = Target
        self.Edge = Edge
        self.NodeSet = NodeSet
        self.PipelineContext = PipelineContext
        self.set_database_context = set_database_context
        self.create_authorized_dataset = create_authorized_dataset
        self.get_authorized_dataset_by_name = get_authorized_dataset_by_name
        self.add_data_points_native = add_data_points
        self.index_data_points_native = index_data_points
        self.get_unified_engine = get_unified_engine
        self.backend_access_control_enabled = backend_access_control_enabled
        self.validate_runtime_boundary = validate_runtime_boundary
        self.fastembed_engine_module = fastembed_engine_module
        self.bounded_fastembed_constructor = bounded_fastembed_constructor
        self.inspect_fastembed_runtime = inspect_fastembed_runtime
        self.forget_native = forget
        self.tenant_id = str(user.tenant_id)
        self.user_id = str(user.id)
        self.dataset_id = str(dataset.id)
        expected_runtime_evidence = {
            "embedding_execution_providers": list(FASTEMBED_EXECUTION_PROVIDERS),
            "embedding_intra_op_threads": FASTEMBED_SESSION_THREADS,
            "embedding_inter_op_threads": FASTEMBED_SESSION_THREADS,
        }
        if embedding_runtime_evidence != expected_runtime_evidence:
            raise CogneeWorkerProtocolError("FastEmbed runtime evidence differs")
        self.embedding_execution_providers = FASTEMBED_EXECUTION_PROVIDERS
        self.embedding_intra_op_threads = FASTEMBED_SESSION_THREADS
        self.embedding_inter_op_threads = FASTEMBED_SESSION_THREADS

    @property
    def scope(self) -> CogneeWorkerScope:
        return self._scope

    @classmethod
    async def load(
        cls,
        *,
        expected_dataset_id: str | None,
        scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
    ) -> "_CogneeBackend":
        scope = exact_worker_scope(scope)
        _validate_runtime_boundary(scope)
        _verify_local_artifacts()
        # Run before Cognee obtains an engine: a copied metadata DB must never
        # redirect a clone's vector writes into its source state root.
        _validate_persisted_dataset_storage(scope)
        python_version = ".".join(str(part) for part in sys.version_info[:3])
        if python_version != COGNEE_PYTHON_VERSION:
            raise CogneeWorkerProtocolError("worker Python version differs")
        for distribution, expected_version in SOFTWARE_VERSIONS.items():
            if distribution_version(distribution) != expected_version:
                raise CogneeWorkerProtocolError(
                    "installed Cognee software identity differs"
                )

        # Imports remain below all environment/artifact checks by design.
        from cognee.api.v1.forget.forget import forget
        from cognee.context_global_variables import (
            backend_access_control_enabled,
            set_database_global_context_variables,
        )
        from cognee.infrastructure.databases.unified import get_unified_engine
        from cognee.infrastructure.databases.vector.embeddings.config import (
            EmbeddingConfig,
        )
        from cognee.infrastructure.engine import DataPoint, Edge
        from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
        from cognee.low_level import setup
        from cognee.modules.data.methods.create_authorized_dataset import (
            create_authorized_dataset,
        )
        from cognee.modules.data.methods.get_authorized_dataset_by_name import (
            get_authorized_dataset_by_name,
        )
        from cognee.modules.engine.models.node_set import NodeSet
        from cognee.modules.pipelines.models.PipelineContext import PipelineContext
        from cognee.modules.users.methods.get_default_user import get_default_user
        from cognee.modules.users.permissions.methods.get_tenant import get_tenant
        from cognee.modules.users.tenants.methods.create_tenant import create_tenant
        from cognee.modules.users.tenants.methods.get_user_tenants import (
            get_user_tenants,
        )
        from cognee.modules.users.tenants.methods.select_tenant import select_tenant
        from cognee.tasks.storage.add_data_points import add_data_points
        from cognee.tasks.storage.index_data_points import index_data_points
        from fastembed import TextEmbedding as NativeTextEmbedding

        fastembed_engine_module = import_module(
            "cognee.infrastructure.databases.vector.embeddings."
            "FastembedEmbeddingEngine"
        )
        embedding_factory_module = import_module(
            "cognee.infrastructure.databases.vector.embeddings."
            "get_embedding_engine"
        )
        create_embedding_engine = getattr(
            embedding_factory_module, "create_embedding_engine", None
        )
        if (
            not callable(create_embedding_engine)
            or create_embedding_engine.cache_info().currsize != 0
        ):
            raise CogneeWorkerProtocolError(
                "Cognee embedding engine existed before qualification binding"
            )
        bounded_fastembed_constructor = _install_bounded_fastembed_constructor(
            engine_module=fastembed_engine_module,
            native_text_embedding=NativeTextEmbedding,
        )

        _validate_runtime_boundary(scope)
        await setup()
        _validate_runtime_boundary(scope)
        if backend_access_control_enabled() is not True:
            raise CogneeWorkerProtocolError("Cognee backend access control is disabled")
        user = await get_default_user()
        user, _tenant = await _ensure_qualification_tenant(
            user=user,
            get_default_user=get_default_user,
            get_user_tenants=get_user_tenants,
            get_tenant=get_tenant,
            create_tenant=create_tenant,
            select_tenant=select_tenant,
            scope=scope,
        )
        _validate_runtime_boundary(scope)
        if backend_access_control_enabled() is not True:
            raise CogneeWorkerProtocolError("Cognee backend access control drifted")
        user_id = canonical_uuid(str(user.id), "persisted Cognee user id")
        tenant_id = canonical_uuid(str(user.tenant_id), "persisted Cognee tenant id")
        modern_id = modern_dataset_id(
            user_id,
            tenant_id,
            dataset_name=scope.dataset_name,
        )
        if expected_dataset_id is not None and expected_dataset_id != modern_id:
            raise CogneeWorkerProtocolError(
                "expected dataset id is not the modern user/tenant identity"
            )
        dataset = await get_authorized_dataset_by_name(
            scope.dataset_name, user, "write"
        )
        if dataset is None:
            dataset = await create_authorized_dataset(scope.dataset_name, user)
        delete_authorized = await get_authorized_dataset_by_name(
            scope.dataset_name, user, "delete"
        )
        if delete_authorized is None or str(delete_authorized.id) != str(dataset.id):
            raise CogneeWorkerProtocolError("dataset delete authority is absent")
        dataset_id = canonical_uuid(str(dataset.id), "Cognee dataset id")
        if (
            str(dataset.owner_id) != user_id
            or str(dataset.tenant_id) != tenant_id
            or dataset_id != modern_id
        ):
            raise CogneeWorkerProtocolError(
                "Cognee dataset is not the persisted modern user/tenant identity"
            )
        if expected_dataset_id is not None and dataset_id != expected_dataset_id:
            raise CogneeWorkerProtocolError("Cognee dataset id differs from the binding")

        embedding_config = EmbeddingConfig(
            embedding_provider="fastembed",
            embedding_model=MODEL_NAME,
            embedding_dimensions=MODEL_DIMENSIONS,
            embedding_max_completion_tokens=512,
            embedding_batch_size=8,
            embedding_max_concurrent_data_points=8,
        )
        if (
            embedding_config.embedding_provider != "fastembed"
            or embedding_config.embedding_model != MODEL_NAME
            or embedding_config.embedding_dimensions != MODEL_DIMENSIONS
        ):
            raise CogneeWorkerProtocolError("Cognee embedding configuration differs")
        Point, Target = _native_types(DataPoint, Edge, NodeSet)
        if (
            str(generate_node_id("NodeSet:" + scope.node_set_name))
            != scope.node_set_id
        ):
            raise CogneeWorkerProtocolError("Cognee NodeSet identity algorithm differs")
        async with set_database_global_context_variables(
            dataset.id,
            user.id,
            embedding_config=embedding_config,
        ):
            unified = await get_unified_engine()
        _validate_persisted_dataset_storage(
            scope,
            expected_user_id=user_id,
            expected_dataset_id=dataset_id,
        )
        if fastembed_engine_module.TextEmbedding is not bounded_fastembed_constructor:
            raise CogneeWorkerProtocolError("Cognee FastEmbed constructor binding drifted")
        embedding_runtime_evidence = _fastembed_runtime_evidence(unified.vector)
        _validate_runtime_boundary(scope)
        return cls(
            user=user,
            dataset=dataset,
            embedding_config=embedding_config,
            Point=Point,
            Target=Target,
            Edge=Edge,
            NodeSet=NodeSet,
            PipelineContext=PipelineContext,
            set_database_context=set_database_global_context_variables,
            create_authorized_dataset=create_authorized_dataset,
            get_authorized_dataset_by_name=get_authorized_dataset_by_name,
            add_data_points=add_data_points,
            index_data_points=index_data_points,
            get_unified_engine=get_unified_engine,
            backend_access_control_enabled=backend_access_control_enabled,
            validate_runtime_boundary=lambda: _validate_runtime_boundary(scope),
            fastembed_engine_module=fastembed_engine_module,
            bounded_fastembed_constructor=bounded_fastembed_constructor,
            inspect_fastembed_runtime=_fastembed_runtime_evidence,
            embedding_runtime_evidence=embedding_runtime_evidence,
            forget=forget,
            scope=scope,
        )

    def hello_result(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "cognee_version": COGNEE_VERSION,
            "python_version": COGNEE_PYTHON_VERSION,
            "software_versions": dict(SOFTWARE_VERSIONS),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.scope.dataset_name,
            "node_set_name": self.scope.node_set_name,
            "node_set_id": self.scope.node_set_id,
            "collection_name": COLLECTION_NAME,
            "model_name": MODEL_NAME,
            "model_dimensions": MODEL_DIMENSIONS,
            "model_manifest": MODEL_MANIFEST,
            "backend_access_control": True,
            "score_semantics": SCORE_SEMANTICS,
            "embedding_execution_providers": list(
                self.embedding_execution_providers
            ),
            "embedding_intra_op_threads": self.embedding_intra_op_threads,
            "embedding_inter_op_threads": self.embedding_inter_op_threads,
        }

    def _point(self, payload: dict[str, Any]) -> Any:
        payload = validate_point_payload(payload, scope=self.scope)
        node_set_payload = payload["belongs_to_set"][0]
        node_set = self.NodeSet(
            id=UUID(self.scope.node_set_id), name=node_set_payload["name"]
        )
        references = []
        for reference in payload["references"]:
            target_payload = reference["target"]
            target = self.Target(
                id=UUID(target_payload["id"]),
                record_ref=target_payload["record_ref"],
            )
            edge = self.Edge(
                relationship_type=reference["relationship_type"],
                properties=reference["properties"],
            )
            references.append((edge, target))
        return self.Point(
            id=UUID(payload["id"]),
            acquisition_ref=payload["acquisition_ref"],
            source_contract=payload["source_contract"],
            source_ref=payload["source_ref"],
            record_ref=payload["record_ref"],
            projection_ref=payload["projection_ref"],
            acquired_ordinal=payload["acquired_ordinal"],
            search_text=payload["search_text"],
            adjacent_record_refs=payload["adjacent_record_refs"],
            references=references,
            visibility=payload["visibility"],
            memory_kind=payload["memory_kind"],
            epistemic_status=payload["epistemic_status"],
            belongs_to_set=[node_set],
        )

    def _require_bound_scope(self) -> None:
        self.validate_runtime_boundary()
        if (
            self.fastembed_engine_module.TextEmbedding
            is not self.bounded_fastembed_constructor
        ):
            raise CogneeWorkerProtocolError("Cognee FastEmbed constructor binding drifted")
        if self.backend_access_control_enabled() is not True:
            raise CogneeWorkerProtocolError("Cognee backend access control drifted")
        if (
            str(self.user.id) != self.user_id
            or str(self.user.tenant_id) != self.tenant_id
            or str(self.dataset.id) != self.dataset_id
            or str(self.dataset.owner_id) != self.user_id
            or str(self.dataset.tenant_id) != self.tenant_id
            or self.dataset_id
            != modern_dataset_id(
                self.user_id,
                self.tenant_id,
                dataset_name=self.scope.dataset_name,
            )
        ):
            raise CogneeWorkerProtocolError("Cognee bound scope drifted")

    async def project(self, payload: dict[str, Any]) -> dict[str, str]:
        self._require_bound_scope()
        point = self._point(payload)
        context = self.PipelineContext(user=self.user, dataset=self.dataset)
        async with self.set_database_context(
            self.dataset.id,
            self.user.id,
            embedding_config=self.embedding_config,
        ):
            await self.add_data_points_native([point], ctx=context, graph_only=True)
            unified = await self.get_unified_engine()
            if self.inspect_fastembed_runtime(unified.vector) != {
                "embedding_execution_providers": list(
                    self.embedding_execution_providers
                ),
                "embedding_intra_op_threads": self.embedding_intra_op_threads,
                "embedding_inter_op_threads": self.embedding_inter_op_threads,
            }:
                raise CogneeWorkerProtocolError("FastEmbed runtime boundary drifted")
            vector_clone = point.model_copy(
                deep=True,
                update={"belongs_to_set": [self.scope.node_set_name]},
            )
            await self.index_data_points_native(
                [vector_clone], vector_engine=unified.vector
            )
        self.validate_runtime_boundary()
        return {"backend_ref": str(point.id)}

    @staticmethod
    def _node_identity(node: object) -> str:
        if type(node) is not dict:
            raise CogneeWorkerProtocolError("graph join returned a malformed node")
        return canonical_uuid(str(node.get("id")), "graph node id")

    def _reference_payload(self, node: dict[str, Any]) -> dict[str, Any]:
        if node.get("type") != "AnglerAcquisitionReferencePoint":
            raise CogneeWorkerProtocolError("graph join returned a different node type")
        record_ref = digest(node.get("record_ref"), "graph node record_ref")
        adjacent = node.get("adjacent_record_refs")
        if (
            type(adjacent) is not list
            or len(adjacent) > MAX_POINT_NEIGHBORS
            or adjacent != sorted(set(adjacent))
        ):
            raise CogneeWorkerProtocolError("graph node adjacency is malformed")
        for target_ref in adjacent:
            digest(target_ref, "graph node adjacent_record_ref")
        # Required native fields prove this is the structured graph point.  They
        # are intentionally validated and then discarded rather than returned.
        for field in ("acquisition_ref", "source_ref", "projection_ref"):
            digest(node.get(field), f"graph node {field}")
        bounded_text(node.get("source_contract"), "graph node source_contract", 256)
        bounded_text(
            node.get("search_text"),
            "graph node search_text",
            16_384,
            allow_layout_controls=True,
        )
        for field in ("visibility", "memory_kind", "epistemic_status"):
            bounded_text(node.get(field), f"graph node {field}", 128)
        if type(node.get("acquired_ordinal")) is not int or node["acquired_ordinal"] < 0:
            raise CogneeWorkerProtocolError("graph node ordinal is malformed")
        if node.get("belongs_to_set") != [self.scope.node_set_name]:
            raise CogneeWorkerProtocolError("graph node NodeSet property differs")
        return {
            "record_ref": record_ref,
            "adjacent_record_refs": adjacent,
            "belongs_to_set": [self.scope.node_set_name],
        }

    async def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        self._require_bound_scope()
        async with self.set_database_context(
            self.dataset.id,
            self.user.id,
            embedding_config=self.embedding_config,
        ):
            unified = await self.get_unified_engine()
            if self.inspect_fastembed_runtime(unified.vector) != {
                "embedding_execution_providers": list(
                    self.embedding_execution_providers
                ),
                "embedding_intra_op_threads": self.embedding_intra_op_threads,
                "embedding_inter_op_threads": self.embedding_inter_op_threads,
            }:
                raise CogneeWorkerProtocolError("FastEmbed runtime boundary drifted")
            vector_hits = await unified.vector.search(
                collection_name=COLLECTION_NAME,
                query_text=query,
                limit=limit,
                include_payload=False,
                node_name=[self.scope.node_set_name],
            )
            if type(vector_hits) is not list or len(vector_hits) > limit:
                raise CogneeWorkerProtocolError("vector search result is malformed")
            ordered_ids = [
                canonical_uuid(str(hit.id), "vector search id") for hit in vector_hits
            ]
            if len(ordered_ids) != len(set(ordered_ids)):
                raise CogneeWorkerProtocolError("vector search returned duplicate ids")
            graph_nodes = await unified.graph.get_nodes(ordered_ids)
            graph_edges = [
                await unified.graph.get_edges(node_id) for node_id in ordered_ids
            ]

        joined: dict[str, dict[str, Any]] = {}
        if type(graph_nodes) is not list:
            raise CogneeWorkerProtocolError("graph join result is malformed")
        for raw_node in graph_nodes:
            node_id = self._node_identity(raw_node)
            if node_id in joined or node_id not in set(ordered_ids):
                raise CogneeWorkerProtocolError("graph join identity differs")
            joined[node_id] = raw_node
        if set(joined) != set(ordered_ids):
            raise CogneeWorkerProtocolError("graph join did not resolve every vector id")
        for node_id, node_edges in zip(ordered_ids, graph_edges, strict=True):
            if type(node_edges) is not list:
                raise CogneeWorkerProtocolError("graph join edges are malformed")
            memberships = []
            for raw_edge in node_edges:
                if type(raw_edge) not in (tuple, list) or len(raw_edge) != 3:
                    raise CogneeWorkerProtocolError("graph join edge is malformed")
                source, relationship, target = raw_edge
                if relationship != "belongs_to_set":
                    continue
                if type(source) is not dict or type(target) is not dict:
                    raise CogneeWorkerProtocolError(
                        "graph join NodeSet edge is malformed"
                    )
                memberships.append(
                    (
                        canonical_uuid(str(source.get("id")), "NodeSet edge source"),
                        canonical_uuid(str(target.get("id")), "NodeSet edge target"),
                        target.get("type"),
                        target.get("name"),
                    )
                )
            if memberships != [
                (
                    node_id,
                    self.scope.node_set_id,
                    "NodeSet",
                    self.scope.node_set_name,
                )
            ]:
                raise CogneeWorkerProtocolError("graph join NodeSet adjacency differs")

        entries = []
        previous_score: float | None = None
        for hit, node_id in zip(vector_hits, ordered_ids, strict=True):
            score = finite_score(hit.score, "vector search score")
            if score is None:
                raise CogneeWorkerProtocolError("vector search score is absent")
            if previous_score is not None and score < previous_score:
                raise CogneeWorkerProtocolError(
                    "vector search scores are not lower-is-better ordered"
                )
            previous_score = score
            entries.append(
                {
                    "id": node_id,
                    "score": score,
                    "payload": self._reference_payload(joined[node_id]),
                }
            )
        self.validate_runtime_boundary()
        return [
            {
                "tenant_id": self.tenant_id,
                "dataset_id": self.dataset_id,
                "dataset_name": self.scope.dataset_name,
                "node_set_name": self.scope.node_set_name,
                "score_semantics": SCORE_SEMANTICS,
                "search_result": entries,
            }
        ]

    async def forget_namespace(self) -> dict[str, Any]:
        self._require_bound_scope()
        frozen_id = self.dataset_id
        await self.forget_native(
            dataset_id=UUID(frozen_id),
            user=self.user,
        )
        recreated = await self.create_authorized_dataset(
            self.scope.dataset_name, self.user
        )
        delete_authorized = await self.get_authorized_dataset_by_name(
            self.scope.dataset_name, self.user, "delete"
        )
        if (
            str(recreated.id) != frozen_id
            or str(recreated.owner_id) != self.user_id
            or str(recreated.tenant_id) != self.tenant_id
            or frozen_id
            != modern_dataset_id(
                self.user_id,
                self.tenant_id,
                dataset_name=self.scope.dataset_name,
            )
            or delete_authorized is None
            or str(delete_authorized.id) != frozen_id
        ):
            raise CogneeWorkerProtocolError("exact dataset recreation failed")
        self.dataset = recreated
        self.validate_runtime_boundary()
        return {
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.scope.dataset_name,
            "node_set_name": self.scope.node_set_name,
            "recreated": True,
        }


async def _serve(
    scope: CogneeWorkerScope = DEFAULT_COGNEE_WORKER_SCOPE,
) -> int:
    scope = exact_worker_scope(scope)
    _validate_runtime_boundary(scope)
    backend: _CogneeBackend | None = None
    while True:
        line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
        if not line:
            return 0
        try:
            raw_request = decode_json_line(line)
            request = validate_request(raw_request, scope=scope)
        except CogneeWorkerProtocolError:
            return 2
        request_id = request["id"]
        operation = request["op"]
        try:
            if backend is None:
                if operation != "hello":
                    response = error_response(
                        request_id, "HELLO_REQUIRED", "Verified hello is required"
                    )
                    sys.stdout.buffer.write(encode_json_line(response))
                    sys.stdout.buffer.flush()
                    return 2
                backend = await _CogneeBackend.load(
                    expected_dataset_id=request["expected_dataset_id"],
                    scope=scope,
                )
                result: Any = backend.hello_result()
            elif operation == "hello":
                raise CogneeWorkerProtocolError("hello was already completed")
            elif operation == "project":
                result = await backend.project(request["point"])
            elif operation == "search":
                result = await backend.search(
                    request["query"], limit=request["limit"]
                )
            elif operation == "forget_namespace":
                result = await backend.forget_namespace()
            elif operation == "close":
                result = {"closed": True}
            else:
                raise CogneeWorkerProtocolError("operation is not admitted")
            response = success_response(request_id, result)
        except CogneeWorkerProtocolError:
            response = error_response(
                request_id,
                "PROTOCOL_VIOLATION",
                "Operation violated the frozen Cognee protocol",
            )
        except Exception:
            response = error_response(
                request_id,
                "COGNEE_OPERATION_FAILED",
                "Cognee operation failed",
            )
        sys.stdout.buffer.write(encode_json_line(response))
        sys.stdout.buffer.flush()
        if operation == "close":
            return 0


def main() -> int:
    try:
        scope = worker_scope_from_launch_arguments(sys.argv[1:])
        return asyncio.run(_serve(scope))
    except (Exception, KeyboardInterrupt):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
