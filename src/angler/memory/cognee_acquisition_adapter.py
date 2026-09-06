"""Structured, reference-only Cognee adapter for generic acquisitions.

Cognee is a disposable search view.  The adapter inserts bounded canonical
record fields but returns only record references and optional adjacency refs;
``AcquisitionSituatedMemory`` revalidates those references against schema-v3
acquisition bytes before exposing content or epistemic state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from typing import Any, Protocol, Sequence, runtime_checkable
import unicodedata
from uuid import UUID, uuid5

from angler.cognition.contracts import RelationType

from .cognitive_acquisition import CognitiveGraphProjectionV2
from .cognitive_acquisition_graph import (
    MAX_ACQUISITION_NEIGHBORS,
    MAX_ACQUISITION_PROVENANCE_REFS,
    AcquisitionReferenceHit,
)


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY_NAMESPACE = UUID("5022f10d-7871-5b67-a448-00ea3fbf0d51")


class CogneeAcquisitionConfigurationError(RuntimeError):
    """The structured Cognee boundary is absent, unscoped, or unauthorized."""


@runtime_checkable
class CogneeAcquisitionBindings(Protocol):
    """Operations pre-bound to one tenant, dataset, and NodeSet."""

    DataPoint: type
    Edge: type
    NodeSet: type
    tenant_id: str
    dataset_id: str
    dataset_name: str
    node_set_name: str

    async def add_data_points(self, data_points: list[Any]) -> Any: ...

    async def search_references(self, query: str, *, limit: int) -> Any: ...

    async def forget_namespace(self) -> None: ...


def _bounded_identity(value: Any, label: str, *, maximum: int = 256) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(
            f"{label} must be non-empty text of at most {maximum} characters"
        )
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{label} must be Unicode NFC-normalized")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


@dataclass(frozen=True, slots=True)
class _Scope:
    tenant_id: str
    dataset_id: str
    dataset_name: str
    node_set_name: str


def _scoped_uuid(scope: _Scope, kind: str, identity: str) -> UUID:
    material = "\x1f".join(
        (
            scope.tenant_id,
            scope.dataset_id,
            scope.node_set_name,
            kind,
            identity,
        )
    )
    return uuid5(_IDENTITY_NAMESPACE, material)


def _entry_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _payload(value: Any) -> dict[str, Any]:
    raw = _entry_value(value, "payload")
    if isinstance(raw, dict):
        return raw
    raw = _entry_value(value, "raw")
    if isinstance(raw, dict):
        nested = raw.get("payload")
        return nested if isinstance(nested, dict) else raw
    return value if isinstance(value, dict) else {}


def _dynamic_point_types(
    bindings: CogneeAcquisitionBindings,
) -> tuple[type, type]:
    data_point = bindings.DataPoint
    point = type(
        "AnglerAcquisitionReferencePoint",
        (data_point,),
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
    target = type(
        "AnglerAcquisitionReferenceTarget",
        (data_point,),
        {
            "__module__": __name__,
            "__annotations__": {"record_ref": str, "metadata": dict},
            "metadata": {"index_fields": [], "identity_fields": ["record_ref"]},
        },
    )
    return point, target


class CogneeAcquisitionAdapter:
    """Structured acquisition projection for one frozen Cognee namespace."""

    def __init__(
        self,
        *,
        bindings: CogneeAcquisitionBindings,
        tenant_id: str,
        dataset_id: str,
        dataset_name: str,
        node_set_name: str,
        local_embeddings_configured: bool = False,
        external_embedding_calls_authorized: bool = False,
        telemetry_authorized: bool = False,
    ) -> None:
        if not isinstance(bindings, CogneeAcquisitionBindings):
            raise TypeError("bindings must implement CogneeAcquisitionBindings")
        self._scope = _Scope(
            tenant_id=_bounded_identity(tenant_id, "tenant_id"),
            dataset_id=_bounded_identity(dataset_id, "dataset_id"),
            dataset_name=_bounded_identity(dataset_name, "dataset_name"),
            node_set_name=_bounded_identity(node_set_name, "node_set_name"),
        )
        for label, value in (
            ("local_embeddings_configured", local_embeddings_configured),
            (
                "external_embedding_calls_authorized",
                external_embedding_calls_authorized,
            ),
            ("telemetry_authorized", telemetry_authorized),
        ):
            if type(value) is not bool:
                raise TypeError(f"{label} must be a bool")
        self._bindings = bindings
        self._local_embeddings_configured = local_embeddings_configured
        self._external_embedding_calls_authorized = (
            external_embedding_calls_authorized
        )
        self._telemetry_authorized = telemetry_authorized
        self._point_type, self._target_type = _dynamic_point_types(bindings)
        self._require_frozen_scope()

    @property
    def dataset_name(self) -> str:
        return self._scope.dataset_name

    @property
    def node_set_name(self) -> str:
        return self._scope.node_set_name

    def _require_telemetry_authority(self) -> None:
        if os.getenv("TELEMETRY_DISABLED") != "1" and not self._telemetry_authorized:
            raise CogneeAcquisitionConfigurationError(
                "Cognee telemetry is not authorized; set TELEMETRY_DISABLED=1 or "
                "explicitly authorize telemetry"
            )

    def _require_embedding_authority(self) -> None:
        if not (
            self._local_embeddings_configured
            or self._external_embedding_calls_authorized
        ):
            raise CogneeAcquisitionConfigurationError(
                "structured projection/search requires a configured local embedding "
                "model or explicit external embedding-call authority"
            )

    def _require_frozen_scope(self) -> None:
        for field in ("tenant_id", "dataset_id", "dataset_name", "node_set_name"):
            binding_value = getattr(self._bindings, field)
            if (
                type(binding_value) is not str
                or binding_value != getattr(self._scope, field)
            ):
                raise CogneeAcquisitionConfigurationError(
                    f"structured Cognee binding {field} does not match the frozen scope"
                )

    def _node_set(self) -> Any:
        return self._bindings.NodeSet(
            id=_scoped_uuid(
                self._scope, "node-set", self._scope.node_set_name
            ),
            name=self._scope.node_set_name,
        )

    @staticmethod
    def _canonical_projection(
        projection: object,
    ) -> CognitiveGraphProjectionV2:
        if type(projection) is not CognitiveGraphProjectionV2:
            raise TypeError(
                "projection must be an exact CognitiveGraphProjectionV2"
            )
        record = projection.record
        total = (
            len(record.relations)
            + len(record.supersedes_refs)
            + len(record.tombstone_refs)
        )
        if total > MAX_ACQUISITION_NEIGHBORS:
            raise ValueError(
                "canonical record exceeds the 384-neighbor local ceiling"
            )
        if len(record.provenance_refs) > MAX_ACQUISITION_PROVENANCE_REFS:
            raise ValueError(
                "canonical record exceeds the 384-provenance-reference local ceiling"
            )
        encoded = projection.canonical_bytes()
        restored = CognitiveGraphProjectionV2.from_json(encoded)
        if (
            restored.canonical_bytes() != encoded
            or restored.projection_ref != projection.projection_ref
        ):
            raise ValueError(
                "projection does not round-trip through its canonical contract"
            )
        return restored

    @staticmethod
    def _adjacent_edges(
        projection: CognitiveGraphProjectionV2,
    ) -> tuple[tuple[str, str], ...]:
        record = projection.record
        total = (
            len(record.relations)
            + len(record.supersedes_refs)
            + len(record.tombstone_refs)
        )
        if total > MAX_ACQUISITION_NEIGHBORS:
            raise ValueError(
                "canonical record exceeds the 384-neighbor local ceiling"
            )
        edges = [
            (relation.relation_type.value, relation.target_ref)
            for relation in record.relations
        ]
        edges.extend(
            (RelationType.SUPERSEDES.value, ref)
            for ref in record.supersedes_refs
        )
        edges.extend(
            (RelationType.TOMBSTONES.value, ref)
            for ref in record.tombstone_refs
        )
        return tuple(sorted(edges))

    def _structured_point(self, projection: CognitiveGraphProjectionV2) -> Any:
        record = projection.record
        edges = self._adjacent_edges(projection)
        references = []
        for relation_type, target_ref in edges:
            target = self._target_type(
                id=_scoped_uuid(
                    self._scope, "target-record", target_ref
                ),
                record_ref=target_ref,
            )
            edge = self._bindings.Edge(
                relationship_type=relation_type,
                properties={"target_ref": target_ref},
            )
            references.append((edge, target))
        adjacent_refs = sorted({target_ref for _, target_ref in edges})
        return self._point_type(
            id=_scoped_uuid(
                self._scope, "acquisition-record", record.record_ref
            ),
            acquisition_ref=projection.acquisition_ref,
            source_contract=projection.source_contract,
            source_ref=projection.source_ref,
            record_ref=record.record_ref,
            projection_ref=projection.projection_ref,
            acquired_ordinal=record.acquired_ordinal,
            search_text=record.content,
            adjacent_record_refs=adjacent_refs,
            references=references,
            visibility=record.visibility,
            memory_kind=record.kind.value,
            epistemic_status=record.epistemic_status.value,
            belongs_to_set=[self._node_set()],
        )

    async def project(self, projection: CognitiveGraphProjectionV2) -> str:
        projection = self._canonical_projection(projection)
        self._require_frozen_scope()
        self._require_telemetry_authority()
        self._require_embedding_authority()
        point = self._structured_point(projection)
        await self._bindings.add_data_points([point])
        return str(
            _scoped_uuid(
                self._scope,
                "acquisition-record",
                projection.record.record_ref,
            )
        )

    def _result_entries(self, results: Any, *, limit: int) -> tuple[Any, ...]:
        if not isinstance(results, (list, tuple)):
            raise CogneeAcquisitionConfigurationError(
                "Cognee reference search returned no list"
            )
        if len(results) > limit:
            raise CogneeAcquisitionConfigurationError(
                "Cognee reference search returned more envelopes than requested"
            )
        entries: list[Any] = []
        for envelope in results:
            dataset_name = _entry_value(envelope, "dataset_name")
            dataset_id = _entry_value(envelope, "dataset_id")
            tenant_id = _entry_value(envelope, "tenant_id")
            node_set_name = _entry_value(envelope, "node_set_name")
            if (
                type(dataset_name) is not str
                or type(dataset_id) is not str
                or type(tenant_id) is not str
                or type(node_set_name) is not str
                or dataset_name != self._scope.dataset_name
                or dataset_id != self._scope.dataset_id
                or tenant_id != self._scope.tenant_id
                or node_set_name != self._scope.node_set_name
            ):
                raise CogneeAcquisitionConfigurationError(
                    "Cognee reference result does not prove the frozen dataset scope"
                )
            search_result = _entry_value(envelope, "search_result")
            if not isinstance(search_result, (list, tuple)):
                raise CogneeAcquisitionConfigurationError(
                    "Cognee reference result has no structured search_result list"
                )
            if len(search_result) > limit - len(entries):
                raise CogneeAcquisitionConfigurationError(
                    "Cognee reference search returned more references than requested"
                )
            entries.extend(search_result)
        return tuple(entries)

    def _hit(self, entry: Any) -> AcquisitionReferenceHit:
        payload = _payload(entry)
        record_ref = _digest(payload.get("record_ref"), "record_ref")
        adjacent = payload.get("adjacent_record_refs", [])
        if type(adjacent) is not list:
            raise ValueError("adjacent_record_refs must be a JSON list")
        if len(adjacent) > MAX_ACQUISITION_NEIGHBORS:
            raise ValueError(
                "adjacent_record_refs exceeds the 384-reference local ceiling"
            )
        adjacent_refs = tuple(adjacent)
        if adjacent_refs != tuple(sorted(set(adjacent_refs))):
            raise ValueError(
                "adjacent_record_refs must be unique and canonically sorted"
            )
        for target_ref in adjacent_refs:
            _digest(target_ref, "adjacent_record_ref")
        belongs_to_set = payload.get("belongs_to_set")
        node_set_value = (
            None
            if not isinstance(belongs_to_set, (list, tuple))
            or len(belongs_to_set) != 1
            else _entry_value(
                belongs_to_set[0], "name", belongs_to_set[0]
            )
        )
        if (
            type(node_set_value) is not str
            or node_set_value != self._scope.node_set_name
        ):
            raise CogneeAcquisitionConfigurationError(
                "Cognee reference payload does not prove the exact frozen NodeSet"
            )
        backend_ref = _entry_value(entry, "id", payload.get("id"))
        expected_ref = str(
            _scoped_uuid(self._scope, "acquisition-record", record_ref)
        )
        if type(backend_ref) is not str or backend_ref != expected_ref:
            raise CogneeAcquisitionConfigurationError(
                "Cognee backend id does not match the deterministic record identity"
            )
        score = _entry_value(entry, "score", payload.get("score"))
        if score is not None:
            if type(score) not in (int, float) or not math.isfinite(float(score)):
                raise ValueError("Cognee score must be finite or absent")
            score = float(score)
        return AcquisitionReferenceHit(
            record_ref=record_ref,
            adjacent_record_refs=adjacent_refs,
            score=score,
            backend_ref=expected_ref,
        )

    async def search(
        self, query: str, *, limit: int
    ) -> Sequence[AcquisitionReferenceHit]:
        if type(query) is not str or not query.strip() or len(query) > 16_384:
            raise ValueError(
                "query must be non-empty text of at most 16384 characters"
            )
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        self._require_frozen_scope()
        self._require_telemetry_authority()
        self._require_embedding_authority()
        results = await self._bindings.search_references(query, limit=limit)
        return tuple(
            self._hit(entry)
            for entry in self._result_entries(results, limit=limit)
        )

    async def forget_namespace(self) -> None:
        self._require_frozen_scope()
        self._require_telemetry_authority()
        await self._bindings.forget_namespace()


__all__ = [
    "CogneeAcquisitionAdapter",
    "CogneeAcquisitionBindings",
    "CogneeAcquisitionConfigurationError",
]
