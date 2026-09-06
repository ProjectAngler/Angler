from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest.mock import patch
from uuid import UUID, uuid5

from angler.cognition.contracts import CognitiveRelation, RelationType
from angler.memory.cognitive_acquisition import CognitiveGraphProjectionV2
from angler.memory.cognitive_acquisition_graph import (
    MAX_ACQUISITION_NEIGHBORS,
    MAX_ACQUISITION_PROVENANCE_REFS,
)
from angler.memory.cognee_acquisition_adapter import (
    CogneeAcquisitionAdapter,
    CogneeAcquisitionConfigurationError,
)
from tests.unit.memory.test_cognitive_acquisition_graph import _chain


_NAMESPACE = UUID("5022f10d-7871-5b67-a448-00ea3fbf0d51")


def _scoped_ref(
    api: "FakeStructuredCognee", kind: str, identity: str
) -> str:
    material = "\x1f".join(
        (api.tenant_id, api.dataset_id, api.node_set_name, kind, identity)
    )
    return str(uuid5(_NAMESPACE, material))


def _backend_ref(api: "FakeStructuredCognee", record_ref: str) -> str:
    return _scoped_ref(api, "acquisition-record", record_ref)


class FakeDataPoint:
    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__()

    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class FakeEdge(FakeDataPoint):
    pass


class FakeNodeSet(FakeDataPoint):
    pass


class FakeStructuredCognee:
    DataPoint = FakeDataPoint
    Edge = FakeEdge
    NodeSet = FakeNodeSet
    tenant_id = "tenant-test"
    dataset_id = "dataset-id-test"
    dataset_name = "angler-acquisition-test"
    node_set_name = "angler-acquisition-test-records"

    def __init__(self) -> None:
        self.batches: list[list[object]] = []
        self.queries: list[tuple[str, int]] = []
        self.results: object = []
        self.forgotten = 0
        self.forbidden_calls: list[str] = []

    async def add_data_points(self, data_points):
        self.batches.append(data_points)
        return data_points

    async def search_references(self, query: str, *, limit: int):
        self.queries.append((query, limit))
        return self.results

    async def forget_namespace(self) -> None:
        self.forgotten += 1

    async def remember(self, *args, **kwargs):
        self.forbidden_calls.append("remember")
        raise AssertionError("LLM extraction path called")

    async def improve(self, *args, **kwargs):
        self.forbidden_calls.append("improve")
        raise AssertionError("improve path called")

    async def memify(self, *args, **kwargs):
        self.forbidden_calls.append("memify")
        raise AssertionError("memify path called")


def _projection() -> CognitiveGraphProjectionV2:
    return CognitiveGraphProjectionV2.from_acquisition(_chain()[1].acquisition)


def _adapter(
    api: FakeStructuredCognee, **overrides
) -> CogneeAcquisitionAdapter:
    values = {
        "bindings": api,
        "tenant_id": api.tenant_id,
        "dataset_id": api.dataset_id,
        "dataset_name": api.dataset_name,
        "node_set_name": api.node_set_name,
        "local_embeddings_configured": True,
    }
    values.update(overrides)
    return CogneeAcquisitionAdapter(**values)


class CogneeAcquisitionAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_projects_exact_structured_point_with_scoped_identity(self) -> None:
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        projection = _projection()
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            first = await adapter.project(projection)
            second = await adapter.project(projection)

        self.assertEqual(first, second)
        self.assertEqual(first, _backend_ref(api, projection.record.record_ref))
        self.assertEqual(len(api.batches), 2)
        point = api.batches[0][0]
        self.assertIsInstance(point, FakeDataPoint)
        self.assertEqual(str(point.id), first)
        self.assertEqual(point.acquisition_ref, projection.acquisition_ref)
        self.assertEqual(point.source_contract, projection.source_contract)
        self.assertEqual(point.source_ref, projection.source_ref)
        self.assertEqual(point.record_ref, projection.record.record_ref)
        self.assertEqual(point.projection_ref, projection.projection_ref)
        self.assertEqual(
            point.acquired_ordinal, projection.record.acquired_ordinal
        )
        self.assertEqual(point.search_text, projection.record.content)
        self.assertEqual(point.visibility, projection.record.visibility)
        self.assertEqual(
            point.epistemic_status,
            projection.record.epistemic_status.value,
        )
        self.assertEqual(point.belongs_to_set[0].name, api.node_set_name)
        self.assertEqual(
            str(point.belongs_to_set[0].id),
            _scoped_ref(api, "node-set", api.node_set_name),
        )
        expected_edges = sorted(
            (
                relation.relation_type.value,
                relation.target_ref,
            )
            for relation in projection.record.relations
        )
        self.assertEqual(
            [
                (edge.relationship_type, target.record_ref)
                for edge, target in point.references
            ],
            expected_edges,
        )
        self.assertEqual(api.forbidden_calls, [])

    async def test_search_returns_only_refs_in_backend_order(self) -> None:
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        first, second = _chain()
        records = (first.record_ref, second.record_ref)
        api.results = [
            {
                "tenant_id": api.tenant_id,
                "dataset_id": api.dataset_id,
                "dataset_name": api.dataset_name,
                "node_set_name": api.node_set_name,
                "search_result": [
                    {
                        "id": _backend_ref(api, record_ref),
                        "score": score,
                        "payload": {
                            "record_ref": record_ref,
                            "adjacent_record_refs": [],
                            "belongs_to_set": [api.node_set_name],
                            "source_ref": "untrusted-and-discarded",
                            "epistemic_status": "VALIDATED",
                            "content": "untrusted-and-discarded",
                        },
                    }
                    for record_ref, score in zip(records, (0.8, 0.7), strict=True)
                ],
            }
        ]
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            hits = await adapter.search("public query", limit=2)

        self.assertEqual(api.queries, [("public query", 2)])
        self.assertEqual(tuple(hit.record_ref for hit in hits), records)
        self.assertEqual(tuple(hit.score for hit in hits), (0.8, 0.7))
        self.assertEqual(
            tuple(hit.backend_ref for hit in hits),
            tuple(_backend_ref(api, record_ref) for record_ref in records),
        )
        self.assertFalse(hasattr(hits[0], "content"))
        self.assertFalse(hasattr(hits[0], "source_ref"))
        self.assertFalse(hasattr(hits[0], "epistemic_status"))
        self.assertEqual(api.forbidden_calls, [])

    async def test_search_rejects_scope_nodeset_identity_order_and_limit_tamper(
        self,
    ) -> None:
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        record_ref = _projection().record.record_ref
        base_entry = {
            "id": _backend_ref(api, record_ref),
            "payload": {
                "record_ref": record_ref,
                "adjacent_record_refs": [],
                "belongs_to_set": [api.node_set_name],
            },
        }
        base = {
            "tenant_id": api.tenant_id,
            "dataset_id": api.dataset_id,
            "dataset_name": api.dataset_name,
            "node_set_name": api.node_set_name,
            "search_result": [base_entry],
        }
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            api.results = [{**base, "dataset_name": "other"}]
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "scope"
            ):
                await adapter.search("query", limit=1)

            api.results = [
                {
                    **base,
                    "search_result": [
                        {
                            **base_entry,
                            "payload": {
                                **base_entry["payload"],
                                "belongs_to_set": ["other"],
                            },
                        }
                    ],
                }
            ]
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "NodeSet"
            ):
                await adapter.search("query", limit=1)

            api.results = [
                {
                    **base,
                    "search_result": [{**base_entry, "id": "forged"}],
                }
            ]
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "backend id"
            ):
                await adapter.search("query", limit=1)

            api.results = [
                {
                    **base,
                    "search_result": [base_entry, base_entry],
                }
            ]
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError,
                "more references than requested",
            ):
                await adapter.search("query", limit=1)

            api.results = [
                {
                    **base,
                    "search_result": [
                        {
                            **base_entry,
                            "payload": {
                                **base_entry["payload"],
                                "adjacent_record_refs": [
                                    records := _chain()[1].record_ref,
                                    records,
                                ],
                            },
                        }
                    ],
                }
            ]
            with self.assertRaisesRegex(ValueError, "canonically sorted"):
                await adapter.search("query", limit=1)

    async def test_authority_and_constructor_scope_fail_closed(self) -> None:
        api = FakeStructuredCognee()
        with self.assertRaisesRegex(
            CogneeAcquisitionConfigurationError, "dataset_name"
        ):
            _adapter(api, dataset_name="wrong")
        for field in (
            "local_embeddings_configured",
            "external_embedding_calls_authorized",
            "telemetry_authorized",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, field):
                    _adapter(api, **{field: "true"})

        projection = _projection()
        no_authority = _adapter(api, local_embeddings_configured=False)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "telemetry"
            ):
                await no_authority.project(projection)
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=True):
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "embedding"
            ):
                await no_authority.project(projection)
        self.assertEqual(api.batches, [])

    async def test_search_scope_ids_require_exact_strings(self) -> None:
        api = FakeStructuredCognee()
        api.tenant_id = "8"
        api.dataset_id = "7"
        adapter = _adapter(api)
        api.results = [
            {
                "tenant_id": 8,
                "dataset_id": 7,
                "dataset_name": api.dataset_name,
                "node_set_name": api.node_set_name,
                "search_result": [],
            }
        ]
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "scope"
            ):
                await adapter.search("query", limit=1)

    def test_scope_rejects_ambiguous_uuid_delimiter_material(self) -> None:
        first = FakeStructuredCognee()
        first.tenant_id = "tenant\x1fdataset"
        with self.assertRaisesRegex(ValueError, "control characters"):
            _adapter(first)

        second = FakeStructuredCognee()
        second.dataset_id = "dataset\x1fnode-set"
        with self.assertRaisesRegex(ValueError, "control characters"):
            _adapter(second)

    async def test_binding_scope_drift_blocks_every_external_call(self) -> None:
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        api.dataset_id = "drifted-after-construction"
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "dataset_id"
            ):
                await adapter.project(_projection())
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "dataset_id"
            ):
                await adapter.search("query", limit=1)
            with self.assertRaisesRegex(
                CogneeAcquisitionConfigurationError, "dataset_id"
            ):
                await adapter.forget_namespace()
        self.assertEqual(api.batches, [])
        self.assertEqual(api.queries, [])
        self.assertEqual(api.forgotten, 0)

    async def test_canonical_and_untrusted_neighbors_are_resource_bounded(
        self,
    ) -> None:
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        base = _projection()
        refs = tuple(
            f"sha256:{index:064x}"
            for index in range(1, MAX_ACQUISITION_NEIGHBORS + 2)
        )

        def with_relations(count: int) -> CognitiveGraphProjectionV2:
            record = replace(
                base.record,
                relations=tuple(
                    CognitiveRelation(RelationType.ABOUT, target_ref)
                    for target_ref in refs[:count]
                ),
            )
            return CognitiveGraphProjectionV2(
                acquisition_ref=base.acquisition_ref,
                source_contract=base.source_contract,
                source_ref=base.source_ref,
                record=record,
            )

        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            await adapter.project(with_relations(MAX_ACQUISITION_NEIGHBORS))
            self.assertEqual(
                len(api.batches[-1][0].references), MAX_ACQUISITION_NEIGHBORS
            )
            with self.assertRaisesRegex(ValueError, "neighbor local ceiling"):
                await adapter.project(
                    with_relations(MAX_ACQUISITION_NEIGHBORS + 1)
                )

            record_ref = base.record.record_ref
            envelope = {
                "tenant_id": api.tenant_id,
                "dataset_id": api.dataset_id,
                "dataset_name": api.dataset_name,
                "node_set_name": api.node_set_name,
                "search_result": [
                    {
                        "id": _backend_ref(api, record_ref),
                        "payload": {
                            "record_ref": record_ref,
                            "adjacent_record_refs": list(refs),
                            "belongs_to_set": [api.node_set_name],
                        },
                    }
                ],
            }
            api.results = [envelope]
            with self.assertRaisesRegex(ValueError, "local ceiling"):
                await adapter.search("query", limit=1)

    async def test_canonical_provenance_is_bounded_before_serialization(self) -> None:
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        base = _projection()
        extra_refs = tuple(
            f"sha256:{index:064x}"
            for index in range(1, MAX_ACQUISITION_PROVENANCE_REFS + 1)
        )

        def with_provenance(extra_count: int) -> CognitiveGraphProjectionV2:
            record = replace(
                base.record,
                provenance_refs=tuple(
                    sorted({base.source_ref, *extra_refs[:extra_count]})
                ),
            )
            return CognitiveGraphProjectionV2(
                acquisition_ref=base.acquisition_ref,
                source_contract=base.source_contract,
                source_ref=base.source_ref,
                record=record,
            )

        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            accepted = with_provenance(MAX_ACQUISITION_PROVENANCE_REFS - 1)
            self.assertEqual(
                len(accepted.record.provenance_refs),
                MAX_ACQUISITION_PROVENANCE_REFS,
            )
            await adapter.project(accepted)
            rejected = with_provenance(MAX_ACQUISITION_PROVENANCE_REFS)
            self.assertEqual(
                len(rejected.record.provenance_refs),
                MAX_ACQUISITION_PROVENANCE_REFS + 1,
            )
            with self.assertRaisesRegex(ValueError, "provenance-reference"):
                await adapter.project(rejected)

    async def test_same_record_has_distinct_ids_in_two_scopes(self) -> None:
        first_api = FakeStructuredCognee()
        second_api = FakeStructuredCognee()
        second_api.tenant_id = "tenant-other"
        second_api.dataset_id = "dataset-id-other"
        second_api.dataset_name = "angler-acquisition-other"
        second_api.node_set_name = "angler-acquisition-other-records"
        projection = _projection()
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            first = await _adapter(first_api).project(projection)
            second = await _adapter(second_api).project(projection)
        self.assertEqual(first, _backend_ref(first_api, projection.record.record_ref))
        self.assertEqual(
            second, _backend_ref(second_api, projection.record.record_ref)
        )
        self.assertNotEqual(first, second)

    async def test_forget_calls_only_prebound_namespace(self) -> None:
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            await adapter.forget_namespace()
        self.assertEqual(api.forgotten, 1)
        self.assertEqual(api.forbidden_calls, [])


if __name__ == "__main__":
    unittest.main()
