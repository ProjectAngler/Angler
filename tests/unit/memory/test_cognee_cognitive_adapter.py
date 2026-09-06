from __future__ import annotations

from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch
from uuid import UUID, uuid5

from angler.cognition.contracts import (
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    CognitiveRelation,
    EpistemicStatus,
    RelationType,
)
from angler.memory.cognitive_graph import CognitiveGraphProjection
from angler.memory.cognee_cognitive_adapter import (
    CogneeCognitiveAdapter,
    CogneeCognitiveConfigurationError,
)


_NAMESPACE = UUID("5022f10d-7871-5b67-a448-00ea3fbf0d51")


def _ref(index: int) -> str:
    return f"sha256:{index:064x}"


def _scoped_ref(api: "FakeStructuredCognee", kind: str, identity: str) -> str:
    material = "\x1f".join(
        (api.tenant_id, api.dataset_id, api.node_set_name, kind, identity)
    )
    return str(uuid5(_NAMESPACE, material))


def _backend_ref(api: "FakeStructuredCognee", record_ref: str) -> str:
    return _scoped_ref(api, "record", record_ref)


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
    dataset_name = "angler-cognitive-test"
    node_set_name = "angler-cognitive-test-records"

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


def _projection() -> CognitiveGraphProjection:
    record = CognitiveMemoryRecord(
        kind=CognitiveMemoryKind.EPISODIC,
        epistemic_status=EpistemicStatus.OBSERVED,
        content="Public request, selected procedure, objective outcome, and feedback.",
        provenance_refs=tuple(sorted((_ref(10), _ref(11), _ref(12)))),
        visibility="LEARNER_VISIBLE",
        producer_id="angler.cognitive-cycle",
        producer_checkpoint_ref=_ref(13),
        competence_ref=_ref(14),
        acquired_ordinal=4,
        world_valid_from=20,
        relations=(
            CognitiveRelation(RelationType.ABOUT, _ref(21)),
            CognitiveRelation(RelationType.ACTUAL_OF, _ref(22)),
        ),
        supersedes_refs=(_ref(23),),
    )
    return CognitiveGraphProjection(source_episode_ref=_ref(10), record=record)


def _adapter(api: FakeStructuredCognee, **overrides) -> CogneeCognitiveAdapter:
    values = {
        "bindings": api,
        "tenant_id": api.tenant_id,
        "dataset_id": api.dataset_id,
        "dataset_name": api.dataset_name,
        "node_set_name": api.node_set_name,
        "local_embeddings_configured": True,
    }
    values.update(overrides)
    return CogneeCognitiveAdapter(**values)


class CogneeCognitiveAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_projects_native_points_with_deterministic_ids_and_typed_edges(self):
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
        self.assertEqual(point.record_ref, projection.record.record_ref)
        self.assertEqual(point.source_episode_ref, projection.source_episode_ref)
        self.assertEqual(point.projection_ref, projection.projection_ref)
        self.assertEqual(
            point.adjacent_record_refs,
            sorted((_ref(21), _ref(22), _ref(23))),
        )
        self.assertEqual(
            [(edge.relationship_type, target.record_ref) for edge, target in point.references],
            [
                ("ABOUT", _ref(21)),
                ("ACTUAL_OF", _ref(22)),
                ("SUPERSEDES", _ref(23)),
            ],
        )
        self.assertEqual(point.belongs_to_set[0].name, api.node_set_name)
        self.assertEqual(api.forbidden_calls, [])

    async def test_search_returns_only_exact_reference_hits_from_frozen_scope(self):
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        projection = _projection()
        api.results = [
            {
                "tenant_id": api.tenant_id,
                "dataset_id": api.dataset_id,
                "dataset_name": api.dataset_name,
                "node_set_name": api.node_set_name,
                "search_result": [
                    {
                        "id": _backend_ref(api, projection.record.record_ref),
                        "score": 0.75,
                        "payload": {
                            "record_ref": projection.record.record_ref,
                            "source_episode_ref": projection.source_episode_ref,
                            "adjacent_record_refs": sorted((_ref(21), _ref(22))),
                            "belongs_to_set": [api.node_set_name],
                            "untrusted_content": "must never leave the adapter",
                        },
                    }
                ],
            }
        ]
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            hits = await adapter.search("public query", limit=3)

        self.assertEqual(api.queries, [("public query", 3)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].record_ref, projection.record.record_ref)
        self.assertEqual(hits[0].source_episode_ref, projection.source_episode_ref)
        self.assertEqual(hits[0].adjacent_record_refs, (_ref(21), _ref(22)))
        self.assertEqual(
            hits[0].backend_ref, _backend_ref(api, projection.record.record_ref)
        )
        self.assertFalse(hasattr(hits[0], "untrusted_content"))
        self.assertEqual(api.forbidden_calls, [])

    async def test_wrong_dataset_nodeset_or_backend_identity_fails_closed(self):
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        projection = _projection()
        base = {
            "tenant_id": api.tenant_id,
            "dataset_id": api.dataset_id,
            "dataset_name": api.dataset_name,
            "node_set_name": api.node_set_name,
            "search_result": [],
        }
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            api.results = [{**base, "dataset_name": "other"}]
            with self.assertRaisesRegex(CogneeCognitiveConfigurationError, "scope"):
                await adapter.search("query", limit=1)

            api.results = [{**base, "search_result": [{
                "id": _backend_ref(api, projection.record.record_ref),
                "payload": {
                    "record_ref": projection.record.record_ref,
                    "source_episode_ref": projection.source_episode_ref,
                    "adjacent_record_refs": [],
                    "belongs_to_set": ["other-node-set"],
                },
            }]}]
            with self.assertRaisesRegex(CogneeCognitiveConfigurationError, "NodeSet"):
                await adapter.search("query", limit=1)

            api.results[0]["search_result"][0]["payload"]["belongs_to_set"] = [
                api.node_set_name
            ]
            api.results[0]["search_result"][0]["id"] = "forged"
            with self.assertRaisesRegex(CogneeCognitiveConfigurationError, "backend id"):
                await adapter.search("query", limit=1)

    async def test_authority_and_constructor_scope_are_fail_closed(self):
        api = FakeStructuredCognee()
        with self.assertRaisesRegex(CogneeCognitiveConfigurationError, "dataset_name"):
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
            with self.assertRaisesRegex(CogneeCognitiveConfigurationError, "telemetry"):
                await no_authority.project(projection)
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "true"}, clear=True):
            with self.assertRaisesRegex(CogneeCognitiveConfigurationError, "telemetry"):
                await no_authority.project(projection)
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=True):
            with self.assertRaisesRegex(CogneeCognitiveConfigurationError, "embedding"):
                await no_authority.project(projection)
        self.assertEqual(api.batches, [])

    async def test_search_rejects_more_references_than_requested(self):
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        api.results = [
            {
                "tenant_id": api.tenant_id,
                "dataset_id": api.dataset_id,
                "dataset_name": api.dataset_name,
                "node_set_name": api.node_set_name,
                "search_result": [{}, {}],
            }
        ]
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            with self.assertRaisesRegex(
                CogneeCognitiveConfigurationError, "more references than requested"
            ):
                await adapter.search("public query", limit=1)

    async def test_same_record_has_distinct_recomputable_ids_in_two_scopes(self):
        first_api = FakeStructuredCognee()
        second_api = FakeStructuredCognee()
        second_api.tenant_id = "tenant-other"
        second_api.dataset_id = "dataset-id-other"
        second_api.dataset_name = "angler-cognitive-other"
        second_api.node_set_name = "angler-cognitive-other-records"
        projection = _projection()

        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            first_id = await _adapter(first_api).project(projection)
            second_id = await _adapter(second_api).project(projection)

        self.assertEqual(first_id, _backend_ref(first_api, projection.record.record_ref))
        self.assertEqual(second_id, _backend_ref(second_api, projection.record.record_ref))
        self.assertNotEqual(first_id, second_id)
        first_node_set_id = first_api.batches[0][0].belongs_to_set[0].id
        second_node_set_id = second_api.batches[0][0].belongs_to_set[0].id
        self.assertEqual(
            str(first_node_set_id),
            _scoped_ref(first_api, "node-set", first_api.node_set_name),
        )
        self.assertEqual(
            str(second_node_set_id),
            _scoped_ref(second_api, "node-set", second_api.node_set_name),
        )
        self.assertNotEqual(first_node_set_id, second_node_set_id)

    async def test_forget_calls_only_the_prebound_namespace(self):
        api = FakeStructuredCognee()
        adapter = _adapter(api)
        with patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}, clear=False):
            await adapter.forget_namespace()
        self.assertEqual(api.forgotten, 1)
        self.assertEqual(api.forbidden_calls, [])


if __name__ == "__main__":
    unittest.main()
