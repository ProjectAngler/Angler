from __future__ import annotations

from dataclasses import dataclass
import unittest

from angler.memory import (
    AFTER,
    BEFORE,
    CogneeConfigurationError,
    CogneeProjectionBackend,
    FairNaiveTemporalView,
    MemoryHit,
    MemoryProjection,
    MovingOriginIndex,
    SituatedMemory,
    decode_projection,
    encode_projection,
    projection_id,
)


def _ref(index: int) -> str:
    return f"sha256:{index:064x}"


class FakeBackend:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.hits: tuple[MemoryHit, ...] = ()
        self.forgotten = False

    async def remember(self, document: str) -> str:
        self.documents.append(document)
        return f"backend-{len(self.documents)}"

    async def remember_many(self, documents: tuple[str, ...]) -> tuple[str, ...]:
        self.documents.extend(documents)
        return tuple(f"backend-{index + 1}" for index in range(len(documents)))

    async def search(self, query: str, *, limit: int) -> tuple[MemoryHit, ...]:
        return self.hits[:limit]

    async def forget_dataset(self) -> None:
        self.forgotten = True
        self.documents.clear()


class FailingBackend(FakeBackend):
    async def remember(self, document: str) -> str:
        raise RuntimeError("injected projection failure")


class MovingOriginTests(unittest.TestCase):
    def test_origin_moves_without_wall_clock_and_live_age_re_resolves(self) -> None:
        origin = MovingOriginIndex()
        first = origin.append(_ref(1), _ref(101), world_valid_from=20, world_valid_until=30)
        second = origin.append(_ref(2), _ref(102))

        self.assertEqual(first.ordinal, 0)
        self.assertEqual(second.ordinal, 1)
        self.assertEqual(origin.now, 1)
        self.assertEqual(origin.position(_ref(1)).age, 1)
        self.assertEqual(origin.position(_ref(2)).age, 0)
        self.assertEqual(origin.frozen_position(_ref(1)).age, 0)

        origin.append(_ref(3), _ref(103))
        self.assertEqual(origin.position(_ref(1)).age, 2)
        self.assertEqual(origin.frozen_position(_ref(1)).age, 0)

    def test_landmark_is_a_designation_event_and_frozen_view_stays_at_encoding(self) -> None:
        origin = MovingOriginIndex()
        origin.append(_ref(1), _ref(101))
        origin.append(_ref(2), _ref(102))
        landmark = origin.designate_landmark(
            "first_success",
            target_event_ref=_ref(1),
            designation_event_ref=_ref(3),
            projection_id=_ref(103),
        )
        origin.append(_ref(4), _ref(104))

        self.assertEqual(landmark.designated_at, 2)
        self.assertEqual(
            dict(origin.position(_ref(1)).landmark_relations)["first_success"],
            BEFORE,
        )
        self.assertEqual(
            dict(origin.position(_ref(4)).landmark_relations)["first_success"],
            AFTER,
        )
        self.assertEqual(origin.frozen_position(_ref(1)).landmark_relations, ())
        self.assertEqual(
            dict(origin.frozen_position(_ref(4)).landmark_relations)["first_success"],
            AFTER,
        )

    def test_maintained_index_matches_fair_naive_answer_with_less_work(self) -> None:
        origin = MovingOriginIndex()
        for index in range(1_000):
            origin.append(_ref(index + 1), _ref(index + 2_000))
        candidate = origin.recent(5)
        naive = FairNaiveTemporalView.from_index(origin).recent(5)

        self.assertEqual(candidate.event_refs, naive.event_refs)
        self.assertEqual(candidate.inspected, 5)
        self.assertEqual(naive.inspected, 1_000)

        maintained_position = origin.position(_ref(501))
        naive_position, inspected = FairNaiveTemporalView.from_index(origin).position(
            _ref(501)
        )
        self.assertEqual(maintained_position, naive_position)
        self.assertEqual(inspected, 1_000)

    def test_snapshot_round_trip_preserves_coordinates_and_rejects_bad_chain(self) -> None:
        origin = MovingOriginIndex()
        origin.append(_ref(1), _ref(101))
        origin.designate_landmark(
            "milestone",
            target_event_ref=_ref(1),
            designation_event_ref=_ref(2),
            projection_id=_ref(102),
        )
        origin.append(_ref(3), _ref(103))

        snapshot = origin.snapshot()
        restored = MovingOriginIndex.restore(snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.position(_ref(1)), origin.position(_ref(1)))

        snapshot["anchors"][1]["previous_event_ref"] = _ref(999)
        with self.assertRaisesRegex(ValueError, "predecessor chain"):
            MovingOriginIndex.restore(snapshot)


class SituatedMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_projection_round_trip_and_tamper_detection(self) -> None:
        projection = MemoryProjection.from_mapping(
            artifact_ref=_ref(1),
            text="A failed approach revealed a reusable ordering constraint.",
            source_ref=_ref(91),
            context={"domain": "software", "outcome": "failure"},
        )
        document = encode_projection(projection)
        decoded, declared = decode_projection(document)
        self.assertEqual(decoded, projection)
        self.assertEqual(declared, projection_id(projection))

        with self.assertRaisesRegex(ValueError, "identity"):
            decode_projection(document.replace("reusable", "different"))
        with self.assertRaisesRegex(ValueError, "fields"):
            decode_projection(document.replace('\n{', '\n{"extra":"x",', 1))

    async def test_cognee_candidates_are_validated_and_temporally_situated(self) -> None:
        backend = FakeBackend()
        memory = SituatedMemory(backend)
        first = MemoryProjection.from_mapping(
            artifact_ref=_ref(1),
            text="First observation about a shared procedure.",
            source_ref=_ref(91),
            context={"domain": "software"},
        )
        second = MemoryProjection.from_mapping(
            artifact_ref=_ref(2),
            text="Later evidence revises the procedure.",
            source_ref=_ref(92),
        )
        await memory.remember(first, world_valid_from=10, world_valid_until=20)
        await memory.remember(second, world_valid_from=21)

        tampered = backend.documents[0].replace("First", "False")
        unknown = encode_projection(
            MemoryProjection.from_mapping(
                artifact_ref=_ref(99), text="Unknown", source_ref=_ref(199)
            )
        )
        backend.hits = (
            MemoryHit(backend.documents[1], score=0.90, backend_ref="chunk-2"),
            MemoryHit(backend.documents[0], score=0.80, backend_ref="chunk-1"),
            MemoryHit(tampered, score=0.99),
            MemoryHit(unknown, score=0.70),
        )
        batch = await memory.recall("procedure", world_time=15)

        self.assertEqual([item.artifact_ref for item in batch.items], [_ref(2), _ref(1)])
        self.assertEqual([item.age for item in batch.items], [0, 1])
        self.assertEqual([item.backend_score for item in batch.items], [0.90, 0.80])
        self.assertFalse(batch.items[0].world_valid_at_query)
        self.assertTrue(batch.items[1].world_valid_at_query)
        self.assertEqual(batch.rejected, ("PROJECTION_INVALID", "PROJECTION_UNKNOWN"))

        frozen = await memory.recall("procedure", frozen_origin=True)
        self.assertEqual([item.age for item in frozen.items], [0, 0])

    async def test_visibility_is_revalidated_outside_cognee(self) -> None:
        backend = FakeBackend()
        memory = SituatedMemory(backend)
        sealed = MemoryProjection.from_mapping(
            artifact_ref=_ref(5),
            text="Sealed evaluator material",
            source_ref=_ref(95),
            visibility="SEALED_EVALUATION",
        )
        await memory.remember(sealed)
        backend.hits = (MemoryHit(backend.documents[0]),)

        denied = await memory.recall("material")
        self.assertEqual(denied.items, ())
        self.assertEqual(denied.rejected, ("PROJECTION_VISIBILITY_DENIED",))
        allowed = await memory.recall(
            "material", allowed_visibility=("SEALED_EVALUATION",)
        )
        self.assertEqual(allowed.items[0].artifact_ref, _ref(5))

    async def test_backend_failure_does_not_erase_canonical_temporal_event(self) -> None:
        memory = SituatedMemory(FailingBackend())
        projection = MemoryProjection.from_mapping(
            artifact_ref=_ref(7), text="Evidence survives index failure", source_ref=_ref(97)
        )
        with self.assertRaisesRegex(RuntimeError, "injected"):
            await memory.remember(projection)
        self.assertEqual(memory.origin.anchor(_ref(7)).ordinal, 0)

        recovery = FakeBackend()
        memory.backend = recovery
        self.assertEqual(await memory.reproject(projection), "backend-1")
        self.assertEqual(len(recovery.documents), 1)

    async def test_forget_deletes_projection_not_origin(self) -> None:
        backend = FakeBackend()
        memory = SituatedMemory(backend)
        projection = MemoryProjection.from_mapping(
            artifact_ref=_ref(8), text="Disposable projection", source_ref=_ref(98)
        )
        await memory.remember(projection)
        await memory.forget_projection()
        self.assertTrue(backend.forgotten)
        self.assertEqual(memory.origin.anchor(_ref(8)).ordinal, 0)

    async def test_bulk_projection_uses_one_backend_batch_and_preserves_order(self) -> None:
        backend = FakeBackend()
        memory = SituatedMemory(backend)
        projections = tuple(
            MemoryProjection.from_mapping(
                artifact_ref=_ref(index),
                text=f"Evidence {index}",
                source_ref=_ref(index + 100),
            )
            for index in range(20, 23)
        )
        ordinals = await memory.remember_many(projections)
        self.assertEqual(ordinals, (0, 1, 2))
        self.assertEqual(len(backend.documents), 3)
        self.assertEqual(
            [decode_projection(item)[0] for item in backend.documents],
            list(projections),
        )


@dataclass
class _FakeRememberResult:
    status: str = "completed"
    items: tuple[dict[str, str], ...] = ({"id": "data-1"},)


@dataclass
class _FakeSearchResult:
    text: str
    score: float
    metadata: dict[str, str]


class _FakeSearchType:
    CHUNKS = "CHUNKS"


class _FakeCogneeModule:
    SearchType = _FakeSearchType

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def remember(self, document: str, **kwargs: object) -> _FakeRememberResult:
        self.calls.append(("remember", {"document": document, **kwargs}))
        return _FakeRememberResult()

    async def search(self, **kwargs: object) -> tuple[_FakeSearchResult, ...]:
        self.calls.append(("search", kwargs))
        return (
            _FakeSearchResult(
                text="document", score=0.75, metadata={"chunk_id": "chunk-1"}
            ),
        )

    async def forget(self, **kwargs: object) -> dict[str, int]:
        self.calls.append(("forget", kwargs))
        return {"datasets_removed": 1}


class CogneeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_calls_fail_closed_without_explicit_configuration(self) -> None:
        backend = CogneeProjectionBackend("angler-test", cognee_module=_FakeCogneeModule())
        with self.assertRaises(CogneeConfigurationError):
            await backend.remember("document")
        with self.assertRaises(CogneeConfigurationError):
            await backend.search("query", limit=2)

    async def test_official_public_api_shape_is_used_without_answer_generation(self) -> None:
        module = _FakeCogneeModule()
        backend = CogneeProjectionBackend(
            "angler-test",
            cognee_module=module,
            local_models_configured=True,
            telemetry_authorized=True,
        )
        identifier = await backend.remember("document")
        hits = await backend.search("query", limit=4)
        await backend.forget_dataset()

        self.assertEqual(identifier, "data-1")
        self.assertEqual(hits, (MemoryHit("document", 0.75, "chunk-1"),))
        self.assertEqual(module.calls[0][0], "remember")
        self.assertEqual(module.calls[0][1]["dataset_name"], "angler-test")
        self.assertFalse(module.calls[0][1]["self_improvement"])
        self.assertEqual(module.calls[1][1]["query_type"], "CHUNKS")
        self.assertEqual(module.calls[1][1]["datasets"], ["angler-test"])
        self.assertEqual(module.calls[2], ("forget", {"dataset": "angler-test"}))

    async def test_cognee_1_5_chunk_dictionary_results_are_not_discarded(self) -> None:
        module = _FakeCogneeModule()

        async def search(**kwargs: object) -> list[dict[str, object]]:
            module.calls.append(("search", kwargs))
            return [
                {
                    "id": "chunk-live-1",
                    "document_id": "data-live-1",
                    "text": "live chunk document",
                    "score": 0.125,
                    "chunk_index": 0,
                }
            ]

        module.search = search  # type: ignore[method-assign]
        backend = CogneeProjectionBackend(
            "angler-test",
            cognee_module=module,
            local_models_configured=True,
            telemetry_authorized=True,
        )

        hits = await backend.search("query", limit=4)

        self.assertEqual(
            hits,
            (MemoryHit("live chunk document", 0.125, "chunk-live-1"),),
        )

    async def test_access_control_chunk_envelope_is_supported(self) -> None:
        module = _FakeCogneeModule()

        async def search(**kwargs: object) -> list[dict[str, object]]:
            module.calls.append(("search", kwargs))
            return [
                {
                    "dataset_name": "angler-test",
                    "search_result": [
                        {"text": "wrapped chunk", "document_id": "data-wrapped"}
                    ],
                }
            ]

        module.search = search  # type: ignore[method-assign]
        backend = CogneeProjectionBackend(
            "angler-test",
            cognee_module=module,
            local_models_configured=True,
            telemetry_authorized=True,
        )

        hits = await backend.search("query", limit=4)

        self.assertEqual(hits, (MemoryHit("wrapped chunk", None, "data-wrapped"),))


if __name__ == "__main__":
    unittest.main()
