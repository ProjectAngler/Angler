from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import unittest

from angler.cognition.contracts import (
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    CognitiveRelation,
    EpistemicStatus,
    ProspectiveCommitment,
    RelationType,
)
from angler.memory.cognitive_graph import (
    CognitiveGraphProjection,
    CognitiveReferenceHit,
    TypedSituatedMemory,
    episode_record,
)
from angler.memory.moving_origin import MovingOriginIndex


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _episode(
    number: int,
    *,
    previous: str | None = None,
    visibility: str = "LEARNER_VISIBLE",
) -> CognitiveEpisode:
    task_id = f"task-{number}"
    proposals = (
        "Inspect public evidence.",
        "Request a missing constraint.",
        "Apply the bounded procedure.",
        "Decline unsupported execution.",
    )
    return CognitiveEpisode(
        task_id=task_id,
        request=f"Handle bounded public request {number}.",
        recalled_refs=(
            _ref(f"recalled-{number}"),
            _ref(f"support-{number}"),
        ),
        proposals=proposals,
        selected_index=2,
        commitment=ProspectiveCommitment(
            parent_event_ref=previous,
            task_id=task_id,
            candidate_index=2,
            candidate_trace=proposals[2],
            predicted_score=0.75,
            uncertainty=0.25,
            horizon=1,
            competence_state_digest=_ref(f"state-{number - 1}"),
        ),
        response=f"Bounded response {number}.",
        observations=(f"Objective observation {number}.",),
        outcome="success" if number % 2 else "failure",
        feedback_text=f"Objective verifier result {number}.",
        feedback_source_ref=_ref(f"feedback-{number}"),
        parent_state_digest=_ref(f"state-{number - 1}"),
        child_state_digest=_ref(f"state-{number}"),
        model_ref=_ref("model"),
        encoder_ref=_ref("encoder"),
        supporting_evidence_refs=(_ref(f"support-{number}"),),
        visibility=visibility,
    )


@dataclass(frozen=True)
class _Item:
    sequence: int
    episode_ref: str
    episode: CognitiveEpisode


def _item(number: int, **kwargs: object) -> _Item:
    episode = _episode(number, **kwargs)
    return _Item(number, episode.episode_ref, episode)


class _Store:
    def __init__(self, items: tuple[_Item, ...]) -> None:
        self.items = items
        self.by_ref = {item.episode_ref: item for item in items}

    def get_episode_item(self, episode_ref: str) -> _Item:
        return self.by_ref[episode_ref]

    def episode_items(self, after_sequence: int = 0, limit: int = 128) -> tuple[_Item, ...]:
        return tuple(
            item for item in self.items if item.sequence > after_sequence
        )[:limit]


class _Backend:
    def __init__(self) -> None:
        self.projections: dict[str, CognitiveGraphProjection] = {}
        self.hits: tuple[object, ...] = ()
        self.fail_project = False
        self.forget_count = 0
        self.project_calls = 0

    async def project(self, projection: CognitiveGraphProjection) -> str:
        self.project_calls += 1
        if self.fail_project:
            raise RuntimeError("injected backend failure")
        self.projections[projection.record.record_ref] = projection
        return f"backend-{self.project_calls}"

    async def search(self, query: str, *, limit: int):
        return self.hits[:limit]

    async def forget_namespace(self) -> None:
        self.forget_count += 1
        self.projections.clear()


class CognitiveGraphContractTests(unittest.TestCase):
    def test_episode_record_and_projection_are_deterministic_and_strict(self) -> None:
        item = _item(1)
        first = episode_record(item)
        second = episode_record(item)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.record_ref, second.record_ref)
        self.assertEqual(first.acquired_ordinal, 0)
        self.assertIn(item.episode_ref, first.provenance_refs)
        self.assertIn(item.episode.feedback_source_ref, first.provenance_refs)
        self.assertEqual(first.kind, CognitiveMemoryKind.EPISODIC)
        self.assertEqual(first.epistemic_status, EpistemicStatus.OBSERVED)

        projection = CognitiveGraphProjection(item.episode_ref, first)
        restored = CognitiveGraphProjection.from_json(projection.canonical_bytes())
        self.assertEqual(restored, projection)
        self.assertEqual(restored.projection_ref, projection.projection_ref)
        noncanonical = json.dumps(projection.to_payload(), sort_keys=True)
        with self.assertRaises(ValueError):
            CognitiveGraphProjection.from_json(noncanonical)
        with self.assertRaises(ValueError):
            CognitiveGraphProjection(_ref("wrong-source"), first)

    def test_relation_types_and_dedicated_revision_links_round_trip(self) -> None:
        source = _ref("source")
        ordinary = tuple(
            sorted(
                (
                    CognitiveRelation(kind, _ref(kind.value))
                    for kind in RelationType
                    if kind not in {RelationType.SUPERSEDES, RelationType.TOMBSTONES}
                ),
                key=lambda value: (value.relation_type.value, value.target_ref),
            )
        )
        record = CognitiveMemoryRecord(
            kind=CognitiveMemoryKind.SEMANTIC,
            epistemic_status=EpistemicStatus.VALIDATED,
            content="Typed relation round-trip.",
            provenance_refs=(source,),
            visibility="LEARNER_VISIBLE",
            producer_id="angler.test",
            producer_checkpoint_ref=_ref("model"),
            competence_ref=_ref("competence"),
            acquired_ordinal=0,
            relations=ordinary,
            supersedes_refs=(_ref("superseded"),),
        )
        projection = CognitiveGraphProjection(source, record)
        restored = CognitiveGraphProjection.from_json(projection.canonical_bytes())
        self.assertEqual(restored.record.relations, ordinary)
        self.assertEqual(restored.record.supersedes_refs, record.supersedes_refs)

        retracted = CognitiveMemoryRecord(
            kind=CognitiveMemoryKind.SEMANTIC,
            epistemic_status=EpistemicStatus.RETRACTED,
            content="Canonical retraction link.",
            provenance_refs=(source,),
            visibility="LEARNER_VISIBLE",
            producer_id="angler.test",
            producer_checkpoint_ref=_ref("model"),
            competence_ref=_ref("competence"),
            acquired_ordinal=0,
            tombstone_refs=(_ref("tombstoned"),),
        )
        self.assertEqual(
            CognitiveGraphProjection.from_json(
                CognitiveGraphProjection(source, retracted).canonical_bytes()
            ).record.tombstone_refs,
            retracted.tombstone_refs,
        )


class TypedSituatedMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        first = _item(1)
        second = _item(2, previous=first.episode_ref)
        self.items = (first, second)
        self.store = _Store(self.items)
        self.backend = _Backend()
        self.memory = TypedSituatedMemory(source=self.store, backend=self.backend)

    async def test_one_anchor_retry_and_cycle_compatible_recall_surface(self) -> None:
        first = await self.memory.remember_episode_item(self.items[0])
        await self.memory.remember_episode_item(self.items[1])
        retry = await self.memory.remember_episode_item(self.items[0])
        self.assertEqual(first.record_ref, retry.record_ref)
        self.assertEqual(self.memory.origin.size, 2)

        self.backend.hits = (
            CognitiveReferenceHit(
                self.items[0].episode_ref,
                first.record_ref,
                score=0.5,
                backend_ref="backend-hit",
            ),
        )
        recalled = await self.memory.recall("bounded request", world_time=17)
        self.assertEqual(recalled.rejected, ())
        value = recalled.items[0]
        self.assertEqual(value.artifact_ref, first.record_ref)
        self.assertEqual(value.text, value.record.content)
        self.assertEqual(value.source_ref, self.items[0].episode_ref)
        self.assertEqual(value.visibility, "LEARNER_VISIBLE")
        self.assertEqual(value.acquired_ordinal, 0)
        self.assertEqual(value.age, 1)
        self.assertIsInstance(value.context, tuple)
        self.assertEqual(value.world_valid_at_query, True)

    async def test_constant_time_synchronization_requires_rebuild_then_accepts(self) -> None:
        self.memory.assert_synchronized(0, None)
        with self.assertRaisesRegex(ValueError, "rebuild_from_store"):
            self.memory.assert_synchronized(2, self.items[1].episode_ref)

        await self.memory.rebuild_from_store(page_size=1)
        self.memory.assert_synchronized(2, self.items[1].episode_ref)

        with self.assertRaisesRegex(ValueError, "rebuild_from_store"):
            self.memory.assert_synchronized(2, self.items[0].episode_ref)
        with self.assertRaisesRegex(ValueError, "rebuild_from_store"):
            self.memory.assert_synchronized(0, self.items[0].episode_ref)

    async def test_backend_failure_preserves_single_anchor_and_retry(self) -> None:
        self.backend.fail_project = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            await self.memory.remember_episode_item(self.items[0])
        self.assertEqual(self.memory.origin.size, 1)
        self.backend.fail_project = False
        receipt = await self.memory.reproject_episode(self.items[0].episode_ref)
        self.assertEqual(receipt.acquired_ordinal, 0)
        self.assertEqual(self.memory.origin.size, 1)

    async def test_reprojection_rejects_wrong_temporal_ordinal(self) -> None:
        receipt = await self.memory.remember_episode_item(self.items[0])
        wrong_origin = MovingOriginIndex()
        wrong_origin.append(_ref("unrelated-record"), _ref("unrelated-projection"))
        wrong_origin.append(receipt.record_ref, receipt.projection_ref)
        self.memory.origin = wrong_origin
        with self.assertRaisesRegex(ValueError, "existing anchor"):
            await self.memory.reproject_episode(self.items[0].episode_ref)

        self.backend.hits = (
            CognitiveReferenceHit(self.items[0].episode_ref, receipt.record_ref),
        )
        recalled = await self.memory.recall("query")
        self.assertEqual(recalled.rejected, ("REFERENCE_STALE_OR_TAMPERED",))

    async def test_unknown_mismatch_visibility_and_stale_hits_are_rejected(self) -> None:
        receipt = await self.memory.remember_episode_item(self.items[0])
        restricted_item = _item(1, visibility="HUMAN_AUTHORITY")
        restricted_memory = TypedSituatedMemory(
            source=_Store((restricted_item,)), backend=_Backend()
        )
        restricted_receipt = await restricted_memory.remember_episode_item(restricted_item)

        self.backend.hits = (
            CognitiveReferenceHit(_ref("unknown-episode"), receipt.record_ref),
            CognitiveReferenceHit(self.items[0].episode_ref, _ref("wrong-record")),
        )
        rejected = await self.memory.recall("query")
        self.assertEqual(
            rejected.rejected,
            ("REFERENCE_UNKNOWN_OR_INVALID", "REFERENCE_IDENTITY_MISMATCH"),
        )

        restricted_memory.backend.hits = (
            CognitiveReferenceHit(restricted_item.episode_ref, restricted_receipt.record_ref),
        )
        restricted = await restricted_memory.recall("query")
        self.assertEqual(restricted.rejected, ("REFERENCE_VISIBILITY_DENIED",))

        stale_origin = MovingOriginIndex()
        stale_origin.append(receipt.record_ref, _ref("stale-projection"))
        self.memory.origin = stale_origin
        self.backend.hits = (
            CognitiveReferenceHit(self.items[0].episode_ref, receipt.record_ref),
        )
        stale = await self.memory.recall("query")
        self.assertEqual(stale.rejected, ("REFERENCE_STALE_OR_TAMPERED",))

    async def test_backend_cannot_exceed_requested_recall_limit(self) -> None:
        async def oversized_search(query: str, *, limit: int):
            return (object(), object())

        self.backend.search = oversized_search
        with self.assertRaisesRegex(ValueError, "more hits than requested"):
            await self.memory.recall("query", limit=1)

    async def test_neighbor_is_revalidated_without_being_a_seed_hit(self) -> None:
        first = await self.memory.remember_episode_item(self.items[0])
        second = await self.memory.remember_episode_item(self.items[1])
        self.backend.hits = (
            CognitiveReferenceHit(
                self.items[0].episode_ref,
                first.record_ref,
                adjacent_record_refs=(second.record_ref, _ref("poison")),
            ),
        )
        recalled = await self.memory.recall("query")
        self.assertEqual(
            tuple(item.record_ref for item in recalled.items[0].adjacent_records),
            (second.record_ref,),
        )
        self.assertEqual(recalled.rejected, ("ADJACENT_REFERENCE_UNKNOWN",))

    async def test_rebuild_pages_canonical_items_and_recreates_exact_views(self) -> None:
        receipts = tuple(
            [await self.memory.remember_episode_item(item) for item in self.items]
        )
        expected_snapshot = self.memory.origin.snapshot()
        expected_refs = tuple(item.record_ref for item in receipts)
        self.backend.projections.clear()

        rebuilt = await self.memory.rebuild_from_store(page_size=1)
        self.assertEqual(rebuilt, expected_refs)
        self.assertEqual(self.memory.origin.snapshot(), expected_snapshot)
        self.assertEqual(tuple(self.backend.projections), expected_refs)
        self.assertEqual(self.backend.forget_count, 1)

    async def test_rebuild_rejects_noncontiguous_canonical_pages(self) -> None:
        bad = _Item(2, self.items[0].episode_ref, self.items[0].episode)
        memory = TypedSituatedMemory(source=_Store((bad,)), backend=_Backend())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            await memory.rebuild_from_store(page_size=1)


if __name__ == "__main__":
    unittest.main()
