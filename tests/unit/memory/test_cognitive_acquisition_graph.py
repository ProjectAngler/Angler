from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import unittest

from angler.cognition.contracts import (
    CognitiveMemoryKind,
    EpistemicStatus,
    CognitiveRelation,
    RelationType,
)
from angler.cognition.prospective_origin import (
    ProspectiveResolution,
    ResolutionDisposition,
)
from angler.memory.cognitive_acquisition import (
    CognitiveAcquisition,
    CognitiveGraphProjectionV2,
)
from angler.memory.cognitive_acquisition_graph import (
    MAX_ACQUISITION_NEIGHBORS,
    MAX_ACQUISITION_PROVENANCE_REFS,
    AcquisitionReferenceHit,
    AcquisitionSituatedMemory,
    prospective_batch_record,
    prospective_resolution_record,
)
from angler.memory.moving_origin import MovingOriginIndex
from tests.unit.cognition.test_prospective_origin import (
    batch as _batch,
    observed_resolution as _observed_resolution,
    wrapper as _wrapper,
)


def _ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _cancelled_resolution() -> ProspectiveResolution:
    return ProspectiveResolution.lifecycle(
        disposition=ResolutionDisposition.CANCELLED,
        reservation=_wrapper(),
    )


@dataclass(frozen=True, slots=True)
class _Item:
    ordinal: int
    acquisition_ref: str
    record_ref: str
    acquisition: CognitiveAcquisition


@dataclass(frozen=True, slots=True)
class _Outbox:
    ordinal: int
    acquisition_ref: str
    record_ref: str
    projection_ref: str
    projection: CognitiveGraphProjectionV2


@dataclass(frozen=True, slots=True)
class _Head:
    next_ordinal: int
    acquisition_ref: str | None
    record_ref: str | None


def _chain() -> tuple[_Item, _Item]:
    batch = _batch()
    batch_record = prospective_batch_record(batch, 0)
    batch_acquisition = CognitiveAcquisition.from_source(
        batch,
        ordinal=0,
        predecessor_acquisition_ref=None,
        record=batch_record,
    )
    resolution = _observed_resolution()[0]
    resolution_record = prospective_resolution_record(resolution, 1)
    resolution_acquisition = CognitiveAcquisition.from_source(
        resolution,
        ordinal=1,
        predecessor_acquisition_ref=batch_acquisition.acquisition_ref,
        record=resolution_record,
    )
    return (
        _Item(
            0,
            batch_acquisition.acquisition_ref,
            batch_record.record_ref,
            batch_acquisition,
        ),
        _Item(
            1,
            resolution_acquisition.acquisition_ref,
            resolution_record.record_ref,
            resolution_acquisition,
        ),
    )


def _observed_lifecycle_chain() -> tuple[_Item, _Item]:
    observed = _observed_resolution()[0]
    observed_record = prospective_resolution_record(observed, 0)
    observed_acquisition = CognitiveAcquisition.from_source(
        observed,
        ordinal=0,
        predecessor_acquisition_ref=None,
        record=observed_record,
    )
    cancelled = _cancelled_resolution()
    cancelled_record = prospective_resolution_record(cancelled, 1)
    cancelled_acquisition = CognitiveAcquisition.from_source(
        cancelled,
        ordinal=1,
        predecessor_acquisition_ref=observed_acquisition.acquisition_ref,
        record=cancelled_record,
    )
    return (
        _Item(
            0,
            observed_acquisition.acquisition_ref,
            observed_record.record_ref,
            observed_acquisition,
        ),
        _Item(
            1,
            cancelled_acquisition.acquisition_ref,
            cancelled_record.record_ref,
            cancelled_acquisition,
        ),
    )


def _outbox(item: _Item) -> _Outbox:
    projection = CognitiveGraphProjectionV2.from_acquisition(item.acquisition)
    return _Outbox(
        item.ordinal,
        item.acquisition_ref,
        item.record_ref,
        projection.projection_ref,
        projection,
    )


class _Store:
    def __init__(self, items: tuple[_Item, ...]) -> None:
        self.items = items
        self.by_record = {item.record_ref: item for item in items}
        self.pending = [_outbox(item) for item in items]
        self.pending_limits: list[int] = []
        self.page_limits: list[tuple[int, int]] = []
        self.acknowledged: list[str] = []
        self.fail_ack = False

    def acquisition_head(self) -> _Head:
        if not self.items:
            return _Head(0, None, None)
        item = self.items[-1]
        return _Head(item.ordinal + 1, item.acquisition_ref, item.record_ref)

    def get_acquisition_item(self, record_ref: str) -> _Item:
        return self.by_record[record_ref]

    def acquisition_items(
        self, after_ordinal: int = -1, limit: int = 64
    ) -> tuple[_Item, ...]:
        self.page_limits.append((after_ordinal, limit))
        return tuple(
            item for item in self.items if item.ordinal > after_ordinal
        )[:limit]

    def pending_acquisition_projections(
        self, limit: int = 64
    ) -> tuple[_Outbox, ...]:
        self.pending_limits.append(limit)
        return tuple(self.pending[:limit])

    def ack_acquisition_projection(self, projection_ref: str) -> None:
        if self.fail_ack:
            raise RuntimeError("injected acknowledgement failure")
        matches = [
            item for item in self.pending if item.projection_ref == projection_ref
        ]
        if not matches:
            if projection_ref in self.acknowledged:
                return
            raise KeyError(projection_ref)
        self.pending.remove(matches[0])
        self.acknowledged.append(projection_ref)


class _Backend:
    def __init__(self) -> None:
        self.projections: dict[str, CognitiveGraphProjectionV2] = {}
        self.project_order: list[str] = []
        self.hits: tuple[object, ...] = ()
        self.queries: list[tuple[str, int]] = []
        self.fail_project = False
        self.fail_record_ref: str | None = None
        self.fail_forget = False
        self.forget_count = 0

    async def project(self, projection: CognitiveGraphProjectionV2) -> str:
        self.project_order.append(projection.record.record_ref)
        if self.fail_project or projection.record.record_ref == self.fail_record_ref:
            raise RuntimeError("injected backend failure")
        self.projections[projection.record.record_ref] = projection
        return f"backend-{projection.record.record_ref[-8:]}"

    async def search(self, query: str, *, limit: int):
        self.queries.append((query, limit))
        return self.hits[:limit]

    async def forget_namespace(self) -> None:
        self.forget_count += 1
        if self.fail_forget:
            raise RuntimeError("injected forget failure")
        self.projections.clear()


class ProspectiveAcquisitionRecordTests(unittest.TestCase):
    def test_batch_record_has_one_frozen_hypothetical_rendering(self) -> None:
        batch = _batch(count=7)
        record = prospective_batch_record(batch, 9)
        self.assertEqual(
            record.content,
            "Prospective dynamics batch\n"
            "Hypothetical status: PROPOSED (not observed, validated, or authorized).\n"
            "Request:\n"
            "Solve the bounded public task.\n"
            "Branch count: 7\n"
            "Selected proposed trace:\n"
            "proposal 3",
        )
        self.assertEqual(record.kind, CognitiveMemoryKind.COUNTERFACTUAL)
        self.assertEqual(record.epistemic_status, EpistemicStatus.PROPOSED)
        self.assertEqual(record.visibility, "LEARNER_VISIBLE")
        self.assertEqual(record.producer_id, "angler.prospective-cycle")
        self.assertEqual(record.acquired_ordinal, 9)
        self.assertIsNone(record.confidence_ppm)
        self.assertIsNone(record.world_valid_from)
        self.assertIn(batch.batch_ref, record.provenance_refs)
        self.assertIn(batch.parent_lineage.lineage_ref, record.provenance_refs)
        self.assertEqual(
            tuple(
                relation.target_ref
                for relation in record.relations
                if relation.relation_type is RelationType.PREDICTS
            ),
            tuple(sorted(branch.branch_ref for branch in batch.branches)),
        )
        self.assertEqual(
            {relation.relation_type for relation in record.relations},
            {RelationType.ABOUT, RelationType.PREDICTS},
        )
        self.assertEqual(
            prospective_batch_record(batch, 9).canonical_bytes(),
            record.canonical_bytes(),
        )

    def test_resolution_rendering_exposes_outcome_only_when_observed(self) -> None:
        observed = _observed_resolution()[0]
        record = prospective_resolution_record(observed, 4)
        self.assertEqual(
            record.content,
            "Prospective lifecycle resolution\n"
            "Disposition: OBSERVED\n"
            "Selected trace:\n"
            "proposal 3\n"
            "Executor status: COMPLETED\n"
            "Objective outcome: success\n"
            "Objective feedback:\n"
            "The external objective check passed.\n"
            "Child competence update: PRESENT",
        )
        self.assertEqual(record.kind, CognitiveMemoryKind.EPISODIC)
        self.assertEqual(record.epistemic_status, EpistemicStatus.OBSERVED)
        self.assertEqual(
            record.competence_ref,
            observed.child_lineage.competence_state_digest,
        )
        self.assertIn(observed.resolution_ref, record.provenance_refs)
        self.assertIn(
            observed.objective_feedback.feedback_ref, record.provenance_refs
        )

        cancelled = _cancelled_resolution()
        cancelled_record = prospective_resolution_record(cancelled, 5)
        self.assertEqual(
            cancelled_record.content,
            "Prospective lifecycle resolution\n"
            "Disposition: CANCELLED\n"
            "Selected trace:\n"
            "proposal 3\n"
            "Executor status: ABSENT\n"
            "Objective outcome: ABSENT\n"
            "Child competence update: ABSENT",
        )
        self.assertNotIn("feedback", cancelled_record.content.lower())
        self.assertEqual(
            cancelled_record.competence_ref,
            cancelled.parent_lineage.competence_state_digest,
        )

    def test_renderers_reject_wrong_type_and_invalid_ordinal(self) -> None:
        with self.assertRaises(TypeError):
            prospective_batch_record(object(), 0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            prospective_resolution_record(object(), 0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            prospective_batch_record(_batch(), -1)
        with self.assertRaises(ValueError):
            prospective_resolution_record(_cancelled_resolution(), True)
        with self.assertRaisesRegex(ValueError, "64-branch"):
            prospective_batch_record(_batch(count=65, branch_capacity=65), 0)

    def test_backend_hit_adjacency_has_a_local_resource_ceiling(self) -> None:
        refs = tuple(
            f"sha256:{index:064x}"
            for index in range(1, MAX_ACQUISITION_NEIGHBORS + 2)
        )
        accepted = AcquisitionReferenceHit(
            _ref("seed"),
            adjacent_record_refs=refs[:MAX_ACQUISITION_NEIGHBORS],
        )
        self.assertEqual(
            len(accepted.adjacent_record_refs), MAX_ACQUISITION_NEIGHBORS
        )
        with self.assertRaisesRegex(ValueError, "local ceiling"):
            AcquisitionReferenceHit(
                _ref("seed"),
                adjacent_record_refs=refs,
            )


class AcquisitionSituatedMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.items = _chain()
        self.store = _Store(self.items)
        self.backend = _Backend()
        self.memory = AcquisitionSituatedMemory(
            source=self.store,
            backend=self.backend,
        )

    async def test_retry_forwards_exact_limit_oldest_first_and_acknowledges(self) -> None:
        first = await self.memory.retry_pending_projections(limit=1)
        second = await self.memory.retry_pending_projections(limit=1)
        self.assertEqual(first, (self.items[0].record_ref,))
        self.assertEqual(second, (self.items[1].record_ref,))
        self.assertEqual(self.store.pending_limits, [1, 1])
        self.assertEqual(
            self.backend.project_order,
            [self.items[0].record_ref, self.items[1].record_ref],
        )
        self.assertEqual(len(self.store.acknowledged), 2)
        self.memory.assert_synchronized()
        for invalid in (0, 257, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    await self.memory.retry_pending_projections(invalid)  # type: ignore[arg-type]

    async def test_projection_and_ack_failures_leave_retryable_single_anchor(self) -> None:
        self.backend.fail_project = True
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            await self.memory.retry_pending_projections(limit=1)
        self.assertEqual(self.memory.origin.size, 1)
        self.assertEqual(self.store.acknowledged, [])
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.memory.assert_synchronized()

        self.backend.fail_project = False
        self.store.fail_ack = True
        with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
            await self.memory.retry_pending_projections(limit=1)
        self.assertEqual(self.memory.origin.size, 1)
        self.store.fail_ack = False
        retried = await self.memory.retry_pending_projections(limit=1)
        self.assertEqual(retried, (self.items[0].record_ref,))
        self.assertEqual(self.memory.origin.size, 1)

    async def test_ack_failure_restart_rebuilds_then_retries_exactly(self) -> None:
        self.store.fail_ack = True
        with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
            await self.memory.retry_pending_projections(limit=1)
        self.assertEqual(len(self.backend.projections), 1)

        self.store.fail_ack = False
        restarted = AcquisitionSituatedMemory(
            source=self.store,
            backend=self.backend,
        )
        rebuilt = await restarted.rebuild(limit=1)
        self.assertEqual(rebuilt, tuple(item.record_ref for item in self.items))
        restarted.assert_synchronized()
        retried = await restarted.retry_pending_projections(limit=1)
        self.assertEqual(retried, (self.items[0].record_ref,))
        self.assertEqual(restarted.origin.size, 2)
        self.assertEqual(len(self.backend.projections), 2)

    async def test_rebuild_pages_canonical_stream_without_acknowledging(self) -> None:
        rebuilt = await self.memory.rebuild_from_store(page_size=1)
        self.assertEqual(rebuilt, tuple(item.record_ref for item in self.items))
        self.assertEqual(self.backend.forget_count, 1)
        self.assertEqual(self.store.acknowledged, [])
        self.assertEqual(self.store.pending, [_outbox(item) for item in self.items])
        self.assertEqual(self.store.page_limits, [(-1, 1), (0, 1), (1, 1)])
        self.memory.assert_synchronized()

    async def test_recall_rejoins_canonical_bytes_and_denies_proposed(self) -> None:
        await self.memory.rebuild()
        self.backend.hits = (
            AcquisitionReferenceHit(self.items[0].record_ref, score=0.9),
            AcquisitionReferenceHit(
                self.items[1].record_ref,
                score=0.8,
                backend_ref="backend-result",
            ),
        )
        recalled = await self.memory.recall("bounded public query", limit=12)
        self.assertEqual(self.backend.queries, [("bounded public query", 12)])
        self.assertEqual(
            recalled.rejected, ("REFERENCE_EPISTEMIC_STATUS_DENIED",)
        )
        self.assertEqual(len(recalled.items), 1)
        item = recalled.items[0]
        self.assertEqual(item.artifact_ref, self.items[1].record_ref)
        self.assertEqual(item.source_ref, self.items[1].acquisition.source_ref)
        self.assertEqual(item.epistemic_status, "OBSERVED")
        self.assertEqual(item.acquired_ordinal, 1)
        self.assertEqual(item.age, 0)
        self.assertEqual(item.context[0], ("epistemic_status", "OBSERVED"))

    async def test_recall_requires_synchronization_and_frozen_origin_is_fair(self) -> None:
        self.backend.hits = (AcquisitionReferenceHit(self.items[1].record_ref),)
        with self.assertRaisesRegex(ValueError, "rebuild"):
            await self.memory.recall("query")
        await self.memory.rebuild()
        self.backend.hits = (AcquisitionReferenceHit(self.items[1].record_ref),)
        live = await self.memory.recall("query")
        frozen = await self.memory.recall("query", frozen_origin=True)
        self.assertEqual(live.items[0].record_ref, frozen.items[0].record_ref)
        self.assertEqual(live.items[0].text, frozen.items[0].text)
        self.assertEqual(live.items[0].age, 0)
        self.assertEqual(frozen.items[0].age, 0)

        # Replaying a fixed, already-returned hit manifest uses the same
        # canonical status/visibility checks without consulting the backend.
        replay = self.memory.recall_from_hits(self.backend.hits)
        self.assertEqual(replay.items[0].record_ref, live.items[0].record_ref)

        older_items = _observed_lifecycle_chain()
        older_store = _Store(older_items)
        older_backend = _Backend()
        older_memory = AcquisitionSituatedMemory(
            source=older_store,
            backend=older_backend,
        )
        await older_memory.rebuild()
        fixed_manifest = (AcquisitionReferenceHit(older_items[0].record_ref),)
        older_backend.hits = fixed_manifest
        live_older = await older_memory.recall("same query")
        frozen_older = await older_memory.recall(
            "same query", frozen_origin=True
        )
        self.assertEqual(
            tuple(item.record_ref for item in live_older.items),
            tuple(item.record_ref for item in frozen_older.items),
        )
        self.assertEqual(
            tuple(item.text for item in live_older.items),
            tuple(item.text for item in frozen_older.items),
        )
        self.assertEqual(live_older.items[0].age, 1)
        self.assertEqual(frozen_older.items[0].age, 0)

    async def test_hit_and_neighbor_tamper_fail_closed(self) -> None:
        await self.memory.rebuild()
        self.backend.hits = (
            AcquisitionReferenceHit(_ref("unknown")),
            AcquisitionReferenceHit(
                self.items[1].record_ref,
                adjacent_record_refs=(self.items[0].record_ref,),
            ),
        )
        recalled = await self.memory.recall("query")
        self.assertEqual(
            recalled.rejected,
            (
                "REFERENCE_UNKNOWN_OR_INVALID",
                "ADJACENT_REFERENCE_NOT_CANONICAL",
            ),
        )
        self.assertEqual(len(recalled.items), 1)
        self.assertEqual(recalled.items[0].adjacent_records, ())

    async def test_external_generic_relation_targets_are_not_record_rejections(
        self,
    ) -> None:
        await self.memory.rebuild()
        record = self.items[1].acquisition.record
        external_refs = tuple(
            sorted({relation.target_ref for relation in record.relations})
        )
        self.assertTrue(external_refs)
        self.backend.hits = (
            AcquisitionReferenceHit(
                record.record_ref,
                adjacent_record_refs=external_refs,
            ),
        )

        recalled = await self.memory.recall("query")

        self.assertEqual(recalled.rejected, ())
        self.assertEqual(
            tuple(item.record_ref for item in recalled.items),
            (record.record_ref,),
        )
        self.assertEqual(recalled.items[0].adjacent_records, ())
        self.assertEqual(recalled.items[0].record.relations, record.relations)

    async def test_mixed_record_and_external_relations_join_only_the_record(
        self,
    ) -> None:
        first_source = _observed_resolution()[0]
        first_record = prospective_resolution_record(first_source, 0)
        first_acquisition = CognitiveAcquisition.from_source(
            first_source,
            ordinal=0,
            predecessor_acquisition_ref=None,
            record=first_record,
        )
        second_source = _cancelled_resolution()
        second_base = prospective_resolution_record(second_source, 1)
        external_ref = second_base.relations[0].target_ref
        second_record = replace(
            second_base,
            relations=tuple(
                sorted(
                    (
                        *second_base.relations,
                        CognitiveRelation(
                            RelationType.DERIVED_FROM,
                            first_record.record_ref,
                        ),
                    ),
                    key=lambda relation: (
                        relation.relation_type.value,
                        relation.target_ref,
                    ),
                )
            ),
        )
        second_acquisition = CognitiveAcquisition.from_source(
            second_source,
            ordinal=1,
            predecessor_acquisition_ref=first_acquisition.acquisition_ref,
            record=second_record,
        )
        items = (
            _Item(
                0,
                first_acquisition.acquisition_ref,
                first_record.record_ref,
                first_acquisition,
            ),
            _Item(
                1,
                second_acquisition.acquisition_ref,
                second_record.record_ref,
                second_acquisition,
            ),
        )
        backend = _Backend()
        memory = AcquisitionSituatedMemory(source=_Store(items), backend=backend)
        await memory.rebuild()
        backend.hits = (
            AcquisitionReferenceHit(
                second_record.record_ref,
                adjacent_record_refs=tuple(
                    sorted((first_record.record_ref, external_ref))
                ),
            ),
        )

        recalled = await memory.recall("query")

        self.assertEqual(recalled.rejected, ())
        self.assertEqual(len(recalled.items), 1)
        self.assertEqual(
            tuple(
                record.record_ref
                for record in recalled.items[0].adjacent_records
            ),
            (first_record.record_ref,),
        )

    async def test_missing_dedicated_revision_target_remains_rejected(self) -> None:
        first = self.items[0]
        source = _observed_resolution()[0]
        missing_ref = _ref("missing-superseded-record")
        record = replace(
            prospective_resolution_record(source, 1),
            supersedes_refs=(missing_ref,),
        )
        acquisition = CognitiveAcquisition.from_source(
            source,
            ordinal=1,
            predecessor_acquisition_ref=first.acquisition_ref,
            record=record,
        )
        item = _Item(
            1,
            acquisition.acquisition_ref,
            record.record_ref,
            acquisition,
        )
        backend = _Backend()
        memory = AcquisitionSituatedMemory(
            source=_Store((first, item)),
            backend=backend,
        )
        await memory.rebuild()
        backend.hits = (
            AcquisitionReferenceHit(
                record.record_ref,
                adjacent_record_refs=(missing_ref,),
            ),
        )

        recalled = await memory.recall("query")

        self.assertEqual(recalled.rejected, ("ADJACENT_UNKNOWN_OR_INVALID",))
        self.assertEqual(len(recalled.items), 1)
        self.assertEqual(recalled.items[0].adjacent_records, ())

    async def test_head_advance_during_backend_search_blocks_stale_recall(self) -> None:
        await self.memory.rebuild()
        batch = _batch()
        record = prospective_batch_record(batch, 2)
        acquisition = CognitiveAcquisition.from_source(
            batch,
            ordinal=2,
            predecessor_acquisition_ref=self.items[1].acquisition_ref,
            record=record,
        )
        third = _Item(
            2,
            acquisition.acquisition_ref,
            record.record_ref,
            acquisition,
        )

        async def advancing_search(query: str, *, limit: int):
            self.store.items = (*self.store.items, third)
            self.store.by_record[third.record_ref] = third
            self.store.pending.append(_outbox(third))
            return (AcquisitionReferenceHit(self.items[1].record_ref),)

        self.backend.search = advancing_search
        with self.assertRaisesRegex(ValueError, "rebuild"):
            await self.memory.recall("query")

    async def test_outbox_projection_and_origin_tamper_fail_closed(self) -> None:
        outbox = _outbox(self.items[0])
        with self.assertRaisesRegex(ValueError, "projection item"):
            await self.memory.project_pending_item(
                replace(outbox, projection_ref=_ref("wrong-projection"))
            )

        await self.memory.rebuild()
        wrong_origin = MovingOriginIndex()
        wrong_origin.append(
            self.items[0].record_ref,
            _ref("wrong-projection"),
        )
        wrong_origin.append(
            self.items[1].record_ref,
            CognitiveGraphProjectionV2.from_acquisition(
                self.items[1].acquisition
            ).projection_ref,
        )
        self.memory.origin = wrong_origin
        self.backend.hits = (AcquisitionReferenceHit(self.items[0].record_ref),)
        recalled = await self.memory.recall("query")
        self.assertEqual(
            recalled.rejected,
            ("REFERENCE_EPISTEMIC_STATUS_DENIED",),
        )

        # The O(1) synchronization proof covers the canonical tail.  Once a
        # normally admissible interior record is requested, its own anchor is
        # rejoined before exposure.  Directly exercise that trust edge here.
        result = self.memory._resolve_record(
            self.items[0].record_ref,
            allowed_visibility=frozenset(("LEARNER_VISIBLE",)),
            admitted_epistemic_statuses=frozenset((EpistemicStatus.PROPOSED,)),
        )
        self.assertEqual(result, "REFERENCE_STALE_OR_TAMPERED")

    async def test_rebuild_rejects_noncontiguous_page(self) -> None:
        bad_store = _Store((self.items[1],))
        memory = AcquisitionSituatedMemory(source=bad_store, backend=_Backend())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            await memory.rebuild_from_store(page_size=1)

    async def test_rebuild_rejects_canonical_but_wrong_predecessor(self) -> None:
        second = self.items[1]
        wrong_acquisition = replace(
            second.acquisition,
            predecessor_acquisition_ref=_ref("wrong-predecessor"),
        )
        wrong_second = _Item(
            ordinal=second.ordinal,
            acquisition_ref=wrong_acquisition.acquisition_ref,
            record_ref=second.record_ref,
            acquisition=wrong_acquisition,
        )
        memory = AcquisitionSituatedMemory(
            source=_Store((self.items[0], wrong_second)),
            backend=_Backend(),
        )
        with self.assertRaisesRegex(ValueError, "predecessor"):
            await memory.rebuild_from_store(page_size=2)

    async def test_overdegree_record_never_reaches_alternate_backend(self) -> None:
        batch = _batch()
        base = prospective_batch_record(batch, 0)
        refs = tuple(
            f"sha256:{index:064x}"
            for index in range(1, MAX_ACQUISITION_NEIGHBORS + 2)
        )
        record = replace(
            base,
            relations=tuple(
                CognitiveRelation(RelationType.ABOUT, target_ref)
                for target_ref in refs
            ),
        )
        acquisition = CognitiveAcquisition.from_source(
            batch,
            ordinal=0,
            predecessor_acquisition_ref=None,
            record=record,
        )
        item = _Item(
            0,
            acquisition.acquisition_ref,
            record.record_ref,
            acquisition,
        )
        backend = _Backend()
        memory = AcquisitionSituatedMemory(
            source=_Store((item,)),
            backend=backend,
        )
        with self.assertRaisesRegex(ValueError, "neighbor local ceiling"):
            await memory.rebuild()
        self.assertEqual(backend.project_order, [])

    async def test_overprovenance_record_never_reaches_alternate_backend(self) -> None:
        batch = _batch()
        base = prospective_batch_record(batch, 0)
        extra_refs = {
            f"sha256:{index:064x}"
            for index in range(1, MAX_ACQUISITION_PROVENANCE_REFS + 1)
        }
        record = replace(
            base,
            provenance_refs=tuple(sorted({batch.batch_ref, *extra_refs})),
        )
        self.assertEqual(
            len(record.provenance_refs),
            MAX_ACQUISITION_PROVENANCE_REFS + 1,
        )
        acquisition = CognitiveAcquisition.from_source(
            batch,
            ordinal=0,
            predecessor_acquisition_ref=None,
            record=record,
        )
        item = _Item(
            0,
            acquisition.acquisition_ref,
            record.record_ref,
            acquisition,
        )
        backend = _Backend()
        memory = AcquisitionSituatedMemory(
            source=_Store((item,)),
            backend=backend,
        )
        with self.assertRaisesRegex(ValueError, "provenance-reference"):
            await memory.rebuild()
        self.assertEqual(backend.project_order, [])

    async def test_failed_destructive_rebuild_invalidates_old_live_view(self) -> None:
        await self.memory.rebuild()
        self.memory.assert_synchronized()

        self.backend.fail_forget = True
        with self.assertRaisesRegex(RuntimeError, "forget failure"):
            await self.memory.rebuild()
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.memory.assert_synchronized()

        self.backend.fail_forget = False
        await self.memory.rebuild()
        self.backend.fail_record_ref = self.items[1].record_ref
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            await self.memory.rebuild(limit=1)
        self.assertEqual(
            tuple(self.backend.projections),
            (self.items[0].record_ref,),
        )
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.memory.assert_synchronized()
        self.backend.hits = (AcquisitionReferenceHit(self.items[0].record_ref),)
        with self.assertRaisesRegex(ValueError, "rebuild"):
            await self.memory.recall("query")

    async def test_empty_store_forget_failure_invalidates_generation(self) -> None:
        store = _Store(())
        backend = _Backend()
        memory = AcquisitionSituatedMemory(source=store, backend=backend)
        memory.assert_synchronized()
        backend.fail_forget = True
        with self.assertRaisesRegex(RuntimeError, "forget failure"):
            await memory.rebuild()
        with self.assertRaisesRegex(ValueError, "generation is invalid"):
            memory.assert_synchronized()


if __name__ == "__main__":
    unittest.main()
