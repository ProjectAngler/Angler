"""Opt-in, networkless live qualification for acquisition-scoped Cognee.

Run only with ``ANGLER_RUN_LIVE_COGNEE_ACQUISITION=1``.  Cognee stays in the
isolated worker; this test module and the Angler process never import it.

The current acquisition source-binding table has exact sources for PROPOSED
and OBSERVED records, but none for VALIDATED records.  This qualification
therefore proves PROPOSED denial and OBSERVED admission without fabricating a
VALIDATED acquisition.  The worker also has no public live fault-injection
surface; bounded transport/backend failures remain fake-backed unit claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import sys
import unittest
from unittest.mock import patch
from uuid import NAMESPACE_OID, uuid5

from angler.cognition.contracts import (
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    CognitiveRelation,
    EpistemicStatus,
    ProspectiveCommitment,
    RelationType,
)
from angler.cognition.prospective_origin import (
    Perspective,
    ProspectiveBranch,
    ProspectiveDynamicsBatch,
    ProspectiveResourceEnvelope,
    RealityMode,
    SituatedContext,
    SituatedStateLineage,
)
from angler.memory.cognee_acquisition_adapter import CogneeAcquisitionAdapter
from angler.memory.cognee_subprocess_bindings import CogneeSubprocessBindings
from angler.memory.cognee_worker_protocol import (
    DATASET_NAME,
    FASTEMBED_EXECUTION_PROVIDERS,
    FASTEMBED_SESSION_THREADS,
    NODE_SET_ID,
    NODE_SET_NAME,
    SCORE_SEMANTICS,
    modern_dataset_id,
)
from angler.memory.cognitive_acquisition import (
    CognitiveAcquisition,
    CognitiveGraphProjectionV2,
)
from angler.memory.cognitive_acquisition_graph import (
    AcquisitionReferenceHit,
    AcquisitionSituatedMemory,
)


_LIVE_FLAG = "ANGLER_RUN_LIVE_COGNEE_ACQUISITION"
_QUERY = "angler live qualification cobalt river"
_TIMEOUT_SECONDS = 120.0


def _ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _prospective_batch() -> ProspectiveDynamicsBatch:
    simulated = SituatedContext(
        reality_mode=RealityMode.SIMULATED,
        perspective=Perspective.AGENT_SITUATED,
        subject_ref=_ref("live-subject"),
        scope_ref=_ref("live-scope"),
        world_ref=_ref("live-world"),
        evidence_refs=tuple(
            sorted((_ref("live-context-a"), _ref("live-context-b")))
        ),
    )
    parent = SituatedStateLineage(
        context=simulated,
        agent_ref=_ref("live-agent"),
        parent_lineage_ref=None,
        state_step=0,
        checkpoint_ref=_ref("live-checkpoint"),
        config_ref=_ref("live-config"),
        competence_state_digest=_ref("live-competence-0"),
        snapshot_digest=_ref("live-snapshot-0"),
        world_state_digest=_ref("live-world-state-0"),
        self_state_digest=_ref("live-self-state-0"),
        focus_state_digest=_ref("live-focus-state-0"),
        outcome_state_digest=_ref("live-outcome-state-0"),
        state_evidence_refs=(_ref("live-state-evidence"),),
    )
    context_refs = tuple(
        sorted((_ref("live-world-row-a"), _ref("live-world-row-b")))
    )
    hypothetical = SituatedContext(
        reality_mode=RealityMode.HYPOTHETICAL,
        perspective=simulated.perspective,
        subject_ref=simulated.subject_ref,
        scope_ref=simulated.scope_ref,
        world_ref=simulated.world_ref,
        evidence_refs=context_refs,
    )
    request = f"Consider the synthetic {_QUERY} request."
    request_ref = CognitiveEpisode.proposal_ref(request)
    support = (_ref("live-recalled-support"),)
    branches = tuple(
        ProspectiveBranch(
            parent_lineage_ref=parent.lineage_ref,
            task_id="cognee-live-proposed",
            request_ref=request_ref,
            candidate_index=index,
            candidate_trace=f"synthetic candidate {index}",
            context=hypothetical,
            predicted_next_latent=(float(index), float(index + 1)),
            outcome_logit=float(index) - 0.5,
            uncertainty=0.25 + float(index) / 10.0,
            selection_residual=float(index) / 10.0,
            decision_logit=float(index),
            horizon=1,
            recurrent_steps=1,
            focused_context_indices=(0, 1),
            focus_weights=(0.5, 0.5),
            supporting_evidence_refs=support,
        )
        for index in range(2)
    )
    return ProspectiveDynamicsBatch(
        parent_lineage=parent,
        parent_sequence=0,
        parent_event_ref=None,
        parent_acquisition_ref=None,
        task_id="cognee-live-proposed",
        request=request,
        model_ref=_ref("live-model"),
        encoder_ref=_ref("live-encoder"),
        dynamics_checkpoint_ref=parent.checkpoint_ref,
        dynamics_config_ref=parent.config_ref,
        decision_evidence_ref=_ref("live-decision-evidence"),
        recalled_refs=support,
        context_refs=context_refs,
        eligibility_rows=((0, 1), (0, 1)),
        resources=ProspectiveResourceEnvelope(
            branch_capacity=2,
            read_capacity=2,
            latent_width=2,
            recurrent_step_capacity=1,
            horizon_capacity=1,
        ),
        branches=branches,
        selected_branch_ref=branches[0].branch_ref,
    )


def _episode(label: str) -> CognitiveEpisode:
    parent_ref = _ref(f"live-{label}-parent")
    support = (_ref(f"live-{label}-support"),)
    commitment = ProspectiveCommitment(
        parent_event_ref=None,
        task_id=f"cognee-live-{label}",
        candidate_index=0,
        candidate_trace=f"synthetic {label} candidate a",
        predicted_score=0.5,
        uncertainty=0.25,
        horizon=1,
        competence_state_digest=parent_ref,
    )
    return CognitiveEpisode(
        task_id=f"cognee-live-{label}",
        request=f"Observe the synthetic {_QUERY} {label} event.",
        recalled_refs=support,
        proposals=(
            f"synthetic {label} candidate a",
            f"synthetic {label} candidate b",
        ),
        selected_index=0,
        commitment=commitment,
        response=f"synthetic {label} response",
        observations=(f"synthetic {label} observation",),
        outcome="success",
        feedback_text=f"synthetic {label} objective feedback",
        feedback_source_ref=_ref(f"live-{label}-feedback"),
        parent_state_digest=parent_ref,
        child_state_digest=_ref(f"live-{label}-child"),
        model_ref=_ref("live-model"),
        encoder_ref=_ref("live-encoder"),
        supporting_evidence_refs=support,
    )


def _record(
    *,
    ordinal: int,
    source_ref: str,
    kind: CognitiveMemoryKind,
    status: EpistemicStatus,
    label: str,
    relations: tuple[CognitiveRelation, ...] = (),
) -> CognitiveMemoryRecord:
    return CognitiveMemoryRecord(
        kind=kind,
        epistemic_status=status,
        content=f"{_QUERY}. Synthetic canonical {label} record.",
        provenance_refs=tuple(
            sorted((source_ref, _ref(f"live-{label}-record-evidence")))
        ),
        visibility="LEARNER_VISIBLE",
        producer_id="ANG-TEST-COGNEE-ACQUISITION-LIVE-001",
        producer_checkpoint_ref=_ref("live-checkpoint"),
        competence_ref=_ref(f"live-{label}-competence"),
        acquired_ordinal=ordinal,
        world_valid_from=ordinal,
        relations=relations,
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


def _item(acquisition: CognitiveAcquisition) -> _Item:
    return _Item(
        ordinal=acquisition.ordinal,
        acquisition_ref=acquisition.acquisition_ref,
        record_ref=acquisition.record.record_ref,
        acquisition=acquisition,
    )


def _outbox(item: _Item) -> _Outbox:
    projection = CognitiveGraphProjectionV2.from_acquisition(item.acquisition)
    return _Outbox(
        ordinal=item.ordinal,
        acquisition_ref=item.acquisition_ref,
        record_ref=item.record_ref,
        projection_ref=projection.projection_ref,
        projection=projection,
    )


def _chain() -> tuple[_Item, _Item, _Item]:
    proposed_source = _prospective_batch()
    proposed_record = _record(
        ordinal=0,
        source_ref=proposed_source.batch_ref,
        kind=CognitiveMemoryKind.COUNTERFACTUAL,
        status=EpistemicStatus.PROPOSED,
        label="proposed",
    )
    proposed = CognitiveAcquisition.from_source(
        proposed_source,
        ordinal=0,
        predecessor_acquisition_ref=None,
        record=proposed_record,
    )

    observed_source_a = _episode("observed-a")
    observed_record_a = _record(
        ordinal=1,
        source_ref=observed_source_a.episode_ref,
        kind=CognitiveMemoryKind.EPISODIC,
        status=EpistemicStatus.OBSERVED,
        label="observed-a",
    )
    observed_a = CognitiveAcquisition.from_source(
        observed_source_a,
        ordinal=1,
        predecessor_acquisition_ref=proposed.acquisition_ref,
        record=observed_record_a,
    )

    observed_source_b = _episode("observed-b")
    observed_record_b = _record(
        ordinal=2,
        source_ref=observed_source_b.episode_ref,
        kind=CognitiveMemoryKind.EPISODIC,
        status=EpistemicStatus.OBSERVED,
        label="observed-b",
        relations=(
            CognitiveRelation(
                RelationType.DERIVED_FROM,
                observed_record_a.record_ref,
            ),
        ),
    )
    observed_b = CognitiveAcquisition.from_source(
        observed_source_b,
        ordinal=2,
        predecessor_acquisition_ref=observed_a.acquisition_ref,
        record=observed_record_b,
    )
    return _item(proposed), _item(observed_a), _item(observed_b)


class _Store:
    def __init__(self, items: tuple[_Item, ...]) -> None:
        self._items = items
        self._by_record = {item.record_ref: item for item in items}
        self._pending = [_outbox(item) for item in items]
        self._acknowledged: set[str] = set()

    def acquisition_head(self) -> _Head:
        if not self._items:
            return _Head(0, None, None)
        last = self._items[-1]
        return _Head(last.ordinal + 1, last.acquisition_ref, last.record_ref)

    def get_acquisition_item(self, record_ref: str) -> _Item:
        return self._by_record[record_ref]

    def acquisition_items(
        self, after_ordinal: int = -1, limit: int = 64
    ) -> tuple[_Item, ...]:
        return tuple(
            item for item in self._items if item.ordinal > after_ordinal
        )[:limit]

    def pending_acquisition_projections(
        self, limit: int = 64
    ) -> tuple[_Outbox, ...]:
        return tuple(self._pending[:limit])

    def ack_acquisition_projection(self, projection_ref: str) -> None:
        for index, item in enumerate(self._pending):
            if item.projection_ref == projection_ref:
                self._pending.pop(index)
                self._acknowledged.add(projection_ref)
                return
        if projection_ref not in self._acknowledged:
            raise KeyError(projection_ref)


def _adapter(bindings: CogneeSubprocessBindings) -> CogneeAcquisitionAdapter:
    return CogneeAcquisitionAdapter(
        bindings=bindings,
        tenant_id=bindings.tenant_id,
        dataset_id=bindings.dataset_id,
        dataset_name=bindings.dataset_name,
        node_set_name=bindings.node_set_name,
        local_embeddings_configured=True,
        external_embedding_calls_authorized=False,
        telemetry_authorized=False,
    )


@unittest.skipUnless(
    os.environ.get(_LIVE_FLAG) == "1",
    f"set {_LIVE_FLAG}=1 to run the isolated live Cognee qualification",
)
class CogneeAcquisitionLiveQualification(unittest.IsolatedAsyncioTestCase):
    def assert_cognee_not_imported(self) -> None:
        self.assertFalse(
            any(name == "cognee" or name.startswith("cognee.") for name in sys.modules)
        )

    def assert_raw_reference_search(
        self,
        raw: list[dict[str, object]],
        *,
        bindings: CogneeSubprocessBindings,
        items: tuple[_Item, ...],
    ) -> tuple[str, ...]:
        self.assertEqual(len(raw), 1)
        envelope = raw[0]
        self.assertEqual(
            set(envelope),
            {
                "tenant_id",
                "dataset_id",
                "dataset_name",
                "node_set_name",
                "score_semantics",
                "search_result",
            },
        )
        self.assertEqual(envelope["tenant_id"], bindings.tenant_id)
        self.assertEqual(envelope["dataset_id"], bindings.dataset_id)
        self.assertEqual(envelope["dataset_name"], DATASET_NAME)
        self.assertEqual(envelope["node_set_name"], NODE_SET_NAME)
        self.assertEqual(envelope["score_semantics"], SCORE_SEMANTICS)
        entries = envelope["search_result"]
        self.assertIs(type(entries), list)
        assert type(entries) is list
        self.assertEqual(len(entries), len(items))
        scores: list[float] = []
        record_refs: list[str] = []
        backend_ids: list[str] = []
        expected_adjacency = {
            items[0].record_ref: [],
            items[1].record_ref: [],
            items[2].record_ref: [items[1].record_ref],
        }
        for entry in entries:
            self.assertIs(type(entry), dict)
            assert type(entry) is dict
            self.assertEqual(set(entry), {"id", "score", "payload"})
            backend_ids.append(entry["id"])
            scores.append(float(entry["score"]))
            payload = entry["payload"]
            self.assertIs(type(payload), dict)
            assert type(payload) is dict
            self.assertEqual(
                set(payload),
                {"record_ref", "adjacent_record_refs", "belongs_to_set"},
            )
            record_ref = payload["record_ref"]
            self.assertIs(type(record_ref), str)
            assert type(record_ref) is str
            record_refs.append(record_ref)
            self.assertEqual(
                payload["adjacent_record_refs"], expected_adjacency[record_ref]
            )
            self.assertEqual(payload["belongs_to_set"], [NODE_SET_NAME])
        self.assertEqual(len(set(backend_ids)), len(items))
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(set(record_refs), {item.record_ref for item in items})
        return tuple(record_refs)

    def assert_canonical_rejoin(
        self,
        memory: AcquisitionSituatedMemory,
        hits: tuple[AcquisitionReferenceHit, ...],
        items: tuple[_Item, ...],
    ) -> None:
        by_ref = {item.record_ref: item for item in items}
        result = memory.recall_from_hits(hits, limit=3, world_time=2)
        expected_refs = tuple(
            hit.record_ref
            for hit in hits
            if by_ref[hit.record_ref].acquisition.record.epistemic_status
            is EpistemicStatus.OBSERVED
        )
        self.assertEqual(
            tuple(item.record_ref for item in result.items), expected_refs
        )
        self.assertEqual(
            result.rejected, ("REFERENCE_EPISTEMIC_STATUS_DENIED",)
        )
        for recalled in result.items:
            canonical = by_ref[recalled.record_ref].acquisition.record
            self.assertEqual(
                recalled.record.canonical_bytes(), canonical.canonical_bytes()
            )
            self.assertIs(recalled.record.epistemic_status, EpistemicStatus.OBSERVED)
            self.assertEqual(recalled.acquired_ordinal, canonical.acquired_ordinal)
            self.assertTrue(recalled.world_valid_at_query)
        joined = next(
            item for item in result.items if item.record_ref == items[2].record_ref
        )
        self.assertEqual(
            tuple(record.record_ref for record in joined.adjacent_records),
            (items[1].record_ref,),
        )

    async def test_live_projection_search_rejoin_restart_and_exact_forget(self) -> None:
        self.assert_cognee_not_imported()
        items = _chain()
        self.assertEqual(len(items), 3)
        self.assertEqual(
            tuple(item.acquisition.record.epistemic_status for item in items),
            (
                EpistemicStatus.PROPOSED,
                EpistemicStatus.OBSERVED,
                EpistemicStatus.OBSERVED,
            ),
        )

        active_bindings: CogneeSubprocessBindings | None = None
        active_adapter: CogneeAcquisitionAdapter | None = None
        namespace_empty = False
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "TELEMETRY_DISABLED": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            clear=False,
        ):
            try:
                first = await CogneeSubprocessBindings.start(
                    timeout_seconds=_TIMEOUT_SECONDS
                )
                active_bindings = first
                first_adapter = _adapter(first)
                active_adapter = first_adapter
                dataset_id = first.dataset_id
                self.assertEqual(
                    dataset_id, modern_dataset_id(first.user_id, first.tenant_id)
                )
                legacy_id = str(uuid5(NAMESPACE_OID, f"{DATASET_NAME}{first.user_id}"))
                self.assertNotEqual(dataset_id, legacy_id)
                self.assertTrue(first.backend_access_control)
                self.assertEqual(first.dataset_name, DATASET_NAME)
                self.assertEqual(first.node_set_name, NODE_SET_NAME)
                self.assertEqual(first.node_set_id, NODE_SET_ID)
                self.assertEqual(first.score_semantics, SCORE_SEMANTICS)
                self.assertEqual(
                    first.embedding_execution_providers,
                    FASTEMBED_EXECUTION_PROVIDERS,
                )
                self.assertEqual(
                    first.embedding_intra_op_threads,
                    FASTEMBED_SESSION_THREADS,
                )
                self.assertEqual(
                    first.embedding_inter_op_threads,
                    FASTEMBED_SESSION_THREADS,
                )
                self.assert_cognee_not_imported()

                await first_adapter.forget_namespace()
                namespace_empty = True
                first_store = _Store(items)
                first_memory = AcquisitionSituatedMemory(
                    source=first_store,
                    backend=first_adapter,
                )
                namespace_empty = False
                projected = await first_memory.retry_pending_projections(limit=3)
                self.assertEqual(projected, tuple(item.record_ref for item in items))
                first_memory.assert_synchronized()

                projection = CognitiveGraphProjectionV2.from_acquisition(
                    items[1].acquisition
                )
                backend_ref_a = await first_adapter.project(projection)
                backend_ref_b = await first_adapter.project(projection)
                self.assertEqual(backend_ref_a, backend_ref_b)

                raw = await first.search_references(_QUERY, limit=3)
                self.assert_raw_reference_search(raw, bindings=first, items=items)
                hits = tuple(await first_adapter.search(_QUERY, limit=3))
                self.assertEqual(
                    {hit.record_ref for hit in hits},
                    {item.record_ref for item in items},
                )
                self.assertTrue(all(hit.score is not None for hit in hits))
                self.assertEqual(
                    [hit.score for hit in hits],
                    sorted(hit.score for hit in hits if hit.score is not None),
                )
                self.assertEqual(
                    len({hit.backend_ref for hit in hits}), len(items)
                )
                self.assert_canonical_rejoin(first_memory, hits, items)
                self.assert_cognee_not_imported()

                await first.close()
                active_bindings = None
                active_adapter = None

                second = await CogneeSubprocessBindings.start(
                    expected_dataset_id=dataset_id,
                    timeout_seconds=_TIMEOUT_SECONDS,
                )
                active_bindings = second
                second_adapter = _adapter(second)
                active_adapter = second_adapter
                self.assertEqual(second.dataset_id, dataset_id)
                self.assertEqual(second.user_id, first.user_id)
                self.assertEqual(second.tenant_id, first.tenant_id)
                self.assertEqual(
                    second.embedding_execution_providers,
                    FASTEMBED_EXECUTION_PROVIDERS,
                )
                self.assertEqual(
                    second.embedding_intra_op_threads,
                    FASTEMBED_SESSION_THREADS,
                )
                self.assertEqual(
                    second.embedding_inter_op_threads,
                    FASTEMBED_SESSION_THREADS,
                )
                persisted = await second.search_references(_QUERY, limit=3)
                self.assert_raw_reference_search(
                    persisted, bindings=second, items=items
                )

                second_store = _Store(items)
                second_memory = AcquisitionSituatedMemory(
                    source=second_store,
                    backend=second_adapter,
                )
                namespace_empty = False
                rebuilt = await second_memory.rebuild_from_store(page_size=3)
                self.assertEqual(rebuilt, tuple(item.record_ref for item in items))
                second_memory.assert_synchronized()
                restarted = await second_memory.recall(_QUERY, limit=3, world_time=2)
                self.assertEqual(
                    {item.record_ref for item in restarted.items},
                    {items[1].record_ref, items[2].record_ref},
                )
                self.assertEqual(
                    restarted.rejected,
                    ("REFERENCE_EPISTEMIC_STATUS_DENIED",),
                )

                await second_adapter.forget_namespace()
                namespace_empty = True
                self.assertEqual(second.dataset_id, dataset_id)
                self.assert_cognee_not_imported()
            finally:
                propagating = sys.exc_info()[0] is not None
                if active_adapter is not None and not namespace_empty:
                    try:
                        await active_adapter.forget_namespace()
                    except Exception:
                        if not propagating:
                            raise
                if active_bindings is not None:
                    await active_bindings.close()
                self.assert_cognee_not_imported()


if __name__ == "__main__":
    unittest.main()
