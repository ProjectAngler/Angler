from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from angler.cognition import CognitiveExecutionReceipt
from angler.memory import (
    MemoryHit,
    MemoryProjection,
    SituatedMemory,
    encode_projection,
)
from angler.memory.cognitive_graph import (
    CognitiveGraphProjection,
    TypedSituatedMemory,
)
from angler.reasoning.structure_keyed_credit_memory import StructureKeyedCreditMemoryCore
from angler.runtime.cognitive_cycle import (
    CognitiveCycle,
    CognitiveExecution,
    CognitiveExecutionRecovery,
    ExecutionReconciliationRequired,
    ExecutedCognitiveTurn,
)
from angler.runtime.cognitive_transaction_store import (
    CognitiveEpisodeItem,
    CognitiveTransactionStore,
    StateHead,
)
from angler.runtime.durable_ability_bridge import (
    AbilityDecision,
    DurableAbilityLearner,
    EncodedProcedureCandidate,
)


def _ref(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


class Backend:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.hits: tuple[MemoryHit, ...] = ()
        self.fail_remember = False

    async def remember(self, document: str) -> str:
        if self.fail_remember:
            raise RuntimeError("injected projection failure")
        self.documents.append(document)
        return f"projection-{len(self.documents)}"

    async def search(self, query: str, *, limit: int) -> tuple[MemoryHit, ...]:
        return self.hits[:limit]

    async def forget_dataset(self) -> None:
        self.documents.clear()


class TextAdapter:
    def __init__(self) -> None:
        self.proposals = ("inspect evidence", "form hypothesis", "test hypothesis", "revise plan")

    def propose_procedure_traces(self, task, recalled_evidence, *, count=4):
        self.last_recall = recalled_evidence
        self.count = count
        if count <= len(self.proposals):
            return self.proposals[:count]
        return (*self.proposals, *(f"candidate procedure {index}" for index in range(4, count)))

    def execute_with_procedure(self, task, selected_trace, observations):
        raise AssertionError("the cognitive cycle uses its generic executor")


class RelationAdapter:
    def __init__(self, evidence_ref: str) -> None:
        self.evidence_ref = evidence_ref

    def encode_candidates(self, task, proposals, recall):
        return tuple(
            EncodedProcedureCandidate(
                trace=trace,
                relation_features=torch.tensor([float(index)]),
                temporal_features=torch.tensor([0.0]),
                base_logit=torch.tensor(float(index)),
                supporting_evidence_refs=(self.evidence_ref,) if index == 2 else (),
            )
            for index, trace in enumerate(proposals)
        )


class NoEvidenceRelationAdapter(RelationAdapter):
    def encode_candidates(self, task, proposals, recall):
        return tuple(
            EncodedProcedureCandidate(
                trace=trace,
                relation_features=torch.tensor([float(index)]),
                temporal_features=torch.tensor([0.0]),
                base_logit=torch.tensor(float(index)),
                supporting_evidence_refs=(),
            )
            for index, trace in enumerate(proposals)
        )


class RealCoreRelationAdapter:
    def __init__(self, temporal_width: int) -> None:
        self.temporal_width = temporal_width

    def encode_candidates(self, task, proposals, recall):
        return tuple(
            EncodedProcedureCandidate(
                trace=trace,
                relation_features=torch.linspace(0.0, 1.0, 64)
                + float(index) / 100.0,
                temporal_features=torch.tensor(
                    [
                        float(index + offset) / 100.0
                        for offset in range(self.temporal_width)
                    ]
                ),
                base_logit=torch.tensor(float(index) / 10.0),
                supporting_evidence_refs=(),
            )
            for index, trace in enumerate(proposals)
        )


class TypedBackend:
    def __init__(self) -> None:
        self.projections: list[CognitiveGraphProjection] = []
        self.fail_project = False

    async def project(self, projection: CognitiveGraphProjection) -> str:
        if self.fail_project:
            raise RuntimeError("injected typed projection failure")
        self.projections.append(projection)
        return f"typed-{len(self.projections)}"

    async def search(self, query: str, *, limit: int):
        return ()

    async def forget_namespace(self) -> None:
        self.projections.clear()


class Learner:
    def __init__(self, state: int = 0) -> None:
        self.state = state
        self._pending = None
        self.updates = 0
        self.fail_update = False

    @property
    def pending_decision(self):
        return self._pending

    def capture_state(self) -> bytes:
        return str(self.state).encode()

    def restore_state(self, state: bytes) -> None:
        self.state = int(state.decode())
        self._pending = None

    def state_digest(self) -> str:
        return _ref(str(self.state))

    def select(self, *, task_id, request, candidates):
        if self._pending is not None:
            raise RuntimeError("pending")
        selected_index = min(2, len(candidates) - 1)
        logits = tuple(
            2.0 if index == selected_index else float(index) / 4.0
            for index in range(len(candidates))
        )
        decision = AbilityDecision(
            task_id=task_id,
            request=request,
            proposals=tuple(item.trace for item in candidates),
            selected_index=selected_index,
            selected_trace=candidates[selected_index].trace,
            logits=logits,
            evidence_ref=_ref(f"{task_id}:{self.state}"),
            supporting_evidence_refs=candidates[selected_index].supporting_evidence_refs,
            parent_state_digest=self.state_digest(),
        )
        self._pending = decision
        return decision

    def capture_pending_state(self) -> bytes:
        if self._pending is None:
            raise RuntimeError("no pending decision")
        value = self._pending
        return json.dumps(
            {
                "task_id": value.task_id,
                "request": value.request,
                "proposals": list(value.proposals),
                "selected_index": value.selected_index,
                "selected_trace": value.selected_trace,
                "logits": [item.hex() for item in value.logits],
                "evidence_ref": value.evidence_ref,
                "supporting_evidence_refs": list(value.supporting_evidence_refs),
                "parent_state_digest": value.parent_state_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def restore_pending_state(self, payload: bytes):
        if self._pending is not None:
            raise RuntimeError("pending")
        value = json.loads(payload.decode())
        decision = AbilityDecision(
            task_id=value["task_id"],
            request=value["request"],
            proposals=tuple(value["proposals"]),
            selected_index=value["selected_index"],
            selected_trace=value["selected_trace"],
            logits=tuple(float.fromhex(item) for item in value["logits"]),
            evidence_ref=value["evidence_ref"],
            supporting_evidence_refs=tuple(value["supporting_evidence_refs"]),
            parent_state_digest=value["parent_state_digest"],
        )
        if decision.parent_state_digest != self.state_digest():
            raise ValueError("pending decision belongs to another parent")
        self._pending = decision
        return decision

    def apply_outcome(self, outcome):
        if self.fail_update:
            raise RuntimeError("injected learner failure")
        if self._pending is None:
            raise RuntimeError("no pending decision")
        self.state += 1 if outcome.disposition == "success" else 2
        self.updates += 1
        self._pending = None


class Executor:
    def __init__(self, status: str = "COMPLETED") -> None:
        self.status = status
        self.calls = []
        self.recorded = {}

    def execute(self, request):
        self.calls.append(request)
        result = CognitiveExecution(
            self.status,
            request.selected_trace,
            "observable result" if self.status == "COMPLETED" else "need clarification",
            ("tool observation",),
        )
        self.recorded[request.execution_request_ref] = CognitiveExecutionReceipt.from_request(
            request,
            status=result.status,
            executed_trace=result.executed_trace,
            response=result.response,
            output_observations=result.observations,
        )
        return result

    def recover(self, request):
        receipt = self.recorded.get(request.execution_request_ref)
        if receipt is None:
            return CognitiveExecutionRecovery("NOT_STARTED")
        return CognitiveExecutionRecovery("RECORDED", receipt)


class BlockingExecutor(Executor):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, request):
        self.calls.append(request)
        self.started.set()
        await self.release.wait()
        result = CognitiveExecution("COMPLETED", request.selected_trace, "observable result")
        self.recorded[request.execution_request_ref] = CognitiveExecutionReceipt.from_request(
            request,
            status=result.status,
            executed_trace=result.executed_trace,
            response=result.response,
            output_observations=result.observations,
        )
        return result


class CrashAfterEffectExecutor(Executor):
    """Simulate process loss after the executor durably records its effect."""

    def __init__(self) -> None:
        super().__init__()
        self.crash_once = True

    def execute(self, request):
        result = super().execute(request)
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("simulated crash after durable effect")
        return result


class WrongReceiptRecoveryExecutor(CrashAfterEffectExecutor):
    def recover(self, request):
        receipt = self.recorded[request.execution_request_ref]
        return CognitiveExecutionRecovery(
            "RECORDED",
            replace(receipt, execution_request_ref=_ref("another-execution-request")),
        )


class NotStartedExecutor(Executor):
    """First call fails before mutation; recovery can prove NOT_STARTED."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def execute(self, request):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("simulated crash before mutation")
        return super().execute(request)


class EmptyResponseExecutor(Executor):
    def execute(self, request):
        self.calls.append(request)
        result = CognitiveExecution("COMPLETED", request.selected_trace, "", ())
        self.recorded[request.execution_request_ref] = CognitiveExecutionReceipt.from_request(
            request,
            status=result.status,
            executed_trace=result.executed_trace,
            response=result.response,
            output_observations=result.observations,
        )
        return result


class UnknownExecutor(Executor):
    def execute(self, request):
        self.calls.append(request)
        raise RuntimeError("effect status became unknown")

    def recover(self, request):
        return CognitiveExecutionRecovery("UNKNOWN")


class BlockingRecoveryExecutor(UnknownExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.recovery_started = asyncio.Event()
        self.release_recovery = asyncio.Event()

    async def recover(self, request):
        self.recovery_started.set()
        await self.release_recovery.wait()
        return CognitiveExecutionRecovery("UNKNOWN")


class CognitiveCycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Backend()
        self.memory = SituatedMemory(self.backend)
        self.evidence_ref = _ref("recalled evidence")
        projection = MemoryProjection.from_mapping(
            artifact_ref=self.evidence_ref,
            text="A validated prior attempt.",
            source_ref=_ref("source"),
        )
        await self.memory.remember(projection)
        self.backend.hits = (MemoryHit(self.backend.documents[0]),)
        self.learner = Learner()
        self.executor = Executor()
        self.store = CognitiveTransactionStore(Path(self.temp.name) / "cycle.sqlite")
        self.cycle = self._cycle()

    def _cycle(
        self,
        *,
        learner=None,
        executor=None,
        store=None,
        memory=None,
        model_ref=None,
        encoder_ref=None,
        agent_ref=None,
        world_ref=None,
        relation_adapter=None,
        proposal_count=4,
    ):
        return CognitiveCycle(
            text_adapter=TextAdapter(),
            relation_adapter=relation_adapter or RelationAdapter(self.evidence_ref),
            learner=learner or self.learner,
            memory=memory or self.memory,
            executor=executor or self.executor,
            transaction_store=store or self.store,
            model_ref=model_ref or _ref("frozen-model"),
            encoder_ref=encoder_ref or _ref("v13-encoder"),
            agent_ref=agent_ref or _ref("angler-agent"),
            world_ref=world_ref or _ref("actual-world"),
            proposal_count=proposal_count,
        )

    async def test_typed_memory_projects_canonical_record_and_retries_one_anchor(self):
        learner = Learner()
        store = CognitiveTransactionStore(Path(self.temp.name) / "typed-cycle.sqlite")
        backend = TypedBackend()
        memory = TypedSituatedMemory(source=store, backend=backend)
        cycle = self._cycle(
            learner=learner,
            store=store,
            memory=memory,
            relation_adapter=NoEvidenceRelationAdapter(self.evidence_ref),
        )
        turn = await cycle.begin_turn("typed-turn", "Exercise typed memory projection")
        completed = await cycle.execute_turn(turn)
        result = await cycle.record_outcome(
            completed,
            reservation_ref=turn.reservation.reservation_ref,
            task_id=turn.decision.task_id,
            outcome="success",
            feedback_text="Objective typed-memory verification passed.",
            feedback_source_ref=_ref("typed-verifier"),
        )
        self.assertFalse(result.projection_pending)
        self.assertEqual(len(backend.projections), 1)
        projected = backend.projections[0]
        self.assertEqual(projected.source_episode_ref, result.episode_ref)
        self.assertNotEqual(projected.record.record_ref, result.episode_ref)
        self.assertEqual(memory.origin.anchor(projected.record.record_ref).ordinal, 0)
        self.assertEqual(store.pending_projections(), ())

        second_learner = Learner()
        second_store = CognitiveTransactionStore(
            Path(self.temp.name) / "typed-cycle-retry.sqlite"
        )
        failing_backend = TypedBackend()
        failing_backend.fail_project = True
        second_memory = TypedSituatedMemory(
            source=second_store, backend=failing_backend
        )
        second_cycle = self._cycle(
            learner=second_learner,
            store=second_store,
            memory=second_memory,
            relation_adapter=NoEvidenceRelationAdapter(self.evidence_ref),
        )
        second_turn = await second_cycle.begin_turn(
            "typed-retry", "Retry one typed projection"
        )
        second_completed = await second_cycle.execute_turn(second_turn)
        pending = await second_cycle.record_outcome(
            second_completed,
            reservation_ref=second_turn.reservation.reservation_ref,
            task_id=second_turn.decision.task_id,
            outcome="failure",
            feedback_text="Projection backend was unavailable.",
            feedback_source_ref=_ref("typed-retry-verifier"),
        )
        self.assertTrue(pending.projection_pending)
        self.assertEqual(second_memory.origin.size, 1)
        failing_backend.fail_project = False
        self.assertEqual(
            await second_cycle.retry_pending_projections(), (pending.episode_ref,)
        )
        self.assertEqual(second_memory.origin.size, 1)
        self.assertEqual(second_store.pending_projections(), ())

    async def test_typed_memory_requires_the_cycle_canonical_store(self):
        cycle_store = CognitiveTransactionStore(
            Path(self.temp.name) / "typed-source-cycle.sqlite"
        )
        other_store = CognitiveTransactionStore(
            Path(self.temp.name) / "typed-source-other.sqlite"
        )
        memory = TypedSituatedMemory(source=other_store, backend=TypedBackend())
        with self.assertRaisesRegex(ValueError, "share one canonical store"):
            self._cycle(
                learner=Learner(),
                store=cycle_store,
                memory=memory,
                relation_adapter=NoEvidenceRelationAdapter(self.evidence_ref),
            )

    async def test_typed_restart_requires_explicit_bounded_rebuild(self):
        learner = Learner()
        store = CognitiveTransactionStore(Path(self.temp.name) / "typed-restart.sqlite")
        memory = TypedSituatedMemory(source=store, backend=TypedBackend())
        cycle = self._cycle(
            learner=learner,
            store=store,
            memory=memory,
            relation_adapter=NoEvidenceRelationAdapter(self.evidence_ref),
        )
        turn = await cycle.begin_turn("typed-restart", "Create one canonical episode")
        completed = await cycle.execute_turn(turn)
        await cycle.record_outcome(
            completed,
            reservation_ref=turn.reservation.reservation_ref,
            task_id=turn.decision.task_id,
            outcome="success",
            feedback_text="Verified for restart.",
            feedback_source_ref=_ref("restart-verifier"),
        )

        fresh_memory = TypedSituatedMemory(source=store, backend=TypedBackend())
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self._cycle(
                learner=Learner(999),
                store=store,
                memory=fresh_memory,
                relation_adapter=NoEvidenceRelationAdapter(self.evidence_ref),
            )
        await fresh_memory.rebuild_from_store(page_size=1)
        restarted = self._cycle(
            learner=Learner(999),
            store=store,
            memory=fresh_memory,
            relation_adapter=NoEvidenceRelationAdapter(self.evidence_ref),
        )
        self.assertIsNone(restarted.pending_turn)
        self.assertEqual(fresh_memory.origin.size, 1)

    async def test_typed_canonical_reload_mismatch_surfaces_after_durable_commit(self):
        learner = Learner()
        store = CognitiveTransactionStore(Path(self.temp.name) / "typed-reload.sqlite")
        memory = TypedSituatedMemory(source=store, backend=TypedBackend())
        cycle = self._cycle(
            learner=learner,
            store=store,
            memory=memory,
            relation_adapter=NoEvidenceRelationAdapter(self.evidence_ref),
        )
        turn = await cycle.begin_turn("typed-reload", "Verify canonical reload")
        completed = await cycle.execute_turn(turn)
        real_get = store.get_episode_item

        def mismatched_item(episode_ref):
            item = real_get(episode_ref)
            altered = replace(item.episode, feedback_text="Different canonical reload")
            return CognitiveEpisodeItem(item.sequence, item.episode_ref, altered)

        store.get_episode_item = mismatched_item
        with self.assertRaisesRegex(ValueError, "differs from the committed episode"):
            await cycle.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="failure",
                feedback_text="Canonical reload must remain exact.",
                feedback_source_ref=_ref("reload-verifier"),
            )
        self.assertEqual(store.head().sequence, 1)
        self.assertEqual(len(store.pending_projections()), 1)
        self.assertEqual(memory.origin.size, 0)
        store.get_episode_item = real_get
        pending_ref = store.head().episode_ref
        self.assertEqual(await cycle.retry_pending_projections(), (pending_ref,))

    async def _executed(self):
        turn = await self.cycle.begin_turn("task-1", "Investigate a novel failure")
        return await self.cycle.execute_turn(turn, observations=("public input",))

    async def test_configured_validated_proposals_one_pending_and_score_before_execution(self):
        turn = await self.cycle.begin_turn("task-1", "Investigate a novel failure")
        self.assertEqual(len(turn.decision.proposals), 4)
        self.assertEqual(len(set(turn.decision.proposals)), 4)
        self.assertEqual(turn.decision.supporting_evidence_refs, (self.evidence_ref,))
        self.assertEqual(self.learner.updates, 0)
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            await self.cycle.begin_turn("task-2", "Another task")
        completed = await self.cycle.execute_turn(turn)
        self.assertEqual(self.executor.calls[0].selected_trace, turn.decision.selected_trace)
        self.assertEqual(completed.execution.status, "COMPLETED")
        self.assertEqual(self.learner.updates, 0)

    async def test_candidate_cardinality_is_resource_bounded_not_fixed_at_four(self):
        learner = Learner()
        store = CognitiveTransactionStore(Path(self.temp.name) / "six-candidates.sqlite")
        cycle = self._cycle(
            learner=learner,
            store=store,
            proposal_count=6,
        )
        turn = await cycle.begin_turn("six", "Compare a wider sibling set")
        self.assertEqual(len(turn.decision.proposals), 6)
        self.assertEqual(len(turn.reservation.proposals), 6)
        self.assertEqual(turn.reservation.agent_ref, _ref("angler-agent"))
        self.assertEqual(turn.reservation.world_ref, _ref("actual-world"))

    async def test_real_v14_learner_cycle_handles_two_six_and_sixty_four(self):
        for count in (2, 6, 64):
            with self.subTest(count=count):
                torch.manual_seed(20260831)
                core = StructureKeyedCreditMemoryCore(temporal_width=3)
                learner = DurableAbilityLearner(
                    core,
                    core.initial_state(),
                    checkpoint_identity=_ref(f"v14-real-{count}"),
                )
                store = CognitiveTransactionStore(
                    Path(self.temp.name) / f"real-v14-{count}.sqlite"
                )
                cycle = self._cycle(
                    learner=learner,
                    store=store,
                    relation_adapter=RealCoreRelationAdapter(core.temporal_width),
                    proposal_count=count,
                )
                turn = await cycle.begin_turn(
                    f"real-{count}",
                    f"Evaluate {count} learned candidate relations.",
                )
                self.assertEqual(len(turn.decision.proposals), count)
                self.assertEqual(len(turn.decision.logits), count)

                restarted_learner = DurableAbilityLearner(
                    core,
                    core.initial_state(),
                    checkpoint_identity=_ref(f"v14-real-{count}"),
                )
                restarted = self._cycle(
                    learner=restarted_learner,
                    store=store,
                    relation_adapter=RealCoreRelationAdapter(core.temporal_width),
                    proposal_count=count,
                )
                self.assertEqual(restarted.pending_turn.decision, turn.decision)
                completed = await restarted.execute_turn(restarted.pending_turn)
                result = await restarted.record_outcome(
                    completed,
                    reservation_ref=turn.reservation.reservation_ref,
                    task_id=turn.decision.task_id,
                    outcome="success",
                    feedback_text="The bounded real learner cycle completed.",
                    feedback_source_ref=_ref(f"real-v14-verifier-{count}"),
                )
                self.assertEqual(result.sequence, 1)

    async def test_concurrent_begin_has_one_winner_without_clearing_it(self):
        recall_started = asyncio.Event()
        release_recall = asyncio.Event()
        real_recall = self.memory.recall

        async def blocking_recall(*args, **kwargs):
            recall_started.set()
            await release_recall.wait()
            return await real_recall(*args, **kwargs)

        self.memory.recall = blocking_recall
        winner_task = asyncio.create_task(
            self.cycle.begin_turn("winner", "Keep exactly one prospective turn")
        )
        await recall_started.wait()
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            await self.cycle.begin_turn("loser", "Do not disturb the winner")
        release_recall.set()
        winner = await winner_task
        self.assertIs(self.cycle.pending_turn, winner)
        self.assertEqual(
            self.store.active_reservation().reservation_ref,
            winner.reservation.reservation_ref,
        )
        completed = await self.cycle.execute_turn(winner)
        self.assertEqual(completed.execution.status, "COMPLETED")

    async def test_execute_is_exactly_once_concurrent_safe_and_fabrication_resistant(self):
        blocking = BlockingExecutor()
        cycle = self._cycle(executor=blocking)
        turn = await cycle.begin_turn("once", "Execute one public procedure")
        active = asyncio.create_task(cycle.execute_turn(turn))
        await blocking.started.wait()
        with self.assertRaisesRegex(ValueError, "currently pending"):
            await cycle.execute_turn(turn)
        blocking.release.set()
        completed = await active
        with self.assertRaisesRegex(ValueError, "currently pending"):
            await cycle.execute_turn(turn)
        fabricated = ExecutedCognitiveTurn(
            completed.turn,
            completed.execution,
            completed.input_observations,
            completed.execution_request,
            completed.execution_receipt,
        )
        with self.assertRaisesRegex(ValueError, "awaiting objective feedback"):
            await cycle.record_outcome(
                fabricated,
                reservation_ref=turn.reservation.reservation_ref,
                task_id="once",
                outcome="success",
                feedback_text="Verified.",
                feedback_source_ref=_ref("verifier"),
            )
        self.assertEqual(self.learner.updates, 0)

    async def test_head_is_exactly_bound_before_selection_execution_and_feedback(self):
        original = self.store.head()
        stale = StateHead(
            original.sequence + 1,
            _ref("aba-event"),
            original.state_digest,
            original.snapshot_sha256,
        )
        calls = 0
        real_head = self.store.head

        def changes_after_capture():
            nonlocal calls
            calls += 1
            return original if calls == 1 else stale

        self.store.head = changes_after_capture
        with self.assertRaisesRegex(ValueError, "changed while"):
            await self.cycle.begin_turn("aba-select", "Detect same-digest ABA")
        self.store.head = real_head
        self.assertEqual(self.learner.capture_state(), b"0")

        turn = await self.cycle.begin_turn("aba-execute", "Detect stale execution")
        self.store.head = lambda: stale
        with self.assertRaisesRegex(ValueError, "changed before procedure"):
            await self.cycle.execute_turn(turn)
        self.store.head = real_head
        self.assertEqual(self.learner.capture_state(), b"0")

        turn = await self.cycle.begin_turn("aba-feedback", "Detect stale feedback")
        completed = await self.cycle.execute_turn(turn)
        self.store.head = lambda: stale
        with self.assertRaisesRegex(ValueError, "changed before objective"):
            await self.cycle.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id="aba-feedback",
                outcome="success",
                feedback_text="Verified.",
                feedback_source_ref=_ref("verifier"),
            )
        self.store.head = real_head
        self.assertEqual(self.learner.updates, 0)

    async def test_feedback_identity_mismatch_never_updates(self):
        completed = await self._executed()
        parent = self.learner.capture_state()
        with self.assertRaisesRegex(ValueError, "feedback identity"):
            await self.cycle.record_outcome(
                completed,
                reservation_ref=_ref("wrong"),
                task_id="task-1",
                outcome="success",
                feedback_text="Externally verified.",
                feedback_source_ref=_ref("verifier"),
            )
        self.assertEqual(self.learner.capture_state(), parent)
        self.assertEqual(self.learner.updates, 0)

    async def test_success_and_failure_each_apply_one_update_and_commit_exact_state(self):
        completed = await self._executed()
        result = await self.cycle.record_outcome(
            completed,
            reservation_ref=completed.turn.reservation.reservation_ref,
            task_id="task-1",
            outcome="failure",
            feedback_text="The external check failed.",
            feedback_source_ref=_ref("verifier"),
        )
        self.assertEqual(self.learner.updates, 1)
        self.assertFalse(result.projection_pending)
        episode = self.store.get_episode(result.episode_ref)
        self.assertEqual(episode.parent_state_digest, _ref("0"))
        self.assertEqual(episode.child_state_digest, _ref("2"))
        self.assertEqual(episode.commitment, completed.turn.commitment)
        self.assertEqual(episode.observations, ("public input", "tool observation"))
        self.assertEqual(self.store.load_head_state(), b"2")
        self.assertEqual(self.store.pending_projections(), ())
        with self.assertRaisesRegex(ValueError, "awaiting objective feedback"):
            await self.cycle.record_outcome(
                completed,
                reservation_ref=completed.turn.reservation.reservation_ref,
                task_id="task-1",
                outcome="failure",
                feedback_text="Duplicate feedback.",
                feedback_source_ref=_ref("verifier"),
            )
        self.assertEqual(self.learner.updates, 1)

    async def test_clarification_and_executor_error_restore_without_update(self):
        for status in ("CLARIFICATION_REQUIRED", "ERROR"):
            learner = Learner()
            executor = Executor(status)
            store = CognitiveTransactionStore(
                Path(self.temp.name) / f"{status.lower()}.sqlite"
            )
            cycle = self._cycle(learner=learner, executor=executor, store=store)
            turn = await cycle.begin_turn(status.lower(), "Need a bounded action")
            completed = await cycle.execute_turn(turn)
            self.assertEqual(completed.execution.status, status)
            self.assertIsNone(cycle.pending_turn)
            self.assertIsNone(learner.pending_decision)
            self.assertEqual(learner.updates, 0)
            self.assertEqual(store.head().sequence, 0)

    async def test_noncompletion_does_not_collide_with_changed_request_or_world(self):
        learner = Learner()
        executor = Executor("CLARIFICATION_REQUIRED")
        store = CognitiveTransactionStore(Path(self.temp.name) / "context-key.sqlite")
        first_cycle = self._cycle(learner=learner, executor=executor, store=store)
        first = await first_cycle.begin_turn(
            "same-task",
            "First request under the actual world.",
        )
        await first_cycle.execute_turn(first)
        self.assertIsNone(store.active_reservation())

        executor.status = "COMPLETED"
        second_cycle = self._cycle(
            learner=learner,
            executor=executor,
            store=store,
            world_ref=_ref("another-world"),
        )
        second = await second_cycle.begin_turn(
            "same-task",
            "Changed request under another world.",
        )
        self.assertEqual(
            first.commitment.commitment_ref,
            second.commitment.commitment_ref,
        )
        self.assertNotEqual(first.reservation.reservation_ref, second.reservation.reservation_ref)
        completed = await second_cycle.execute_turn(second)
        self.assertNotEqual(
            executor.calls[0].idempotency_key,
            completed.execution_request.idempotency_key,
        )
        self.assertEqual(len(executor.calls), 2)
        with self.assertRaisesRegex(ValueError, "exact reservation"):
            await second_cycle.record_outcome(
                completed,
                reservation_ref=first.reservation.reservation_ref,
                task_id=second.decision.task_id,
                outcome="success",
                feedback_text="This feedback carries the stale reservation.",
                feedback_source_ref=_ref("stale-context-verifier"),
            )
        self.assertEqual(learner.updates, 0)
        result = await second_cycle.record_outcome(
            completed,
            reservation_ref=second.reservation.reservation_ref,
            task_id=second.decision.task_id,
            outcome="success",
            feedback_text="This feedback carries the exact reservation.",
            feedback_source_ref=_ref("exact-context-verifier"),
        )
        self.assertEqual(result.sequence, 1)
        self.assertEqual(learner.updates, 1)

    async def test_empty_completed_response_rehydrates_after_restart(self):
        executor = EmptyResponseExecutor()
        store = CognitiveTransactionStore(Path(self.temp.name) / "empty-response.sqlite")
        cycle = self._cycle(learner=Learner(), executor=executor, store=store)
        turn = await cycle.begin_turn("empty-response", "Permit an exact empty response")
        completed = await cycle.execute_turn(turn)
        self.assertEqual(completed.execution.response, "")

        restarted = self._cycle(
            learner=Learner(999),
            executor=executor,
            store=store,
        )
        self.assertIsNotNone(restarted.executed_turn)
        self.assertEqual(restarted.executed_turn.execution.response, "")
        result = await restarted.record_outcome(
            restarted.executed_turn,
            reservation_ref=turn.reservation.reservation_ref,
            task_id=turn.decision.task_id,
            outcome="success",
            feedback_text="The empty public response was valid.",
            feedback_source_ref=_ref("empty-response-verifier"),
        )
        self.assertEqual(result.sequence, 1)

    async def test_learner_and_store_failure_restore_exact_parent(self):
        completed = await self._executed()
        self.learner.fail_update = True
        with self.assertRaisesRegex(RuntimeError, "learner failure"):
            await self.cycle.record_outcome(
                completed,
                reservation_ref=completed.turn.reservation.reservation_ref,
                task_id="task-1",
                outcome="success",
                feedback_text="Verified.",
                feedback_source_ref=_ref("verifier"),
            )
        self.assertEqual(self.learner.capture_state(), b"0")
        self.assertEqual(self.store.head().sequence, 0)

        def fail(stage):
            if stage == "before_commit":
                raise RuntimeError("injected store failure")

        learner = Learner()
        store = CognitiveTransactionStore(
            Path(self.temp.name) / "store-fail.sqlite", fault_injector=fail
        )
        cycle = self._cycle(learner=learner, store=store)
        turn = await cycle.begin_turn("task-store", "Test atomic store rollback")
        executed = await cycle.execute_turn(turn)
        with self.assertRaisesRegex(RuntimeError, "store failure"):
            await cycle.record_outcome(
                executed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id="task-store",
                outcome="success",
                feedback_text="Verified.",
                feedback_source_ref=_ref("verifier"),
            )
        self.assertEqual(learner.capture_state(), b"0")
        self.assertEqual(store.head().sequence, 0)

    async def test_projection_failure_preserves_commit_and_retry_acks_outbox(self):
        completed = await self._executed()
        self.backend.fail_remember = True
        result = await self.cycle.record_outcome(
            completed,
            reservation_ref=completed.turn.reservation.reservation_ref,
            task_id="task-1",
            outcome="success",
            feedback_text="Verified despite projection outage.",
            feedback_source_ref=_ref("verifier"),
        )
        self.assertTrue(result.projection_pending)
        self.assertEqual(self.store.head().state_digest, self.learner.state_digest())
        self.assertEqual(len(self.store.pending_projections()), 1)
        self.assertEqual(self.learner.capture_state(), b"1")

        self.backend.fail_remember = False
        requested_limits = []
        pending_projections = self.store.pending_projections

        def bounded_pending_projections(*, limit):
            requested_limits.append(limit)
            return pending_projections(limit=limit)

        self.store.pending_projections = bounded_pending_projections
        self.assertEqual(
            await self.cycle.retry_pending_projections(limit=1),
            (result.episode_ref,),
        )
        self.assertEqual(requested_limits, [1])
        self.store.pending_projections = pending_projections
        self.assertEqual(self.store.pending_projections(), ())

    async def test_ack_failure_after_projection_is_pending_and_retryable(self):
        completed = await self._executed()
        real_ack = self.store.ack_projection
        self.store.ack_projection = lambda _ref_value: (_ for _ in ()).throw(
            RuntimeError("injected ack failure")
        )
        result = await self.cycle.record_outcome(
            completed,
            reservation_ref=completed.turn.reservation.reservation_ref,
            task_id="task-1",
            outcome="success",
            feedback_text="Verified.",
            feedback_source_ref=_ref("verifier"),
        )
        self.assertTrue(result.projection_pending)
        self.assertEqual(len(self.store.pending_projections()), 1)
        self.assertEqual(self.learner.capture_state(), b"1")
        self.store.ack_projection = real_ack
        self.assertEqual(await self.cycle.retry_pending_projections(), (result.episode_ref,))

    async def test_restart_restores_canonical_competence_bytes(self):
        completed = await self._executed()
        await self.cycle.record_outcome(
            completed,
            reservation_ref=completed.turn.reservation.reservation_ref,
            task_id="task-1",
            outcome="success",
            feedback_text="Verified.",
            feedback_source_ref=_ref("verifier"),
        )
        restarted = Learner(999)
        cycle = self._cycle(learner=restarted)
        self.assertEqual(restarted.capture_state(), b"1")
        self.assertEqual(restarted.state_digest(), self.store.head().state_digest)
        self.assertIsNone(cycle.pending_turn)
        with self.assertRaisesRegex(ValueError, "another model or encoder"):
            CognitiveCycle(
                text_adapter=TextAdapter(),
                relation_adapter=RelationAdapter(self.evidence_ref),
                learner=Learner(),
                memory=self.memory,
                executor=self.executor,
                transaction_store=self.store,
                model_ref=_ref("different-model"),
                encoder_ref=_ref("v13-encoder"),
                agent_ref=_ref("angler-agent"),
                world_ref=_ref("actual-world"),
            )

    async def test_sequence_zero_store_rejects_model_and_encoder_drift(self):
        store = CognitiveTransactionStore(Path(self.temp.name) / "identity.sqlite")
        self._cycle(learner=Learner(), store=store)
        with self.assertRaises(ValueError):
            self._cycle(
                learner=Learner(),
                store=store,
                model_ref=_ref("different-model"),
            )
        with self.assertRaises(ValueError):
            self._cycle(
                learner=Learner(),
                store=store,
                encoder_ref=_ref("different-encoder"),
            )

    async def test_reserved_restart_restores_exact_pending_decision_and_world(self):
        turn = await self.cycle.begin_turn("reserved-restart", "Resume before execution")
        restarted_learner = Learner(999)
        restarted = self._cycle(learner=restarted_learner)
        restored = restarted.pending_turn
        self.assertIsNotNone(restored)
        self.assertEqual(restored.decision, turn.decision)
        self.assertEqual(restored.reservation.reservation_ref, turn.reservation.reservation_ref)
        self.assertEqual(restored.reservation.agent_ref, _ref("angler-agent"))
        self.assertEqual(restored.reservation.world_ref, _ref("actual-world"))
        completed = await restarted.execute_turn(restored)
        self.assertEqual(completed.execution_request.idempotency_key, turn.reservation.reservation_ref)
        self.assertEqual(len(self.executor.calls), 1)

    async def test_reserved_restart_rejects_agent_or_world_identity_drift(self):
        await self.cycle.begin_turn("identity-restart", "Bind the acting world")
        with self.assertRaisesRegex(ValueError, "canonical head"):
            self._cycle(learner=Learner(999), world_ref=_ref("counterfactual-world"))
        with self.assertRaisesRegex(ValueError, "canonical head"):
            self._cycle(learner=Learner(999), agent_ref=_ref("another-agent"))

    async def test_recorded_effect_recovery_never_executes_a_second_time(self):
        executor = CrashAfterEffectExecutor()
        store = CognitiveTransactionStore(Path(self.temp.name) / "recorded-recovery.sqlite")
        cycle = self._cycle(learner=Learner(), executor=executor, store=store)
        turn = await cycle.begin_turn("recorded", "Perform one recoverable effect")
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.execute_turn(turn)
        self.assertEqual(store.active_reservation().status, "CLAIMED")
        self.assertEqual(len(executor.calls), 1)

        restarted_learner = Learner(999)
        restarted = self._cycle(
            learner=restarted_learner,
            executor=executor,
            store=store,
        )
        with self.assertRaises(ExecutionReconciliationRequired):
            await restarted.execute_turn(restarted.pending_turn)
        completed = await restarted.recover_claimed_execution()
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(completed.execution.status, "COMPLETED")
        result = await restarted.record_outcome(
            completed,
            reservation_ref=turn.reservation.reservation_ref,
            task_id=turn.decision.task_id,
            outcome="success",
            feedback_text="Recovered effect verified once.",
            feedback_source_ref=_ref("recorded-recovery-verifier"),
        )
        self.assertEqual(result.sequence, 1)
        self.assertEqual(restarted_learner.updates, 1)

    async def test_recorded_recovery_rejects_a_receipt_from_another_request(self):
        executor = WrongReceiptRecoveryExecutor()
        store = CognitiveTransactionStore(Path(self.temp.name) / "wrong-receipt.sqlite")
        cycle = self._cycle(learner=Learner(), executor=executor, store=store)
        turn = await cycle.begin_turn("wrong-receipt", "Bind the executor receipt exactly")
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.execute_turn(turn)
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.recover_claimed_execution()
        self.assertEqual(store.active_reservation().status, "CLAIMED")
        self.assertEqual(len(executor.calls), 1)

    async def test_not_started_recovery_reuses_the_claimed_request_once(self):
        executor = NotStartedExecutor()
        store = CognitiveTransactionStore(Path(self.temp.name) / "not-started.sqlite")
        cycle = self._cycle(learner=Learner(), executor=executor, store=store)
        turn = await cycle.begin_turn("not-started", "Retry only with proof of no mutation")
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.execute_turn(turn)
        request = store.active_reservation().execution_request
        completed = await cycle.recover_claimed_execution()
        self.assertEqual(executor.attempts, 2)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(completed.execution_request, request)
        self.assertEqual(completed.execution_request.idempotency_key, turn.reservation.reservation_ref)

    async def test_unknown_claim_blocks_retry_and_new_work(self):
        executor = UnknownExecutor()
        store = CognitiveTransactionStore(Path(self.temp.name) / "unknown.sqlite")
        cycle = self._cycle(learner=Learner(), executor=executor, store=store)
        turn = await cycle.begin_turn("unknown", "Never guess about an external effect")
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.execute_turn(turn)
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.recover_claimed_execution()
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.execute_turn(turn)
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            await cycle.begin_turn("blocked", "No second turn")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(store.active_reservation().status, "CLAIMED")

    async def test_claimed_recovery_allows_only_one_reconciler(self):
        executor = BlockingRecoveryExecutor()
        store = CognitiveTransactionStore(Path(self.temp.name) / "recovery-race.sqlite")
        cycle = self._cycle(learner=Learner(), executor=executor, store=store)
        turn = await cycle.begin_turn("recovery-race", "Serialize effect reconciliation")
        with self.assertRaises(ExecutionReconciliationRequired):
            await cycle.execute_turn(turn)
        active = asyncio.create_task(cycle.recover_claimed_execution())
        await executor.recovery_started.wait()
        with self.assertRaisesRegex(ValueError, "already in progress"):
            await cycle.recover_claimed_execution()
        executor.release_recovery.set()
        with self.assertRaises(ExecutionReconciliationRequired):
            await active
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(store.active_reservation().status, "CLAIMED")

    async def test_recorded_execution_and_staged_feedback_resume_after_restart(self):
        fail_commit = True

        def fault(stage):
            if fail_commit and stage == "before_reserved_commit":
                raise RuntimeError("injected reserved commit failure")

        store = CognitiveTransactionStore(
            Path(self.temp.name) / "staged-restart.sqlite",
            fault_injector=fault,
        )
        learner = Learner()
        cycle = self._cycle(learner=learner, store=store)
        turn = await cycle.begin_turn("staged", "Resume feedback without re-execution")
        completed = await cycle.execute_turn(turn)
        with self.assertRaisesRegex(RuntimeError, "reserved commit failure"):
            await cycle.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="success",
                feedback_text="Persist this objective result.",
                feedback_source_ref=_ref("staged-verifier"),
            )
        active = store.active_reservation()
        self.assertEqual(active.status, "EXECUTION_RECORDED")
        self.assertIsNotNone(active.feedback)
        self.assertEqual(store.head().sequence, 0)

        fail_commit = False
        restarted_learner = Learner(999)
        restarted = self._cycle(learner=restarted_learner, store=store)
        self.assertIsNotNone(restarted.executed_turn)
        result = await restarted.resume_staged_outcome()
        self.assertEqual(result.sequence, 1)
        self.assertEqual(store.head().state_digest, restarted_learner.state_digest())
        self.assertIsNone(store.active_reservation())
        self.assertEqual(len(self.executor.calls), 1)

    async def test_retryable_feedback_mismatch_then_correct_feedback_updates_once(self):
        completed = await self._executed()
        with self.assertRaises(ValueError):
            await self.cycle.record_outcome(
                completed,
                reservation_ref=completed.turn.reservation.reservation_ref,
                task_id="wrong-task",
                outcome="success",
                feedback_text="Wrong identity.",
                feedback_source_ref=_ref("verifier"),
            )
        result = await self.cycle.record_outcome(
            completed,
            reservation_ref=completed.turn.reservation.reservation_ref,
            task_id="task-1",
            outcome="success",
            feedback_text="Correct identity.",
            feedback_source_ref=_ref("verifier"),
        )
        self.assertEqual(self.learner.updates, 1)
        self.assertEqual(result.sequence, 1)

    async def test_begin_failures_restore_exact_parent_and_executor_failure_stays_claimed(self):
        stages = []

        async def failing_recall(*args, **kwargs):
            raise RuntimeError("recall failure")

        real_recall = self.memory.recall
        self.memory.recall = failing_recall
        stages.append((self.cycle, "recall failure"))
        with self.assertRaisesRegex(RuntimeError, "recall failure"):
            await self.cycle.begin_turn("recall", "Task")
        self.memory.recall = real_recall

        self.cycle.text_adapter.propose_procedure_traces = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("text failure")
        )
        with self.assertRaisesRegex(RuntimeError, "text failure"):
            await self.cycle.begin_turn("text", "Task")
        self.cycle.text_adapter = TextAdapter()

        self.cycle.relation_adapter.encode_candidates = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("relation failure")
        )
        with self.assertRaisesRegex(RuntimeError, "relation failure"):
            await self.cycle.begin_turn("relation", "Task")
        self.cycle.relation_adapter = RelationAdapter(self.evidence_ref)

        real_select = self.learner.select
        self.learner.select = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("select failure"))
        with self.assertRaisesRegex(RuntimeError, "select failure"):
            await self.cycle.begin_turn("select", "Task")
        self.learner.select = real_select

        class RaisingExecutor(Executor):
            def execute(self, *args):
                raise RuntimeError("executor failure")

        self.cycle.executor = RaisingExecutor()
        turn = await self.cycle.begin_turn("execute", "Task")
        with self.assertRaises(ExecutionReconciliationRequired):
            await self.cycle.execute_turn(turn)
        self.assertEqual(self.learner.capture_state(), b"0")
        self.assertEqual(self.store.head().sequence, 0)
        self.assertEqual(self.store.active_reservation().status, "CLAIMED")
        self.assertIs(self.cycle.pending_turn, turn)

    async def test_invalid_proposals_or_fabricated_recall_citations_fail_closed(self):
        self.cycle.text_adapter.proposals = ("same", "same", "three", "four")
        with self.assertRaisesRegex(ValueError, "requested number of distinct"):
            await self.cycle.begin_turn("bad-proposals", "Task")
        self.assertIsNone(self.learner.pending_decision)

        bad = CognitiveCycle(
            text_adapter=TextAdapter(),
            relation_adapter=RelationAdapter(_ref("not recalled")),
            learner=self.learner,
            memory=self.memory,
            executor=self.executor,
            transaction_store=self.store,
            model_ref=_ref("frozen-model"),
            encoder_ref=_ref("v13-encoder"),
            agent_ref=_ref("angler-agent"),
            world_ref=_ref("actual-world"),
        )
        with self.assertRaisesRegex(ValueError, "outside validated recall"):
            await bad.begin_turn("bad-citation", "Task")
        self.assertIsNone(self.learner.pending_decision)

    async def test_observations_reject_scalar_text_before_claim(self):
        turn = await self.cycle.begin_turn("observations", "Keep items structurally distinct")
        with self.assertRaisesRegex(TypeError, "complete text items"):
            await self.cycle.execute_turn(turn, observations="not a collection")
        self.assertEqual(self.store.active_reservation().status, "RESERVED")
        completed = await self.cycle.execute_turn(turn, observations=("one item",))
        self.assertEqual(completed.input_observations, ("one item",))


if __name__ == "__main__":
    unittest.main()
