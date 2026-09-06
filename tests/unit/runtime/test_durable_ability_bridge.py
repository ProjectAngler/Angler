from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from angler.memory import MemoryHit, MovingOriginIndex, SituatedMemory
from angler.reasoning.prospective_dynamics import (
    CompositeProspectiveCreditCore,
    CompositeProspectiveCreditEvent,
    ProspectiveDynamicsConfig,
    ProspectiveResourceBudget,
)
from angler.reasoning.structure_keyed_credit_memory import (
    StructureKeyedCreditMemoryCore,
)
from angler.runtime.autonomous_apprentice import AtomicCompetenceStore, OutcomeContext
from angler.runtime.durable_ability_bridge import (
    DurableAbilityBridge,
    DurableAbilityLearner,
    EncodedProcedureCandidate,
    FrozenProcedureTextAdapter,
    PendingProspectiveMaterial,
    ProcedureRelationAdapter,
)
from angler.runtime import durable_ability_bridge as ability_module
from angler.runtime.situated_conversation import ConversationJournal


class _Backend:
    def __init__(self) -> None:
        self.documents: list[str] = []

    async def remember(self, document: str) -> str:
        self.documents.append(document)
        return str(len(self.documents))

    async def remember_many(self, documents) -> tuple[str, ...]:
        self.documents.extend(documents)
        return tuple(str(index + 1) for index in range(len(documents)))

    async def search(self, query: str, *, limit: int):
        del query
        return tuple(MemoryHit(item, score=1.0) for item in self.documents[-limit:])

    async def forget_dataset(self) -> None:
        self.documents.clear()


class _TextAdapter:
    proposals = (
        "Step 1: inspect the declared input.",
        "Step 1: stage the proposed change.",
        "Step 1: verify the changed artifact.",
        "Step 1: preserve a rollback point.",
    )

    def __init__(self) -> None:
        self.proposal_calls = []
        self.execution_calls = []

    def propose_procedure_traces(self, task, recalled_evidence, *, count=4):
        self.proposal_calls.append((task, recalled_evidence, count))
        return _proposals(count)

    def execute_with_procedure(self, task, selected_trace, observations):
        self.execution_calls.append((task, selected_trace, observations))
        return f"executed: {selected_trace}"


class _RelationAdapter:
    def __init__(self, temporal_width: int) -> None:
        self.temporal_width = temporal_width
        self.calls = []

    def encode_candidates(self, task, proposals, recall):
        self.calls.append((task, proposals, recall))
        refs = tuple(item.artifact_ref for item in recall.items)
        return tuple(
            EncodedProcedureCandidate(
                trace=trace,
                relation_features=torch.linspace(0.0, 1.0, 64)
                + float(index) / 10.0,
                temporal_features=torch.tensor(
                    [float(index + offset) / 10.0 for offset in range(self.temporal_width)]
                ),
                base_logit=torch.tensor(float(index)),
                supporting_evidence_refs=refs,
            )
            for index, trace in enumerate(proposals)
        )


class _FailingStore(AtomicCompetenceStore):
    def save(self, state: bytes) -> str:
        del state
        raise OSError("injected durable-state failure")


class _FabricatedEvidenceAdapter(_RelationAdapter):
    def encode_candidates(self, task, proposals, recall):
        rows = super().encode_candidates(task, proposals, recall)
        return tuple(
            EncodedProcedureCandidate(
                row.trace,
                row.relation_features,
                row.temporal_features,
                row.base_logit,
                ("sha256:fabricated",),
            )
            for row in rows
        )


def _core(temporal_width: int = 3) -> StructureKeyedCreditMemoryCore:
    torch.manual_seed(20260831)
    return StructureKeyedCreditMemoryCore(temporal_width=temporal_width)


def _learner(core: StructureKeyedCreditMemoryCore) -> DurableAbilityLearner:
    return DurableAbilityLearner(
        core,
        core.initial_state(),
        checkpoint_identity="sha256:" + "a" * 64,
    )


def _prospective_core(
    temporal_width: int = 5,
    *,
    seed: int = 20260831,
) -> CompositeProspectiveCreditCore:
    torch.manual_seed(seed)
    credit = StructureKeyedCreditMemoryCore(temporal_width=temporal_width)
    return CompositeProspectiveCreditCore(
        credit,
        ProspectiveDynamicsConfig(temporal_width=temporal_width),
    )


def _prospective_learner(
    core: CompositeProspectiveCreditCore,
    *,
    checkpoint_identity: str | None = None,
    prospective_lesion: bool = False,
) -> DurableAbilityLearner:
    return DurableAbilityLearner(
        core,
        core.initial_state(),
        checkpoint_identity=(
            core.checkpoint_identity
            if checkpoint_identity is None
            else checkpoint_identity
        ),
        prospective_lesion=prospective_lesion,
    )


def _proposals(count: int = 4) -> tuple[str, ...]:
    if count <= len(_TextAdapter.proposals):
        return _TextAdapter.proposals[:count]
    return (
        *_TextAdapter.proposals,
        *(f"Step 1: inspect candidate pathway {index}." for index in range(4, count)),
    )


def _encoded(temporal_width: int = 3, count: int = 4):
    adapter = _RelationAdapter(temporal_width)
    return adapter.encode_candidates("repair task", _proposals(count), _empty_recall())


def _sha_ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _prospective_encoded(temporal_width: int = 5, count: int = 4):
    recalled = tuple(sorted((_sha_ref("recall-a"), _sha_ref("recall-b"))))
    rows = tuple(
        EncodedProcedureCandidate(
            row.trace,
            row.relation_features,
            row.temporal_features,
            row.base_logit,
            (recalled[index % len(recalled)],),
        )
        for index, row in enumerate(_encoded(temporal_width, count))
    )
    return rows, recalled


def _empty_recall():
    from angler.memory import RecallBatch

    return RecallBatch(())


def _pending_payload(value: bytes) -> dict:
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


def _encode_pending_payload(payload: dict) -> bytes:
    body = {
        key: payload[key]
        for key in (
            "version",
            "checkpoint_identity",
            "parent_state_digest",
            "public_decision",
            "selected_output",
        )
    }
    payload["content_sha256"] = ability_module._pending_content_digest(body)
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _encode_prospective_pending_payload(payload: dict) -> bytes:
    body = {
        key: payload[key]
        for key in (
            "version",
            "checkpoint_identity",
            "parent_state_digest",
            "public_decision",
            "selected_output",
            "prospective_material",
        )
    }
    payload["content_sha256"] = ability_module._pending_prospective_content_digest(body)
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


class DurableAbilityLearnerTests(unittest.TestCase):
    def test_protocols_are_model_agnostic_and_selection_is_label_free(self) -> None:
        text = _TextAdapter()
        relation = _RelationAdapter(3)
        self.assertIsInstance(text, FrozenProcedureTextAdapter)
        self.assertIsInstance(relation, ProcedureRelationAdapter)

        learner = _learner(_core())
        decision = learner.select(
            task_id="task-a",
            request="repair task",
            candidates=_encoded(),
        )
        self.assertEqual(decision.selected_index, 3)
        self.assertEqual(decision.selected_trace, _TextAdapter.proposals[3])
        self.assertEqual(len(decision.logits), 4)
        self.assertTrue(decision.evidence_ref.startswith("sha256:"))
        self.assertTrue(all(not parameter.requires_grad for parameter in learner.core.parameters()))
        self.assertIsNotNone(learner.pending_decision)

    def test_real_learner_batches_two_six_and_sixty_four_candidates(self) -> None:
        for count in (2, 6, 64):
            with self.subTest(count=count):
                core = _core()
                learner = _learner(core)
                decision = learner.select(
                    task_id=f"task-{count}",
                    request="repair task",
                    candidates=_encoded(count=count),
                )
                self.assertEqual(len(decision.proposals), count)
                self.assertEqual(len(decision.logits), count)
                self.assertTrue(0 <= decision.selected_index < count)
                pending = learner.capture_pending_state()
                restored = _learner(_core())
                self.assertEqual(restored.restore_pending_state(pending), decision)
                outcome = OutcomeContext(
                    f"task-{count}", "repair task", (), "success", "verifier"
                )
                learner.apply_outcome(outcome)
                restored.apply_outcome(outcome)
                self.assertEqual(restored.state_digest(), learner.state_digest())

    def test_objective_feedback_snapshot_restore_and_derangement_separate_values(self) -> None:
        core = _core()
        first = _learner(core)
        second = _learner(core)
        candidates = _encoded()
        first.select(task_id="task", request="repair task", candidates=candidates)
        second.select(task_id="task", request="repair task", candidates=candidates)
        first.apply_outcome(OutcomeContext("task", "repair task", (), "success", "verifier"))
        second.apply_outcome(OutcomeContext("task", "repair task", (), "failure", "verifier"))

        self.assertTrue(torch.equal(first.state.keys, second.state.keys))
        self.assertTrue(torch.equal(first.state.usage, second.state.usage))
        self.assertTrue(torch.equal(first.state.acquisition, second.state.acquisition))
        self.assertFalse(torch.equal(first.state.values, second.state.values))
        self.assertEqual(first.state.step, 1)
        self.assertEqual(len(first.events), 1)

        snapshot = first.capture_state()
        digest = first.state_digest()
        restored = _learner(core)
        restored.restore_state(snapshot)
        self.assertEqual(restored.state_digest(), digest)
        self.assertTrue(torch.equal(restored.state.values, first.state.values))
        self.assertEqual(restored.events[0].to_record(), first.events[0].to_record())

        damaged = bytearray(snapshot)
        damaged[len(damaged) // 2] ^= 0xFF
        with self.assertRaises(ValueError):
            restored.restore_state(bytes(damaged))

    def test_invalidation_removes_dependent_events_and_replay_is_exact(self) -> None:
        core = _core()
        learner = _learner(core)
        decision = learner.select(
            task_id="task",
            request="repair task",
            candidates=_encoded(),
        )
        learner.apply_outcome(
            OutcomeContext("task", "repair task", (), "success", "verifier")
        )
        event = learner.events[0]
        learned_digest = learner.state_digest()

        clean = _learner(core)
        self.assertEqual(clean.rebuild_from_events((event,)), learned_digest)
        invalidated = learner.invalidate_evidence((decision.evidence_ref,))
        self.assertEqual(learner.state.step, 0)
        self.assertEqual(learner.events, ())
        self.assertIn(decision.evidence_ref, learner.tombstoned_evidence_refs)
        self.assertEqual(invalidated, learner.state_digest())
        with self.assertRaisesRegex(ValueError, "tombstoned"):
            learner.rebuild_from_events((event,))

    def test_pending_decision_round_trip_restores_exact_update_context(self) -> None:
        source = _learner(_core())
        decision = source.select(
            task_id="task-a",
            request="repair task",
            candidates=_encoded(),
        )
        pending = source.capture_pending_state()
        self.assertLessEqual(len(pending), 16 * 1024 * 1024)
        self.assertEqual(source.capture_pending_state(), pending)

        restored = _learner(_core())
        self.assertEqual(restored.restore_pending_state(pending), decision)
        self.assertEqual(restored.pending_decision, decision)
        outcome = OutcomeContext("task-a", "repair task", (), "success", "verifier")
        source.apply_outcome(outcome)
        restored.apply_outcome(outcome)
        self.assertEqual(restored.state_digest(), source.state_digest())
        self.assertEqual(restored.events[0].to_record(), source.events[0].to_record())

    def test_pending_state_requires_exact_checkpoint_parent_and_open_slot(self) -> None:
        source = _learner(_core())
        with self.assertRaisesRegex(RuntimeError, "no ability decision"):
            source.capture_pending_state()
        source.select(task_id="task-a", request="repair task", candidates=_encoded())
        pending = source.capture_pending_state()
        with self.assertRaisesRegex(RuntimeError, "already awaiting"):
            source.restore_pending_state(pending)

        wrong_checkpoint = DurableAbilityLearner(
            _core(),
            _core().initial_state(),
            checkpoint_identity="sha256:" + "b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            wrong_checkpoint.restore_pending_state(pending)

        wrong_parent = _learner(_core())
        wrong_parent.select(
            task_id="other", request="repair task", candidates=_encoded()
        )
        wrong_parent.apply_outcome(
            OutcomeContext("other", "repair task", (), "success", "verifier")
        )
        with self.assertRaisesRegex(ValueError, "another parent state"):
            wrong_parent.restore_pending_state(pending)
        with self.assertRaisesRegex(ValueError, "16 MiB"):
            _learner(_core()).restore_pending_state(b"x" * (16 * 1024 * 1024 + 1))

    def test_pending_state_rejects_public_decision_and_tensor_tampering(self) -> None:
        source = _learner(_core())
        source.select(task_id="task-a", request="repair task", candidates=_encoded())
        original = source.capture_pending_state()

        selected_trace = _pending_payload(original)
        selected_trace["public_decision"]["selected_trace"] = "not the selected proposal"
        with self.assertRaisesRegex(ValueError, "selected trace"):
            _learner(_core()).restore_pending_state(
                _encode_pending_payload(selected_trace)
            )

        wrong_shape = _pending_payload(original)
        wrong_shape["selected_output"]["tensors"]["read_queries"] = torch.zeros(1, 31)
        wrong_shape["selected_output"]["tensor_specs"]["read_queries"]["shape"] = [1, 31]
        with self.assertRaisesRegex(ValueError, "neural output"):
            _learner(_core()).restore_pending_state(_encode_pending_payload(wrong_shape))

        wrong_dtype = _pending_payload(original)
        wrong_dtype["selected_output"]["tensors"]["read_queries"] = wrong_dtype[
            "selected_output"
        ]["tensors"]["read_queries"].to(torch.float64)
        wrong_dtype["selected_output"]["tensor_specs"]["read_queries"][
            "dtype"
        ] = "torch.float64"
        with self.assertRaisesRegex(ValueError, "dtype/shape/value"):
            _learner(_core()).restore_pending_state(_encode_pending_payload(wrong_dtype))

        nonfinite = _pending_payload(original)
        nonfinite["selected_output"]["tensors"]["write_keys"][0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "dtype/shape/value"):
            _learner(_core()).restore_pending_state(_encode_pending_payload(nonfinite))

        wrong_value = _pending_payload(original)
        wrong_value["selected_output"]["tensors"]["write_keys"][0, 0] += 0.125
        with self.assertRaisesRegex(ValueError, "frozen parent prediction"):
            _learner(_core()).restore_pending_state(_encode_pending_payload(wrong_value))

    def test_pending_state_rejects_corruption_and_unsafe_object_payloads(self) -> None:
        source = _learner(_core())
        source.select(task_id="task-a", request="repair task", candidates=_encoded())
        encoded = source.capture_pending_state()
        damaged = bytearray(encoded)
        damaged[len(damaged) // 2] ^= 0xFF
        with self.assertRaises(ValueError):
            _learner(_core()).restore_pending_state(bytes(damaged))

        unsafe = _pending_payload(encoded)
        unsafe["selected_output"]["unsafe_path"] = Path("not-a-primitive")
        stream = io.BytesIO()
        torch.save(unsafe, stream)
        with self.assertRaisesRegex(ValueError, "decoded safely"):
            _learner(_core()).restore_pending_state(stream.getvalue())

    def test_composite_pending_snapshot_event_and_replay_are_exact(self) -> None:
        core = _prospective_core()
        source = _prospective_learner(core)
        decision = source.select(
            task_id="prospective-task",
            request="repair task",
            candidates=_encoded(temporal_width=5, count=7),
        )
        pending = source.capture_pending_state()
        self.assertLessEqual(len(pending), 16 * 1024 * 1024)

        restored = _prospective_learner(_prospective_core())
        self.assertEqual(restored.restore_pending_state(pending), decision)
        outcome = OutcomeContext(
            "prospective-task", "repair task", (), "success", "verifier"
        )
        source.apply_outcome(outcome)
        restored.apply_outcome(outcome)
        self.assertEqual(restored.state_digest(), source.state_digest())
        self.assertIsInstance(source.events[0], CompositeProspectiveCreditEvent)
        self.assertEqual(
            restored.events[0].to_record(),
            source.events[0].to_record(),
        )

        snapshot = source.capture_state()
        restarted = _prospective_learner(_prospective_core())
        restarted.restore_state(snapshot)
        self.assertEqual(restarted.state_digest(), source.state_digest())

        context = ability_module._event_context(source.events[0])
        self.assertLessEqual(int(context["ability_event_chunks"]), 8)
        parsed = ability_module._event_from_context(context, core=restarted.core)
        self.assertEqual(parsed.to_record(), source.events[0].to_record())
        rebuilt = _prospective_learner(_prospective_core())
        self.assertEqual(
            rebuilt.rebuild_from_events((parsed,)),
            source.state_digest(),
        )

        wrong_core = _prospective_core(seed=20260830)
        wrong = _prospective_learner(
            wrong_core,
            checkpoint_identity=core.checkpoint_identity,
        )
        with self.assertRaisesRegex(ValueError, "snapshot content"):
            wrong.restore_state(snapshot)

    def test_prospective_material_covers_variable_batches_binds_and_restores(self) -> None:
        for count in (2, 7, 64):
            with self.subTest(count=count):
                core = _prospective_core()
                source = _prospective_learner(core)
                rows, recalled = _prospective_encoded(count=count)
                budget = ProspectiveResourceBudget(1, count, 1)
                decision = source.select_prospective(
                    task_id=f"prospective-{count}",
                    request="repair task",
                    candidates=rows,
                    recalled_refs=recalled,
                    context_refs=(_sha_ref("situated-context"),),
                    eligibility_rows=((0,),) * count,
                    resource_budget=budget,
                )
                material = source.pending_prospective_material
                self.assertIsInstance(material, PendingProspectiveMaterial)
                material = material  # type: ignore[assignment]
                self.assertIsNone(material.batch_ref)
                self.assertEqual(material.candidate_traces, decision.proposals)
                self.assertEqual(material.resource_budget, budget)
                self.assertEqual(material.relation_features.shape, (count, 64))
                self.assertEqual(material.temporal_features.shape, (count, 5))
                self.assertEqual(
                    material.prospective_output.future_latents.shape,
                    (count, core.config.latent_width),
                )
                selected = decision.selected_index
                with torch.no_grad():
                    _combined, exact = core.predict_with_prospective(
                        material.relation_features[selected : selected + 1],
                        material.temporal_features[selected : selected + 1],
                        material.base_logits[selected : selected + 1],
                        state=source.state,
                    )
                self.assertTrue(
                    torch.equal(
                        material.prospective_output.future_latents[
                            selected : selected + 1
                        ],
                        exact.future_latents,
                    )
                )
                with self.assertRaisesRegex(ValueError, "batch binding"):
                    source.capture_pending_state()

                batch_ref = _sha_ref(f"batch-{count}")
                bound = source.bind_pending_prospective_batch(batch_ref)
                self.assertEqual(bound.batch_ref, batch_ref)
                bound.relation_features[0, 0] += 99.0
                self.assertNotEqual(
                    source.pending_prospective_material.relation_features[0, 0].item(),  # type: ignore[union-attr]
                    bound.relation_features[0, 0].item(),
                )
                pending = source.capture_pending_state()
                self.assertLessEqual(len(pending), 16 * 1024 * 1024)
                self.assertEqual(source.capture_pending_state(), pending)

                restored = _prospective_learner(_prospective_core())
                self.assertEqual(restored.restore_pending_state(pending), decision)
                restored_material = restored.pending_prospective_material
                self.assertEqual(restored_material.batch_ref, batch_ref)  # type: ignore[union-attr]
                self.assertTrue(
                    torch.equal(
                        restored_material.prospective_output.future_latents,  # type: ignore[union-attr]
                        source.pending_prospective_material.prospective_output.future_latents,  # type: ignore[union-attr]
                    )
                )
                outcome = OutcomeContext(
                    f"prospective-{count}",
                    "repair task",
                    (),
                    "success",
                    "verifier",
                )
                source.apply_outcome(outcome)
                restored.apply_outcome(outcome)
                self.assertEqual(restored.state_digest(), source.state_digest())

    def test_prospective_pending_rejects_coherently_rehashed_material_tamper(self) -> None:
        source = _prospective_learner(_prospective_core())
        rows, recalled = _prospective_encoded(count=7)
        source.select_prospective(
            task_id="prospective-tamper",
            request="repair task",
            candidates=rows,
            recalled_refs=recalled,
            context_refs=(_sha_ref("situated-context"),),
            eligibility_rows=((0,),) * 7,
            resource_budget=ProspectiveResourceBudget(1, 7, 1),
        )
        source.bind_pending_prospective_batch(_sha_ref("tamper-batch"))
        original = source.capture_pending_state()
        cases = (
            "candidate",
            "evidence",
            "eligibility",
            "budget",
            "input",
            "prediction",
            "parent",
            "component",
        )
        for case in cases:
            with self.subTest(case=case):
                payload = _pending_payload(original)
                material = payload["prospective_material"]
                if case == "candidate":
                    material["candidate_traces"][0] = "different candidate trace"
                elif case == "evidence":
                    material["candidate_supporting_evidence_refs"][0] = [
                        _sha_ref("fabricated")
                    ]
                elif case == "eligibility":
                    material["eligibility_rows"][0] = [1]
                elif case == "budget":
                    material["resource_budget"]["branches"] = 6
                elif case == "input":
                    material["candidate_inputs"]["tensors"]["relation_features"][
                        0, 0
                    ] += 0.125
                elif case == "prediction":
                    material["prospective_output"]["tensors"]["future_latents"][
                        0, 0
                    ] += 0.125
                elif case == "parent":
                    material["parent_competence_digest"] = _sha_ref("wrong-parent")
                else:
                    material["component_state"]["world_state_digest"] = _sha_ref(
                        "wrong-world-state"
                    )
                encoded = _encode_prospective_pending_payload(payload)
                restored = _prospective_learner(_prospective_core())
                with self.assertRaises(ValueError):
                    restored.restore_pending_state(encoded)
                self.assertIsNone(restored.pending_decision)

    def test_fixed_prospective_lesion_preserves_inputs_and_replays(self) -> None:
        normal = _prospective_learner(_prospective_core())
        lesion = _prospective_learner(
            _prospective_core(), prospective_lesion=True
        )
        for learner in (normal, lesion):
            learner.select(
                task_id="adapt-parent",
                request="repair task",
                candidates=_encoded(temporal_width=5, count=7),
            )
            learner.apply_outcome(
                OutcomeContext(
                    "adapt-parent", "repair task", (), "success", "verifier"
                )
            )
        self.assertEqual(normal.state_digest(), lesion.state_digest())
        parent_snapshot = lesion.capture_state()

        rows, recalled = _prospective_encoded(count=7)
        arguments = {
            "task_id": "lesion-turn",
            "request": "repair task",
            "candidates": rows,
            "recalled_refs": recalled,
            "context_refs": (_sha_ref("situated-context"),),
            "eligibility_rows": ((0,),) * 7,
            "resource_budget": ProspectiveResourceBudget(1, 7, 1),
        }
        normal_decision = normal.select_prospective(**arguments)
        lesion_decision = lesion.select_prospective(**arguments)
        normal_material = normal.pending_prospective_material
        lesion_material = lesion.pending_prospective_material
        self.assertTrue(normal_material.prospective_read_enabled)  # type: ignore[union-attr]
        self.assertFalse(lesion_material.prospective_read_enabled)  # type: ignore[union-attr]
        for name in ("relation_features", "temporal_features", "base_logits"):
            self.assertTrue(
                torch.equal(
                    getattr(normal_material, name),  # type: ignore[arg-type]
                    getattr(lesion_material, name),  # type: ignore[arg-type]
                )
            )
        self.assertEqual(normal_material.context_refs, lesion_material.context_refs)  # type: ignore[union-attr]
        self.assertEqual(normal_material.eligibility_rows, lesion_material.eligibility_rows)  # type: ignore[union-attr]
        self.assertEqual(normal_material.resource_budget, lesion_material.resource_budget)  # type: ignore[union-attr]
        credit = lesion.core.credit_core.predict(
            lesion_material.relation_features,  # type: ignore[union-attr]
            lesion_material.temporal_features,  # type: ignore[union-attr]
            lesion_material.base_logits,  # type: ignore[union-attr]
            state=lesion.state.credit_state,
        )
        credit_logits = [float(value) for value in credit.logits.tolist()]
        selected = lesion_decision.selected_index
        exact_credit = lesion.core.credit_core.predict(
            lesion_material.relation_features[selected : selected + 1],  # type: ignore[union-attr]
            lesion_material.temporal_features[selected : selected + 1],  # type: ignore[union-attr]
            lesion_material.base_logits[selected : selected + 1],  # type: ignore[union-attr]
            state=lesion.state.credit_state,
        )
        credit_logits[selected] = float(exact_credit.logits.item())
        self.assertEqual(lesion_decision.logits, tuple(credit_logits))
        self.assertTrue(
            bool(
                (
                    normal_material.prospective_output.selection_residuals != 0  # type: ignore[union-attr]
                ).any().item()
            )
        )

        lesion.bind_pending_prospective_batch(_sha_ref("lesion-batch"))
        pending = lesion.capture_pending_state()
        wrong_mode = _prospective_learner(_prospective_core())
        wrong_mode.restore_state(parent_snapshot)
        with self.assertRaisesRegex(ValueError, "removal condition"):
            wrong_mode.restore_pending_state(pending)
        outcome = OutcomeContext(
            "lesion-turn", "repair task", (), "success", "verifier"
        )
        lesion.apply_outcome(outcome)
        snapshot = lesion.capture_state()
        replayed = _prospective_learner(
            _prospective_core(), prospective_lesion=True
        )
        replayed.restore_state(snapshot)
        self.assertEqual(replayed.state_digest(), lesion.state_digest())

    def test_composite_child_failures_preserve_parent_and_pending(self) -> None:
        for target_name, method_name in (
            ("prospective", "_loss_directed_hypothesis_update"),
            ("credit", "apply_feedback"),
        ):
            with self.subTest(child=target_name):
                core = _prospective_core()
                learner = _prospective_learner(core)
                decision = learner.select(
                    task_id=f"failure-{target_name}",
                    request="repair task",
                    candidates=_encoded(temporal_width=5, count=7),
                )
                parent = learner.state_digest()
                pending = learner.capture_pending_state()
                target = core if target_name == "prospective" else core.credit_core
                with mock.patch.object(
                    target,
                    method_name,
                    side_effect=RuntimeError(f"injected {target_name} failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        learner.apply_outcome(
                            OutcomeContext(
                                f"failure-{target_name}",
                                "repair task",
                                (),
                                "success",
                                "verifier",
                            )
                        )
                self.assertEqual(learner.state_digest(), parent)
                self.assertEqual(learner.capture_pending_state(), pending)
                self.assertEqual(learner.pending_decision, decision)
                self.assertEqual(learner.events, ())


class DurableAbilityBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_bridge_uses_bounded_variable_proposal_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = _core()
            text = _TextAdapter()
            bridge = DurableAbilityBridge(
                text_adapter=text,
                relation_adapter=_RelationAdapter(core.temporal_width),
                learner=_learner(core),
                memory=SituatedMemory(_Backend()),
                journal=ConversationJournal(root / "journal.jsonl"),
                state_store=AtomicCompetenceStore(root / "ability.pt"),
                proposal_count=6,
            )
            turn = await bridge.begin_turn("six", "repair task")
            self.assertEqual(text.proposal_calls[0][2], 6)
            self.assertEqual(len(turn.decision.proposals), 6)

            for invalid in (1, 65, True):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    DurableAbilityBridge(
                        text_adapter=_TextAdapter(),
                        relation_adapter=_RelationAdapter(core.temporal_width),
                        learner=_learner(core),
                        memory=SituatedMemory(_Backend()),
                        journal=ConversationJournal(root / f"invalid-{invalid}.jsonl"),
                        state_store=AtomicCompetenceStore(root / f"invalid-{invalid}.pt"),
                        proposal_count=invalid,
                    )

    async def test_relation_adapter_cannot_fabricate_evidence_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = _core()
            bridge = DurableAbilityBridge(
                text_adapter=_TextAdapter(),
                relation_adapter=_FabricatedEvidenceAdapter(core.temporal_width),
                learner=_learner(core),
                memory=SituatedMemory(_Backend()),
                journal=ConversationJournal(root / "journal.jsonl"),
                state_store=AtomicCompetenceStore(root / "ability.pt"),
            )
            with self.assertRaisesRegex(ValueError, "outside the validated recall"):
                await bridge.begin_turn("task-a", "repair task")

    async def test_turn_commit_restart_and_journal_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _Backend()
            memory = SituatedMemory(backend, origin=MovingOriginIndex())
            journal = ConversationJournal(root / "journal.jsonl")
            core = _core()
            learner = _learner(core)
            text = _TextAdapter()
            bridge = DurableAbilityBridge(
                text_adapter=text,
                relation_adapter=_RelationAdapter(core.temporal_width),
                learner=learner,
                memory=memory,
                journal=journal,
                state_store=AtomicCompetenceStore(root / "ability.pt"),
            )

            turn = await bridge.begin_turn("task-a", "repair task")
            completed = bridge.execute_turn(turn, observations=("workspace is ready",))
            record = await bridge.record_outcome(
                completed,
                outcome="success",
                source_ref="sha256:" + "b" * 64,
                feedback_text="the objective verifier passed",
            )
            self.assertEqual(record.projection.artifact_ref, turn.decision.evidence_ref)
            self.assertGreaterEqual(memory.origin.size, 1)
            self.assertEqual(
                memory.origin.anchor(record.projection.artifact_ref).ordinal,
                0,
            )
            self.assertEqual(len(backend.documents), 1)
            self.assertEqual(len(journal.records()), 1)
            self.assertTrue((root / "ability.pt").is_file())
            committed_digest = learner.state_digest()

            restarted = _learner(core)
            restarted.restore_state((root / "ability.pt").read_bytes())
            self.assertEqual(restarted.state_digest(), committed_digest)
            bridge.learner.zero()
            self.assertEqual(bridge.learner.state.step, 0)
            self.assertEqual(bridge.rebuild_ability_from_journal(), committed_digest)

    async def test_store_failure_restores_state_but_preserves_replayable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _Backend()
            memory = SituatedMemory(backend)
            journal = ConversationJournal(root / "journal.jsonl")
            core = _core()
            learner = _learner(core)
            bridge = DurableAbilityBridge(
                text_adapter=_TextAdapter(),
                relation_adapter=_RelationAdapter(core.temporal_width),
                learner=learner,
                memory=memory,
                journal=journal,
                state_store=_FailingStore(root / "ability.pt"),
            )
            parent = learner.state_digest()
            turn = await bridge.begin_turn("task-a", "repair task")
            completed = bridge.execute_turn(turn)
            with self.assertRaisesRegex(OSError, "injected"):
                await bridge.record_outcome(
                    completed,
                    outcome="failure",
                    source_ref="verifier:1",
                    feedback_text="the verifier rejected the result",
                )
            self.assertEqual(learner.state_digest(), parent)
            self.assertEqual(learner.state.step, 0)
            self.assertEqual(len(journal.records()), 1)
            rebuilt = bridge.rebuild_ability_from_journal()
            self.assertEqual(learner.state.step, 1)
            self.assertEqual(rebuilt, learner.state_digest())

    async def test_composite_store_failure_restores_and_replays_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = _prospective_core()
            learner = _prospective_learner(core)
            bridge = DurableAbilityBridge(
                text_adapter=_TextAdapter(),
                relation_adapter=_RelationAdapter(core.temporal_width),
                learner=learner,
                memory=SituatedMemory(_Backend()),
                journal=ConversationJournal(root / "journal.jsonl"),
                state_store=_FailingStore(root / "ability.pt"),
                proposal_count=7,
            )
            parent = learner.state_digest()
            turn = await bridge.begin_turn("prospective-task", "repair task")
            with self.assertRaisesRegex(OSError, "injected"):
                await bridge.record_outcome(
                    bridge.execute_turn(turn),
                    outcome="failure",
                    source_ref="verifier:prospective",
                    feedback_text="the verifier rejected the result",
                )
            self.assertEqual(learner.state_digest(), parent)
            self.assertEqual(learner.state.step, 0)
            self.assertEqual(len(bridge.journal.records()), 1)

            rebuilt = bridge.rebuild_ability_from_journal()
            self.assertEqual(learner.state.step, 1)
            self.assertEqual(rebuilt, learner.state_digest())
            self.assertIsInstance(learner.events[0], CompositeProspectiveCreditEvent)

    async def test_tombstone_rebuild_excludes_cognee_origin_and_neural_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _Backend()
            memory = SituatedMemory(backend)
            journal = ConversationJournal(root / "journal.jsonl")
            core = _core()
            learner = _learner(core)
            bridge = DurableAbilityBridge(
                text_adapter=_TextAdapter(),
                relation_adapter=_RelationAdapter(core.temporal_width),
                learner=learner,
                memory=memory,
                journal=journal,
                state_store=AtomicCompetenceStore(root / "ability.pt"),
            )
            turn = await bridge.begin_turn("task-a", "repair task")
            await bridge.record_outcome(
                bridge.execute_turn(turn),
                outcome="success",
                source_ref="verifier:1",
                feedback_text="passed",
            )
            learner.invalidate_evidence((turn.decision.evidence_ref,))
            await bridge.rebuild_projection()
            self.assertEqual(backend.documents, [])
            self.assertEqual(memory.origin.size, 0)
            bridge.rebuild_ability_from_journal()
            self.assertEqual(learner.state.step, 0)
            self.assertEqual(learner.events, ())


if __name__ == "__main__":
    unittest.main()
