from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import torch

from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
    ObjectiveFeedbackRecord,
)
from angler.episodes.canonical import canonical_bytes
from angler.memory.cognitive_acquisition_graph import (
    AcquisitionReferenceHit,
    AcquisitionSituatedMemory,
)
from angler.reasoning.prospective_dynamics import (
    CompositeProspectiveCreditCore,
    ProspectiveDynamicsConfig,
)
from angler.reasoning.structure_keyed_credit_memory import (
    StructureKeyedCreditMemoryCore,
)
from angler.runtime.cognitive_cycle import (
    CognitiveCycle,
    CognitiveExecution,
)
from angler.runtime.cognitive_transaction_store import CognitiveTransactionStore
from angler.runtime.durable_ability_bridge import (
    DurableAbilityLearner,
    EncodedProcedureCandidate,
)
from angler.runtime import prospective_observation
from angler.runtime.prospective_observation import (
    FrozenObservedStateEncoder,
    MAX_SYNTHETIC_OBSERVED_LATENT_WIDTH,
    ObservedStateEncoding,
    QwenReceiptObservedStateEncoderV1,
    SYNTHETIC_OBSERVED_STATE_ALGORITHM,
    SyntheticObservedStateEncoderV1,
)
from angler.runtime.qwen_cognitive import (
    FrozenQwenCognitiveExecutorV1,
    FrozenQwenCycleManifestV1,
    FrozenQwenProcedureAdapterV1,
    FrozenQwenProcedureRelationAdapterV1,
    SQLiteQwenExecutionJournal,
)
from angler.runtime.situated_qwen import LocalQwenIO


def _execution() -> tuple[CognitiveExecutionRequest, CognitiveExecutionReceipt]:
    reservation_ref = "sha256:" + "1" * 64
    request = CognitiveExecutionRequest(
        reservation_ref=reservation_ref,
        commitment_ref="sha256:" + "2" * 64,
        idempotency_key=reservation_ref,
        task_id="task-fixture",
        request="Observe this exact public request.",
        selected_trace="Use the selected public procedure.",
        input_observations=("input alpha", "input beta"),
    )
    receipt = CognitiveExecutionReceipt.from_request(
        request,
        status="COMPLETED",
        executed_trace=request.selected_trace,
        response="Exact public response.",
        output_observations=("output gamma",),
    )
    return request, receipt


class SyntheticObservedStateEncoderV1Tests(unittest.TestCase):
    def test_exact_identity_framing_dyadics_and_evidence(self) -> None:
        request, receipt = _execution()
        encoder = SyntheticObservedStateEncoderV1(latent_width=3)

        encoding = encoder.encode(request, receipt, latent_width=3)

        self.assertIsInstance(encoder, FrozenObservedStateEncoder)
        self.assertEqual(SYNTHETIC_OBSERVED_STATE_ALGORITHM, encoder.ALGORITHM)
        self.assertEqual(
            "sha256:3bfaf345f3ead48aec25084fd3e927085a7c4aa03736920df7d02487a0cfb6d9",
            encoder.encoder_ref,
        )
        self.assertEqual(
            tuple(
                sorted(
                    (
                        "sha256:c3ed7d4d0d3e875db4e8b950de7aecd5e1cc4664d8a61cb003a0ccfcbe59f081",
                        "sha256:96dab22a96200ec000f2a2aeaace5790869fa567fb4b4432d46cfdaffeebc92d",
                    )
                )
            ),
            encoding.evidence_refs,
        )
        self.assertEqual(
            (
                "0x1.7665207d3b076p-1",
                "-0x1.f8ba387542c9ep-1",
                "0x1.5d06fb2d0c39ap-1",
            ),
            tuple(value.hex() for value in encoding.latent),
        )
        self.assertIs(encoding.observed_next_latent, encoding.latent)
        self.assertIs(encoding.observation_evidence_refs, encoding.evidence_refs)
        self.assertTrue(all(math.isfinite(value) for value in encoding.latent))
        self.assertTrue(all(-1.0 <= value < 1.0 for value in encoding.latent))

        identity_bytes = canonical_bytes(
            {
                "algorithm": "angler.synthetic-observed-state.v1",
                "latent_width": 3,
            }
        )
        self.assertEqual(
            "sha256:" + hashlib.sha256(identity_bytes).hexdigest(),
            encoder.encoder_ref,
        )
        request_bytes = request.canonical_bytes()
        receipt_bytes = receipt.canonical_bytes()
        framed = b"".join(
            (
                b"angler.synthetic-observed-state.v1\0",
                struct.pack(">Q", 3),
                struct.pack(">Q", len(request_bytes)),
                request_bytes,
                struct.pack(">Q", len(receipt_bytes)),
                receipt_bytes,
            )
        )
        independent = []
        for index in range(3):
            digest = hashlib.sha256(framed + struct.pack(">Q", index)).digest()
            leading_u64 = int.from_bytes(digest[:8], "big", signed=False)
            top53 = leading_u64 >> 11
            independent.append((top53 / 2**52) - 1.0)
        self.assertEqual(tuple(independent), encoding.latent)

    def test_frozen_dyadic_boundary_values(self) -> None:
        self.assertEqual(
            "-0x1.0000000000000p+0",
            prospective_observation._dyadic_from_leading_u64(0).hex(),
        )
        self.assertEqual(
            "0x1.ffffffffffffep-1",
            prospective_observation._dyadic_from_leading_u64(2**64 - 1).hex(),
        )
        for invalid in (-1, 2**64, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    prospective_observation._dyadic_from_leading_u64(
                        invalid  # type: ignore[arg-type]
                    )

    def test_width_is_exact_bounded_and_bound_into_identity(self) -> None:
        request, receipt = _execution()
        for invalid in (0, -1, True, 1.0, MAX_SYNTHETIC_OBSERVED_LATENT_WIDTH + 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    SyntheticObservedStateEncoderV1(invalid)  # type: ignore[arg-type]

        for width in (1, MAX_SYNTHETIC_OBSERVED_LATENT_WIDTH):
            with self.subTest(width=width):
                encoder = SyntheticObservedStateEncoderV1(width)
                encoding = encoder.encode(request, receipt, latent_width=width)
                self.assertEqual(width, len(encoding.latent))
                self.assertEqual(width, encoding.latent_width)

        encoder = SyntheticObservedStateEncoderV1(3)
        with self.assertRaisesRegex(ValueError, "differs from the frozen encoder"):
            encoder.encode(request, receipt, latent_width=4)
        self.assertNotEqual(encoder.encoder_ref, SyntheticObservedStateEncoderV1(4).encoder_ref)

    def test_only_exact_completed_request_and_receipt_are_accepted(self) -> None:
        request, receipt = _execution()
        encoder = SyntheticObservedStateEncoderV1(3)

        for status in ("CLARIFICATION_REQUIRED", "ERROR"):
            with self.subTest(status=status):
                noncompleted = CognitiveExecutionReceipt.from_request(
                    request,
                    status=status,
                    executed_trace=request.selected_trace,
                    response="public non-completion",
                )
                with self.assertRaisesRegex(ValueError, "requires a completed receipt"):
                    encoder.encode(request, noncompleted, latent_width=3)

        other_request = CognitiveExecutionRequest(
            reservation_ref=request.reservation_ref,
            commitment_ref=request.commitment_ref,
            idempotency_key=request.idempotency_key,
            task_id=request.task_id,
            request=request.request + " Changed.",
            selected_trace=request.selected_trace,
            input_observations=request.input_observations,
        )
        mismatched_receipt = CognitiveExecutionReceipt.from_request(
            other_request,
            status="COMPLETED",
            executed_trace=other_request.selected_trace,
            response=receipt.response,
            output_observations=receipt.output_observations,
        )
        with self.assertRaisesRegex(ValueError, "does not match the exact request"):
            encoder.encode(request, mismatched_receipt, latent_width=3)
        with self.assertRaises(TypeError):
            encoder.encode(object(), receipt, latent_width=3)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            encoder.encode(request, object(), latent_width=3)  # type: ignore[arg-type]

    def test_surface_rejects_outcome_prediction_and_caller_evidence(self) -> None:
        request, receipt = _execution()
        encoder = SyntheticObservedStateEncoderV1(3)
        forbidden = (
            {"outcome": "success"},
            {"predicted_latent": (0.0, 0.0, 0.0)},
            {"evidence_refs": (request.execution_request_ref,)},
        )
        for values in forbidden:
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    encoder.encode(  # type: ignore[call-arg]
                        request,
                        receipt,
                        latent_width=3,
                        **values,
                    )

    def test_encoding_rejects_nonfinite_invented_evidence_and_identity_drift(self) -> None:
        request, receipt = _execution()
        encoder = SyntheticObservedStateEncoderV1(3)
        exact = encoder.encode(request, receipt, latent_width=3)

        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite exact floats"):
                    ObservedStateEncoding(
                        encoder_ref=exact.encoder_ref,
                        latent_width=3,
                        latent=(invalid, 0.0, 0.0),
                        evidence_refs=exact.evidence_refs,
                    )

        invented = ObservedStateEncoding(
            encoder_ref=exact.encoder_ref,
            latent_width=3,
            latent=exact.latent,
            evidence_refs=tuple(sorted(("sha256:" + "3" * 64, "sha256:" + "4" * 64))),
        )
        with self.assertRaisesRegex(ValueError, "evidence does not match"):
            invented.assert_context(
                request,
                receipt,
                encoder_ref=encoder.encoder_ref,
                latent_width=3,
            )
        with self.assertRaisesRegex(ValueError, "identity drifted"):
            exact.assert_context(
                request,
                receipt,
                encoder_ref="sha256:" + "5" * 64,
                latent_width=3,
            )
        with self.assertRaisesRegex(ValueError, "width drifted"):
            exact.assert_context(
                request,
                receipt,
                encoder_ref=encoder.encoder_ref,
                latent_width=4,
            )

    def test_request_or_receipt_byte_change_changes_encoding(self) -> None:
        request, receipt = _execution()
        encoder = SyntheticObservedStateEncoderV1(3)
        original = encoder.encode(request, receipt, latent_width=3)
        changed_receipt = CognitiveExecutionReceipt.from_request(
            request,
            status="COMPLETED",
            executed_trace=request.selected_trace,
            response=receipt.response + " Changed.",
            output_observations=receipt.output_observations,
        )
        changed = encoder.encode(request, changed_receipt, latent_width=3)
        self.assertNotEqual(original.latent, changed.latent)
        self.assertNotEqual(original.evidence_refs, changed.evidence_refs)

    def test_encoder_and_encoding_are_immutable(self) -> None:
        request, receipt = _execution()
        encoder = SyntheticObservedStateEncoderV1(3)
        encoding = encoder.encode(request, receipt, latent_width=3)
        with self.assertRaises(FrozenInstanceError):
            encoder.latent_width = 4  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            encoding.latent = (0.0, 0.0, 0.0)  # type: ignore[misc]


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


class _AcquisitionBackend:
    def __init__(self) -> None:
        self.projections = []
        self.fail_project = False
        self.queries = []

    async def project(self, projection):
        if self.fail_project:
            raise RuntimeError("injected acquisition projection failure")
        if all(
            item.projection_ref != projection.projection_ref
            for item in self.projections
        ):
            self.projections.append(projection)
        return f"backend-{len(self.projections)}"

    async def search(self, query: str, *, limit: int):
        self.queries.append((query, limit))
        return tuple(
            AcquisitionReferenceHit(
                record_ref=item.record.record_ref,
                score=1.0,
                backend_ref="synthetic-backend",
            )
            for item in self.projections[-limit:]
        )

    async def forget_namespace(self) -> None:
        self.projections.clear()


class _TextAdapter:
    def propose_procedure_traces(self, task, recalled_evidence, *, count=4):
        del task, recalled_evidence
        return tuple(f"Inspect bounded candidate {index}." for index in range(count))

    def execute_with_procedure(self, task, selected_trace, observations):
        raise AssertionError("the successor cycle uses its generic executor")


class _RelationAdapter:
    def __init__(self, encoder_ref: str, temporal_width: int) -> None:
        self.encoder_ref = encoder_ref
        self.temporal_width = temporal_width
        self.last_recall = None

    def encode_candidates(self, task, proposals, recall):
        del task
        self.last_recall = recall
        support = tuple(sorted(item.artifact_ref for item in recall.items))
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
                supporting_evidence_refs=support,
            )
            for index, trace in enumerate(proposals)
        )


class _FabricatedRelationAdapter(_RelationAdapter):
    def encode_candidates(self, task, proposals, recall):
        rows = super().encode_candidates(task, proposals, recall)
        return tuple(
            EncodedProcedureCandidate(
                row.trace,
                row.relation_features,
                row.temporal_features,
                row.base_logit,
                (_ref("fabricated-outside-recall"),),
            )
            for row in rows
        )


class _Executor:
    def __init__(self, status: str = "COMPLETED") -> None:
        self.status = status
        self.calls = []
        self.recorded = {}

    def execute(self, request):
        self.calls.append(request)
        result = CognitiveExecution(
            self.status,
            request.selected_trace,
            "bounded public response" if self.status == "COMPLETED" else "",
            ("bounded public observation",),
        )
        receipt = CognitiveExecutionReceipt.from_request(
            request,
            status=result.status,
            executed_trace=result.executed_trace,
            response=result.response,
            output_observations=result.observations,
        )
        self.recorded[request.execution_request_ref] = receipt
        return result

    def recover(self, request):
        from angler.runtime.cognitive_cycle import CognitiveExecutionRecovery

        receipt = self.recorded.get(request.execution_request_ref)
        return (
            CognitiveExecutionRecovery("NOT_STARTED")
            if receipt is None
            else CognitiveExecutionRecovery("RECORDED", receipt)
        )


class _ForgedObservedStateEncoder:
    """Protocol-shaped encoder that must not enter this frozen local lane."""

    def __init__(self, encoder_ref: str, latent_width: int = 16) -> None:
        self.encoder_ref = encoder_ref
        self.latent_width = latent_width

    def encode(self, execution_request, execution_receipt, *, latent_width):
        return ObservedStateEncoding(
            encoder_ref=self.encoder_ref,
            latent_width=latent_width,
            latent=tuple(0.0 for _ in range(latent_width)),
            evidence_refs=tuple(
                sorted(
                    (
                        execution_request.execution_request_ref,
                        execution_receipt.execution_receipt_ref,
                    )
                )
            ),
        )


def _prospective_learner(
    seed: int = 20260831, *, prospective_lesion: bool = False
) -> DurableAbilityLearner:
    torch.manual_seed(seed)
    credit = StructureKeyedCreditMemoryCore(temporal_width=3)
    core = CompositeProspectiveCreditCore(
        credit,
        ProspectiveDynamicsConfig(temporal_width=3, latent_width=16),
    )
    return DurableAbilityLearner(
        core,
        core.initial_state(),
        checkpoint_identity=core.checkpoint_identity,
        prospective_lesion=prospective_lesion,
    )


def _qwen_prospective_learner(
    *, prospective_lesion: bool = False
) -> DurableAbilityLearner:
    """Reconstruct the exact frozen CPU learner genesis from the live leaf."""

    torch.manual_seed(2_026_083_101)
    credit = StructureKeyedCreditMemoryCore(temporal_width=8)
    core = CompositeProspectiveCreditCore(
        credit,
        ProspectiveDynamicsConfig(temporal_width=8, latent_width=16),
    )
    return DurableAbilityLearner(
        core,
        core.initial_state(),
        checkpoint_identity=core.checkpoint_identity,
        prospective_lesion=prospective_lesion,
    )


def _qwen_manifest(model_ref: str, tokenizer_ref: str) -> FrozenQwenCycleManifestV1:
    return FrozenQwenCycleManifestV1(
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
        genesis_config_ref=(
            "sha256:ec5df84ea8e3946ce9e2bc387ef874c5bd719d962ee0c9c08bd2ab6a279c05b7"
        ),
        prospective_config_ref=(
            "sha256:f3bab0f86a3671588a3bd66538448cf276e3a6e965d7d866f7698a092d1e7313"
        ),
        learner_checkpoint_ref=(
            "sha256:5b2de1bb50091570dd92a790fe179cf3681f1202a1660d843863c46845f14f4c"
        ),
        initial_competence_state_digest=(
            "sha256:5a8bac6f60ba9d0d73c4cbc3965479a7f1b1f8d4fe4b3e9c9bb4530a924a4286"
        ),
        genesis_snapshot_sha256=(
            "93ab9deee7b0b67eabade8d43bb2b09e466f1dcd25405931d5a3d55eca91902d"
        ),
        genesis_snapshot_bytes=139_913,
        genesis_seed=2_026_083_101,
        prospective_lesion=False,
    )


class CognitiveCycleV3Tests(unittest.IsolatedAsyncioTestCase):
    def _fixture(
        self,
        directory: str,
        *,
        proposal_count: int = 4,
        executor_status: str = "COMPLETED",
        seed_store: bool = True,
        seed_count: int = 1,
        prospective_lesion: bool = False,
        removal_condition: str = "FULL",
        frozen_recall_hits: tuple[AcquisitionReferenceHit, ...] = (),
    ):
        from tests.unit.runtime.test_cognitive_transaction_store import (
            _episode as legacy_episode,
        )

        path = Path(directory) / "cycle-v3.sqlite3"
        learner = _prospective_learner(prospective_lesion=prospective_lesion)
        encoder = SyntheticObservedStateEncoderV1(16)
        model_ref = _ref("cycle-v3-model")
        store = CognitiveTransactionStore(path)
        if seed_store:
            snapshot = learner.capture_state()
            state_digest = learner.state_digest()
            store.initialize(
                state_digest,
                snapshot,
                model_ref=model_ref,
                encoder_ref=encoder.encoder_ref,
            )
            previous_episode_ref = None
            for index in range(seed_count):
                seed = legacy_episode(
                    state_digest,
                    state_digest,
                    task_id=(
                        "observed-cold-start-seed"
                        if index == 0
                        else f"observed-cold-start-seed-{index}"
                    ),
                    previous_episode_ref=previous_episode_ref,
                    model_ref=model_ref,
                    encoder_ref=encoder.encoder_ref,
                )
                store.commit_episode(
                    seed,
                    snapshot,
                    expected_parent_digest=state_digest,
                )
                previous_episode_ref = seed.episode_ref
        backend = _AcquisitionBackend()
        memory = AcquisitionSituatedMemory(source=store, backend=backend)
        relation = _RelationAdapter(encoder.encoder_ref, temporal_width=3)
        executor = _Executor(executor_status)
        cycle = CognitiveCycle(
            text_adapter=_TextAdapter(),
            relation_adapter=relation,
            learner=learner,
            memory=memory,
            executor=executor,
            transaction_store=store,
            model_ref=model_ref,
            encoder_ref=encoder.encoder_ref,
            agent_ref=_ref("cycle-v3-agent"),
            world_ref=_ref("cycle-v3-world"),
            recall_limit=12,
            proposal_count=proposal_count,
            successor_mode=True,
            reality_mode="SIMULATED",
            subject_ref=_ref("cycle-v3-subject"),
            scope_ref=_ref("cycle-v3-scope"),
            bootstrap_evidence_refs=(_ref("cycle-v3-bootstrap"),),
            observation_encoder=encoder,
            removal_condition=removal_condition,
            frozen_recall_hits=frozen_recall_hits,
        )
        return cycle, learner, memory, backend, store, executor, encoder, model_ref

    def _qwen_fixture(
        self,
        directory: str,
        *,
        seed_store: bool,
        backend: _AcquisitionBackend | None = None,
        io: LocalQwenIO | None = None,
        manifest: FrozenQwenCycleManifestV1 | None = None,
        relation_manifest: FrozenQwenCycleManifestV1 | None = None,
        observation_encoder: object | None = None,
        learner: DurableAbilityLearner | None = None,
    ):
        from tests.unit.runtime.test_cognitive_transaction_store import (
            _episode as legacy_episode,
        )
        from tests.unit.runtime.test_qwen_cognitive import _Model, _Tokenizer

        if learner is None:
            learner = _qwen_prospective_learner()
        model_ref = _ref("cycle-v3-qwen-model")
        tokenizer_ref = _ref("cycle-v3-qwen-tokenizer")
        manifest = manifest or _qwen_manifest(model_ref, tokenizer_ref)
        if io is None:
            model = _Model(width=manifest.input_width)
            tokenizer = _Tokenizer()
            model.requires_grad_(False)
            model.eval()
            io = LocalQwenIO(
                model,
                tokenizer,
                embedding_batch_size=manifest.embedding_batch_size,
                max_input_tokens=manifest.max_input_tokens,
                max_new_tokens=manifest.max_new_tokens,
                enable_thinking=False,
                model_ref=manifest.model_ref,
                tokenizer_ref=manifest.tokenizer_ref,
            )
        encoder = observation_encoder or QwenReceiptObservedStateEncoderV1(
            model_ref=manifest.model_ref,
            encoder_ref=manifest.encoder_ref,
            manifest_ref=manifest.manifest_ref,
            latent_width=manifest.latent_width,
        )
        path = Path(directory) / "cycle-v3-qwen.sqlite3"
        store = CognitiveTransactionStore(path)
        if seed_store:
            snapshot = learner.capture_state()
            self.assertEqual(len(snapshot), manifest.genesis_snapshot_bytes)
            self.assertEqual(
                hashlib.sha256(snapshot).hexdigest(),
                manifest.genesis_snapshot_sha256,
            )
            self.assertEqual(
                learner.state_digest(), manifest.initial_competence_state_digest
            )
            store.initialize(
                learner.state_digest(),
                snapshot,
                model_ref=manifest.model_ref,
                encoder_ref=manifest.encoder_ref,
            )
            seed = legacy_episode(
                learner.state_digest(),
                learner.state_digest(),
                task_id="qwen-observed-cold-start-seed",
                model_ref=manifest.model_ref,
                encoder_ref=manifest.encoder_ref,
            )
            store.commit_episode(
                seed,
                snapshot,
                expected_parent_digest=learner.state_digest(),
            )
        backend = backend or _AcquisitionBackend()
        memory = AcquisitionSituatedMemory(source=store, backend=backend)
        text = FrozenQwenProcedureAdapterV1(io, manifest)
        relation = FrozenQwenProcedureRelationAdapterV1(
            io, relation_manifest or manifest
        )
        journal = SQLiteQwenExecutionJournal(
            Path(directory) / "cycle-v3-qwen-journal.sqlite3"
        )
        executor = FrozenQwenCognitiveExecutorV1(io, manifest, journal)
        cycle = CognitiveCycle(
            text_adapter=text,
            relation_adapter=relation,
            learner=learner,
            memory=memory,
            executor=executor,
            transaction_store=store,
            model_ref=manifest.model_ref,
            encoder_ref=manifest.encoder_ref,
            agent_ref=_ref("cycle-v3-qwen-agent"),
            world_ref=_ref("cycle-v3-qwen-world"),
            recall_limit=12,
            proposal_count=2,
            successor_mode=True,
            reality_mode="SIMULATED",
            subject_ref=_ref("cycle-v3-qwen-subject"),
            scope_ref=_ref("cycle-v3-qwen-scope"),
            bootstrap_evidence_refs=(_ref("cycle-v3-qwen-bootstrap"),),
            observation_encoder=encoder,
        )
        return (
            cycle,
            learner,
            memory,
            backend,
            store,
            executor,
            encoder,
            manifest,
            io,
        )

    def _restart_fixture(
        self,
        path: Path,
        *,
        backend: _AcquisitionBackend,
        executor: _Executor,
        proposal_count: int = 4,
        subject_ref: str | None = None,
        observation_encoder=None,
    ):
        learner = _prospective_learner()
        encoder = observation_encoder or SyntheticObservedStateEncoderV1(16)
        store = CognitiveTransactionStore(path)
        memory = AcquisitionSituatedMemory(source=store, backend=backend)
        cycle = CognitiveCycle(
            text_adapter=_TextAdapter(),
            relation_adapter=_RelationAdapter(encoder.encoder_ref, 3),
            learner=learner,
            memory=memory,
            executor=executor,
            transaction_store=store,
            model_ref=_ref("cycle-v3-model"),
            encoder_ref=encoder.encoder_ref,
            agent_ref=_ref("cycle-v3-agent"),
            world_ref=_ref("cycle-v3-world"),
            recall_limit=12,
            proposal_count=proposal_count,
            successor_mode=True,
            reality_mode="SIMULATED",
            subject_ref=subject_ref or _ref("cycle-v3-subject"),
            scope_ref=_ref("cycle-v3-scope"),
            bootstrap_evidence_refs=(_ref("cycle-v3-bootstrap"),),
            observation_encoder=encoder,
        )
        return cycle, learner, memory, store

    async def test_concurrent_reservation_loser_rehydrates_canonical_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            first, _learner, _memory, _backend, store, _executor, _encoder, _model = (
                self._fixture(directory)
            )
            second_backend = _AcquisitionBackend()
            second_executor = _Executor()
            second, second_learner, _second_memory, _second_store = (
                self._restart_fixture(
                    Path(directory) / "cycle-v3.sqlite3",
                    backend=second_backend,
                    executor=second_executor,
                )
            )

            winner = await first.begin_turn(
                "canonical-winner", "Use the one canonical observed seed."
            )
            with self.assertRaises(ValueError):
                await second.begin_turn(
                    "concurrent-loser", "Use the one canonical observed seed."
                )

            self.assertIsNotNone(store.active_prospective_turn())
            self.assertIsNotNone(second.pending_turn)
            self.assertEqual(second.pending_turn.reservation, winner.reservation)
            self.assertEqual(
                second.pending_turn.successor_reservation,
                winner.successor_reservation,
            )
            self.assertEqual(second_learner.pending_decision, winner.decision)
            store.audit_integrity()

    def test_successor_lane_rejects_protocol_shaped_forged_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            _cycle, _learner, _memory, _backend, _store, _executor, encoder, _model = (
                self._fixture(directory)
            )
            forged = _ForgedObservedStateEncoder(encoder.encoder_ref)
            self.assertIsInstance(forged, FrozenObservedStateEncoder)

            with self.assertRaisesRegex(
                TypeError, "exact SyntheticObservedStateEncoderV1"
            ):
                self._restart_fixture(
                    Path(directory) / "cycle-v3.sqlite3",
                    backend=_AcquisitionBackend(),
                    executor=_Executor(),
                    observation_encoder=forged,
                )

    async def test_exact_qwen_successor_cycle_survives_receipt_stage_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            (
                cycle,
                _learner,
                _memory,
                backend,
                store,
                executor,
                _encoder,
                manifest,
                io,
            ) = self._qwen_fixture(directory, seed_store=True)
            io.model.queue(  # type: ignore[attr-defined]
                '{"procedures":["Inspect.","Compare."]}',
                "Exact frozen Qwen response.",
            )

            turn = await cycle.begin_turn(
                "qwen-successor-turn",
                "Use the exact observed public evidence.",
            )
            completed = await cycle.execute_turn(
                turn,
                observations=("public execution observation",),
            )
            self.assertEqual(io.model.generate_calls, 2)  # type: ignore[attr-defined]
            self.assertEqual(completed.execution.executed_trace, turn.decision.selected_trace)
            self.assertEqual(
                executor.journal.get(turn.reservation.idempotency_key).receipt,
                completed.execution_receipt,
            )
            self.assertIsNotNone(store.active_prospective_turn())

            (
                restarted,
                restarted_learner,
                _restarted_memory,
                _backend,
                restarted_store,
                restarted_executor,
                _restarted_encoder,
                _restarted_manifest,
                _restarted_io,
            ) = self._qwen_fixture(
                directory,
                seed_store=False,
                backend=backend,
                io=io,
                manifest=manifest,
            )
            self.assertIsNotNone(restarted.executed_turn)
            self.assertEqual(
                restarted.executed_turn.execution_receipt,
                completed.execution_receipt,
            )
            result = await restarted.record_outcome(
                restarted.executed_turn,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="success",
                feedback_text="External synthetic verifier accepted the response.",
                feedback_source_ref=_ref("cycle-v3-qwen-feedback"),
            )
            self.assertFalse(result.projection_pending)
            self.assertEqual(restarted_store.head().sequence, 2)
            self.assertNotEqual(
                restarted_learner.state_digest(),
                manifest.initial_competence_state_digest,
            )
            self.assertEqual(io.model.generate_calls, 2)  # type: ignore[attr-defined]
            restarted_executor.journal.audit_integrity()
            restarted_store.audit_integrity()

    async def test_qwen_advanced_store_restart_does_not_require_genesis_reseed(self):
        with tempfile.TemporaryDirectory() as directory:
            (
                cycle,
                learner,
                _memory,
                backend,
                store,
                _executor,
                _encoder,
                manifest,
                io,
            ) = self._qwen_fixture(directory, seed_store=True)
            genesis_shell = copy.deepcopy(learner)
            io.model.queue(  # type: ignore[attr-defined]
                '{"procedures":["Inspect.","Compare."]}',
                "Exact frozen Qwen response.",
            )
            turn = await cycle.begin_turn(
                "qwen-advanced-restart",
                "Advance the exact canonical learner state.",
            )
            completed = await cycle.execute_turn(
                turn,
                observations=("public execution observation",),
            )
            result = await cycle.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="success",
                feedback_text="External synthetic verifier accepted the response.",
                feedback_source_ref=_ref("cycle-v3-qwen-advanced-feedback"),
            )
            self.assertFalse(result.projection_pending)
            advanced_digest = learner.state_digest()
            advanced_snapshot = store.load_head_state()
            advanced_head = store.head()
            self.assertNotEqual(
                advanced_digest,
                manifest.initial_competence_state_digest,
            )
            advanced_shell = copy.deepcopy(learner)

            with mock.patch.object(
                torch,
                "manual_seed",
                side_effect=AssertionError("restart must not reseed"),
            ):
                (
                    restored_from_genesis,
                    genesis_shell,
                    _memory_from_genesis,
                    _backend_from_genesis,
                    store_from_genesis,
                    _executor_from_genesis,
                    _encoder_from_genesis,
                    _manifest_from_genesis,
                    _io_from_genesis,
                ) = self._qwen_fixture(
                    directory,
                    seed_store=False,
                    backend=backend,
                    io=io,
                    manifest=manifest,
                    learner=genesis_shell,
                )
            self.assertEqual(genesis_shell.state_digest(), advanced_digest)
            self.assertEqual(genesis_shell.capture_state(), advanced_snapshot)
            self.assertEqual(store_from_genesis.head(), advanced_head)
            self.assertEqual(
                restored_from_genesis.current_lineage.competence_state_digest,
                advanced_digest,
            )

            with mock.patch.object(
                torch,
                "manual_seed",
                side_effect=AssertionError("restart must not reseed"),
            ):
                (
                    restored_from_advanced,
                    advanced_shell,
                    _memory_from_advanced,
                    _backend_from_advanced,
                    store_from_advanced,
                    _executor_from_advanced,
                    _encoder_from_advanced,
                    _manifest_from_advanced,
                    _io_from_advanced,
                ) = self._qwen_fixture(
                    directory,
                    seed_store=False,
                    backend=backend,
                    io=io,
                    manifest=manifest,
                    learner=advanced_shell,
                )
            self.assertEqual(advanced_shell.state_digest(), advanced_digest)
            self.assertEqual(advanced_shell.capture_state(), advanced_snapshot)
            self.assertEqual(store_from_advanced.head(), advanced_head)
            self.assertEqual(
                restored_from_advanced.current_lineage.competence_state_digest,
                advanced_digest,
            )

            with tempfile.TemporaryDirectory() as empty_directory:
                with mock.patch.object(
                    torch,
                    "manual_seed",
                    side_effect=AssertionError("restart must not reseed"),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "does not begin from the frozen genesis snapshot",
                    ):
                        self._qwen_fixture(
                            empty_directory,
                            seed_store=False,
                            io=io,
                            manifest=manifest,
                            learner=copy.deepcopy(advanced_shell),
                        )

    def test_qwen_successor_lane_rejects_mixed_manifest_and_forged_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            (
                _cycle,
                _learner,
                _memory,
                backend,
                _store,
                _executor,
                encoder,
                manifest,
                io,
            ) = self._qwen_fixture(directory, seed_store=True)
            mixed = replace(manifest, genesis_seed=manifest.genesis_seed + 1)
            with self.assertRaisesRegex(ValueError, "manifest identities differ"):
                self._qwen_fixture(
                    directory,
                    seed_store=False,
                    backend=backend,
                    io=io,
                    manifest=manifest,
                    relation_manifest=mixed,
                )

            forged = _ForgedObservedStateEncoder(
                encoder.encoder_ref,
                latent_width=manifest.latent_width,
            )
            with self.assertRaisesRegex(TypeError, "exact SyntheticObservedStateEncoderV1"):
                self._qwen_fixture(
                    directory,
                    seed_store=False,
                    backend=backend,
                    io=io,
                    manifest=manifest,
                    observation_encoder=forged,
                )

    async def test_observed_successor_cycle_commits_one_lineage_and_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle, learner, memory, _backend, store, executor, _encoder, _model = (
                self._fixture(directory)
            )
            parent_lineage = cycle.current_lineage
            turn = await cycle.begin_turn("successor-turn", "Use observed evidence.")
            self.assertIsNotNone(turn.successor_reservation)
            self.assertEqual(len(executor.calls), 0)
            self.assertEqual(store.acquisition_head().next_ordinal, 2)
            self.assertEqual(
                turn.successor_reservation.idempotency_key,
                turn.reservation.reservation_ref,
            )
            completed = await cycle.execute_turn(turn)
            result = await cycle.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="success",
                feedback_text="Objective synthetic check passed.",
                feedback_source_ref=_ref("cycle-v3-feedback"),
            )
            self.assertFalse(result.projection_pending)
            self.assertEqual(store.head().sequence, 2)
            self.assertEqual(store.acquisition_head().next_ordinal, 3)
            self.assertIsNone(store.active_prospective_turn())
            self.assertIsNone(learner.pending_decision)
            self.assertEqual(memory.origin.size, 3)
            child = cycle.current_lineage
            self.assertIsNotNone(child)
            parent_lineage.assert_successor(child)
            aggregate = store.prospective_turn_for_episode(result.episode_ref)
            self.assertEqual(aggregate.episode_v2.resolution.child_lineage, child)
            self.assertEqual(
                aggregate.resolution_acquisition_ref,
                store.acquisition_head().acquisition_ref,
            )
            store.audit_integrity()

    async def test_two_seven_and_sixty_four_candidates_keep_full_learned_batches(self):
        for count in (2, 7, 64):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                cycle, learner, _memory, _backend, store, executor, _encoder, _model = (
                    self._fixture(directory, proposal_count=count)
                )
                turn = await cycle.begin_turn(
                    f"candidate-count-{count}",
                    "Use the same bounded observed evidence.",
                )
                wrapper = turn.successor_reservation
                self.assertEqual(len(wrapper.batch.branches), count)
                self.assertEqual(
                    tuple(branch.candidate_index for branch in wrapper.batch.branches),
                    tuple(range(count)),
                )
                self.assertEqual(
                    tuple(branch.decision_logit for branch in wrapper.batch.branches),
                    turn.decision.logits,
                )
                self.assertTrue(
                    all(len(branch.predicted_next_latent) == 16 for branch in wrapper.batch.branches)
                )
                self.assertTrue(
                    all(branch.focused_context_indices == (0,) for branch in wrapper.batch.branches)
                )
                self.assertEqual(len(executor.calls), 0)
                result = await cycle.cancel_pending_turn(turn)
                self.assertFalse(result.projection_pending)
                self.assertIsNone(learner.pending_decision)
                self.assertIsNone(store.active_prospective_turn())

    async def test_fixed_removals_reuse_one_frozen_manifest_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fixture(name: str, **kwargs):
                location = root / name
                location.mkdir()
                return self._fixture(str(location), seed_count=2, **kwargs)

            # Freeze one reference-only manifest before constructing any of
            # the compared condition fixtures.  Every seeded store has exact
            # content identities, so the same manifest rejoins each one.
            manifest_source = fixture("manifest-source")
            source_store = manifest_source[4]
            manifest = tuple(
                AcquisitionReferenceHit(
                    item.record_ref,
                    score=1.0,
                    backend_ref="synthetic-backend",
                )
                for item in source_store.acquisition_items(limit=12)
            )
            request = "Use the exact shared canonical evidence manifest."

            full = fixture("full", frozen_recall_hits=manifest)
            (
                full_cycle,
                full_learner,
                _full_memory,
                full_backend,
                _full_store,
                _full_executor,
                _full_encoder,
                _full_model,
            ) = full
            full_turn = await full_cycle.begin_turn("fair-removal-turn", request)
            full_recall = full_cycle.relation_adapter.last_recall
            self.assertEqual(full_cycle.removal_condition, "FULL")
            self.assertEqual(full_backend.queries, [(request, 12)])
            self.assertGreater(full_recall.items[0].position.age, 0)

            frozen = fixture(
                "frozen-origin",
                removal_condition="FROZEN_ORIGIN",
                frozen_recall_hits=manifest,
            )
            frozen_cycle, _frozen_learner, *_ = frozen
            frozen_turn = await frozen_cycle.begin_turn("fair-removal-turn", request)
            frozen_recall = frozen_cycle.relation_adapter.last_recall
            self.assertEqual(frozen_cycle.removal_condition, "FROZEN_ORIGIN")
            self.assertEqual(
                tuple(item.record for item in frozen_recall.items),
                tuple(item.record for item in full_recall.items),
            )
            self.assertEqual(
                tuple(item.record_ref for item in frozen_recall.items),
                tuple(item.record_ref for item in full_recall.items),
            )
            self.assertEqual(
                tuple(item.position.acquired_ordinal for item in frozen_recall.items),
                tuple(item.position.acquired_ordinal for item in full_recall.items),
            )
            self.assertTrue(
                all(item.position.age == 0 for item in frozen_recall.items)
            )
            self.assertEqual(
                frozen_turn.successor_reservation.batch,
                full_turn.successor_reservation.batch,
            )

            backend_removed = fixture(
                "backend-removal",
                removal_condition="BACKEND_REMOVAL",
                frozen_recall_hits=manifest,
            )
            backend_cycle, _backend_learner, _memory, backend, *_ = backend_removed
            backend_turn = await backend_cycle.begin_turn(
                "fair-removal-turn", request
            )
            backend_recall = backend_cycle.relation_adapter.last_recall
            self.assertEqual(backend_cycle.removal_condition, "BACKEND_REMOVAL")
            self.assertEqual(backend.queries, [])
            self.assertEqual(backend_recall, full_recall)
            self.assertEqual(
                backend_turn.successor_reservation.batch,
                full_turn.successor_reservation.batch,
            )

            lesioned = fixture(
                "prospective-removal",
                prospective_lesion=True,
                removal_condition="PROSPECTIVE_REMOVAL",
                frozen_recall_hits=manifest,
            )
            lesion_cycle, lesion_learner, *_ = lesioned
            lesion_turn = await lesion_cycle.begin_turn(
                "fair-removal-turn", request
            )
            self.assertEqual(
                lesion_cycle.removal_condition, "PROSPECTIVE_REMOVAL"
            )
            full_material = full_learner.pending_prospective_material
            lesion_material = lesion_learner.pending_prospective_material
            self.assertIsNotNone(full_material)
            self.assertIsNotNone(lesion_material)
            self.assertTrue(full_material.prospective_read_enabled)
            self.assertFalse(lesion_material.prospective_read_enabled)
            for field in ("relation_features", "temporal_features", "base_logits"):
                self.assertTrue(
                    torch.equal(
                        getattr(full_material, field),
                        getattr(lesion_material, field),
                    )
                )
            for field in (
                "candidate_traces",
                "candidate_supporting_evidence_refs",
                "recalled_refs",
                "context_refs",
                "eligibility_rows",
                "resource_budget",
                "parent_competence_digest",
                "checkpoint_ref",
                "config_ref",
                "component_state",
            ):
                self.assertEqual(
                    getattr(full_material, field),
                    getattr(lesion_material, field),
                )
            self.assertEqual(
                lesion_turn.successor_reservation.batch.resources,
                full_turn.successor_reservation.batch.resources,
            )
            self.assertEqual(lesion_turn.recalled_refs, full_turn.recalled_refs)
            with self.assertRaises(AttributeError):
                full_cycle.removal_condition = "FROZEN_ORIGIN"

    async def test_cold_start_and_fabricated_support_fail_before_selection_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle, learner, _memory, _backend, store, executor, _encoder, _model = (
                self._fixture(directory, seed_store=False)
            )
            with self.assertRaisesRegex(ValueError, "no admissible observed evidence"):
                await cycle.begin_turn("cold-start", "No evidence exists.")
            self.assertIsNone(learner.pending_decision)
            self.assertIsNone(store.active_prospective_turn())
            self.assertEqual(store.acquisition_head().next_ordinal, 0)
            self.assertEqual(executor.calls, [])

        with tempfile.TemporaryDirectory() as directory:
            cycle, learner, _memory, _backend, store, executor, encoder, _model = (
                self._fixture(directory)
            )
            cycle.relation_adapter = _FabricatedRelationAdapter(
                encoder.encoder_ref,
                temporal_width=3,
            )
            with self.assertRaisesRegex(ValueError, "canonical recalled support"):
                await cycle.begin_turn("fabricated", "Reject invented evidence.")
            self.assertIsNone(learner.pending_decision)
            self.assertIsNone(store.active_prospective_turn())
            self.assertEqual(store.acquisition_head().next_ordinal, 1)
            self.assertEqual(executor.calls, [])

    async def test_every_nonobserved_disposition_closes_without_learning_or_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle, learner, memory, _backend, store, executor, _encoder, _model = (
                self._fixture(directory)
            )
            parent_digest = learner.state_digest()
            parent_lineage = cycle.current_lineage

            cancelled = await cycle.begin_turn("cancelled", "Cancel before claim.")
            cancelled_wrapper = cancelled.successor_reservation
            cancelled_result = await cycle.cancel_pending_turn(cancelled)
            self.assertEqual(
                store.get_prospective_turn(cancelled_wrapper.reservation_ref).resolution_disposition,
                "CANCELLED",
            )
            self.assertFalse(cancelled_result.projection_pending)

            for status, disposition in (
                ("CLARIFICATION_REQUIRED", "CLARIFICATION_REQUIRED"),
                ("ERROR", "ERROR"),
            ):
                executor.status = status
                turn = await cycle.begin_turn(
                    disposition.lower(),
                    f"Exercise {disposition} without outcome truth.",
                )
                wrapper = turn.successor_reservation
                completed = await cycle.execute_turn(turn)
                self.assertEqual(completed.execution.status, status)
                record = store.get_prospective_turn(wrapper.reservation_ref)
                self.assertEqual(record.resolution_disposition, disposition)
                self.assertIsNone(record.feedback)
                self.assertIsNone(record.episode_v2)
                self.assertIsNone(cycle.pending_turn)

            executor.status = "COMPLETED"
            unevaluated = await cycle.begin_turn(
                "unevaluated",
                "Complete while objective feedback is explicitly unavailable.",
            )
            wrapper = unevaluated.successor_reservation
            completed = await cycle.execute_turn(unevaluated)
            result = await cycle.resolve_completed_unevaluated(
                completed,
                reservation_ref=unevaluated.reservation.reservation_ref,
            )
            record = store.get_prospective_turn(wrapper.reservation_ref)
            self.assertEqual(record.resolution_disposition, "COMPLETED_UNEVALUATED")
            self.assertIsNone(record.feedback)
            self.assertIsNone(record.episode_v2)
            self.assertFalse(result.projection_pending)

            self.assertEqual(learner.state_digest(), parent_digest)
            self.assertEqual(cycle.current_lineage, parent_lineage)
            self.assertEqual(store.head().sequence, 1)
            self.assertEqual(len(store.episode_items()), 1)
            self.assertEqual(store.acquisition_head().next_ordinal, 9)
            self.assertEqual(memory.origin.size, 9)
            store.audit_integrity()

    async def test_reserved_restart_restores_exact_batch_then_commits_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle-v3.sqlite3"
            cycle, _learner, _memory, backend, store, executor, _encoder, _model = (
                self._fixture(directory)
            )
            turn = await cycle.begin_turn("restart-reserved", "Persist before claim.")
            wrapper = turn.successor_reservation
            before = store.active_prospective_turn()

            restarted, learner2, memory2, store2 = self._restart_fixture(
                path,
                backend=backend,
                executor=executor,
            )
            self.assertEqual(restarted.pending_turn.successor_reservation, wrapper)
            self.assertEqual(store2.active_prospective_turn(), before)
            self.assertEqual(
                learner2.pending_prospective_material.batch_ref,
                wrapper.batch_ref,
            )
            completed = await restarted.execute_turn(restarted.pending_turn)
            result = await restarted.record_outcome(
                completed,
                reservation_ref=wrapper.legacy_reservation_ref,
                task_id=turn.decision.task_id,
                outcome="failure",
                feedback_text="Objective synthetic check failed.",
                feedback_source_ref=_ref("restart-feedback"),
            )
            self.assertFalse(result.projection_pending)
            self.assertEqual(len(executor.calls), 1)
            self.assertIsNone(store2.active_prospective_turn())
            self.assertEqual(memory2.origin.size, store2.acquisition_head().next_ordinal)
            store2.audit_integrity()

    async def test_staged_feedback_restart_reencodes_then_applies_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle-v3.sqlite3"
            cycle, _learner, _memory, backend, store, executor, _encoder, _model = (
                self._fixture(directory)
            )
            turn = await cycle.begin_turn("restart-staged", "Stage exact feedback.")
            completed = await cycle.execute_turn(turn)
            feedback = ObjectiveFeedbackRecord.from_execution(
                turn.reservation,
                completed.execution_request,
                completed.execution_receipt,
                outcome="success",
                feedback_text="Objective staged success.",
                feedback_source_ref=_ref("staged-feedback"),
            )
            store.stage_prospective_feedback(feedback)

            restarted, learner2, _memory2, store2 = self._restart_fixture(
                path,
                backend=backend,
                executor=executor,
            )
            self.assertIsNotNone(restarted.executed_turn)
            self.assertEqual(
                store2.active_prospective_turn().feedback,
                feedback,
            )
            result = await restarted.resume_staged_outcome()
            self.assertFalse(result.projection_pending)
            self.assertEqual(learner2.component_state_integrity().step, 1)
            self.assertIsNone(store2.active_prospective_turn())
            with self.assertRaisesRegex(RuntimeError, "no staged objective feedback"):
                await restarted.resume_staged_outcome()
            store2.audit_integrity()

    async def test_observation_encoder_failure_precedes_feedback_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle, learner, _memory, _backend, store, _executor, _encoder, _model = (
                self._fixture(directory)
            )
            turn = await cycle.begin_turn(
                "observation-failure", "Keep execution pending if encoding fails."
            )
            completed = await cycle.execute_turn(turn)
            before_head = store.head()
            before_acquisition_head = store.acquisition_head()

            with mock.patch.object(
                SyntheticObservedStateEncoderV1,
                "encode",
                side_effect=RuntimeError("injected observation encoder failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "injected observation encoder failure"
                ):
                    await cycle.record_outcome(
                        completed,
                        reservation_ref=turn.reservation.reservation_ref,
                        task_id=turn.decision.task_id,
                        outcome="success",
                        feedback_text="Objective result remains unstaged.",
                        feedback_source_ref=_ref("observation-failure-feedback"),
                    )

            active = store.active_prospective_turn()
            self.assertEqual(active.status, "EXECUTION_RECORDED")
            self.assertIsNone(active.feedback)
            self.assertEqual(store.head(), before_head)
            self.assertEqual(store.acquisition_head(), before_acquisition_head)
            self.assertIs(cycle.executed_turn, completed)
            self.assertEqual(learner.pending_decision, turn.decision)

            result = await cycle.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="success",
                feedback_text="Objective result remains unstaged.",
                feedback_source_ref=_ref("observation-failure-feedback"),
            )
            self.assertFalse(result.projection_pending)
            self.assertIsNone(store.active_prospective_turn())

    async def test_execution_recorded_restart_resumes_before_feedback_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle-v3.sqlite3"
            cycle, _learner, _memory, backend, store, executor, _encoder, _model = (
                self._fixture(directory)
            )
            turn = await cycle.begin_turn(
                "restart-recorded", "Restart after the exact completed receipt."
            )
            completed = await cycle.execute_turn(turn)
            self.assertIsNone(store.active_prospective_turn().feedback)

            restarted, learner2, _memory2, store2 = self._restart_fixture(
                path,
                backend=backend,
                executor=executor,
            )
            restored = restarted.executed_turn
            self.assertIsNotNone(restored)
            self.assertEqual(restored.execution_request, completed.execution_request)
            self.assertEqual(restored.execution_receipt, completed.execution_receipt)
            self.assertIsNone(store2.active_prospective_turn().feedback)

            result = await restarted.record_outcome(
                restored,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="failure",
                feedback_text="Objective result arrived after restart.",
                feedback_source_ref=_ref("recorded-restart-feedback"),
            )
            self.assertFalse(result.projection_pending)
            self.assertEqual(learner2.component_state_integrity().step, 1)
            self.assertIsNone(store2.active_prospective_turn())
            store2.audit_integrity()

    async def test_claimed_restart_requires_reconciliation_and_reuses_effect_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle-v3.sqlite3"
            cycle, _learner, _memory, backend, store, executor, _encoder, _model = (
                self._fixture(directory)
            )
            turn = await cycle.begin_turn("claimed-restart", "Claim exactly once.")
            request = CognitiveExecutionRequest.from_reservation(turn.reservation)
            store.claim_prospective_turn(request)

            restarted, _learner2, _memory2, store2 = self._restart_fixture(
                path,
                backend=backend,
                executor=executor,
            )
            from angler.runtime.cognitive_cycle import ExecutionReconciliationRequired

            with self.assertRaises(ExecutionReconciliationRequired):
                await restarted.execute_turn(restarted.pending_turn)
            completed = await restarted.recover_claimed_execution()
            self.assertEqual(len(executor.calls), 1)
            self.assertEqual(
                executor.calls[0].idempotency_key,
                turn.reservation.reservation_ref,
            )
            result = await restarted.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="success",
                feedback_text="Reconciled objective success.",
                feedback_source_ref=_ref("claimed-restart-feedback"),
            )
            self.assertFalse(result.projection_pending)
            self.assertIsNone(store2.active_prospective_turn())

    async def test_learner_and_projection_failure_preserve_canonical_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle, learner, _memory, backend, store, _executor, _encoder, _model = (
                self._fixture(directory)
            )
            turn = await cycle.begin_turn("learner-fault", "Retry exact staged feedback.")
            completed = await cycle.execute_turn(turn)
            parent_head = store.head()
            with mock.patch.object(
                learner,
                "apply_outcome",
                side_effect=RuntimeError("injected learner failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected learner failure"):
                    await cycle.record_outcome(
                        completed,
                        reservation_ref=turn.reservation.reservation_ref,
                        task_id=turn.decision.task_id,
                        outcome="success",
                        feedback_text="Staged before learner failure.",
                        feedback_source_ref=_ref("learner-fault-feedback"),
                    )
            self.assertEqual(store.head(), parent_head)
            self.assertIsNotNone(store.active_prospective_turn().feedback)
            self.assertIsNotNone(cycle.executed_turn)

            backend.fail_project = True
            committed = await cycle.resume_staged_outcome()
            self.assertTrue(committed.projection_pending)
            committed_head = store.head()
            self.assertIsNone(store.active_prospective_turn())
            self.assertEqual(learner.component_state_integrity().step, 1)
            backend.fail_project = False
            acknowledged = await cycle.retry_pending_projections()
            self.assertGreaterEqual(len(acknowledged), 1)
            self.assertEqual(store.head(), committed_head)
            self.assertEqual(store.pending_acquisition_projections(), ())
            store.audit_integrity()

    async def test_observed_restart_rejects_fixed_situated_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle-v3.sqlite3"
            cycle, _learner, _memory, backend, _store, executor, _encoder, _model = (
                self._fixture(directory)
            )
            turn = await cycle.begin_turn("lineage-identity", "Bind fixed identity.")
            completed = await cycle.execute_turn(turn)
            await cycle.record_outcome(
                completed,
                reservation_ref=turn.reservation.reservation_ref,
                task_id=turn.decision.task_id,
                outcome="success",
                feedback_text="Identity-bound objective success.",
                feedback_source_ref=_ref("lineage-identity-feedback"),
            )
            with self.assertRaisesRegex(ValueError, "fixed identity"):
                self._restart_fixture(
                    path,
                    backend=backend,
                    executor=executor,
                    subject_ref=_ref("different-subject"),
                )


if __name__ == "__main__":
    unittest.main()
