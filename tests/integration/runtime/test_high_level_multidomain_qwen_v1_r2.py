"""Exact-Qwen CPU integration for the high-level multi-domain removal controls.

This test deliberately uses the production LocalQwenIO, proposal/relation
adapters, journaled executor, successor CognitiveCycle, transaction store, and
AcquisitionSituatedMemory.  Only the tiny frozen model/tokenizer and the
reference backend are fakes; no live model, Cognee process, network, seed
artifact, evaluation manifest, or project state root is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from angler.cognition.contracts import CognitiveEpisode, ProspectiveCommitment
from angler.memory.cognitive_acquisition_graph import (
    AcquisitionRecallBatch,
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
from angler.runtime.cognitive_cycle import CognitiveCycle, ExecutedCognitiveTurn
from angler.runtime.cognitive_transaction_store import CognitiveTransactionStore
from angler.runtime.durable_ability_bridge import DurableAbilityLearner
from angler.runtime.prospective_observation import QwenReceiptObservedStateEncoderV1
from angler.runtime.qwen_cognitive import (
    FrozenQwenCognitiveExecutorV1,
    FrozenQwenCycleManifestV1,
    FrozenQwenProcedureAdapterV1,
    FrozenQwenProcedureRelationAdapterV1,
    QWEN_PROCEDURE_PROPOSAL_SCHEMA,
    QWEN_SELECTED_PROCEDURE_EXECUTION_SCHEMA,
    QwenExecutionJournalEntry,
    SQLiteQwenExecutionJournal,
)
from angler.runtime.situated_qwen import FrozenQwenGenerationRecord, LocalQwenIO
from experiments.evaluators import high_level_multidomain_v1 as high_evaluator
from experiments.runners import high_level_multidomain_v1_r2 as runner


_GENESIS_SEED = 2_026_083_101
_GENESIS_CONFIG_REF = (
    "sha256:ec5df84ea8e3946ce9e2bc387ef874c5bd719d962ee0c9c08bd2ab6a279c05b7"
)
_TASK_ID = "high-level-multidomain-fake-task"
_PUBLIC_TASK = "Apply one bounded public procedure to this synthetic task."
_PUBLIC_OBSERVATIONS = ("One bounded public observation.",)
_PROPOSAL_RESPONSE = '{"procedures":["Inspect evidence.","Apply procedure."]}'
_EXECUTION_RESPONSE = "EXACT_PUBLIC_RESPONSE"


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Batch(dict[str, torch.Tensor]):
    def to(self, device: torch.device | str) -> "_Batch":
        return _Batch({name: value.to(device) for name, value in self.items()})


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.chat_prompts: list[str] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        if kwargs != {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }:
            raise AssertionError("unexpected chat-template settings")
        if messages != [{"role": "user", "content": messages[0]["content"]}]:
            raise AssertionError("unexpected chat message shape")
        prompt = messages[0]["content"]
        self.chat_prompts.append(prompt)
        return prompt

    def __call__(
        self,
        texts: str | list[str] | tuple[str, ...],
        *,
        return_tensors: str,
        padding: bool = False,
        add_special_tokens: bool = False,
    ) -> _Batch:
        if return_tensors != "pt":
            raise AssertionError("unexpected tensor format")
        rows = [texts] if isinstance(texts, str) else list(texts)
        encoded_rows = [
            [2, *(2 + (value % 250) for value in row.encode("utf-8"))]
            if add_special_tokens
            else [2 + (value % 250) for value in row.encode("utf-8")]
            for row in rows
        ]
        maximum = max(len(row) for row in encoded_rows)
        input_ids = torch.zeros((len(rows), maximum), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, values in enumerate(encoded_rows):
            input_ids[index, : len(values)] = torch.tensor(values)
            attention_mask[index, : len(values)] = 1
        return _Batch(input_ids=input_ids, attention_mask=attention_mask)

    def batch_decode(self, generated: torch.Tensor, **kwargs: object) -> list[str]:
        if kwargs != {"skip_special_tokens": True}:
            raise AssertionError("unexpected decode settings")
        return [
            bytes(int(value) for value in row.detach().cpu().tolist()).decode("utf-8")
            for row in generated
        ]


class _TinyBackbone(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.linspace(0.25, 1.0, width))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask
        if use_cache or not return_dict:
            raise AssertionError("unexpected representation settings")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(
            1, -1, 1
        )
        hidden = (
            input_ids.float().unsqueeze(-1) + positions.float()
        ) * self.anchor.view(1, 1, -1)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.model = _TinyBackbone(width)
        self.responses: list[str] = []
        self.generate_calls = 0

    def queue(self, *responses: str) -> None:
        self.responses.extend(responses)

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del attention_mask
        if kwargs.get("do_sample") is not False or kwargs.get("max_new_tokens") != 64:
            raise AssertionError("unexpected generation settings")
        self.generate_calls += 1
        if not self.responses:
            raise AssertionError("no fake response is queued")
        values = torch.tensor(
            [list(self.responses.pop(0).encode("utf-8"))],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat((input_ids, values), dim=1)


class Qwen3ForCausalLM(_TinyModel):
    """Tiny CPU construction seam with the frozen public architecture metadata."""

    def __init__(self) -> None:
        super().__init__(8)
        self.config = SimpleNamespace(
            hidden_size=2_560,
            num_hidden_layers=36,
            max_position_embeddings=40_960,
        )
        self.to(dtype=torch.bfloat16)


class _GuardStorage:
    def __init__(self, pointer: int) -> None:
        self.pointer = pointer

    def data_ptr(self) -> int:
        return self.pointer


class _GuardParameter:
    def __init__(self, pointer: int) -> None:
        self.shape = (2, 3)
        self.dtype = "torch.bfloat16"
        self.device = "cpu"
        self.requires_grad = False
        self._version = 0
        self._storage = _GuardStorage(pointer)
        self._storage_offset = 0
        self._stride = (3, 1)

    def untyped_storage(self) -> _GuardStorage:
        return self._storage

    def storage_offset(self) -> int:
        return self._storage_offset

    def stride(self) -> tuple[int, ...]:
        return self._stride


class _GuardModel:
    def __init__(self) -> None:
        self.training = False
        self.rows = [
            ("first.weight", _GuardParameter(101)),
            ("second.weight", _GuardParameter(202)),
        ]

    def named_parameters(self) -> tuple[tuple[str, _GuardParameter], ...]:
        return tuple(self.rows)


class _FrozenReferenceBackend:
    """One bounded fake search surface over real canonical projections."""

    def __init__(self, hits: tuple[AcquisitionReferenceHit, ...]) -> None:
        self.hits = hits
        self.projections: dict[str, object] = {}
        self.search_calls = 0
        self.queries: list[tuple[str, int]] = []
        self.forget_calls = 0

    async def project(self, projection: object) -> str:
        projection_ref = getattr(projection, "projection_ref")
        record_ref = getattr(getattr(projection, "record"), "record_ref")
        previous = self.projections.get(record_ref)
        if previous is not None and getattr(previous, "projection_ref") != projection_ref:
            raise AssertionError("one record was projected with different canonical bytes")
        self.projections[record_ref] = projection
        return "fake-reference:" + record_ref

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[AcquisitionReferenceHit, ...]:
        self.search_calls += 1
        self.queries.append((query, limit))
        if not set(hit.record_ref for hit in self.hits).issubset(self.projections):
            raise AssertionError("search preceded canonical projection rebuild")
        return self.hits[:limit]

    async def forget_namespace(self) -> None:
        self.forget_calls += 1
        self.projections.clear()


class _ScopedFakeBinding:
    def __init__(self, service: "_ScopedFakeAcquisitionService", key: str) -> None:
        self.service = service
        self.key = key
        self.dataset_id = service.namespaces[key]["dataset_id"]
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            raise AssertionError("fake scoped binding closed twice")
        self._closed = True
        self.service.active[self.key] -= 1
        self.service.close_calls += 1


class _ScopedFakeBackend:
    def __init__(self, service: "_ScopedFakeAcquisitionService", key: str) -> None:
        self.service = service
        self.key = key

    @property
    def namespace(self) -> dict[str, object]:
        return self.service.namespaces[self.key]

    async def project(self, projection: object) -> str:
        record = getattr(projection, "record")
        record_ref = getattr(record, "record_ref")
        projections = self.namespace["projections"]
        assert isinstance(projections, dict)
        projections[record_ref] = projection
        return f"fake-reference:{self.key}:{record_ref}"

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[AcquisitionReferenceHit, ...]:
        self.service.search_calls.append((self.key, query, limit))
        projections = self.namespace["projections"]
        assert isinstance(projections, dict)
        refs = list(sorted(projections))
        extra = self.service.raw_extra_refs.get(self.key)
        if extra is not None and extra not in refs:
            refs.insert(0, extra)
        admitted_primary = next(
            (record_ref for record_ref in refs if record_ref != extra),
            None,
        )
        return tuple(
            AcquisitionReferenceHit(
                record_ref,
                adjacent_record_refs=(
                    (extra,)
                    if extra is not None and record_ref == admitted_primary
                    else ()
                ),
                score=float(index) / 100.0,
                backend_ref=f"fake-reference:{self.key}:{record_ref}",
            )
            for index, record_ref in enumerate(refs[:limit])
        )

    async def forget_namespace(self) -> None:
        self.service.forget_calls += 1
        projections = self.namespace["projections"]
        assert isinstance(projections, dict)
        projections.clear()


class _ScopedFakeAcquisitionService:
    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, object]] = {}
        self.active: dict[str, int] = {}
        self.raw_extra_refs: dict[str, str] = {}
        self.search_calls: list[tuple[str, str, int]] = []
        self.open_calls = 0
        self.close_calls = 0
        self.forget_calls = 0
        self.maximum_active = 0

    async def __call__(
        self,
        scope_spec: runner.AcquisitionScopeSpec,
        *,
        expected_dataset_id: str | None = None,
    ) -> tuple[_ScopedFakeBinding, _ScopedFakeBackend]:
        key = scope_spec.dataset_name
        if self.active.get(key, 0) != 0:
            raise AssertionError("shared fake scope lifetime overlapped")
        if key not in self.namespaces:
            self.namespaces[key] = {
                "dataset_id": "fake-dataset-" + hashlib.sha256(
                    key.encode("ascii")
                ).hexdigest()[:24],
                "projections": {},
            }
            self.active[key] = 0
        dataset_id = self.namespaces[key]["dataset_id"]
        if expected_dataset_id is not None and dataset_id != expected_dataset_id:
            raise AssertionError("fake dataset identity drifted")
        self.active[key] += 1
        self.maximum_active = max(self.maximum_active, sum(self.active.values()))
        self.open_calls += 1
        return _ScopedFakeBinding(self, key), _ScopedFakeBackend(self, key)


class _ObjectiveFakeEvaluator:
    async def judge_response(
        self,
        task_id: str,
        arm: str,
        attempt_receipt_ref: str,
        raw_response: str,
    ) -> dict[str, object]:
        score = 1.0
        return {
            "arm": arm,
            "attempt_receipt_ref": attempt_receipt_ref,
            "disposition": "SUCCESS",
            "raw_response": raw_response,
            "response_commitment": runner.evaluator_record_digest(
                "response",
                {
                    "arm": arm,
                    "attempt_receipt_ref": attempt_receipt_ref,
                    "raw_response": raw_response,
                    "task_id": task_id,
                },
            ),
            "score": score,
            "task_id": task_id,
        }


@dataclass(frozen=True, slots=True)
class _Baseline:
    store_path: Path
    journal_path: Path
    hit_manifest: tuple[AcquisitionReferenceHit, ...]
    store_sha256: str
    journal_sha256: str


@dataclass(frozen=True, slots=True)
class _ArmRuntime:
    arm: str
    hit_manifest: tuple[AcquisitionReferenceHit, ...]
    cycle: CognitiveCycle
    learner: DurableAbilityLearner
    memory: AcquisitionSituatedMemory
    backend: _FrozenReferenceBackend
    store: CognitiveTransactionStore
    journal: SQLiteQwenExecutionJournal
    text_adapter: FrozenQwenProcedureAdapterV1


@dataclass(frozen=True, slots=True)
class _ArmResult:
    runtime: _ArmRuntime
    recall: AcquisitionRecallBatch
    proposal_prompt: str
    execution_prompt: str
    proposal_generation: FrozenQwenGenerationRecord
    executed: ExecutedCognitiveTurn
    journal_entry: QwenExecutionJournalEntry


def _learner(*, prospective_lesion: bool = False) -> DurableAbilityLearner:
    torch.manual_seed(_GENESIS_SEED)
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


def _manifest(learner: DurableAbilityLearner) -> FrozenQwenCycleManifestV1:
    snapshot = learner.capture_state()
    integrity = learner.component_state_integrity()
    return FrozenQwenCycleManifestV1(
        model_ref=_ref("high-level-multidomain-tiny-qwen-model"),
        tokenizer_ref=_ref("high-level-multidomain-tiny-qwen-tokenizer"),
        genesis_config_ref=_GENESIS_CONFIG_REF,
        prospective_config_ref=integrity.config_ref,
        learner_checkpoint_ref=integrity.checkpoint_ref,
        initial_competence_state_digest=learner.state_digest(),
        genesis_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
        genesis_snapshot_bytes=len(snapshot),
        genesis_seed=_GENESIS_SEED,
        prospective_lesion=False,
        input_width=8,
        relation_width=64,
        temporal_width=8,
        latent_width=16,
        max_input_tokens=4_096,
        max_new_tokens=64,
        embedding_batch_size=1,
    )


def _seed_episode(
    state_digest: str,
    *,
    task_id: str,
    previous_episode_ref: str | None,
    manifest: FrozenQwenCycleManifestV1,
) -> CognitiveEpisode:
    proposals = (
        "Inspect the supplied public evidence.",
        "Apply the bounded public procedure.",
    )
    selected_index = 1
    return CognitiveEpisode(
        task_id=task_id,
        request="Use the public evidence to select a bounded procedure.",
        recalled_refs=(_ref("baseline-public-evidence"),),
        proposals=proposals,
        selected_index=selected_index,
        commitment=ProspectiveCommitment(
            parent_event_ref=previous_episode_ref,
            task_id=task_id,
            candidate_index=selected_index,
            candidate_trace=proposals[selected_index],
            predicted_score=0.75,
            uncertainty=0.25,
            horizon=1,
            competence_state_digest=state_digest,
        ),
        response="The bounded procedure was applied.",
        observations=("The objective synthetic check returned success.",),
        outcome="success",
        feedback_text="Objective synthetic feedback: success.",
        feedback_source_ref=_ref("baseline-feedback:" + task_id),
        parent_state_digest=state_digest,
        child_state_digest=state_digest,
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
        supporting_evidence_refs=(_ref("baseline-public-evidence"),),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_tensors(model: nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, value.detach().cpu().clone())
        for name, value in sorted(model.state_dict().items())
    )


def _same_model_tensors(
    left: tuple[tuple[str, torch.Tensor], ...],
    right: tuple[tuple[str, torch.Tensor], ...],
) -> bool:
    return len(left) == len(right) and all(
        left_name == right_name and _same_tensor(left_value, right_value)
        for (left_name, left_value), (right_name, right_value) in zip(
            left, right, strict=True
        )
    )


def _build_baseline(
    root: Path,
    manifest: FrozenQwenCycleManifestV1,
) -> _Baseline:
    learner = _learner()
    snapshot = learner.capture_state()
    state_digest = learner.state_digest()
    store_path = root / "closed-baseline-store.sqlite3"
    store = CognitiveTransactionStore(store_path)
    store.initialize(
        state_digest,
        snapshot,
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
    )
    previous_episode_ref = None
    for index in range(2):
        episode = _seed_episode(
            state_digest,
            task_id=f"baseline-observed-{index}",
            previous_episode_ref=previous_episode_ref,
            manifest=manifest,
        )
        store.commit_episode(
            episode,
            snapshot,
            expected_parent_digest=state_digest,
        )
        previous_episode_ref = episode.episode_ref
    store.audit_integrity()
    acquisition_items = tuple(store.acquisition_items(limit=12))
    if len(acquisition_items) != 2:
        raise AssertionError("baseline must expose exactly two observed acquisitions")
    hit_manifest = tuple(
        AcquisitionReferenceHit(
            item.record_ref,
            score=float(index) / 10.0,
            backend_ref="fake-reference:" + item.record_ref,
        )
        for index, item in enumerate(acquisition_items)
    )
    journal_path = root / "closed-baseline-qwen-journal.sqlite3"
    journal = SQLiteQwenExecutionJournal(journal_path)
    journal.audit_integrity()
    return _Baseline(
        store_path=store_path,
        journal_path=journal_path,
        hit_manifest=hit_manifest,
        store_sha256=_file_sha256(store_path),
        journal_sha256=_file_sha256(journal_path),
    )


def _clone_runtime(
    root: Path,
    *,
    arm: str,
    baseline: _Baseline,
    io: LocalQwenIO,
    manifest: FrozenQwenCycleManifestV1,
) -> _ArmRuntime:
    arm_root = root / arm.lower().replace("_", "-")
    arm_root.mkdir()
    store_path = arm_root / "cycle.sqlite3"
    journal_path = arm_root / "qwen-journal.sqlite3"
    shutil.copy2(baseline.store_path, store_path)
    shutil.copy2(baseline.journal_path, journal_path)
    learner = _learner(prospective_lesion=arm == "PROSPECTIVE_REMOVAL")
    store = CognitiveTransactionStore(store_path)
    backend = _FrozenReferenceBackend(baseline.hit_manifest)
    memory = AcquisitionSituatedMemory(source=store, backend=backend)
    text_adapter = FrozenQwenProcedureAdapterV1(io, manifest)
    relation_adapter = FrozenQwenProcedureRelationAdapterV1(io, manifest)
    journal = SQLiteQwenExecutionJournal(journal_path)
    executor = FrozenQwenCognitiveExecutorV1(io, manifest, journal)
    encoder = QwenReceiptObservedStateEncoderV1(
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
        manifest_ref=manifest.manifest_ref,
        latent_width=manifest.latent_width,
    )
    cycle = CognitiveCycle(
        text_adapter=text_adapter,
        relation_adapter=relation_adapter,
        learner=learner,
        memory=memory,
        executor=executor,
        transaction_store=store,
        model_ref=manifest.model_ref,
        encoder_ref=manifest.encoder_ref,
        agent_ref=_ref("high-level-multidomain-agent"),
        world_ref=_ref("high-level-multidomain-world"),
        recall_limit=12,
        proposal_count=2,
        successor_mode=True,
        reality_mode="SIMULATED",
        subject_ref=_ref("high-level-multidomain-subject"),
        scope_ref=_ref("high-level-multidomain-scope"),
        bootstrap_evidence_refs=(_ref("high-level-multidomain-bootstrap"),),
        observation_encoder=encoder,
        removal_condition=arm,
        frozen_recall_hits=baseline.hit_manifest,
    )
    return _ArmRuntime(
        arm=arm,
        hit_manifest=baseline.hit_manifest,
        cycle=cycle,
        learner=learner,
        memory=memory,
        backend=backend,
        store=store,
        journal=journal,
        text_adapter=text_adapter,
    )


async def _run_arm(
    runtime: _ArmRuntime,
    *,
    model: _TinyModel,
    tokenizer: _TinyTokenizer,
) -> _ArmResult:
    await runtime.memory.retry_pending_projections(limit=64)
    runtime.memory.assert_synchronized(runtime.store.acquisition_head())
    recall = runtime.memory.recall_from_hits(
        runtime.hit_manifest,
        limit=12,
        frozen_origin=runtime.arm == "FROZEN_ORIGIN",
    )
    prompt_start = len(tokenizer.chat_prompts)
    model.queue(_PROPOSAL_RESPONSE, _EXECUTION_RESPONSE)
    turn = await runtime.cycle.begin_turn(_TASK_ID, _PUBLIC_TASK)
    proposal_generation = runtime.text_adapter.last_proposal_generation
    if proposal_generation is None:
        raise AssertionError("proposal adapter did not retain its generation")
    executed = await runtime.cycle.execute_turn(
        turn,
        observations=_PUBLIC_OBSERVATIONS,
    )
    prompts = tuple(tokenizer.chat_prompts[prompt_start:])
    if len(prompts) != 2:
        raise AssertionError("one arm must issue one proposal and one execution prompt")
    entry = runtime.journal.get(executed.execution_request.idempotency_key)
    if entry is None:
        raise AssertionError("exact Qwen execution was not journaled")
    runtime.journal.audit_integrity()
    runtime.store.audit_integrity()
    return _ArmResult(
        runtime=runtime,
        recall=recall,
        proposal_prompt=prompts[0],
        execution_prompt=prompts[1],
        proposal_generation=proposal_generation,
        executed=executed,
        journal_entry=entry,
    )


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)


def _public_task(
    *,
    phase: str,
    ordinal: int,
    replicate_commitment: str,
) -> dict[str, object]:
    replicate = "qualification-01"
    family = "causal-operator"
    payload = {
        "family": family,
        "fixture": f"factory-{phase}-{ordinal}",
        "instruction": "Use only the bounded public synthetic input.",
    }
    payload_json = runner.canonical_json_bytes(payload).decode("utf-8")
    public_commitment = high_evaluator._public_commitment(
        replicate,
        replicate_commitment,
        family,
        phase,
        ordinal,
        payload_json,
    )
    task_id = high_evaluator._task_id(
        replicate,
        family,
        phase,
        ordinal,
        public_commitment,
    )
    return {
        "family": family,
        "ordinal": ordinal,
        "payload": payload,
        "phase": phase,
        "public_commitment": public_commitment,
        "replicate": replicate,
        "replicate_commitment": replicate_commitment,
        "schema": runner.SUITE_SCHEMA,
        "task_id": task_id,
    }


class HighLevelMultidomainConstructionIntegrationTests(unittest.TestCase):
    def test_fresh_genesis_is_exact_rng_preserving_and_independently_owned(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh-genesis"
            torch.manual_seed(9_193)
            rng_before = torch.get_rng_state().clone()
            bundle = runner.reconstruct_fresh_cpu_genesis(root)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            comparison_bundle = runner.reconstruct_fresh_cpu_genesis(
                Path(directory) / "fresh-genesis-comparison"
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertEqual(comparison_bundle.binding_ref, bundle.binding_ref)
            self.assertEqual(
                comparison_bundle.core_state_sha256,
                bundle.core_state_sha256,
            )
            self.assertEqual(
                bundle.binding_ref,
                runner.content_ref("fresh-genesis", bundle.to_canonical()),
            )
            self.assertEqual(
                bundle.to_canonical()["snapshot_sha256"],
                runner.QUALIFIED_GENESIS_SNAPSHOT_SHA256,
            )
            self.assertEqual(
                bundle.to_canonical()["snapshot_bytes"],
                runner.QUALIFIED_GENESIS_SNAPSHOT_BYTES,
            )
            for path in (
                bundle.snapshot_path,
                bundle.core_state_path,
                bundle.binding_path,
            ):
                self.assertEqual(path.stat().st_mode & 0o7777, 0o600)
            self.assertEqual(root.stat().st_mode & 0o7777, 0o700)

            first = runner.restore_fresh_cpu_genesis(bundle)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            lesion = runner.restore_fresh_cpu_genesis(
                bundle,
                prospective_lesion=True,
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertIsNot(first, lesion)
            self.assertIsNot(first.core, lesion.core)
            self.assertFalse(first.prospective_lesion)
            self.assertTrue(lesion.prospective_lesion)
            self.assertEqual(
                first.state_digest(),
                runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST,
            )
            self.assertEqual(lesion.state_digest(), first.state_digest())
            self.assertEqual(first.capture_state(), lesion.capture_state())
            first_parameters = tuple(first.core.parameters())
            lesion_parameters = tuple(lesion.core.parameters())
            self.assertEqual(len(first_parameters), len(lesion_parameters))
            self.assertTrue(
                all(
                    left is not right
                    and left.untyped_storage().data_ptr()
                    != right.untyped_storage().data_ptr()
                    and torch.equal(left, right)
                    for left, right in zip(
                        first_parameters,
                        lesion_parameters,
                        strict=True,
                    )
                )
            )
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "already consumed",
            ):
                runner.reconstruct_fresh_cpu_genesis(root)

    def test_foundation_guard_rejects_every_declared_metadata_drift(self) -> None:
        digest = _ref("foundation-guard-synthetic-digest")

        def parameter(model: _GuardModel, index: int = 0) -> _GuardParameter:
            return model.rows[index][1]

        def rename(model: _GuardModel) -> None:
            model.rows[0] = ("renamed.weight", parameter(model))

        def reorder(model: _GuardModel) -> None:
            model.rows.reverse()

        def replace_parameter(model: _GuardModel) -> None:
            model.rows[0] = (model.rows[0][0], _GuardParameter(101))

        def change_shape(model: _GuardModel) -> None:
            parameter(model).shape = (3, 2)

        def change_dtype(model: _GuardModel) -> None:
            parameter(model).dtype = "torch.float32"

        def change_device(model: _GuardModel) -> None:
            parameter(model).device = "cuda:0"

        def change_storage(model: _GuardModel) -> None:
            parameter(model)._storage.pointer += 1

        def change_version(model: _GuardModel) -> None:
            parameter(model)._version += 1

        def change_requires_grad(model: _GuardModel) -> None:
            parameter(model).requires_grad = True

        def change_training(model: _GuardModel) -> None:
            model.training = True

        mutations = {
            "name": rename,
            "order": reorder,
            "replacement": replace_parameter,
            "shape": change_shape,
            "dtype": change_dtype,
            "device": change_device,
            "storage": change_storage,
            "version": change_version,
            "requires_grad": change_requires_grad,
            "training": change_training,
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                model = _GuardModel()
                guard = runner.FoundationGuard(
                    model,
                    tensor_digest=lambda _model: digest,
                )
                self.assertEqual(guard.probe(), digest)
                mutate(model)
                with self.assertRaises(runner.RunnerInvariantError):
                    guard.probe()

    def test_foundation_guard_recomputes_one_exact_whole_boundary_digest(
        self,
    ) -> None:
        digest = _ref("foundation-guard-boundary")
        calls = 0

        def tensor_digest(_model: object) -> str:
            nonlocal calls
            calls += 1
            return digest

        guard = runner.FoundationGuard(
            _GuardModel(),
            tensor_digest=tensor_digest,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(guard.verify_boundary_digest(), digest)
        self.assertEqual(calls, 2)

        observed = digest

        def drifting_digest(_model: object) -> str:
            return observed

        drifting = runner.FoundationGuard(
            _GuardModel(),
            tensor_digest=drifting_digest,
        )
        observed = _ref("foundation-guard-boundary-drift")
        with self.assertRaisesRegex(
            runner.RunnerInvariantError,
            "tensor bytes changed",
        ):
            drifting.verify_boundary_digest()

    def test_injected_cpu_loader_constructs_one_exact_local_qwen_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "tiny-model-fixture"
            model_root.mkdir()
            calls: list[tuple[str, Path, dict[str, object]]] = []
            model = Qwen3ForCausalLM()
            tokenizer = _TinyTokenizer()

            def tokenizer_factory(path: Path, **kwargs: object) -> _TinyTokenizer:
                calls.append(("tokenizer", path, dict(kwargs)))
                return tokenizer

            def model_factory(path: Path, **kwargs: object) -> Qwen3ForCausalLM:
                calls.append(("model", path, dict(kwargs)))
                return model

            loaded = runner.load_qualified_qwen_foundation(
                model_root,
                model_factory=model_factory,
                tokenizer_factory=tokenizer_factory,
            )
            self.assertIs(loaded.model, model)
            self.assertIs(loaded.tokenizer, tokenizer)
            self.assertIs(loaded.io.model, model)
            self.assertIs(loaded.io.tokenizer, tokenizer)
            self.assertFalse(loaded.production_file_identity_verified)
            self.assertEqual(loaded.manifest.model_ref, runner.QUALIFIED_MODEL_REF)
            self.assertEqual(
                loaded.manifest.tokenizer_ref,
                runner.QUALIFIED_TOKENIZER_REF,
            )
            self.assertEqual(loaded.guard.probe(), loaded.guard.initial_tensor_digest)
            self.assertEqual(
                loaded.guard.verify_boundary_digest(),
                loaded.guard.initial_tensor_digest,
            )
            self.assertEqual([name for name, _, _ in calls], ["tokenizer", "model"])
            self.assertTrue(all(path == model_root for _, path, _ in calls))
            self.assertEqual(
                calls[0][2],
                {
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
            )
            self.assertEqual(
                calls[1][2],
                {
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "dtype": torch.bfloat16,
                    "device_map": {"": "cuda:0"},
                    "low_cpu_mem_usage": True,
                },
            )
            self.assertTrue(all(not value.requires_grad for value in model.parameters()))
            self.assertFalse(model.training)


class HighLevelMultidomainExactQwenIntegrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_live_evaluation_owner_constructs_four_real_lineages(self) -> None:
        class _Cuda:
            def __init__(self) -> None:
                self.reset_calls = 0
                self.synchronize_calls = 0
                self.empty_cache_calls = 0

            def reset_peak_memory_stats(self, device: int) -> None:
                if device != 0:
                    raise AssertionError("unexpected fake CUDA device")
                self.reset_calls += 1

            def max_memory_allocated(self, device: int) -> int:
                if device != 0:
                    raise AssertionError("unexpected fake CUDA device")
                return 0

            def max_memory_reserved(self, device: int) -> int:
                if device != 0:
                    raise AssertionError("unexpected fake CUDA device")
                return 0

            def synchronize(self, device: int) -> None:
                if device != 0:
                    raise AssertionError("unexpected fake CUDA device")
                self.synchronize_calls += 1

            def empty_cache(self) -> None:
                self.empty_cache_calls += 1

        class _EvaluationClient:
            def __init__(self, snapshot: dict[str, object]) -> None:
                self.commitment_snapshot = snapshot
                self.invalidate_calls = 0

            def commitments(self) -> dict[str, object]:
                return self.commitment_snapshot

            def release_phase(self, phase: str) -> tuple[()]:
                del phase
                return ()

            def complete_phase(self, phase: str) -> None:
                del phase

            def admit_final(self, admission_digest: str) -> None:
                del admission_digest

            def judge_response(self, *args: object) -> None:
                del args

            def random_feedback(self, replicate: str) -> tuple[()]:
                del replicate
                return ()

            def final_metrics(self) -> dict[str, object]:
                return {}

            def close(self) -> None:
                pass

            def invalidate(self) -> None:
                self.invalidate_calls += 1

        def foundation_evidence() -> dict[str, object]:
            return {
                "fastembed_cache_tree": dict(
                    runner.EXPECTED_FASTEMBED_CACHE_TREE
                ),
                "file_count": runner.QUALIFIED_MODEL_FILE_COUNT,
                "model_cache_tree": {
                    "exists": False,
                    "files": 0,
                    "manifest_sha256": hashlib.sha256(b"ABSENT\n").hexdigest(),
                    "total_bytes": 0,
                },
                "root_manifest_sha256": runner.QUALIFIED_MODEL_ROOT_SHA256,
                "schema": runner.FOUNDATION_LOAD_SCHEMA,
                "tiktoken_cache_tree": dict(
                    runner.EXPECTED_TIKTOKEN_CACHE_TREE
                ),
                "tokenizer_manifest_sha256": runner.QUALIFIED_TOKENIZER_SHA256,
                "total_bytes": runner.QUALIFIED_MODEL_TOTAL_BYTES,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            scratch = root / "scratch"
            sealed = state / "sealed"
            results = root / "results"
            qualification_runtime = state / "qualification-runtime"
            qualification_scopes = state / "qualification-scopes"
            model_root = root / "model"
            repository_guard = root / "repository-guard"
            for path in (
                state,
                scratch,
                sealed,
                results,
                qualification_runtime,
                qualification_scopes,
                model_root,
                repository_guard,
            ):
                path.mkdir(mode=0o700)
            runtime_root = state / "evaluation-runtime"
            scope_root = state / "evaluation-scopes"
            manifest_path = root / "manifest.json"
            seed_path = sealed / "seeds.json"
            qualification_path = results / "qualification.json"
            result_path = results / "evaluation.json"
            qualification_release_path = sealed / "qualification-release.json"
            claim_path = sealed / "evaluation-claim.json"

            raw_seeds = (bytes(range(32)), bytes(range(32, 64)))
            commitments = tuple(
                high_evaluator.commit_replicate_seed(value)
                for value in raw_seeds
            )
            committed = high_evaluator.make_evaluation_evaluator(
                raw_seeds,
                commitments,
                _ref("evaluation-owner-preconstruction-admission"),
            )
            source_hashes = runner.completed_source_manifest(runner.REPOSITORY_ROOT)
            leaf_sha256 = hashlib.sha256(
                runner.ACTIVE_LEAF_PATH.read_bytes()
            ).hexdigest()
            public_manifest = runner.build_source_manifest(
                evaluator_commitments=committed.commitments.to_canonical(),
                seed_commitments=commitments,
                accepted_leaf_sha256=leaf_sha256,
                completed_source_hashes=source_hashes,
            )
            manifest_sha256 = hashlib.sha256(
                runner.canonical_json_bytes(public_manifest)
            ).hexdigest()
            qualification = {"classification": "QUALIFICATION_PASS"}
            qualification_sha256 = hashlib.sha256(
                b"synthetic-qualified-result"
            ).hexdigest()
            seal = runner.SeedSeal(
                schema=runner.SEED_SEAL_SCHEMA,
                seed_commitments=commitments,
                artifact_sha256=hashlib.sha256(b"synthetic-seed-bytes").hexdigest(),
                artifact_bytes=1,
                path=str(seed_path),
            )
            claim, claim_ref = runner.build_evaluation_admission_claim(
                manifest_sha256=manifest_sha256,
                qualification_result_sha256=qualification_sha256,
                evaluator_commitment_digest=public_manifest[
                    "evaluator_commitment_digest"
                ],
                source_hashes=source_hashes,
                manifest_path=manifest_path,
                seed_path=seed_path,
                qualification_result_path=qualification_path,
                evaluation_result_path=result_path,
                claim_path=claim_path,
                repository_root=runner.REPOSITORY_ROOT,
                seed_seal_ref=seal.seal_ref,
                path_binding_mode="INJECTED_CPU_TEST",
            )
            runner._write_consuming_admission_claim(claim_path, claim)
            payload = {
                "claim": claim,
                "claim_path": str(claim_path),
                "claim_ref": claim_ref,
                "manifest": public_manifest,
                "manifest_sha256": manifest_sha256,
                "qualification_result": qualification,
                "qualification_result_sha256": qualification_sha256,
                "result_path": str(result_path),
                "seed_seal": seal.to_canonical(),
                "seed_seal_ref": seal.seal_ref,
            }

            cycle_manifest = runner.qualified_qwen_cycle_manifest()
            model = _TinyModel(cycle_manifest.input_width)
            model.requires_grad_(False)
            model.eval()
            tokenizer = _TinyTokenizer()
            io = LocalQwenIO(
                model,
                tokenizer,
                embedding_batch_size=cycle_manifest.embedding_batch_size,
                max_input_tokens=cycle_manifest.max_input_tokens,
                max_new_tokens=cycle_manifest.max_new_tokens,
                enable_thinking=False,
                model_ref=cycle_manifest.model_ref,
                tokenizer_ref=cycle_manifest.tokenizer_ref,
            )
            foundation_guard = runner.FoundationGuard(model)
            foundation_before = _model_tensors(model)
            evidence = foundation_evidence()
            service = _ScopedFakeAcquisitionService()
            cuda = _Cuda()
            torch_fake = SimpleNamespace(cuda=cuda)
            evaluator_clients: list[_EvaluationClient] = []
            evaluator_specs: list[runner.EvaluatorWorkerSpec] = []

            def evaluator_factory(
                spec: runner.EvaluatorWorkerSpec,
            ) -> _EvaluationClient:
                evaluator_specs.append(spec)
                client = _EvaluationClient(
                    {
                        "commitments": public_manifest[
                            "evaluator_commitments"
                        ],
                        "digest": public_manifest[
                            "evaluator_commitment_digest"
                        ],
                    }
                )
                evaluator_clients.append(client)
                return client

            def foundation_loader() -> runner.LoadedQwenFoundation:
                return runner.LoadedQwenFoundation(
                    model=model,
                    tokenizer=tokenizer,
                    io=io,
                    manifest=cycle_manifest,
                    guard=foundation_guard,
                    production_file_evidence=evidence,
                )

            repository_before = runner._repository_tree_manifest(
                repository_guard
            )
            preconstruction = (
                public_manifest,
                manifest_sha256,
                qualification,
                qualification_sha256,
                seal,
                {"path_binding_mode": "INJECTED_CPU_TEST"},
                _ref("qualification-release"),
            )

            async def owner_factory(
                owner_payload: dict[str, object],
            ) -> runner._LiveDependenciesOwner:
                return await runner._build_live_evaluation_owner(
                    owner_payload,
                    parent_boundary_validator=lambda: {"boundary": "exact"},
                    expected_parent_boundary={"boundary": "exact"},
                    evaluator_client_factory=evaluator_factory,
                    genesis_factory=runner.reconstruct_fresh_cpu_genesis,
                    foundation_loader=foundation_loader,
                    foundation_model_root=model_root,
                    foundation_file_verifier=lambda path: dict(evidence),
                    arm_factory_builder=runner.HighLevelArmFactory.create,
                    acquisition_opener=service,
                    torch_module=torch_fake,
                    state_root=state,
                    scratch_root=scratch,
                    qualification_runtime_root=qualification_runtime,
                    qualification_scope_root=qualification_scopes,
                    qualification_release_path=qualification_release_path,
                    runtime_root=runtime_root,
                    scope_root=scope_root,
                    expected_repository_tree=repository_before,
                    repository_tree_probe=lambda: runner._repository_tree_manifest(
                        repository_guard
                    ),
                    injected_boundaries=True,
                )

            with mock.patch.object(
                runner,
                "_load_evaluation_preconstruction_evidence",
                return_value=preconstruction,
            ), mock.patch.object(
                runner,
                "_validate_evaluation_preconstruction_roots",
            ), mock.patch.object(
                runner,
                "SEALED_SEED_PATH",
                seed_path,
            ):
                owner = await runner._construct_live_dependencies_owner(
                    owner_factory,
                    payload,
                    purpose="evaluation",
                )
                executor = owner.dependencies.execute_arm
                self.assertIs(type(executor), runner._LiveArmExecutorOwner)
                factory = executor.factory
                self.assertIs(type(factory), runner.HighLevelArmFactory)
                assert factory is not None
                expected_lineages = {
                    (commitment, arm)
                    for commitment in commitments
                    for arm in ("FULL", "RANDOM_FEEDBACK")
                }
                self.assertEqual(set(factory.lineages), expected_lineages)
                self.assertEqual(factory.purpose, "evaluation")
                self.assertEqual(
                    len({owner.initial_hashes for owner in factory.lineages.values()}),
                    1,
                )
                self.assertEqual(
                    owner.dependencies.expected_learner_genesis_digests,
                    {
                        commitment: runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST
                        for commitment in commitments
                    },
                )
                self.assertEqual(
                    owner.dependencies.accounting.mode,
                    runner.RunMode.EVALUATION,
                )
                self.assertEqual(sum(service.active.values()), 0)
                owner.dependencies.accounting._finalizer()
                await owner.close()

            self.assertEqual(len(evaluator_specs), 1)
            self.assertFalse(evaluator_specs[0].commitment_only)
            self.assertEqual(evaluator_specs[0].expected_commitments, commitments)
            self.assertEqual(
                evaluator_specs[0].expected_final_admission,
                claim["admission_digest"],
            )
            self.assertEqual(evaluator_clients[0].invalidate_calls, 1)
            self.assertEqual(sum(service.active.values()), 0)
            self.assertEqual(cuda.reset_calls, 1)
            self.assertEqual(cuda.synchronize_calls, 1)
            self.assertEqual(cuda.empty_cache_calls, 1)
            self.assertTrue((runtime_root / "arm-factory-audit.json").is_file())
            self.assertTrue(
                _same_model_tensors(_model_tensors(model), foundation_before)
            )

    async def test_managed_factory_separates_lineages_and_runs_seven_arm_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genesis = runner.reconstruct_fresh_cpu_genesis(root / "genesis")
            manifest = runner.qualified_qwen_cycle_manifest()
            model = _TinyModel(manifest.input_width)
            model.requires_grad_(False)
            model.eval()
            tokenizer = _TinyTokenizer()
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
            guard = runner.FoundationGuard(model)
            foundation_before = _model_tensors(model)
            service = _ScopedFakeAcquisitionService()
            replicate_commitments = (
                _ref("managed-factory-replicate-01"),
                _ref("managed-factory-replicate-02"),
            )
            factory = await runner.HighLevelArmFactory.create(
                root / "factory",
                purpose="qualification",
                replicate_commitments=replicate_commitments,
                genesis=genesis,
                io=io,
                manifest=manifest,
                foundation_guard=guard,
                acquisition_opener=service,
            )
            owners = tuple(factory.lineages.values())
            self.assertEqual(len(owners), 4)
            self.assertEqual(runner.QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE, 1)
            self.assertEqual(
                len({owner.initial_hashes for owner in owners}),
                1,
            )
            for owner in owners:
                self.assertIs(type(owner), runner.LineageBaselineOwner)
                head = owner.store.head()
                self.assertIsNotNone(head)
                assert head is not None and head.episode_ref is not None
                self.assertEqual(
                    head.sequence,
                    runner.QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE,
                )
                self.assertEqual(head.state_digest, owner.genesis_digest)
                self.assertEqual(
                    owner.genesis_digest,
                    runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST,
                )
                expected_episode = runner._qualified_unrelated_bootstrap_episode(
                    runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST,
                    manifest,
                )
                self.assertEqual(
                    owner.store.get_episode(head.episode_ref),
                    expected_episode,
                )
                self.assertFalse(owner.store.pending_acquisition_projections(limit=1))
                self.assertEqual(owner.paths.learner.stat().st_mode & 0o7777, 0o600)

            ledger = runner.EvidenceLedger(
                root / "evidence" / "attempts.sqlite3",
                _ref("managed-factory-run-intent"),
            )
            budget = runner.AttemptBudget(
                proposal_ceiling=9,
                execution_ceiling=9,
                arm_task_ceiling=9,
            )
            evaluator = _ObjectiveFakeEvaluator()
            adaptation_task = _public_task(
                phase="adaptation",
                ordinal=0,
                replicate_commitment=replicate_commitments[0],
            )
            model.queue(
                _PROPOSAL_RESPONSE,
                _EXECUTION_RESPONSE,
                _PROPOSAL_RESPONSE,
                _EXECUTION_RESPONSE,
            )
            full_adaptation = await factory.execute(
                runner.ArmTaskSpec(
                    purpose="qualification",
                    phase="adaptation",
                    arm="FULL",
                    task=adaptation_task,
                    judge_arm="QUALIFICATION",
                ),
                evaluator=evaluator,
                ledger=ledger,
                budget=budget,
            )
            random_adaptation = await factory.execute(
                runner.ArmTaskSpec(
                    purpose="qualification",
                    phase="adaptation",
                    arm="RANDOM_FEEDBACK",
                    task=adaptation_task,
                    judge_arm=None,
                    feedback_value=0.0,
                ),
                evaluator=evaluator,
                ledger=ledger,
                budget=budget,
            )
            self.assertNotEqual(
                full_adaptation.integrity.learner_child_digest,
                random_adaptation.integrity.learner_child_digest,
            )
            full_owner = factory.lineages[(replicate_commitments[0], "FULL")]
            random_owner = factory.lineages[
                (replicate_commitments[0], "RANDOM_FEEDBACK")
            ]
            first_learned_sequence = (
                runner.QUALIFIED_BOOTSTRAP_TRANSACTION_SEQUENCE + 1
            )
            self.assertEqual(first_learned_sequence, 2)
            self.assertEqual(
                full_adaptation.integrity.learner_sequence,
                first_learned_sequence,
            )
            self.assertEqual(
                random_adaptation.integrity.learner_sequence,
                first_learned_sequence,
            )
            self.assertEqual(
                full_adaptation.integrity.learner_parent_digest,
                runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST,
            )
            self.assertEqual(
                random_adaptation.integrity.learner_parent_digest,
                runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST,
            )
            full_head = full_owner.store.head()
            random_head = random_owner.store.head()
            self.assertIsNotNone(full_head)
            self.assertIsNotNone(random_head)
            assert full_head is not None and random_head is not None
            self.assertEqual(full_head.sequence, first_learned_sequence)
            self.assertEqual(random_head.sequence, first_learned_sequence)
            self.assertEqual(
                full_head.state_digest,
                full_adaptation.integrity.learner_child_digest,
            )
            self.assertEqual(
                random_head.state_digest,
                random_adaptation.integrity.learner_child_digest,
            )
            self.assertNotEqual(full_owner.hashes(), random_owner.hashes())
            adapted_full_hashes = full_owner.hashes()

            proposed_refs = tuple(
                item.record_ref
                for item in full_owner.store.acquisition_items(limit=64)
                if item.acquisition.record.epistemic_status.value == "PROPOSED"
            )
            self.assertTrue(proposed_refs)
            development_task = _public_task(
                phase="development",
                ordinal=1,
                replicate_commitment=replicate_commitments[0],
            )
            comparison_scope = runner.AcquisitionScopeSpec.for_clone(
                run_label="qualification",
                replicate=replicate_commitments[0],
                arm="FULL",
                task_id=str(development_task["task_id"]),
                state_parent=factory.scopes_parent,
            )
            service.raw_extra_refs[comparison_scope.dataset_name] = proposed_refs[0]
            model.queue(
                *(
                    response
                    for _arm in runner.EVALUATION_ARMS
                    for response in (_PROPOSAL_RESPONSE, _EXECUTION_RESPONSE)
                )
            )
            results: dict[str, runner.ArmTaskResult] = {}
            for arm in runner.EVALUATION_ARMS:
                results[arm] = await factory.execute(
                    runner.ArmTaskSpec(
                        purpose="qualification",
                        phase="development",
                        arm=arm,
                        task=development_task,
                        judge_arm=("QUALIFICATION" if arm == "FULL" else None),
                    ),
                    evaluator=evaluator,
                    ledger=ledger,
                    budget=budget,
                )

            full_batch = results["FULL"].frozen_recall
            self.assertIsNotNone(full_batch)
            assert full_batch is not None
            self.assertNotIn(
                proposed_refs[0],
                tuple(item.record_ref for item in full_batch.items),
            )
            self.assertNotIn(
                proposed_refs[0],
                tuple(
                    adjacent_ref
                    for item in full_batch.items
                    for adjacent_ref in item.adjacent_record_refs
                ),
            )
            for arm in (
                "FULL",
                "RETRIEVAL_ONLY",
                "FROZEN_ORIGIN",
                "PROSPECTIVE_REMOVAL",
                "BACKEND_REMOVAL",
            ):
                self.assertEqual(results[arm].frozen_recall, full_batch, arm)
            self.assertEqual(
                {
                    arm: results[arm].evidence["backend_search_calls"]
                    for arm in (
                        "FULL",
                        "RETRIEVAL_ONLY",
                        "FROZEN_ORIGIN",
                        "PROSPECTIVE_REMOVAL",
                        "BACKEND_REMOVAL",
                    )
                },
                {
                    "FULL": 1,
                    "RETRIEVAL_ONLY": 0,
                    "FROZEN_ORIGIN": 1,
                    "PROSPECTIVE_REMOVAL": 1,
                    "BACKEND_REMOVAL": 0,
                },
            )
            self.assertGreaterEqual(
                results["FULL"].evidence["backend_rejected_hit_count"],
                1,
            )
            self.assertEqual(
                results["FROZEN_ORIGIN"].evidence["backend_rejected_hit_count"],
                results["FULL"].evidence["backend_rejected_hit_count"],
            )
            self.assertEqual(
                results["PROSPECTIVE_REMOVAL"].evidence[
                    "backend_rejected_hit_count"
                ],
                results["FULL"].evidence["backend_rejected_hit_count"],
            )
            runner.validate_removal_fairness(
                results["FULL"].evidence,
                results["FROZEN_ORIGIN"].evidence,
                results["PROSPECTIVE_REMOVAL"].evidence,
                results["BACKEND_REMOVAL"].evidence,
                results["RETRIEVAL_ONLY"].evidence,
                require_score=False,
            )
            self.assertIn(
                runner.NO_PERSISTENT_RECALLED_EVIDENCE_V1,
                "\n".join(tokenizer.chat_prompts),
            )
            self.assertEqual(full_owner.hashes(), adapted_full_hashes)
            self.assertFalse(any(factory.clones_parent.iterdir()))
            self.assertFalse(any(factory.controls_parent.iterdir()))
            self.assertTrue(all(value == 0 for value in service.active.values()))
            self.assertEqual(service.maximum_active, 1)
            self.assertFalse(factory._active_scopes)
            self.assertFalse(factory._active_lineages)
            factory_audit = factory.audit_lineages()
            dispositions = factory_audit["managed_dispositions"]
            disposition_paths = tuple(
                factory.dispositions_parent.glob("*.json")
            )
            self.assertEqual(len(dispositions), 9)
            self.assertEqual(len(disposition_paths), 9)
            for envelope in dispositions:
                disposition = envelope["disposition"]
                self.assertEqual(
                    envelope["disposition_ref"],
                    runner.content_ref(
                        "managed-runtime-disposition",
                        disposition,
                    ),
                )
                self.assertEqual(
                    disposition["schema"],
                    runner.MANAGED_RUNTIME_DISPOSITION_SCHEMA,
                )
                disposable_root = disposition["disposable_root"]
                if disposable_root is not None:
                    self.assertFalse(Path(disposable_root).exists())
                if disposition["arm"] == "QWEN_ONLY":
                    self.assertIsNone(disposition["namespace"])
                    self.assertEqual(disposition["namespace_rebuild_calls"], 0)
                    self.assertIsNone(disposition["recall_provider_disposed"])
                else:
                    self.assertEqual(disposition["namespace_rebuild_calls"], 1)
                    self.assertTrue(disposition["namespace_forget_completed"])
                    self.assertTrue(disposition["binding_close_completed"])
                    self.assertTrue(disposition["scope_lease_released"])
                    self.assertTrue(disposition["recall_provider_disposed"])
                    self.assertIn(
                        disposition["recall_provider_kind"],
                        ("CAPTURE_ONCE", "REPLAY_ONLY"),
                    )
            self.assertTrue(
                all(
                    path.stat().st_mode & 0o7777 == 0o600
                    for path in disposition_paths
                )
            )
            retained_root = Path(
                next(
                    disposition["disposition"]["disposable_root"]
                    for disposition in dispositions
                    if disposition["disposition"]["disposable_root"] is not None
                )
            )
            retained_root.mkdir(mode=0o700)
            try:
                with self.assertRaisesRegex(
                    runner.RunnerInvariantError,
                    "root was not removed",
                ):
                    factory.audit_lineages()
            finally:
                retained_root.rmdir()
            disposition_path = disposition_paths[0]
            original_disposition = disposition_path.read_bytes()
            disposition_path.write_bytes(b'{"tampered":true}')
            try:
                with self.assertRaises(runner.RunnerInvariantError):
                    factory.audit_lineages()
            finally:
                disposition_path.write_bytes(original_disposition)
            factory.audit_lineages()
            self.assertEqual(guard.verify_boundary_digest(), guard.initial_tensor_digest)
            self.assertTrue(_same_model_tensors(_model_tensors(model), foundation_before))
            self.assertEqual(model.responses, [])

    async def test_managed_random_feedback_malformed_is_one_consumed_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genesis = runner.reconstruct_fresh_cpu_genesis(root / "genesis")
            manifest = runner.qualified_qwen_cycle_manifest()
            model = _TinyModel(manifest.input_width)
            model.requires_grad_(False)
            model.eval()
            tokenizer = _TinyTokenizer()
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
            guard = runner.FoundationGuard(model)
            foundation_before = _model_tensors(model)
            service = _ScopedFakeAcquisitionService()
            replicate_commitment = _ref(
                "managed-malformed-random-feedback-replicate"
            )
            factory = await runner.HighLevelArmFactory.create(
                root / "factory",
                purpose="qualification",
                replicate_commitments=(replicate_commitment,),
                genesis=genesis,
                io=io,
                manifest=manifest,
                foundation_guard=guard,
                acquisition_opener=service,
            )
            random_owner = factory.lineages[
                (replicate_commitment, "RANDOM_FEEDBACK")
            ]
            lineage_before = random_owner.hashes()
            ledger = runner.EvidenceLedger(
                root / "evidence" / "attempts.sqlite3",
                _ref("managed-malformed-run-intent"),
            )
            budget = runner.AttemptBudget(
                proposal_ceiling=1,
                execution_ceiling=0,
                arm_task_ceiling=1,
            )
            task = _public_task(
                phase="adaptation",
                ordinal=0,
                replicate_commitment=replicate_commitment,
            )
            malformed = '{"procedures":["only one"]}'
            model.queue(malformed)

            result = await factory.execute(
                runner.ArmTaskSpec(
                    purpose="qualification",
                    phase="adaptation",
                    arm="RANDOM_FEEDBACK",
                    task=task,
                    judge_arm=None,
                    feedback_value=0.0,
                ),
                evaluator=None,
                ledger=ledger,
                budget=budget,
            )

            self.assertEqual(result.parser_disposition, "MALFORMED")
            self.assertIsNone(result.judgment)
            self.assertEqual(result.raw_response, "")
            self.assertIsNone(result.execution_request)
            self.assertIsNone(result.execution_receipt)
            self.assertIsNone(result.journal_entry)
            self.assertIsNone(result.learner_pending_material)
            self.assertEqual(result.evidence["proposals"], [])
            for key in (
                "execution_request_ref",
                "execution_receipt_ref",
                "selection_ref",
                "task_response_generation_ref",
            ):
                self.assertIsNone(result.evidence[key], key)
            self.assertIsNone(result.integrity.learner_child_digest)
            self.assertIsNone(result.integrity.learner_sequence)
            self.assertIsNotNone(result.frozen_recall)
            assert result.frozen_recall is not None
            self.assertEqual(
                result.evidence["frozen_recall_ref"],
                result.frozen_recall.batch_ref,
            )
            self.assertEqual(result.evidence["backend_search_calls"], 1)
            self.assertEqual(len(service.search_calls), 1)
            self.assertIsNotNone(result.proposal_generation)
            assert result.proposal_generation is not None
            self.assertEqual(result.proposal_generation.response, malformed)
            self.assertEqual(
                result.proposal_generation.generated_token_ids,
                tuple(malformed.encode("utf-8")),
            )
            self.assertEqual(model.generate_calls, 1)
            self.assertEqual(io.generation_calls, 1)
            self.assertEqual(len(tokenizer.chat_prompts), 1)
            self.assertIn(QWEN_PROCEDURE_PROPOSAL_SCHEMA, tokenizer.chat_prompts[0])
            self.assertNotIn(
                QWEN_SELECTED_PROCEDURE_EXECUTION_SCHEMA,
                tokenizer.chat_prompts[0],
            )
            self.assertEqual(model.responses, [])
            self.assertEqual(
                budget.snapshot(),
                {
                    "arm_tasks": 1,
                    "execution_attempts": 0,
                    "generation_attempts": 1,
                    "proposal_attempts": 1,
                },
            )
            self.assertEqual(random_owner.hashes(), lineage_before)
            self.assertIsNone(random_owner.learner.pending_decision)
            self.assertIsNone(random_owner.store.active_prospective_turn())
            random_owner.store.audit_integrity()
            random_owner.journal.audit_integrity()
            rows = ledger.read_all()
            self.assertEqual(len(rows), 1)
            attempt = rows[0]["attempt"]
            judgment = rows[0]["judgment"]
            self.assertEqual(attempt["arm"], "RANDOM_FEEDBACK")
            self.assertEqual(attempt["parser_disposition"], "MALFORMED")
            self.assertEqual(attempt["proposals"], [])
            for key in (
                "execution_request",
                "execution_receipt",
                "selection",
                "task_response_generation",
            ):
                self.assertIsNone(attempt[key], key)
            self.assertIsInstance(judgment, dict)
            assert isinstance(judgment, dict)
            for key in (
                "feedback_record",
                "learner_transition",
                "objective_judgment",
                "unevaluated_resolution",
            ):
                self.assertIsNone(judgment[key], key)
            ledger.audit()
            self.assertFalse(any(factory.clones_parent.iterdir()))
            self.assertFalse(any(factory.controls_parent.iterdir()))
            self.assertTrue(all(value == 0 for value in service.active.values()))
            self.assertEqual(service.maximum_active, 1)
            self.assertFalse(factory._active_scopes)
            self.assertFalse(factory._active_lineages)
            audit = factory.audit_lineages()
            dispositions = audit["managed_dispositions"]
            self.assertEqual(len(dispositions), 1)
            disposition = dispositions[0]["disposition"]
            self.assertEqual(disposition["arm"], "RANDOM_FEEDBACK")
            self.assertEqual(disposition["phase"], "adaptation")
            self.assertIsNone(disposition["disposable_root"])
            self.assertEqual(disposition["namespace_rebuild_calls"], 1)
            self.assertTrue(disposition["namespace_forget_completed"])
            self.assertTrue(disposition["binding_close_completed"])
            self.assertTrue(disposition["scope_lease_released"])
            self.assertTrue(disposition["recall_provider_disposed"])
            self.assertEqual(disposition["recall_provider_kind"], "CAPTURE_ONCE")
            self.assertEqual(disposition["lineage_hashes"], lineage_before)
            disposition_paths = tuple(factory.dispositions_parent.glob("*.json"))
            self.assertEqual(len(disposition_paths), 1)
            self.assertEqual(disposition_paths[0].stat().st_mode & 0o7777, 0o600)
            self.assertEqual(guard.verify_boundary_digest(), guard.initial_tensor_digest)
            self.assertTrue(_same_model_tensors(_model_tensors(model), foundation_before))

    async def test_qwen_post_finalization_journal_mutation_is_retained_and_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genesis = runner.reconstruct_fresh_cpu_genesis(root / "genesis")
            manifest = runner.qualified_qwen_cycle_manifest()
            model = _TinyModel(manifest.input_width)
            model.requires_grad_(False)
            model.eval()
            tokenizer = _TinyTokenizer()
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
            guard = runner.FoundationGuard(model)
            service = _ScopedFakeAcquisitionService()
            replicate_commitment = _ref("qwen-mutation-replicate")
            factory = await runner.HighLevelArmFactory.create(
                root / "factory",
                purpose="qualification",
                replicate_commitments=(replicate_commitment,),
                genesis=genesis,
                io=io,
                manifest=manifest,
                foundation_guard=guard,
                acquisition_opener=service,
            )
            ledger = runner.EvidenceLedger(
                root / "evidence" / "attempts.sqlite3",
                _ref("qwen-mutation-run-intent"),
            )
            budget = runner.AttemptBudget(
                proposal_ceiling=1,
                execution_ceiling=1,
                arm_task_ceiling=1,
            )
            task = _public_task(
                phase="development",
                ordinal=0,
                replicate_commitment=replicate_commitment,
            )
            spec = runner.ArmTaskSpec(
                purpose="qualification",
                phase="development",
                arm="QWEN_ONLY",
                task=task,
                judge_arm=None,
            )
            model.queue(_PROPOSAL_RESPONSE, _EXECUTION_RESPONSE)
            original_finalize = runner.EvidenceLedger.finalize_attempt
            mutated_path: Path | None = None

            def finalize_then_mutate(
                owned: runner.EvidenceLedger,
                key: tuple[str, str, str],
                final: dict[str, object],
            ) -> str:
                nonlocal mutated_path
                final_ref = original_finalize(owned, key, final)
                journals = tuple(
                    factory.controls_parent.rglob("qwen-journal.sqlite3")
                )
                if len(journals) != 1:
                    raise AssertionError("QWEN mutation witness lacks one journal")
                mutated_path = journals[0]
                with mutated_path.open("ab") as stream:
                    stream.write(b"POST-FINALIZATION-MUTATION")
                    stream.flush()
                    os.fsync(stream.fileno())
                return final_ref

            with mock.patch.object(
                runner.EvidenceLedger,
                "finalize_attempt",
                new=finalize_then_mutate,
            ):
                with self.assertRaisesRegex(
                    BaseExceptionGroup,
                    "managed arm cleanup failed",
                ):
                    await factory.execute(
                        spec,
                        evaluator=_ObjectiveFakeEvaluator(),
                        ledger=ledger,
                        budget=budget,
                    )
            self.assertIsNotNone(mutated_path)
            assert mutated_path is not None
            self.assertTrue(mutated_path.exists())
            self.assertTrue(mutated_path.parent.exists())
            self.assertFalse(any(factory.dispositions_parent.iterdir()))
            with self.assertRaisesRegex(
                runner.RunnerInvariantError,
                "disposition set differs",
            ):
                factory.audit_lineages()

    async def test_fixed_removals_share_frozen_inputs_and_foundation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genesis = _learner()
            manifest = _manifest(genesis)
            baseline = _build_baseline(root, manifest)
            model = _TinyModel(manifest.input_width)
            model.requires_grad_(False)
            model.eval()
            tokenizer = _TinyTokenizer()
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
            foundation_before = _model_tensors(model)

            arms = (
                "FULL",
                "FROZEN_ORIGIN",
                "PROSPECTIVE_REMOVAL",
                "BACKEND_REMOVAL",
            )
            results: dict[str, _ArmResult] = {}
            for arm in arms:
                runtime = _clone_runtime(
                    root,
                    arm=arm,
                    baseline=baseline,
                    io=io,
                    manifest=manifest,
                )
                results[arm] = await _run_arm(
                    runtime,
                    model=model,
                    tokenizer=tokenizer,
                )

            full = results["FULL"]
            frozen = results["FROZEN_ORIGIN"]
            prospective = results["PROSPECTIVE_REMOVAL"]
            backend = results["BACKEND_REMOVAL"]

            # One reference-only manifest controls every arm; only BACKEND skips search.
            self.assertTrue(
                all(
                    result.runtime.hit_manifest is baseline.hit_manifest
                    for result in results.values()
                )
            )
            self.assertEqual(full.runtime.backend.search_calls, 1)
            self.assertEqual(frozen.runtime.backend.search_calls, 1)
            self.assertEqual(prospective.runtime.backend.search_calls, 1)
            self.assertEqual(backend.runtime.backend.search_calls, 0)
            self.assertEqual(full.runtime.backend.queries, [(_PUBLIC_TASK, 12)])

            # Public records/text/order, task, proposal prompts, and model proposal
            # payloads are byte-identical before either declared intervention.
            expected_record_bytes = tuple(
                item.record.canonical_bytes() for item in full.recall.items
            )
            expected_text = tuple(item.text for item in full.recall.items)
            expected_refs = tuple(item.artifact_ref for item in full.recall.items)
            expected_proposals = full.executed.turn.decision.proposals
            for result in results.values():
                self.assertEqual(
                    tuple(item.record.canonical_bytes() for item in result.recall.items),
                    expected_record_bytes,
                )
                self.assertEqual(tuple(item.text for item in result.recall.items), expected_text)
                self.assertEqual(
                    tuple(item.artifact_ref for item in result.recall.items), expected_refs
                )
                self.assertEqual(result.executed.execution_request.task_id, _TASK_ID)
                self.assertEqual(result.executed.execution_request.request, _PUBLIC_TASK)
                self.assertEqual(
                    result.executed.execution_request.input_observations,
                    _PUBLIC_OBSERVATIONS,
                )
                self.assertEqual(result.proposal_prompt, full.proposal_prompt)
                self.assertEqual(result.proposal_generation, full.proposal_generation)
                self.assertEqual(result.executed.turn.decision.proposals, expected_proposals)
                self.assertEqual(result.execution_prompt, full.execution_prompt)
                self.assertEqual(result.executed.execution.response, _EXECUTION_RESPONSE)

            # BACKEND is a pure search lesion: same selection, executor input,
            # generated output, durable journal entry, and receipt with zero searches.
            self.assertEqual(backend.recall, full.recall)
            self.assertEqual(
                backend.executed.execution_request,
                full.executed.execution_request,
            )
            self.assertEqual(backend.execution_prompt, full.execution_prompt)
            self.assertEqual(backend.executed.execution, full.executed.execution)
            self.assertEqual(
                backend.executed.execution_receipt,
                full.executed.execution_receipt,
            )
            self.assertEqual(backend.journal_entry, full.journal_entry)
            self.assertEqual(
                backend.executed.execution_receipt.response,
                _EXECUTION_RESPONSE,
            )

            # Frozen Origin changes only temporal position/features before any
            # downstream selection consequence; canonical content remains exact.
            self.assertTrue(any(item.age > 0 for item in full.recall.items))
            self.assertTrue(all(item.age == 0 for item in frozen.recall.items))
            for full_item, frozen_item in zip(
                full.recall.items, frozen.recall.items, strict=True
            ):
                self.assertEqual(full_item.record, frozen_item.record)
                self.assertEqual(full_item.acquisition_ref, frozen_item.acquisition_ref)
                self.assertEqual(full_item.source_contract, frozen_item.source_contract)
                self.assertEqual(full_item.source_ref, frozen_item.source_ref)
                self.assertEqual(full_item.adjacent_records, frozen_item.adjacent_records)
                self.assertEqual(full_item.backend_score, frozen_item.backend_score)
                self.assertEqual(full_item.backend_ref, frozen_item.backend_ref)
                self.assertEqual(
                    full_item.world_valid_at_query,
                    frozen_item.world_valid_at_query,
                )
                self.assertEqual(
                    full_item.position.acquired_ordinal,
                    frozen_item.position.acquired_ordinal,
                )
                self.assertEqual(
                    full_item.position.landmark_relations,
                    frozen_item.position.landmark_relations,
                )
                self.assertEqual(
                    full_item.position.world_valid_from,
                    frozen_item.position.world_valid_from,
                )
                self.assertEqual(
                    full_item.position.world_valid_until,
                    frozen_item.position.world_valid_until,
                )
            full_material = full.runtime.learner.pending_prospective_material
            frozen_material = frozen.runtime.learner.pending_prospective_material
            self.assertIsNotNone(full_material)
            self.assertIsNotNone(frozen_material)
            assert full_material is not None and frozen_material is not None
            self.assertTrue(
                _same_tensor(
                    full_material.relation_features,
                    frozen_material.relation_features,
                )
            )
            self.assertTrue(
                _same_tensor(full_material.base_logits, frozen_material.base_logits)
            )
            self.assertFalse(
                _same_tensor(
                    full_material.temporal_features,
                    frozen_material.temporal_features,
                )
            )

            # Prospective removal receives the same live recall/candidate inputs;
            # the frozen read flag is the sole lesion at that boundary.
            prospective_material = prospective.runtime.learner.pending_prospective_material
            self.assertIsNotNone(prospective_material)
            assert prospective_material is not None
            self.assertEqual(prospective.recall, full.recall)
            self.assertTrue(full_material.prospective_read_enabled)
            self.assertFalse(prospective_material.prospective_read_enabled)
            for field in ("relation_features", "temporal_features", "base_logits"):
                self.assertTrue(
                    _same_tensor(
                        getattr(full_material, field),
                        getattr(prospective_material, field),
                    ),
                    field,
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
                    getattr(prospective_material, field),
                    field,
                )

            self.assertTrue(_same_model_tensors(_model_tensors(model), foundation_before))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            self.assertFalse(model.training)
            self.assertEqual(model.generate_calls, 8)
            self.assertEqual(io.generation_calls, 8)
            self.assertEqual(model.responses, [])
            self.assertEqual(_file_sha256(baseline.store_path), baseline.store_sha256)
            self.assertEqual(_file_sha256(baseline.journal_path), baseline.journal_sha256)

    async def test_malformed_proposal_is_retained_without_retry_or_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genesis = _learner()
            manifest = _manifest(genesis)
            baseline = _build_baseline(root, manifest)
            model = _TinyModel(manifest.input_width)
            model.requires_grad_(False)
            model.eval()
            tokenizer = _TinyTokenizer()
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
            foundation_before = _model_tensors(model)
            runtime = _clone_runtime(
                root,
                arm="FULL",
                baseline=baseline,
                io=io,
                manifest=manifest,
            )
            await runtime.memory.retry_pending_projections(limit=64)
            runtime.memory.assert_synchronized(runtime.store.acquisition_head())
            malformed = '{"procedures":["only one"]}'
            model.queue(malformed)

            with self.assertRaisesRegex(ValueError, "candidate cardinality"):
                await runtime.cycle.begin_turn(_TASK_ID, _PUBLIC_TASK)

            generation = runtime.text_adapter.last_proposal_generation
            self.assertIsNotNone(generation)
            assert generation is not None
            self.assertEqual(generation.response, malformed)
            self.assertEqual(generation.generated_token_ids, tuple(malformed.encode("utf-8")))
            self.assertEqual(model.generate_calls, 1)
            self.assertEqual(io.generation_calls, 1)
            self.assertEqual(len(tokenizer.chat_prompts), 1)
            self.assertIn(QWEN_PROCEDURE_PROPOSAL_SCHEMA, tokenizer.chat_prompts[0])
            self.assertNotIn(
                QWEN_SELECTED_PROCEDURE_EXECUTION_SCHEMA,
                tokenizer.chat_prompts[0],
            )
            self.assertEqual(model.responses, [])
            self.assertIsNone(runtime.cycle.pending_turn)
            self.assertIsNone(runtime.cycle.executed_turn)
            self.assertIsNone(runtime.learner.pending_decision)
            self.assertIsNone(runtime.store.active_prospective_turn())
            with sqlite3.connect(runtime.journal.path) as connection:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM qwen_execution_journal"
                    ).fetchone()[0]
                )
            self.assertEqual(count, 0)
            runtime.journal.audit_integrity()
            runtime.store.audit_integrity()
            self.assertTrue(_same_model_tensors(_model_tensors(model), foundation_before))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
