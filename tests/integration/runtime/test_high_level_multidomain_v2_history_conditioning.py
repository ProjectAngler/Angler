"""CPU-only integration invariants for the V2 history isolation runtime."""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import struct
import tempfile
import types
import unittest
from unittest import mock


# The construction gate explicitly forbids model or GPU work.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in os.sys.path:
        os.sys.path.insert(0, str(location))


from angler.cognition.contracts import (  # noqa: E402
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    EpistemicStatus,
    ProspectiveCommitment,
)
from angler.memory.cognitive_acquisition import (  # noqa: E402
    CognitiveAcquisition,
    LEGACY_COGNITIVE_EPISODE_CONTRACT,
    PROSPECTIVE_DYNAMICS_BATCH_CONTRACT,
    PROSPECTIVE_RESOLUTION_CONTRACT,
)
from angler.memory.cognitive_acquisition_graph import (  # noqa: E402
    AcquisitionRecall,
    AcquisitionRecallBatch,
    AcquisitionReferenceHit,
    AcquisitionSituatedMemory,
)
from angler.memory.moving_origin import TemporalPosition  # noqa: E402
from angler.reasoning.prospective_dynamics import (  # noqa: E402
    ProspectiveComponentStateIntegrity,
)
from angler.runtime.cognitive_cycle import CognitiveCycle  # noqa: E402
from angler.runtime.cognitive_transaction_store import (  # noqa: E402
    AcquisitionHead,
    CognitiveAcquisitionItem,
    CognitiveTransactionStore,
)
from angler.runtime.prospective_observation import (  # noqa: E402
    SyntheticObservedStateEncoderV1,
)
from angler.runtime.situated_qwen import (  # noqa: E402
    FrozenQwenGenerationRecord,
    LocalQwenIO,
)
from experiments.evaluators import (  # noqa: E402
    high_level_multidomain_v2_history_conditioning as evaluator,
)
from experiments.runners import (  # noqa: E402
    high_level_multidomain_v2_history_conditioning as runner,
)


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _raw(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_inherited_parent_attestation(
    network_namespace: str = "net:[41002]",
) -> dict[str, object]:
    return {
        "capabilities": {
            name: runner._V2_ZERO_CAPABILITY_SET
            for name in runner._V2_CAPABILITY_STATUS_FIELDS
        },
        "gids": [1000, 1000, 1000, 1000],
        "groups": [],
        "interfaces": [[1, "lo"]],
        "ipv4_route_count": 0,
        "ipv6_route_count": 1,
        "loopback_operstate": "unknown",
        "network_namespace": network_namespace,
        "no_new_privileges": True,
        "uids": [1000, 1000, 1000, 1000],
    }


def _result_publication_receipt(
    path: Path,
    result: dict[str, object],
    *,
    result_sha256: str | None = None,
) -> dict[str, object]:
    canonical = runner.canonical_json_bytes(result)
    return {
        "bytes": len(canonical),
        "path": str(path),
        "result_ref": result["result_ref"],
        "sha256": (
            hashlib.sha256(canonical).hexdigest()
            if result_sha256 is None
            else result_sha256
        ),
    }


def _synthetic_preflight_namespace_proof() -> dict[str, object]:
    return {
        "filesystem": "tmpfs",
        "mount_device": "0:32",
        "mount_id": 36,
        "mount_options": ["nodev", "noexec", "nosuid", "rw"],
        "mount_parent_id": 25,
        "mount_root": "/",
        "mount_size_bytes": runner.FOUNDATION_MIRROR_PINNED_SLOT_BYTES,
        "mount_target": "/opt/angler/scratch",
        "parent_mount_namespace": "mnt:[51001]",
        "parent_network_namespace": "net:[51002]",
        "private_child_mode": "0700",
        "private_child_names": ["cache", "tmp"],
        "private_child_owner_gid": 1000,
        "private_child_owner_uid": 1000,
        "private_root": str(runner.FOUNDATION_PREFLIGHT_PRIVATE_ROOT),
        "private_root_mode": "0700",
        "private_root_owner_gid": 1000,
        "private_root_owner_uid": 1000,
        "scratch_mount_mode": "0700",
        "scratch_mount_owner_gid": 1000,
        "scratch_mount_owner_uid": 1000,
        "self_mount_namespace": "mnt:[41001]",
        "self_network_namespace": "net:[41002]",
        "super_options": [
            "gid=1000",
            "mode=700",
            "rw",
            "size=1048576k",
            "uid=1000",
        ],
    }


def _synthetic_preflight_boundary(
    source_inventory_ref: str,
    *,
    unassigned_compute_processes: int = 0,
) -> dict[str, object]:
    namespace_proof = _synthetic_preflight_namespace_proof()
    payload = {
        "assigned_gpu_name": "NVIDIA GeForce RTX 5080",
        "assigned_gpu_uuid": runner.ASSIGNED_GPU_UUID,
        "environment_sha256": hashlib.sha256(
            runner.canonical_json_bytes(
                runner._frozen_foundation_preflight_environment()
            )
        ).hexdigest(),
        "hostname": runner.EXPECTED_HOSTNAME,
        "logical_device_count": 1,
        "logical_device_index": 0,
        "mount_namespace": namespace_proof["self_mount_namespace"],
        "namespace_boundary_ref": runner.canonical_ref(
            "foundation-preflight-namespace",
            namespace_proof,
        ),
        "namespace_proof": namespace_proof,
        "network_namespace": namespace_proof["self_network_namespace"],
        "python": runner.EXPECTED_PYTHON_VERSION,
        "python_executable": str(runner.ANGLER_PYTHON),
        "source_inventory_ref": source_inventory_ref,
        "unassigned_compute_processes": unassigned_compute_processes,
        "unassigned_gpu_name": "NVIDIA GeForce RTX 5070",
        "unassigned_gpu_uuid": runner.UNASSIGNED_GPU_UUID,
    }
    return {
        "boundary": "FOUNDATION_MIRROR_PREFLIGHT_CHILD",
        **payload,
        "host_boundary_ref": runner.canonical_ref(
            "foundation-preflight-host",
            payload,
        ),
    }


def _synthetic_preflight_host_closure(
    source_inventory_ref: str,
) -> dict[str, object]:
    payload = {
        "child_exit_terminal": True,
        "host_live_paths_absent": True,
        "host_private_path_absent": True,
        "host_scratch_device": 7,
        "host_scratch_gid": 1000,
        "host_scratch_identity_unchanged": True,
        "host_scratch_inode": 41,
        "host_scratch_mode": "0700",
        "host_scratch_path": "/opt/angler/scratch",
        "host_scratch_uid": 1000,
        "source_inventory_ref": source_inventory_ref,
        "stdout_single_canonical_receipt": True,
    }
    return {
        **payload,
        "host_closure_ref": runner.canonical_ref(
            "foundation-preflight-host-closure",
            payload,
        ),
    }


def _synthetic_preflight_receipt(
    source_inventory: object,
    *,
    durations_ns: tuple[int, int, int] = (11, 17, 23),
) -> dict[str, object]:
    sources = [dict(row) for row in source_inventory]
    source_inventory_ref = runner.canonical_ref("source-inventory", sources)
    boundary = _synthetic_preflight_boundary(source_inventory_ref)
    host_closure = _synthetic_preflight_host_closure(source_inventory_ref)
    baseline = runner._PinnedAllocatorSnapshot(3, 17, 1, 5)
    allocated = runner._PinnedAllocatorSnapshot(
        5,
        17 + runner.FOUNDATION_MIRROR_PINNED_BYTES,
        1 + runner.FOUNDATION_MIRROR_PINNED_SLOT_COUNT,
        5 + runner.FOUNDATION_MIRROR_PINNED_BYTES,
    )
    return runner.FoundationMirrorPreflightReceipt(
        source_inventory_ref=source_inventory_ref,
        source_count=len(sources),
        environment_sha256=boundary["environment_sha256"],
        hostname=boundary["hostname"],
        python=boundary["python"],
        python_executable=boundary["python_executable"],
        mount_namespace=boundary["mount_namespace"],
        network_namespace=boundary["network_namespace"],
        namespace_boundary_ref=boundary["namespace_boundary_ref"],
        _namespace_proof_json=runner.canonical_json_bytes(
            boundary["namespace_proof"]
        ),
        assigned_gpu_uuid=boundary["assigned_gpu_uuid"],
        assigned_gpu_name=boundary["assigned_gpu_name"],
        unassigned_gpu_uuid=boundary["unassigned_gpu_uuid"],
        unassigned_gpu_name=boundary["unassigned_gpu_name"],
        unassigned_compute_processes=boundary["unassigned_compute_processes"],
        logical_device_count=boundary["logical_device_count"],
        logical_device_index=boundary["logical_device_index"],
        host_boundary_ref=boundary["host_boundary_ref"],
        host_closure_ref=host_closure["host_closure_ref"],
        _host_closure_json=runner.canonical_json_bytes(host_closure),
        verification_durations_ns=durations_ns,
        maximum_duration_ns=max(durations_ns),
        vm_locked_bytes_before=0,
        memlock_soft_limit_bytes=runner.resource.RLIM_INFINITY,
        rss_bytes_before=32,
        rss_reservation_bytes=(
            32
            + runner.QUALIFIED_MODEL_PARAMETER_BYTES
            + runner.FOUNDATION_MIRROR_PINNED_BYTES
            + runner.FOUNDATION_MIRROR_TRANSIENT_BYTES
        ),
        allocator_baseline=baseline,
        allocator_post_allocation=allocated,
        allocator_cleanup=baseline,
    ).to_canonical()


_STATE_REF = _ref("v2-integration-genesis-state")
_MODEL_REF = _ref("v2-integration-frozen-model")
_ENCODER_REF = _ref("v2-integration-frozen-encoder")
_SNAPSHOT = b"exact-v2-integration-genesis-snapshot"
_EVALUATION_SEEDS = (
    bytes(range(0, 32)),
    bytes(range(32, 64)),
    bytes(range(64, 96)),
)
_EVALUATION_COMMITMENTS = tuple(
    evaluator.commit_replicate_seed(seed) for seed in _EVALUATION_SEEDS
)
_REPLICATE_REF = _EVALUATION_COMMITMENTS[0]
_ADMISSION_REF = _ref("v2-integration-final-admission")


def _synthetic_evaluation_admission() -> tuple[dict[str, object], str]:
    boundary = runner._evaluation_admission_boundary(
        manifest_ref=_ref("synthetic-admission-manifest"),
        manifest_sha256=_raw("synthetic-admission-manifest-bytes"),
        qualification_result_ref=_ref("synthetic-admission-qualification"),
        qualification_result_sha256=_raw(
            "synthetic-admission-qualification-bytes"
        ),
        qualification_release_ref=_ref("synthetic-admission-release"),
        source_seal_ref=_ref("synthetic-admission-source"),
    )
    suite = evaluator.make_evaluation_evaluator(
        _EVALUATION_SEEDS,
        _EVALUATION_COMMITMENTS,
        _ADMISSION_REF,
    )
    commitments = suite.commitments.to_canonical()
    receipt_payload = {
        **boundary,
        "artifact_bytes": 512,
        "artifact_device": 81,
        "artifact_inode": 91,
        "artifact_mode": "0600",
        "artifact_path": str(runner.SEALED_SEED_PATH),
        "artifact_sha256": _raw("synthetic-sealed-seed-artifact"),
        "commitments": list(_EVALUATION_COMMITMENTS),
        "evaluator_commitments_ref": runner.canonical_ref(
            "evaluator-commitments", commitments
        ),
        "schema": runner.SEED_SEAL_RECEIPT_SCHEMA,
    }
    receipt = {
        **receipt_payload,
        "seed_seal_ref": runner.canonical_ref(
            "evaluation-seed-seal", receipt_payload
        ),
    }
    admission = runner.build_evaluation_admission_claim(
        boundary=boundary,
        seed_receipt=receipt,
        evaluator_commitments=commitments,
    )
    return admission, hashlib.sha256(
        runner.canonical_json_bytes(admission)
    ).hexdigest()


def _bootstrap_episode() -> CognitiveEpisode:
    proposals = (
        "Retain the bounded synthetic bootstrap state.",
        "Stop without changing the bounded synthetic bootstrap state.",
    )
    return CognitiveEpisode(
        task_id="V2_NEUTRAL_BOOTSTRAP",
        request="Create the qualified local synthetic bootstrap episode.",
        recalled_refs=(),
        proposals=proposals,
        selected_index=0,
        commitment=ProspectiveCommitment(
            parent_event_ref=None,
            task_id="V2_NEUTRAL_BOOTSTRAP",
            candidate_index=0,
            candidate_trace=proposals[0],
            predicted_score=0.0,
            uncertainty=1.0,
            horizon=1,
            competence_state_digest=_STATE_REF,
        ),
        response="Qualified local synthetic bootstrap complete.",
        observations=("The local synthetic bootstrap remained bounded.",),
        outcome="success",
        feedback_text="Objective local bootstrap check passed.",
        feedback_source_ref=_ref("v2-neutral-bootstrap-feedback"),
        parent_state_digest=_STATE_REF,
        child_state_digest=_STATE_REF,
        model_ref=_MODEL_REF,
        encoder_ref=_ENCODER_REF,
        supporting_evidence_refs=(),
    )


def _neutral_episodes() -> tuple[CognitiveEpisode, ...]:
    bootstrap = _bootstrap_episode()
    values = runner.build_neutral_fixture_chain(
        bootstrap_episode_ref=bootstrap.episode_ref,
        competence_state_digest=_STATE_REF,
        model_ref=_MODEL_REF,
        encoder_ref=_ENCODER_REF,
    )
    return tuple(values)  # type: ignore[return-value]


@dataclass(frozen=True)
class _OwnerPaths:
    root: Path
    store: Path
    journal: Path
    learner: Path


@dataclass(frozen=True)
class _OwnerManifest:
    model_ref: str = _MODEL_REF
    encoder_ref: str = _ENCODER_REF


class _OwnerLearner:
    pending_decision = None
    pending_prospective_material = None

    def __init__(self) -> None:
        self._snapshot = b"v2-owner-exact-qualified-genesis"

    def capture_state(self) -> bytes:
        return self._snapshot

    def restore_state(self, state: bytes) -> None:
        self._snapshot = state

    def state_digest(self) -> str:
        return runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST


class _OwnerJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.write_bytes(b"v2-owner-empty-journal-v1")
        path.chmod(0o600)

    def audit_integrity(self) -> None:
        if not self.path.is_file() or not self.path.read_bytes():
            raise RuntimeError("fake owner journal is not intact")

    def get(self, _key: str) -> None:
        return None


class _OwnerAcquisitionBinding:
    def __init__(
        self,
        scope_spec: runner.V2AcquisitionScopeSpec,
        events: list[str],
        dataset_id: str,
    ) -> None:
        self.dataset_name = scope_spec.dataset_name
        self.tenant_name = scope_spec.tenant_name
        self.node_set_name = scope_spec.node_set_name
        self.dataset_id = dataset_id
        self._arm = scope_spec.arm
        self._events = events

    async def close(self) -> None:
        self._events.append(f"close:{self._arm}")


class _OwnerAcquisitionBackend:
    def __init__(
        self,
        scope_spec: runner.V2AcquisitionScopeSpec,
        events: list[str],
    ) -> None:
        self.scope_spec = scope_spec
        self._events = events

    def project(self, *_args: object, **_kwargs: object) -> None:
        self._events.append(f"project:{self.scope_spec.arm}")

    def search(self, *_args: object, **_kwargs: object) -> tuple[()]:
        self._events.append(f"search:{self.scope_spec.arm}")
        return ()

    async def forget_namespace(self) -> None:
        self._events.append(f"forget:{self.scope_spec.arm}")


class _OwnerAcquisitionOpener:
    def __init__(self, events: list[str], *, incarnation: str = "genesis") -> None:
        self.events = events
        self.incarnation = incarnation

    async def __call__(
        self,
        scope_spec: runner.V2AcquisitionScopeSpec,
        *,
        expected_dataset_id: str | None = None,
    ) -> tuple[_OwnerAcquisitionBinding, _OwnerAcquisitionBackend]:
        dataset_id = runner.canonical_ref(
            "fake-cognee-incarnation",
            {
                "incarnation": self.incarnation,
                "scope_ref": scope_spec.scope_ref,
            },
        )
        if (
            expected_dataset_id is not None
            and expected_dataset_id != dataset_id
        ):
            raise AssertionError("owner scope requested a different dataset id")
        root = Path(scope_spec.state_root)
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        state = root / "fake-cognee-state"
        state.write_bytes(b"bounded fake state")
        state.chmod(0o600)
        self.events.append(f"open:{scope_spec.arm}")
        return (
            _OwnerAcquisitionBinding(scope_spec, self.events, dataset_id),
            _OwnerAcquisitionBackend(scope_spec, self.events),
        )


class _OwnerMemory:
    def __init__(
        self,
        *,
        source: CognitiveTransactionStore,
        backend: _OwnerAcquisitionBackend,
        fail_rebuild: bool,
    ) -> None:
        self.source = source
        self.backend = backend
        self.fail_rebuild = fail_rebuild
        inventory = source.acquisition_items(limit=64)
        if len(inventory) != 13 or source.acquisition_head().next_ordinal != 13:
            raise AssertionError("memory was constructed before the full genesis existed")
        backend._events.append(f"memory-nonempty:{backend.scope_spec.arm}:13")

    async def rebuild(self, *, limit: int) -> None:
        if limit != runner.MAXIMUM_PROJECTION_RETRY:
            raise AssertionError("lineage rebuild bound drifted")
        arm = self.backend.scope_spec.arm
        self.backend._events.append(f"rebuild:{arm}")
        if self.fail_rebuild:
            raise RuntimeError("injected rebuild failure")
        pending = self.source.pending_acquisition_projections(limit=limit)
        if len(pending) != 13:
            raise AssertionError("rebuild did not receive the exact full genesis")
        for item in pending:
            self.backend.project(item)
            self.source.ack_acquisition_projection(item.projection_ref)

    async def retry_pending_projections(self, *, limit: int) -> None:
        if limit != runner.MAXIMUM_PROJECTION_RETRY:
            raise AssertionError("lineage retry bound drifted")
        arm = self.backend.scope_spec.arm
        self.backend._events.append(f"retry:{arm}")
        if self.source.pending_acquisition_projections(limit=1):
            raise AssertionError("rebuild left a pending acquisition projection")

    def assert_synchronized(self, head: object) -> None:
        arm = self.backend.scope_spec.arm
        self.backend._events.append(f"synchronized:{arm}")
        if head != self.source.acquisition_head():
            raise AssertionError("memory synchronized against a different head")


def _owner_bootstrap_episode(state_digest: str, manifest: object) -> CognitiveEpisode:
    base = _bootstrap_episode()
    return replace(
        base,
        commitment=replace(
            base.commitment,
            competence_state_digest=state_digest,
        ),
        parent_state_digest=state_digest,
        child_state_digest=state_digest,
        model_ref=getattr(manifest, "model_ref"),
        encoder_ref=getattr(manifest, "encoder_ref"),
    )


def _owner_mechanics(
    events: list[str],
    *,
    ledger: runner.EvidenceLedger,
    fail_rebuild_arms: frozenset[str] = frozenset(),
) -> runner.V2LineageMechanics:
    def clone_paths_factory(**paths: Path) -> _OwnerPaths:
        durable_allocations = ledger._connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind=?",
            ("persistent-lineage-allocation",),
        ).fetchone()[0]
        if durable_allocations < 1:
            raise AssertionError("creating constructor ran before durable allocation")
        events.append(f"paths-after-allocation:{durable_allocations}")
        return _OwnerPaths(**paths)

    def restore_genesis(_genesis: object, *, prospective_lesion: bool) -> _OwnerLearner:
        if prospective_lesion is not False:
            raise AssertionError("V2 lineage changed the qualified learner")
        events.append("restore-genesis")
        return _OwnerLearner()

    def transaction_store_factory(path: Path) -> CognitiveTransactionStore:
        events.append("create-store")
        return CognitiveTransactionStore(path)

    def journal_factory(path: Path) -> _OwnerJournal:
        events.append("create-journal")
        return _OwnerJournal(path)

    def memory_factory(
        *,
        source: CognitiveTransactionStore,
        backend: _OwnerAcquisitionBackend,
    ) -> _OwnerMemory:
        return _OwnerMemory(
            source=source,
            backend=backend,
            fail_rebuild=backend.scope_spec.arm in fail_rebuild_arms,
        )

    def closed_hashes(
        paths: _OwnerPaths,
        *,
        store: CognitiveTransactionStore,
        journal: _OwnerJournal,
    ) -> dict[str, str]:
        journal.audit_integrity()
        acquisitions = store.acquisition_items(limit=64)
        episodes = store.episode_items(limit=64)
        acquisition_bytes = json.dumps(
            [item.acquisition_ref for item in acquisitions],
            separators=(",", ":"),
        ).encode("ascii")
        store_bytes = json.dumps(
            {
                "episodes": [
                    [item.sequence, item.episode_ref] for item in episodes
                ],
                "head_state": store.audit_integrity().state_digest,
                "snapshot": store.load_head_state().hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return {
            "acquisition_sequence": hashlib.sha256(acquisition_bytes).hexdigest(),
            "journal": hashlib.sha256(paths.journal.read_bytes()).hexdigest(),
            "learner": hashlib.sha256(paths.learner.read_bytes()).hexdigest(),
            "store": hashlib.sha256(store_bytes).hexdigest(),
        }

    def assert_quiescent(
        *,
        learner: _OwnerLearner,
        store: CognitiveTransactionStore,
        journal: _OwnerJournal,
        cycle: object | None = None,
    ) -> None:
        if cycle is not None and any(
            (
                getattr(cycle, "pending_turn", None) is not None,
                getattr(cycle, "executed_turn", None) is not None,
                getattr(cycle, "_beginning", None) is not False,
                getattr(cycle, "_executing", None) is not False,
            )
        ):
            raise AssertionError("fake owner received a non-quiescent cycle")
        if (
            learner.pending_decision is not None
            or learner.pending_prospective_material is not None
            or store.pending_acquisition_projections(limit=1)
            or store.audit_integrity().state_digest != learner.state_digest()
            or store.load_head_state() != learner.capture_state()
        ):
            raise AssertionError("fake lineage is not quiescent")
        journal.audit_integrity()
        events.append("quiescent")

    return runner.V2LineageMechanics(
        restore_genesis=restore_genesis,
        clone_paths_factory=clone_paths_factory,
        transaction_store_factory=transaction_store_factory,
        journal_factory=journal_factory,
        bootstrap_episode_factory=_owner_bootstrap_episode,
        memory_factory=memory_factory,
        closed_hashes=closed_hashes,
        assert_quiescent=assert_quiescent,
    )


def _memory_record(
    *,
    ordinal: int,
    source_ref: str,
    content: str,
    confidence_ppm: int | None = None,
) -> CognitiveMemoryRecord:
    return CognitiveMemoryRecord(
        kind=CognitiveMemoryKind.EPISODIC,
        epistemic_status=EpistemicStatus.OBSERVED,
        content=content,
        provenance_refs=(source_ref,),
        visibility="LEARNER_VISIBLE",
        producer_id="ANG-TEST-V2-HISTORY-INTEGRATION",
        producer_checkpoint_ref=_ref("v2-donor-producer"),
        competence_ref=_ref("v2-donor-competence"),
        acquired_ordinal=ordinal,
        confidence_ppm=confidence_ppm,
    )


def _proposed_batch_record(
    *,
    ordinal: int,
    source_ref: str,
    content: str,
) -> CognitiveMemoryRecord:
    return CognitiveMemoryRecord(
        kind=CognitiveMemoryKind.COUNTERFACTUAL,
        epistemic_status=EpistemicStatus.PROPOSED,
        content=content,
        provenance_refs=(source_ref,),
        visibility="LEARNER_VISIBLE",
        producer_id="ANG-TEST-V2-HISTORY-INTEGRATION",
        producer_checkpoint_ref=_ref("v2-donor-producer"),
        competence_ref=_ref("v2-donor-competence"),
        acquired_ordinal=ordinal,
    )


def _catalog_bundle(
    *,
    replicate: str = "replicate-01",
    replicate_commitment: str = _REPLICATE_REF,
    purpose: runner.Purpose = "evaluation",
    task_bindings: dict[
        tuple[runner.Family, int], evaluator.PublicEvaluationTask
    ]
    | None = None,
) -> tuple[
    runner.DonorCatalog,
    tuple[runner.AttemptFinal, ...],
    tuple[CognitiveAcquisitionItem, ...],
]:
    attempts: list[runner.AttemptFinal] = []
    items: list[CognitiveAcquisitionItem] = []
    predecessor: str | None = None
    ordinal = 0
    for family in runner.FAMILIES:
        for family_ordinal in range(12):
            task = (
                None
                if task_bindings is None
                else task_bindings[(family, family_ordinal)]
            )
            source_ref = _ref(f"v2-resolution-{family}-{family_ordinal}")
            attempt = _attempt_final(
                task_id=(
                    _ref(f"v2-task-{family}-{family_ordinal}")
                    if task is None
                    else task.task_id
                ),
                public_task_ref=(
                    _ref(f"v2-public-{family}-{family_ordinal}")
                    if task is None
                    else task.public_task_ref
                ),
                replicate=replicate if task is None else task.replicate,
                replicate_commitment=(
                    replicate_commitment
                    if task is None
                    else task.replicate_commitment
                ),
                family=family,
                arm="IO_FULL_ORDINARY_12",
                phase="adaptation",
                purpose=purpose,
                task_ordinal=family_ordinal,
                block_ordinal=ordinal,
                success=True,
                resolution_ref=source_ref,
            )
            record = _memory_record(
                ordinal=ordinal,
                source_ref=source_ref,
                content=f"Public observed donor {family} {family_ordinal:02d}.",
            )
            acquisition = CognitiveAcquisition(
                ordinal=ordinal,
                predecessor_acquisition_ref=predecessor,
                source_contract="ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0",
                source_ref=source_ref,
                record=record,
            )
            items.append(
                CognitiveAcquisitionItem(
                    ordinal,
                    acquisition.acquisition_ref,
                    record.record_ref,
                    acquisition,
                )
            )
            attempts.append(attempt)
            predecessor = acquisition.acquisition_ref
            ordinal += 1

    # A canonical legacy acquisition proves that the complete store head is
    # accounted for while only admitted observed resolutions enter the pool.
    legacy_source = _ref("v2-legacy-source-excluded")
    legacy_record = _memory_record(
        ordinal=ordinal,
        source_ref=legacy_source,
        content="Canonical legacy source that is ineligible as a V2 donor.",
    )
    legacy_acquisition = CognitiveAcquisition(
        ordinal=ordinal,
        predecessor_acquisition_ref=predecessor,
        source_contract="ANG-CTR-COGNITIVE-EPISODE-001@0.1.0",
        source_ref=legacy_source,
        record=legacy_record,
    )
    items.append(
        CognitiveAcquisitionItem(
            ordinal,
            legacy_acquisition.acquisition_ref,
            legacy_record.record_ref,
            legacy_acquisition,
        )
    )

    catalog = runner.build_donor_catalog(
        items,
        attempts,
        replicate_commitment=replicate_commitment,
        expected_next_ordinal=len(items),
    )
    return catalog, tuple(attempts), tuple(items)


def _ranked_controls(
    catalog: runner.DonorCatalog,
) -> tuple[
    runner.RankedSelectionPlan,
    runner.RankedSelectionPlan,
    runner.RankedSelectionPlan,
]:
    embeddings = {
        entry.record_ref: (
            1.0,
            float((entry.acquired_ordinal % 6) + 1),
        )
        for entry in catalog.entries
    }

    task = _selection_task()
    low = runner.build_ranked_selection_plan(
        task=task,
        catalog=catalog,
        mode="same-low",
        task_embedding=(1.0, 0.0),
        record_embeddings=embeddings,
    )
    high = runner.build_ranked_selection_plan(
        task=task,
        catalog=catalog,
        mode="same-high",
        task_embedding=(1.0, 0.0),
        record_embeddings=embeddings,
    )
    cross = runner.build_ranked_selection_plan(
        task=task,
        catalog=catalog,
        mode="cross-low",
        task_embedding=(1.0, 0.0),
        record_embeddings=embeddings,
    )
    return low, high, cross


def _selection_task() -> evaluator.PublicEvaluationTask:
    suite = evaluator.make_evaluation_evaluator(
        _EVALUATION_SEEDS,
        _EVALUATION_COMMITMENTS,
        _ADMISSION_REF,
    )
    return next(
        item
        for item in suite.release_phase("adaptation")
        if item.replicate == "replicate-01"
        and item.family == "symbolic-demonstration-transfer"
        and item.ordinal == 0
    )


def _frozen_recall_capture(
    task: evaluator.PublicEvaluationTask,
    batch: object,
) -> runner.FrozenRecallCapture:
    source_hashes = {
        name: _raw(f"v2-recall-source-{task.task_id}-{name}")
        for name in runner.BASELINE_HASH_NAMES
    }
    return runner.freeze_recall_capture(
        task=task,
        kind="ordinary",
        recall_batch=batch,
        query_ref=_ref(f"v2-recall-query-{task.task_id}"),
        backend_binding_ref=_ref("v2-recall-backend-binding"),
        source_baseline_seal_ref=_ref("v2-recall-source-baseline"),
        capture_clone_ref=_ref(f"v2-recall-capture-clone-{task.task_id}"),
        capture_quiescent=True,
        capture_disposed=True,
        cleanup_absence_verified=True,
        source_root=str((ROOT / "tests").resolve()),
        source_hashes=source_hashes,
    )


@dataclass(frozen=True, slots=True)
class _RecallItem:
    record: CognitiveMemoryRecord
    record_ref: str
    acquisition_ref: str
    source_ref: str
    text: str
    ordinal: int
    source_contract: str
    temporal_marker: str


@dataclass(frozen=True, slots=True)
class _RecallBatch:
    items: tuple[_RecallItem, ...]
    rejected: tuple[str, ...] = ()


class _ReferenceBackend:
    def search(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        return ()


class _PresentationMemory:
    def __init__(
        self,
        batch: _RecallBatch,
        *,
        source: object | None = None,
    ) -> None:
        self.batch = batch
        self.source = source
        self.backend = _ReferenceBackend()
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def recall(self, *args: object, **kwargs: object) -> _RecallBatch:
        self.calls.append(("recall", args, kwargs))
        return self.batch

    def recall_from_hits(self, *args: object, **kwargs: object) -> _RecallBatch:
        self.calls.append(("recall_from_hits", args, kwargs))
        return self.batch

    def retry_pending_projections(self, limit: int = 64) -> int:
        self.calls.append(("retry_pending_projections", (limit,), {}))
        return 0

    def assert_synchronized(self) -> None:
        self.calls.append(("assert_synchronized", (), {}))

    def delegated_marker(self) -> str:
        return "delegated-canonical-memory"


class _BoundedOrdinaryRecallSource:
    def __init__(self, items: tuple[CognitiveAcquisitionItem, ...]) -> None:
        self.items = items
        self.by_record = {item.record_ref: item for item in items}

    def acquisition_head(self) -> AcquisitionHead:
        final = self.items[-1]
        return AcquisitionHead(
            len(self.items), final.acquisition_ref, final.record_ref
        )

    def get_acquisition_item(self, record_ref: str) -> CognitiveAcquisitionItem:
        try:
            return self.by_record[record_ref]
        except KeyError:
            raise KeyError(record_ref) from None

    def acquisition_items(
        self, after_ordinal: int = -1, limit: int = 64
    ) -> tuple[CognitiveAcquisitionItem, ...]:
        return tuple(
            item for item in self.items if item.ordinal > after_ordinal
        )[:limit]

    def pending_acquisition_projections(
        self, limit: int = 64
    ) -> tuple[object, ...]:
        del limit
        return ()

    def ack_acquisition_projection(self, projection_ref: str) -> None:
        del projection_ref


class _BoundedOrdinaryRecallBackend:
    def __init__(self) -> None:
        self.hits: tuple[AcquisitionReferenceHit, ...] = ()
        self.search_limits: list[int] = []

    async def project(self, projection: object) -> str:
        return f"test-projection:{getattr(projection, 'projection_ref')}"

    async def search(
        self, query: str, *, limit: int
    ) -> tuple[AcquisitionReferenceHit, ...]:
        if query != "bounded ordinary recall query":
            raise AssertionError("ordinary recall test query differs")
        self.search_limits.append(limit)
        return self.hits

    async def forget_namespace(self) -> None:
        return None


def _bounded_ordinary_recall_fixture(
    *,
    memory_type: type[AcquisitionSituatedMemory] = AcquisitionSituatedMemory,
    ordinary_record_count: int = 12,
) -> tuple[
    _BoundedOrdinaryRecallSource,
    _BoundedOrdinaryRecallBackend,
    AcquisitionSituatedMemory,
    tuple[AcquisitionReferenceHit, ...],
    str,
    str,
]:
    rows: list[CognitiveAcquisitionItem] = []
    predecessor: str | None = None

    def append_record(
        *,
        source_contract: str,
        source_ref: str,
        kind: CognitiveMemoryKind,
        epistemic_status: EpistemicStatus,
        content: str,
    ) -> str:
        nonlocal predecessor
        ordinal = len(rows)
        record = CognitiveMemoryRecord(
            kind=kind,
            epistemic_status=epistemic_status,
            content=content,
            provenance_refs=(source_ref,),
            visibility="LEARNER_VISIBLE",
            producer_id="ANG-TEST-V2-BOUNDED-ORDINARY-RECALL",
            producer_checkpoint_ref=_ref("bounded-ordinary-producer"),
            competence_ref=_ref("bounded-ordinary-competence"),
            acquired_ordinal=ordinal,
        )
        acquisition = CognitiveAcquisition(
            ordinal=ordinal,
            predecessor_acquisition_ref=predecessor,
            source_contract=source_contract,
            source_ref=source_ref,
            record=record,
        )
        rows.append(
            CognitiveAcquisitionItem(
                ordinal,
                acquisition.acquisition_ref,
                record.record_ref,
                acquisition,
            )
        )
        predecessor = acquisition.acquisition_ref
        return record.record_ref

    if type(ordinary_record_count) is not int or ordinary_record_count < 1:
        raise ValueError("ordinary record count differs")
    legacy_refs = tuple(
        append_record(
            source_contract=LEGACY_COGNITIVE_EPISODE_CONTRACT,
            source_ref=_ref(f"bounded-ordinary-legacy-{index}"),
            kind=CognitiveMemoryKind.EPISODIC,
            epistemic_status=EpistemicStatus.OBSERVED,
            content=f"Observed ordinary memory {index:02d}.",
        )
        for index in range(ordinary_record_count)
    )
    resolution_ref = append_record(
        source_contract=PROSPECTIVE_RESOLUTION_CONTRACT,
        source_ref=_ref("bounded-ordinary-prospective-resolution"),
        kind=CognitiveMemoryKind.EPISODIC,
        epistemic_status=EpistemicStatus.OBSERVED,
        content="Observed prospective lifecycle resolution.",
    )
    proposed_ref = append_record(
        source_contract=PROSPECTIVE_DYNAMICS_BATCH_CONTRACT,
        source_ref=_ref("bounded-ordinary-prospective-batch"),
        kind=CognitiveMemoryKind.COUNTERFACTUAL,
        epistemic_status=EpistemicStatus.PROPOSED,
        content="Proposed prospective batch; not an observed outcome.",
    )
    source = _BoundedOrdinaryRecallSource(tuple(rows))
    backend = _BoundedOrdinaryRecallBackend()
    memory = memory_type(source=source, backend=backend)
    asyncio.run(memory.rebuild(limit=64))
    ranking = (proposed_ref, resolution_ref, *legacy_refs)
    hits = tuple(
        AcquisitionReferenceHit(
            record_ref=record_ref,
            score=float(index),
            backend_ref="bounded-ordinary-recall-test",
        )
        for index, record_ref in enumerate(ranking)
    )
    backend.hits = hits
    return source, backend, memory, hits, proposed_ref, resolution_ref


class _TinyTensorRows:
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[int]]:
        return self._rows


class _TinyTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        self.template_calls.append({"messages": messages, **kwargs})
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, rendered: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        values = list(range(1, len(rendered.split()) + 1))
        return {
            "input_ids": _TinyTensorRows([values]),
            "attention_mask": _TinyTensorRows([[1] * len(values)]),
        }


class _TextAdapter:
    def propose_procedure_traces(
        self,
        task: str,
        recalled_evidence: tuple[str, ...],
        *,
        count: int = 4,
    ) -> tuple[str, ...]:
        del task, recalled_evidence
        return tuple(f"candidate-{index}" for index in range(count))

    def execute_with_procedure(
        self,
        task: str,
        selected_trace: str,
        observations: tuple[str, ...],
    ) -> str:
        del task, selected_trace, observations
        return "unused"


class _RelationAdapter:
    def __init__(self, encoder_ref: str) -> None:
        self.encoder_ref = encoder_ref

    def encode_candidates(
        self,
        task: str,
        proposals: tuple[str, ...],
        recall: object,
    ) -> tuple[object, ...]:
        del task, proposals, recall
        return ()


class _Learner:
    prospective_lesion = False
    pending_decision = None
    pending_prospective_material = None

    def select(self, **kwargs: object) -> object:
        del kwargs
        return object()

    def apply_outcome(self, outcome: object) -> None:
        del outcome

    def capture_state(self) -> bytes:
        return _SNAPSHOT

    def restore_state(self, state: bytes) -> None:
        del state

    def state_digest(self) -> str:
        return _STATE_REF

    def capture_pending_state(self) -> bytes:
        return b"pending"

    def restore_pending_state(self, state: bytes) -> object:
        del state
        return object()

    def select_prospective(self, **kwargs: object) -> object:
        del kwargs
        return object()

    def bind_pending_prospective_batch(self, batch_ref: str) -> object:
        del batch_ref
        return object()

    def component_state_integrity(self) -> object:
        return ProspectiveComponentStateIntegrity(
            step=0,
            checkpoint_ref=_ref("v2-cycle-checkpoint"),
            config_ref=_ref("v2-cycle-config"),
            world_state_digest=_ref("v2-cycle-world-state"),
            self_state_digest=_ref("v2-cycle-self-state"),
            focus_state_digest=_ref("v2-cycle-focus-state"),
            outcome_state_digest=_ref("v2-cycle-outcome-state"),
        )


class _Executor:
    def execute(self, request: object) -> object:
        del request
        return object()

    def recover(self, request: object) -> object:
        del request
        return object()


def _finalize_evaluator_phase(
    suite: evaluator.HighLevelMultidomainEvaluator,
    tasks: tuple[evaluator.PublicEvaluationTask, ...],
) -> None:
    for task in tasks:
        for arm in suite.expected_arms_for(task.task_id):
            suite.record_attempt(
                task.task_id,
                arm,
                _ref(f"schedule-{task.task_id}-{arm}"),
                "",
                proposal_admitted=False,
            )
    suite.complete_phase(tasks[0].phase)


def _attempt_final(
    *,
    task_id: str,
    public_task_ref: str,
    replicate: str,
    replicate_commitment: str,
    family: runner.Family,
    arm: runner.Arm,
    block_ordinal: int,
    success: bool,
    response_conforms: bool = True,
    purpose: runner.Purpose = "evaluation",
    phase: runner.Phase = "final",
    task_ordinal: int | None = None,
    resolution_ref: str | None = None,
    clone_parent: Path | None = None,
) -> runner.AttemptFinal:
    arm_position = runner.ARMS.index(arm)
    proposal_prompt_ref = _ref(f"finite-prompt-{task_id}-{arm}")
    stage = runner.AttemptStage(
        purpose=purpose,
        phase=phase,
        task_id=task_id,
        public_task_ref=public_task_ref,
        replicate=replicate,
        replicate_commitment=replicate_commitment,
        family=family,
        task_ordinal=block_ordinal if task_ordinal is None else task_ordinal,
        arm=arm,
        block_ordinal=block_ordinal,
        arm_position=arm_position,
        history_plan_ref=(None if arm == "N0_NATIVE_NO_HISTORY" else _ref(
            f"finite-history-{task_id}-{arm}"
        )),
        prompt_plan_ref=_ref(f"finite-prompt-plan-{task_id}-{arm}"),
        proposal_prompt_ref=proposal_prompt_ref,
        proposal_prompt_tokens=64,
    )
    receipt_ref = stage.attempt_receipt_ref
    proposals = (
        "Use the first bounded public procedure.",
        "Use the second bounded public procedure.",
    )
    raw_response = "SYNTHETIC_CONFORMING_RESPONSE"
    execution_request_ref = _ref(f"finite-request-{task_id}-{arm}")
    execution_receipt_ref = _ref(f"finite-receipt-{task_id}-{arm}")
    response_payload = {
        "arm": arm,
        "attempt_receipt_ref": receipt_ref,
        "protocol_identity": runner.PROTOCOL_IDENTITY,
        "raw_response": raw_response,
        "record_kind": "response",
        "task_id": task_id,
    }
    response_ref = runner.domain_ref(
        runner.EVALUATOR_RECORD_DOMAIN,
        response_payload,
    )
    stateful = arm in runner.STATEFUL_ARMS
    terminal_disposition_ref = _ref(f"finite-disposition-{task_id}-{arm}")
    clone_lifecycle: runner.CloneLifecycleEvidence | None = None
    if stateful and phase in ("development", "final"):
        if clone_parent is None:
            raise AssertionError("stateful post-adaptation fixture needs a clone root")
        source_root = clone_parent / f"source-{replicate}-{arm}"
        clone_root = clone_parent / f"clone-{task_id[-12:]}-{arm}"
        source_root.mkdir(exist_ok=True)
        clone_root.mkdir()
        hashes = {
            "acquisition_sequence": _raw(f"finite-acquisitions-{replicate}-{arm}"),
            "journal": _raw(f"finite-journal-{replicate}-{arm}"),
            "learner": _raw(f"finite-learner-{replicate}-{arm}"),
            "store": _raw(f"finite-store-{replicate}-{arm}"),
        }
        terminal_hashes = {
            **hashes,
            "journal": _raw(f"finite-terminal-{task_id}-{arm}"),
        }
        terminal = runner.CloneTerminalEvidence(
            task_id=task_id,
            arm=arm,
            source_root=str(source_root),
            clone_root=str(clone_root),
            source_before=tuple(sorted(hashes.items())),
            source_after=tuple(sorted(hashes.items())),
            clone_before=tuple(sorted(hashes.items())),
            clone_terminal=tuple(sorted(terminal_hashes.items())),
            quiescent=True,
        )
        clone_root.rmdir()
        clone_lifecycle = runner.CloneLifecycleEvidence(
            terminal=terminal,
            terminal_disposition_ref=terminal_disposition_ref,
            disposed=True,
            cleanup_absence_verified=True,
        )
    observation = runner.AttemptObservation(
        task_id=task_id,
        arm=arm,
        proposal_admitted=True,
        proposals=proposals,
        selected_index=0,
        selected_trace=proposals[0],
        raw_response=raw_response,
        proposal_generation_ref=_ref(f"finite-generation-{task_id}-{arm}"),
        proposal_prompt_ref=proposal_prompt_ref,
        proposal_prompt_tokens=64,
        proposal_output_tokens=8,
        execution_request_ref=execution_request_ref,
        execution_receipt_ref=execution_receipt_ref,
        execution_prompt_ref=_ref(f"finite-execution-prompt-{task_id}-{arm}"),
        execution_prompt_tokens=64,
        execution_output_tokens=8,
        selected_trace_bound=True,
        response_bound=True,
        selected_grammar_class="PUBLIC_SYNTHETIC",
        response_grammar_class="PUBLIC_SYNTHETIC",
        resolution_ref=(
            (resolution_ref or _ref(f"finite-resolution-{task_id}-{arm}"))
            if stateful
            else None
        ),
        prospective_reservation_ref=(
            _ref(f"finite-reservation-{task_id}-{arm}") if stateful else None
        ),
        prospective_batch_ref=(
            _ref(f"finite-batch-{task_id}-{arm}") if stateful else None
        ),
        clone_lifecycle=clone_lifecycle,
    )
    judgment_without_ref = {
        "arm": arm,
        "attempt_receipt_ref": receipt_ref,
        "disposition": "SUCCESS" if success else "UNSUCCESSFUL",
        "protocol_identity": runner.PROTOCOL_IDENTITY,
        "proposal_admitted": True,
        "public_task_ref": public_task_ref,
        "raw_response": raw_response,
        "record_kind": "objective-judgment",
        "response_commitment": response_ref,
        "response_conforms": response_conforms,
        "score": 1.0 if success else 0.0,
        "success": success,
        "task_id": task_id,
    }
    objective_ref = runner.domain_ref(
        runner.EVALUATOR_RECORD_DOMAIN,
        judgment_without_ref,
    )
    judgment = {
        **judgment_without_ref,
        "evaluator_record_ref": objective_ref,
    }
    witness = runner.make_congruence_witness(
        task_id=task_id,
        arm=arm,
        attempt_receipt_ref=receipt_ref,
        selected_trace_ref=runner.content_ref({"selected_trace": proposals[0]}),
        execution_request_ref=execution_request_ref,
        execution_receipt_ref=execution_receipt_ref,
        response_ref=response_ref,
        objective_judgment_ref=objective_ref,
        selected_trace_bound=True,
        response_bound=True,
        response_conforms=response_conforms,
        objective_success=success,
        selected_grammar_class="PUBLIC_SYNTHETIC",
        response_grammar_class="PUBLIC_SYNTHETIC",
    )
    return runner.AttemptFinal(
        stage=stage,
        attempt_receipt_ref=receipt_ref,
        terminal_disposition_ref=terminal_disposition_ref,
        observation=observation,
        judgment=judgment,
        witness=witness,
    )


def _finite_cohort(clone_parent: Path) -> tuple[runner.AttemptFinal, ...]:
    values: list[runner.AttemptFinal] = []
    for replicate_index in range(3):
        replicate = f"replicate-{replicate_index + 1:02d}"
        replicate_ref = _EVALUATION_COMMITMENTS[replicate_index]
        for family in runner.FAMILIES:
            for ordinal in range(6):
                task_id = _ref(f"finite-task-{replicate}-{family}-{ordinal}")
                public_ref = _ref(f"finite-public-{replicate}-{family}-{ordinal}")
                for arm in runner.ARMS:
                    if arm in (
                        "N0_NATIVE_NO_HISTORY",
                        "IN_FULL_NEUTRAL_12",
                    ):
                        success = family == "causal-operator"
                    elif arm in (
                        "F0_NEUTRAL_FORMATTED_12",
                        "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6",
                    ):
                        success = family != "causal-operator"
                    elif arm == "S0_SAME_FAMILY_LOW_6_NEUTRAL_6":
                        success = False
                    elif arm in (
                        "S1_SAME_FAMILY_HIGH_6_NEUTRAL_6",
                        "O0_ORDINARY_FROZEN_12",
                    ):
                        success = True
                    else:
                        success = False
                    values.append(
                        _attempt_final(
                            task_id=task_id,
                            public_task_ref=public_ref,
                            replicate=replicate,
                            replicate_commitment=replicate_ref,
                            family=family,
                            arm=arm,
                            block_ordinal=ordinal,
                            success=success,
                            clone_parent=clone_parent,
                        )
                    )
    return tuple(values)


def _qualification_frozen_state(
    root: Path,
    *,
    donor_attempts: tuple[runner.AttemptFinal, ...] | None = None,
) -> tuple[
    runner.FrozenAdaptationState,
    runner.DonorCatalog,
    tuple[CognitiveAcquisitionItem, ...],
]:
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    commitment = runner.qualification_replicate_commitment()
    if donor_attempts is None:
        qualification = evaluator.make_qualification_evaluator()
        task_bindings = {
            (task.family, task.ordinal): task
            for task in qualification.release_phase("adaptation")
            if task.ordinal < 12
        }
        catalog, _attempts, items = _catalog_bundle(
            replicate="qualification-01",
            replicate_commitment=commitment,
            purpose="qualification",
            task_bindings=task_bindings,
        )
    else:
        io_attempts = tuple(
            item
            for item in donor_attempts
            if item.stage.arm == "IO_FULL_ORDINARY_12"
        )
        if len(io_attempts) != 45:
            raise AssertionError("qualification fixture requires all 45 IO donors")
        acquisition_items: list[CognitiveAcquisitionItem] = []
        predecessor: str | None = None
        for ordinal in range(13):
            source_ref = _ref(f"qualification-bootstrap-neutral-source-{ordinal}")
            record = _memory_record(
                ordinal=ordinal,
                source_ref=source_ref,
                content=f"Qualification bootstrap/neutral record {ordinal:02d}.",
            )
            acquisition = CognitiveAcquisition(
                ordinal=ordinal,
                predecessor_acquisition_ref=predecessor,
                source_contract="ANG-CTR-COGNITIVE-EPISODE-001@0.1.0",
                source_ref=source_ref,
                record=record,
            )
            acquisition_items.append(
                CognitiveAcquisitionItem(
                    ordinal,
                    acquisition.acquisition_ref,
                    record.record_ref,
                    acquisition,
                )
            )
            predecessor = acquisition.acquisition_ref
        next_ordinal = 13
        for attempt in io_attempts:
            batch_source_ref = attempt.observation.prospective_batch_ref
            source_ref = attempt.observation.resolution_ref
            if batch_source_ref is None or source_ref is None:
                raise AssertionError(
                    "admitted IO donor lacks its dynamics batch or resolution"
                )
            batch_record = _proposed_batch_record(
                ordinal=next_ordinal,
                source_ref=batch_source_ref,
                content=(
                    "Public proposed qualification dynamics batch "
                    f"{attempt.stage.family} {attempt.stage.task_ordinal:02d}."
                ),
            )
            batch_acquisition = CognitiveAcquisition(
                ordinal=next_ordinal,
                predecessor_acquisition_ref=predecessor,
                source_contract=PROSPECTIVE_DYNAMICS_BATCH_CONTRACT,
                source_ref=batch_source_ref,
                record=batch_record,
            )
            acquisition_items.append(
                CognitiveAcquisitionItem(
                    next_ordinal,
                    batch_acquisition.acquisition_ref,
                    batch_record.record_ref,
                    batch_acquisition,
                )
            )
            predecessor = batch_acquisition.acquisition_ref
            next_ordinal += 1
            record = _memory_record(
                ordinal=next_ordinal,
                source_ref=source_ref,
                content=(
                    "Public observed qualification donor "
                    f"{attempt.stage.family} {attempt.stage.task_ordinal:02d}."
                ),
            )
            acquisition = CognitiveAcquisition(
                ordinal=next_ordinal,
                predecessor_acquisition_ref=predecessor,
                source_contract="ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0",
                source_ref=source_ref,
                record=record,
            )
            acquisition_items.append(
                CognitiveAcquisitionItem(
                    next_ordinal,
                    acquisition.acquisition_ref,
                    record.record_ref,
                    acquisition,
                )
            )
            predecessor = acquisition.acquisition_ref
            next_ordinal += 1
        items = tuple(acquisition_items)
        catalog = runner.build_donor_catalog(
            items,
            io_attempts,
            replicate_commitment=commitment,
            expected_next_ordinal=len(items),
        )
    baseline_hashes = {
        "acquisition_sequence": _raw("v2-qualification-acquisition-sequence"),
        "journal": _raw("v2-qualification-journal"),
        "learner": _raw("v2-qualification-learner"),
        "store": _raw("v2-qualification-store"),
    }
    bootstrap = _bootstrap_episode()
    neutral = _neutral_episodes()
    neutral_refs = tuple(item.episode_ref for item in neutral)
    chain = runner.validate_neutral_fixture_chain(
        neutral,
        bootstrap_episode_ref=bootstrap.episode_ref,
        competence_state_digest=_STATE_REF,
        model_ref=_MODEL_REF,
        encoder_ref=_ENCODER_REF,
    )
    seals: list[runner.BaselineSeal] = []
    for arm, name in (
        ("IN_FULL_NEUTRAL_12", "qualification-in"),
        ("IO_FULL_ORDINARY_12", "qualification-io"),
    ):
        baseline_root = root / name
        baseline_root.mkdir(mode=0o700)
        baseline_root.chmod(0o700)
        seals.append(
            runner.BaselineSeal(
                purpose="qualification",
                replicate_commitment=commitment,
                arm=arm,
                root=str(baseline_root.resolve()),
                hashes=tuple(sorted(baseline_hashes.items())),
                neutral_episode_refs=neutral_refs,
                neutral_chain_ref=str(chain["chain_ref"]),
                bootstrap_episode_ref=bootstrap.episode_ref,
                genesis_competence_digest=_STATE_REF,
                model_ref=_MODEL_REF,
                encoder_ref=_ENCODER_REF,
                quiescent=True,
            )
        )
    return (
        runner.FrozenAdaptationState(
            purpose="qualification",
            baseline_seals=tuple(seals),
            donor_catalogs=(catalog,),
        ),
        catalog,
        items,
    )


def _qualification_development_tasks() -> tuple[evaluator.PublicEvaluationTask, ...]:
    suite = evaluator.make_qualification_evaluator()
    adaptation = suite.release_phase("adaptation")
    _finalize_evaluator_phase(suite, adaptation)
    return suite.release_phase("development")


def _ordinary_private_replay(
    *,
    task: evaluator.PublicEvaluationTask,
    catalog: runner.DonorCatalog,
    items: tuple[CognitiveAcquisitionItem, ...],
    source_baseline_seal_ref: str,
) -> runner.PrivateRecallReplay:
    by_record = {item.record_ref: item for item in items}
    entries = catalog.entries[:12]
    public_batch = _RecallBatch(
        tuple(
            _RecallItem(
                record=by_record[entry.record_ref].acquisition.record,
                record_ref=entry.record_ref,
                acquisition_ref=entry.acquisition_ref,
                source_ref=entry.source_ref,
                text=entry.public_record_text,
                ordinal=entry.acquired_ordinal,
                source_contract=entry.source_contract,
                temporal_marker=f"ordinal:{entry.acquired_ordinal}",
            )
            for entry in entries
        )
    )
    source_hashes = {
        name: _raw(f"qualification-private-source-{task.task_id}-{name}")
        for name in runner.BASELINE_HASH_NAMES
    }
    capture = runner.freeze_recall_capture(
        task=task,
        kind="ordinary",
        recall_batch=public_batch,
        query_ref=_ref(f"qualification-private-recall-query-{task.task_id}"),
        backend_binding_ref=_ref("qualification-private-recall-backend"),
        source_baseline_seal_ref=source_baseline_seal_ref,
        capture_clone_ref=_ref(
            f"qualification-private-recall-clone-{task.task_id}"
        ),
        capture_quiescent=True,
        capture_disposed=True,
        cleanup_absence_verified=True,
        source_root=str((ROOT / "tests").resolve()),
        source_hashes=source_hashes,
    )
    hits = tuple(
        AcquisitionReferenceHit(
            record_ref=entry.record_ref,
            score=float(12 - index),
            backend_ref="v2-private-replay-test",
        )
        for index, entry in enumerate(entries)
    )
    recalls = tuple(
        AcquisitionRecall(
            record=by_record[entry.record_ref].acquisition.record,
            acquisition_ref=entry.acquisition_ref,
            source_contract=entry.source_contract,
            source_ref=entry.source_ref,
            position=TemporalPosition(
                acquired_ordinal=entry.acquired_ordinal,
                age=0,
                landmark_relations=(),
                world_valid_from=None,
                world_valid_until=None,
            ),
            adjacent_records=(),
            backend_score=float(12 - index),
            backend_ref="v2-private-replay-test",
            world_valid_at_query=None,
        )
        for index, entry in enumerate(entries)
    )
    return runner.PrivateRecallReplay(
        capture=capture,
        hits=hits,
        batch=AcquisitionRecallBatch(items=recalls, rejected=()),
    )


def _prompt_plan_for_stateful_arm(
    *,
    task: evaluator.PublicEvaluationTask,
    arm: runner.Arm,
    ordinary_capture: runner.FrozenRecallCapture | None = None,
) -> runner.PromptPlan:
    if arm == "IN_FULL_NEUTRAL_12":
        history = runner.neutral_history_plan(task.task_id, arm, _neutral_episodes())
    elif arm == "IO_FULL_ORDINARY_12" and ordinary_capture is not None:
        history = runner.recall_history_plan(
            task.task_id,
            arm,
            ordinary_capture,
            neutral=False,
        )
    else:
        raise AssertionError("stateful prompt fixture requires its exact history")
    prompt = runner.build_v1_proposal_prompt(
        "One bounded public synthetic qualification task.",
        history.evidence_texts,
    )
    token_ids = tuple(range(1, len(prompt.split()) + 1))
    return runner.PromptPlan(
        arm=arm,
        task_id=task.task_id,
        history_plan=history,
        prompt=prompt,
        prompt_ref=runner.content_ref({"prompt": prompt}),
        token_ids=token_ids,
        padding_atoms=0,
        target_tokens=len(token_ids),
    )


def _record_malformed_terminal_disposition(
    *,
    ledger: runner.EvidenceLedger,
    stage: runner.AttemptStage,
    prompt_plan: runner.PromptPlan,
    observation: runner.AttemptObservation,
) -> str:
    if observation.proposal_admitted:
        raise AssertionError("malformed disposition helper received an admission")
    receipt_ref = ledger.append_stage(stage, prompt_plan)
    ledger.append_artifact_durable(
        observation.observation_ref,
        "attempt-observation",
        observation.to_canonical(),
    )
    response_payload = {
        "arm": stage.arm,
        "attempt_receipt_ref": receipt_ref,
        "protocol_identity": runner.PROTOCOL_IDENTITY,
        "raw_response": "",
        "record_kind": "response",
        "task_id": stage.task_id,
    }
    response_ref = runner.domain_ref(
        runner.EVALUATOR_RECORD_DOMAIN,
        response_payload,
    )
    judgment_without_ref = {
        "arm": stage.arm,
        "attempt_receipt_ref": receipt_ref,
        "disposition": "UNSUCCESSFUL",
        "protocol_identity": runner.PROTOCOL_IDENTITY,
        "proposal_admitted": False,
        "public_task_ref": stage.public_task_ref,
        "raw_response": "",
        "record_kind": "objective-judgment",
        "response_commitment": response_ref,
        "response_conforms": False,
        "score": 0.0,
        "success": False,
        "task_id": stage.task_id,
    }
    objective_ref = runner.domain_ref(
        runner.EVALUATOR_RECORD_DOMAIN,
        judgment_without_ref,
    )
    judgment = {
        **judgment_without_ref,
        "evaluator_record_ref": objective_ref,
    }
    ledger.append_artifact_durable(
        runner.canonical_ref("objective-judgment", judgment),
        "objective-judgment",
        judgment,
    )
    witness = runner.make_congruence_witness(
        task_id=stage.task_id,
        arm=stage.arm,
        attempt_receipt_ref=receipt_ref,
        selected_trace_ref=None,
        execution_request_ref=None,
        execution_receipt_ref=None,
        response_ref=None,
        objective_judgment_ref=objective_ref,
        selected_trace_bound=None,
        response_bound=None,
        response_conforms=None,
        objective_success=None,
        selected_grammar_class=None,
        response_grammar_class=None,
    )
    return ledger.record_terminal_disposition(
        stage=stage,
        attempt_receipt_ref=receipt_ref,
        pre_feedback_observation_ref=observation.observation_ref,
        observation=observation,
        judgment=judgment,
        witness=witness,
        clone_terminal=None,
    )


def _qualification_result_fixture(root: Path) -> dict[str, object]:
    clone_parent = root / "result-clones"
    clone_parent.mkdir()
    suite = evaluator.make_qualification_evaluator()
    attempts: list[runner.AttemptFinal] = []
    pair_equalities: list[runner.PairEqualityEvidence] = []
    adaptation_tasks = suite.release_phase("adaptation")
    for scheduled in runner.adaptation_schedule(
        adaptation_tasks,
        purpose="qualification",
    ):
        task = scheduled.task
        attempt = _attempt_final(
            task_id=scheduled.task_id,
            public_task_ref=str(task["public_task_ref"]),
            replicate=str(task["replicate"]),
            replicate_commitment=scheduled.replicate_commitment,
            family=scheduled.family,
            arm=scheduled.arm,
            block_ordinal=scheduled.block_ordinal,
            success=False,
            response_conforms=False,
            purpose="qualification",
            phase="adaptation",
            task_ordinal=int(task["ordinal"]),
        )
        judgment = suite.record_attempt(
            scheduled.task_id,
            scheduled.arm,
            attempt.attempt_receipt_ref,
            attempt.observation.raw_response,
            proposal_admitted=True,
        )
        if judgment.success or judgment.response_conforms:
            raise AssertionError("qualification result fixture must remain unsuccessful")
        attempts.append(attempt)
    adaptation_attempts = tuple(attempts)
    suite.complete_phase("adaptation")
    development_tasks = suite.release_phase("development")
    for task in development_tasks:
        for left, right in (
            ("F0_NEUTRAL_FORMATTED_12", "IN_FULL_NEUTRAL_12"),
            ("O0_ORDINARY_FROZEN_12", "IO_FULL_ORDINARY_12"),
        ):
            pair_equalities.append(
                runner.PairEqualityEvidence(
                    left_arm=left,
                    right_arm=right,
                    task_id=task.task_id,
                    history_text_equal=True,
                    prompt_bytes_equal=True,
                    token_ids_equal=True,
                    padding_atoms_equal=True,
                    proposal_sets_equal=True,
                )
            )
    for scheduled in runner.development_final_schedule(
        development_tasks,
        purpose="qualification",
        phase="development",
    ):
        task = scheduled.task
        attempt = _attempt_final(
            task_id=scheduled.task_id,
            public_task_ref=str(task["public_task_ref"]),
            replicate=str(task["replicate"]),
            replicate_commitment=scheduled.replicate_commitment,
            family=scheduled.family,
            arm=scheduled.arm,
            block_ordinal=scheduled.block_ordinal,
            success=False,
            response_conforms=False,
            purpose="qualification",
            phase="development",
            task_ordinal=int(task["ordinal"]),
            clone_parent=clone_parent,
        )
        judgment = suite.record_attempt(
            scheduled.task_id,
            scheduled.arm,
            attempt.attempt_receipt_ref,
            attempt.observation.raw_response,
            proposal_admitted=True,
        )
        if judgment.success or judgment.response_conforms:
            raise AssertionError("qualification result fixture must remain unsuccessful")
        attempts.append(attempt)
    suite.complete_phase("development")
    metrics = suite.final_metrics().to_canonical()
    receipts_sha256 = hashlib.sha256(
        runner.canonical_json_bytes(
            sorted(item.attempt_receipt_ref for item in attempts)
        )
    ).hexdigest()
    terminals_sha256 = hashlib.sha256(
        runner.canonical_json_bytes(sorted(item.final_ref for item in attempts))
    ).hexdigest()
    stages_sha256 = hashlib.sha256(
        runner.canonical_json_bytes(sorted(item.stage.stage_ref for item in attempts))
    ).hexdigest()
    dispositions_sha256 = hashlib.sha256(
        runner.canonical_json_bytes(
            sorted(item.terminal_disposition_ref for item in attempts)
        )
    ).hexdigest()
    witnesses_sha256 = hashlib.sha256(
        runner.canonical_json_bytes(
            sorted(item.witness.witness_ref for item in attempts)
        )
    ).hexdigest()
    final_outcomes_sha256 = hashlib.sha256(
        runner.canonical_json_bytes([])
    ).hexdigest()
    witness_dispositions = {
        name: sum(item.witness.disposition == name for item in attempts)
        for name in ("SATISFIED", "UNKNOWN", "VIOLATED")
    }
    attempt_bindings = sorted(
        (runner._ledger_attempt_binding_from_final(item) for item in attempts),
        key=lambda item: (item["task_id"], item["arm"]),
    )
    accounting = {
        "arm_tasks": 72,
        "execution_attempts": 72,
        "generation_attempts": 144,
        "proposal_attempts": 72,
    }
    resources = {
        "configured_pools": 0,
        "embedding_rows": 0,
        "generation_calls": 144,
        "mode": "qualification",
        "output_tokens": 1_152,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
        "peak_processes": 0,
        "peak_rss_bytes": 0,
        "peak_state_scratch_bytes": 0,
        "prompt_tokens": 9_216,
        "runtime_embedding_rows": 54 * 15,
        "samples": 144,
        "schema": runner.RESOURCE_SCHEMA,
        "wall_seconds": 0.0,
    }
    frozen, _catalog, _items = _qualification_frozen_state(
        runner.STATE_ROOT / "qualification-v1" / "arm-runtime",
        donor_attempts=adaptation_attempts,
    )
    commitments = suite.commitments.to_canonical()
    source_seal_ref = _ref("qualification-result-source-seal")
    qualification_release_ref = _ref("qualification-result-release")
    metrics_artifact_ref = runner.canonical_ref(
        "evaluator-final-metrics",
        metrics,
    )
    closure = runner._run_provenance_closure_payload(
        purpose="qualification",
        authorization_ref=qualification_release_ref,
        source_seal_ref=source_seal_ref,
        evaluator_commitments_ref=runner._evaluator_digest(
            "evaluator-commitments",
            commitments,
        ),
        frozen_adaptation_ref=frozen.freeze_ref,
        evaluator_metrics_artifact_ref=metrics_artifact_ref,
    )
    return runner.build_run_result(
        purpose="qualification",
        attempts=tuple(attempts),
        pair_equalities=tuple(pair_equalities),
        frozen=frozen,
        evaluator_commitments=commitments,
        evaluator_metrics=metrics,
        statistics=None,
        accounting=accounting,
        resources=resources,
        source_seal_ref=source_seal_ref,
        authorization_ref=qualification_release_ref,
        ledger_witness={
            "attempts": 72,
            "attempt_bindings": attempt_bindings,
            "artifact_refs_sha256": _raw("qualification-ledger-artifacts"),
            "artifacts": 1,
            "disposition_refs_sha256": dispositions_sha256,
            "evaluator_metrics_artifact_ref": metrics_artifact_ref,
            "final_outcomes_sha256": final_outcomes_sha256,
            "finalized": 72,
            "ledger_bytes": 1,
            "ledger_sha256": _raw("qualification-ledger"),
            "receipt_refs_sha256": receipts_sha256,
            "run_provenance_closure": {
                "closure": closure,
                "closure_ref": runner.canonical_ref(
                    "run-provenance-closure",
                    closure,
                ),
            },
            "stage_refs_sha256": stages_sha256,
            "terminal_refs_sha256": terminals_sha256,
            "witness_dispositions": witness_dispositions,
            "witness_refs_sha256": witnesses_sha256,
        },
    )


class NeutralStoreIntegrationTests(unittest.TestCase):
    def test_exact_neutral_chain_round_trips_through_transaction_store(self) -> None:
        bootstrap = _bootstrap_episode()
        episodes = _neutral_episodes()
        with tempfile.TemporaryDirectory() as directory:
            store = CognitiveTransactionStore(Path(directory) / "neutral.sqlite3")
            store.initialize(
                _STATE_REF,
                _SNAPSHOT,
                model_ref=_MODEL_REF,
                encoder_ref=_ENCODER_REF,
            )
            store.commit_episode(
                bootstrap,
                _SNAPSHOT,
                expected_parent_digest=_STATE_REF,
            )
            for episode in episodes:
                store.commit_episode(
                    episode,
                    _SNAPSHOT,
                    expected_parent_digest=_STATE_REF,
                )

            evidence = runner.validate_neutral_fixture_chain(
                episodes,
                bootstrap_episode_ref=bootstrap.episode_ref,
                competence_state_digest=_STATE_REF,
                model_ref=_MODEL_REF,
                encoder_ref=_ENCODER_REF,
            )
            stored = store.episode_items(limit=64)
            acquisitions = store.acquisition_items(limit=64)
            self.assertEqual(tuple(item.sequence for item in stored), tuple(range(1, 14)))
            self.assertEqual(
                tuple(item.episode for item in stored[1:]),
                episodes,
            )
            self.assertEqual(
                tuple(item.episode_ref for item in stored[1:]),
                tuple(episode.episode_ref for episode in episodes),
            )
            self.assertEqual(
                tuple(item.ordinal for item in acquisitions),
                tuple(range(13)),
            )
            all_episodes = (bootstrap, *episodes)
            self.assertEqual(
                tuple(item.acquisition.source_ref for item in acquisitions),
                tuple(item.episode_ref for item in all_episodes),
            )
            self.assertEqual(
                tuple(
                    item.acquisition.predecessor_acquisition_ref
                    for item in acquisitions
                ),
                (None, *tuple(item.acquisition_ref for item in acquisitions[:-1])),
            )
            self.assertTrue(
                all(
                    item.acquisition.source_ref
                    in item.acquisition.record.provenance_refs
                    for item in acquisitions
                )
            )
            self.assertTrue(
                all(
                    item.acquisition.source_contract
                    == "ANG-CTR-COGNITIVE-EPISODE-001@0.1.0"
                    for item in acquisitions
                )
            )
            head = store.audit_integrity()
            self.assertEqual(head.sequence, 13)
            self.assertEqual(head.episode_ref, episodes[-1].episode_ref)
            self.assertEqual(head.state_digest, _STATE_REF)
            self.assertEqual(store.load_head_state(), _SNAPSHOT)
            acquisition_head = store.acquisition_head()
            self.assertEqual(acquisition_head.next_ordinal, 13)
            self.assertEqual(
                (acquisition_head.acquisition_ref, acquisition_head.record_ref),
                (acquisitions[-1].acquisition_ref, acquisitions[-1].record_ref),
            )
            self.assertEqual(
                tuple(item.episode_ref for item in store.pending_projections(64)),
                tuple(item.episode_ref for item in all_episodes),
            )
            self.assertEqual(
                tuple(
                    (item.ordinal, item.acquisition_ref, item.record_ref)
                    for item in store.pending_acquisition_projections(64)
                ),
                tuple(
                    (item.ordinal, item.acquisition_ref, item.record_ref)
                    for item in acquisitions
                ),
            )
            self.assertEqual(evidence["episode_refs"], [item.episode_ref for item in episodes])
            self.assertEqual(len(set(evidence["presentation_sha256"])), 12)
            with self.assertRaisesRegex(runner.V2InvariantError, "fixture 00|commitment"):
                runner.validate_neutral_fixture_chain(
                    (episodes[1], episodes[0], *episodes[2:]),
                    bootstrap_episode_ref=bootstrap.episode_ref,
                    competence_state_digest=_STATE_REF,
                    model_ref=_MODEL_REF,
                    encoder_ref=_ENCODER_REF,
                )


class V2ProductionLineageOwnerTests(unittest.TestCase):
    @staticmethod
    def _private_parent(path: Path) -> Path:
        path.mkdir(mode=0o700)
        path.chmod(0o700)
        return path

    def test_durable_allocation_and_empty_parent_gates_precede_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            runtime_parent = self._private_parent(base / "runtime")
            scope_parent = self._private_parent(base / "scopes")
            root = self._private_parent(runtime_parent / "allocated-lineage")
            scope_spec = runner.V2AcquisitionScopeSpec.create(
                purpose="qualification",
                replicate_commitment=_REPLICATE_REF,
                arm="IN_FULL_NEUTRAL_12",
                scope_kind="persistent-adaptation",
                state_parent=scope_parent,
            )
            allocation = runner.V2PersistentLineageAllocation(
                purpose="qualification",
                replicate_commitment=_REPLICATE_REF,
                arm="IN_FULL_NEUTRAL_12",
                runtime_root=str(root),
                cognee_scope_root=scope_spec.state_root,
                injected_paths=True,
            )
            ledger = runner.EvidenceLedger(base / "allocation.sqlite3", create=True)
            events: list[str] = []
            mechanics = _owner_mechanics(events, ledger=ledger)
            opener = _OwnerAcquisitionOpener(events)

            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "required durable ledger artifact is absent",
            ):
                asyncio.run(
                    runner.V2LineageOwner.create_from_allocated_root(
                        root,
                        purpose="qualification",
                        replicate_commitment=_REPLICATE_REF,
                        arm="IN_FULL_NEUTRAL_12",
                        genesis=object(),
                        manifest=_OwnerManifest(),
                        scope_spec=scope_spec,
                        allocation=allocation,
                        ledger=ledger,
                        acquisition_opener=opener,
                        mechanics=mechanics,
                    )
                )
            self.assertEqual(events, [])
            self.assertEqual(tuple(root.iterdir()), ())
            self.assertFalse(Path(scope_spec.state_root).exists())

            blocked_runtime = self._private_parent(base / "blocked-runtime")
            blocked_scopes = self._private_parent(base / "blocked-scopes")
            (blocked_runtime / "unexpected-child").write_bytes(b"occupied")
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "runtime parent emptiness differs",
            ):
                asyncio.run(
                    runner.V2LineageSetOwner.create_from_phase_parents(
                        purpose="qualification",
                        replicate_commitments=(_REPLICATE_REF,),
                        arm_runtime_parent=blocked_runtime,
                        cognee_scope_parent=blocked_scopes,
                        genesis=object(),
                        manifest=_OwnerManifest(),
                        ledger=ledger,
                        acquisition_opener=opener,
                        mechanics=mechanics,
                        injected_paths=True,
                    )
                )
            artifact_count = ledger._connection.execute(
                "SELECT COUNT(*) FROM artifacts"
            ).fetchone()[0]
            self.assertEqual(artifact_count, 0)
            self.assertEqual(events, [])
            ledger.close()

    def test_lineage_set_builds_exact_disjoint_neutral_geneses_and_seals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            runtime_parent = self._private_parent(base / "runtime")
            scope_parent = self._private_parent(base / "scopes")
            ledger = runner.EvidenceLedger(base / "lineages.sqlite3", create=True)
            events: list[str] = []
            mechanics = _owner_mechanics(events, ledger=ledger)
            opener = _OwnerAcquisitionOpener(events)

            lineages = asyncio.run(
                runner.V2LineageSetOwner.create_from_phase_parents(
                    purpose="qualification",
                    replicate_commitments=(_REPLICATE_REF,),
                    arm_runtime_parent=runtime_parent,
                    cognee_scope_parent=scope_parent,
                    genesis=object(),
                    manifest=_OwnerManifest(),
                    ledger=ledger,
                    acquisition_opener=opener,
                    mechanics=mechanics,
                    injected_paths=True,
                )
            )
            owners = {
                owner.arm: owner
                for owner in lineages.owners
            }
            self.assertEqual(set(owners), set(runner.STATEFUL_ARMS))
            self.assertEqual(
                events.count("paths-after-allocation:1"),
                1,
            )
            self.assertEqual(
                events.count("paths-after-allocation:2"),
                1,
            )
            self.assertEqual(
                ledger._connection.execute(
                    "SELECT COUNT(*) FROM artifacts "
                    "WHERE kind='persistent-lineage-allocation'"
                ).fetchone()[0],
                2,
            )

            for arm, owner in owners.items():
                with self.subTest(arm=arm):
                    stored = owner.store.episode_items(limit=64)
                    acquisitions = owner.store.acquisition_items(limit=64)
                    head = owner.store.audit_integrity()
                    self.assertEqual(
                        tuple(item.sequence for item in stored),
                        tuple(range(1, 14)),
                    )
                    self.assertEqual(head.sequence, 13)
                    self.assertEqual(owner.store.acquisition_head().next_ordinal, 13)
                    self.assertEqual(len(acquisitions), 13)
                    self.assertEqual(
                        tuple(item.episode_ref for item in stored[1:]),
                        owner.neutral_episode_refs,
                    )
                    self.assertNotIn(
                        owner.bootstrap_episode_ref,
                        owner.neutral_episode_refs,
                    )
                    self.assertEqual(
                        owner.genesis_competence_digest,
                        runner.QUALIFIED_INITIAL_COMPETENCE_DIGEST,
                    )
                    self.assertEqual(owner.model_ref, _MODEL_REF)
                    self.assertEqual(owner.encoder_ref, _ENCODER_REF)
                    self.assertFalse(owner.store.pending_acquisition_projections(limit=1))
                    self.assertFalse(Path(owner.scope_spec.state_root).exists())
                    self.assertLess(
                        events.index(f"memory-nonempty:{arm}:13"),
                        events.index(f"rebuild:{arm}"),
                    )
                    self.assertLess(
                        events.index(f"rebuild:{arm}"),
                        events.index(f"retry:{arm}"),
                    )
                    self.assertLess(
                        events.index(f"retry:{arm}"),
                        events.index(f"synchronized:{arm}"),
                    )
                    self.assertLess(
                        events.index(f"synchronized:{arm}"),
                        events.index(f"forget:{arm}"),
                    )
                    self.assertLess(
                        events.index(f"forget:{arm}"),
                        events.index(f"close:{arm}"),
                    )
                    self.assertEqual(events.count(f"project:{arm}"), 13)

            left = owners["IN_FULL_NEUTRAL_12"]
            right = owners["IO_FULL_ORDINARY_12"]
            self.assertNotEqual(left.root, right.root)
            self.assertNotEqual(
                os.stat(left.root).st_ino,
                os.stat(right.root).st_ino,
            )
            self.assertNotEqual(left.scope_spec.scope_ref, right.scope_spec.scope_ref)
            self.assertFalse(hasattr(left, "dataset_id"))
            self.assertFalse(hasattr(right, "dataset_id"))
            self.assertEqual(left.neutral_episode_refs, right.neutral_episode_refs)
            self.assertEqual(left.neutral_chain_ref, right.neutral_chain_ref)
            self.assertEqual(tuple(scope_parent.iterdir()), ())
            self.assertNotIn("search:IN_FULL_NEUTRAL_12", events)
            self.assertNotIn("search:IO_FULL_ORDINARY_12", events)

            original_allocation = left.allocation
            left.allocation = right.allocation
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "allocation binding changed",
            ):
                left.hashes()
            left.allocation = original_allocation

            original_paths = left.paths
            left.paths = replace(left.paths, journal=right.paths.journal)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "owner path binding changed",
            ):
                left.hashes()
            left.paths = original_paths

            substitute_ledger = runner.EvidenceLedger(
                base / "substitute-ledger.sqlite3",
                create=True,
            )
            original_ledger = left.ledger
            left.ledger = substitute_ledger
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "required durable ledger artifact is absent",
            ):
                left.hashes()
            left.ledger = original_ledger
            substitute_ledger.close()

            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "adaptation freeze attempt count differs",
            ):
                lineages.freeze(())
            seals = {
                arm: owner.persist_and_seal()
                for arm, owner in owners.items()
            }
            self.assertEqual(
                runner.validate_genesis_pair(
                    dict(seals["IN_FULL_NEUTRAL_12"].hashes),
                    dict(seals["IO_FULL_ORDINARY_12"].hashes),
                    in_root=left.root,
                    io_root=right.root,
                ),
                dict(seals["IN_FULL_NEUTRAL_12"].hashes),
            )
            for arm, seal in seals.items():
                with self.subTest(seal_arm=arm):
                    self.assertEqual(set(dict(seal.hashes)), runner.BASELINE_HASH_NAMES)
                    self.assertTrue(
                        all(
                            len(value) == 64
                            and set(value) <= set("0123456789abcdef")
                            for value in dict(seal.hashes).values()
                        )
                    )
                    self.assertEqual(
                        dict(seal.hashes),
                        dict(owners[arm].initial_hashes),
                    )
                    self.assertIs(owners[arm].verify_seal(), seal)

            with left.paths.journal.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "frozen V2 lineage bytes changed",
            ):
                left.verify_seal()
            ledger.close()

    def test_scope_cleanup_is_one_use_and_projection_failure_preserves_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            scope_parent = self._private_parent(base / "standalone-scopes")
            scope_spec = runner.V2AcquisitionScopeSpec.create(
                purpose="qualification",
                replicate_commitment=_REPLICATE_REF,
                arm="IO_FULL_ORDINARY_12",
                scope_kind="ordinary-recall-capture",
                state_parent=scope_parent,
                task_id=_ref("standalone-recall-task"),
            )
            events: list[str] = []
            opener = _OwnerAcquisitionOpener(events)
            binding, backend = asyncio.run(opener(scope_spec))
            session = runner.V2AcquisitionSession(
                scope_spec,
                binding,
                backend,
                binding.dataset_id,
            )
            asyncio.run(session.forget_close_and_remove())
            self.assertEqual(
                events,
                [
                    "open:IO_FULL_ORDINARY_12",
                    "forget:IO_FULL_ORDINARY_12",
                    "close:IO_FULL_ORDINARY_12",
                ],
            )
            self.assertFalse(Path(scope_spec.state_root).exists())
            self.assertIsNone(session.binding)
            self.assertIsNone(session.backend)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "already closed",
            ):
                asyncio.run(session.forget_close_and_remove())

            runtime_parent = self._private_parent(base / "failure-runtime")
            failure_scopes = self._private_parent(base / "failure-scopes")
            root = runtime_parent / "preserved-lineage"
            failed_scope = runner.V2AcquisitionScopeSpec.create(
                purpose="qualification",
                replicate_commitment=_REPLICATE_REF,
                arm="IO_FULL_ORDINARY_12",
                scope_kind="persistent-adaptation",
                state_parent=failure_scopes,
            )
            allocation = runner.V2PersistentLineageAllocation(
                purpose="qualification",
                replicate_commitment=_REPLICATE_REF,
                arm="IO_FULL_ORDINARY_12",
                runtime_root=str(root),
                cognee_scope_root=failed_scope.state_root,
                injected_paths=True,
            )
            ledger = runner.EvidenceLedger(base / "failure.sqlite3", create=True)
            ledger.append_artifact_durable(
                allocation.allocation_ref,
                "persistent-lineage-allocation",
                allocation.to_canonical(),
            )
            self._private_parent(root)
            failure_events: list[str] = []
            mechanics = _owner_mechanics(
                failure_events,
                ledger=ledger,
                fail_rebuild_arms=frozenset({"IO_FULL_ORDINARY_12"}),
            )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "V2_LINEAGE_PROJECTION_FAILED",
            ):
                asyncio.run(
                    runner.V2LineageOwner.create_from_allocated_root(
                        root,
                        purpose="qualification",
                        replicate_commitment=_REPLICATE_REF,
                        arm="IO_FULL_ORDINARY_12",
                        genesis=object(),
                        manifest=_OwnerManifest(),
                        scope_spec=failed_scope,
                        allocation=allocation,
                        ledger=ledger,
                        acquisition_opener=_OwnerAcquisitionOpener(failure_events),
                        mechanics=mechanics,
                    )
                )
            ledger.require_artifact(
                allocation.allocation_ref,
                "persistent-lineage-allocation",
                allocation.to_canonical(),
            )
            self.assertEqual(
                {item.name for item in root.iterdir()},
                {
                    "cognitive.sqlite3",
                    "qwen-execution-journal.sqlite3",
                    "learner-state.bin",
                },
            )
            preserved = CognitiveTransactionStore(root / "cognitive.sqlite3")
            self.assertEqual(preserved.audit_integrity().sequence, 13)
            self.assertEqual(preserved.acquisition_head().next_ordinal, 13)
            self.assertEqual(
                len(preserved.pending_acquisition_projections(limit=64)),
                13,
            )
            self.assertTrue(Path(failed_scope.state_root).exists())
            self.assertLess(
                failure_events.index("rebuild:IO_FULL_ORDINARY_12"),
                failure_events.index("close:IO_FULL_ORDINARY_12"),
            )
            self.assertNotIn("forget:IO_FULL_ORDINARY_12", failure_events)
            self.assertNotIn("retry:IO_FULL_ORDINARY_12", failure_events)
            self.assertNotIn("synchronized:IO_FULL_ORDINARY_12", failure_events)
            ledger.close()


class V2AttemptRuntimeOwnerTests(unittest.TestCase):
    def test_stateless_graph_is_guard_bound_and_cleanup_is_retriable(self) -> None:
        import gc
        import weakref

        from angler.runtime.qwen_cognitive import (
            FrozenQwenCognitiveExecutorV1,
            FrozenQwenProcedureAdapterV1,
            SQLiteQwenExecutionJournal,
        )

        manifest = object()

        def exact_adapter(io: object) -> FrozenQwenProcedureAdapterV1:
            value = object.__new__(FrozenQwenProcedureAdapterV1)
            value.io = io
            value.manifest = manifest
            value._last_proposal_generation = None
            return value

        def exact_executor(
            io: object,
            journal: object,
        ) -> FrozenQwenCognitiveExecutorV1:
            value = object.__new__(FrozenQwenCognitiveExecutorV1)
            value.io = io
            value.manifest = manifest
            value.journal = journal
            return value

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            state_root = base / "state"
            runtime_parent = state_root / "qualification-v1" / "arm-runtime"
            runtime_parent.mkdir(parents=True, mode=0o700)
            runtime_parent.chmod(0o700)
            ledger = runner.EvidenceLedger(base / "runtime.sqlite3", create=True)
            task = _qualification_development_tasks()[0]
            task_id = task.task_id
            arm = "F0_NEUTRAL_FORMATTED_12"
            history = runner.neutral_history_plan(
                task_id,
                arm,
                _neutral_episodes(),
            )
            prompt = runner.build_v1_proposal_prompt(
                "One bounded public synthetic qualification task.",
                history.evidence_texts,
            )
            token_ids = tuple(range(1, len(prompt.split()) + 1))
            prompt_plan = runner.PromptPlan(
                arm=arm,
                task_id=task_id,
                history_plan=history,
                prompt=prompt,
                prompt_ref=runner.content_ref({"prompt": prompt}),
                token_ids=token_ids,
                padding_atoms=0,
                target_tokens=len(token_ids),
            )
            scheduled = runner.ScheduledAttempt(
                purpose="qualification",
                phase="development",
                arm=arm,
                task=task.to_canonical(),
                block_ordinal=0,
                arm_position=runner.ARMS.index(arm),
            )
            stage = runner.AttemptStage.from_schedule(scheduled, prompt_plan)
            with mock.patch.object(runner, "STATE_ROOT", state_root):
                allocation = runner.DisposablePathAllocation(
                    purpose="qualification",
                    task_id=task_id,
                    replicate_commitment=task.replicate_commitment,
                    role=arm,
                    runtime_root=str(runtime_parent / "stateless-owner"),
                    cognee_scope_roots=(),
                )
                registry = runner.PathAllocationRegistry("qualification", ledger)
                registry.register(allocation)
            runtime_root = Path(allocation.runtime_root)
            runtime_root.mkdir(mode=0o700)
            runtime_root.chmod(0o700)
            journal = SQLiteQwenExecutionJournal(
                runtime_root / "qwen-execution-journal.sqlite3"
            )
            journal.path.chmod(0o600)
            io = object()
            guard = object.__new__(runner.GuardedLocalQwenBoundary)
            guard.io = io
            adapter = exact_adapter(io)
            executor = exact_executor(io, journal)

            def construct(
                *,
                observed_adapter: object = adapter,
                observed_executor: object = executor,
            ) -> runner.V2AttemptExecutionRuntime:
                return runner.V2AttemptExecutionRuntime(
                    arm=arm,
                    proposal_adapter=observed_adapter,
                    relation_adapter=None,
                    executor=observed_executor,
                    journal=journal,
                    guard=guard,
                    runtime_root=runtime_root,
                    allocation_ref=allocation.allocation_ref,
                    allocation=allocation,
                    path_registry=registry,
                )

            for label, observed_adapter, observed_executor in (
                ("adapter-io", exact_adapter(object()), executor),
                ("executor-io", adapter, exact_executor(object(), journal)),
                ("executor-journal", adapter, exact_executor(io, object())),
            ):
                with self.subTest(bypass=label), self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "bypasses its guard",
                ):
                    construct(
                        observed_adapter=observed_adapter,
                        observed_executor=observed_executor,
                    )

            runtime = construct()
            sentinel = type("GuardReleaseSentinel", (), {})()
            sentinel_ref = weakref.ref(sentinel)
            guard._release_sentinel = sentinel
            sentinel = None
            guard = None
            gc.collect()
            self.assertIsNotNone(sentinel_ref())
            runtime.allocation = None
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "cleanup boundary differs",
            ):
                runtime.dispose_stateless_durable(
                    stage=stage,
                    terminal_disposition_ref=_ref("stateless-disposition"),
                    ledger=ledger,
                )
            self.assertTrue(runtime_root.exists())
            runtime.allocation = allocation

            observation = runner.AttemptObservation(
                task_id=task_id,
                arm=arm,
                proposal_admitted=False,
                proposals=(),
                selected_index=None,
                selected_trace=None,
                raw_response="",
                proposal_generation_ref=_ref("stateless-owner-generation"),
                proposal_prompt_ref=prompt_plan.prompt_ref,
                proposal_prompt_tokens=len(prompt_plan.token_ids),
                proposal_output_tokens=1,
                execution_request_ref=None,
                execution_receipt_ref=None,
                execution_prompt_ref=None,
                execution_prompt_tokens=None,
                execution_output_tokens=None,
                selected_trace_bound=None,
                response_bound=None,
                selected_grammar_class=None,
                response_grammar_class=None,
                recalled_record_refs=tuple(
                    item.record_ref
                    for item in prompt_plan.history_plan.presentations
                ),
            )
            disposition_ref = _record_malformed_terminal_disposition(
                ledger=ledger,
                stage=stage,
                prompt_plan=prompt_plan,
                observation=observation,
            )

            original_remove = runner._remove_exact_private_tree
            removal_calls = 0

            def fail_once(path: Path, *, expected_parent: Path) -> None:
                nonlocal removal_calls
                removal_calls += 1
                if removal_calls == 1:
                    raise runner.V2IntegrityStop("injected pre-delete failure")
                original_remove(path, expected_parent=expected_parent)

            with (
                mock.patch.object(
                    runner,
                    "_remove_exact_private_tree",
                    side_effect=fail_once,
                ),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "injected pre-delete failure",
                ),
            ):
                runtime.dispose_stateless_durable(
                    stage=stage,
                    terminal_disposition_ref=disposition_ref,
                    ledger=ledger,
                )
            self.assertTrue(runtime_root.exists())
            self.assertIs(runtime.proposal_adapter, adapter)
            self.assertIs(runtime.executor, executor)
            self.assertIsNotNone(sentinel_ref())
            self.assertEqual(
                ledger._connection.execute(
                    "SELECT COUNT(*) FROM artifacts "
                    "WHERE kind='stateless-runtime-terminal'"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                ledger._connection.execute(
                    "SELECT COUNT(*) FROM artifacts "
                    "WHERE kind='stateless-runtime-cleanup'"
                ).fetchone()[0],
                0,
            )

            with mock.patch.object(
                runner,
                "_remove_exact_private_tree",
                side_effect=fail_once,
            ):
                cleanup = runtime.dispose_stateless_durable(
                    stage=stage,
                    terminal_disposition_ref=disposition_ref,
                    ledger=ledger,
                )
            self.assertEqual(removal_calls, 2)
            self.assertFalse(runtime_root.exists())
            for value in (
                runtime.proposal_adapter,
                runtime.relation_adapter,
                runtime.executor,
                runtime.journal,
                runtime.guard,
                runtime.cycle,
                runtime.learner,
                runtime.store,
                runtime.memory,
                runtime.session,
            ):
                self.assertIsNone(value)
            gc.collect()
            self.assertIsNone(sentinel_ref())
            ledger.require_artifact(
                cleanup.terminal.terminal_ref,
                "stateless-runtime-terminal",
                cleanup.terminal.to_canonical(),
            )
            ledger.require_artifact(
                cleanup.cleanup_ref,
                "stateless-runtime-cleanup",
                cleanup.to_canonical(),
            )
            registry.require_stateless(cleanup.terminal)
            registry.mark_closed(cleanup.terminal.allocation_ref)
            registry.assert_closed()
            ledger.close()

    def test_persistent_scope_cleanup_is_durable_for_both_feedback_states(self) -> None:
        class QuiescentCycle:
            pending_turn = None
            executed_turn = None
            _beginning = False
            _executing = False

        class Memory:
            def __init__(self, source: object, backend: object) -> None:
                self.source = source
                self.backend = backend

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            runtime_parent = base / "runtime"
            scope_parent = base / "scopes"
            replicate_commitment = runner.qualification_replicate_commitment()
            for parent in (runtime_parent, scope_parent):
                parent.mkdir(mode=0o700)
                parent.chmod(0o700)
            ledger = runner.EvidenceLedger(base / "persistent.sqlite3", create=True)
            construction_events: list[str] = []
            mechanics = _owner_mechanics(construction_events, ledger=ledger)
            opener = _OwnerAcquisitionOpener(construction_events)
            lineages = asyncio.run(
                runner.V2LineageSetOwner.create_from_phase_parents(
                    purpose="qualification",
                    replicate_commitments=(replicate_commitment,),
                    arm_runtime_parent=runtime_parent,
                    cognee_scope_parent=scope_parent,
                    genesis=object(),
                    manifest=_OwnerManifest(),
                    ledger=ledger,
                    acquisition_opener=opener,
                    mechanics=mechanics,
                    injected_paths=True,
                )
            )
            source_owner = lineages.owner_for(
                replicate_commitment,
                "IN_FULL_NEUTRAL_12",
            )
            qualification = evaluator.make_qualification_evaluator()
            scheduled_rows = tuple(
                item
                for item in runner.adaptation_schedule(
                    qualification.release_phase("adaptation"),
                    purpose="qualification",
                )
                if item.arm == "IN_FULL_NEUTRAL_12"
            )
            self.assertGreaterEqual(len(scheduled_rows), 2)
            incarnation_ids: list[str] = []

            for ordinal, feedback_applied in enumerate((False, True)):
                with self.subTest(feedback_applied=feedback_applied):
                    events: list[str] = []
                    attempt_opener = _OwnerAcquisitionOpener(
                        events,
                        incarnation=f"adaptation-{ordinal}",
                    )
                    session = asyncio.run(
                        runner._open_v2_acquisition_session(
                            source_owner.scope_spec,
                            opener=attempt_opener,
                        )
                    )
                    self.assertNotIn(session.dataset_id, incarnation_ids)
                    incarnation_ids.append(session.dataset_id)
                    memory = Memory(source_owner.store, session.backend)
                    cycle = QuiescentCycle()
                    runtime = object.__new__(runner.V2AttemptExecutionRuntime)
                    runtime.arm = source_owner.arm
                    runtime.proposal_adapter = object()
                    runtime.relation_adapter = object()
                    runtime.executor = object()
                    runtime.journal = source_owner.journal
                    runtime.cycle = cycle
                    runtime.learner = source_owner.learner
                    runtime.store = source_owner.store
                    runtime.memory = memory
                    runtime.session = session

                    def quiesce_runtime(**owners: object) -> None:
                        self.assertIs(owners["cycle"], cycle)
                        self.assertIs(owners["learner"], source_owner.learner)
                        self.assertIs(owners["store"], source_owner.store)
                        self.assertIs(owners["journal"], source_owner.journal)
                        self.assertIs(owners["memory"], memory)
                        events.append("quiesce")

                    attempt_owner = runner.V2StatefulAttemptOwner(
                        runtime=runtime,
                        source_owner=source_owner,
                        ledger=ledger,
                        quiesce_runtime=quiesce_runtime,
                    )
                    original_append = ledger.append_artifact_durable

                    def append_artifact(
                        artifact_ref: str,
                        kind: str,
                        payload: dict[str, object],
                    ) -> str:
                        value = original_append(artifact_ref, kind, payload)
                        if kind in (
                            "persistent-runtime-terminal",
                            "persistent-runtime-cleanup",
                        ):
                            events.append(f"{kind}-durable")
                        return value

                    scheduled = scheduled_rows[ordinal]
                    history = runner.neutral_history_plan(
                        scheduled.task_id,
                        scheduled.arm,
                        _neutral_episodes(),
                    )
                    prompt = runner.build_v1_proposal_prompt(
                        "One bounded public synthetic qualification task.",
                        history.evidence_texts,
                    )
                    token_ids = tuple(range(1, len(prompt.split()) + 1))
                    prompt_plan = runner.PromptPlan(
                        arm=scheduled.arm,
                        task_id=scheduled.task_id,
                        history_plan=history,
                        prompt=prompt,
                        prompt_ref=runner.content_ref({"prompt": prompt}),
                        token_ids=token_ids,
                        padding_atoms=0,
                        target_tokens=len(token_ids),
                    )
                    stage = runner.AttemptStage.from_schedule(
                        scheduled,
                        prompt_plan,
                    )
                    receipt_ref = ledger.append_stage(stage, prompt_plan)
                    raw_response = (
                        "SYNTHETIC_CONFORMING_RESPONSE"
                        if feedback_applied
                        else ""
                    )
                    proposals = (
                        (
                            "Use the first bounded public procedure.",
                            "Use the second bounded public procedure.",
                        )
                        if feedback_applied
                        else ()
                    )
                    observation = runner.AttemptObservation(
                        task_id=stage.task_id,
                        arm=stage.arm,
                        proposal_admitted=feedback_applied,
                        proposals=proposals,
                        selected_index=0 if feedback_applied else None,
                        selected_trace=(proposals[0] if feedback_applied else None),
                        raw_response=raw_response,
                        proposal_generation_ref=_ref(
                            f"persistent-proposal-generation-{ordinal}"
                        ),
                        proposal_prompt_ref=prompt_plan.prompt_ref,
                        proposal_prompt_tokens=len(prompt_plan.token_ids),
                        proposal_output_tokens=1,
                        execution_request_ref=(
                            _ref(f"persistent-execution-request-{ordinal}")
                            if feedback_applied
                            else None
                        ),
                        execution_receipt_ref=(
                            _ref(f"persistent-execution-receipt-{ordinal}")
                            if feedback_applied
                            else None
                        ),
                        execution_prompt_ref=(
                            _ref(f"persistent-execution-prompt-{ordinal}")
                            if feedback_applied
                            else None
                        ),
                        execution_prompt_tokens=1 if feedback_applied else None,
                        execution_output_tokens=1 if feedback_applied else None,
                        selected_trace_bound=True if feedback_applied else None,
                        response_bound=True if feedback_applied else None,
                        selected_grammar_class=(
                            "PUBLIC_SYNTHETIC" if feedback_applied else None
                        ),
                        response_grammar_class=(
                            "PUBLIC_SYNTHETIC" if feedback_applied else None
                        ),
                        resolution_ref=(
                            _ref(f"persistent-resolution-{ordinal}")
                            if feedback_applied
                            else None
                        ),
                        prospective_reservation_ref=(
                            _ref(f"persistent-reservation-{ordinal}")
                            if feedback_applied
                            else None
                        ),
                        prospective_batch_ref=(
                            _ref(f"persistent-batch-{ordinal}")
                            if feedback_applied
                            else None
                        ),
                        recalled_record_refs=tuple(
                            item.record_ref
                            for item in history.presentations
                        ),
                    )
                    ledger.append_artifact_durable(
                        observation.observation_ref,
                        "attempt-observation",
                        observation.to_canonical(),
                    )
                    response_payload = {
                        "arm": stage.arm,
                        "attempt_receipt_ref": receipt_ref,
                        "protocol_identity": runner.PROTOCOL_IDENTITY,
                        "raw_response": raw_response,
                        "record_kind": "response",
                        "task_id": stage.task_id,
                    }
                    response_ref = runner.domain_ref(
                        runner.EVALUATOR_RECORD_DOMAIN,
                        response_payload,
                    )
                    judgment_without_ref = {
                        "arm": stage.arm,
                        "attempt_receipt_ref": receipt_ref,
                        "disposition": "UNSUCCESSFUL",
                        "protocol_identity": runner.PROTOCOL_IDENTITY,
                        "proposal_admitted": feedback_applied,
                        "public_task_ref": stage.public_task_ref,
                        "raw_response": raw_response,
                        "record_kind": "objective-judgment",
                        "response_commitment": response_ref,
                        "response_conforms": False,
                        "score": 0.0,
                        "success": False,
                        "task_id": stage.task_id,
                    }
                    objective_ref = runner.domain_ref(
                        runner.EVALUATOR_RECORD_DOMAIN,
                        judgment_without_ref,
                    )
                    judgment = {
                        **judgment_without_ref,
                        "evaluator_record_ref": objective_ref,
                    }
                    ledger.append_artifact_durable(
                        runner.canonical_ref("objective-judgment", judgment),
                        "objective-judgment",
                        judgment,
                    )
                    witness = runner.make_congruence_witness(
                        task_id=stage.task_id,
                        arm=stage.arm,
                        attempt_receipt_ref=receipt_ref,
                        selected_trace_ref=(
                            runner.content_ref(
                                {"selected_trace": proposals[0]}
                            )
                            if feedback_applied
                            else None
                        ),
                        execution_request_ref=observation.execution_request_ref,
                        execution_receipt_ref=observation.execution_receipt_ref,
                        response_ref=(response_ref if feedback_applied else None),
                        objective_judgment_ref=objective_ref,
                        selected_trace_bound=observation.selected_trace_bound,
                        response_bound=observation.response_bound,
                        response_conforms=(False if feedback_applied else None),
                        objective_success=(False if feedback_applied else None),
                        selected_grammar_class=observation.selected_grammar_class,
                        response_grammar_class=observation.response_grammar_class,
                    )
                    disposition_ref = ledger.record_terminal_disposition(
                        stage=stage,
                        attempt_receipt_ref=receipt_ref,
                        pre_feedback_observation_ref=observation.observation_ref,
                        observation=observation,
                        judgment=judgment,
                        witness=witness,
                        clone_terminal=None,
                    )
                    events.append("disposition-durable")
                    with mock.patch.object(
                        ledger,
                        "append_artifact_durable",
                        side_effect=append_artifact,
                    ):
                        cleanup = asyncio.run(
                            attempt_owner.finalize_persistent(
                                stage=stage,
                                attempt_receipt_ref=receipt_ref,
                                terminal_disposition_ref=disposition_ref,
                                feedback_applied=feedback_applied,
                            )
                        )
                    self.assertEqual(
                        events,
                        [
                            "open:IN_FULL_NEUTRAL_12",
                            "disposition-durable",
                            "quiesce",
                            "persistent-runtime-terminal-durable",
                            "forget:IN_FULL_NEUTRAL_12",
                            "close:IN_FULL_NEUTRAL_12",
                            "persistent-runtime-cleanup-durable",
                        ],
                    )
                    self.assertEqual(cleanup.terminal.task_id, stage.task_id)
                    self.assertEqual(cleanup.terminal.stage_ref, stage.stage_ref)
                    self.assertEqual(
                        cleanup.terminal.attempt_receipt_ref,
                        receipt_ref,
                    )
                    self.assertEqual(
                        cleanup.terminal.terminal_disposition_ref,
                        disposition_ref,
                    )
                    self.assertIs(
                        cleanup.terminal.feedback_applied,
                        feedback_applied,
                    )
                    self.assertEqual(
                        cleanup.terminal.allocation_ref,
                        source_owner.allocation_ref,
                    )
                    self.assertEqual(
                        cleanup.terminal.dataset_id,
                        incarnation_ids[ordinal],
                    )
                    self.assertEqual(
                        dict(cleanup.terminal.hashes),
                        source_owner.hashes(),
                    )
                    self.assertTrue(source_owner.root.exists())
                    self.assertFalse(
                        Path(cleanup.terminal.cognee_scope_root).exists()
                    )
                    ledger.require_artifact(
                        cleanup.terminal.terminal_ref,
                        "persistent-runtime-terminal",
                        cleanup.terminal.to_canonical(),
                    )
                    ledger.require_artifact(
                        cleanup.cleanup_ref,
                        "persistent-runtime-cleanup",
                        cleanup.to_canonical(),
                    )
                    self.assertTrue(attempt_owner._closed)
                    self.assertIsNone(attempt_owner.runtime)
            self.assertEqual(len(set(incarnation_ids)), 2)
            ledger.close()

    def test_malformed_product_runtime_retains_f0_in_and_io_staged_recall(self) -> None:
        from angler.runtime.cognitive_cycle import CognitiveCycle
        from angler.runtime.prospective_observation import (
            QwenReceiptObservedStateEncoderV1,
        )
        from angler.runtime.qwen_cognitive import (
            FrozenQwenCognitiveExecutorV1,
            FrozenQwenProcedureAdapterV1,
            FrozenQwenProcedureRelationAdapterV1,
            SQLiteQwenExecutionJournal,
        )

        class AttemptContext:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *_args: object) -> bool:
                return False

        class Memory:
            def __init__(self, source: object, backend: object) -> None:
                self.source = source
                self.backend = backend

        class Situated:
            reality_mode = "SIMULATED"

        manifest = object()

        def products(
            *,
            io: object,
            journal: SQLiteQwenExecutionJournal,
            prompt_plan: runner.PromptPlan,
            captured_evidence: list[tuple[str, ...]],
        ) -> tuple[
            FrozenQwenProcedureAdapterV1,
            FrozenQwenProcedureRelationAdapterV1,
            FrozenQwenCognitiveExecutorV1,
        ]:
            adapter = object.__new__(FrozenQwenProcedureAdapterV1)
            adapter.io = io
            adapter.manifest = manifest
            adapter._last_proposal_generation = None

            def malformed_proposal(
                _request: str,
                evidence: tuple[str, ...],
                *,
                count: int,
            ) -> tuple[str, ...]:
                self.assertEqual(count, runner.PROPOSAL_COUNT)
                captured_evidence.append(tuple(evidence))
                adapter._last_proposal_generation = FrozenQwenGenerationRecord(
                    model_ref=None,
                    tokenizer_ref=None,
                    generation_config_ref=_ref("malformed-runtime-config"),
                    prompt_ref=prompt_plan.prompt_ref,
                    prompt_tokens=len(prompt_plan.token_ids),
                    generated_token_ids=(1,),
                    response="malformed single procedure",
                )
                raise ValueError("malformed proposal schema")

            adapter.propose_procedure_traces = malformed_proposal
            relation = object.__new__(FrozenQwenProcedureRelationAdapterV1)
            relation.io = io
            relation.manifest = manifest
            executor = object.__new__(FrozenQwenCognitiveExecutorV1)
            executor.io = io
            executor.manifest = manifest
            executor.journal = journal
            return adapter, relation, executor

        qualification = evaluator.make_qualification_evaluator()
        adaptation_task = qualification.release_phase("adaptation")[0]
        development_task = _qualification_development_tasks()[0]

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frozen, catalog, items = _qualification_frozen_state(base / "frozen")
            adaptation_baselines = frozen.baselines_for(
                adaptation_task.replicate_commitment
            )
            ordinary_replay = _ordinary_private_replay(
                task=adaptation_task,
                catalog=catalog,
                items=items,
                source_baseline_seal_ref=adaptation_baselines[
                    "IO_FULL_ORDINARY_12"
                ].seal_ref,
            )

            f0_history = runner.neutral_history_plan(
                development_task.task_id,
                "F0_NEUTRAL_FORMATTED_12",
                _neutral_episodes(),
            )
            f0_prompt = runner.build_v1_proposal_prompt(
                "One bounded public synthetic qualification task.",
                f0_history.evidence_texts,
            )
            f0_tokens = tuple(range(1, len(f0_prompt.split()) + 1))
            f0_plan = runner.PromptPlan(
                arm="F0_NEUTRAL_FORMATTED_12",
                task_id=development_task.task_id,
                history_plan=f0_history,
                prompt=f0_prompt,
                prompt_ref=runner.content_ref({"prompt": f0_prompt}),
                token_ids=f0_tokens,
                padding_atoms=0,
                target_tokens=len(f0_tokens),
            )
            plans = {
                "F0_NEUTRAL_FORMATTED_12": (
                    development_task,
                    f0_plan,
                ),
                "IN_FULL_NEUTRAL_12": (
                    adaptation_task,
                    _prompt_plan_for_stateful_arm(
                        task=adaptation_task,
                        arm="IN_FULL_NEUTRAL_12",
                    ),
                ),
                "IO_FULL_ORDINARY_12": (
                    adaptation_task,
                    _prompt_plan_for_stateful_arm(
                        task=adaptation_task,
                        arm="IO_FULL_ORDINARY_12",
                        ordinary_capture=ordinary_replay.capture,
                    ),
                ),
            }

            for arm in (
                "F0_NEUTRAL_FORMATTED_12",
                "IN_FULL_NEUTRAL_12",
                "IO_FULL_ORDINARY_12",
            ):
                with self.subTest(arm=arm):
                    task, prompt_plan = plans[arm]
                    scheduled = runner.ScheduledAttempt(
                        purpose="qualification",
                        phase=(
                            "development"
                            if arm == "F0_NEUTRAL_FORMATTED_12"
                            else "adaptation"
                        ),
                        arm=arm,
                        task=task.to_canonical(),
                        block_ordinal=0,
                        arm_position=runner.ARMS.index(arm),
                    )
                    stage = runner.AttemptStage.from_schedule(
                        scheduled,
                        prompt_plan,
                    )
                    budget = runner.AttemptBudget(runner.RunMode.QUALIFICATION)
                    permit = runner.AttemptGenerationPermit(
                        budget,
                        task.task_id,
                        arm,
                    )
                    io = object()
                    guard = object.__new__(runner.GuardedLocalQwenBoundary)
                    guard.io = io
                    if arm == "F0_NEUTRAL_FORMATTED_12":
                        state_root = base / "stateless-state"
                        runtime_parent = (
                            state_root / "qualification-v1" / "arm-runtime"
                        )
                        runtime_parent.mkdir(parents=True, mode=0o700)
                        runtime_parent.chmod(0o700)
                        attempt_root = runtime_parent / "attempt-F0"
                        ledger = runner.EvidenceLedger(
                            base / "f0-ledger.sqlite3",
                            create=True,
                        )
                        with mock.patch.object(runner, "STATE_ROOT", state_root):
                            allocation = runner.DisposablePathAllocation(
                                purpose="qualification",
                                task_id=task.task_id,
                                replicate_commitment=task.replicate_commitment,
                                role=arm,
                                runtime_root=str(attempt_root),
                                cognee_scope_roots=(),
                            )
                            registry = runner.PathAllocationRegistry(
                                "qualification",
                                ledger,
                            )
                            registry.register(allocation)
                    else:
                        attempt_root = base / f"attempt-{arm}"
                    attempt_root.mkdir(mode=0o700)
                    attempt_root.chmod(0o700)
                    journal = SQLiteQwenExecutionJournal(
                        attempt_root / "qwen-execution-journal.sqlite3"
                    )
                    journal.path.chmod(0o600)
                    captured_evidence: list[tuple[str, ...]] = []
                    adapter, relation, executor = products(
                        io=io,
                        journal=journal,
                        prompt_plan=prompt_plan,
                        captured_evidence=captured_evidence,
                    )
                    executor.execute = mock.Mock(
                        side_effect=AssertionError(
                            "malformed proposal reached execution"
                        )
                    )
                    session: runner.V2AcquisitionSession | None = None
                    store: CognitiveTransactionStore | None = None
                    cycle: CognitiveCycle | None = None
                    scope_events: list[str] = []
                    if arm in runner.STATEFUL_ARMS:
                        learner = _OwnerLearner()
                        store = CognitiveTransactionStore(
                            attempt_root / "cognitive.sqlite3"
                        )
                        store.initialize(
                            learner.state_digest(),
                            learner.capture_state(),
                            model_ref=_MODEL_REF,
                            encoder_ref=_ENCODER_REF,
                        )
                        scope_parent = base / f"scope-parent-{arm}"
                        scope_parent.mkdir(mode=0o700)
                        scope_parent.chmod(0o700)
                        scope_spec = runner.V2AcquisitionScopeSpec.create(
                            purpose="qualification",
                            replicate_commitment=task.replicate_commitment,
                            arm=arm,
                            scope_kind="persistent-adaptation",
                            state_parent=scope_parent,
                        )
                        scope_opener = _OwnerAcquisitionOpener(scope_events)
                        session = asyncio.run(
                            runner._open_v2_acquisition_session(
                                scope_spec,
                                opener=scope_opener,
                            )
                        )
                        memory = Memory(store, session.backend)
                        cycle = object.__new__(CognitiveCycle)
                        cycle.text_adapter = adapter
                        cycle.relation_adapter = relation
                        cycle.executor = executor
                        cycle.learner = learner
                        cycle.transaction_store = store
                        cycle.memory = memory
                        cycle.proposal_count = runner.PROPOSAL_COUNT
                        cycle.recall_limit = runner.RECALL_SLOTS
                        cycle._successor_mode = True
                        cycle._removal_condition = "FULL"
                        cycle._situated_context = Situated()
                        cycle._observation_encoder = object.__new__(
                            QwenReceiptObservedStateEncoderV1
                        )
                        cycle._pending = None
                        cycle._executed = None
                        cycle._beginning = False
                        cycle._executing = False

                        async def begin_turn(
                            _task_id: str,
                            _request: str,
                        ) -> object:
                            return adapter.propose_procedure_traces(
                                _request,
                                prompt_plan.history_plan.evidence_texts,
                                count=runner.PROPOSAL_COUNT,
                            )

                        cycle.begin_turn = begin_turn
                        cycle.execute_turn = mock.Mock(
                            side_effect=AssertionError(
                                "malformed proposal reached cycle execution"
                            )
                        )
                        cycle.record_outcome = mock.Mock(
                            side_effect=AssertionError(
                                "malformed proposal reached feedback"
                            )
                        )
                        cycle.retry_pending_projections = mock.Mock()
                        runtime = runner.V2AttemptExecutionRuntime(
                            arm=arm,
                            proposal_adapter=adapter,
                            relation_adapter=relation,
                            executor=executor,
                            journal=journal,
                            guard=guard,
                            cycle=cycle,
                            learner=learner,
                            store=store,
                            memory=memory,
                            session=session,
                        )
                    else:
                        runtime = runner.V2AttemptExecutionRuntime(
                            arm=arm,
                            proposal_adapter=adapter,
                            relation_adapter=None,
                            executor=executor,
                            journal=journal,
                            guard=guard,
                            runtime_root=attempt_root,
                            allocation_ref=allocation.allocation_ref,
                            allocation=allocation,
                            path_registry=registry,
                        )

                    with mock.patch.object(
                        runner.GuardedLocalQwenBoundary,
                        "attempt",
                        return_value=AttemptContext(),
                    ):
                        observation = asyncio.run(
                            runtime.execute_observation(
                                scheduled,
                                stage,
                                prompt_plan,
                                permit,
                            )
                        )
                    expected_refs = tuple(
                        item.record_ref
                        for item in prompt_plan.history_plan.presentations
                    )
                    self.assertFalse(observation.proposal_admitted)
                    self.assertEqual(len(expected_refs), runner.RECALL_SLOTS)
                    self.assertEqual(
                        observation.recalled_record_refs,
                        expected_refs,
                    )
                    self.assertEqual(
                        captured_evidence,
                        [prompt_plan.history_plan.evidence_texts],
                    )
                    self.assertEqual(
                        budget.snapshot(),
                        {
                            "arm_tasks": 0,
                            "execution_attempts": 0,
                            "generation_attempts": 1,
                            "proposal_attempts": 1,
                        },
                    )
                    executor.execute.assert_not_called()
                    with self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "objective feedback runtime state differs",
                    ):
                        asyncio.run(
                            runtime.apply_objective_feedback(
                                observation,
                                {"score": 1.0},
                            )
                        )

                    if arm in runner.STATEFUL_ARMS:
                        assert session is not None and store is not None and cycle is not None
                        self.assertIsNone(cycle.pending_turn)
                        self.assertIsNone(cycle.executed_turn)
                        self.assertIsNone(store.active_prospective_turn())
                        dataset_id = session.dataset_id
                        scope_spec = session.scope_spec
                        asyncio.run(session.forget_close_and_remove())
                        self.assertFalse(Path(scope_spec.state_root).exists())
                        reopened = asyncio.run(
                            runner._open_v2_acquisition_session(
                                scope_spec,
                                opener=_OwnerAcquisitionOpener(scope_events),
                                expected_dataset_id=dataset_id,
                            )
                        )
                        asyncio.run(reopened.forget_close_and_remove())
                        self.assertEqual(
                            scope_events,
                            [
                                f"open:{arm}",
                                f"forget:{arm}",
                                f"close:{arm}",
                                f"open:{arm}",
                                f"forget:{arm}",
                                f"close:{arm}",
                            ],
                        )
                    else:
                        disposition_ref = _record_malformed_terminal_disposition(
                            ledger=ledger,
                            stage=stage,
                            prompt_plan=prompt_plan,
                            observation=observation,
                        )
                        cleanup = runtime.dispose_stateless_durable(
                            stage=stage,
                            terminal_disposition_ref=disposition_ref,
                            ledger=ledger,
                        )
                        registry.require_stateless(cleanup.terminal)
                        registry.mark_closed(cleanup.terminal.allocation_ref)
                        registry.assert_closed()
                        ledger.close()


class DonorHistoryAndPresentationTests(unittest.TestCase):
    def test_bounded_ordinary_recall_filters_only_proposed_batch_after_rejoin(
        self,
    ) -> None:
        source, backend, memory, hits, proposed_ref, resolution_ref = (
            _bounded_ordinary_recall_fixture()
        )
        selected_hits, batch = asyncio.run(
            runner._bounded_ordinary_recall(
                query="bounded ordinary recall query",
                store=source,
                backend=backend,
                memory=memory,
                frozen_origin=False,
                label="bounded ordinary recall test",
            )
        )
        expected = tuple(
            hit for hit in hits if hit.record_ref != proposed_ref
        )[: runner.RECALL_SLOTS]
        self.assertEqual(backend.search_limits, [len(source.items)])
        self.assertEqual(selected_hits, expected)
        self.assertEqual(batch.rejected, ())
        self.assertEqual(
            tuple(item.record_ref for item in batch.items),
            tuple(hit.record_ref for hit in expected),
        )
        self.assertNotIn(proposed_ref, {item.record_ref for item in batch.items})
        self.assertEqual(selected_hits[-1], hits[12])
        resolution = next(
            item for item in batch.items if item.record_ref == resolution_ref
        )
        self.assertEqual(
            resolution.source_contract, PROSPECTIVE_RESOLUTION_CONTRACT
        )
        self.assertEqual(resolution.epistemic_status, "OBSERVED")

    def test_bounded_ordinary_recall_rejects_unexplained_policy_failure(
        self,
    ) -> None:
        class ExtraRejectMemory(AcquisitionSituatedMemory):
            def recall_from_hits(
                self,
                hits: tuple[AcquisitionReferenceHit, ...],
                *,
                limit: int = 12,
                allowed_visibility: tuple[str, ...] = ("LEARNER_VISIBLE",),
                world_time: int | None = None,
                frozen_origin: bool = False,
            ) -> AcquisitionRecallBatch:
                batch = super().recall_from_hits(
                    hits,
                    limit=limit,
                    allowed_visibility=allowed_visibility,
                    world_time=world_time,
                    frozen_origin=frozen_origin,
                )
                return AcquisitionRecallBatch(
                    batch.items,
                    (*batch.rejected, "REFERENCE_UNKNOWN_OR_INVALID"),
                )

        source, backend, memory, _hits, _proposed, _resolution = (
            _bounded_ordinary_recall_fixture(memory_type=ExtraRejectMemory)
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "ordinary policy rejection differs",
        ):
            asyncio.run(
                runner._bounded_ordinary_recall(
                    query="bounded ordinary recall query",
                    store=source,
                    backend=backend,
                    memory=memory,
                    frozen_origin=True,
                    label="bounded unexplained rejection test",
                )
            )

    def test_bounded_ordinary_recall_validates_proposed_adjacency_before_policy(
        self,
    ) -> None:
        source, backend, memory, hits, proposed_ref, _resolution = (
            _bounded_ordinary_recall_fixture()
        )
        backend.hits = tuple(
            (
                AcquisitionReferenceHit(
                    record_ref=hit.record_ref,
                    adjacent_record_refs=(_ref("noncanonical-proposed-adjacent"),),
                    score=hit.score,
                    backend_ref=hit.backend_ref,
                )
                if hit.record_ref == proposed_ref
                else hit
            )
            for hit in hits
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "exact canonical adjacency differs",
        ):
            asyncio.run(
                runner._bounded_ordinary_recall(
                    query="bounded ordinary recall query",
                    store=source,
                    backend=backend,
                    memory=memory,
                    frozen_origin=False,
                    label="bounded proposed adjacency test",
                )
            )

    def test_bounded_ordinary_recall_requires_complete_unique_hit_population(
        self,
    ) -> None:
        for case in ("missing", "duplicate", "unknown"):
            with self.subTest(case=case):
                source, backend, memory, hits, _proposed, _resolution = (
                    _bounded_ordinary_recall_fixture()
                )
                if case == "missing":
                    backend.hits = hits[:-1]
                elif case == "duplicate":
                    backend.hits = (*hits[:-1], hits[1])
                else:
                    backend.hits = (
                        *hits[:-1],
                        replace(hits[-1], record_ref=_ref("unknown-record")),
                    )
                with self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "full reference-hit population differs",
                ):
                    asyncio.run(
                        runner._bounded_ordinary_recall(
                            query="bounded ordinary recall query",
                            store=source,
                            backend=backend,
                            memory=memory,
                            frozen_origin=False,
                            label=f"bounded {case} population test",
                        )
                    )

    def test_bounded_ordinary_recall_requires_twelve_eligible_memories(
        self,
    ) -> None:
        source, backend, memory, _hits, _proposed, _resolution = (
            _bounded_ordinary_recall_fixture(ordinary_record_count=10)
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "fewer than 12 eligible memories",
        ):
            asyncio.run(
                runner._bounded_ordinary_recall(
                    query="bounded ordinary recall query",
                    store=source,
                    backend=backend,
                    memory=memory,
                    frozen_origin=False,
                    label="bounded insufficient eligibility test",
                )
            )

    def test_bounded_acquisition_inventory_refuses_more_than_256_records(
        self,
    ) -> None:
        source, _backend, _memory, _hits, _proposed, _resolution = (
            _bounded_ordinary_recall_fixture()
        )
        terminal = source.acquisition_head()
        source.acquisition_head = lambda: AcquisitionHead(  # type: ignore[method-assign]
            257,
            terminal.acquisition_ref,
            terminal.record_ref,
        )
        with self.assertRaisesRegex(ValueError, "acquisition inventory head"):
            runner._bounded_acquisition_inventory(source)

    def test_bounded_ordinary_recall_requires_lower_is_better_score_order(
        self,
    ) -> None:
        source, backend, memory, hits, _proposed, _resolution = (
            _bounded_ordinary_recall_fixture()
        )
        first, second, *remaining = hits
        backend.hits = (
            replace(first, score=1.0),
            replace(second, score=0.0),
            *remaining,
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "score order differs",
        ):
            asyncio.run(
                runner._bounded_ordinary_recall(
                    query="bounded ordinary recall query",
                    store=source,
                    backend=backend,
                    memory=memory,
                    frozen_origin=False,
                    label="bounded score order test",
                )
            )

    def test_catalog_eligibility_temporal_strata_and_disjoint_six_plus_six(self) -> None:
        catalog, bindings, items = _catalog_bundle()
        self.assertEqual(len(catalog.entries), 36)
        self.assertEqual(
            catalog.excluded_counts,
            (("SOURCE_CONTRACT", 1),),
        )
        for family in runner.FAMILIES:
            pool = catalog.family_pool(family)
            early, late = runner.temporal_strata(pool)
            self.assertEqual((len(early), len(late)), (6, 6))
            self.assertLess(early[-1].acquired_ordinal, late[0].acquired_ordinal)

        low, high, cross = _ranked_controls(catalog)
        runner.validate_disjoint_same_family_selections(low.selected, high.selected)
        self.assertEqual({item.stratum for item in low.selected}, {"early", "late"})
        self.assertEqual({item.stratum for item in high.selected}, {"early", "late"})
        self.assertTrue(
            all(
                item.entry.family == "symbolic-demonstration-transfer"
                for item in (*low.selected, *high.selected)
            )
        )
        self.assertTrue(
            all(item.entry.family == "glyph-machine" for item in cross.selected)
        )

        history = runner.ranked_control_history_plan(
            low.task_id,
            "S0_SAME_FAMILY_LOW_6_NEUTRAL_6",
            low,
            _neutral_episodes(),
        )
        self.assertEqual(len(history.presentations), 12)
        self.assertTrue(all(not item.neutral for item in history.presentations[::2]))
        self.assertTrue(all(item.neutral for item in history.presentations[1::2]))
        self.assertEqual(
            tuple(item.record_ref for item in history.presentations[::2]),
            tuple(item.entry.record_ref for item in low.selected),
        )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "complete acquisition head",
        ):
            runner.build_donor_catalog(
                items[:-1],
                bindings,
                replicate_commitment=_REPLICATE_REF,
                expected_next_ordinal=len(items),
            )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "fewer than 12|exactly cover admitted IO resolutions",
        ):
            runner.build_donor_catalog(
                items,
                bindings[:-1],
                replicate_commitment=_REPLICATE_REF,
                expected_next_ordinal=len(items),
            )

    def test_presentation_proxy_delegates_everything_except_recall_text(self) -> None:
        catalog, _bindings, items = _catalog_bundle()
        records = {item.record_ref: item.acquisition.record for item in items}
        batch = _RecallBatch(
            tuple(
                _RecallItem(
                    record=records[entry.record_ref],
                    record_ref=entry.record_ref,
                    acquisition_ref=entry.acquisition_ref,
                    source_ref=entry.source_ref,
                    text=entry.public_record_text,
                    ordinal=entry.acquired_ordinal,
                    source_contract=entry.source_contract,
                    temporal_marker=f"ordinal:{entry.acquired_ordinal}",
                )
                for entry in catalog.entries[:12]
            ),
            ("sha256:" + "f" * 64,),
        )
        task = _selection_task()
        capture = _frozen_recall_capture(task, batch)
        plan = runner.recall_history_plan(
            task.task_id,
            "O0_ORDINARY_FROZEN_12",
            capture,
            neutral=False,
        )
        source = _PresentationMemory(batch)
        proxy = runner.AcquisitionMemoryPresentationProxy(source, plan)

        recalled = asyncio.run(proxy.recall("public query", limit=12))
        replayed = proxy.recall_from_hits(("frozen-hit",))
        self.assertEqual(proxy.presentation_calls, 2)
        self.assertEqual(recalled.rejected, batch.rejected)
        self.assertEqual(replayed.rejected, batch.rejected)
        self.assertEqual(proxy.backend, source.backend)
        self.assertEqual(proxy.delegated_marker(), "delegated-canonical-memory")
        self.assertEqual(
            tuple(item.text for item in recalled.items),
            plan.evidence_texts,
        )
        for canonical, presented in zip(batch.items, recalled.items, strict=True):
            self.assertIs(presented.canonical_item, canonical)
            self.assertIs(presented.record, canonical.record)
            self.assertEqual(presented.record_ref, canonical.record_ref)
            self.assertEqual(presented.acquisition_ref, canonical.acquisition_ref)
            self.assertEqual(presented.source_ref, canonical.source_ref)
            self.assertEqual(presented.ordinal, canonical.ordinal)
            self.assertEqual(presented.source_contract, canonical.source_contract)
            self.assertEqual(presented.temporal_marker, canonical.temporal_marker)
        self.assertEqual(source.calls[0][0], "recall")
        self.assertEqual(source.calls[1][0], "recall_from_hits")

        wrong_batch = _RecallBatch(tuple(reversed(batch.items)))
        wrong_source = _PresentationMemory(wrong_batch)
        wrong_proxy = runner.AcquisitionMemoryPresentationProxy(wrong_source, plan)
        with self.assertRaisesRegex(runner.V2InvariantError, "frozen presentation order"):
            asyncio.run(wrong_proxy.recall("public query", limit=12))

    def test_proxy_preserves_store_identity_for_real_successor_cycle_constructor(self) -> None:
        catalog, _bindings, items = _catalog_bundle()
        records = {item.record_ref: item.acquisition.record for item in items}
        batch = _RecallBatch(
            tuple(
                _RecallItem(
                    record=records[entry.record_ref],
                    record_ref=entry.record_ref,
                    acquisition_ref=entry.acquisition_ref,
                    source_ref=entry.source_ref,
                    text=entry.public_record_text,
                    ordinal=entry.acquired_ordinal,
                    source_contract=entry.source_contract,
                    temporal_marker=f"ordinal:{entry.acquired_ordinal}",
                )
                for entry in catalog.entries[:12]
            )
        )
        task = _selection_task()
        capture = _frozen_recall_capture(task, batch)
        plan = runner.recall_history_plan(
            task.task_id,
            "IO_FULL_ORDINARY_12",
            capture,
            neutral=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            observation_encoder = SyntheticObservedStateEncoderV1(1)
            store = CognitiveTransactionStore(Path(directory) / "cycle.sqlite3")
            store.initialize(
                _STATE_REF,
                _SNAPSHOT,
                model_ref=_MODEL_REF,
                encoder_ref=observation_encoder.encoder_ref,
            )
            memory = _PresentationMemory(batch, source=store)
            proxy = runner.AcquisitionMemoryPresentationProxy(memory, plan)
            self.assertIs(proxy.source, store)
            cycle = CognitiveCycle(
                text_adapter=_TextAdapter(),
                relation_adapter=_RelationAdapter(observation_encoder.encoder_ref),
                learner=_Learner(),
                memory=proxy,
                executor=_Executor(),
                transaction_store=store,
                model_ref=_MODEL_REF,
                encoder_ref=observation_encoder.encoder_ref,
                agent_ref=_ref("v2-cycle-agent"),
                world_ref=_ref("v2-cycle-world"),
                recall_limit=12,
                proposal_count=2,
                successor_mode=True,
                reality_mode="SIMULATED",
                subject_ref=_ref("v2-cycle-subject"),
                scope_ref=_ref("v2-cycle-scope"),
                bootstrap_evidence_refs=(_ref("v2-cycle-bootstrap"),),
                observation_encoder=observation_encoder,
            )
            self.assertIs(cycle.memory, proxy)
            self.assertIs(cycle.transaction_store, store)
            self.assertEqual(proxy.retry_pending_projections(12), 0)
            proxy.assert_synchronized()
            self.assertEqual(
                [call[0] for call in memory.calls[-2:]],
                ["retry_pending_projections", "assert_synchronized"],
            )


class PromptScheduleAndBudgetTests(unittest.TestCase):
    def test_stateful_runtime_embedding_rows_are_exact_v2_rows(self) -> None:
        self.assertEqual(runner.PROPOSAL_COUNT, 2)
        self.assertEqual(runner.RECALL_SLOTS, 12)
        self.assertEqual(
            runner.RUNTIME_EMBEDDING_ROWS_PER_STATEFUL_ATTEMPT,
            1 + runner.PROPOSAL_COUNT + runner.RECALL_SLOTS,
        )
        self.assertEqual(runner.RUNTIME_EMBEDDING_ROWS_PER_STATEFUL_ATTEMPT, 15)
        self.assertEqual(
            runner.MAXIMUM_QUALIFICATION_RUNTIME_EMBEDDING_ROWS,
            54 * 15,
        )
        self.assertEqual(
            runner.MAXIMUM_EVALUATION_RUNTIME_EMBEDDING_ROWS,
            432 * 15,
        )
        self.assertEqual(
            runner.ResourceLedger(
                runner.RunMode.QUALIFICATION
            ).runtime_embedding_ceiling,
            810,
        )
        self.assertEqual(
            runner.ResourceLedger(
                runner.RunMode.EVALUATION
            ).runtime_embedding_ceiling,
            6_480,
        )

    def test_paired_malformed_generations_close_only_on_exact_shared_identity(self) -> None:
        task_id = _ref("paired-malformed-task")
        public_task = _ref("paired-malformed-public-task")
        replicate_commitment = runner.qualification_replicate_commitment()

        def malformed(
            base: runner.AttemptFinal,
            generation_ref: str,
        ) -> runner.AttemptFinal:
            observation = replace(
                base.observation,
                proposal_admitted=False,
                proposals=(),
                selected_index=None,
                selected_trace=None,
                raw_response="",
                proposal_generation_ref=generation_ref,
                execution_request_ref=None,
                execution_receipt_ref=None,
                execution_prompt_ref=None,
                execution_prompt_tokens=None,
                execution_output_tokens=None,
                selected_trace_bound=None,
                response_bound=None,
                selected_grammar_class=None,
                response_grammar_class=None,
                resolution_ref=None,
                prospective_reservation_ref=None,
                prospective_batch_ref=None,
            )
            response_payload = {
                "arm": base.stage.arm,
                "attempt_receipt_ref": base.attempt_receipt_ref,
                "protocol_identity": runner.PROTOCOL_IDENTITY,
                "raw_response": "",
                "record_kind": "response",
                "task_id": task_id,
            }
            response_ref = runner.domain_ref(
                runner.EVALUATOR_RECORD_DOMAIN,
                response_payload,
            )
            judgment_without_ref = {
                "arm": base.stage.arm,
                "attempt_receipt_ref": base.attempt_receipt_ref,
                "disposition": "UNSUCCESSFUL",
                "protocol_identity": runner.PROTOCOL_IDENTITY,
                "proposal_admitted": False,
                "public_task_ref": public_task,
                "raw_response": "",
                "record_kind": "objective-judgment",
                "response_commitment": response_ref,
                "response_conforms": False,
                "score": 0.0,
                "success": False,
                "task_id": task_id,
            }
            objective_ref = runner.domain_ref(
                runner.EVALUATOR_RECORD_DOMAIN,
                judgment_without_ref,
            )
            judgment = {
                **judgment_without_ref,
                "evaluator_record_ref": objective_ref,
            }
            witness = runner.make_congruence_witness(
                task_id=task_id,
                arm=base.stage.arm,
                attempt_receipt_ref=base.attempt_receipt_ref,
                selected_trace_ref=None,
                execution_request_ref=None,
                execution_receipt_ref=None,
                response_ref=None,
                objective_judgment_ref=objective_ref,
                selected_trace_bound=None,
                response_bound=None,
                response_conforms=None,
                objective_success=None,
                selected_grammar_class=None,
                response_grammar_class=None,
            )
            return runner.AttemptFinal(
                stage=base.stage,
                attempt_receipt_ref=base.attempt_receipt_ref,
                terminal_disposition_ref=base.terminal_disposition_ref,
                observation=observation,
                judgment=judgment,
                witness=witness,
            )

        with tempfile.TemporaryDirectory() as directory:
            clone_parent = Path(directory)
            left_admitted = _attempt_final(
                task_id=task_id,
                public_task_ref=public_task,
                replicate="qualification-01",
                replicate_commitment=replicate_commitment,
                family="glyph-machine",
                arm="F0_NEUTRAL_FORMATTED_12",
                block_ordinal=0,
                success=False,
                purpose="qualification",
                phase="development",
            )
            right_admitted = _attempt_final(
                task_id=task_id,
                public_task_ref=public_task,
                replicate="qualification-01",
                replicate_commitment=replicate_commitment,
                family="glyph-machine",
                arm="IN_FULL_NEUTRAL_12",
                block_ordinal=0,
                success=False,
                purpose="qualification",
                phase="development",
                clone_parent=clone_parent,
            )
            shared_generation_ref = _ref("paired-malformed-generation")
            left = malformed(left_admitted, shared_generation_ref)
            right = malformed(right_admitted, shared_generation_ref)
            initial = runner.PairEqualityEvidence(
                left_arm="F0_NEUTRAL_FORMATTED_12",
                right_arm="IN_FULL_NEUTRAL_12",
                task_id=task_id,
                history_text_equal=True,
                prompt_bytes_equal=True,
                token_ids_equal=True,
                padding_atoms_equal=True,
            )
            closed = runner._paired_generation_equality(initial, left, right)
            self.assertTrue(closed.proposal_sets_equal)

            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "generation identity differs",
            ):
                runner._paired_generation_equality(
                    initial,
                    left,
                    malformed(right_admitted, _ref("different-generation")),
                )
            mixed_right = replace(
                right_admitted,
                observation=replace(
                    right_admitted.observation,
                    proposal_generation_ref=shared_generation_ref,
                ),
            )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "admission differs",
            ):
                runner._paired_generation_equality(initial, left, mixed_right)

    def test_tiny_tokenizer_reaches_exact_padding_and_binds_pair_equality(self) -> None:
        catalog, _bindings, items = _catalog_bundle()
        records = {item.record_ref: item.acquisition.record for item in items}
        low, high, cross = _ranked_controls(catalog)
        neutral = _neutral_episodes()
        task = _selection_task()
        task_id = task.task_id
        ordinary_batch = _RecallBatch(
            tuple(
                _RecallItem(
                    record=records[entry.record_ref],
                    record_ref=entry.record_ref,
                    acquisition_ref=entry.acquisition_ref,
                    source_ref=entry.source_ref,
                    text=entry.public_record_text,
                    ordinal=entry.acquired_ordinal,
                    source_contract=entry.source_contract,
                    temporal_marker=f"ordinal:{entry.acquired_ordinal}",
                )
                for entry in catalog.entries[:12]
            )
        )
        ordinary_capture = _frozen_recall_capture(task, ordinary_batch)
        histories = {
            "F0_NEUTRAL_FORMATTED_12": runner.neutral_history_plan(
                task_id, "F0_NEUTRAL_FORMATTED_12", neutral
            ),
            "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6": runner.ranked_control_history_plan(
                task_id,
                "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6",
                cross,
                neutral,
            ),
            "S0_SAME_FAMILY_LOW_6_NEUTRAL_6": runner.ranked_control_history_plan(
                task_id,
                "S0_SAME_FAMILY_LOW_6_NEUTRAL_6",
                low,
                neutral,
            ),
            "S1_SAME_FAMILY_HIGH_6_NEUTRAL_6": runner.ranked_control_history_plan(
                task_id,
                "S1_SAME_FAMILY_HIGH_6_NEUTRAL_6",
                high,
                neutral,
            ),
            "O0_ORDINARY_FROZEN_12": runner.recall_history_plan(
                task_id,
                "O0_ORDINARY_FROZEN_12",
                ordinary_capture,
                neutral=False,
            ),
            "IN_FULL_NEUTRAL_12": runner.neutral_history_plan(
                task_id, "IN_FULL_NEUTRAL_12", neutral
            ),
            "IO_FULL_ORDINARY_12": runner.recall_history_plan(
                task_id,
                "IO_FULL_ORDINARY_12",
                ordinary_capture,
                neutral=False,
            ),
        }
        prompt_plans, equalities = runner.plan_equalized_prompts(
            task_text="One bounded public synthetic task.",
            history_plans=histories,
            tokenize_chat_prompt=lambda prompt: tuple(range(len(prompt.split()))),
        )
        unpadded_lengths = {
            arm: len(
                runner.build_v1_proposal_prompt(
                    "One bounded public synthetic task.",
                    history.evidence_texts,
                ).split()
            )
            for arm, history in histories.items()
        }
        expected_target = max(unpadded_lengths.values()) + runner.PADDING_TARGET_MARGIN
        self.assertEqual(set(prompt_plans), set(runner.TWELVE_SLOT_ARMS))
        self.assertEqual(
            {len(item.token_ids) for item in prompt_plans.values()},
            {expected_target},
        )
        self.assertLessEqual(expected_target, runner.MAXIMUM_INPUT_TOKENS)
        self.assertTrue(all(item.padding_atoms >= 32 for item in prompt_plans.values()))
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "cannot be repadded",
        ):
            prompt_plans[
                "F0_NEUTRAL_FORMATTED_12"
            ].history_plan.presentations[-1].with_padding(1)
        pre_padded_histories = dict(histories)
        pre_padded = histories["F0_NEUTRAL_FORMATTED_12"]
        pre_padded_presentations = list(pre_padded.presentations)
        pre_padded_presentations[-1] = pre_padded_presentations[-1].with_padding(1)
        pre_padded_histories["F0_NEUTRAL_FORMATTED_12"] = runner.HistoryPlan(
            arm=pre_padded.arm,
            task_id=pre_padded.task_id,
            presentations=tuple(pre_padded_presentations),
        )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "task/arm matrix",
        ):
            runner.plan_equalized_prompts(
                task_text="One bounded public synthetic task.",
                history_plans=pre_padded_histories,
                tokenize_chat_prompt=lambda prompt: tuple(
                    range(len(prompt.split()))
                ),
            )
        for arm, plan in prompt_plans.items():
            self.assertEqual(plan.target_tokens, expected_target)
            self.assertTrue(
                all(
                    presentation.padding_atoms == 0
                    for presentation in plan.history_plan.presentations[:-1]
                )
            )
            self.assertEqual(
                plan.history_plan.presentations[-1].padding_atoms,
                plan.padding_atoms,
            )
            base = histories[arm]
            for count in range(plan.padding_atoms):
                presentations = list(base.presentations)
                presentations[-1] = presentations[-1].with_padding(count)
                candidate = runner.HistoryPlan(
                    arm=base.arm,
                    task_id=base.task_id,
                    presentations=tuple(presentations),
                    selection_rows=base.selection_rows,
                    selection_plan=base.selection_plan,
                )
                candidate_prompt = runner.build_v1_proposal_prompt(
                    "One bounded public synthetic task.",
                    candidate.evidence_texts,
                )
                self.assertNotEqual(len(candidate_prompt.split()), expected_target)
        self.assertTrue(
            all(
                item.history_text_equal
                and item.prompt_bytes_equal
                and item.token_ids_equal
                and item.padding_atoms_equal
                for item in equalities
            )
        )
        for left, right in (
            ("F0_NEUTRAL_FORMATTED_12", "IN_FULL_NEUTRAL_12"),
            ("O0_ORDINARY_FROZEN_12", "IO_FULL_ORDINARY_12"),
        ):
            self.assertEqual(prompt_plans[left].prompt_ref, prompt_plans[right].prompt_ref)
            self.assertEqual(prompt_plans[left].token_ids, prompt_plans[right].token_ids)
        for equality in equalities:
            bound = runner.bind_paired_proposal_sets(
                equality,
                ("proposal-a", "proposal-b"),
                ("proposal-a", "proposal-b"),
            )
            self.assertTrue(bound.proposal_sets_equal)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "greedy proposal sets differ",
            ):
                runner.bind_paired_proposal_sets(
                    equality,
                    ("proposal-a", "proposal-b"),
                    ("proposal-a", "proposal-c"),
                )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "cannot reach the exact target",
        ):
            runner.plan_equalized_prompts(
                task_text="One bounded public synthetic task.",
                history_plans=histories,
                tokenize_chat_prompt=lambda _prompt: (0,),
            )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "target exceeds the input-token ceiling",
        ):
            runner.plan_equalized_prompts(
                task_text="One bounded public synthetic task.",
                history_plans=histories,
                tokenize_chat_prompt=lambda _prompt: tuple(
                    range(runner.MAXIMUM_INPUT_TOKENS - 16)
                ),
            )

        tokenizer = _TinyTokenizer()
        native = runner.native_prompt_plan(
            task_id=task_id,
            task_text="One bounded public synthetic task.",
            tokenize_chat_prompt=lambda prompt: tuple(range(len(prompt.split()))),
        )
        self.assertEqual(native.arm, "N0_NATIVE_NO_HISTORY")
        self.assertEqual(native.history_plan.presentations, ())
        self.assertEqual(native.history_plan.evidence_texts, ())
        self.assertEqual(native.padding_atoms, 0)
        self.assertIn(runner.NO_PERSISTENT_RECALLED_EVIDENCE_V1, native.prompt)
        self.assertEqual(
            native.to_canonical()["native_evidence_sha256"],
            hashlib.sha256(
                runner.NO_PERSISTENT_RECALLED_EVIDENCE_V1.encode("utf-8")
            ).hexdigest(),
        )
        token_ids = runner.frozen_qwen_chat_token_ids(
            tokenizer,
            prompt_plans["F0_NEUTRAL_FORMATTED_12"].prompt,
        )
        self.assertEqual(token_ids, tuple(range(1, len(token_ids) + 1)))
        self.assertEqual(
            tokenizer.template_calls[0]["enable_thinking"],
            False,
        )
        self.assertEqual(
            tokenizer.template_calls[0]["add_generation_prompt"],
            True,
        )

    def test_latin_schedules_and_one_use_attempt_budgets_close_exactly(self) -> None:
        qualification = evaluator.make_qualification_evaluator()
        evaluation_suite = evaluator.make_evaluation_evaluator(
            _EVALUATION_SEEDS,
            _EVALUATION_COMMITMENTS,
            _ADMISSION_REF,
        )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "reuses the qualification commitment",
        ):
            runner.validate_evaluation_seed_commitments(
                (
                    runner.qualification_replicate_commitment(),
                    _EVALUATION_COMMITMENTS[1],
                    _EVALUATION_COMMITMENTS[2],
                )
            )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "already consumed",
        ):
            runner.validate_evaluation_seed_commitments(
                (
                    sorted(runner.CONSUMED_SEED_COMMITMENTS)[0],
                    _EVALUATION_COMMITMENTS[1],
                    _EVALUATION_COMMITMENTS[2],
                )
            )
        self.assertIn(
            runner.CONSUMED_R3_QUALIFICATION_REPLICATE_COMMITMENT,
            runner.CONSUMED_SEED_COMMITMENTS,
        )
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "already consumed",
        ):
            runner.validate_evaluation_seed_commitments(
                (
                    runner.CONSUMED_R3_QUALIFICATION_REPLICATE_COMMITMENT,
                    _EVALUATION_COMMITMENTS[1],
                    _EVALUATION_COMMITMENTS[2],
                )
            )
        qualification_adaptation_tasks = qualification.release_phase("adaptation")
        untouched_budget = runner.AttemptBudget(runner.RunMode.QUALIFICATION)
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "schedule phase/family ordinal inventory differs",
        ):
            runner.adaptation_schedule(
                qualification_adaptation_tasks[:-1],
                purpose="qualification",
            )
        self.assertEqual(
            untouched_budget.snapshot(),
            {
                "arm_tasks": 0,
                "execution_attempts": 0,
                "generation_attempts": 0,
                "proposal_attempts": 0,
            },
        )
        q_adaptation = runner.adaptation_schedule(
            qualification_adaptation_tasks,
            purpose="qualification",
        )
        _finalize_evaluator_phase(qualification, qualification_adaptation_tasks)
        qualification_development_tasks = qualification.release_phase("development")
        q_development = runner.development_final_schedule(
            qualification_development_tasks,
            purpose="qualification",
            phase="development",
        )
        evaluation_adaptation_tasks = evaluation_suite.release_phase("adaptation")
        e_adaptation = runner.adaptation_schedule(
            evaluation_adaptation_tasks,
            purpose="evaluation",
        )
        _finalize_evaluator_phase(evaluation_suite, evaluation_adaptation_tasks)
        evaluation_development_tasks = evaluation_suite.release_phase("development")
        e_development = runner.development_final_schedule(
            evaluation_development_tasks,
            purpose="evaluation",
            phase="development",
        )
        _finalize_evaluator_phase(evaluation_suite, evaluation_development_tasks)
        evaluation_suite.admit_final(_ADMISSION_REF)
        evaluation_final_tasks = evaluation_suite.release_phase("final")
        e_final = runner.development_final_schedule(
            evaluation_final_tasks,
            purpose="evaluation",
            phase="final",
        )
        self.assertEqual(
            tuple(map(len, (q_adaptation, q_development))),
            (48, 24),
        )
        self.assertEqual(
            tuple(map(len, (e_adaptation, e_development, e_final))),
            (270, 216, 432),
        )
        self.assertEqual(
            Counter(item.arm for item in q_adaptation),
            {"IO_FULL_ORDINARY_12": 45, "IN_FULL_NEUTRAL_12": 3},
        )
        qualification_in = tuple(
            item for item in q_adaptation if item.arm == "IN_FULL_NEUTRAL_12"
        )
        self.assertEqual(
            {(item.family, int(item.task["ordinal"])) for item in qualification_in},
            {(family, 0) for family in runner.FAMILIES},
        )
        for block_ordinal in range(3):
            pair = tuple(
                item
                for item in q_adaptation
                if item.block_ordinal == block_ordinal
            )
            expected = (
                ("IN_FULL_NEUTRAL_12", "IO_FULL_ORDINARY_12")
                if block_ordinal % 2 == 0
                else ("IO_FULL_ORDINARY_12", "IN_FULL_NEUTRAL_12")
            )
            self.assertEqual(tuple(item.arm for item in pair), expected)
        self.assertEqual(
            sorted({item.block_ordinal for item in q_adaptation}),
            list(range(45)),
        )
        self.assertEqual(
            tuple(e_adaptation[index].block_ordinal for index in range(0, 270, 2)),
            tuple(range(135)),
        )
        for index in range(0, len(e_adaptation), 2):
            pair = e_adaptation[index : index + 2]
            first = pair[0]
            replicate_index = int(first.task["replicate"].split("-")[1]) - 1
            expected = (
                ("IN_FULL_NEUTRAL_12", "IO_FULL_ORDINARY_12")
                if (first.block_ordinal + replicate_index) % 2 == 0
                else ("IO_FULL_ORDINARY_12", "IN_FULL_NEUTRAL_12")
            )
            self.assertEqual(tuple(item.arm for item in pair), expected)

        for phase, schedule, phase_offset in (
            ("development", e_development, 0),
            ("final", e_final, 1),
        ):
            groups = [schedule[index : index + 8] for index in range(0, len(schedule), 8)]
            self.assertEqual(
                tuple(group[0].block_ordinal for group in groups),
                tuple(range(len(groups))),
            )
            for group in groups:
                first = group[0]
                replicate_index = int(first.task["replicate"].split("-")[1]) - 1
                offset = (
                    first.block_ordinal + 3 * replicate_index + 5 * phase_offset
                ) % 8
                self.assertEqual({item.task_id for item in group}, {first.task_id})
                self.assertEqual(
                    tuple(item.arm for item in group),
                    tuple(runner.ARMS[(offset + index) % 8] for index in range(8)),
                )
                self.assertEqual(
                    tuple(item.arm_position for item in group),
                    tuple(range(8)),
                )
                self.assertTrue(all(item.phase == phase for item in group))

        budget = runner.AttemptBudget(runner.RunMode.QUALIFICATION)
        for index, attempt in enumerate((*q_adaptation, *q_development)):
            budget.consume_proposal(attempt.task_id, attempt.arm)
            if index % 2 == 0:
                budget.consume_execution(attempt.task_id, attempt.arm)
            budget.finish(attempt.task_id, attempt.arm)
        self.assertEqual(
            budget.snapshot(),
            {
                "arm_tasks": 72,
                "execution_attempts": 36,
                "generation_attempts": 108,
                "proposal_attempts": 72,
            },
        )
        with self.assertRaisesRegex(runner.V2IntegrityStop, "retry is forbidden"):
            budget.consume_proposal(q_adaptation[0].task_id, q_adaptation[0].arm)
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "missing a unique proposal boundary",
        ):
            budget.consume_execution(q_adaptation[0].task_id, q_adaptation[0].arm)
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "finalization is missing or duplicated",
        ):
            budget.finish(q_adaptation[0].task_id, q_adaptation[0].arm)

        all_admitted_qualification = runner.AttemptBudget(runner.RunMode.QUALIFICATION)
        for attempt in (*q_adaptation, *q_development):
            all_admitted_qualification.consume_proposal(attempt.task_id, attempt.arm)
            all_admitted_qualification.consume_execution(attempt.task_id, attempt.arm)
            all_admitted_qualification.finish(attempt.task_id, attempt.arm)
        self.assertEqual(
            all_admitted_qualification.snapshot(),
            {
                "arm_tasks": 72,
                "execution_attempts": 72,
                "generation_attempts": 144,
                "proposal_attempts": 72,
            },
        )

        all_admitted_evaluation = runner.AttemptBudget(runner.RunMode.EVALUATION)
        for attempt in (*e_adaptation, *e_development, *e_final):
            all_admitted_evaluation.consume_proposal(attempt.task_id, attempt.arm)
            all_admitted_evaluation.consume_execution(attempt.task_id, attempt.arm)
            all_admitted_evaluation.finish(attempt.task_id, attempt.arm)
        self.assertEqual(
            all_admitted_evaluation.snapshot(),
            {
                "arm_tasks": 918,
                "execution_attempts": 918,
                "generation_attempts": 1_836,
                "proposal_attempts": 918,
            },
        )

        fresh = runner.AttemptBudget(runner.RunMode.EVALUATION)
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "missing a unique proposal boundary",
        ):
            fresh.consume_execution(e_adaptation[0].task_id, e_adaptation[0].arm)
        self.assertEqual(
            (
                fresh.proposal_ceiling,
                fresh.execution_ceiling,
                fresh.arm_task_ceiling,
                fresh.generation_ceiling,
            ),
            (918, 918, 918, 2_048),
        )


class CongruenceBaselineAndAccountingTests(unittest.TestCase):
    def test_successor_output_bound_accepts_160_and_rejects_161(self) -> None:
        resources = runner.ResourceLedger(runner.RunMode.QUALIFICATION)
        resources.record_generation(1, runner.MAXIMUM_OUTPUT_TOKENS)
        self.assertEqual(resources.output_tokens, 160)
        with self.assertRaisesRegex(ValueError, "output tokens"):
            runner.ResourceLedger(
                runner.RunMode.QUALIFICATION
            ).record_generation(1, 161)

        common = {
            "task_id": _ref("v2-output-bound-task"),
            "arm": "N0_NATIVE_NO_HISTORY",
            "proposal_admitted": False,
            "proposals": (),
            "selected_index": None,
            "selected_trace": None,
            "raw_response": "",
            "proposal_generation_ref": _ref("v2-output-bound-generation"),
            "proposal_prompt_ref": _ref("v2-output-bound-prompt"),
            "proposal_prompt_tokens": 1,
            "proposal_output_tokens": 160,
            "execution_request_ref": None,
            "execution_receipt_ref": None,
            "execution_prompt_ref": None,
            "execution_prompt_tokens": None,
            "execution_output_tokens": None,
            "selected_trace_bound": None,
            "response_bound": None,
            "selected_grammar_class": None,
            "response_grammar_class": None,
        }
        self.assertEqual(
            runner.AttemptObservation(**common).proposal_output_tokens,
            160,
        )
        with self.assertRaisesRegex(ValueError, "proposal output tokens"):
            runner.AttemptObservation(
                **{**common, "proposal_output_tokens": 161}
            )

    def test_congruence_is_shadow_only_and_binding_mismatch_is_an_integrity_stop(self) -> None:
        common = {
            "task_id": _ref("v2-congruence-task"),
            "arm": "IO_FULL_ORDINARY_12",
            "attempt_receipt_ref": _ref("v2-congruence-attempt"),
            "selected_grammar_class": "PUBLIC_SYNTHETIC",
            "response_grammar_class": "PUBLIC_SYNTHETIC",
        }
        unknown = runner.make_congruence_witness(
            **common,
            selected_trace_ref=None,
            execution_request_ref=None,
            execution_receipt_ref=None,
            response_ref=None,
            objective_judgment_ref=None,
            selected_trace_bound=None,
            response_bound=None,
            response_conforms=None,
            objective_success=None,
        )
        self.assertEqual(unknown.disposition, "UNKNOWN")
        unknown.assert_integrity()

        refs = {
            "selected_trace_ref": _ref("v2-congruence-trace"),
            "execution_request_ref": _ref("v2-congruence-request"),
            "execution_receipt_ref": _ref("v2-congruence-receipt"),
            "response_ref": _ref("v2-congruence-response"),
            "objective_judgment_ref": _ref("v2-congruence-objective"),
        }
        satisfied = runner.make_congruence_witness(
            **common,
            **refs,
            selected_trace_bound=True,
            response_bound=True,
            response_conforms=True,
            objective_success=True,
        )
        self.assertEqual(satisfied.disposition, "SATISFIED")
        satisfied.assert_integrity()

        outcome_failure = runner.make_congruence_witness(
            **common,
            **refs,
            selected_trace_bound=True,
            response_bound=True,
            response_conforms=True,
            objective_success=False,
        )
        self.assertEqual(outcome_failure.disposition, "VIOLATED")
        outcome_failure.assert_integrity()

        conformance_failure = runner.make_congruence_witness(
            **common,
            **refs,
            selected_trace_bound=True,
            response_bound=True,
            response_conforms=False,
            objective_success=False,
        )
        self.assertEqual(conformance_failure.disposition, "VIOLATED")
        self.assertFalse(conformance_failure.binding_mismatch)
        conformance_failure.assert_integrity()

        response_mismatch = runner.make_congruence_witness(
            **common,
            **refs,
            selected_trace_bound=True,
            response_bound=False,
            response_conforms=True,
            objective_success=True,
        )
        self.assertEqual(response_mismatch.disposition, "VIOLATED")
        self.assertTrue(response_mismatch.binding_mismatch)
        with self.assertRaisesRegex(runner.V2IntegrityStop, "binding mismatch"):
            response_mismatch.assert_integrity()

        mismatch = runner.make_congruence_witness(
            **common,
            **refs,
            selected_trace_bound=False,
            response_bound=True,
            response_conforms=True,
            objective_success=True,
        )
        self.assertEqual(mismatch.disposition, "VIOLATED")
        self.assertTrue(mismatch.binding_mismatch)
        with self.assertRaisesRegex(runner.V2IntegrityStop, "binding mismatch"):
            mismatch.assert_integrity()
        self.assertNotIn("repair", json.dumps(mismatch.to_canonical()).lower())
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "value types/disposition",
        ):
            _attempt_final(
                task_id=_ref("v2-invalid-success-task"),
                public_task_ref=_ref("v2-invalid-success-public"),
                replicate="replicate-01",
                replicate_commitment=_EVALUATION_COMMITMENTS[0],
                family="glyph-machine",
                arm="F0_NEUTRAL_FORMATTED_12",
                block_ordinal=0,
                success=True,
                response_conforms=False,
            )

    def test_genesis_baseline_and_disposable_clone_evidence_fail_closed(self) -> None:
        hashes = {
            "acquisition_sequence": _raw("v2-genesis-acquisition-sequence"),
            "journal": _raw("v2-genesis-journal"),
            "learner": _raw("v2-genesis-learner"),
            "store": _raw("v2-genesis-store"),
        }
        bootstrap = _bootstrap_episode()
        neutral_refs = tuple(item.episode_ref for item in _neutral_episodes())
        chain = runner.validate_neutral_fixture_chain(
            _neutral_episodes(),
            bootstrap_episode_ref=bootstrap.episode_ref,
            competence_state_digest=_STATE_REF,
            model_ref=_MODEL_REF,
            encoder_ref=_ENCODER_REF,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            in_root = root / "in"
            io_root = root / "io"
            clone_root = root / "clone"
            mutated_clone_root = root / "mutated-clone"
            mismatched_clone_root = root / "mismatched-clone"
            nonquiescent_clone_root = root / "nonquiescent-clone"
            for path in (
                in_root,
                io_root,
                clone_root,
                mutated_clone_root,
                mismatched_clone_root,
                nonquiescent_clone_root,
            ):
                path.mkdir()

            self.assertEqual(
                runner.validate_genesis_pair(
                    hashes,
                    dict(reversed(tuple(hashes.items()))),
                    in_root=in_root,
                    io_root=io_root,
                ),
                hashes,
            )
            with self.assertRaisesRegex(runner.V2InvariantError, "genesis differs"):
                runner.validate_genesis_pair(
                    hashes,
                    {**hashes, "store": _raw("changed")},
                    in_root=in_root,
                    io_root=io_root,
                )
            with self.assertRaisesRegex(ValueError, "exact baseline hash inventory"):
                runner.validate_genesis_pair(
                    {name: value for name, value in hashes.items() if name != "journal"},
                    hashes,
                    in_root=in_root,
                    io_root=io_root,
                )
            with self.assertRaisesRegex(runner.V2InvariantError, "roots alias"):
                runner.validate_genesis_pair(
                    hashes,
                    hashes,
                    in_root=in_root,
                    io_root=in_root,
                )
            symlink_root = root / "in-alias"
            symlink_root.symlink_to(in_root, target_is_directory=True)
            with self.assertRaisesRegex(runner.V2InvariantError, "owned real directory"):
                runner.validate_genesis_pair(
                    hashes,
                    hashes,
                    in_root=symlink_root,
                    io_root=io_root,
                )

            seal = runner.BaselineSeal(
                purpose="evaluation",
                replicate_commitment=_REPLICATE_REF,
                arm="IO_FULL_ORDINARY_12",
                root=str(io_root),
                hashes=tuple(sorted(hashes.items())),
                neutral_episode_refs=neutral_refs,
                neutral_chain_ref=str(chain["chain_ref"]),
                bootstrap_episode_ref=bootstrap.episode_ref,
                genesis_competence_digest=_STATE_REF,
                model_ref=_MODEL_REF,
                encoder_ref=_ENCODER_REF,
                quiescent=True,
            )
            self.assertTrue(seal.seal_ref.startswith("sha256:"))
            terminal_hashes = {**hashes, "journal": _raw("v2-terminal-journal")}
            disposition_ref = _ref("v2-clone-disposition")
            terminal = runner.CloneTerminalEvidence(
                task_id=_ref("v2-clone-task"),
                arm="IO_FULL_ORDINARY_12",
                source_root=str(io_root),
                clone_root=str(clone_root),
                source_before=tuple(sorted(hashes.items())),
                source_after=tuple(sorted(hashes.items())),
                clone_before=tuple(sorted(hashes.items())),
                clone_terminal=tuple(sorted(terminal_hashes.items())),
                quiescent=True,
            )
            lifecycle = runner.CloneLifecycleEvidence(
                terminal=terminal,
                terminal_disposition_ref=disposition_ref,
                disposed=True,
                cleanup_absence_verified=True,
            )
            self.assertTrue(lifecycle.lifecycle_ref.startswith("sha256:"))
            with self.assertRaisesRegex(runner.V2IntegrityStop, "source baseline changed"):
                runner.CloneTerminalEvidence(
                    task_id=_ref("v2-clone-mutated-task"),
                    arm="IO_FULL_ORDINARY_12",
                    source_root=str(io_root),
                    clone_root=str(mutated_clone_root),
                    source_before=tuple(sorted(hashes.items())),
                    source_after=tuple(sorted(terminal_hashes.items())),
                    clone_before=tuple(sorted(hashes.items())),
                    clone_terminal=tuple(sorted(terminal_hashes.items())),
                    quiescent=True,
                )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "does not match frozen baseline",
            ):
                runner.CloneTerminalEvidence(
                    task_id=_ref("v2-clone-mismatched-task"),
                    arm="IO_FULL_ORDINARY_12",
                    source_root=str(io_root),
                    clone_root=str(mismatched_clone_root),
                    source_before=tuple(sorted(hashes.items())),
                    source_after=tuple(sorted(hashes.items())),
                    clone_before=tuple(sorted(terminal_hashes.items())),
                    clone_terminal=tuple(sorted(terminal_hashes.items())),
                    quiescent=True,
                )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "not quiescent",
            ):
                runner.CloneTerminalEvidence(
                    task_id=_ref("v2-clone-nonquiescent-task"),
                    arm="IO_FULL_ORDINARY_12",
                    source_root=str(io_root),
                    clone_root=str(nonquiescent_clone_root),
                    source_before=tuple(sorted(hashes.items())),
                    source_after=tuple(sorted(hashes.items())),
                    clone_before=tuple(sorted(hashes.items())),
                    clone_terminal=tuple(sorted(terminal_hashes.items())),
                    quiescent=False,
                )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "cleanup was not verified",
            ):
                runner.CloneLifecycleEvidence(
                    terminal=terminal,
                    terminal_disposition_ref=disposition_ref,
                    disposed=False,
                    cleanup_absence_verified=False,
                )

    def test_finite_cohort_is_signed_descriptive_and_uses_all_54_pairs(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        cohort = _finite_cohort(Path(temporary.name))
        self.assertEqual(len(cohort), 432)
        statistics = runner.finite_cohort_statistics(cohort)
        self.assertEqual(statistics["schema"], runner.FINITE_COHORT_SCHEMA)
        self.assertEqual(statistics["denominator"], 54)
        self.assertEqual(statistics["primary_contrast_count"], 7)
        self.assertIn("descriptive", statistics["classification_boundary"])
        self.assertIn("no support/rule-out/promotion", statistics["classification_boundary"])
        contrasts = {
            (item["left_arm"], item["right_arm"]): item
            for item in statistics["contrasts"]
        }
        self.assertEqual(set(contrasts), set(runner.PRIMARY_CONTRASTS))
        self.assertTrue(all(item["denominator"] == 54 for item in contrasts.values()))
        cells = {
            (item.stage.task_id, item.stage.arm): item
            for item in cohort
        }
        task_ids = tuple(sorted({item.stage.task_id for item in cohort}))
        task_metadata = {
            item.stage.task_id: (item.stage.family, item.stage.replicate)
            for item in cohort
        }
        self.assertEqual(len(task_ids), 54)
        for left_arm, right_arm in runner.PRIMARY_CONTRASTS:
            row = contrasts[(left_arm, right_arm)]
            outcomes = tuple(
                (
                    cells[(task_id, left_arm)].success,
                    cells[(task_id, right_arm)].success,
                )
                for task_id in task_ids
            )
            left_failure_right_success = sum(
                not left and right for left, right in outcomes
            )
            left_success_right_failure = sum(
                left and not right for left, right in outcomes
            )
            tied_failure = sum(not left and not right for left, right in outcomes)
            tied_success = sum(left and right for left, right in outcomes)
            self.assertEqual(
                (
                    row["left_failure_right_success"],
                    row["left_success_right_failure"],
                    row["tied_failure"],
                    row["tied_success"],
                    row["effect_numerator"],
                ),
                (
                    left_failure_right_success,
                    left_success_right_failure,
                    tied_failure,
                    tied_success,
                    left_failure_right_success - left_success_right_failure,
                ),
            )
            for stratum_key, metadata_index, names in (
                ("family_strata", 0, runner.FAMILIES),
                (
                    "replicate_strata",
                    1,
                    ("replicate-01", "replicate-02", "replicate-03"),
                ),
            ):
                strata = {item["name"]: item for item in row[stratum_key]}
                self.assertEqual(set(strata), set(names))
                for name in names:
                    selected = tuple(
                        task_id
                        for task_id in task_ids
                        if task_metadata[task_id][metadata_index] == name
                    )
                    self.assertEqual(len(selected), 18)
                    self.assertEqual(strata[name]["denominator"], 18)
                    self.assertEqual(
                        strata[name]["numerator"],
                        sum(
                            int(cells[(task_id, right_arm)].success)
                            - int(cells[(task_id, left_arm)].success)
                            for task_id in selected
                        ),
                    )

        first = contrasts[
            ("N0_NATIVE_NO_HISTORY", "F0_NEUTRAL_FORMATTED_12")
        ]
        self.assertEqual(first["effect_numerator"], 18)
        self.assertEqual(first["direction"], "OBSERVED_RIGHT_HIGHER")
        self.assertTrue(first["recurrent_in_two_families"])
        self.assertTrue(first["recurrent_in_two_replicates"])
        self.assertIn("family:causal-operator", first["reversals"])
        self.assertTrue(
            0.0 <= float.fromhex(first["exact_mcnemar_two_sided_hex"]) <= 1.0
        )
        self.assertTrue(0.0 <= float.fromhex(first["holm_adjusted_hex"]) <= 1.0)

        equal = contrasts[
            (
                "F0_NEUTRAL_FORMATTED_12",
                "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6",
            )
        ]
        self.assertEqual(equal["effect_numerator"], 0)
        self.assertEqual(equal["direction"], "OBSERVED_EQUAL")
        self.assertFalse(equal["recurrent_in_two_families"])
        self.assertFalse(equal["recurrent_in_two_replicates"])
        self.assertEqual(float.fromhex(equal["exact_mcnemar_two_sided_hex"]), 1.0)

        interaction = statistics["history_runtime_interaction"]
        self.assertEqual(interaction["classification"], "SECONDARY_DESCRIPTIVE_ONLY")
        self.assertEqual(interaction["formula"], "(IO-O0)-(IN-F0)")
        self.assertEqual(interaction["effect_numerator"], -36)
        self.assertEqual(interaction["direction"], "OBSERVED_LEFT_HIGHER")
        statistics_payload = {
            key: value
            for key, value in statistics.items()
            if key != "statistics_ref"
        }
        self.assertEqual(
            statistics["statistics_ref"],
            runner.canonical_ref("finite-cohort", statistics_payload),
        )

        cohort_rows = statistics["cohort"]
        self.assertEqual(len(cohort_rows), 54)
        self.assertEqual(
            [row["task_id"] for row in cohort_rows],
            sorted(row["task_id"] for row in cohort_rows),
        )
        self.assertTrue(
            all(
                set(row["successes"]) == set(runner.ARMS)
                and all(type(value) is bool for value in row["successes"].values())
                for row in cohort_rows
            )
        )

        def cohort_anchors(
            rows: list[dict[str, object]],
        ) -> tuple[
            dict[str, int],
            dict[tuple[str, str], int],
            str,
            dict[str, tuple[str, str, str]],
        ]:
            arm_successes = {
                arm: sum(bool(row["successes"][arm]) for row in rows)
                for arm in runner.ARMS
            }
            family_arm_successes = {
                (family, arm): sum(
                    bool(row["successes"][arm])
                    for row in rows
                    if row["family"] == family
                )
                for family in runner.FAMILIES
                for arm in runner.ARMS
            }
            outcomes_sha256 = hashlib.sha256(
                runner.canonical_json_bytes(
                    sorted(
                        (
                            {
                                "arm": arm,
                                "success": row["successes"][arm],
                                "task_id": row["task_id"],
                            }
                            for row in rows
                            for arm in runner.ARMS
                        ),
                        key=lambda item: (item["task_id"], item["arm"]),
                    )
                )
            ).hexdigest()
            metadata = {
                row["task_id"]: (
                    row["family"],
                    row["replicate"],
                    row["replicate_commitment"],
                )
                for row in rows
            }
            return arm_successes, family_arm_successes, outcomes_sha256, metadata

        original_anchors = cohort_anchors(cohort_rows)
        ledger_bindings = {
            (final.stage.task_id, final.stage.arm): (
                runner._ledger_attempt_binding_from_final(final)
            )
            for final in cohort
        }
        self.assertEqual(
            runner._validate_serialized_statistics(
                statistics,
                final_arm_successes=original_anchors[0],
                final_family_arm_successes=original_anchors[1],
                final_outcomes_sha256=original_anchors[2],
                final_task_metadata=original_anchors[3],
                ledger_bindings=ledger_bindings,
            ),
            statistics,
        )
        rotated = json.loads(json.dumps(statistics))
        first = next(
            index
            for index, row in enumerate(rotated["cohort"])
            if row["replicate"] == "replicate-01"
            and row["family"] == "symbolic-demonstration-transfer"
        )
        second = next(
            index
            for index, row in enumerate(rotated["cohort"])
            if row["replicate"] == "replicate-01"
            and row["family"] == "causal-operator"
        )
        rotated["cohort"][first]["successes"], rotated["cohort"][second][
            "successes"
        ] = (
            rotated["cohort"][second]["successes"],
            rotated["cohort"][first]["successes"],
        )
        rotated_payload = {
            key: value for key, value in rotated.items() if key != "statistics_ref"
        }
        rotated["statistics_ref"] = runner.canonical_ref(
            "finite-cohort",
            rotated_payload,
        )
        rotated_anchors = cohort_anchors(rotated["cohort"])
        with self.assertRaises(runner.V2InvariantError):
            runner._validate_serialized_statistics(
                rotated,
                final_arm_successes=rotated_anchors[0],
                final_family_arm_successes=rotated_anchors[1],
                final_outcomes_sha256=rotated_anchors[2],
                final_task_metadata=rotated_anchors[3],
                ledger_bindings=ledger_bindings,
            )

        with self.assertRaisesRegex(runner.V2InvariantError, "exactly 432"):
            runner.finite_cohort_statistics(cohort[:-1])
        duplicate_cell_cohort = (*cohort[:-1], cohort[0])
        self.assertEqual(len(duplicate_cell_cohort), 432)
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "repeats a task/arm cell",
        ):
            runner.finite_cohort_statistics(duplicate_cell_cohort)

    def test_malformed_observation_cannot_acquire_execution_or_repair_material(self) -> None:
        task_id = _ref("v2-malformed-task")
        arm = "IO_FULL_ORDINARY_12"
        common = {
            "task_id": task_id,
            "arm": arm,
            "proposal_admitted": False,
            "proposals": (),
            "selected_index": None,
            "selected_trace": None,
            "raw_response": "",
            "proposal_generation_ref": _ref("v2-malformed-generation"),
            "proposal_prompt_ref": _ref("v2-malformed-prompt"),
            "proposal_prompt_tokens": 64,
            "proposal_output_tokens": 1,
            "execution_request_ref": None,
            "execution_receipt_ref": None,
            "execution_prompt_ref": None,
            "execution_prompt_tokens": None,
            "execution_output_tokens": None,
            "selected_trace_bound": None,
            "response_bound": None,
            "selected_grammar_class": None,
            "response_grammar_class": None,
        }
        observation = runner.AttemptObservation(**common)
        self.assertFalse(observation.proposal_admitted)
        self.assertEqual(observation.proposals, ())
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "malformed proposal acquired execution material",
        ):
            runner.AttemptObservation(
                **{
                    **common,
                    "execution_request_ref": _ref("forbidden-repair-request"),
                }
            )


class _MirrorFakeDtype:
    def __init__(self, name: str, itemsize: int) -> None:
        self.name = name
        self.itemsize = itemsize

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class _MirrorFakeDevice:
    type: str
    index: int | None


class _MirrorFakeParameter:
    def __init__(
        self,
        torch_module: "_MirrorFakeTorch",
        name: str,
        raw: bytes,
        *,
        shape: tuple[int, ...],
        dtype: _MirrorFakeDtype,
    ) -> None:
        self._torch_module = torch_module
        self.name = name
        self._raw = bytearray(raw)
        self.shape = shape
        self.dtype = dtype
        self.device = torch_module.cuda_device
        self.layout = torch_module.strided

    @property
    def data(self) -> "_MirrorFakeParameter":
        return self

    def numel(self) -> int:
        return len(self._raw) // self.dtype.itemsize

    def element_size(self) -> int:
        return self.dtype.itemsize

    def is_contiguous(self) -> bool:
        return True

    def detach(self) -> "_MirrorFakeParameter":
        return self

    def reshape(self, *shape: int) -> "_MirrorFakeParameter":
        if shape != (-1,):
            raise AssertionError("fake parameter supports only flat reshape")
        return self

    def view(self, dtype: object) -> "_MirrorFakeByteView":
        if dtype is not self._torch_module.uint8:
            raise AssertionError("fake parameter supports only raw-byte views")
        return _MirrorFakeByteView(self, 0, len(self._raw))

    def raw_bytes(self) -> bytes:
        return bytes(self._raw)

    def write_bytes(self, offset: int, value: bytes) -> None:
        self._raw[offset : offset + len(value)] = value

    def _read(self, offset: int, length: int) -> bytes:
        return bytes(self._raw[offset : offset + length])


class _MirrorFakePinnedSlot:
    def __init__(
        self,
        torch_module: "_MirrorFakeTorch",
        slot_id: int,
        length: int,
    ) -> None:
        self._torch_module = torch_module
        self.slot_id = slot_id
        self._length = length
        self._buffer = (ctypes.c_ubyte * length)()
        self._released = False
        self.dtype = torch_module.uint8
        self.device = torch_module.cpu_device

    @property
    def nbytes(self) -> int:
        return self._length

    def __del__(self) -> None:
        if not self._released:
            self._released = True
            self._torch_module._release_pinned_slot(self._length)

    def numel(self) -> int:
        return self._length

    def is_contiguous(self) -> bool:
        return True

    def is_pinned(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return ctypes.addressof(self._buffer)

    def narrow(self, dimension: int, offset: int, length: int) -> "_MirrorFakeByteView":
        if dimension != 0 or offset < 0 or length < 0 or offset + length > self._length:
            raise AssertionError("fake pinned-slot narrow differs")
        return _MirrorFakeByteView(self, offset, length)

    def _read(self, offset: int, length: int) -> bytes:
        return ctypes.string_at(self.data_ptr() + offset, length)

    def _write(self, offset: int, value: bytes) -> None:
        ctypes.memmove(self.data_ptr() + offset, value, len(value))


class _MirrorFakeByteView:
    def __init__(self, owner: object, offset: int, length: int) -> None:
        self._owner = owner
        self._offset = offset
        self._length = length

    def numel(self) -> int:
        return self._length

    def narrow(self, dimension: int, offset: int, length: int) -> "_MirrorFakeByteView":
        if dimension != 0 or offset < 0 or length < 0 or offset + length > self._length:
            raise AssertionError("fake raw-byte narrow differs")
        return _MirrorFakeByteView(
            self._owner,
            self._offset + offset,
            length,
        )

    def _read(self) -> bytes:
        return self._owner._read(self._offset, self._length)

    def _write(self, value: bytes) -> None:
        if len(value) != self._length:
            raise AssertionError("fake asynchronous copy length differs")
        self._owner._write(self._offset, value)

    def copy_(
        self,
        source: "_MirrorFakeByteView",
        *,
        non_blocking: bool,
    ) -> "_MirrorFakeByteView":
        torch_module = self._owner._torch_module
        stream = torch_module.active_stream
        if (
            not isinstance(self._owner, _MirrorFakePinnedSlot)
            or not isinstance(source, _MirrorFakeByteView)
            or non_blocking is not True
            or stream is None
            or stream.kind != "copy"
        ):
            raise AssertionError("fake asynchronous copy boundary differs")
        scan = torch_module.current_scan
        slot_id = self._owner.slot_id
        torch_module.operations.append(
            ("enqueue", scan, slot_id, self._offset, source._offset, self._length)
        )

        def transfer() -> None:
            self._write(source._read())

        stream.enqueue(
            transfer,
            (scan, slot_id, self._offset, source._offset, self._length),
        )
        return self


class _MirrorFakeStream:
    def __init__(self, torch_module: "_MirrorFakeTorch", kind: str) -> None:
        self._torch_module = torch_module
        self.kind = kind
        self._operations: list[tuple[object, tuple[int, ...]] | None] = []
        self._executed = 0

    def enqueue(self, operation: object, metadata: tuple[int, ...]) -> None:
        self._operations.append((operation, metadata))

    def execute_through(self, boundary: int) -> None:
        while self._executed < boundary:
            item = self._operations[self._executed]
            self._operations[self._executed] = None
            self._executed += 1
            if item is None:
                raise AssertionError("fake copy operation was reused")
            operation, metadata = item
            self._torch_module.operations.append(("execute", *metadata))
            operation()

    def wait_event(self, event: "_MirrorFakeEvent") -> None:
        if self.kind != "copy" or event.recorded_kind != "model":
            raise AssertionError("fake model-stream dependency differs")
        self._torch_module.copy_stream_waits += 1
        self._torch_module.operations.append(
            ("dependency", self._torch_module.current_scan, event.event_id)
        )

    def synchronize(self) -> None:
        self._torch_module.copy_stream_synchronizations += 1
        self._torch_module.operations.append(
            ("copy-stream-sync", self._torch_module.current_scan)
        )
        self.execute_through(len(self._operations))


class _MirrorFakeEvent:
    def __init__(self, torch_module: "_MirrorFakeTorch", event_id: int) -> None:
        self._torch_module = torch_module
        self.event_id = event_id
        self.recorded_kind: str | None = None
        self._stream: _MirrorFakeStream | None = None
        self._boundary = 0
        self._scan = 0

    def record(self, stream: _MirrorFakeStream) -> None:
        self.recorded_kind = stream.kind
        if stream.kind == "model":
            self._torch_module.current_scan += 1
            self._torch_module.model_stream_records += 1
            self._scan = self._torch_module.current_scan
            self._stream = None
            self._boundary = 0
            self._torch_module.operations.append(
                ("model-record", self._scan, self.event_id)
            )
            return
        if stream.kind != "copy":
            raise AssertionError("fake event stream differs")
        self._scan = self._torch_module.current_scan
        self._stream = stream
        self._boundary = len(stream._operations)
        self._torch_module.operations.append(
            ("copy-record", self._scan, self.event_id, self._boundary)
        )

    def synchronize(self) -> None:
        if self.recorded_kind != "copy" or self._stream is None:
            raise AssertionError("fake event completion is absent")
        self._torch_module.operations.append(
            ("event-sync", self._scan, self.event_id, self._boundary)
        )
        self._stream.execute_through(self._boundary)


class _MirrorFakeStreamContext:
    def __init__(
        self,
        torch_module: "_MirrorFakeTorch",
        stream: _MirrorFakeStream,
    ) -> None:
        self._torch_module = torch_module
        self._stream = stream
        self._prior: _MirrorFakeStream | None = None

    def __enter__(self) -> _MirrorFakeStream:
        self._prior = self._torch_module.active_stream
        self._torch_module.active_stream = self._stream
        return self._stream

    def __exit__(self, *_exc: object) -> None:
        self._torch_module.active_stream = self._prior


class _MirrorFakeCuda:
    def __init__(self, torch_module: "_MirrorFakeTorch") -> None:
        self._torch_module = torch_module
        self._model_stream = _MirrorFakeStream(torch_module, "model")

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def current_device(self) -> int:
        return 0

    def host_memory_stats(self) -> dict[str, int]:
        self._torch_module.host_stats_calls += 1
        if self._torch_module.host_stats_calls in self._torch_module.stats_fail_calls:
            raise RuntimeError("private fake allocator failure")
        observed = dict(self._torch_module.allocator)
        if self._torch_module.host_stats_calls in self._torch_module.stats_drift_calls:
            observed["active_bytes.current"] += 1
        return observed

    def Stream(self, *, device: object) -> _MirrorFakeStream:
        if device != self._torch_module.cuda_device:
            raise AssertionError("fake copy-stream device differs")
        self._torch_module.copy_stream_creations += 1
        return _MirrorFakeStream(self._torch_module, "copy")

    def Event(
        self,
        *,
        enable_timing: bool,
        blocking: bool,
        interprocess: bool,
    ) -> _MirrorFakeEvent:
        if enable_timing or blocking or interprocess:
            raise AssertionError("fake completion event flags differ")
        event = _MirrorFakeEvent(
            self._torch_module,
            self._torch_module.event_creations,
        )
        self._torch_module.event_creations += 1
        return event

    def current_stream(self, *, device: object) -> _MirrorFakeStream:
        if device != self._torch_module.cuda_device:
            raise AssertionError("fake current-stream device differs")
        return self._model_stream

    def stream(self, stream: _MirrorFakeStream) -> _MirrorFakeStreamContext:
        if stream.kind != "copy":
            raise AssertionError("fake copy-stream context differs")
        return _MirrorFakeStreamContext(self._torch_module, stream)

    def synchronize(self) -> None:
        self._torch_module.global_synchronizations += 1
        raise AssertionError("global CUDA synchronization is forbidden")


class _MirrorFakeAccelerator:
    def __init__(self, torch_module: "_MirrorFakeTorch") -> None:
        self._torch_module = torch_module

    def empty_host_cache(self) -> None:
        self._torch_module.empty_host_cache_calls += 1
        if self._torch_module.empty_host_cache_failures:
            self._torch_module.empty_host_cache_failures -= 1
            raise RuntimeError("private fake host-cache failure")
        baseline = self._torch_module.allocator_baseline
        if (
            self._torch_module.allocator["active_requests.current"]
            != baseline["active_requests.current"]
            or self._torch_module.allocator["active_bytes.current"]
            != baseline["active_bytes.current"]
        ):
            raise RuntimeError("private fake pinned allocation remains active")
        self._torch_module.allocator["allocations.current"] = baseline[
            "allocations.current"
        ]
        self._torch_module.allocator["allocated_bytes.current"] = baseline[
            "allocated_bytes.current"
        ]


class _MirrorFakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.uint8 = _MirrorFakeDtype("torch.uint8", 1)
        self.float32 = _MirrorFakeDtype("torch.float32", 4)
        self.alternate_float32 = _MirrorFakeDtype("torch.alternate_float32", 4)
        self.strided = object()
        self.cuda_device = _MirrorFakeDevice("cuda", 0)
        self.cpu_device = _MirrorFakeDevice("cpu", None)
        self.allocator_baseline = {
            "allocations.current": 3,
            "allocated_bytes.current": 17,
            "active_requests.current": 1,
            "active_bytes.current": 5,
        }
        self.allocator = dict(self.allocator_baseline)
        self.operations: list[tuple[object, ...]] = []
        self.active_stream: _MirrorFakeStream | None = None
        self.current_scan = 0
        self.copy_stream_creations = 0
        self.event_creations = 0
        self.model_stream_records = 0
        self.copy_stream_waits = 0
        self.copy_stream_synchronizations = 0
        self.global_synchronizations = 0
        self.pinned_allocation_calls = 0
        self.successful_pinned_allocations = 0
        self.released_pinned_allocations = 0
        self.host_stats_calls = 0
        self.stats_fail_calls: set[int] = set()
        self.stats_drift_calls: set[int] = set()
        self.fail_allocation_call: int | None = None
        self.empty_host_cache_calls = 0
        self.empty_host_cache_failures = 0
        self.cuda = _MirrorFakeCuda(self)
        self.accelerator = _MirrorFakeAccelerator(self)

    def empty(
        self,
        length: int,
        *,
        dtype: object,
        device: str,
        pin_memory: bool,
    ) -> _MirrorFakePinnedSlot:
        self.pinned_allocation_calls += 1
        if self.fail_allocation_call == self.pinned_allocation_calls:
            raise RuntimeError("private fake pinned allocation failure")
        if dtype is not self.uint8 or device != "cpu" or pin_memory is not True:
            raise AssertionError("fake pinned allocation request differs")
        slot = _MirrorFakePinnedSlot(
            self,
            self.successful_pinned_allocations,
            length,
        )
        self.successful_pinned_allocations += 1
        self.allocator["allocations.current"] += 1
        self.allocator["allocated_bytes.current"] += length
        self.allocator["active_requests.current"] += 1
        self.allocator["active_bytes.current"] += length
        self.operations.append(
            ("allocate", slot.slot_id, length, slot.data_ptr())
        )
        return slot

    def _release_pinned_slot(self, length: int) -> None:
        self.released_pinned_allocations += 1
        self.allocator["active_requests.current"] -= 1
        self.allocator["active_bytes.current"] -= length


class _MirrorTinyModel:
    def __init__(
        self,
        rows: tuple[tuple[str, _MirrorFakeParameter], ...],
    ) -> None:
        self.rows = rows

    def named_parameters(self) -> object:
        return iter(self.rows)

    def parameters(self) -> object:
        return iter(parameter for _name, parameter in self.rows)


class _MirrorFoundationGuard:
    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.probes = 0
        self.full_verifications = 0

    def probe(self) -> str:
        self.probes += 1
        return self.digest

    def verify_boundary_digest(self) -> str:
        self.full_verifications += 1
        return self.digest


def _mirror_tiny_fixture(
    torch_module: _MirrorFakeTorch,
) -> tuple[_MirrorTinyModel, str]:
    rows = (
        (
            "nan_parameter",
            _MirrorFakeParameter(
                torch_module,
                "nan_parameter",
                struct.pack("<II", 0x7FC00001, 0x3F800000),
                shape=(2,),
                dtype=torch_module.float32,
            ),
        ),
        (
            "zero_parameter",
            _MirrorFakeParameter(
                torch_module,
                "zero_parameter",
                struct.pack("<I", 0x00000000),
                shape=(1,),
                dtype=torch_module.float32,
            ),
        ),
        (
            "tail_parameter",
            _MirrorFakeParameter(
                torch_module,
                "tail_parameter",
                struct.pack("<II", 0x40000000, 0x40400000),
                shape=(2,),
                dtype=torch_module.float32,
            ),
        ),
    )
    model = _MirrorTinyModel(rows)
    digest = runner._foundation_mirror_digest(
        tuple(
            runner._FoundationByteMirrorRow(
                name=name,
                shape=parameter.shape,
                dtype=str(parameter.dtype),
                raw=parameter.raw_bytes(),
            )
            for name, parameter in rows
        )
    )
    return model, digest


def _mirror_fake_boundary(
    torch_module: _MirrorFakeTorch,
    *,
    current_rss: int = 32,
    peak_rss: int = 48,
    memlock_failure: BaseException | None = None,
) -> ExitStack:
    stack = ExitStack()
    constants = {
        "QUALIFIED_MODEL_PARAMETER_COUNT": 3,
        "QUALIFIED_MODEL_PARAMETER_BYTES": 20,
        "QUALIFIED_MODEL_LARGEST_PARAMETER_BYTES": 8,
        "FOUNDATION_MIRROR_PINNED_SLOT_BYTES": 8,
        "FOUNDATION_MIRROR_PINNED_SLOT_COUNT": 2,
        "FOUNDATION_MIRROR_PINNED_BYTES": 16,
        "FOUNDATION_MIRROR_TRANSIENT_BYTES": 4,
        "FOUNDATION_MIRROR_MAXIMUM_VERIFICATION_NS": 100,
    }
    for name, value in constants.items():
        stack.enter_context(mock.patch.object(runner, name, value))
    stack.enter_context(mock.patch.dict(os.sys.modules, {"torch": torch_module}))
    if memlock_failure is None:
        stack.enter_context(
            mock.patch.object(
                runner,
                "_foundation_mirror_memlock_precheck",
                return_value=(4, 1_024),
            )
        )
    else:
        stack.enter_context(
            mock.patch.object(
                runner,
                "_foundation_mirror_memlock_precheck",
                side_effect=memlock_failure,
            )
        )
    stack.enter_context(
        mock.patch.object(
            runner,
            "_current_process_rss_bytes",
            return_value=current_rss,
        )
    )
    stack.enter_context(
        mock.patch.object(
            runner,
            "_peak_process_rss_bytes",
            return_value=peak_rss,
        )
    )
    return stack


class ExactFoundationByteMirrorTests(unittest.TestCase):
    def test_two_slot_pipeline_is_exact_deferred_and_first_three_timed(self) -> None:
        torch_module = _MirrorFakeTorch()
        model, digest = _mirror_tiny_fixture(torch_module)
        guard = _MirrorFoundationGuard(digest)
        real_fill = runner._foundation_mirror_fill_bytes_from_slot
        real_match = runner._foundation_mirror_slot_matches_bytes

        def traced_fill(**values: object) -> None:
            slot = values["slot"]
            torch_module.operations.append(
                (
                    "fill",
                    torch_module.current_scan,
                    slot.slot_id,
                    values["slot_offset"],
                    values["row_offset"],
                    values["length"],
                )
            )
            real_fill(**values)

        def traced_match(**values: object) -> bool:
            slot = values["slot"]
            torch_module.operations.append(
                (
                    "compare",
                    torch_module.current_scan,
                    slot.slot_id,
                    values["slot_offset"],
                    values["row_offset"],
                    values["length"],
                )
            )
            return real_match(**values)

        with (
            _mirror_fake_boundary(torch_module),
            mock.patch.object(
                runner,
                "_foundation_mirror_fill_bytes_from_slot",
                new=traced_fill,
            ),
            mock.patch.object(
                runner,
                "_foundation_mirror_slot_matches_bytes",
                new=traced_match,
            ),
            mock.patch.object(
                runner.time,
                "perf_counter_ns",
                side_effect=(100, 110, 200, 220, 300, 330),
            ),
        ):
            mirror = runner._ExactFoundationByteMirror(
                model,
                foundation_guard=guard,
                expected_digest=digest,
                resources=runner.ResourceLedger(runner.RunMode.QUALIFICATION),
            )
            self.assertEqual(mirror._verification_durations_ns, [])
            self.assertEqual(
                tuple(
                    (
                        chunk.global_offset,
                        chunk.length,
                        tuple(
                            (
                                segment.row_index,
                                segment.row_offset,
                                segment.slot_offset,
                                segment.length,
                            )
                            for segment in chunk.segments
                        ),
                    )
                    for chunk in mirror._plan
                ),
                (
                    (0, 8, ((0, 0, 0, 8),)),
                    (8, 8, ((1, 0, 0, 4), (2, 0, 4, 4))),
                    (16, 4, ((2, 4, 0, 4),)),
                ),
            )
            self.assertEqual(
                tuple(row.raw for row in mirror._rows),
                tuple(parameter.raw_bytes() for _name, parameter in model.rows),
            )
            self.assertTrue(all(type(row.raw) is bytes for row in mirror._rows))
            self.assertEqual(torch_module.pinned_allocation_calls, 2)
            self.assertEqual(torch_module.successful_pinned_allocations, 2)
            self.assertEqual(torch_module.copy_stream_creations, 1)
            self.assertEqual(torch_module.event_creations, 2)
            self.assertEqual(guard.full_verifications, 2)

            self.assertEqual(mirror.verify_exact(), digest)
            self.assertEqual(mirror.verify_exact(), digest)
            self.assertEqual(mirror.verify_exact(), digest)
            self.assertEqual(mirror._verification_durations_ns, [10, 20, 30])
            self.assertEqual(mirror.assert_first_three_performance_gate(), (10, 20, 30))
            self.assertEqual(guard.probes, 6)
            self.assertEqual(torch_module.current_scan, 4)
            self.assertEqual(torch_module.model_stream_records, 4)
            self.assertEqual(torch_module.copy_stream_waits, 4)
            self.assertEqual(torch_module.global_synchronizations, 0)

            for scan in range(1, 5):
                indexed = [
                    (index, item)
                    for index, item in enumerate(torch_module.operations)
                    if len(item) > 1 and item[1] == scan
                ]
                slot_zero_enqueues = [
                    index
                    for index, item in indexed
                    if item[0] == "enqueue" and item[2] == 0
                ]
                slot_zero_drains = [
                    index
                    for index, item in indexed
                    if item[0] == "event-sync" and item[2] == 0
                ]
                comparisons = [
                    index
                    for index, item in indexed
                    if item[0] in {"fill", "compare"} and item[2] == 0
                ]
                self.assertEqual(len(slot_zero_enqueues), 2)
                self.assertEqual(len(slot_zero_drains), 2)
                self.assertGreaterEqual(len(comparisons), 2)
                self.assertLess(slot_zero_enqueues[0], slot_zero_drains[0])
                self.assertLess(slot_zero_drains[0], comparisons[0])
                self.assertLess(comparisons[0], slot_zero_enqueues[1])
                self.assertLess(slot_zero_enqueues[1], slot_zero_drains[1])
                self.assertLess(slot_zero_drains[1], comparisons[-1])
                self.assertTrue(
                    any(item[0] == "event-sync" and item[2] == 1 for _, item in indexed)
                )

            live_receipt = mirror.audit_receipt
            self.assertEqual(live_receipt.verification_durations_ns, (10, 20, 30))
            self.assertTrue(live_receipt.performance_gate_passed)
            self.assertFalse(live_receipt.cleanup_proven)
            mirror.close()
            closed_receipt = mirror.audit_receipt
            self.assertTrue(closed_receipt.closed)
            self.assertTrue(closed_receipt.cleanup_proven)
            self.assertEqual(
                closed_receipt.allocator_cleanup,
                closed_receipt.allocator_baseline,
            )
            self.assertEqual(torch_module.allocator, torch_module.allocator_baseline)
            self.assertEqual(torch_module.released_pinned_allocations, 2)
            self.assertEqual(mirror._slots, ())
            self.assertEqual(mirror._events, ())
            self.assertIsNone(mirror._copy_stream)
            self.assertIsNone(mirror._model)
            self.assertEqual(torch_module.global_synchronizations, 0)

    def test_raw_and_metadata_drift_terminally_poison_the_mirror(self) -> None:
        cases = ("nan-payload", "signed-zero", "dtype", "order")
        for case in cases:
            with self.subTest(case=case):
                torch_module = _MirrorFakeTorch()
                model, digest = _mirror_tiny_fixture(torch_module)
                guard = _MirrorFoundationGuard(digest)
                with _mirror_fake_boundary(torch_module):
                    mirror = runner._ExactFoundationByteMirror(
                        model,
                        foundation_guard=guard,
                        expected_digest=digest,
                        resources=runner.ResourceLedger(
                            runner.RunMode.QUALIFICATION
                        ),
                    )
                    original_rows = model.rows
                    if case == "nan-payload":
                        parameter = model.rows[0][1]
                        original = parameter.raw_bytes()
                        changed = struct.pack("<I", 0x7FC00002)
                        self.assertTrue(math.isnan(struct.unpack("<f", changed)[0]))
                        self.assertTrue(
                            math.isnan(struct.unpack("<f", original[:4])[0])
                        )
                        parameter.data.write_bytes(0, changed)
                    elif case == "signed-zero":
                        parameter = model.rows[1][1]
                        original = parameter.raw_bytes()
                        changed = struct.pack("<f", -0.0)
                        self.assertEqual(struct.unpack("<f", changed)[0], 0.0)
                        self.assertNotEqual(changed, original)
                        parameter.data.write_bytes(0, changed)
                    elif case == "dtype":
                        parameter = model.rows[2][1]
                        original = parameter.dtype
                        parameter.dtype = torch_module.alternate_float32
                    else:
                        original = model.rows
                        model.rows = tuple(reversed(model.rows))

                    expected = (
                        "parameter bytes changed"
                        if case in {"nan-payload", "signed-zero"}
                        else "metadata differs"
                    )
                    with self.assertRaisesRegex(runner.V2IntegrityStop, expected):
                        mirror.verify_exact()
                    self.assertTrue(mirror._poisoned)

                    if case in {"nan-payload", "signed-zero"}:
                        parameter.data.write_bytes(0, original[:4])
                    elif case == "dtype":
                        parameter.dtype = original
                    else:
                        model.rows = original_rows
                    with self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "terminally poisoned",
                    ):
                        mirror.verify_exact()
                    mirror.close()
                    self.assertTrue(mirror.audit_receipt.poisoned)
                    self.assertEqual(
                        torch_module.allocator,
                        torch_module.allocator_baseline,
                    )
                    self.assertEqual(torch_module.global_synchronizations, 0)

    def test_memlock_rss_and_allocator_failures_clean_partial_state(self) -> None:
        cases = ("memlock", "rss", "allocator-stats", "allocation", "delta")
        for case in cases:
            with self.subTest(case=case):
                torch_module = _MirrorFakeTorch()
                model, digest = _mirror_tiny_fixture(torch_module)
                guard = _MirrorFoundationGuard(digest)
                memlock_failure = None
                current_rss = 32
                expected = ""
                if case == "memlock":
                    memlock_failure = runner.V2IntegrityStop(
                        "foundation mirror memlock reservation is unavailable"
                    )
                    expected = "FOUNDATION_MIRROR_CONSTRUCTION_FAILED"
                elif case == "rss":
                    current_rss = runner.MAXIMUM_RSS_BYTES
                    expected = "FOUNDATION_MIRROR_CONSTRUCTION_FAILED"
                elif case == "allocator-stats":
                    torch_module.stats_fail_calls.add(1)
                    expected = "FOUNDATION_MIRROR_CONSTRUCTION_FAILED"
                elif case == "allocation":
                    torch_module.fail_allocation_call = 2
                    expected = "FOUNDATION_MIRROR_CONSTRUCTION_FAILED"
                else:
                    torch_module.stats_drift_calls.add(2)
                    expected = "FOUNDATION_MIRROR_CONSTRUCTION_FAILED"
                with (
                    _mirror_fake_boundary(
                        torch_module,
                        current_rss=current_rss,
                        memlock_failure=memlock_failure,
                    ),
                    self.assertRaisesRegex(runner.V2IntegrityStop, expected),
                ):
                    runner._ExactFoundationByteMirror(
                        model,
                        foundation_guard=guard,
                        expected_digest=digest,
                        resources=runner.ResourceLedger(
                            runner.RunMode.QUALIFICATION
                        ),
                    )
                self.assertEqual(
                    torch_module.allocator,
                    torch_module.allocator_baseline,
                )
                self.assertEqual(torch_module.global_synchronizations, 0)
                if case in {"memlock", "rss", "allocator-stats"}:
                    self.assertEqual(torch_module.successful_pinned_allocations, 0)
                elif case == "allocation":
                    self.assertEqual(torch_module.successful_pinned_allocations, 1)
                    self.assertEqual(torch_module.released_pinned_allocations, 1)
                else:
                    self.assertEqual(torch_module.successful_pinned_allocations, 2)
                    self.assertEqual(torch_module.released_pinned_allocations, 2)

    def test_close_retries_cleanup_and_restores_allocator_baseline(self) -> None:
        torch_module = _MirrorFakeTorch()
        model, digest = _mirror_tiny_fixture(torch_module)
        guard = _MirrorFoundationGuard(digest)
        with _mirror_fake_boundary(torch_module):
            mirror = runner._ExactFoundationByteMirror(
                model,
                foundation_guard=guard,
                expected_digest=digest,
                resources=runner.ResourceLedger(runner.RunMode.QUALIFICATION),
            )
            self.assertIsNotNone(mirror._copy_stream)
            mirror._stream_may_have_work = True
            with (
                mock.patch.object(
                    mirror._copy_stream,
                    "synchronize",
                    side_effect=RuntimeError(
                        "private mirror drain at /secret/mirror/stream"
                    ),
                ),
                self.assertRaises(runner.V2IntegrityStop) as raised,
            ):
                mirror.close()
            self.assertEqual(
                str(raised.exception),
                "FOUNDATION_MIRROR_CLOSE_DRAIN_FAILED",
            )
            self.assertEqual(
                raised.exception.boundary_code_chain,
                ("FOUNDATION_MIRROR_CLOSE_DRAIN_FAILED",),
            )
            self.assertIsNone(raised.exception.__cause__)
            self.assertIsNone(raised.exception.__context__)
            self.assertNotIn("private mirror drain", str(raised.exception))
            self.assertNotIn("/secret/mirror/stream", str(raised.exception))
            self.assertFalse(mirror._closed)
            self.assertTrue(mirror._poisoned)
            self.assertFalse(mirror._resources_released)

            mirror.close()
            self.assertTrue(mirror._closed)
            self.assertTrue(mirror._cleanup_proven)
            self.assertEqual(torch_module.allocator, torch_module.allocator_baseline)

        torch_module = _MirrorFakeTorch()
        model, digest = _mirror_tiny_fixture(torch_module)
        guard = _MirrorFoundationGuard(digest)
        with _mirror_fake_boundary(torch_module):
            mirror = runner._ExactFoundationByteMirror(
                model,
                foundation_guard=guard,
                expected_digest=digest,
                resources=runner.ResourceLedger(runner.RunMode.QUALIFICATION),
            )
            torch_module.empty_host_cache_failures = 1
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "FOUNDATION_MIRROR_CLOSE_CLEANUP_FAILED",
            ) as raised:
                mirror.close()
            self.assertEqual(
                str(raised.exception),
                "FOUNDATION_MIRROR_CLOSE_CLEANUP_FAILED",
            )
            self.assertEqual(
                raised.exception.boundary_code_chain,
                ("FOUNDATION_MIRROR_CLOSE_CLEANUP_FAILED",),
            )
            self.assertIsNone(raised.exception.__cause__)
            self.assertIsNone(raised.exception.__context__)
            self.assertNotIn(
                "private fake host-cache failure",
                str(raised.exception),
            )
            self.assertFalse(mirror._closed)
            self.assertTrue(mirror._poisoned)
            self.assertTrue(mirror._resources_released)
            self.assertEqual(mirror._slots, ())
            self.assertEqual(mirror._events, ())
            self.assertIsNone(mirror._copy_stream)
            self.assertEqual(torch_module.released_pinned_allocations, 2)
            self.assertGreater(
                torch_module.allocator["allocations.current"],
                torch_module.allocator_baseline["allocations.current"],
            )

            mirror.close()
            self.assertTrue(mirror._closed)
            self.assertTrue(mirror._cleanup_proven)
            self.assertEqual(torch_module.empty_host_cache_calls, 2)
            self.assertEqual(torch_module.allocator, torch_module.allocator_baseline)
            receipt = mirror.audit_receipt
            self.assertEqual(receipt.allocator_cleanup, receipt.allocator_baseline)
            self.assertTrue(receipt.closed)
            self.assertTrue(receipt.cleanup_proven)
            self.assertTrue(receipt.poisoned)
            mirror.close()

    def test_redacted_failure_releases_traceback_owners(self) -> None:
        import gc
        import weakref

        class Owner:
            pass

        def fail_with_owner(value: object) -> None:
            retained_only_by_traceback = value
            if retained_only_by_traceback is value:
                raise RuntimeError("private failure detail")

        owner: Owner | None = Owner()
        owner_ref = weakref.ref(owner)
        original: BaseException | None = None
        try:
            fail_with_owner(owner)
        except BaseException as error:
            original = error
        self.assertIsNotNone(original)
        redacted = runner._redacted_failure(
            original,
            "BOUNDED_TEST_FAILURE",
        )
        self.assertEqual(str(redacted), "BOUNDED_TEST_FAILURE")
        self.assertEqual(redacted.boundary_code_chain, ())
        self.assertIsNone(original.__traceback__)
        self.assertIsNone(original.__cause__)
        self.assertIsNone(original.__context__)
        owner = None
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_cleanup_redaction_retains_only_capped_allowlisted_code_chain(
        self,
    ) -> None:
        import gc
        import weakref

        class Owner:
            pass

        def fail_with_owner(value: object) -> None:
            retained_only_by_traceback = value
            if retained_only_by_traceback is value:
                raise RuntimeError("private detail at /secret/runtime/path")

        owner: Owner | None = Owner()
        owner_ref = weakref.ref(owner)
        private: BaseException | None = None
        try:
            fail_with_owner(owner)
        except BaseException as error:
            private = error
        self.assertIsNotNone(private)
        nested = ExceptionGroup(
            "private cleanup group",
            [
                runner.V2IntegrityStop("V2_PRODUCT_FOUNDATION_CLOSE_FAILED"),
                ExceptionGroup(
                    "private nested cleanup group",
                    [
                        runner.V2IntegrityStop(
                            "QUALIFIED_FOUNDATION_MIRROR_CLOSE_FAILED"
                        ),
                        runner.V2IntegrityStop(
                            "FOUNDATION_MIRROR_CLOSE_CLEANUP_FAILED"
                        ),
                        runner.V2IntegrityStop("UNALLOWLISTED_UPPERCASE"),
                        private,
                    ],
                ),
            ],
        )
        redacted = runner._redacted_cleanup_failure(
            nested,
            "V2_RUNTIME_CLOSE_FAILED",
        )
        expected = (
            "V2_RUNTIME_CLOSE_FAILED",
            "V2_PRODUCT_FOUNDATION_CLOSE_FAILED",
            "QUALIFIED_FOUNDATION_MIRROR_CLOSE_FAILED",
            "FOUNDATION_MIRROR_CLOSE_CLEANUP_FAILED",
        )
        self.assertEqual(redacted.boundary_code_chain, expected)
        self.assertEqual(str(redacted), ">".join(expected))
        self.assertNotIn("secret", str(redacted))
        self.assertNotIn("UNALLOWLISTED", str(redacted))
        self.assertIsNone(private.__traceback__)
        self.assertIsNone(private.__cause__)
        self.assertIsNone(private.__context__)
        owner = None
        private = None
        nested = None
        gc.collect()
        self.assertIsNone(owner_ref())

        many = ExceptionGroup(
            "private overlong cleanup group",
            [
                runner.V2IntegrityStop(code)
                for code in (
                    "V2_EVALUATOR_CLOSE_FAILED",
                    "V2_EVALUATOR_REAP_UNPROVEN",
                    "V2_RUNTIME_CLOSE_FAILED",
                    "V2_PATH_REGISTRY_PRECLOSE_FAILED",
                    "V2_PATH_REGISTRY_POSTCLOSE_FAILED",
                    "V2_LEDGER_CLOSE_FAILED",
                    "V2_PRODUCT_FOUNDATION_CLOSE_FAILED",
                    "QUALIFIED_FOUNDATION_GUARD_CLOSE_FAILED",
                    "QUALIFIED_FOUNDATION_MIRROR_CLOSE_FAILED",
                )
            ],
        )
        capped = runner._redacted_cleanup_failure(
            many,
            "V2_RUN_CLEANUP_FAILED",
        )
        self.assertEqual(len(capped.boundary_code_chain), 8)
        self.assertEqual(
            capped.boundary_code_chain[0],
            "V2_RUN_CLEANUP_FAILED",
        )
        self.assertEqual(
            capped.boundary_code_chain[-1],
            "V2_BOUNDARY_CODE_CHAIN_TRUNCATED",
        )
        self.assertEqual(
            len(set(capped.boundary_code_chain)),
            len(capped.boundary_code_chain),
        )

        class FailingInjectedBinding:
            async def close(self) -> None:
                raise RuntimeError(
                    "private injected binding at /secret/cognee/binding"
                )

        with self.assertRaises(runner.V2IntegrityStop) as raised:
            asyncio.run(
                runner._close_v2_acquisition_binding_preserving_state(
                    FailingInjectedBinding()
                )
            )
        self.assertEqual(
            str(raised.exception),
            "V2_INJECTED_COGNEE_BINDING_CLOSE_FAILED",
        )
        self.assertEqual(
            raised.exception.boundary_code_chain,
            ("V2_INJECTED_COGNEE_BINDING_CLOSE_FAILED",),
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertNotIn("private injected binding", str(raised.exception))
        self.assertNotIn("/secret/cognee/binding", str(raised.exception))

        stateful_owner = object.__new__(runner.V2StatefulAttemptOwner)
        stateful_owner._closed = False
        stateful_owner.runtime = types.SimpleNamespace(
            cycle=object(),
            learner=object(),
            store=object(),
            journal=object(),
            memory=object(),
        )

        def fail_quiescence(**_owners: object) -> None:
            raise RuntimeError(
                "private quiescence detail at /secret/runtime/quiescence"
            )

        stateful_owner.quiesce_runtime = fail_quiescence
        with self.assertRaises(runner.V2IntegrityStop) as raised:
            asyncio.run(stateful_owner._quiesce())
        self.assertEqual(
            str(raised.exception),
            "V2_STATEFUL_RUNTIME_QUIESCENCE_FAILED",
        )
        self.assertEqual(
            raised.exception.boundary_code_chain,
            ("V2_STATEFUL_RUNTIME_QUIESCENCE_FAILED",),
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertNotIn("private quiescence detail", str(raised.exception))
        self.assertNotIn("/secret/runtime/quiescence", str(raised.exception))

    def test_injected_run_redacts_primary_before_closing_owned_runtime(self) -> None:
        import gc
        import weakref

        class Owner:
            pass

        evaluator_owner: Owner | None = Owner()
        runtime_owner: Owner | None = Owner()
        evaluator_owner_ref = weakref.ref(evaluator_owner)
        runtime_owner_ref = weakref.ref(runtime_owner)
        events: list[str] = []

        class FailingEvaluator:
            protocol_identity = runner.PROTOCOL_IDENTITY
            identity = runner.QUALIFICATION_IDENTITY

            def __init__(self, owner: Owner) -> None:
                self.owner: Owner | None = owner

            def release_phase(self, _phase: str) -> object:
                retained_only_by_failure_traceback = self.owner
                if retained_only_by_failure_traceback is not None:
                    raise RuntimeError("private evaluator failure")
                raise AssertionError("evaluator owner unexpectedly absent")

            def close(self) -> None:
                events.append("evaluator-close")
                self.owner = None

        evaluator_instance = FailingEvaluator(evaluator_owner)
        runtime_box = {"owner": runtime_owner}

        def close_runtime() -> None:
            events.append("runtime-close")
            runtime_box["owner"] = None

        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("failed run reached an attempt dependency")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = runner.EvidenceLedger(root / "failure.sqlite3", create=True)
            dependencies = runner.OrchestrationDependencies(
                mode=runner.RunMode.QUALIFICATION,
                evaluator=evaluator_instance,
                ledger=ledger,
                path_allocations=runner.PathAllocationRegistry(
                    "qualification",
                    ledger,
                ),
                attempt_budget=runner.AttemptBudget(
                    runner.RunMode.QUALIFICATION
                ),
                resources=runner.ResourceLedger(runner.RunMode.QUALIFICATION),
                prepare_adaptation_attempt=forbidden,
                freeze_adaptation=forbidden,
                prepare_task_block=forbidden,
                dispose_recall_captures=forbidden,
                execute_attempt=forbidden,
                apply_objective_feedback=forbidden,
                terminalize_stateful_attempt=forbidden,
                dispose_clone=forbidden,
                dispose_stateless_attempt=forbidden,
                finalize_persistent_stateful_attempt=forbidden,
                verify_frozen_adaptation=forbidden,
                sample_resources=lambda: {
                    "configured_pools": 0,
                    "cuda_allocated_bytes": 0,
                    "cuda_reserved_bytes": 0,
                    "process_count": 0,
                    "rss_bytes": 0,
                    "state_scratch_bytes": 0,
                },
                verify_source_seal=lambda: _ref("failure-source-seal"),
                close_runtime=close_runtime,
            )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "V2_QUALIFICATION_RUN_FAILED",
            ) as raised:
                asyncio.run(
                    runner.run_qualification_injected(
                        dependencies,
                        qualification_release_ref=_ref(
                            "failure-qualification-release"
                        ),
                    )
                )
            self.assertIsNone(raised.exception.__cause__)
            self.assertIsNone(raised.exception.__context__)
            self.assertEqual(events, ["evaluator-close", "runtime-close"])

        evaluator_owner = None
        runtime_owner = None
        gc.collect()
        self.assertIsNone(evaluator_owner_ref())
        self.assertIsNone(runtime_owner_ref())


class FoundationMirrorPreflightGlueTests(unittest.TestCase):
    @staticmethod
    def _sources() -> list[dict[str, object]]:
        return [
            {
                "bytes": index + 1,
                "mode": "0664",
                "path": relative,
                "sha256": _raw(f"preflight-source-{index}"),
            }
            for index, relative in enumerate(runner.SOURCE_MANIFEST_INVENTORY)
        ]

    def test_boundary_receipt_and_source_inventory_join_are_strict(self) -> None:
        sources = self._sources()
        source_ref = runner.canonical_ref("source-inventory", sources)
        boundary = _synthetic_preflight_boundary(source_ref)
        self.assertEqual(
            runner.validate_foundation_preflight_boundary_record(
                boundary,
                expected_source_inventory_ref=source_ref,
            ),
            boundary,
        )
        receipt = _synthetic_preflight_receipt(sources)
        self.assertEqual(
            runner.validate_foundation_mirror_preflight_receipt(
                receipt,
                source_inventory=sources,
                injected_root=True,
            ),
            receipt,
        )

        for label, changes in (
            (
                "formula",
                {"rss_reservation_bytes": receipt["rss_reservation_bytes"] + 1},
            ),
            (
                "ceiling",
                {
                    "rss_bytes_before": runner.MAXIMUM_RSS_BYTES,
                    "rss_reservation_bytes": (
                        runner.MAXIMUM_RSS_BYTES
                        + runner.QUALIFIED_MODEL_PARAMETER_BYTES
                        + runner.FOUNDATION_MIRROR_PINNED_BYTES
                        + runner.FOUNDATION_MIRROR_TRANSIENT_BYTES
                    ),
                },
            ),
        ):
            mutated = {**receipt, **changes}
            mutated_without_ref = dict(mutated)
            mutated_without_ref.pop("preflight_ref")
            mutated["preflight_ref"] = runner.canonical_ref(
                "foundation-mirror-preflight",
                mutated_without_ref,
            )
            with self.subTest(rss=label), self.assertRaisesRegex(
                runner.V2InvariantError,
                "preflight receipt differs",
            ):
                runner.FoundationMirrorPreflightReceipt.from_canonical(mutated)

        active_5070 = _synthetic_preflight_boundary(
            source_ref,
            unassigned_compute_processes=1,
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "boundary identity differs",
        ):
            runner.validate_foundation_preflight_boundary_record(
                active_5070,
                expected_source_inventory_ref=source_ref,
            )

        changed_sources = json.loads(json.dumps(sources))
        changed_sources[-1]["sha256"] = _raw("changed-preflight-source")
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "preflight source inventory changed",
        ):
            runner.validate_foundation_mirror_preflight_receipt(
                receipt,
                source_inventory=changed_sources,
                injected_root=True,
            )

    def test_namespace_and_host_boundary_are_strict_without_mounting(self) -> None:
        environment = runner._frozen_foundation_preflight_environment()
        namespace_values = {
            "/proc/self/ns/mnt": "mnt:[41001]",
            "/proc/self/ns/net": "net:[41002]",
        }
        status = "\n".join(
            (
                "NoNewPrivs:\t1",
                "CapEff:\t0000000000000000",
                "CapPrm:\t0000000000000000",
                "CapInh:\t0000000000000000",
                "CapAmb:\t0000000000000000",
                "CapBnd:\t0000000000000000",
            )
        )
        mountinfo = (
            "36 25 0:32 / /opt/angler/scratch rw,nosuid,nodev,noexec "
            "- tmpfs tmpfs rw,size=1048576k,mode=700,uid=1000,gid=1000"
        )
        path_checks: list[str] = []

        def namespace_reader(path: Path) -> str:
            return namespace_values[str(path)]

        def path_facts() -> dict[str, object]:
            path_checks.append("validated")
            return {
                "private_child_names": ["cache", "tmp"],
                "private_child_mode": "0700",
                "private_child_owner_gid": 1000,
                "private_child_owner_uid": 1000,
                "private_root": str(runner.FOUNDATION_PREFLIGHT_PRIVATE_ROOT),
                "private_root_mode": "0700",
                "private_root_owner_gid": 1000,
                "private_root_owner_uid": 1000,
                "scratch_mount_mode": "0700",
                "scratch_mount_owner_gid": 1000,
                "scratch_mount_owner_uid": 1000,
            }

        namespace = runner.validate_foundation_preflight_namespace_boundary(
            parent_mount_namespace="mnt:[51001]",
            parent_network_namespace="net:[51002]",
            environment=environment,
            effective_uid=1000,
            effective_gid=1000,
            supplementary_groups=(),
            namespace_reader=namespace_reader,
            status_reader=lambda: status,
            mountinfo_reader=lambda: mountinfo,
            validate_paths=path_facts,
        )
        self.assertEqual(path_checks, ["validated"])
        self.assertEqual(namespace["mount_namespace"], "mnt:[41001]")
        self.assertEqual(namespace["network_namespace"], "net:[41002]")
        self.assertEqual(
            namespace["environment_sha256"],
            hashlib.sha256(runner.canonical_json_bytes(environment)).hexdigest(),
        )

        sources = self._sources()
        captured: list[dict[str, object]] = []

        def parent_validator(**values: object) -> dict[str, object]:
            captured.append(values)
            return {
                "environment_sha256": namespace["environment_sha256"],
                "hostname": runner.EXPECTED_HOSTNAME,
                "logical_cuda_device": 0,
                "network_namespace": namespace["network_namespace"],
                "physical_gpus": [
                    {
                        "name": "NVIDIA GeForce RTX 5080",
                        "uuid": runner.ASSIGNED_GPU_UUID,
                    },
                    {
                        "name": "NVIDIA GeForce RTX 5070",
                        "uuid": runner.UNASSIGNED_GPU_UUID,
                    },
                ],
                "python": runner.EXPECTED_PYTHON_VERSION,
                "python_executable": str(runner.ANGLER_PYTHON),
                "unassigned_compute_processes": 0,
            }

        boundary = runner.collect_foundation_preflight_child_boundary(
            namespace_boundary=namespace,
            source_inventory=sources,
            live_parent_validator=parent_validator,
            torch_module=object(),
        )
        self.assertEqual(boundary["boundary"], "FOUNDATION_MIRROR_PREFLIGHT_CHILD")
        self.assertEqual(boundary["unassigned_compute_processes"], 0)
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]["require_unassigned_zero_processes"])
        self.assertEqual(captured[0]["expected_environment"], environment)

        invalid_cases = (
            {
                "supplementary_groups": (27,),
                "status_reader": lambda: status,
                "mountinfo_reader": lambda: mountinfo,
            },
            {
                "supplementary_groups": (),
                "status_reader": lambda: status.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0"),
                "mountinfo_reader": lambda: mountinfo,
            },
            {
                "supplementary_groups": (),
                "status_reader": lambda: status.replace(
                    "CapBnd:\t0000000000000000",
                    "CapBnd:\t0000000000000400",
                ),
                "mountinfo_reader": lambda: mountinfo,
            },
            {
                "supplementary_groups": (),
                "status_reader": lambda: status,
                "mountinfo_reader": lambda: mountinfo.replace(
                    "size=1048576k", "size=1024k"
                ),
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values), self.assertRaises(
                runner.V2IntegrityStop
            ):
                runner.validate_foundation_preflight_namespace_boundary(
                    parent_mount_namespace="mnt:[51001]",
                    parent_network_namespace="net:[51002]",
                    environment=environment,
                    effective_uid=1000,
                    effective_gid=1000,
                    namespace_reader=namespace_reader,
                    validate_paths=lambda: {
                        key: value
                        for key, value in path_facts().items()
                    },
                    **values,
                )

    def test_live_parent_temporary_directory_states_and_thread_setters_are_exact(
        self,
    ) -> None:
        environment = runner._frozen_parent_environment()
        inventory = "\n".join(
            (
                f"{runner.ASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5080, "
                "00000000:01:00.0, 12.0, 16303, 14336",
                f"{runner.UNASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5070, "
                "00000000:05:00.0, 12.0, 12227, 12000",
            )
        )

        def text_reader(path: Path) -> str:
            return {
                "/proc/net/route": "Iface\tDestination\n",
                "/proc/net/ipv6_route": "0 0 0 0 0 0 0 0 0 lo\n",
                "/sys/class/net/lo/operstate": "unknown\n",
            }[str(path)]

        def make_torch() -> types.SimpleNamespace:
            return types.SimpleNamespace(
                __version__=runner.EXPECTED_TORCH_VERSION,
                version=types.SimpleNamespace(cuda=runner.EXPECTED_CUDA_VERSION),
                backends=types.SimpleNamespace(
                    cudnn=types.SimpleNamespace(
                        version=mock.Mock(
                            return_value=runner.EXPECTED_CUDNN_RUNTIME
                        )
                    )
                ),
                cuda=types.SimpleNamespace(
                    is_available=mock.Mock(return_value=True),
                    device_count=mock.Mock(return_value=1),
                    set_device=mock.Mock(),
                    current_device=mock.Mock(return_value=0),
                    get_device_name=mock.Mock(
                        return_value="NVIDIA GeForce RTX 5080"
                    ),
                ),
                set_num_threads=mock.Mock(),
                set_num_interop_threads=mock.Mock(),
                get_num_threads=mock.Mock(return_value=2),
                get_num_interop_threads=mock.Mock(return_value=1),
            )

        def validate(
            state: object,
            *,
            owned: mock.Mock,
            absent: mock.Mock,
            torch_module: types.SimpleNamespace,
        ) -> dict[str, object]:
            return runner.validate_live_parent_boundary(
                environment=environment,
                expected_environment=environment,
                executable=runner.ANGLER_PYTHON,
                runner_path=runner.RUNNER_PATH,
                working_directory=runner.REPOSITORY_ROOT,
                interface_inventory=lambda: ((1, "lo"),),
                text_reader=text_reader,
                namespace_reader=lambda _path: "net:[41002]",
                gpu_inventory_runner=lambda *_args, **_kwargs: types.SimpleNamespace(
                    stdout=inventory
                ),
                hostname_reader=lambda: runner.EXPECTED_HOSTNAME,
                python_version_reader=lambda: runner.EXPECTED_PYTHON_VERSION,
                distribution_reader=lambda name: dict(
                    runner.EXPECTED_PARENT_DISTRIBUTIONS
                )[name],
                torch_module=torch_module,
                temporary_directory_state=state,  # type: ignore[arg-type]
                activate_cuda_runtime=(state == "owned-empty"),
                inherited_worker_parent_attestor=(
                    lambda expected, **_kwargs: (
                        _synthetic_inherited_parent_attestation(str(expected))
                    )
                ),
            )

        phase_observations: dict[str, dict[str, object]] = {}
        for state in ("absent", "owned-empty"):
            with self.subTest(state=state):
                owned = mock.Mock()
                absent = mock.Mock()
                torch_module = make_torch()
                with (
                    mock.patch.object(
                        runner, "_require_owned_directory", owned
                    ),
                    mock.patch.object(runner, "_require_absent", absent),
                ):
                    observed = validate(
                        state,
                        owned=owned,
                        absent=absent,
                        torch_module=torch_module,
                    )
                phase_observations[state] = observed
                self.assertEqual(
                    observed["temporary_directory"], environment["TMPDIR"]
                )
                if state == "absent":
                    owned.assert_called_once_with(
                        runner.SCRATCH_ROOT.parent,
                        expected_mode=0o700,
                        label="live scratch parent",
                    )
                    self.assertEqual(
                        absent.call_args_list,
                        [
                            mock.call(
                                runner.SCRATCH_ROOT, "live scratch root"
                            ),
                            mock.call(
                                Path(environment["TMPDIR"]),
                                "live temporary directory",
                            ),
                            mock.call(
                                runner.SCRATCH_ROOT,
                                "post-check live scratch root",
                            ),
                            mock.call(
                                Path(environment["TMPDIR"]),
                                "post-check live temporary directory",
                            ),
                        ],
                    )
                else:
                    self.assertEqual(
                        owned.call_args_list,
                        [
                            mock.call(
                                Path(environment["TMPDIR"]),
                                expected_mode=0o700,
                                empty=True,
                                label="live temporary directory",
                            ),
                            mock.call(
                                Path(environment["TMPDIR"]),
                                expected_mode=0o700,
                                empty=True,
                                label="post-check live temporary directory",
                            ),
                        ],
                    )
                    absent.assert_not_called()
                torch_module.set_num_threads.assert_not_called()
                torch_module.set_num_interop_threads.assert_not_called()
                self.assertEqual(torch_module.get_num_threads.call_count, 2)
                self.assertEqual(
                    torch_module.get_num_interop_threads.call_count, 2
                )
                torch_module.cuda.device_count.assert_called_once_with()
                if state == "absent":
                    torch_module.backends.cudnn.version.assert_not_called()
                    torch_module.cuda.is_available.assert_not_called()
                    torch_module.cuda.set_device.assert_not_called()
                    torch_module.cuda.current_device.assert_not_called()
                    torch_module.cuda.get_device_name.assert_not_called()
                else:
                    torch_module.backends.cudnn.version.assert_called_once_with()
                    torch_module.cuda.is_available.assert_called_once_with()
                    torch_module.cuda.set_device.assert_called_once_with(0)
                    torch_module.cuda.current_device.assert_called_once_with()
                    torch_module.cuda.get_device_name.assert_called_once_with(0)

        runner.validate_live_parent_boundary_rejoin(
            phase_observations["absent"],
            phase_observations["owned-empty"],
        )
        changed = dict(phase_observations["owned-empty"])
        changed["hostname"] = "changed-host"
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "live parent changed across CUDA activation",
        ):
            runner.validate_live_parent_boundary_rejoin(
                phase_observations["absent"], changed
            )

        owned = mock.Mock()
        absent = mock.Mock()
        torch_module = make_torch()
        torch_module.get_num_threads.side_effect = (4, 2)
        torch_module.get_num_interop_threads.side_effect = (2, 1)
        with (
            mock.patch.object(runner, "_require_owned_directory", owned),
            mock.patch.object(runner, "_require_absent", absent),
        ):
            validate(
                "owned-empty",
                owned=owned,
                absent=absent,
                torch_module=torch_module,
            )
        torch_module.set_num_threads.assert_called_once_with(2)
        torch_module.set_num_interop_threads.assert_called_once_with(1)
        self.assertEqual(torch_module.get_num_threads.call_count, 2)
        self.assertEqual(torch_module.get_num_interop_threads.call_count, 2)

        owned = mock.Mock()
        absent = mock.Mock()
        with (
            mock.patch.object(runner, "_require_owned_directory", owned),
            mock.patch.object(runner, "_require_absent", absent),
            self.assertRaisesRegex(
                ValueError,
                "live temporary-directory state differs",
            ),
        ):
            validate(
                "unchecked",
                owned=owned,
                absent=absent,
                torch_module=make_torch(),
            )
        owned.assert_not_called()
        absent.assert_not_called()

    def test_parent_preload_memory_floor_and_postload_probe_are_distinct(self) -> None:
        inventory = "\n".join(
            (
                f"{runner.ASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5080, "
                "00000000:01:00.0, 12.0, 16303, 14335",
                f"{runner.UNASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5070, "
                "00000000:05:00.0, 12.0, 12227, 12000",
            )
        )

        def text_reader(path: Path) -> str:
            return {
                "/proc/net/route": "Iface\tDestination\n",
                "/proc/net/ipv6_route": "0 0 0 0 0 0 0 0 0 lo\n",
                "/sys/class/net/lo/operstate": "unknown\n",
            }[str(path)]

        with (
            mock.patch.object(runner, "_require_owned_directory"),
            self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "free-memory floor is not met",
            ),
        ):
            runner.validate_live_parent_boundary(
                environment=runner._frozen_parent_environment(),
                expected_environment=runner._frozen_parent_environment(),
                executable=runner.ANGLER_PYTHON,
                runner_path=runner.RUNNER_PATH,
                working_directory=runner.REPOSITORY_ROOT,
                interface_inventory=lambda: ((1, "lo"),),
                text_reader=text_reader,
                namespace_reader=lambda _path: "net:[41002]",
                gpu_inventory_runner=lambda *_args, **_kwargs: types.SimpleNamespace(
                    stdout=inventory
                ),
                hostname_reader=lambda: runner.EXPECTED_HOSTNAME,
                python_version_reader=lambda: runner.EXPECTED_PYTHON_VERSION,
                distribution_reader=lambda name: dict(
                    runner.EXPECTED_PARENT_DISTRIBUTIONS
                )[name],
                torch_module=object(),
                inherited_worker_parent_attestor=(
                    lambda expected, **_kwargs: (
                        _synthetic_inherited_parent_attestation(str(expected))
                    )
                ),
            )

        postload_inventory = inventory.replace("14335", "4200")

        class Cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def current_device() -> int:
                return 0

            @staticmethod
            def get_device_name(_index: int) -> str:
                return "NVIDIA GeForce RTX 5080"

        inventory_runner = mock.Mock(
            return_value=types.SimpleNamespace(stdout=postload_inventory)
        )
        process_runner = mock.Mock(
            return_value=types.SimpleNamespace(
                stdout=f"{runner.ASSIGNED_GPU_UUID}, {os.getpid()}\n"
            )
        )
        for _ordinal in range(2):
            observed = runner._foundation_preflight_gpu_probe(
                gpu_inventory_runner=inventory_runner,
                gpu_process_runner=process_runner,
                torch_module=types.SimpleNamespace(cuda=Cuda()),
            )
            self.assertEqual(observed["assigned_uuid"], runner.ASSIGNED_GPU_UUID)
            self.assertEqual(observed["unassigned_compute_processes"], 0)
        self.assertEqual(inventory_runner.call_count, 2)
        self.assertEqual(process_runner.call_count, 2)

        for label, row in (
            (
                "foreign-assigned",
                f"{runner.ASSIGNED_GPU_UUID}, {os.getpid() + 1}\n",
            ),
            ("unassigned", f"{runner.UNASSIGNED_GPU_UUID}, 4242\n"),
        ):
            process_runner.return_value = types.SimpleNamespace(stdout=row)
            with self.subTest(label=label), self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "GPU ownership differs",
            ):
                runner._foundation_preflight_gpu_probe(
                    gpu_inventory_runner=inventory_runner,
                    gpu_process_runner=process_runner,
                    torch_module=types.SimpleNamespace(cuda=Cuda()),
                )

    def test_injected_preflight_uses_first_three_scans_and_closes_owner(self) -> None:
        sources = self._sources()
        source_ref = runner.canonical_ref("source-inventory", sources)
        boundary = _synthetic_preflight_boundary(source_ref)
        host_closure = _synthetic_preflight_host_closure(source_ref)
        torch_module = _MirrorFakeTorch()
        model, digest = _mirror_tiny_fixture(torch_module)
        guard = _MirrorFoundationGuard(digest)
        resources = runner.ResourceLedger(runner.RunMode.QUALIFICATION)
        owner_box: dict[str, object] = {}
        mirror_box: dict[str, object] = {}
        calls = Counter()

        def load_sources() -> object:
            calls["sources"] += 1
            return sources

        def validate_boundary() -> dict[str, object]:
            calls["boundary"] += 1
            return boundary

        def sample_resources() -> dict[str, int]:
            calls["resources"] += 1
            return {
                "configured_pools": 0,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
                "process_count": 0,
                "rss_bytes": 48,
                "state_scratch_bytes": 0,
            }

        def probe_gpu() -> dict[str, object]:
            calls["gpu"] += 1
            return {
                "allocated_bytes": 0,
                "logical_device_count": 1,
                "logical_device_index": 0,
                "name": "NVIDIA GeForce RTX 5080",
                "reserved_bytes": 0,
                "uuid": runner.ASSIGNED_GPU_UUID,
            }

        def owner_factory(**values: object) -> runner.V2FoundationOwner:
            calls["owner"] += 1
            self.assertIs(values["resources"], resources)
            self.assertEqual(values["expected_source_inventory_ref"], source_ref)
            self.assertEqual(values["verify_source_inventory"](), source_ref)
            self.assertEqual(values["validate_boundary"](), boundary)
            self.assertEqual(values["sample_resources"](), sample_resources())
            self.assertEqual(values["probe_gpu"](), probe_gpu())
            mirror = runner._ExactFoundationByteMirror(
                model,
                foundation_guard=guard,
                expected_digest=digest,
                resources=resources,
            )
            # Guard construction is the first measured scan; the coordinator
            # must perform exactly two more and may not discard this sample.
            mirror.verify_exact()
            owner = runner.V2FoundationOwner(
                loaded=object(),
                guard=None,
                _mirror=mirror,
                helpers={},
                _construction_capability=(
                    runner._V2_FOUNDATION_OWNER_CONSTRUCTION_CAPABILITY
                ),
            )
            owner_box["value"] = owner
            mirror_box["value"] = mirror
            return owner

        with (
            _mirror_fake_boundary(torch_module),
            mock.patch.object(
                runner.V2FoundationOwner,
                "_release_cuda",
                return_value=None,
            ),
            mock.patch.object(
                runner.time,
                "perf_counter_ns",
                side_effect=(100, 110, 200, 220, 300, 330),
            ),
        ):
            boundary = _synthetic_preflight_boundary(source_ref)
            receipt = runner.run_foundation_mirror_preflight_injected(
                validate_boundary=validate_boundary,
                host_closure=host_closure,
                sample_resources=sample_resources,
                probe_gpu=probe_gpu,
                source_inventory_loader=load_sources,
                owner_factory=owner_factory,
                resources=resources,
            )
            self.assertEqual(
                runner.validate_foundation_mirror_preflight_receipt(
                    receipt,
                    source_inventory=sources,
                    injected_root=True,
                ),
                receipt,
            )
            self.assertEqual(receipt["verification_durations_ns"], [10, 20, 30])
            self.assertEqual(receipt["maximum_duration_ns"], 30)
            self.assertEqual(
                mirror_box["value"]._verification_durations_ns,
                [10, 20, 30],
            )
            self.assertTrue(mirror_box["value"]._closed)
            self.assertTrue(owner_box["value"]._closed)
            self.assertEqual(torch_module.current_scan, 4)
            self.assertEqual(torch_module.model_stream_records, 4)
            self.assertEqual(torch_module.copy_stream_waits, 4)
            self.assertEqual(torch_module.global_synchronizations, 0)
            self.assertEqual(torch_module.allocator, torch_module.allocator_baseline)
            self.assertEqual(calls["owner"], 1)
            self.assertEqual(calls["sources"], 4)

    def test_injected_preflight_failures_still_close_the_exact_owner(self) -> None:
        for failure in ("mirror", "source"):
            with self.subTest(failure=failure):
                sources = self._sources()
                changed_sources = json.loads(json.dumps(sources))
                changed_sources[0]["sha256"] = _raw("preflight-drift")
                source_ref = runner.canonical_ref("source-inventory", sources)
                boundary: dict[str, object] = {}
                host_closure = _synthetic_preflight_host_closure(source_ref)
                torch_module = _MirrorFakeTorch()
                model, digest = _mirror_tiny_fixture(torch_module)
                guard = _MirrorFoundationGuard(digest)
                resources = runner.ResourceLedger(runner.RunMode.QUALIFICATION)
                owner_box: dict[str, object] = {}
                mirror_box: dict[str, object] = {}
                source_calls = 0

                def load_sources() -> object:
                    nonlocal source_calls
                    source_calls += 1
                    if failure == "source" and source_calls >= 2:
                        return changed_sources
                    return sources

                def owner_factory(**_values: object) -> runner.V2FoundationOwner:
                    mirror = runner._ExactFoundationByteMirror(
                        model,
                        foundation_guard=guard,
                        expected_digest=digest,
                        resources=resources,
                    )
                    mirror.verify_exact()
                    if failure == "mirror":
                        model.rows[0][1].data.write_bytes(
                            0,
                            struct.pack("<I", 0x7FC00002),
                        )
                    owner = runner.V2FoundationOwner(
                        loaded=object(),
                        guard=None,
                        _mirror=mirror,
                        helpers={},
                        _construction_capability=(
                            runner._V2_FOUNDATION_OWNER_CONSTRUCTION_CAPABILITY
                        ),
                    )
                    owner_box["value"] = owner
                    mirror_box["value"] = mirror
                    return owner

                with (
                    _mirror_fake_boundary(torch_module),
                    mock.patch.object(
                        runner.V2FoundationOwner,
                        "_release_cuda",
                        return_value=None,
                    ),
                    mock.patch.object(
                        runner.time,
                        "perf_counter_ns",
                        side_effect=(100, 110, 200, 220, 300, 330),
                    ),
                    self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "FOUNDATION_PREFLIGHT_FAILED",
                    ),
                ):
                    boundary = _synthetic_preflight_boundary(source_ref)
                    runner.run_foundation_mirror_preflight_injected(
                        validate_boundary=lambda: boundary,
                        host_closure=host_closure,
                        sample_resources=lambda: {},
                        probe_gpu=lambda: {},
                        source_inventory_loader=load_sources,
                        owner_factory=owner_factory,
                        resources=resources,
                    )
                self.assertTrue(owner_box["value"]._closed)
                self.assertTrue(mirror_box["value"]._closed)
                self.assertEqual(
                    torch_module.allocator,
                    torch_module.allocator_baseline,
                )
                self.assertEqual(torch_module.global_synchronizations, 0)


class FoundationMirrorPreflightLauncherTests(unittest.TestCase):
    @staticmethod
    def _sources() -> list[dict[str, object]]:
        return FoundationMirrorPreflightGlueTests._sources()

    @staticmethod
    def _scratch_metadata(*, inode: int = 41) -> os.stat_result:
        return os.stat_result(
            (
                stat.S_IFDIR | 0o700,
                inode,
                7,
                1,
                os.geteuid(),
                os.getegid(),
                0,
                0,
                0,
                0,
            )
        )

    def test_exact_argv_mocked_mount_tree_and_host_closure(self) -> None:
        root_environment = runner._frozen_foundation_preflight_root_environment()
        self.assertEqual(
            runner._foundation_preflight_host_argv(),
            (
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/unshare",
                "--mount",
                "--net",
                "--fork",
                "--kill-child=SIGKILL",
                "--propagation",
                "private",
                "--",
                "/usr/bin/env",
                "-i",
                *(
                    f"{name}={root_environment[name]}"
                    for name in sorted(root_environment)
                ),
                str(runner.ANGLER_PYTHON),
                str(runner.RUNNER_PATH),
                "--foundation-mirror-preflight-root",
            ),
        )
        child_environment = runner._frozen_foundation_preflight_environment()
        self.assertEqual(
            child_environment["HF_HUB_DISABLE_PROGRESS_BARS"], "1"
        )
        child_argv = runner._foundation_preflight_child_argv(
            parent_mount_namespace="mnt:[51001]",
            parent_network_namespace="net:[51002]",
        )
        self.assertEqual(
            child_argv,
            (
                "/usr/bin/setpriv",
                "--reuid=1000",
                "--regid=1000",
                "--clear-groups",
                "--inh-caps=-all",
                "--ambient-caps=-all",
                "--bounding-set=-all",
                "--no-new-privs",
                "--pdeathsig=SIGKILL",
                "/usr/bin/env",
                "-i",
                *(
                    f"{name}={child_environment[name]}"
                    for name in sorted(child_environment)
                ),
                str(runner.ANGLER_PYTHON),
                str(runner.RUNNER_PATH),
                "--foundation-mirror-preflight-child",
                "mnt:[51001]",
                "net:[51002]",
            ),
        )
        self.assertNotIn("-c", runner._foundation_preflight_host_argv())
        self.assertNotIn("-c", child_argv)

        class MountCall:
            def __init__(self) -> None:
                self.argtypes: object = None
                self.restype: object = None
                self.calls: list[tuple[object, ...]] = []

            def __call__(self, *values: object) -> int:
                self.calls.append(values)
                return 0

        mount = MountCall()
        with mock.patch.object(
            runner.ctypes,
            "CDLL",
            return_value=types.SimpleNamespace(mount=mount),
        ):
            runner._foundation_preflight_mount_tmpfs()
        self.assertEqual(
            mount.calls,
            [
                (
                    b"tmpfs",
                    b"/opt/angler/scratch",
                    b"tmpfs",
                    2 | 4 | 8,
                    b"size=1073741824,mode=0700,uid=1000,gid=1000",
                )
            ],
        )

        with (
            mock.patch.object(runner.os, "mkdir") as mkdir,
            mock.patch.object(runner.os, "chown") as chown,
            mock.patch.object(runner.os, "chmod") as chmod,
        ):
            runner._foundation_preflight_make_private_tree()
        private_paths = (
            runner.FOUNDATION_PREFLIGHT_PRIVATE_ROOT,
            runner.FOUNDATION_PREFLIGHT_TMP,
            runner.FOUNDATION_PREFLIGHT_CACHE,
        )
        self.assertEqual(
            mkdir.call_args_list,
            [mock.call(path, mode=0o700) for path in private_paths],
        )
        self.assertEqual(
            chown.call_args_list,
            [mock.call(path, 1000, 1000) for path in private_paths],
        )
        self.assertEqual(
            chmod.call_args_list,
            [mock.call(path, 0o700) for path in private_paths],
        )

        source_ref = runner.canonical_ref("source-inventory", self._sources())
        closure = runner._foundation_preflight_host_closure(
            scratch_metadata=self._scratch_metadata(),
            source_inventory_ref=source_ref,
        )
        self.assertEqual(
            runner.validate_foundation_preflight_host_closure(
                closure,
                expected_source_inventory_ref=source_ref,
            ),
            closure,
        )

    def test_root_revalidates_sources_as_uid_gid_1000(self) -> None:
        class SnapshotReached(Exception):
            pass

        class Child:
            returncode = 0

            def __init__(self) -> None:
                self.stdout = types.SimpleNamespace(close=mock.Mock())
                self.stderr = types.SimpleNamespace(close=mock.Mock())

        base_descriptor = 91
        metadata = os.stat_result(
            (
                stat.S_IFDIR | 0o700,
                41,
                7,
                1,
                1000,
                1000,
                0,
                0,
                0,
                0,
            )
        )
        real_stat = os.stat

        def missing_private_paths(
            path: object,
            *args: object,
            **kwargs: object,
        ) -> os.stat_result:
            if kwargs.get("dir_fd") == base_descriptor:
                raise FileNotFoundError(str(path))
            return real_stat(path, *args, **kwargs)

        source_snapshot = mock.Mock(return_value=self._sources())
        with (
            mock.patch.object(runner.os, "geteuid", return_value=0),
            mock.patch.object(runner.os, "getegid", return_value=0),
            mock.patch.object(
                runner.os,
                "environ",
                runner._frozen_foundation_preflight_root_environment(),
            ),
            mock.patch.object(runner.Path, "cwd", return_value=runner.REPOSITORY_ROOT),
            mock.patch.object(runner.sys, "executable", str(runner.ANGLER_PYTHON)),
            mock.patch.object(
                runner.os,
                "readlink",
                side_effect=lambda path: {
                    "/proc/self/ns/mnt": "mnt:[41001]",
                    "/proc/1/ns/mnt": "mnt:[51001]",
                    "/proc/self/ns/net": "net:[41002]",
                    "/proc/1/ns/net": "net:[51002]",
                }[str(path)],
            ),
            mock.patch.object(runner.os, "open", return_value=base_descriptor),
            mock.patch.object(runner.os, "fstat", return_value=metadata),
            mock.patch.object(runner.os, "stat", side_effect=missing_private_paths),
            mock.patch.object(runner.os, "close"),
            mock.patch.object(runner, "_foundation_preflight_mount_tmpfs"),
            mock.patch.object(runner, "_foundation_preflight_make_private_tree"),
            mock.patch.object(
                runner.subprocess,
                "run",
                return_value=types.SimpleNamespace(stdout=b"", stderr=b""),
            ),
            mock.patch.object(runner.subprocess, "Popen", return_value=Child()),
            mock.patch.object(
                runner,
                "_read_foundation_preflight_child_process",
                return_value=(b"{}\n", b""),
            ),
            mock.patch.object(
                runner,
                "_source_inventory_snapshot",
                source_snapshot,
            ),
            mock.patch.object(
                runner,
                "validate_foundation_mirror_child_evidence",
                side_effect=SnapshotReached,
            ),
            self.assertRaises(SnapshotReached),
        ):
            runner.foundation_mirror_preflight_root_main()
        source_snapshot.assert_called_once_with(
            expected_owner_uid=1000,
            expected_owner_gid=1000,
        )

    def test_live_launcher_accepts_one_canonical_frame_and_rechecks_host(self) -> None:
        sources = self._sources()
        receipt = _synthetic_preflight_receipt(sources)
        stdout = runner.canonical_json_bytes(receipt) + b"\n"
        scratch = self._scratch_metadata()
        captured: list[tuple[tuple[str, ...], dict[str, object]]] = []

        class Process:
            def __init__(self, argv: tuple[str, ...]) -> None:
                self.args = argv
                self.returncode = 0
                self.pid = 4242
                self.stdout = types.SimpleNamespace(close=mock.Mock())
                self.stderr = types.SimpleNamespace(close=mock.Mock())

            def poll(self) -> int:
                return self.returncode

        def process_factory(
            argv: tuple[str, ...], **kwargs: object
        ) -> Process:
            captured.append((argv, kwargs))
            return Process(argv)

        def scratch_metadata(_path: object) -> os.stat_result:
            return scratch

        def namespace_reader(path: str | Path) -> str:
            return {
                "/proc/self/ns/mnt": "mnt:[51001]",
                "/proc/self/ns/net": "net:[51002]",
            }[str(path)]

        with (
            mock.patch.object(
                runner,
                "_source_inventory_snapshot",
                side_effect=(sources, sources),
            ) as source_snapshot,
            mock.patch.object(
                runner.os,
                "lstat",
                side_effect=scratch_metadata,
            ),
            mock.patch.object(
                runner.os,
                "readlink",
                side_effect=namespace_reader,
            ),
            mock.patch.object(
                runner.Path,
                "resolve",
                autospec=True,
                side_effect=lambda path, strict=False: path,
            ),
            mock.patch.object(runner, "_require_absent") as require_absent,
            mock.patch.object(
                runner, "assert_all_live_paths_absent"
            ) as live_absent,
            mock.patch.object(
                runner,
                "_read_foundation_preflight_process",
                return_value=(stdout, b""),
            ),
        ):
            observed = runner.run_foundation_mirror_preflight_live(
                process_factory=process_factory
            )
        self.assertEqual(observed, receipt)
        self.assertEqual(source_snapshot.call_count, 2)
        self.assertEqual(require_absent.call_count, 3)
        self.assertEqual(live_absent.call_count, 2)
        self.assertEqual(len(captured), 1)
        argv, kwargs = captured[0]
        self.assertEqual(argv, runner._foundation_preflight_host_argv())
        self.assertNotIn("shell", kwargs)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], runner.subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], runner.subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], runner.subprocess.PIPE)

    def test_live_launcher_rejects_frames_failures_and_postcheck_drift(self) -> None:
        sources = self._sources()
        changed_sources = json.loads(json.dumps(sources))
        changed_sources[0]["sha256"] = _raw("launcher-source-drift")
        receipt = _synthetic_preflight_receipt(sources)
        canonical = runner.canonical_json_bytes(receipt) + b"\n"
        scratch = self._scratch_metadata()

        class Process:
            def __init__(self, argv: tuple[str, ...], returncode: int | None) -> None:
                self.args = argv
                self.returncode = returncode
                self.pid = 4242
                self.stdout = types.SimpleNamespace(close=mock.Mock())
                self.stderr = types.SimpleNamespace(close=mock.Mock())

            def poll(self) -> int | None:
                return self.returncode

        def invoke(
            *,
            frame: bytes = canonical,
            returncode: int | None = 0,
            read_error: BaseException | None = None,
            sources_after: object = sources,
            scratch_after: os.stat_result = scratch,
            teardown: mock.Mock | None = None,
        ) -> None:
            process = Process(
                runner._foundation_preflight_host_argv(), returncode
            )
            teardown_callback = mock.Mock() if teardown is None else teardown
            read = mock.Mock(
                side_effect=read_error,
                return_value=(frame, b""),
            )
            lstat_calls = 0

            def scratch_metadata(_path: object) -> os.stat_result:
                nonlocal lstat_calls
                lstat_calls += 1
                return scratch if lstat_calls <= 2 else scratch_after

            def namespace_reader(path: str | Path) -> str:
                return {
                    "/proc/self/ns/mnt": "mnt:[51001]",
                    "/proc/self/ns/net": "net:[51002]",
                }[str(path)]

            with (
                mock.patch.object(
                    runner,
                    "_source_inventory_snapshot",
                    side_effect=(sources, sources_after),
                ),
                mock.patch.object(
                    runner.os,
                    "lstat",
                    side_effect=scratch_metadata,
                ),
                mock.patch.object(
                    runner.os,
                    "readlink",
                    side_effect=namespace_reader,
                ),
                mock.patch.object(
                    runner.Path,
                    "resolve",
                    autospec=True,
                    side_effect=lambda path, strict=False: path,
                ),
                mock.patch.object(runner, "_require_absent"),
                mock.patch.object(runner, "assert_all_live_paths_absent"),
                mock.patch.object(
                    runner,
                    "_read_foundation_preflight_process",
                    read,
                ),
                mock.patch.object(
                    runner,
                    "_terminate_foundation_preflight_process_group",
                    teardown_callback,
                ),
            ):
                runner.run_foundation_mirror_preflight_live(
                    process_factory=lambda *_args, **_kwargs: process
                )

        bad_cases = (
            ("noncanonical", b" " + canonical, 0, None),
            ("multiple-frames", canonical + canonical, 0, None),
            ("nonzero", canonical, 7, None),
        )
        for label, frame, returncode, read_error in bad_cases:
            with self.subTest(label=label), self.assertRaises(
                runner.V2IntegrityStop
            ):
                invoke(
                    frame=frame,
                    returncode=returncode,
                    read_error=read_error,
                )

        timeout_teardown = mock.Mock()
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "FOUNDATION_PREFLIGHT_LAUNCH_FAILED",
        ):
            invoke(
                returncode=None,
                read_error=TimeoutError("synthetic timeout"),
                teardown=timeout_teardown,
            )
        timeout_teardown.assert_called_once()

        for label, values in (
            ("source", {"sources_after": changed_sources}),
            (
                "scratch",
                {"scratch_after": self._scratch_metadata(inode=42)},
            ),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "host closure changed",
            ):
                invoke(**values)

    def test_bounded_pipe_read_and_terminal_teardown_are_fail_closed(self) -> None:
        read_descriptor, write_descriptor = os.pipe()
        error_read, error_write = os.pipe()
        stdout = os.fdopen(read_descriptor, "rb", closefd=True)
        stderr = os.fdopen(error_read, "rb", closefd=True)
        self.addCleanup(stdout.close)
        self.addCleanup(stderr.close)
        os.write(write_descriptor, b"123456789")
        os.close(write_descriptor)
        os.close(error_write)

        class PipeProcess:
            pid = 4242
            args = runner._foundation_preflight_host_argv()
            returncode: int | None = None

            def __init__(self) -> None:
                self.stdout = stdout
                self.stderr = stderr

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout: float) -> int:
                self.returncode = 0
                return 0

        process = PipeProcess()
        with (
            mock.patch.object(
                runner,
                "FOUNDATION_PREFLIGHT_MAXIMUM_OUTPUT_BYTES",
                8,
            ),
            mock.patch.object(
                runner,
                "_terminate_foundation_preflight_process_group",
            ) as teardown,
            self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "FOUNDATION_PREFLIGHT_PROCESS_FAILED",
            ),
        ):
            runner._read_foundation_preflight_process(process)
        teardown.assert_called_once_with(process)

        class SlowProcess:
            pid = 4343
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout: float) -> int:
                raise AssertionError("terminal process must not be waited twice")

        slow = SlowProcess()
        def fake_killpg(_pid: int, requested: int) -> None:
            if requested == signal.SIGKILL:
                slow.returncode = -signal.SIGKILL
            elif requested == 0:
                raise ProcessLookupError

        with (
            mock.patch.object(
                runner.os,
                "killpg",
                side_effect=fake_killpg,
            ) as killpg,
            mock.patch.object(
                runner.time,
                "monotonic",
                side_effect=(0.0, 6.0, 10.0, 11.0),
            ),
            mock.patch.object(runner.time, "sleep"),
        ):
            runner._terminate_foundation_preflight_process_group(slow)
        self.assertEqual(
            [call.args for call in killpg.call_args_list if call.args[1] != 0],
            [
                (slow.pid, signal.SIGTERM),
                (slow.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(slow.returncode, -signal.SIGKILL)

    def test_outer_namespace_rejoin_binds_exact_host_mount_and_network(self) -> None:
        receipt = _synthetic_preflight_receipt(self._sources())
        host_namespaces = {
            "/proc/self/ns/mnt": "mnt:[51001]",
            "/proc/self/ns/net": "net:[51002]",
        }

        def reader(path: Path) -> str:
            return host_namespaces[str(path)]

        self.assertEqual(
            runner._rejoin_foundation_preflight_outer_namespaces(
                receipt,
                namespace_reader=reader,
            ),
            receipt,
        )
        for label, changed in (
            ("mount", {"/proc/self/ns/mnt": "mnt:[61001]"}),
            ("network", {"/proc/self/ns/net": "net:[61002]"}),
        ):
            mismatched = {**host_namespaces, **changed}
            with self.subTest(label=label), self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "outer namespace binding differs",
            ):
                runner._rejoin_foundation_preflight_outer_namespaces(
                    receipt,
                    namespace_reader=lambda path, values=mismatched: values[
                        str(path)
                    ],
                )


class LiveBoundaryAndPathTests(unittest.TestCase):
    def test_live_path_absence_accepts_posixpath_and_rejects_invalid_inventory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            absent = (root / "absent-a", root / "absent-b")
            self.assertTrue(all(isinstance(path, Path) for path in absent))
            self.assertEqual(type(absent[0]).__name__, "PosixPath")
            runner.assert_all_live_paths_absent(absent)

            with self.assertRaisesRegex(
                ValueError,
                "non-absolute path",
            ):
                runner.assert_all_live_paths_absent((str(absent[0]),))  # type: ignore[arg-type]
            with self.assertRaisesRegex(
                ValueError,
                "non-absolute path",
            ):
                runner.assert_all_live_paths_absent((Path("relative-live-path"),))

            existing = root / "existing"
            existing.write_bytes(b"already consumed")
            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "one-use live path already exists",
            ):
                runner.assert_all_live_paths_absent((existing,))

    def test_state_scratch_measurement_is_bounded_and_never_follows_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state_root = root / "state"
            scratch_root = root / "scratch"
            state_root.mkdir(mode=0o700)
            scratch_root.mkdir(mode=0o700)
            state_root.chmod(0o700)
            scratch_root.chmod(0o700)
            state_file = state_root / "state.bin"
            state_file.write_bytes(b"abc")
            state_file.chmod(0o600)
            nested = scratch_root / "nested"
            nested.mkdir(mode=0o700)
            nested.chmod(0o700)
            scratch_file = nested / "scratch.bin"
            scratch_file.write_bytes(b"12345")
            scratch_file.chmod(0o600)

            self.assertEqual(
                runner._bounded_no_follow_state_scratch_bytes(
                    state_root,
                    scratch_root,
                    maximum_bytes=8,
                    maximum_entries=5,
                ),
                8,
            )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "byte ceiling exceeded",
            ):
                runner._bounded_no_follow_state_scratch_bytes(
                    state_root,
                    scratch_root,
                    maximum_bytes=7,
                    maximum_entries=5,
                )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "entry ceiling exceeded",
            ):
                runner._bounded_no_follow_state_scratch_bytes(
                    state_root,
                    scratch_root,
                    maximum_bytes=8,
                    maximum_entries=4,
                )

            outside = root / "outside.bin"
            outside.write_bytes(b"outside-must-never-be-counted")
            outside.chmod(0o600)
            link = scratch_root / "outside-link"
            link.symlink_to(outside)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "metadata differs",
            ):
                runner._bounded_no_follow_state_scratch_bytes(
                    state_root,
                    scratch_root,
                    maximum_bytes=64,
                    maximum_entries=8,
                )
            self.assertEqual(outside.read_bytes(), b"outside-must-never-be-counted")

    def test_live_gpu_probe_rejects_any_rtx5070_compute_activity(self) -> None:
        inventory = "\n".join(
            (
                f"{runner.ASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5080, "
                "00000000:01:00.0, 12.0",
                f"{runner.UNASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5070, "
                "00000000:05:00.0, 12.0",
            )
        )

        class Cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def current_device() -> int:
                return 0

            @staticmethod
            def get_device_name(_index: int) -> str:
                return "NVIDIA GeForce RTX 5080"

            @staticmethod
            def memory_allocated(_index: int) -> int:
                return 13

            @staticmethod
            def memory_reserved(_index: int) -> int:
                return 17

        torch_module = types.SimpleNamespace(
            cuda=Cuda(),
            get_num_threads=lambda: 2,
            get_num_interop_threads=lambda: 1,
        )
        inventory_runner = mock.Mock(
            return_value=types.SimpleNamespace(stdout=inventory)
        )
        process_runner = mock.Mock(
            return_value=types.SimpleNamespace(stdout="")
        )
        environment = {"CUDA_VISIBLE_DEVICES": runner.ASSIGNED_GPU_UUID}
        self.assertEqual(
            runner._live_gpu_probe(
                torch_module=torch_module,
                gpu_inventory_runner=inventory_runner,
                gpu_process_runner=process_runner,
                environment=environment,
            )["unassigned_compute_processes"],
            0,
        )
        process_runner.return_value = types.SimpleNamespace(
            stdout=f"{runner.ASSIGNED_GPU_UUID}, {os.getpid()}\n"
        )
        self.assertEqual(
            runner._live_gpu_probe(
                torch_module=torch_module,
                gpu_inventory_runner=inventory_runner,
                gpu_process_runner=process_runner,
                environment=environment,
            )["assigned_uuid"],
            runner.ASSIGNED_GPU_UUID,
        )
        process_runner.return_value = types.SimpleNamespace(
            stdout=f"{runner.ASSIGNED_GPU_UUID}, {os.getpid() + 1000}\n"
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "another compute-process owner",
        ):
            runner._live_gpu_probe(
                torch_module=torch_module,
                gpu_inventory_runner=inventory_runner,
                gpu_process_runner=process_runner,
                environment=environment,
            )
        process_runner.return_value = types.SimpleNamespace(
            stdout=(
                f"{runner.ASSIGNED_GPU_UUID}, {os.getpid()}\n"
                f"{runner.ASSIGNED_GPU_UUID}, {os.getpid()}\n"
            )
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "repeats a row",
        ):
            runner._live_gpu_probe(
                torch_module=torch_module,
                gpu_inventory_runner=inventory_runner,
                gpu_process_runner=process_runner,
                environment=environment,
            )
        process_runner.return_value = types.SimpleNamespace(
            stdout=f"{runner.UNASSIGNED_GPU_UUID}, 4242\n"
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "RTX 5070 acquired a compute process",
        ):
            runner._live_gpu_probe(
                torch_module=torch_module,
                gpu_inventory_runner=inventory_runner,
                gpu_process_runner=process_runner,
                environment=environment,
            )

        parent_environment = runner._frozen_parent_environment()
        with (
            mock.patch.object(runner.os, "environ", parent_environment),
            mock.patch.object(
                runner,
                "_v2_descendant_depths",
                return_value={101: 1, 102: 2},
            ),
            mock.patch.object(
                runner,
                "_peak_process_rss_bytes",
                return_value=23,
            ),
        ):
            sample = runner._exact_live_resource_sample(
                torch_module=torch_module,
                configured_pools=runner.MAXIMUM_CONFIGURED_POOLS,
                measure_state_scratch_bytes=lambda: 29,
            )
        self.assertEqual(sample["process_count"], 3)
        self.assertEqual(sample["cuda_allocated_bytes"], 13)
        self.assertEqual(sample["cuda_reserved_bytes"], 17)
        self.assertEqual(sample["state_scratch_bytes"], 29)

    def test_retained_runtime_reaudits_exact_ledger_witness(self) -> None:
        for purpose, expected_attempts in (("qualification", 72), ("evaluation", 918)):
            with self.subTest(purpose=purpose):
                runtime_root = Path(f"/synthetic/{purpose}-runtime")
                state_root = Path("/synthetic/state")
                witness = {"ledger_sha256": _raw(f"{purpose}-ledger")}
                audit = mock.Mock(return_value=witness)
                with (
                    mock.patch.object(
                        runner,
                        "validate_run_result",
                        return_value={
                            "purpose": purpose,
                            "integrity": {"ledger": witness},
                        },
                    ),
                    mock.patch.object(runner, "audit_evidence_ledger", audit),
                    mock.patch.object(
                        runner,
                        "_exact_child_names",
                        side_effect=runner.V2IntegrityStop("after ledger rejoin"),
                    ),
                    self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "after ledger rejoin",
                    ),
                ):
                    runner._validate_retained_runtime_from_result(
                        purpose=purpose,
                        runtime_root=runtime_root,
                        state_root=state_root,
                        result={"fixture": purpose},
                    )
                audit.assert_called_once_with(
                    runtime_root / "attempt-evidence.sqlite3",
                    expected_attempts=expected_attempts,
                )

                with (
                    mock.patch.object(
                        runner,
                        "validate_run_result",
                        return_value={
                            "purpose": purpose,
                            "integrity": {"ledger": witness},
                        },
                    ),
                    mock.patch.object(
                        runner,
                        "audit_evidence_ledger",
                        return_value={
                            "ledger_sha256": _raw(f"{purpose}-changed-ledger")
                        },
                    ),
                    mock.patch.object(runner, "_exact_child_names") as topology,
                    self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "retained evidence ledger witness differs",
                    ),
                ):
                    runner._validate_retained_runtime_from_result(
                        purpose=purpose,
                        runtime_root=runtime_root,
                        state_root=state_root,
                        result={"fixture": purpose},
                    )
                topology.assert_not_called()

    def test_seed_receipt_rejoins_inode_device_size_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "evaluation-seeds-v1.json"
            target.write_bytes(b"synthetic sealed seed bytes")
            target.chmod(0o600)
            metadata = os.lstat(target)
            receipt = {
                "artifact_bytes": metadata.st_size,
                "artifact_device": metadata.st_dev,
                "artifact_inode": metadata.st_ino,
                "artifact_path": str(target),
                "seed_seal_ref": _ref("metadata-seed-seal"),
            }
            self.assertEqual(
                runner._rejoin_sealed_seed_artifact_metadata(
                    receipt,
                    expected_seed_path=target,
                )["seed_seal_ref"],
                receipt["seed_seal_ref"],
            )
            for field, value in (
                ("artifact_bytes", metadata.st_size + 1),
                ("artifact_device", metadata.st_dev + 1),
                ("artifact_inode", metadata.st_ino + 1),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "metadata differs from receipt",
                ):
                    runner._rejoin_sealed_seed_artifact_metadata(
                        {**receipt, field: value},
                        expected_seed_path=target,
                    )
            target.chmod(0o640)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "metadata differs from receipt",
            ):
                runner._rejoin_sealed_seed_artifact_metadata(
                    receipt,
                    expected_seed_path=target,
                )

        admission, _admission_sha256 = _synthetic_evaluation_admission()
        sealed_receipt = admission["seed_seal_receipt"]
        boundary_names = (
            "preseed_boundary_ref",
            "identity",
            "manifest_ref",
            "manifest_sha256",
            "protocol_identity",
            "qualification_release_ref",
            "qualification_result_ref",
            "qualification_result_sha256",
            "source_seal_ref",
        )
        boundary = {name: admission[name] for name in boundary_names}
        self.assertEqual(
            runner._validate_seed_seal_receipt(
                sealed_receipt,
                boundary=boundary,
            ),
            sealed_receipt,
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "receipt identity/metadata differs",
        ):
            runner._validate_seed_seal_receipt(
                {**sealed_receipt, "artifact_mode": "0640"},
                boundary=boundary,
            )

    def test_evaluator_private_paths_are_empty_and_nonaliasing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cwd = root / "cwd"
            temporary = root / "tmp"
            cwd.mkdir(mode=0o700)
            temporary.mkdir(mode=0o700)
            cwd.chmod(0o700)
            temporary.chmod(0o700)
            observed = runner._require_empty_evaluator_private_paths(
                cwd=cwd,
                tmpdir=temporary,
                injected_paths=True,
            )
            self.assertEqual(observed["cwd"], str(cwd))
            self.assertEqual(observed["tmpdir"], str(temporary))

            leftover = cwd / "leftover"
            leftover.write_bytes(b"x")
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "emptiness differs",
            ):
                runner._require_empty_evaluator_private_paths(
                    cwd=cwd,
                    tmpdir=temporary,
                    injected_paths=True,
                )
            leftover.unlink()

            with (
                mock.patch.object(
                    runner,
                    "_require_owned_directory",
                    side_effect=((str(cwd), 7, 11), (str(temporary), 7, 11)),
                ),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "private paths alias",
                ),
            ):
                runner._require_empty_evaluator_private_paths(
                    cwd=cwd,
                    tmpdir=temporary,
                    injected_paths=True,
                )

    def test_injected_phase_runtime_has_exact_topology_and_closes_partial_ledger(
        self,
    ) -> None:
        class Genesis:
            def __init__(self, root: Path) -> None:
                self.root = root
                self.validations = 0

            def validate_artifacts(self) -> None:
                self.validations += 1

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            runtime_root = parent / "qualification-v1"

            async def genesis_factory(root: Path) -> Genesis:
                root.mkdir(mode=0o700)
                root.chmod(0o700)
                return Genesis(root)

            phase = asyncio.run(
                runner._create_live_phase_runtime(
                    "qualification",
                    runtime_root=runtime_root,
                    genesis_factory=genesis_factory,
                    injected_paths=True,
                )
            )
            self.assertEqual(
                sorted(item.name for item in runtime_root.iterdir()),
                [
                    "arm-runtime",
                    "attempt-evidence.sqlite3",
                    "cognee-scopes",
                    "genesis",
                ],
            )
            self.assertEqual(phase.genesis.validations, 1)
            self.assertIs(phase.path_registry.ledger, phase.ledger)
            self.assertIs(phase.resources.mode, runner.RunMode.QUALIFICATION)
            for child in (phase.arm_runtime_parent, phase.cognee_scope_parent):
                self.assertEqual(child.stat().st_mode & 0o777, 0o700)
            self.assertEqual(phase.ledger.path.stat().st_mode & 0o777, 0o600)
            phase.ledger.close()

            failure_root = parent / "evaluation-v1"
            captured: list[runner.EvidenceLedger] = []

            def ledger_factory(
                path: str | Path, *, create: bool
            ) -> runner.EvidenceLedger:
                ledger = runner.EvidenceLedger(path, create=create)
                captured.append(ledger)
                (failure_root / "unexpected-child").mkdir(mode=0o700)
                return ledger

            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "V2_PHASE_RUNTIME_CONSTRUCTION_FAILED",
            ):
                asyncio.run(
                    runner._create_live_phase_runtime(
                        "evaluation",
                        runtime_root=failure_root,
                        genesis_factory=genesis_factory,
                        ledger_factory=ledger_factory,
                        injected_paths=True,
                    )
                )
            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0]._closed)
            self.assertEqual(
                sorted(item.name for item in failure_root.iterdir()),
                [
                    "arm-runtime",
                    "attempt-evidence.sqlite3",
                    "cognee-scopes",
                    "genesis",
                    "unexpected-child",
                ],
            )

    def test_live_callbacks_bind_source_gpu_resources_and_detect_drift(self) -> None:
        manifest = {"fixture": "live-callback-manifest"}
        manifest_sha256 = _raw("live-callback-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("live-callback-source-seal")}
        current = {
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
            "source_seal": source_seal,
        }
        loader_calls = 0

        def loader() -> tuple[dict[str, object], str, dict[str, object]]:
            nonlocal loader_calls
            loader_calls += 1
            return (
                dict(current["manifest"]),
                current["manifest_sha256"],
                dict(current["source_seal"]),
            )

        inventory = "\n".join(
            (
                f"{runner.ASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5080, "
                "00000000:01:00.0, 12.0",
                f"{runner.UNASSIGNED_GPU_UUID}, NVIDIA GeForce RTX 5070, "
                "00000000:05:00.0, 12.0",
            )
        )
        gpu_inventory_runner = mock.Mock(
            return_value=types.SimpleNamespace(stdout=inventory)
        )
        gpu_process_runner = mock.Mock(
            return_value=types.SimpleNamespace(stdout="")
        )

        class Cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def current_device() -> int:
                return 0

            @staticmethod
            def get_device_name(_index: int) -> str:
                return "NVIDIA GeForce RTX 5080"

            @staticmethod
            def memory_allocated(_index: int) -> int:
                return 0

            @staticmethod
            def memory_reserved(_index: int) -> int:
                return 0

        torch_module = types.SimpleNamespace(
            cuda=Cuda(),
            get_num_threads=lambda: 2,
            get_num_interop_threads=lambda: 1,
        )
        parent_calls: list[dict[str, object]] = []

        def parent_validator(**kwargs: object) -> dict[str, object]:
            parent_calls.append(kwargs)
            return {
                "configured_pool_sum": runner.MAXIMUM_CONFIGURED_POOLS,
                "unassigned_compute_processes": 0,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state_root = root / "state"
            scratch_root = root / "scratch"
            state_root.mkdir(mode=0o700)
            scratch_root.mkdir(mode=0o700)
            state_root.chmod(0o700)
            scratch_root.chmod(0o700)
            parent_environment = runner._frozen_parent_environment()
            with (
                mock.patch.object(runner.os, "environ", parent_environment),
                mock.patch.object(
                    runner,
                    "_v2_descendant_depths",
                    return_value={},
                ),
                mock.patch.object(
                    runner,
                    "_peak_process_rss_bytes",
                    return_value=31,
                ),
            ):
                callbacks = runner.LiveBoundaryCallbacks.create_live(
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    source_seal=source_seal,
                    torch_module=torch_module,
                    parent_boundary_validator=parent_validator,
                    manifest_loader=loader,
                    gpu_inventory_runner=gpu_inventory_runner,
                    gpu_process_runner=gpu_process_runner,
                    state_root=state_root,
                    scratch_root=scratch_root,
                )
                self.assertEqual(callbacks.verify_source_seal(), source_seal["source_seal_ref"])
                self.assertEqual(callbacks.measure_state_scratch_bytes(), 0)
                self.assertEqual(callbacks.sample_resources()["process_count"], 1)
                current["manifest"] = {"fixture": "drifted-manifest"}
                with self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "source manifest or seal changed",
                ):
                    callbacks.verify_source_seal()
        self.assertGreaterEqual(loader_calls, 3)
        self.assertEqual(len(parent_calls), 1)
        self.assertTrue(parent_calls[0]["require_unassigned_zero_processes"])
        callback_ids = {
            id(callbacks.verify_source_seal),
            id(callbacks.probe_gpu),
            id(callbacks.measure_state_scratch_bytes),
            id(callbacks.sample_resources),
        }
        self.assertEqual(len(callback_ids), 4)

    def test_live_orchestration_reverses_every_partial_owner_boundary(self) -> None:
        source_ref = _ref("live-orchestration-source-seal")

        def verify_source() -> str:
            return source_ref

        def probe_gpu() -> dict[str, object]:
            return {"fixture": "gpu"}

        def measure_bytes() -> int:
            return 0

        def sample_resources() -> dict[str, int]:
            return {
                "configured_pools": runner.MAXIMUM_CONFIGURED_POOLS,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
                "process_count": 1,
                "rss_bytes": 0,
                "state_scratch_bytes": 0,
            }

        callbacks = runner.LiveBoundaryCallbacks(
            manifest_sha256=_raw("live-orchestration-manifest"),
            source_seal_ref=source_ref,
            configured_pools=runner.MAXIMUM_CONFIGURED_POOLS,
            verify_source_seal=verify_source,
            probe_gpu=probe_gpu,
            measure_state_scratch_bytes=measure_bytes,
            sample_resources=sample_resources,
        )
        expected_by_failure = {
            "foundation": ["phase", "foundation", "ledger-close"],
            "lineages": [
                "phase",
                "foundation",
                "mechanics",
                "lineages",
                "foundation-close",
                "ledger-close",
            ],
            "product": [
                "phase",
                "foundation",
                "mechanics",
                "lineages",
                "product",
                "lineages-close",
                "foundation-close",
                "ledger-close",
            ],
            "evaluator": [
                "phase",
                "foundation",
                "mechanics",
                "lineages",
                "product",
                "evaluator",
                "product-close",
                "ledger-close",
            ],
            "dependencies": [
                "phase",
                "foundation",
                "mechanics",
                "lineages",
                "product",
                "evaluator",
                "dependencies",
                "evaluator-close",
                "product-close",
                "ledger-close",
            ],
        }

        for failure, expected in expected_by_failure.items():
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                events: list[str] = []
                runtime_root = Path(directory).resolve() / "qualification-v1"
                runtime_root.mkdir(mode=0o700)
                runtime_root.chmod(0o700)
                arm_runtime = runtime_root / "arm-runtime"
                cognee_scopes = runtime_root / "cognee-scopes"
                arm_runtime.mkdir(mode=0o700)
                cognee_scopes.mkdir(mode=0o700)
                arm_runtime.chmod(0o700)
                cognee_scopes.chmod(0o700)
                ledger = runner.EvidenceLedger(
                    runtime_root / "attempt-evidence.sqlite3",
                    create=True,
                )
                original_ledger_close = ledger.close

                def close_ledger() -> None:
                    events.append("ledger-close")
                    original_ledger_close()

                ledger.close = close_ledger  # type: ignore[method-assign]
                resources = runner.ResourceLedger(runner.RunMode.QUALIFICATION)
                registry = runner.PathAllocationRegistry("qualification", ledger)
                genesis = types.SimpleNamespace(root=runtime_root / "genesis")
                phase = types.SimpleNamespace(
                    resources=resources,
                    ledger=ledger,
                    path_registry=registry,
                    genesis=genesis,
                    arm_runtime_parent=arm_runtime,
                    cognee_scope_parent=cognee_scopes,
                )

                class Foundation:
                    manifest: dict[str, object] = {}

                    def close(self) -> None:
                        events.append("foundation-close")

                class Lineages:
                    def close_unfrozen_preserving_state(self) -> None:
                        events.append("lineages-close")

                class Product:
                    def close(self) -> None:
                        events.append("product-close")

                class Evaluator:
                    purpose = "qualification"
                    expected_seed_commitments = (
                        runner.qualification_replicate_commitment(),
                    )

                    def close(self) -> None:
                        events.append("evaluator-close")

                def phase_factory(_purpose: str) -> object:
                    events.append("phase")
                    return phase

                def foundation_factory(**_kwargs: object) -> object:
                    events.append("foundation")
                    if failure == "foundation":
                        raise RuntimeError("synthetic foundation failure")
                    return Foundation()

                def mechanics_factory() -> object:
                    events.append("mechanics")
                    return object()

                def lineage_factory(**_kwargs: object) -> object:
                    events.append("lineages")
                    if failure == "lineages":
                        raise RuntimeError("synthetic lineage failure")
                    return Lineages()

                def product_factory(**_kwargs: object) -> object:
                    events.append("product")
                    if failure == "product":
                        raise RuntimeError("synthetic product failure")
                    return Product()

                def evaluator_factory(**_kwargs: object) -> object:
                    events.append("evaluator")
                    if failure == "evaluator":
                        raise RuntimeError("synthetic evaluator failure")
                    return Evaluator()

                def dependency_factory(**_kwargs: object) -> object:
                    events.append("dependencies")
                    raise RuntimeError("synthetic dependency failure")

                with self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "V2_LIVE_ORCHESTRATION_CONSTRUCTION_FAILED",
                ):
                    asyncio.run(
                        runner._create_live_orchestration(
                            "qualification",
                            callbacks=callbacks,
                            replicate_commitments=(
                                runner.qualification_replicate_commitment(),
                            ),
                            evaluator_factory=evaluator_factory,
                            phase_factory=phase_factory,
                            foundation_factory=foundation_factory,
                            lineage_factory=lineage_factory,
                            product_factory=product_factory,
                            mechanics_factory=mechanics_factory,
                            dependency_factory=dependency_factory,
                            injected_factories=True,
                        )
                    )
                self.assertEqual(events, expected)
                self.assertTrue(ledger._closed)

    def test_evaluator_factory_unproven_process_retains_all_owners(self) -> None:
        retained_processes = runner._UNRESOLVED_EVALUATOR_PROCESSES
        retained_owners = runner._UNRESOLVED_LIVE_ORCHESTRATION_OWNERS
        retained_processes.clear()
        retained_owners.clear()
        source_ref = _ref("factory-reap-source")

        def verify_source() -> str:
            return source_ref

        def probe() -> dict[str, object]:
            return {}

        def measure() -> int:
            return 0

        def sample() -> dict[str, int]:
            return {}

        callbacks = runner.LiveBoundaryCallbacks(
            manifest_sha256=_raw("factory-reap-manifest"),
            source_seal_ref=source_ref,
            configured_pools=runner.MAXIMUM_CONFIGURED_POOLS,
            verify_source_seal=verify_source,
            probe_gpu=probe,
            measure_state_scratch_bytes=measure,
            sample_resources=sample,
        )
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            root = Path(directory).resolve()
            ledger = runner.EvidenceLedger(root / "attempt-evidence.sqlite3", create=True)
            resources = runner.ResourceLedger(runner.RunMode.QUALIFICATION)
            registry = runner.PathAllocationRegistry("qualification", ledger)
            phase = types.SimpleNamespace(
                resources=resources,
                ledger=ledger,
                path_registry=registry,
                genesis=types.SimpleNamespace(root=root / "genesis"),
                arm_runtime_parent=root / "arm-runtime",
                cognee_scope_parent=root / "cognee-scopes",
            )

            class Owner:
                manifest: dict[str, object] = {}

                def __init__(self, label: str) -> None:
                    self.label = label

                def close(self) -> None:
                    events.append(f"{self.label}-close")

                def close_unfrozen_preserving_state(self) -> None:
                    events.append(f"{self.label}-close")

            unresolved_process = types.SimpleNamespace(pid=4242)

            def evaluator_factory(**_kwargs: object) -> object:
                events.append("evaluator")
                retained_processes[4242] = unresolved_process
                raise runner.V2IntegrityStop("synthetic evaluator start failure")

            try:
                with self.assertRaises(ExceptionGroup) as raised:
                    asyncio.run(
                        runner._create_live_orchestration(
                            "qualification",
                            callbacks=callbacks,
                            replicate_commitments=(
                                runner.qualification_replicate_commitment(),
                            ),
                            phase_factory=lambda _purpose: events.append("phase")
                            or phase,
                            foundation_factory=lambda **_kwargs: events.append(
                                "foundation"
                            )
                            or Owner("foundation"),
                            mechanics_factory=lambda: events.append("mechanics")
                            or object(),
                            lineage_factory=lambda **_kwargs: events.append("lineages")
                            or Owner("lineages"),
                            product_factory=lambda **_kwargs: events.append("product")
                            or Owner("product"),
                            evaluator_factory=evaluator_factory,
                            dependency_factory=lambda **_kwargs: object(),
                            injected_factories=True,
                        )
                    )
                self.assertIn("construction and cleanup failed", str(raised.exception))
                self.assertEqual(
                    events,
                    ["phase", "foundation", "mechanics", "lineages", "product", "evaluator"],
                )
                self.assertFalse(ledger._closed)
                self.assertEqual(len(retained_owners), 1)
                retained = next(iter(retained_owners.values()))
                self.assertIs(retained[0], unresolved_process)
                self.assertTrue(any(getattr(item, "label", None) == "product" for item in retained))
                self.assertTrue(any(item is phase for item in retained))
            finally:
                retained_processes.clear()
                retained_owners.clear()
                if not ledger._closed:
                    ledger.close()

    def test_orchestration_cleanup_requires_terminal_evaluator_proof(self) -> None:
        retained = runner._UNRESOLVED_LIVE_ORCHESTRATION_OWNERS
        retained.clear()

        class Paths:
            def __init__(self, events: list[str]) -> None:
                self.events = events

            def assert_closed(self) -> None:
                self.events.append("paths")

        class Ledger:
            def __init__(self, events: list[str]) -> None:
                self.events = events

            def close(self) -> None:
                self.events.append("ledger")

        class Evaluator:
            def __init__(
                self,
                events: list[str],
                *,
                close_failure: bool,
                terminal: bool,
            ) -> None:
                self.events = events
                self.close_failure = close_failure
                self.terminal = terminal

            def close(self) -> None:
                self.events.append("evaluator")
                if self.close_failure:
                    raise runner.V2IntegrityStop("synthetic evaluator close")

            def terminally_closed(self) -> bool:
                return self.terminal

        def dependencies(
            *, close_failure: bool, terminal: bool
        ) -> tuple[object, Evaluator, list[str]]:
            events: list[str] = []
            evaluator_owner = Evaluator(
                events,
                close_failure=close_failure,
                terminal=terminal,
            )

            def close_runtime() -> None:
                events.append("runtime")

            value = types.SimpleNamespace(
                path_allocations=Paths(events),
                evaluator=evaluator_owner,
                close_runtime=close_runtime,
                ledger=Ledger(events),
            )
            return value, evaluator_owner, events

        try:
            successful, _evaluator, events = dependencies(
                close_failure=False,
                terminal=True,
            )
            asyncio.run(runner._close_orchestration(successful))
            self.assertEqual(
                events,
                ["paths", "evaluator", "runtime", "paths", "ledger"],
            )

            failed_but_terminal, _evaluator, events = dependencies(
                close_failure=True,
                terminal=True,
            )
            with self.assertRaises(ExceptionGroup) as raised:
                asyncio.run(runner._close_orchestration(failed_but_terminal))
            self.assertIn("V2 cleanup failed", str(raised.exception))
            self.assertEqual(
                raised.exception.exceptions[0].boundary_code_chain,
                ("V2_EVALUATOR_CLOSE_FAILED",),
            )
            self.assertEqual(
                events,
                ["paths", "evaluator", "runtime", "paths", "ledger"],
            )
            self.assertFalse(retained)

            retry_events: list[str] = []
            retry_evaluator = Evaluator(
                retry_events,
                close_failure=False,
                terminal=True,
            )
            runtime_calls = 0

            def retry_runtime_close() -> None:
                nonlocal runtime_calls
                runtime_calls += 1
                retry_events.append("runtime")
                if runtime_calls == 1:
                    raise runner.V2IntegrityStop(
                        "synthetic durable evidence append failure"
                    )

            retrying = types.SimpleNamespace(
                path_allocations=Paths(retry_events),
                evaluator=retry_evaluator,
                close_runtime=retry_runtime_close,
                ledger=Ledger(retry_events),
            )
            with self.assertRaises(ExceptionGroup) as raised:
                asyncio.run(runner._close_orchestration(retrying))
            self.assertEqual(
                raised.exception.exceptions[0].boundary_code_chain,
                ("V2_RUNTIME_CLOSE_FAILED",),
            )
            self.assertEqual(
                retry_events,
                ["paths", "evaluator", "runtime"],
            )
            self.assertEqual(len(retained), 1)
            self.assertIs(next(iter(retained.values()))[1], retrying)
            asyncio.run(runner._close_orchestration(retrying))
            self.assertEqual(
                retry_events,
                [
                    "paths",
                    "evaluator",
                    "runtime",
                    "paths",
                    "evaluator",
                    "runtime",
                    "paths",
                    "ledger",
                ],
            )
            self.assertFalse(retained)

            unresolved, evaluator_owner, events = dependencies(
                close_failure=True,
                terminal=False,
            )
            with self.assertRaises(ExceptionGroup) as raised:
                asyncio.run(runner._close_orchestration(unresolved))
            self.assertIn("V2 cleanup failed", str(raised.exception))
            self.assertEqual(
                tuple(
                    error.boundary_code_chain
                    for error in raised.exception.exceptions
                ),
                (
                    ("V2_EVALUATOR_CLOSE_FAILED",),
                    (),
                ),
            )
            self.assertEqual(
                str(raised.exception.exceptions[1]),
                "V2_EVALUATOR_REAP_UNPROVEN",
            )
            self.assertEqual(events, ["paths", "evaluator"])
            self.assertEqual(len(retained), 1)
            retained_owners = next(iter(retained.values()))
            self.assertIs(retained_owners[0], evaluator_owner)
            self.assertIs(retained_owners[1], unresolved)

            def verify_source() -> str:
                return _ref("blocked-orchestration-source")

            def probe_gpu() -> dict[str, object]:
                return {}

            def measure() -> int:
                return 0

            def sample() -> dict[str, int]:
                return {}

            callbacks = runner.LiveBoundaryCallbacks(
                manifest_sha256=_raw("blocked-orchestration-manifest"),
                source_seal_ref=verify_source(),
                configured_pools=runner.MAXIMUM_CONFIGURED_POOLS,
                verify_source_seal=verify_source,
                probe_gpu=probe_gpu,
                measure_state_scratch_bytes=measure,
                sample_resources=sample,
            )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "unresolved live orchestration is retained",
            ):
                asyncio.run(
                    runner._create_live_orchestration(
                        "qualification",
                        callbacks=callbacks,
                        replicate_commitments=(
                            runner.qualification_replicate_commitment(),
                        ),
                    )
                )

            retained.clear()
            combined, _evaluator, combined_events = dependencies(
                close_failure=True,
                terminal=True,
            )
            primary = runner.V2IntegrityStop("V2_QUALIFICATION_RUN_FAILED")
            with self.assertRaises(ExceptionGroup) as raised:
                asyncio.run(
                    runner._finish_injected_run(
                        combined,
                        result=None,
                        primary=primary,
                    )
                )
            self.assertEqual(len(raised.exception.exceptions), 2)
            self.assertIs(raised.exception.exceptions[0], primary)
            cleanup = raised.exception.exceptions[1]
            self.assertIs(type(cleanup), runner.V2IntegrityStop)
            self.assertEqual(
                cleanup.boundary_code_chain,
                (
                    "V2_RUN_CLEANUP_FAILED",
                    "V2_EVALUATOR_CLOSE_FAILED",
                ),
            )
            self.assertEqual(
                str(cleanup),
                "V2_RUN_CLEANUP_FAILED>V2_EVALUATOR_CLOSE_FAILED",
            )
            self.assertNotIn("synthetic evaluator close", str(cleanup))
            self.assertEqual(
                combined_events,
                ["paths", "evaluator", "runtime", "paths", "ledger"],
            )
        finally:
            retained.clear()

    def test_primary_failure_discharges_provisional_path_preclose(self) -> None:
        events: list[str] = []

        class Paths:
            def assert_closed(self) -> None:
                events.append("strict")
                raise runner.V2IntegrityStop("synthetic live failed allocation")

            def assert_failure_accounted(self) -> None:
                events.append("failure-accounted")

        class Evaluator:
            def close(self) -> None:
                events.append("evaluator")

            def terminally_closed(self) -> bool:
                return True

        class Ledger:
            def close(self) -> None:
                events.append("ledger")

        def close_runtime() -> None:
            events.append("runtime")

        dependencies = types.SimpleNamespace(
            path_allocations=Paths(),
            evaluator=Evaluator(),
            close_runtime=close_runtime,
            ledger=Ledger(),
        )
        primary = runner.V2IntegrityStop("V2_SYNTHETIC_PRIMARY_FAILURE")
        with self.assertRaises(runner.V2IntegrityStop) as raised:
            asyncio.run(
                runner._finish_injected_run(
                    dependencies,
                    result=None,
                    primary=primary,
                )
            )
        self.assertIs(raised.exception, primary)
        self.assertEqual(
            events,
            ["strict", "evaluator", "runtime", "failure-accounted", "ledger"],
        )


class LiveWrapperAndCliTests(unittest.TestCase):
    def test_result_publication_rejoin_binds_result_and_exact_receipt(self) -> None:
        result = {
            "result_ref": _ref("publication-rejoin-result"),
            "synthetic": True,
        }
        path = Path("/synthetic/result.json")
        receipt = _result_publication_receipt(path, result)
        runner._validate_result_publication_rejoin(
            expected_result=result,
            loaded_result=result,
            publication_receipt=receipt,
            result_path=path,
            result_sha256=str(receipt["sha256"]),
        )

        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "published result bytes differ from run result",
        ):
            runner._validate_result_publication_rejoin(
                expected_result=result,
                loaded_result={**result, "synthetic": False},
                publication_receipt=receipt,
                result_path=path,
                result_sha256=str(receipt["sha256"]),
            )

        mutations: dict[str, object] = {
            "bytes": int(receipt["bytes"]) + 1,
            "path": "/synthetic/other-result.json",
            "result_ref": _ref("publication-rejoin-other-result"),
            "sha256": _raw("publication-rejoin-other-bytes"),
        }
        for field, value in mutations.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "published result receipt differs",
            ):
                runner._validate_result_publication_rejoin(
                    expected_result=result,
                    loaded_result=result,
                    publication_receipt={**receipt, field: value},
                    result_path=path,
                    result_sha256=str(receipt["sha256"]),
                )

    def test_private_seed_verifier_returns_only_public_proof_and_closes(self) -> None:
        admission, admission_sha256 = _synthetic_evaluation_admission()
        expected = runner._evaluation_seed_verification_payload(
            admission, admission_sha256
        )
        expected_response = runner._ipc_envelope("ok", 0, expected)
        expected_command = b"\x00".join(
            item.encode("utf-8")
            for item in runner._EVALUATION_SEED_VERIFIER_PROCESS_ARGV
        ) + b"\x00"
        last_calls: dict[str, mock.Mock] = {}

        class Process:
            def __init__(self) -> None:
                self.args = list(runner._EVALUATION_SEED_VERIFIER_PROCESS_ARGV)
                self.pid = 4242
                self.stdin = object()
                self.stdout = object()

            @staticmethod
            def poll() -> int:
                return 0

        def invoke(
            *,
            response: object = expected_response,
            empty_side_effect: object = None,
        ) -> tuple[dict[str, object], mock.Mock, mock.Mock, mock.Mock]:
            process = Process()
            empty = mock.Mock(side_effect=empty_side_effect, return_value={})
            close = mock.Mock()
            write = mock.Mock()
            last_calls.update(empty=empty, close=close, write=write)
            with (
                mock.patch.object(
                    runner,
                    "_rejoin_sealed_seed_artifact_metadata",
                    return_value={"verified": True},
                ),
                mock.patch.object(
                    runner,
                    "_require_empty_evaluator_private_paths",
                    empty,
                ),
                mock.patch.object(
                    runner,
                    "_frozen_evaluator_environment",
                    return_value={"CUDA_VISIBLE_DEVICES": ""},
                ),
                mock.patch.object(
                    runner.subprocess,
                    "Popen",
                    return_value=process,
                ) as popen,
                mock.patch.object(
                    runner,
                    "_evaluator_process_receipt",
                    return_value=(
                        73,
                        str(runner.EVALUATOR_CWD),
                        str(runner.ANGLER_PYTHON),
                        expected_command,
                    ),
                ),
                mock.patch.object(runner, "_write_ipc_line", write),
                mock.patch.object(
                    runner,
                    "_read_ipc_line",
                    side_effect=(
                        response
                        if isinstance(response, BaseException)
                        else None
                    ),
                    return_value=(
                        expected_response
                        if isinstance(response, BaseException)
                        else response
                    ),
                ),
                mock.patch.object(
                    runner,
                    "_close_evaluator_process_terminal",
                    close,
                ),
            ):
                proof = runner.verify_evaluation_seed_artifact_isolated(
                    admission,
                    admission_sha256,
                )
            popen.assert_called_once()
            return proof, empty, close, write

        proof, empty, close, write = invoke()
        self.assertEqual(proof, expected)
        self.assertEqual(
            set(proof),
            {"admission_ref", "admission_sha256", "artifact_sha256", "seed_seal_ref"},
        )
        serialized = json.dumps(proof, sort_keys=True)
        self.assertNotIn("seed_hex", serialized)
        self.assertTrue(
            all(seed.hex() not in serialized for seed in _EVALUATION_SEEDS)
        )
        self.assertEqual(empty.call_count, 2)
        close.assert_called_once()
        self.assertEqual(
            write.call_args.args[1],
            runner._ipc_envelope(
                "verify-sealed-evaluation-seeds", 0, expected
            ),
        )

        process_failure = runner.V2IntegrityStop("synthetic verifier response")
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "V2_EVALUATION_SEED_VERIFICATION_FAILED",
        ):
            invoke(response=process_failure)
        last_calls["close"].assert_called_once()
        self.assertEqual(last_calls["empty"].call_count, 1)

        post_cleanup = (
            {},
            runner.V2IntegrityStop("synthetic post-child leftover"),
        )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "synthetic post-child leftover",
        ):
            invoke(empty_side_effect=post_cleanup)
        last_calls["close"].assert_called_once()
        self.assertEqual(last_calls["empty"].call_count, 2)

    def test_seed_verifier_worker_emits_no_private_seed_material(self) -> None:
        admission, admission_sha256 = _synthetic_evaluation_admission()
        expected = runner._evaluation_seed_verification_payload(
            admission, admission_sha256
        )
        durable = {
            "manifest": {"manifest_ref": admission["manifest_ref"]},
            "manifest_sha256": admission["manifest_sha256"],
            "qualification_release": {
                "release_ref": admission["qualification_release_ref"]
            },
            "qualification_result": {
                "result_ref": admission["qualification_result_ref"]
            },
            "qualification_result_sha256": admission[
                "qualification_result_sha256"
            ],
            "source_seal": {"source_seal_ref": admission["source_seal_ref"]},
        }
        write = mock.Mock()
        with (
            mock.patch.object(
                runner,
                "_validate_private_worker_boundary",
                return_value=durable,
            ),
            mock.patch.object(
                runner,
                "_read_ipc_line",
                return_value=runner._ipc_envelope(
                    "verify-sealed-evaluation-seeds", 0, expected
                ),
            ),
            mock.patch.object(
                runner,
                "load_evaluation_admission_claim",
                return_value=(admission, admission_sha256),
            ),
            mock.patch.object(
                runner,
                "_rejoin_sealed_seed_artifact_metadata",
                return_value={"verified": True},
            ),
            mock.patch.object(
                runner,
                "_load_sealed_evaluation_seeds_for_worker",
                return_value=(
                    _EVALUATION_SEEDS,
                    admission["seed_seal_receipt"],
                ),
            ),
            mock.patch.object(runner, "_write_ipc_line", write),
        ):
            self.assertEqual(runner.evaluation_seed_verifier_worker_main(), 0)
        emitted = write.call_args.args[1]
        self.assertEqual(emitted, runner._ipc_envelope("ok", 0, expected))
        serialized = json.dumps(emitted, sort_keys=True)
        self.assertNotIn("seed_hex", serialized)
        self.assertTrue(
            all(seed.hex() not in serialized for seed in _EVALUATION_SEEDS)
        )

    def test_live_qualification_orders_authorization_run_and_publication(self) -> None:
        events: list[str] = []
        manifest = {"manifest_ref": _ref("live-qualification-manifest")}
        manifest_sha256 = _raw("live-qualification-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("live-qualification-source-seal")}
        release = {"release_ref": _ref("live-qualification-release")}
        release_sha256 = _raw("live-qualification-release-bytes")
        result = {"result_ref": _ref("live-qualification-result")}
        result_sha256 = hashlib.sha256(
            runner.canonical_json_bytes(result)
        ).hexdigest()
        host = {"environment_sha256": _raw("live-qualification-host")}
        host_gate = mock.Mock(return_value=host)
        source_loads = 0
        release_loads = 0

        def topology(phase: str) -> dict[str, object]:
            events.append(f"topology:{phase}")
            return {"phase": phase}

        def load_source() -> tuple[dict[str, object], str, dict[str, object]]:
            nonlocal source_loads
            source_loads += 1
            events.append(f"source-load:{source_loads}")
            return manifest, manifest_sha256, source_seal

        def load_release(*_args: object) -> tuple[dict[str, object], str]:
            nonlocal release_loads
            release_loads += 1
            events.append(f"release-load:{release_loads}")
            return release, release_sha256

        async def compose(*_args: object, **_kwargs: object) -> object:
            events.append("orchestration")
            return object()

        async def execute(*_args: object, **_kwargs: object) -> dict[str, object]:
            events.append("run")
            return result

        def publish_result(path: Path, value: object) -> dict[str, object]:
            events.append("result-publish")
            self.assertEqual(path, runner.QUALIFICATION_RESULT_PATH)
            self.assertEqual(value, result)
            return _result_publication_receipt(
                runner.QUALIFICATION_RESULT_PATH,
                result,
                result_sha256=result_sha256,
            )

        with (
            mock.patch.object(
                runner,
                "validate_live_parent_boundary",
                side_effect=lambda **kwargs: events.append(
                    "host:"
                    + str(
                        kwargs.get(
                            "temporary_directory_state", "owned-empty"
                        )
                    )
                )
                or host_gate(**kwargs),
            ),
            mock.patch.object(
                runner,
                "validate_live_phase_topology",
                side_effect=topology,
            ),
            mock.patch.object(runner, "load_source_manifest", side_effect=load_source),
            mock.patch.object(
                runner,
                "create_qualification_live_roots_once",
                side_effect=lambda: events.append("roots"),
            ),
            mock.patch.object(
                runner,
                "validate_live_parent_boundary_rejoin",
                side_effect=lambda precreate, activated: events.append(
                    "host-rejoin"
                )
                or self.assertEqual((precreate, activated), (host, host)),
            ) as host_rejoin,
            mock.patch.object(
                runner,
                "build_qualification_release_claim",
                side_effect=lambda *_args: events.append("release-build") or release,
            ),
            mock.patch.object(
                runner,
                "publish_qualification_release_claim_create_once",
                side_effect=lambda *_args: events.append("release-publish")
                or (release, release_sha256),
            ),
            mock.patch.object(
                runner,
                "load_qualification_release_claim",
                side_effect=load_release,
            ),
            mock.patch.object(
                runner.LiveBoundaryCallbacks,
                "create_live",
                side_effect=lambda **_kwargs: events.append("callbacks") or object(),
            ) as callbacks_factory,
            mock.patch.object(runner, "_create_live_orchestration", new=compose),
            mock.patch.object(runner, "run_qualification_injected", new=execute),
            mock.patch.object(
                runner,
                "publish_run_result_create_once",
                side_effect=publish_result,
            ),
            mock.patch.object(
                runner,
                "load_qualification_result",
                side_effect=lambda **_kwargs: events.append("result-load")
                or (result, result_sha256),
            ),
            mock.patch.object(
                runner,
                "_require_empty_evaluator_private_paths",
                side_effect=lambda: events.append("private-paths") or {},
            ),
        ):
            summary = asyncio.run(runner.run_live_qualification())
        self.assertEqual(
            events,
            [
                "topology:pre-qualification",
                "source-load:1",
                "host:absent",
                "topology:pre-qualification",
                "source-load:2",
                "roots",
                "host:owned-empty",
                "host-rejoin",
                "source-load:3",
                "callbacks",
                "release-build",
                "release-publish",
                "release-load:1",
                "orchestration",
                "run",
                "source-load:4",
                "release-load:2",
                "private-paths",
                "result-publish",
                "result-load",
                "topology:pre-seed",
            ],
        )
        self.assertEqual(summary["result_ref"], result["result_ref"])
        self.assertEqual(summary["result_sha256"], result_sha256)
        self.assertEqual(
            host_gate.call_args_list,
            [
                mock.call(
                    require_unassigned_zero_processes=True,
                    temporary_directory_state="absent",
                    activate_cuda_runtime=False,
                ),
                mock.call(require_unassigned_zero_processes=True),
            ],
        )
        host_rejoin.assert_called_once_with(host, host)
        callbacks_factory.assert_called_once_with(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            source_seal=source_seal,
        )

    def test_live_qualification_pre_and_post_root_failures_stop_authorization(
        self,
    ) -> None:
        manifest = {"manifest_ref": _ref("root-gate-manifest")}
        changed_manifest = {"manifest_ref": _ref("root-gate-manifest-changed")}
        manifest_sha256 = _raw("root-gate-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("root-gate-source-seal")}
        host = {"environment_sha256": _raw("root-gate-host")}

        for failure in (
            "precreate-gate",
            "precreate-rejoin",
            "root-create",
            "postcreate-gate",
            "host-rejoin",
            "postcreate-rejoin",
            "callbacks",
        ):
            with self.subTest(failure=failure):
                events: list[str] = []
                host_calls = 0
                source_calls = 0

                def validate_host(**kwargs: object) -> dict[str, object]:
                    nonlocal host_calls
                    host_calls += 1
                    state = kwargs.get(
                        "temporary_directory_state", "owned-empty"
                    )
                    events.append(f"host:{state}")
                    if (
                        failure == "precreate-gate" and host_calls == 1
                    ) or (
                        failure == "postcreate-gate" and host_calls == 2
                    ):
                        raise runner.V2IntegrityStop(
                            f"synthetic {failure} failure"
                        )
                    return host

                def load_source() -> tuple[
                    dict[str, object], str, dict[str, object]
                ]:
                    nonlocal source_calls
                    source_calls += 1
                    events.append(f"source:{source_calls}")
                    changed = (
                        failure == "precreate-rejoin" and source_calls == 2
                    ) or (
                        failure == "postcreate-rejoin" and source_calls == 3
                    )
                    return (
                        changed_manifest if changed else manifest,
                        manifest_sha256,
                        source_seal,
                    )

                def create_roots() -> None:
                    events.append("roots")
                    if failure == "root-create":
                        raise RuntimeError("synthetic root creation failure")

                def create_callbacks(**_kwargs: object) -> object:
                    events.append("callbacks")
                    if failure == "callbacks":
                        raise runner.V2IntegrityStop(
                            "synthetic callbacks failure"
                        )
                    return object()

                def rejoin_host(
                    precreate: object, activated: object
                ) -> None:
                    events.append("host-rejoin")
                    self.assertEqual((precreate, activated), (host, host))
                    if failure == "host-rejoin":
                        raise runner.V2IntegrityStop(
                            "synthetic host-rejoin failure"
                        )

                release_builder = mock.Mock()
                release_publisher = mock.Mock()
                release_loader = mock.Mock()
                orchestration = mock.AsyncMock()
                execution = mock.AsyncMock()
                result_loader = mock.Mock()
                private_paths = mock.Mock()
                result_publisher = mock.Mock()
                with (
                    mock.patch.object(
                        runner,
                        "validate_live_phase_topology",
                        side_effect=lambda phase: events.append(
                            f"topology:{phase}"
                        )
                        or {"phase": phase},
                    ),
                    mock.patch.object(
                        runner,
                        "load_source_manifest",
                        side_effect=load_source,
                    ),
                    mock.patch.object(
                        runner,
                        "validate_live_parent_boundary",
                        side_effect=validate_host,
                    ),
                    mock.patch.object(
                        runner,
                        "create_qualification_live_roots_once",
                        side_effect=create_roots,
                    ) as roots,
                    mock.patch.object(
                        runner,
                        "validate_live_parent_boundary_rejoin",
                        side_effect=rejoin_host,
                    ) as host_rejoin,
                    mock.patch.object(
                        runner.LiveBoundaryCallbacks,
                        "create_live",
                        side_effect=create_callbacks,
                    ) as callbacks,
                    mock.patch.object(
                        runner,
                        "build_qualification_release_claim",
                        release_builder,
                    ),
                    mock.patch.object(
                        runner,
                        "publish_qualification_release_claim_create_once",
                        release_publisher,
                    ),
                    mock.patch.object(
                        runner,
                        "load_qualification_release_claim",
                        release_loader,
                    ),
                    mock.patch.object(
                        runner,
                        "_create_live_orchestration",
                        new=orchestration,
                    ),
                    mock.patch.object(
                        runner,
                        "run_qualification_injected",
                        new=execution,
                    ),
                    mock.patch.object(
                        runner,
                        "load_qualification_result",
                        result_loader,
                    ),
                    mock.patch.object(
                        runner,
                        "_require_empty_evaluator_private_paths",
                        private_paths,
                    ),
                    mock.patch.object(
                        runner,
                        "publish_run_result_create_once",
                        result_publisher,
                    ),
                    self.assertRaises((runner.V2IntegrityStop, RuntimeError)),
                ):
                    asyncio.run(runner.run_live_qualification())
                release_builder.assert_not_called()
                release_publisher.assert_not_called()
                release_loader.assert_not_called()
                orchestration.assert_not_awaited()
                execution.assert_not_awaited()
                result_loader.assert_not_called()
                private_paths.assert_not_called()
                result_publisher.assert_not_called()
                if failure in ("precreate-gate", "precreate-rejoin"):
                    roots.assert_not_called()
                    callbacks.assert_not_called()
                else:
                    roots.assert_called_once_with()
                if failure in (
                    "root-create",
                    "postcreate-gate",
                    "host-rejoin",
                    "postcreate-rejoin",
                ):
                    callbacks.assert_not_called()
                if failure in (
                    "precreate-gate",
                    "precreate-rejoin",
                    "root-create",
                    "postcreate-gate",
                ):
                    host_rejoin.assert_not_called()
                else:
                    host_rejoin.assert_called_once_with(host, host)

    def test_live_qualification_never_publishes_after_run_or_chain_failure(self) -> None:
        manifest = {"manifest_ref": _ref("failed-qualification-manifest")}
        changed_manifest = {"manifest_ref": _ref("changed-qualification-manifest")}
        manifest_sha256 = _raw("failed-qualification-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("failed-qualification-source")}
        release = {"release_ref": _ref("failed-qualification-release")}
        release_sha256 = _raw("failed-qualification-release-bytes")
        result = {"result_ref": _ref("failed-qualification-result")}

        for failure in ("run", "chain", "cleanup"):
            with self.subTest(failure=failure):
                source_values = [
                    (manifest, manifest_sha256, source_seal),
                    (manifest, manifest_sha256, source_seal),
                    (manifest, manifest_sha256, source_seal),
                    (
                        changed_manifest if failure == "chain" else manifest,
                        manifest_sha256,
                        source_seal,
                    ),
                ]
                publisher = mock.Mock()
                execute = mock.AsyncMock(
                    side_effect=(
                        RuntimeError("synthetic run failure")
                        if failure == "run"
                        else None
                    ),
                    return_value=result,
                )
                with (
                    mock.patch.object(
                        runner,
                        "validate_live_parent_boundary",
                        return_value={"environment_sha256": _raw("failed-host")},
                    ),
                    mock.patch.object(
                        runner,
                        "validate_live_parent_boundary_rejoin",
                    ),
                    mock.patch.object(runner, "validate_live_phase_topology"),
                    mock.patch.object(
                        runner,
                        "load_source_manifest",
                        side_effect=source_values,
                    ),
                    mock.patch.object(runner, "create_qualification_live_roots_once"),
                    mock.patch.object(
                        runner,
                        "build_qualification_release_claim",
                        return_value=release,
                    ),
                    mock.patch.object(
                        runner,
                        "publish_qualification_release_claim_create_once",
                        return_value=(release, release_sha256),
                    ),
                    mock.patch.object(
                        runner,
                        "load_qualification_release_claim",
                        return_value=(release, release_sha256),
                    ),
                    mock.patch.object(
                        runner.LiveBoundaryCallbacks,
                        "create_live",
                        return_value=object(),
                    ),
                    mock.patch.object(
                        runner,
                        "_create_live_orchestration",
                        new=mock.AsyncMock(return_value=object()),
                    ),
                    mock.patch.object(
                        runner,
                        "run_qualification_injected",
                        new=execute,
                    ),
                    mock.patch.object(
                        runner,
                        "publish_run_result_create_once",
                        publisher,
                    ),
                    mock.patch.object(
                        runner,
                        "_require_empty_evaluator_private_paths",
                        side_effect=(
                            runner.V2IntegrityStop(
                                "synthetic evaluator private path leftover"
                            )
                            if failure == "cleanup"
                            else None
                        ),
                    ),
                ):
                    expected = RuntimeError if failure == "run" else runner.V2IntegrityStop
                    with self.assertRaises(expected):
                        asyncio.run(runner.run_live_qualification())
                publisher.assert_not_called()

    def test_live_evaluation_reopens_result_then_validates_terminal_topology(self) -> None:
        events: list[str] = []
        manifest = {"manifest_ref": _ref("live-evaluation-manifest")}
        manifest_sha256 = _raw("live-evaluation-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("live-evaluation-source")}
        release = {"release_ref": _ref("live-evaluation-release")}
        release_sha256 = _raw("live-evaluation-release-bytes")
        qualification = {"result_ref": _ref("live-qualification-result")}
        qualification_sha256 = _raw("live-qualification-result-bytes")
        admission = {
            "admission_ref": _ref("live-evaluation-admission"),
            "boundary": "accepted",
            "evaluator_commitments": {"fixture": "evaluator"},
            "seed_commitments": list(_EVALUATION_COMMITMENTS),
            "seed_seal_receipt": {"fixture": "seed"},
        }
        admission_sha256 = _raw("live-evaluation-admission-bytes")
        result = {"result_ref": _ref("live-evaluation-result")}
        result_sha256 = hashlib.sha256(
            runner.canonical_json_bytes(result)
        ).hexdigest()
        publication_receipt = _result_publication_receipt(
            runner.EVALUATION_RESULT_PATH,
            result,
            result_sha256=result_sha256,
        )
        host_gate = mock.Mock(
            return_value={"environment_sha256": _raw("evaluation-host")}
        )

        def mark(name: str, value: object) -> object:
            events.append(name)
            return value

        async def compose(*_args: object, **_kwargs: object) -> object:
            return mark("orchestration", object())

        async def execute(*_args: object, **_kwargs: object) -> dict[str, object]:
            return mark("run", result)  # type: ignore[return-value]

        def verify_seed(*_args: object, **_kwargs: object) -> dict[str, object]:
            return mark("seed-verify", {"verified": True})  # type: ignore[return-value]

        with (
            mock.patch.object(
                runner,
                "validate_live_parent_boundary",
                side_effect=lambda **kwargs: mark("host", host_gate(**kwargs)),
            ),
            mock.patch.object(
                runner,
                "validate_live_phase_topology",
                side_effect=lambda phase: mark(f"topology:{phase}", {}),
            ),
            mock.patch.object(
                runner,
                "load_source_manifest",
                side_effect=lambda: mark(
                    "source-load", (manifest, manifest_sha256, source_seal)
                ),
            ),
            mock.patch.object(
                runner,
                "load_qualification_release_claim",
                side_effect=lambda *_args: mark(
                    "release-load", (release, release_sha256)
                ),
            ),
            mock.patch.object(
                runner,
                "load_qualification_result",
                side_effect=lambda **_kwargs: mark(
                    "qualification-load", (qualification, qualification_sha256)
                ),
            ),
            mock.patch.object(
                runner,
                "load_evaluation_admission_claim",
                side_effect=lambda: mark(
                    "admission-load", (admission, admission_sha256)
                ),
            ),
            mock.patch.object(
                runner,
                "_evaluation_admission_boundary",
                return_value={"boundary": "accepted"},
            ),
            mock.patch.object(
                runner,
                "verify_evaluation_seed_artifact_isolated",
                side_effect=verify_seed,
            ),
            mock.patch.object(
                runner.LiveBoundaryCallbacks,
                "create_live",
                side_effect=lambda **_kwargs: mark("callbacks", object()),
            ),
            mock.patch.object(runner, "_create_live_orchestration", new=compose),
            mock.patch.object(runner, "run_evaluation_injected", new=execute),
            mock.patch.object(
                runner,
                "publish_run_result_create_once",
                side_effect=lambda path, value: mark(
                    "result-publish",
                    publication_receipt
                    if path == runner.EVALUATION_RESULT_PATH and value == result
                    else None,
                ),
            ),
            mock.patch.object(
                runner,
                "load_evaluation_result",
                side_effect=lambda **_kwargs: mark(
                    "result-load", (result, result_sha256)
                ),
            ),
            mock.patch.object(
                runner,
                "validate_live_terminal_topology",
                side_effect=lambda: mark(
                    "terminal-topology",
                    {
                        "evaluation_result_ref": result["result_ref"],
                        "evaluation_result_sha256": result_sha256,
                    },
                ),
            ),
        ):
            summary = asyncio.run(runner.run_live_evaluation())
        self.assertLess(events.index("result-publish"), events.index("result-load"))
        self.assertLess(events.index("result-load"), events.index("terminal-topology"))
        self.assertEqual(events.count("source-load"), 2)
        self.assertEqual(events.count("release-load"), 2)
        self.assertEqual(events.count("qualification-load"), 2)
        self.assertEqual(events.count("admission-load"), 2)
        self.assertEqual(events.count("seed-verify"), 2)
        self.assertLess(events.index("seed-verify"), events.index("callbacks"))
        self.assertLess(events.index("run"), len(events) - 1 - events[::-1].index("seed-verify"))
        self.assertLess(
            len(events) - 1 - events[::-1].index("seed-verify"),
            events.index("result-publish"),
        )
        self.assertEqual(summary["result_sha256"], result_sha256)
        host_gate.assert_called_once_with(require_unassigned_zero_processes=True)

    def test_live_evaluation_rejects_reopened_tamper_before_terminal_gate(self) -> None:
        manifest = {"manifest_ref": _ref("tamper-evaluation-manifest")}
        manifest_sha256 = _raw("tamper-evaluation-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("tamper-evaluation-source")}
        release = {"release_ref": _ref("tamper-evaluation-release")}
        qualification = {"result_ref": _ref("tamper-qualification-result")}
        admission = {
            "admission_ref": _ref("tamper-evaluation-admission"),
            "boundary": "accepted",
            "evaluator_commitments": {},
            "seed_commitments": list(_EVALUATION_COMMITMENTS),
            "seed_seal_receipt": {},
        }
        result = {"result_ref": _ref("tamper-evaluation-result")}
        tampered = {"result_ref": _ref("tampered-evaluation-result")}
        terminal = mock.Mock()
        with (
            mock.patch.object(
                runner,
                "validate_live_parent_boundary",
                return_value={"environment_sha256": _raw("tamper-host")},
            ),
            mock.patch.object(runner, "validate_live_phase_topology"),
            mock.patch.object(
                runner,
                "load_source_manifest",
                return_value=(manifest, manifest_sha256, source_seal),
            ),
            mock.patch.object(
                runner,
                "load_qualification_release_claim",
                return_value=(release, _raw("tamper-release-bytes")),
            ),
            mock.patch.object(
                runner,
                "load_qualification_result",
                return_value=(qualification, _raw("tamper-qualification-bytes")),
            ),
            mock.patch.object(
                runner,
                "load_evaluation_admission_claim",
                return_value=(admission, _raw("tamper-admission-bytes")),
            ),
            mock.patch.object(
                runner,
                "_evaluation_admission_boundary",
                return_value={"boundary": "accepted"},
            ),
            mock.patch.object(
                runner,
                "verify_evaluation_seed_artifact_isolated",
                return_value={"verified": True},
            ),
            mock.patch.object(
                runner.LiveBoundaryCallbacks,
                "create_live",
                return_value=object(),
            ),
            mock.patch.object(
                runner,
                "_create_live_orchestration",
                new=mock.AsyncMock(return_value=object()),
            ),
            mock.patch.object(
                runner,
                "run_evaluation_injected",
                new=mock.AsyncMock(return_value=result),
            ),
            mock.patch.object(
                runner,
                "publish_run_result_create_once",
                return_value=_result_publication_receipt(
                    runner.EVALUATION_RESULT_PATH,
                    result,
                ),
            ),
            mock.patch.object(
                runner,
                "load_evaluation_result",
                return_value=(tampered, _raw("tampered-result-bytes")),
            ),
            mock.patch.object(
                runner,
                "validate_live_terminal_topology",
                terminal,
            ),
            self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "published result bytes differ from run result",
            ),
        ):
            asyncio.run(runner.run_live_evaluation())
        terminal.assert_not_called()

    def test_live_evaluation_seed_failure_precedes_callbacks_and_model(self) -> None:
        manifest = {"manifest_ref": _ref("seed-failure-manifest")}
        manifest_sha256 = _raw("seed-failure-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("seed-failure-source")}
        release = {"release_ref": _ref("seed-failure-release")}
        qualification = {"result_ref": _ref("seed-failure-qualification")}
        admission = {
            "admission_ref": _ref("seed-failure-admission"),
            "boundary": "accepted",
            "evaluator_commitments": {},
            "seed_commitments": list(_EVALUATION_COMMITMENTS),
            "seed_seal_receipt": {},
        }
        callbacks = mock.Mock()
        orchestration = mock.AsyncMock()
        execute = mock.AsyncMock()
        publisher = mock.Mock()
        with (
            mock.patch.object(
                runner,
                "validate_live_parent_boundary",
                return_value={"environment_sha256": _raw("seed-failure-host")},
            ),
            mock.patch.object(runner, "validate_live_phase_topology"),
            mock.patch.object(
                runner,
                "load_source_manifest",
                return_value=(manifest, manifest_sha256, source_seal),
            ),
            mock.patch.object(
                runner,
                "load_qualification_release_claim",
                return_value=(release, _raw("seed-failure-release-bytes")),
            ),
            mock.patch.object(
                runner,
                "load_qualification_result",
                return_value=(
                    qualification,
                    _raw("seed-failure-qualification-bytes"),
                ),
            ),
            mock.patch.object(
                runner,
                "load_evaluation_admission_claim",
                return_value=(admission, _raw("seed-failure-admission-bytes")),
            ),
            mock.patch.object(
                runner,
                "_evaluation_admission_boundary",
                return_value={"boundary": "accepted"},
            ),
            mock.patch.object(
                runner,
                "verify_evaluation_seed_artifact_isolated",
                side_effect=runner.V2IntegrityStop("synthetic seed verifier failure"),
            ) as verifier,
            mock.patch.object(
                runner.LiveBoundaryCallbacks,
                "create_live",
                callbacks,
            ),
            mock.patch.object(
                runner,
                "_create_live_orchestration",
                new=orchestration,
            ),
            mock.patch.object(
                runner,
                "run_evaluation_injected",
                new=execute,
            ),
            mock.patch.object(
                runner,
                "publish_run_result_create_once",
                publisher,
            ),
            self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "synthetic seed verifier failure",
            ),
        ):
            asyncio.run(runner.run_live_evaluation())
        verifier.assert_called_once_with(
            admission, _raw("seed-failure-admission-bytes")
        )
        callbacks.assert_not_called()
        orchestration.assert_not_awaited()
        execute.assert_not_awaited()
        publisher.assert_not_called()

    def test_live_evaluation_stops_on_publication_or_terminal_chain_drift(self) -> None:
        manifest = {"manifest_ref": _ref("drift-evaluation-manifest")}
        manifest_sha256 = _raw("drift-evaluation-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("drift-evaluation-source")}
        release = {"release_ref": _ref("drift-evaluation-release")}
        qualification = {"result_ref": _ref("drift-qualification-result")}
        admission = {
            "admission_ref": _ref("drift-evaluation-admission"),
            "boundary": "accepted",
            "evaluator_commitments": {},
            "seed_commitments": list(_EVALUATION_COMMITMENTS),
            "seed_seal_receipt": {},
        }
        result = {"result_ref": _ref("drift-evaluation-result")}
        result_sha256 = hashlib.sha256(
            runner.canonical_json_bytes(result)
        ).hexdigest()
        receipt = _result_publication_receipt(
            runner.EVALUATION_RESULT_PATH,
            result,
            result_sha256=result_sha256,
        )

        for boundary in ("publication", "terminal"):
            with self.subTest(boundary=boundary):
                terminal = mock.Mock(
                    side_effect=(
                        runner.V2IntegrityStop("terminal admission chain drift")
                        if boundary == "terminal"
                        else None
                    )
                )
                publisher = mock.Mock(
                    side_effect=(
                        runner.V2IntegrityStop("publication admission chain drift")
                        if boundary == "publication"
                        else None
                    ),
                    return_value=receipt,
                )
                with (
                    mock.patch.object(
                        runner,
                        "validate_live_parent_boundary",
                        return_value={"environment_sha256": _raw("drift-host")},
                    ),
                    mock.patch.object(runner, "validate_live_phase_topology"),
                    mock.patch.object(
                        runner,
                        "load_source_manifest",
                        return_value=(manifest, manifest_sha256, source_seal),
                    ),
                    mock.patch.object(
                        runner,
                        "load_qualification_release_claim",
                        return_value=(release, _raw("drift-release-bytes")),
                    ),
                    mock.patch.object(
                        runner,
                        "load_qualification_result",
                        return_value=(
                            qualification,
                            _raw("drift-qualification-bytes"),
                        ),
                    ),
                    mock.patch.object(
                        runner,
                        "load_evaluation_admission_claim",
                        return_value=(admission, _raw("drift-admission-bytes")),
                    ),
                    mock.patch.object(
                        runner,
                        "_evaluation_admission_boundary",
                        return_value={"boundary": "accepted"},
                    ),
                    mock.patch.object(
                        runner,
                        "verify_evaluation_seed_artifact_isolated",
                        return_value={"verified": True},
                    ),
                    mock.patch.object(
                        runner.LiveBoundaryCallbacks,
                        "create_live",
                        return_value=object(),
                    ),
                    mock.patch.object(
                        runner,
                        "_create_live_orchestration",
                        new=mock.AsyncMock(return_value=object()),
                    ),
                    mock.patch.object(
                        runner,
                        "run_evaluation_injected",
                        new=mock.AsyncMock(return_value=result),
                    ),
                    mock.patch.object(
                        runner,
                        "publish_run_result_create_once",
                        publisher,
                    ),
                    mock.patch.object(
                        runner,
                        "load_evaluation_result",
                        return_value=(result, result_sha256),
                    ),
                    mock.patch.object(
                        runner,
                        "validate_live_terminal_topology",
                        terminal,
                    ),
                    self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        f"{boundary} admission chain drift",
                    ),
                ):
                    asyncio.run(runner.run_live_evaluation())
                publisher.assert_called_once()
                if boundary == "publication":
                    terminal.assert_not_called()
                else:
                    terminal.assert_called_once()

    def test_terminal_topology_enforces_extra_path_and_non_aliasing_gates(self) -> None:
        manifest = {"manifest_ref": _ref("terminal-manifest")}
        source_seal = {"source_seal_ref": _ref("terminal-source")}
        release = {"release_ref": _ref("terminal-release")}
        qualification = {"result_ref": _ref("terminal-qualification")}
        admission = {
            "admission_ref": _ref("terminal-admission"),
            "boundary": "accepted",
            "seed_seal_receipt": {"fixture": "seed"},
        }
        evaluation = {"result_ref": _ref("terminal-evaluation")}
        evaluation_sha256 = _raw("terminal-evaluation-bytes")

        def run_terminal(
            *,
            child_error: BaseException | None = None,
            alias_error: BaseException | None = None,
        ) -> tuple[dict[str, object], mock.Mock, mock.Mock]:
            exact_children = mock.Mock(side_effect=child_error)
            non_aliasing = mock.Mock(side_effect=alias_error)
            with (
                mock.patch.object(
                    runner,
                    "load_source_manifest",
                    return_value=(manifest, _raw("terminal-manifest-bytes"), source_seal),
                ),
                mock.patch.object(
                    runner,
                    "load_qualification_release_claim",
                    return_value=(release, _raw("terminal-release-bytes")),
                ),
                mock.patch.object(
                    runner,
                    "load_qualification_result",
                    return_value=(qualification, _raw("terminal-qualification-bytes")),
                ),
                mock.patch.object(
                    runner,
                    "load_evaluation_admission_claim",
                    return_value=(admission, _raw("terminal-admission-bytes")),
                ),
                mock.patch.object(
                    runner,
                    "_evaluation_admission_boundary",
                    return_value={"boundary": "accepted"},
                ),
                mock.patch.object(
                    runner,
                    "load_evaluation_result",
                    return_value=(evaluation, evaluation_sha256),
                ),
                mock.patch.object(
                    runner,
                    "_rejoin_sealed_seed_artifact_metadata",
                    return_value={"verified": True},
                ),
                mock.patch.object(
                    runner,
                    "verify_evaluation_seed_artifact_isolated",
                    return_value={
                        "artifact_sha256": _raw("terminal-seed-artifact"),
                        "seed_seal_ref": _ref("terminal-seed-seal"),
                    },
                ),
                mock.patch.object(
                    runner,
                    "_exact_child_names",
                    exact_children,
                ),
                mock.patch.object(runner, "_require_owned_directory"),
                mock.patch.object(
                    runner,
                    "_validate_retained_runtime_from_result",
                    return_value={"baseline_roots": []},
                ),
                mock.patch.object(runner, "_require_owned_regular_file"),
                mock.patch.object(
                    runner,
                    "_require_non_aliasing_existing_paths",
                    non_aliasing,
                ),
                mock.patch.object(
                    runner,
                    "_bounded_no_follow_state_scratch_bytes",
                    return_value=37,
                ),
            ):
                value = runner.validate_live_terminal_topology()
            return value, exact_children, non_aliasing

        terminal, exact_children, non_aliasing = run_terminal()
        self.assertEqual(terminal["evaluation_result_ref"], evaluation["result_ref"])
        self.assertEqual(terminal["evaluation_result_sha256"], evaluation_sha256)
        self.assertGreaterEqual(exact_children.call_count, 3)
        non_aliasing.assert_called_once()
        with self.assertRaisesRegex(runner.V2IntegrityStop, "extra terminal path"):
            run_terminal(
                child_error=runner.V2IntegrityStop("extra terminal path")
            )
        with self.assertRaisesRegex(runner.V2IntegrityStop, "terminal alias"):
            run_terminal(alias_error=runner.V2IntegrityStop("terminal alias"))

    def test_admission_closes_seed_worker_before_publication_and_drift_stops(self) -> None:
        manifest = {"manifest_ref": _ref("admission-order-manifest")}
        manifest_sha256 = _raw("admission-order-manifest-bytes")
        source_seal = {"source_seal_ref": _ref("admission-order-source")}
        release = {"release_ref": _ref("admission-order-release")}
        qualification = {"result_ref": _ref("admission-order-qualification")}
        qualification_sha256 = _raw("admission-order-qualification-bytes")
        boundary = {"boundary_ref": _ref("admission-order-boundary")}
        claim = {"admission_ref": _ref("admission-order-claim")}
        claim_sha256 = _raw("admission-order-claim-bytes")

        class Worker:
            def __init__(self, events: list[str], *, close_failure: bool) -> None:
                self.events = events
                self.close_failure = close_failure
                self.seed_receipt = {
                    "commitments": list(_EVALUATION_COMMITMENTS),
                    "seed_seal_ref": _ref("admission-order-seed-seal"),
                }
                self.evaluator_commitments = {"fixture": "commitments"}

            def revalidate(self, admission_ref: str) -> None:
                self.events.append("worker-revalidate")
                self_ref = claim["admission_ref"]
                if admission_ref != self_ref:
                    raise AssertionError("admission ref differs")

            def close(self) -> None:
                self.events.append("worker-close")
                if self.close_failure:
                    raise runner.V2IntegrityStop("synthetic worker close failure")

        def invoke(*, failure: str | None = None) -> tuple[list[str], mock.Mock]:
            events: list[str] = []
            worker = Worker(events, close_failure=failure == "close")
            publisher = mock.Mock(
                side_effect=lambda *_args, **_kwargs: events.append("publish")
                or (claim, claim_sha256)
            )
            reload_count = 0

            def reload_boundary() -> dict[str, object]:
                nonlocal reload_count
                reload_count += 1
                events.append(f"reload:{reload_count}")
                if failure == "post-close" and reload_count == 1:
                    return {"boundary_ref": _ref("changed-after-close")}
                return boundary

            with (
                mock.patch.object(
                    runner,
                    "validate_live_parent_boundary",
                    side_effect=lambda **_kwargs: events.append("host")
                    or {"environment_sha256": _raw("admission-order-host")},
                ),
                mock.patch.object(
                    runner,
                    "validate_live_phase_topology",
                    side_effect=lambda phase: events.append(f"topology:{phase}"),
                ),
                mock.patch.object(
                    runner,
                    "load_source_manifest",
                    return_value=(manifest, manifest_sha256, source_seal),
                ),
                mock.patch.object(
                    runner,
                    "load_qualification_release_claim",
                    return_value=(release, _raw("admission-order-release-bytes")),
                ),
                mock.patch.object(
                    runner,
                    "load_qualification_result",
                    return_value=(qualification, qualification_sha256),
                ),
                mock.patch.object(
                    runner,
                    "_evaluation_admission_boundary",
                    return_value=boundary,
                ),
                mock.patch.object(
                    runner,
                    "create_evaluation_seed_seal_isolated",
                    side_effect=lambda _boundary: events.append("worker-create")
                    or worker,
                ),
                mock.patch.object(
                    runner,
                    "build_evaluation_admission_claim",
                    side_effect=lambda **_kwargs: events.append("claim-build")
                    or claim,
                ),
                mock.patch.object(
                    runner,
                    "_reload_evaluation_admission_boundary",
                    side_effect=reload_boundary,
                ),
                mock.patch.object(
                    runner,
                    "publish_evaluation_admission_claim_create_once",
                    publisher,
                ),
                mock.patch.object(
                    runner,
                    "load_evaluation_admission_claim",
                    return_value=(claim, claim_sha256),
                ),
            ):
                if failure is None:
                    summary = runner.prepare_evaluation_admission_live()
                    self.assertEqual(summary["admission_ref"], claim["admission_ref"])
                else:
                    expected = (
                        "synthetic worker close failure"
                        if failure == "close"
                        else "changed after worker close"
                    )
                    with self.assertRaisesRegex(runner.V2IntegrityStop, expected):
                        runner.prepare_evaluation_admission_live()
            return events, publisher

        events, publisher = invoke()
        publisher.assert_called_once()
        self.assertLess(events.index("worker-close"), events.index("reload:1"))
        self.assertLess(events.index("reload:1"), events.index("publish"))
        self.assertLess(events.index("publish"), events.index("reload:2"))
        self.assertLess(events.index("topology:pre-evaluation"), events.index("reload:3"))

        for failure in ("close", "post-close"):
            with self.subTest(failure=failure):
                _events, publisher = invoke(failure=failure)
                publisher.assert_not_called()

    def test_cli_defaults_to_denial_and_actions_are_mutually_exclusive(self) -> None:
        public_actions = (
            "run_foundation_mirror_preflight_live",
            "build_source_manifest_candidate",
            "publish_reviewed_source_manifest",
            "run_live_qualification",
            "prepare_evaluation_admission_live",
            "run_live_evaluation",
        )
        for argv in (
            (),
            ("--qualification", "--evaluation"),
            ("--foundation-mirror-preflight", "--source-manifest-candidate"),
            ("--evaluator-worker", "--qualification"),
        ):
            calls = [mock.patch.object(runner, name) for name in public_actions]
            with self.subTest(argv=argv), ExitStack() as stack:
                mocks = [stack.enter_context(patch) for patch in calls]
                stack.enter_context(mock.patch.object(runner.sys, "stderr", io.StringIO()))
                with self.assertRaises(SystemExit) as raised:
                    runner.main(argv)
                self.assertEqual(raised.exception.code, 2)
                for action in mocks:
                    action.assert_not_called()

    def test_cli_rejects_public_and_private_flag_abbreviations(self) -> None:
        dispatch_names = (
            "evaluator_worker_main",
            "evaluation_commitment_worker_main",
            "evaluation_seed_verifier_worker_main",
            "foundation_mirror_preflight_root_main",
            "foundation_mirror_preflight_child_main",
            "run_foundation_mirror_preflight_live",
            "build_source_manifest_candidate",
            "publish_reviewed_source_manifest",
            "run_live_qualification",
            "prepare_evaluation_admission_live",
            "run_live_evaluation",
        )
        for argv in (
            ("--qualific",),
            ("--evaluat",),
            ("--evaluator-w",),
            ("--evaluation-seed-verifier-w",),
            ("--foundation-mirror-preflight-r",),
        ):
            with self.subTest(argv=argv), ExitStack() as stack:
                dispatches = [
                    stack.enter_context(mock.patch.object(runner, name))
                    for name in dispatch_names
                ]
                stack.enter_context(
                    mock.patch.object(runner.sys, "stderr", io.StringIO())
                )
                with self.assertRaises(SystemExit) as raised:
                    runner.main(argv)
                self.assertEqual(raised.exception.code, 2)
                for dispatch in dispatches:
                    dispatch.assert_not_called()

        worker = mock.Mock(return_value=0)
        with mock.patch.object(
            runner,
            "evaluation_seed_verifier_worker_main",
            worker,
        ):
            with self.assertRaises(SystemExit) as raised:
                runner.main(("--evaluation-seed-verifier-worker",))
        self.assertEqual(raised.exception.code, 0)
        worker.assert_called_once_with()


class GuardedFoundationBoundaryTests(unittest.TestCase):
    def test_verified_r2_foundation_is_rebound_to_exact_successor_cap(self) -> None:
        import torch

        helpers = runner._load_approved_r2_low_level_helpers()
        model = torch.nn.Linear(3, 3, bias=False)
        model.requires_grad_(False)
        model.eval()
        tokenizer = object()
        guard = object()
        file_evidence = {"verified": True}
        r2_manifest = helpers["qualified_qwen_cycle_manifest"]()
        loaded = helpers["LoadedQwenFoundation"](
            model=model,
            tokenizer=tokenizer,
            io=object(),
            manifest=r2_manifest,
            guard=guard,
            production_file_evidence=file_evidence,
        )
        original_io = loaded.io
        original_manifest = loaded.manifest

        rebound = runner._rebind_successor_qwen_foundation(
            loaded,
            loaded_type=helpers["LoadedQwenFoundation"],
            r2_manifest=r2_manifest,
        )

        self.assertIs(rebound.model, model)
        self.assertIs(rebound.tokenizer, tokenizer)
        self.assertIs(rebound.guard, guard)
        self.assertIs(rebound.production_file_evidence, file_evidence)
        self.assertIs(rebound.io.model, model)
        self.assertIs(rebound.io.tokenizer, tokenizer)
        self.assertEqual(rebound.io._generation_calls, 0)
        self.assertIs(loaded.io, original_io)
        self.assertIs(loaded.manifest, original_manifest)
        self.assertEqual(loaded.manifest.max_new_tokens, 64)
        self.assertEqual(rebound.manifest.max_new_tokens, 160)
        self.assertEqual(
            rebound.manifest.generation_config_ref,
            runner.QUALIFIED_SUCCESSOR_GENERATION_CONFIG_REF,
        )
        self.assertEqual(
            rebound.manifest.manifest_ref,
            runner.QUALIFIED_SUCCESSOR_CYCLE_MANIFEST_REF,
        )
        rebound.manifest.assert_io(rebound.io)

    def test_successor_foundation_rebind_rejects_wrong_source_or_cap(self) -> None:
        import torch

        helpers = runner._load_approved_r2_low_level_helpers()
        loaded_type = helpers["LoadedQwenFoundation"]
        model = torch.nn.Linear(3, 3, bias=False)
        model.requires_grad_(False)
        model.eval()
        r2_manifest = helpers["qualified_qwen_cycle_manifest"]()
        loaded = loaded_type(
            model=model,
            tokenizer=object(),
            io=object(),
            manifest=r2_manifest,
            guard=object(),
            production_file_evidence={"verified": True},
        )

        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "successor foundation rebind source differs",
        ):
            runner._rebind_successor_qwen_foundation(
                object(),
                loaded_type=loaded_type,
                r2_manifest=r2_manifest,
            )

        mismatched_manifest = replace(r2_manifest, max_input_tokens=4_095)
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "successor foundation rebind source differs",
        ):
            runner._rebind_successor_qwen_foundation(
                loaded,
                loaded_type=loaded_type,
                r2_manifest=mismatched_manifest,
            )

        non_r2_manifest = replace(r2_manifest, max_new_tokens=65)
        non_r2_loaded = replace(loaded, manifest=non_r2_manifest)
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "successor foundation rebind source differs",
        ):
            runner._rebind_successor_qwen_foundation(
                non_r2_loaded,
                loaded_type=loaded_type,
                r2_manifest=non_r2_manifest,
            )

        with (
            mock.patch.object(runner, "MAXIMUM_OUTPUT_TOKENS", 161),
            self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "successor foundation rebind source differs",
            ),
        ):
            runner._rebind_successor_qwen_foundation(
                loaded,
                loaded_type=loaded_type,
                r2_manifest=r2_manifest,
            )

    def test_guard_accepts_160_generated_tokens_and_rejects_161(self) -> None:
        import torch

        class FoundationGuard:
            def __init__(self, digest: str) -> None:
                self.digest = digest

            def probe(self) -> str:
                return self.digest

            def verify_boundary_digest(self) -> str:
                return self.digest

        source_ref = _ref("output-bound-source")
        foundation_ref = _ref("output-bound-foundation")

        def sample_resources() -> dict[str, int]:
            return {
                "configured_pools": 0,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
                "process_count": 0,
                "rss_bytes": 1,
                "state_scratch_bytes": 0,
            }

        def probe_gpu() -> dict[str, object]:
            return {
                "assigned_name": "NVIDIA GeForce RTX 5080",
                "assigned_uuid": runner.ASSIGNED_GPU_UUID,
                "logical_device_count": 1,
                "logical_device_index": 0,
                "unassigned_compute_processes": 0,
                "unassigned_name": "NVIDIA GeForce RTX 5070",
                "unassigned_uuid": runner.UNASSIGNED_GPU_UUID,
            }

        def fake_generate(
            io_owner: LocalQwenIO,
            prompt: str,
        ) -> FrozenQwenGenerationRecord:
            io_owner._generation_calls += 1
            count = io_owner._test_output_tokens
            return FrozenQwenGenerationRecord(
                model_ref=io_owner.model_ref,
                tokenizer_ref=io_owner.tokenizer_ref,
                generation_config_ref=io_owner.generation_config_ref,
                prompt_ref=runner.content_ref({"prompt": prompt}),
                prompt_tokens=1,
                generated_token_ids=tuple(range(count)),
                response="bounded synthetic response",
            )

        def make_boundary(output_tokens: int) -> tuple[
            LocalQwenIO,
            runner.GuardedLocalQwenBoundary,
            runner.ResourceLedger,
        ]:
            model = torch.nn.Linear(2, 2, bias=False)
            model.requires_grad_(False)
            model.eval()
            io_owner = LocalQwenIO(
                model,
                object(),
                embedding_batch_size=1,
                max_input_tokens=4_096,
                max_new_tokens=160,
                enable_thinking=False,
                model_ref=runner.QUALIFIED_MODEL_REF,
                tokenizer_ref=runner.QUALIFIED_TOKENIZER_REF,
            )
            io_owner._test_output_tokens = output_tokens
            resources = runner.ResourceLedger(runner.RunMode.QUALIFICATION)
            foundation = FoundationGuard(foundation_ref)
            boundary = runner.GuardedLocalQwenBoundary(
                io_owner,
                foundation_guard=foundation,
                expected_foundation_digest=foundation_ref,
                resources=resources,
                verify_source_seal=lambda: source_ref,
                expected_source_seal_ref=source_ref,
                sample_resources=sample_resources,
                probe_gpu=probe_gpu,
                verify_exact_foundation=lambda: foundation_ref,
            )
            return io_owner, boundary, resources

        with mock.patch.object(
            LocalQwenIO,
            "generate_record",
            new=fake_generate,
        ):
            io_owner, boundary, resources = make_boundary(160)
            prompt = "exact 160-token guarded output"
            with boundary.attempt(
                task_id=_ref("output-bound-160-task"),
                arm="N0_NATIVE_NO_HISTORY",
                stage_ref=_ref("output-bound-160-stage"),
                proposal_prompt_ref=runner.content_ref({"prompt": prompt}),
                maximum_generation_calls=2,
                maximum_embedding_rows=0,
            ):
                record = io_owner.generate_record(prompt)
            resources.record_generation(
                record.prompt_tokens,
                len(record.generated_token_ids),
            )
            self.assertEqual(resources.output_tokens, 160)
            boundary.close()

            io_owner, boundary, _resources = make_boundary(161)
            prompt = "rejected 161-token guarded output"
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("output-bound-161-task"),
                    arm="N0_NATIVE_NO_HISTORY",
                    stage_ref=_ref("output-bound-161-stage"),
                    proposal_prompt_ref=runner.content_ref({"prompt": prompt}),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=0,
                ):
                    io_owner.generate_record(prompt)
            with self.assertRaises(runner.V2IntegrityStop):
                boundary.close()
            self.assertNotIn(boundary._INSTANCE_MARKER, io_owner.__dict__)

    def test_exact_local_qwen_dispatch_is_guarded_and_restored(self) -> None:
        calls = {"embed": 0, "generate": 0}
        embed_inputs: list[object] = []

        class Matrix:
            def __init__(self, rows: int) -> None:
                self.shape = (rows, 3)

        def fake_embed(_io: LocalQwenIO, texts: object) -> Matrix:
            calls["embed"] += 1
            embed_inputs.append(texts)
            return Matrix(len(tuple(texts)))

        def fake_generate(
            _io: LocalQwenIO,
            prompt: str,
        ) -> FrozenQwenGenerationRecord:
            calls["generate"] += 1
            _io._generation_calls += 1
            return FrozenQwenGenerationRecord(
                model_ref=None,
                tokenizer_ref=None,
                generation_config_ref=_ref("guarded-generation-config"),
                prompt_ref=runner.content_ref({"prompt": prompt}),
                prompt_tokens=1,
                generated_token_ids=(1,),
                response="bounded synthetic response",
            )

        class FoundationGuard:
            def __init__(self, digest: str) -> None:
                self.probe_digest = digest
                self.final_digest = digest
                self.probes = 0
                self.exact_verifications = 0
                self.final_verifications = 0

            def probe(self) -> str:
                self.probes += 1
                return self.probe_digest

            def verify_boundary_digest(self) -> str:
                self.final_verifications += 1
                return self.final_digest

            def verify_exact(self) -> str:
                self.exact_verifications += 1
                return self.final_digest

        def good_gpu() -> dict[str, object]:
            assignment = runner._frozen_manifest_gpu_assignment()
            return {
                "assigned_name": assignment["assigned_name"],
                "assigned_uuid": runner.ASSIGNED_GPU_UUID,
                "logical_device_count": 1,
                "logical_device_index": 0,
                "unassigned_compute_processes": 0,
                "unassigned_name": assignment["unassigned_name"],
                "unassigned_uuid": runner.UNASSIGNED_GPU_UUID,
            }

        def sample() -> dict[str, int]:
            return {
                "configured_pools": 0,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
                "process_count": 0,
                "rss_bytes": 0,
                "state_scratch_bytes": 0,
            }

        def make_boundary() -> tuple[
            LocalQwenIO,
            runner.GuardedLocalQwenBoundary,
            runner.ResourceLedger,
            FoundationGuard,
            dict[str, str],
            dict[str, dict[str, object]],
        ]:
            io = object.__new__(LocalQwenIO)
            io._generation_calls = 0
            resources = runner.ResourceLedger(runner.RunMode.QUALIFICATION)
            expected_digest = _ref("guarded-foundation-digest")
            foundation = FoundationGuard(expected_digest)
            source = {"value": _ref("guarded-source-seal")}
            gpu = {"value": good_gpu()}
            boundary = runner.GuardedLocalQwenBoundary(
                io,
                foundation_guard=foundation,
                expected_foundation_digest=expected_digest,
                resources=resources,
                verify_source_seal=lambda: source["value"],
                expected_source_seal_ref=source["value"],
                sample_resources=sample,
                probe_gpu=lambda: gpu["value"],
                verify_exact_foundation=foundation.verify_exact,
            )
            return io, boundary, resources, foundation, source, gpu

        def assert_poisoned_close_restores(
            io: LocalQwenIO,
            boundary: runner.GuardedLocalQwenBoundary,
            resources: runner.ResourceLedger,
        ) -> None:
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_GUARD_FINAL_BOUNDARY_FAILED",
            ) as raised:
                boundary.close()
            self.assertEqual(
                raised.exception.boundary_code_chain,
                ("QWEN_GUARD_FINAL_BOUNDARY_FAILED",),
            )
            self.assertNotIn("embed", io.__dict__)
            self.assertNotIn("generate_record", io.__dict__)
            self.assertNotIn(
                runner.GuardedLocalQwenBoundary._INSTANCE_MARKER,
                io.__dict__,
            )
            self.assertIsNone(resources._generation_reconciler)

        with (
            mock.patch.object(LocalQwenIO, "embed", new=fake_embed),
            mock.patch.object(
                LocalQwenIO,
                "generate_record",
                new=fake_generate,
            ),
        ):
            for shadow in ("embed", "generate_record"):
                with self.subTest(pre_shadow=shadow):
                    io = object.__new__(LocalQwenIO)
                    setattr(io, shadow, lambda *_args: None)
                    with self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "already shadowed",
                    ):
                        runner.GuardedLocalQwenBoundary(
                            io,
                            foundation_guard=FoundationGuard(
                                _ref("guarded-foundation-digest")
                            ),
                            expected_foundation_digest=_ref(
                                "guarded-foundation-digest"
                            ),
                            resources=runner.ResourceLedger(
                                runner.RunMode.QUALIFICATION
                            ),
                            verify_source_seal=lambda: _ref(
                                "guarded-source-seal"
                            ),
                            expected_source_seal_ref=_ref(
                                "guarded-source-seal"
                            ),
                            sample_resources=sample,
                            probe_gpu=good_gpu,
                        )

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            before = dict(calls)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_GENERATION_BOUNDARY_FAILED",
            ):
                io.generate_record("outside")
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_EMBEDDING_BOUNDARY_FAILED",
            ):
                io.embed(("outside",))
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            before = dict(calls)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("wrong-prompt-task"),
                    arm="N0_NATIVE_NO_HISTORY",
                    stage_ref=_ref("wrong-prompt-stage"),
                    proposal_prompt_ref=_ref("wrong-frozen-prompt"),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=0,
                ):
                    io.generate_record("another prompt")
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            resources.generation_calls = resources.generation_ceiling
            io._generation_calls = resources.generation_ceiling
            boundary._successful_generation_calls = resources.generation_ceiling
            boundary._reserved_generation_calls = resources.generation_ceiling
            boundary._reconciled_generation_calls = resources.generation_ceiling
            before = dict(calls)
            prompt = "generation ceiling prompt"
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("generation-ceiling-task"),
                    arm="N0_NATIVE_NO_HISTORY",
                    stage_ref=_ref("generation-ceiling-stage"),
                    proposal_prompt_ref=runner.content_ref({"prompt": prompt}),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=0,
                ):
                    io.generate_record(prompt)
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            resources.embedding_rows = resources.embedding_ceiling
            before = dict(calls)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_PLANNING_CONTEXT_FAILED",
            ):
                with boundary.planning(
                    task_id=_ref("embedding-ceiling-task"),
                    maximum_embedding_rows=1,
                    planning_ref=_ref("embedding-ceiling-plan"),
                ):
                    io.embed(("one row",))
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            for drift in ("source", "foundation", "gpu"):
                with self.subTest(drift=drift):
                    io, boundary, _resources, foundation, source, gpu = make_boundary()
                    expected_source = source["value"]
                    expected_foundation = foundation.final_digest
                    expected_gpu = dict(gpu["value"])
                    if drift == "source":
                        source["value"] = _ref("drifted-source")
                    elif drift == "foundation":
                        foundation.final_digest = _ref("drifted-foundation")
                    else:
                        gpu["value"] = {**gpu["value"], "logical_device_count": 2}
                    before = dict(calls)
                    prompt = f"{drift} drift prompt"
                    with self.assertRaises(runner.V2IntegrityStop):
                        with boundary.attempt(
                            task_id=_ref(f"{drift}-drift-task"),
                            arm="N0_NATIVE_NO_HISTORY",
                            stage_ref=_ref(f"{drift}-drift-stage"),
                            proposal_prompt_ref=runner.content_ref(
                                {"prompt": prompt}
                            ),
                            maximum_generation_calls=2,
                            maximum_embedding_rows=0,
                        ):
                            io.generate_record(prompt)
                    self.assertEqual(calls, before)
                    source["value"] = expected_source
                    foundation.final_digest = expected_foundation
                    gpu["value"] = expected_gpu
                    assert_poisoned_close_restores(
                        io,
                        boundary,
                        _resources,
                    )

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            before = dict(calls)
            with self.assertRaisesRegex(
                ValueError,
                "generation allowance differs",
            ):
                with boundary.attempt(
                    task_id=_ref("wrong-generation-allowance-task"),
                    arm="N0_NATIVE_NO_HISTORY",
                    stage_ref=_ref("wrong-generation-allowance-stage"),
                    proposal_prompt_ref=_ref("wrong-generation-allowance-prompt"),
                    maximum_generation_calls=1,
                    maximum_embedding_rows=0,
                ):
                    pass
            with self.assertRaisesRegex(
                ValueError,
                "runtime embedding allowance differs",
            ):
                with boundary.attempt(
                    task_id=_ref("wrong-runtime-allowance-task"),
                    arm="IO_FULL_ORDINARY_12",
                    stage_ref=_ref("wrong-runtime-allowance-stage"),
                    proposal_prompt_ref=_ref("wrong-runtime-allowance-prompt"),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=0,
                ):
                    pass
            self.assertEqual(calls, before)
            boundary.close()

            for rows in (14, 16):
                with self.subTest(runtime_rows=rows):
                    io, boundary, resources, _foundation, _source, _gpu = (
                        make_boundary()
                    )
                    before = dict(calls)
                    with self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "QWEN_ATTEMPT_CONTEXT_FAILED",
                    ):
                        with boundary.attempt(
                            task_id=_ref(f"wrong-runtime-rows-{rows}-task"),
                            arm="IN_FULL_NEUTRAL_12",
                            stage_ref=_ref(f"wrong-runtime-rows-{rows}-stage"),
                            proposal_prompt_ref=_ref(
                                f"wrong-runtime-rows-{rows}-prompt"
                            ),
                            maximum_generation_calls=2,
                            maximum_embedding_rows=15,
                        ):
                            io.embed(tuple(f"row {index}" for index in range(rows)))
                    self.assertEqual(calls, before)
                    assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            before = dict(calls)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("stateless-embed-task"),
                    arm="N0_NATIVE_NO_HISTORY",
                    stage_ref=_ref("stateless-embed-stage"),
                    proposal_prompt_ref=_ref("stateless-embed-prompt"),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=0,
                ):
                    io.embed(("forbidden stateless row",))
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            resources.runtime_embedding_rows = resources.runtime_embedding_ceiling
            before = dict(calls)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("runtime-embedding-ceiling-task"),
                    arm="IO_FULL_ORDINARY_12",
                    stage_ref=_ref("runtime-embedding-ceiling-stage"),
                    proposal_prompt_ref=_ref("runtime-embedding-ceiling-prompt"),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=15,
                ):
                    io.embed(tuple(f"row {index}" for index in range(15)))
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            before_embed = calls["embed"]
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("second-runtime-embed-task"),
                    arm="IN_FULL_NEUTRAL_12",
                    stage_ref=_ref("second-runtime-embed-stage"),
                    proposal_prompt_ref=_ref("second-runtime-embed-prompt"),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=15,
                ):
                    rows = tuple(f"row {index}" for index in range(15))
                    io.embed(rows)
                    io.embed(rows)
            self.assertEqual(calls["embed"], before_embed + 1)
            self.assertEqual(resources.runtime_embedding_rows, 15)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            before = dict(calls)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_PLANNING_CONTEXT_FAILED",
            ):
                with boundary.planning(
                    task_id=_ref("planning-generation-task"),
                    planning_ref=_ref("planning-generation-ref"),
                    maximum_embedding_rows=1,
                ):
                    io.generate_record("forbidden planning generation")
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            before = dict(calls)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_PLANNING_CONTEXT_FAILED",
            ):
                with boundary.planning(
                    task_id=_ref("planning-nonstring-task"),
                    planning_ref=_ref("planning-nonstring-ref"),
                    maximum_embedding_rows=2,
                ):
                    io.embed(("one valid row", 2))
            self.assertEqual(calls, before)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_PLANNING_CONTEXT_FAILED",
            ):
                with boundary.planning(
                    task_id=_ref("empty-planning-task"),
                    planning_ref=_ref("empty-planning-ref"),
                    maximum_embedding_rows=1,
                ):
                    pass
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            prompt = "in-context reconciliation proposal"
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("in-context-reconciliation-task"),
                    arm="N0_NATIVE_NO_HISTORY",
                    stage_ref=_ref("in-context-reconciliation-stage"),
                    proposal_prompt_ref=runner.content_ref({"prompt": prompt}),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=0,
                ):
                    record = io.generate_record(prompt)
                    resources.record_generation(
                        record.prompt_tokens,
                        len(record.generated_token_ids),
                    )
            self.assertEqual(resources.generation_calls, 0)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            prompt = "stateful pair without embedding"
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("missing-runtime-embedding-task"),
                    arm="IO_FULL_ORDINARY_12",
                    stage_ref=_ref("missing-runtime-embedding-stage"),
                    proposal_prompt_ref=runner.content_ref({"prompt": prompt}),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=15,
                ):
                    io.generate_record(prompt)
                    io.generate_record("stateful execution without embedding")
            self.assertEqual(resources.runtime_embedding_rows, 0)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            first_prompt = "two-call guarded proposal"
            with boundary.attempt(
                task_id=_ref("two-call-guard-task"),
                arm="IO_FULL_ORDINARY_12",
                stage_ref=_ref("two-call-guard-stage"),
                proposal_prompt_ref=runner.content_ref(
                    {"prompt": first_prompt}
                ),
                maximum_generation_calls=2,
                maximum_embedding_rows=15,
            ):
                first_record = io.generate_record(first_prompt)
                runtime_matrix = io.embed(
                    tuple(f"runtime row {index}" for index in range(15))
                )
                self.assertEqual(runtime_matrix.shape, (15, 3))
                second_record = io.generate_record("two-call guarded execution")
            for record in (first_record, second_record):
                resources.record_generation(
                    record.prompt_tokens,
                    len(record.generated_token_ids),
                )
            next_prompt = "reconciled next proposal"
            with boundary.attempt(
                task_id=_ref("reconciled-next-task"),
                arm="N0_NATIVE_NO_HISTORY",
                stage_ref=_ref("reconciled-next-stage"),
                proposal_prompt_ref=runner.content_ref(
                    {"prompt": next_prompt}
                ),
                maximum_generation_calls=2,
                maximum_embedding_rows=0,
            ):
                next_record = io.generate_record(next_prompt)
            resources.record_generation(
                next_record.prompt_tokens,
                len(next_record.generated_token_ids),
            )
            self.assertEqual(resources.generation_calls, 3)
            self.assertEqual(resources.embedding_rows, 0)
            self.assertEqual(resources.runtime_embedding_rows, 15)
            boundary.close()

            io, boundary, resources, _foundation, _source, _gpu = make_boundary()
            unreconciled_prompt = "unreconciled proposal"
            with boundary.attempt(
                task_id=_ref("unreconciled-task"),
                arm="N0_NATIVE_NO_HISTORY",
                stage_ref=_ref("unreconciled-stage"),
                proposal_prompt_ref=runner.content_ref(
                    {"prompt": unreconciled_prompt}
                ),
                maximum_generation_calls=2,
                maximum_embedding_rows=0,
            ):
                unreconciled_record = io.generate_record(unreconciled_prompt)
            before = dict(calls)
            blocked_prompt = "blocked unreconciled next proposal"
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "QWEN_ATTEMPT_CONTEXT_FAILED",
            ):
                with boundary.attempt(
                    task_id=_ref("blocked-unreconciled-task"),
                    arm="N0_NATIVE_NO_HISTORY",
                    stage_ref=_ref("blocked-unreconciled-stage"),
                    proposal_prompt_ref=runner.content_ref(
                        {"prompt": blocked_prompt}
                    ),
                    maximum_generation_calls=2,
                    maximum_embedding_rows=0,
                ):
                    io.generate_record(blocked_prompt)
            self.assertEqual(calls, before)
            self.assertEqual(unreconciled_record.prompt_tokens, 1)
            assert_poisoned_close_restores(io, boundary, resources)

            io, boundary, resources, foundation, _source, _gpu = make_boundary()
            with boundary.planning(
                task_id=_ref("successful-planning-task"),
                planning_ref=_ref("successful-planning-ref"),
                maximum_embedding_rows=2,
            ):
                matrix = io.embed(["first ranking row", "second ranking row"])
                self.assertEqual(matrix.shape, (2, 3))
            prompt = "malformed stateful proposal"
            with boundary.attempt(
                task_id=_ref("successful-guard-task"),
                arm="IO_FULL_ORDINARY_12",
                stage_ref=_ref("successful-guard-stage"),
                proposal_prompt_ref=runner.content_ref({"prompt": prompt}),
                maximum_generation_calls=2,
                maximum_embedding_rows=15,
            ):
                record = io.generate_record(prompt)
            resources.record_generation(
                record.prompt_tokens,
                len(record.generated_token_ids),
            )
            self.assertEqual(
                boundary.verify_phase_boundary(),
                _ref("guarded-foundation-digest"),
            )
            boundary.close()
            self.assertNotIn("embed", io.__dict__)
            self.assertNotIn("generate_record", io.__dict__)
            self.assertNotIn(boundary._INSTANCE_MARKER, io.__dict__)
            self.assertIs(io.embed.__func__, fake_embed)
            self.assertIs(io.generate_record.__func__, fake_generate)
            self.assertEqual(resources.generation_calls, 1)
            self.assertEqual(resources.embedding_rows, 2)
            self.assertEqual(resources.runtime_embedding_rows, 0)
            self.assertEqual(
                embed_inputs[-1],
                ("first ranking row", "second ranking row"),
            )
            self.assertIs(type(embed_inputs[-1]), tuple)
            self.assertEqual(foundation.exact_verifications, 4)
            self.assertEqual(foundation.final_verifications, 2)
            self.assertEqual(resources.samples, 8)
            boundary.close()


class ClosedResultAndOrchestrationBoundaryTests(unittest.TestCase):
    def test_manifest_successor_is_fresh_and_consumed_predecessor_is_a_source(
        self,
    ) -> None:
        successor = (
            runner.REPOSITORY_ROOT
            / "experiments/manifests/"
            "high-level-multidomain-v2-history-conditioning-isolation-r6.json"
        )
        consumed_r5 = (
            runner.REPOSITORY_ROOT
            / "experiments/manifests/"
            "high-level-multidomain-v2-history-conditioning-isolation-r5.json"
        )
        consumed_r4 = (
            runner.REPOSITORY_ROOT
            / "experiments/manifests/"
            "high-level-multidomain-v2-history-conditioning-isolation-r4.json"
        )
        consumed_r3 = (
            runner.REPOSITORY_ROOT
            / "experiments/manifests/"
            "high-level-multidomain-v2-history-conditioning-isolation-r3.json"
        )
        consumed_r2 = (
            runner.REPOSITORY_ROOT
            / "experiments/manifests/"
            "high-level-multidomain-v2-history-conditioning-isolation-r2.json"
        )
        predecessor = (
            runner.REPOSITORY_ROOT
            / "experiments/manifests/"
            "high-level-multidomain-v2-history-conditioning-isolation-r1.json"
        )
        base = (
            runner.REPOSITORY_ROOT
            / "experiments/manifests/"
            "high-level-multidomain-v2-history-conditioning-isolation.json"
        )
        predecessor_relative = str(
            predecessor.relative_to(runner.REPOSITORY_ROOT)
        )
        successor_relative = str(successor.relative_to(runner.REPOSITORY_ROOT))
        self.assertEqual(runner.MANIFEST_PATH, successor)
        self.assertEqual(runner.CONSUMED_R5_MANIFEST_PATH, consumed_r5)
        self.assertEqual(runner.CONSUMED_R4_MANIFEST_PATH, consumed_r4)
        self.assertEqual(runner.CONSUMED_R3_MANIFEST_PATH, consumed_r3)
        self.assertEqual(runner.CONSUMED_R2_MANIFEST_PATH, consumed_r2)
        self.assertEqual(runner.CONSUMED_PREDECESSOR_MANIFEST_PATH, predecessor)
        self.assertEqual(runner.CONSUMED_BASE_MANIFEST_PATH, base)
        self.assertIn(predecessor_relative, runner.SOURCE_MANIFEST_INVENTORY)
        self.assertIn(
            str(consumed_r5.relative_to(runner.REPOSITORY_ROOT)),
            runner.SOURCE_MANIFEST_INVENTORY,
        )
        self.assertIn(
            str(consumed_r4.relative_to(runner.REPOSITORY_ROOT)),
            runner.SOURCE_MANIFEST_INVENTORY,
        )
        self.assertIn(
            str(consumed_r3.relative_to(runner.REPOSITORY_ROOT)),
            runner.SOURCE_MANIFEST_INVENTORY,
        )
        self.assertIn(
            str(consumed_r2.relative_to(runner.REPOSITORY_ROOT)),
            runner.SOURCE_MANIFEST_INVENTORY,
        )
        self.assertIn(
            str(base.relative_to(runner.REPOSITORY_ROOT)),
            runner.SOURCE_MANIFEST_INVENTORY,
        )
        self.assertEqual(
            hashlib.sha256(base.read_bytes()).hexdigest(),
            runner.CONSUMED_BASE_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(predecessor.read_bytes()).hexdigest(),
            runner.CONSUMED_PREDECESSOR_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(consumed_r5.read_bytes()).hexdigest(),
            runner.CONSUMED_R5_MANIFEST_SHA256,
        )
        self.assertEqual(
            runner.CONSUMED_R5_MANIFEST_REF,
            "sha256:d7a5ecebab5021fbc8a4639fac8e95dd30509ab17ef52516932fa167f77f2275",
        )
        self.assertEqual(
            runner.CONSUMED_R5_SOURCE_SEAL_REF,
            "sha256:ff1439390262b241025002b7236c1b92027526339350694c825928004547ae3c",
        )
        self.assertEqual(
            hashlib.sha256(consumed_r4.read_bytes()).hexdigest(),
            runner.CONSUMED_R4_MANIFEST_SHA256,
        )
        self.assertEqual(
            runner.CONSUMED_R4_MANIFEST_REF,
            "sha256:4e27affc0e719f43c3cd7333baf1b4052e5aa6cc331337464f2d026d25b74ecc",
        )
        self.assertEqual(
            runner.CONSUMED_R4_SOURCE_SEAL_REF,
            "sha256:aa7882f8a86b5a2619f1191bb96e6ad8a3fa2e4c2cf75189bf1802ec522d3f9e",
        )
        self.assertEqual(
            runner._source_manifest_successor_disposition()["consumed"][  # type: ignore[index]
                "qualification_failures"
            ][-1],
            {
                "attempts": 48,
                "evaluation_admission_created": False,
                "evaluation_identity_unreached": (
                    "angler.high-level-multidomain.v2-history-conditioning-"
                    "isolation.evaluation.v3"
                ),
                "evaluation_seed_created": False,
                "evaluator_judgments": 48,
                "feedback_applications": 46,
                "identity": (
                    "angler.high-level-multidomain.v2-history-conditioning-"
                    "isolation.qualification.v5"
                ),
                "ledger_raw_sha256": (
                    "615ed0ff45bbb57dc58097799b1c0dc7b7694892ac283a2d864fd5c9241f35bf"
                ),
                "model_generations": 94,
                "protocol_identity": (
                    "angler.high-level-multidomain.v2-history-conditioning-"
                    "isolation.v3"
                ),
                "qualification_result_created": False,
                "release_raw_sha256": (
                    "671ce190a948aee2d195135e251c28e7556fa99e5ded3f1b67c3bbe3cfbe30bf"
                ),
                "release_ref": (
                    "sha256:07b3feb6d124399e3763ecc4f3dd38a8855d6cc64175d1943bbe151ba7d74257"
                ),
                "replicate_commitment": (
                    "sha256:2382474330f7e2d63e7e02bbe4d02112eecd5ce6d8c5be810b8b3671b9e030af"
                ),
                "scratch_root": (
                    "/opt/angler/scratch/high-level-multidomain-v2-history-"
                    "conditioning-isolation-r5"
                ),
                "seed": 2026090214,
                "state_root": (
                    "/opt/angler/state/project-angler/high-level-multidomain-"
                    "v2-history-conditioning-isolation-r5"
                ),
                "successful_judgments": 5,
                "unsuccessful_judgments": 43,
            },
        )
        self.assertEqual(
            hashlib.sha256(consumed_r3.read_bytes()).hexdigest(),
            runner.CONSUMED_R3_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(consumed_r2.read_bytes()).hexdigest(),
            runner.CONSUMED_R2_MANIFEST_SHA256,
        )
        self.assertNotIn(successor_relative, runner.SOURCE_MANIFEST_INVENTORY)
        self.assertIn(successor, runner.expected_live_paths())
        self.assertNotIn(predecessor, runner.expected_live_paths())
        self.assertNotIn(consumed_r2, runner.expected_live_paths())
        self.assertNotIn(consumed_r3, runner.expected_live_paths())
        self.assertNotIn(consumed_r4, runner.expected_live_paths())
        self.assertNotIn(consumed_r5, runner.expected_live_paths())

    def test_source_manifest_is_reviewable_and_create_once_under_injection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            repository = root / "repository"
            repository.mkdir(mode=0o700)
            for relative in runner.SOURCE_MANIFEST_INVENTORY:
                source = ROOT / relative
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                target.chmod(0o664)
            sources = runner._source_inventory_snapshot(
                repository,
                injected_root=True,
            )
            self.assertEqual(
                runner.validate_frozen_construction_sources(sources),
                {
                    str(
                        runner.ACTIVE_LEAF_PATH.relative_to(
                            runner.REPOSITORY_ROOT
                        )
                    ): runner.CURRENT_AUTHORIZED_LEAF_SHA256,
                    str(
                        runner.EVALUATOR_PATH.relative_to(
                            runner.REPOSITORY_ROOT
                        )
                    ): runner.FROZEN_V2_EVALUATOR_SHA256,
                    str(
                        runner.R2_RUNNER_PATH.relative_to(
                            runner.REPOSITORY_ROOT
                        )
                    ): runner.FROZEN_R2_RUNNER_SHA256,
                },
            )
            predecessor_relative = str(
                runner.CONSUMED_PREDECESSOR_MANIFEST_PATH.relative_to(
                    runner.REPOSITORY_ROOT
                )
            )
            predecessor_bytes = (repository / predecessor_relative).read_bytes()
            self.assertEqual(
                [row for row in sources if row["path"] == predecessor_relative],
                [
                    {
                        "bytes": len(predecessor_bytes),
                        "mode": "0664",
                        "path": predecessor_relative,
                        "sha256": hashlib.sha256(
                            predecessor_bytes
                        ).hexdigest(),
                    }
                ],
            )
            successor_relative = str(
                runner.MANIFEST_PATH.relative_to(runner.REPOSITORY_ROOT)
            )
            self.assertFalse(
                any(row["path"] == successor_relative for row in sources)
            )
            preflight = _synthetic_preflight_receipt(sources)
            self.assertEqual(
                runner.validate_foundation_mirror_preflight_receipt(
                    preflight,
                    repository_root=repository,
                    injected_root=True,
                ),
                preflight,
            )
            candidate = runner.build_source_manifest_candidate(
                preflight,
                repository,
                injected_root=True,
            )
            self.assertEqual(candidate["preflight_receipt"], preflight)
            self.assertEqual(
                candidate["declared_v2_delta"]["runtime_contract_change"],
                "INHERITED_ISOLATION_LAUNCH_AND_REBUILDABLE_PER_ATTEMPT_"
                "COGNEE_PROJECTION_INCARNATIONS",
            )
            self.assertEqual(
                candidate["declared_v2_delta"][
                    "ordinary_recall_selection_change"
                ],
                "FULL_BOUNDED_CANONICAL_INVENTORY_REJOIN_THEN_FIRST_TWELVE_"
                "OBSERVED_OR_VALIDATED_LEARNER_VISIBLE_RECORDS",
            )
            self.assertEqual(
                candidate["declared_v2_delta"][
                    "cleanup_observability_change"
                ],
                "CAPPED_ALLOWLISTED_BOUNDARY_CODE_CHAIN_WITH_PRIVATE_"
                "EXCEPTION_RETENTION_CLEARED",
            )
            self.assertEqual(
                candidate["declared_v2_delta"][
                    "failure_scope_retention_change"
                ],
                "DURABLE_EXACT_PRE_POST_STAGE_AND_DISPOSABLE_FAILURE_"
                "OWNERSHIP_WITHOUT_DISPOSAL_CLAIM",
            )
            self.assertEqual(
                candidate["declared_v2_delta"]["journal_reopen_change"],
                "LOCKING_AWARE_READ_ONLY_EXACT_SCHEMA_VALIDATION_WITHOUT_"
                "MUTATION",
            )
            self.assertEqual(
                candidate["successor_disposition"],
                runner._source_manifest_successor_disposition(),
            )
            self.assertEqual(
                runner.validate_source_manifest(
                    candidate,
                    repository_root=repository,
                    injected_root=True,
                ),
                candidate,
            )

            mismatched_leaf_authority = json.loads(json.dumps(candidate))
            mismatched_leaf_authority["authorized_leaf_sha256"] = _raw(
                "mismatched-authorized-leaf"
            )
            mismatched_payload = dict(mismatched_leaf_authority)
            mismatched_payload.pop("manifest_ref")
            mismatched_leaf_authority["manifest_ref"] = runner.canonical_ref(
                "source-manifest", mismatched_payload
            )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "authorized leaf/source row differs",
            ):
                runner.validate_source_manifest(
                    mismatched_leaf_authority,
                    repository_root=repository,
                    injected_root=True,
                )

            output_161 = json.loads(json.dumps(candidate))
            output_161["budgets"]["max_output_tokens"] = 161
            output_161_payload = dict(output_161)
            output_161_payload.pop("manifest_ref")
            output_161["manifest_ref"] = runner.canonical_ref(
                "source-manifest", output_161_payload
            )
            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "source manifest identity/delta differs",
            ):
                runner.validate_source_manifest(
                    output_161,
                    repository_root=repository,
                    injected_root=True,
                )
            manifest_path = root / "source-manifest.json"
            published, digest = runner.publish_source_manifest_create_once(
                candidate,
                path=manifest_path,
                repository_root=repository,
                injected_paths=True,
            )
            self.assertEqual(published, candidate)
            self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o664)
            loaded, loaded_digest, source_seal = runner.load_source_manifest(
                path=manifest_path,
                repository_root=repository,
                injected_paths=True,
            )
            self.assertEqual(loaded, candidate)
            self.assertEqual(loaded_digest, digest)
            self.assertEqual(
                source_seal["source_seal_ref"],
                runner.canonical_ref(
                    "source-seal",
                    {
                        key: value
                        for key, value in source_seal.items()
                        if key != "source_seal_ref"
                    },
                ),
            )
            with self.assertRaises(FileExistsError):
                runner.publish_source_manifest_create_once(
                    candidate,
                    path=manifest_path,
                    repository_root=repository,
                    injected_paths=True,
                )

            with self.assertRaises(TypeError):
                runner.build_source_manifest_candidate(
                    repository_root=repository,
                    injected_root=True,
                )
            missing_receipt = dict(candidate)
            missing_receipt.pop("preflight_receipt")
            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "source manifest fields differ",
            ):
                runner.validate_source_manifest(
                    missing_receipt,
                    repository_root=repository,
                    injected_root=True,
                )

            rebound_disposition = json.loads(json.dumps(candidate))
            rebound_disposition["successor_disposition"]["fresh"][
                "qualification_seed"
            ] = runner.CONSUMED_QUALIFICATION_SEED
            rebound_payload = dict(rebound_disposition)
            rebound_payload.pop("manifest_ref")
            rebound_disposition["manifest_ref"] = runner.canonical_ref(
                "source-manifest", rebound_payload
            )
            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "source manifest identity/delta differs",
            ):
                runner.validate_source_manifest(
                    rebound_disposition,
                    repository_root=repository,
                    injected_root=True,
                )

            stale_sources = json.loads(json.dumps(sources))
            stale_sources[0]["sha256"] = _raw("stale-preflight-source")
            stale_preflight = _synthetic_preflight_receipt(stale_sources)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "preflight source inventory changed",
            ):
                runner.validate_foundation_mirror_preflight_receipt(
                    stale_preflight,
                    source_inventory=sources,
                    repository_root=repository,
                    injected_root=True,
                )

            predecessor_path = repository / predecessor_relative
            predecessor_path.write_bytes(predecessor_bytes + b"\n")
            predecessor_path.chmod(0o664)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "consumed source manifest identity changed",
            ):
                runner._source_inventory_snapshot(
                    repository,
                    injected_root=True,
                )
            predecessor_path.write_bytes(predecessor_bytes)
            predecessor_path.chmod(0o664)

            changed_source = repository / runner.SOURCE_MANIFEST_INVENTORY[0]
            changed_source.write_bytes(changed_source.read_bytes() + b"\n")
            changed_source.chmod(0o664)
            with self.assertRaises(runner.V2IntegrityStop):
                runner.validate_source_manifest(
                    candidate,
                    repository_root=repository,
                    injected_root=True,
                )

    def test_frozen_construction_source_drift_stops_at_inventory_choke_point(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve() / "repository"
            repository.mkdir(mode=0o700)
            for relative in runner.SOURCE_MANIFEST_INVENTORY:
                source = ROOT / relative
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                target.chmod(0o664)

            protected = (
                str(
                    runner.ACTIVE_LEAF_PATH.relative_to(
                        runner.REPOSITORY_ROOT
                    )
                ),
                str(
                    runner.EVALUATOR_PATH.relative_to(
                        runner.REPOSITORY_ROOT
                    )
                ),
                str(
                    runner.R2_RUNNER_PATH.relative_to(
                        runner.REPOSITORY_ROOT
                    )
                ),
            )
            for relative in protected:
                with self.subTest(relative=relative):
                    target = repository / relative
                    original = target.read_bytes()
                    try:
                        target.write_bytes(original + b"\n")
                        target.chmod(0o664)
                        with self.assertRaisesRegex(
                            runner.V2IntegrityStop,
                            "frozen construction source authority changed",
                        ):
                            runner._source_inventory_snapshot(
                                repository,
                                injected_root=True,
                            )
                    finally:
                        target.write_bytes(original)
                        target.chmod(0o664)

            runner._source_inventory_snapshot(
                repository,
                injected_root=True,
            )

    def test_live_preflight_source_drift_stops_before_process_factory(self) -> None:
        process_factory = mock.Mock()
        with (
            mock.patch.object(runner, "assert_all_live_paths_absent"),
            mock.patch.object(
                runner,
                "_source_inventory_snapshot",
                side_effect=runner.V2IntegrityStop(
                    "frozen construction source authority changed"
                ),
            ),
            self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "frozen construction source authority changed",
            ),
        ):
            runner.run_foundation_mirror_preflight_live(
                process_factory=process_factory
            )
        process_factory.assert_not_called()

    def test_injected_live_topology_rejects_extra_children_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state_root = root / "state"
            sealed_root = state_root / "sealed"
            qualification_runtime = state_root / "qualification-v1"
            scratch_root = root / "scratch"
            for path in (
                state_root,
                sealed_root,
                qualification_runtime,
                scratch_root,
            ):
                path.mkdir(mode=0o700, parents=True, exist_ok=True)
                path.chmod(0o700)
            for child in ("arm-runtime", "cognee-scopes", "genesis"):
                path = qualification_runtime / child
                path.mkdir(mode=0o700)
                path.chmod(0o700)
            for child in runner._frozen_manifest_topology()["scratch_children"]:
                path = scratch_root / child
                path.mkdir(mode=0o700)
                path.chmod(0o700)

            def regular(path: Path, mode: int) -> None:
                path.write_bytes(b"injected-topology-fixture")
                path.chmod(mode)

            manifest = root / "source-manifest.json"
            qualification_result = root / "qualification-result.json"
            qualification_release = (
                sealed_root / "qualification-release-claim-v1.json"
            )
            regular(manifest, 0o664)
            qualification_result.write_bytes(runner.canonical_json_bytes({}))
            qualification_result.chmod(0o600)
            regular(qualification_release, 0o600)
            regular(qualification_runtime / "attempt-evidence.sqlite3", 0o600)
            paths: dict[str, str | Path] = {
                "evaluation_admission": (
                    sealed_root / "evaluation-admission-claim-v1.json"
                ),
                "evaluation_result": root / "evaluation-result.json",
                "evaluation_runtime": state_root / "evaluation-v1",
                "manifest": manifest,
                "qualification_release": qualification_release,
                "qualification_result": qualification_result,
                "qualification_runtime": qualification_runtime,
                "scratch_root": scratch_root,
                "sealed_root": sealed_root,
                "seed": sealed_root / "evaluation-seeds-v1.json",
                "state_root": state_root,
            }
            retained = {
                "baseline_roots": [],
                "genesis_root": str(qualification_runtime / "genesis"),
                "purpose": "qualification",
            }
            with mock.patch.object(
                runner,
                "_validate_retained_runtime_from_result",
                return_value=retained,
            ):
                topology = runner.validate_live_phase_topology(
                    "pre-seed",
                    path_overrides=paths,
                )
            self.assertEqual(topology["phase"], "pre-seed")
            self.assertEqual(
                topology["state_children"],
                ["qualification-v1", "sealed"],
            )

            extra = scratch_root / "undeclared-child"
            extra.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "child inventory differs",
            ), mock.patch.object(
                runner,
                "_validate_retained_runtime_from_result",
                return_value=retained,
            ):
                runner.validate_live_phase_topology(
                    "pre-seed",
                    path_overrides=paths,
                )
            extra.rmdir()

            aliased = dict(paths)
            aliased["evaluation_result"] = qualification_result
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "alias lexically",
            ), mock.patch.object(
                runner,
                "_validate_retained_runtime_from_result",
                return_value=retained,
            ):
                runner.validate_live_phase_topology(
                    "pre-seed",
                    path_overrides=aliased,
                )

    def test_deep_result_validation_rejects_rebound_nested_tampering(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_patch = mock.patch.object(
            runner,
            "STATE_ROOT",
            Path(temporary.name) / "state",
        )
        state_patch.start()
        self.addCleanup(state_patch.stop)
        result = _qualification_result_fixture(Path(temporary.name))
        self.assertEqual(runner.validate_run_result(result), result)
        catalog = result["frozen_adaptation"]["donor_catalogs"][0]["catalog"]
        self.assertEqual(len(catalog["entries"]), 45)
        self.assertEqual(catalog["excluded_counts"], {"SOURCE_CONTRACT": 58})
        injected_result = Path(temporary.name) / "qualification-result.json"
        injected_result.write_bytes(runner.canonical_json_bytes(result))
        injected_result.chmod(0o600)
        loaded, loaded_sha256 = runner.load_qualification_result(
            expected_source_seal_ref=result["integrity"]["source_seal_ref"],
            expected_qualification_release_ref=result["authorization_ref"],
            path=injected_result,
            injected_paths=True,
        )
        self.assertEqual(loaded, result)
        self.assertEqual(
            loaded_sha256,
            hashlib.sha256(runner.canonical_json_bytes(result)).hexdigest(),
        )
        for label, source_ref, release_ref in (
            (
                "source-B",
                _ref("substituted-source-seal"),
                result["authorization_ref"],
            ),
            (
                "release-B",
                result["integrity"]["source_seal_ref"],
                _ref("substituted-qualification-release"),
            ),
            (
                "source-None",
                None,
                result["authorization_ref"],
            ),
            (
                "release-None",
                result["integrity"]["source_seal_ref"],
                None,
            ),
        ):
            with self.subTest(label=label):
                with self.assertRaises(
                    (runner.V2InvariantError, runner.V2IntegrityStop, ValueError)
                ):
                    runner.load_qualification_result(
                        expected_source_seal_ref=source_ref,
                        expected_qualification_release_ref=release_ref,
                        path=injected_result,
                        injected_paths=True,
                    )

        def rebound(value: dict[str, object]) -> dict[str, object]:
            payload = dict(value)
            payload.pop("result_ref", None)
            return {
                **payload,
                "result_ref": runner.canonical_ref("run-result", payload),
            }

        cases: list[tuple[str, tuple[object, ...], object]] = [
            (
                "accounting",
                ("accounting", "execution_attempts"),
                71,
            ),
            (
                "integrity",
                ("integrity", "ledger", "terminal_refs_sha256"),
                _raw("tampered-terminal-inventory"),
            ),
            (
                "evaluator",
                (
                    "evaluator",
                    "metrics",
                    "aggregates",
                    0,
                    "binary_success_total",
                ),
                1,
            ),
            (
                "frozen",
                (
                    "frozen_adaptation",
                    "baseline_seals",
                    0,
                    "seal",
                    "hashes",
                    "store",
                ),
                _raw("tampered-frozen-store"),
            ),
            (
                "pair",
                (
                    "pair_equalities",
                    "items",
                    0,
                    "equality",
                    "proposal_sets_equal",
                ),
                False,
            ),
            (
                "resource",
                ("resources", "generation_calls"),
                143,
            ),
            (
                "runtime-resource",
                ("resources", "runtime_embedding_rows"),
                53 * 15,
            ),
        ]
        for label, path, replacement in cases:
            with self.subTest(label=label):
                tampered = json.loads(json.dumps(result))
                target: object = tampered
                for component in path[:-1]:
                    target = target[component]  # type: ignore[index]
                target[path[-1]] = replacement  # type: ignore[index]
                tampered = rebound(tampered)
                with self.assertRaises(
                    (runner.V2InvariantError, runner.V2IntegrityStop)
                ):
                    runner.validate_run_result(tampered)

        private = json.loads(json.dumps(result))
        private["evaluator"]["metrics"]["seed_hex"] = "00" * 32
        private = rebound(private)
        with self.assertRaisesRegex(
            runner.V2InvariantError,
            "raw/private material",
        ):
            runner.validate_run_result(private)

        def rebind_metrics(
            value: dict[str, object],
            *,
            update_ledger_artifact: bool,
        ) -> dict[str, object]:
            metrics = value["evaluator"]["metrics"]
            value["evaluator"]["metrics_ref"] = runner._evaluator_digest(
                "final-metrics",
                metrics,
            )
            if update_ledger_artifact:
                value["integrity"]["ledger"][
                    "evaluator_metrics_artifact_ref"
                ] = runner.canonical_ref("evaluator-final-metrics", metrics)
            return rebound(value)

        adaptation_success = json.loads(json.dumps(result))
        adaptation_row = next(
            row
            for row in adaptation_success["evaluator"]["metrics"]["aggregates"]
            if row["phase"] == "adaptation"
        )
        adaptation_row["binary_success_total"] = 1
        adaptation_success = rebind_metrics(
            adaptation_success,
            update_ledger_artifact=True,
        )
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(adaptation_success)

        non_float_pairwise = json.loads(json.dumps(result))
        symbolic_row = next(
            row
            for row in non_float_pairwise["evaluator"]["metrics"]["aggregates"]
            if row["family"] == "symbolic-demonstration-transfer"
        )
        symbolic_row["pairwise_agreement_total"] = 0
        non_float_pairwise = rebind_metrics(
            non_float_pairwise,
            update_ledger_artifact=True,
        )
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(non_float_pairwise)

        metric_artifact_drift = json.loads(json.dumps(result))
        symbolic_row = next(
            row
            for row in metric_artifact_drift["evaluator"]["metrics"]["aggregates"]
            if row["family"] == "symbolic-demonstration-transfer"
        )
        symbolic_row["pairwise_agreement_total"] = 0.5
        metric_artifact_drift = rebind_metrics(
            metric_artifact_drift,
            update_ledger_artifact=False,
        )
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(metric_artifact_drift)

        provenance = json.loads(json.dumps(result))
        wrapper = provenance["frozen_adaptation"]["adaptation_provenance"][0]
        wrapper["binding"]["attempt_final_ref"] = _ref(
            "tampered-adaptation-final"
        )
        wrapper["binding_ref"] = runner.canonical_ref(
            "adaptation-provenance",
            wrapper["binding"],
        )
        provenance = rebound(provenance)
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(provenance)

        source_closure = json.loads(json.dumps(result))
        source_closure["integrity"]["source_seal_ref"] = _ref(
            "tampered-source-seal"
        )
        source_closure = rebound(source_closure)
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(source_closure)

        authorization = json.loads(json.dumps(result))
        authorization["authorization_ref"] = _ref(
            "substituted-result-authorization"
        )
        authorization = rebound(authorization)
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(authorization)

        missing_authorization = json.loads(json.dumps(result))
        missing_authorization["authorization_ref"] = None
        missing_authorization = rebound(missing_authorization)
        with self.assertRaises(ValueError):
            runner.validate_run_result(missing_authorization)

        closure_authorization = json.loads(json.dumps(result))
        closure_wrapper = closure_authorization["integrity"]["ledger"][
            "run_provenance_closure"
        ]
        closure_wrapper["closure"]["authorization_ref"] = _ref(
            "substituted-closure-authorization"
        )
        closure_wrapper["closure_ref"] = runner.canonical_ref(
            "run-provenance-closure",
            closure_wrapper["closure"],
        )
        closure_authorization = rebound(closure_authorization)
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(closure_authorization)

        closure = result["integrity"]["ledger"]["run_provenance_closure"][
            "closure"
        ]
        with self.assertRaises(ValueError):
            runner._run_provenance_closure_payload(
                purpose="qualification",
                authorization_ref=None,
                source_seal_ref=closure["source_seal_ref"],
                evaluator_commitments_ref=closure[
                    "evaluator_commitments_ref"
                ],
                frozen_adaptation_ref=closure["frozen_adaptation_ref"],
                evaluator_metrics_artifact_ref=closure[
                    "evaluator_metrics_artifact_ref"
                ],
            )

        commitment_closure = json.loads(json.dumps(result))
        commitments = commitment_closure["evaluator"]["commitments"]
        commitments["tasks"][0]["private_commitment"] = _ref(
            "tampered-private-commitment"
        )
        commitment_closure["evaluator"]["commitments_ref"] = (
            runner._evaluator_digest("evaluator-commitments", commitments)
        )
        commitment_closure = rebound(commitment_closure)
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(commitment_closure)

        frozen_closure = json.loads(json.dumps(result))
        catalog_wrapper = frozen_closure["frozen_adaptation"]["donor_catalogs"][0]
        catalog_wrapper["catalog"]["entries"][0][
            "public_record_text_sha256"
        ] = _raw("tampered-public-record-text")
        catalog_wrapper["catalog_ref"] = runner.canonical_ref(
            "donor-catalog",
            catalog_wrapper["catalog"],
        )
        freeze = frozen_closure["frozen_adaptation"]["freeze"]
        freeze["donor_catalog_refs"] = [catalog_wrapper["catalog_ref"]]
        frozen_closure["frozen_adaptation"]["freeze_ref"] = runner.canonical_ref(
            "frozen-adaptation-state",
            freeze,
        )
        frozen_closure = rebound(frozen_closure)
        with self.assertRaises(runner.V2InvariantError):
            runner.validate_run_result(frozen_closure)

        for exclusion_count in (57, 59):
            with self.subTest(acquisition_head_exclusions=exclusion_count):
                head_closure = json.loads(json.dumps(result))
                catalog_wrapper = head_closure["frozen_adaptation"][
                    "donor_catalogs"
                ][0]
                catalog_wrapper["catalog"]["excluded_counts"] = {
                    "SOURCE_CONTRACT": exclusion_count
                }
                catalog_wrapper["catalog_ref"] = runner.canonical_ref(
                    "donor-catalog", catalog_wrapper["catalog"]
                )
                freeze = head_closure["frozen_adaptation"]["freeze"]
                freeze["donor_catalog_refs"] = [catalog_wrapper["catalog_ref"]]
                frozen_ref = runner.canonical_ref(
                    "frozen-adaptation-state", freeze
                )
                head_closure["frozen_adaptation"]["freeze_ref"] = frozen_ref
                provenance = head_closure["integrity"]["ledger"][
                    "run_provenance_closure"
                ]
                provenance["closure"]["frozen_adaptation_ref"] = frozen_ref
                provenance["closure_ref"] = runner.canonical_ref(
                    "run-provenance-closure", provenance["closure"]
                )
                head_closure = rebound(head_closure)
                with self.assertRaisesRegex(
                    runner.V2InvariantError,
                    "serialized donor acquisition-head closure differs",
                ):
                    runner.validate_run_result(head_closure)

    def test_path_allocations_are_durable_while_absent_and_close_on_enoent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "state"
            runtime_parent = state_root / "qualification-v1" / "arm-runtime"
            cognee_parent = state_root / "qualification-v1" / "cognee-scopes"
            runtime_parent.mkdir(parents=True, mode=0o700)
            cognee_parent.mkdir(parents=True, mode=0o700)
            runtime_parent.chmod(0o700)
            cognee_parent.chmod(0o700)
            ledger = runner.EvidenceLedger(root / "allocation.sqlite3", create=True)
            self.addCleanup(ledger.close)
            with mock.patch.object(runner, "STATE_ROOT", state_root):
                allocation = runner.DisposablePathAllocation(
                    purpose="qualification",
                    task_id=_ref("allocation-task"),
                    replicate_commitment=runner.qualification_replicate_commitment(),
                    role="IO_FULL_ORDINARY_12",
                    runtime_root=str(runtime_parent / "runtime-one"),
                    cognee_scope_roots=(str(cognee_parent / "scope-one"),),
                )
                registry = runner.PathAllocationRegistry("qualification", ledger)
                allocation_ref = registry.register(allocation)
                self.assertFalse(any(Path(item).exists() for item in allocation.targets))
                ledger.require_artifact(
                    allocation_ref,
                    "path-allocation",
                    allocation.to_canonical(),
                )
                for target in map(Path, allocation.targets):
                    target.mkdir(mode=0o700)
                with self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "target remains",
                ):
                    registry.mark_closed(allocation_ref)
                for target in reversed(tuple(map(Path, allocation.targets))):
                    target.rmdir()
                registry.mark_closed(allocation_ref)
                registry.assert_closed()

                recreated = Path(allocation.runtime_root)
                recreated.mkdir(mode=0o700)
                with self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "was recreated",
                ):
                    registry.assert_closed()
                recreated.rmdir()

    def test_invalid_objective_judgment_cannot_reach_state_mutation(self) -> None:
        qualification = evaluator.make_qualification_evaluator()
        scheduled = next(
            item
            for item in runner.adaptation_schedule(
                qualification.release_phase("adaptation"),
                purpose="qualification",
            )
            if item.arm == "IN_FULL_NEUTRAL_12"
        )
        history = runner.neutral_history_plan(
            scheduled.task_id,
            scheduled.arm,
            _neutral_episodes(),
        )
        prompt = runner.build_v1_proposal_prompt(
            "One bounded public synthetic qualification task.",
            history.evidence_texts,
        )
        token_ids = tuple(range(1, len(prompt.split()) + 1))
        prompt_plan = runner.PromptPlan(
            arm=scheduled.arm,
            task_id=scheduled.task_id,
            history_plan=history,
            prompt=prompt,
            prompt_ref=runner.content_ref({"prompt": prompt}),
            token_ids=token_ids,
            padding_atoms=0,
            target_tokens=len(token_ids),
        )
        state = {"feedback_calls": 0, "mutation": b"unchanged"}
        events: list[str] = []
        holder: dict[str, runner.AttemptObservation] = {}

        class BadEvaluator:
            protocol_identity = runner.PROTOCOL_IDENTITY
            identity = runner.QUALIFICATION_IDENTITY

            def record_attempt(
                self,
                task_id: str,
                arm: runner.Arm,
                receipt_ref: str,
                raw_response: str,
                *,
                proposal_admitted: bool,
            ) -> dict[str, object]:
                events.append("evaluator")
                response_payload = {
                    "arm": arm,
                    "attempt_receipt_ref": receipt_ref,
                    "protocol_identity": runner.PROTOCOL_IDENTITY,
                    "raw_response": raw_response,
                    "record_kind": "response",
                    "task_id": task_id,
                }
                without_ref = {
                    "arm": arm,
                    "attempt_receipt_ref": receipt_ref,
                    "disposition": "UNSUCCESSFUL",
                    "protocol_identity": runner.PROTOCOL_IDENTITY,
                    "proposal_admitted": proposal_admitted,
                    "public_task_ref": _ref("substituted-public-task"),
                    "raw_response": raw_response,
                    "record_kind": "objective-judgment",
                    "response_commitment": runner.domain_ref(
                        runner.EVALUATOR_RECORD_DOMAIN,
                        response_payload,
                    ),
                    "response_conforms": False,
                    "score": 0.0,
                    "success": False,
                    "task_id": task_id,
                }
                return {
                    **without_ref,
                    "evaluator_record_ref": runner.domain_ref(
                        runner.EVALUATOR_RECORD_DOMAIN,
                        without_ref,
                    ),
                }

        def execute_attempt(
            _scheduled: runner.ScheduledAttempt,
            _stage: runner.AttemptStage,
            plan: runner.PromptPlan,
            permit: runner.AttemptGenerationPermit,
            _resources: runner.ResourceLedger,
            arm_context: runner.ArmExecutionContext | None,
        ) -> runner.AttemptObservation:
            self.assertIsNone(arm_context)
            permit.before_proposal_generation()
            permit.before_execution_generation()
            events.append("execute")
            proposals = (
                "Use the first bounded public procedure.",
                "Use the second bounded public procedure.",
            )
            observation = runner.AttemptObservation(
                task_id=scheduled.task_id,
                arm=scheduled.arm,
                proposal_admitted=True,
                proposals=proposals,
                selected_index=0,
                selected_trace=proposals[0],
                raw_response="SYNTHETIC_RESPONSE",
                proposal_generation_ref=_ref("bad-judgment-proposal-generation"),
                proposal_prompt_ref=plan.prompt_ref,
                proposal_prompt_tokens=len(plan.token_ids),
                proposal_output_tokens=1,
                execution_request_ref=_ref("bad-judgment-execution-request"),
                execution_receipt_ref=_ref("bad-judgment-execution-receipt"),
                execution_prompt_ref=_ref("bad-judgment-execution-prompt"),
                execution_prompt_tokens=1,
                execution_output_tokens=1,
                selected_trace_bound=True,
                response_bound=True,
                selected_grammar_class="PUBLIC_SYNTHETIC",
                response_grammar_class="PUBLIC_SYNTHETIC",
                recalled_record_refs=tuple(
                    item.record_ref
                    for item in plan.history_plan.presentations
                ),
            )
            holder["observation"] = observation
            return observation

        def mutate_feedback(*_args: object, **_kwargs: object) -> object:
            state["feedback_calls"] = int(state["feedback_calls"]) + 1
            state["mutation"] = b"changed"
            raise AssertionError("invalid judgment reached state mutation")

        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("invalid judgment crossed a terminal boundary")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = runner.EvidenceLedger(root / "bad-judgment.sqlite3", create=True)
            dependencies = runner.OrchestrationDependencies(
                mode=runner.RunMode.QUALIFICATION,
                evaluator=BadEvaluator(),
                ledger=ledger,
                path_allocations=runner.PathAllocationRegistry(
                    "qualification",
                    ledger,
                ),
                attempt_budget=runner.AttemptBudget(runner.RunMode.QUALIFICATION),
                resources=runner.ResourceLedger(runner.RunMode.QUALIFICATION),
                prepare_adaptation_attempt=lambda _scheduled: prompt_plan,
                freeze_adaptation=forbidden,
                prepare_task_block=forbidden,
                dispose_recall_captures=forbidden,
                execute_attempt=execute_attempt,
                apply_objective_feedback=mutate_feedback,
                finalize_persistent_stateful_attempt=forbidden,
                terminalize_stateful_attempt=forbidden,
                dispose_clone=forbidden,
                dispose_stateless_attempt=forbidden,
                verify_frozen_adaptation=forbidden,
                sample_resources=lambda: {
                    "configured_pools": 0,
                    "cuda_allocated_bytes": 0,
                    "cuda_reserved_bytes": 0,
                    "process_count": 0,
                    "rss_bytes": 0,
                    "state_scratch_bytes": 0,
                },
                verify_source_seal=lambda: _ref("bad-judgment-source-seal"),
                close_runtime=lambda: None,
            )
            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "does not bind the attempt",
            ):
                asyncio.run(
                    runner._execute_one_attempt(
                        dependencies,
                        scheduled,
                        prompt_plan,
                    )
                )
            self.assertEqual(events, ["execute", "evaluator"])
            self.assertEqual(
                state,
                {"feedback_calls": 0, "mutation": b"unchanged"},
            )
            self.assertEqual(
                ledger._connection.execute(
                    "SELECT status, disposition_ref, final_ref FROM attempts"
                ).fetchone(),
                (0, None, None),
            )
            observation = holder["observation"]
            ledger.require_artifact(
                observation.observation_ref,
                "attempt-observation",
                observation.to_canonical(),
            )
            self.assertEqual(
                ledger._connection.execute(
                    "SELECT COUNT(*) FROM artifacts "
                    "WHERE kind='objective-judgment'"
                ).fetchone()[0],
                0,
            )
            ledger.close()

    def test_malformed_stateless_attempt_is_durably_disposed_before_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = _qualification_development_tasks()[0]
            arm = "N0_NATIVE_NO_HISTORY"
            prompt_plan = runner.native_prompt_plan(
                task_id=task.task_id,
                task_text="One bounded public synthetic qualification task.",
                tokenize_chat_prompt=lambda prompt: tuple(
                    range(1, len(prompt.split()) + 1)
                ),
            )
            scheduled = runner.ScheduledAttempt(
                purpose="qualification",
                phase="development",
                arm=arm,
                task=task.to_canonical(),
                block_ordinal=0,
                arm_position=runner.ARMS.index(arm),
            )
            arm_context = runner.ArmExecutionContext(
                task_id=task.task_id,
                arm=arm,
                preparation_ref=_ref("malformed-stateless-preparation"),
            )
            state_root = root / "state"
            runtime_parent = state_root / "qualification-v1" / "arm-runtime"
            runtime_parent.mkdir(parents=True, mode=0o700)
            runtime_parent.chmod(0o700)
            ledger = runner.EvidenceLedger(root / "stateless.sqlite3", create=True)
            with mock.patch.object(runner, "STATE_ROOT", state_root):
                allocation = runner.DisposablePathAllocation(
                    purpose="qualification",
                    task_id=task.task_id,
                    replicate_commitment=task.replicate_commitment,
                    role=arm,
                    runtime_root=str(runtime_parent / "malformed-stateless"),
                    cognee_scope_roots=(),
                )
                registry = runner.PathAllocationRegistry("qualification", ledger)
                registry.register(allocation)
            runtime_root = Path(allocation.runtime_root)
            runtime_root.mkdir(mode=0o700)
            runtime_root.chmod(0o700)
            journal_path = runtime_root / "qwen-execution-journal.sqlite3"
            journal_path.write_bytes(b"exact stateless malformed journal")
            journal_path.chmod(0o600)
            events: list[str] = []
            holder: dict[str, runner.AttemptObservation] = {}

            class RecordingEvaluator:
                protocol_identity = runner.PROTOCOL_IDENTITY
                identity = runner.QUALIFICATION_IDENTITY

                def record_attempt(
                    self,
                    task_id: str,
                    observed_arm: runner.Arm,
                    receipt_ref: str,
                    raw_response: str,
                    *,
                    proposal_admitted: bool,
                ) -> dict[str, object]:
                    observation = holder["observation"]
                    ledger.require_artifact(
                        observation.observation_ref,
                        "attempt-observation",
                        observation.to_canonical(),
                    )
                    events.append("evaluator")
                    response = {
                        "arm": observed_arm,
                        "attempt_receipt_ref": receipt_ref,
                        "protocol_identity": runner.PROTOCOL_IDENTITY,
                        "raw_response": raw_response,
                        "record_kind": "response",
                        "task_id": task_id,
                    }
                    response_ref = runner.domain_ref(
                        runner.EVALUATOR_RECORD_DOMAIN,
                        response,
                    )
                    without_ref = {
                        "arm": observed_arm,
                        "attempt_receipt_ref": receipt_ref,
                        "disposition": "UNSUCCESSFUL",
                        "protocol_identity": runner.PROTOCOL_IDENTITY,
                        "proposal_admitted": proposal_admitted,
                        "public_task_ref": task.public_task_ref,
                        "raw_response": raw_response,
                        "record_kind": "objective-judgment",
                        "response_commitment": response_ref,
                        "response_conforms": False,
                        "score": 0.0,
                        "success": False,
                        "task_id": task_id,
                    }
                    return {
                        **without_ref,
                        "evaluator_record_ref": runner.domain_ref(
                            runner.EVALUATOR_RECORD_DOMAIN,
                            without_ref,
                        ),
                    }

            def execute_attempt(
                _scheduled: runner.ScheduledAttempt,
                _stage: runner.AttemptStage,
                plan: runner.PromptPlan,
                permit: runner.AttemptGenerationPermit,
                _resources: runner.ResourceLedger,
                context: runner.ArmExecutionContext | None,
            ) -> runner.AttemptObservation:
                self.assertIs(context, arm_context)
                permit.before_proposal_generation()
                events.append("execute")
                observation = runner.AttemptObservation(
                    task_id=task.task_id,
                    arm=arm,
                    proposal_admitted=False,
                    proposals=(),
                    selected_index=None,
                    selected_trace=None,
                    raw_response="",
                    proposal_generation_ref=_ref("malformed-stateless-generation"),
                    proposal_prompt_ref=plan.prompt_ref,
                    proposal_prompt_tokens=len(plan.token_ids),
                    proposal_output_tokens=1,
                    execution_request_ref=None,
                    execution_receipt_ref=None,
                    execution_prompt_ref=None,
                    execution_prompt_tokens=None,
                    execution_output_tokens=None,
                    selected_trace_bound=None,
                    response_bound=None,
                    selected_grammar_class=None,
                    response_grammar_class=None,
                    recalled_record_refs=(),
                )
                holder["observation"] = observation
                return observation

            def forbidden(*_args: object, **_kwargs: object) -> object:
                raise AssertionError("malformed stateless attempt crossed a boundary")

            def dispose_stateless(
                observed_schedule: runner.ScheduledAttempt,
                observed_stage: runner.AttemptStage,
                observation: runner.AttemptObservation,
                disposition_ref: str,
            ) -> runner.StatelessRuntimeCleanupEvidence:
                self.assertIs(observed_schedule, scheduled)
                self.assertIs(observation, holder["observation"])
                self.assertEqual(observed_stage.task_id, scheduled.task_id)
                self.assertEqual(observed_stage.arm, scheduled.arm)
                row = ledger._connection.execute(
                    "SELECT status, disposition_ref FROM attempts"
                ).fetchone()
                self.assertEqual(row, (1, disposition_ref))
                ledger.require_terminal_disposition_payload(
                    task_id=observed_stage.task_id,
                    arm=observed_stage.arm,
                    terminal_disposition_ref=disposition_ref,
                    expected_clone_terminal_ref=None,
                    stage_ref=observed_stage.stage_ref,
                    attempt_receipt_ref=observed_stage.attempt_receipt_ref,
                )
                journal_sha256, journal_bytes = runner.sha256_file(journal_path)
                terminal = runner.StatelessRuntimeTerminalEvidence(
                    task_id=task.task_id,
                    arm=arm,
                    runtime_root=str(runtime_root),
                    journal_path=str(journal_path),
                    journal_sha256=journal_sha256,
                    journal_bytes=journal_bytes,
                    allocation_ref=allocation.allocation_ref,
                    terminal_disposition_ref=disposition_ref,
                )
                ledger.append_artifact_durable(
                    terminal.terminal_ref,
                    "stateless-runtime-terminal",
                    terminal.to_canonical(),
                )
                events.append("stateless-terminal-durable")
                journal_path.unlink()
                runtime_root.rmdir()
                events.append("stateless-root-removed")
                cleanup = runner.StatelessRuntimeCleanupEvidence(
                    terminal=terminal,
                    disposed=True,
                    cleanup_absence_verified=True,
                )
                ledger.append_artifact_durable(
                    cleanup.cleanup_ref,
                    "stateless-runtime-cleanup",
                    cleanup.to_canonical(),
                )
                events.append("stateless-cleanup-durable")
                return cleanup

            dependencies = runner.OrchestrationDependencies(
                mode=runner.RunMode.QUALIFICATION,
                evaluator=RecordingEvaluator(),
                ledger=ledger,
                path_allocations=registry,
                attempt_budget=runner.AttemptBudget(runner.RunMode.QUALIFICATION),
                resources=runner.ResourceLedger(runner.RunMode.QUALIFICATION),
                prepare_adaptation_attempt=forbidden,
                freeze_adaptation=forbidden,
                prepare_task_block=forbidden,
                dispose_recall_captures=forbidden,
                execute_attempt=execute_attempt,
                apply_objective_feedback=forbidden,
                terminalize_stateful_attempt=forbidden,
                dispose_clone=forbidden,
                dispose_stateless_attempt=dispose_stateless,
                finalize_persistent_stateful_attempt=forbidden,
                verify_frozen_adaptation=forbidden,
                sample_resources=lambda: {
                    "configured_pools": 0,
                    "cuda_allocated_bytes": 0,
                    "cuda_reserved_bytes": 0,
                    "process_count": 0,
                    "rss_bytes": 0,
                    "state_scratch_bytes": 0,
                },
                verify_source_seal=lambda: _ref("source-seal-check"),
                close_runtime=lambda: None,
            )
            original_stage = ledger.append_stage
            original_artifact = ledger.append_artifact_durable
            original_disposition = ledger.record_terminal_disposition
            original_finalize = ledger.finalize

            def append_stage(
                stage: runner.AttemptStage,
                plan: runner.PromptPlan,
            ) -> str:
                value = original_stage(stage, plan)
                events.append("stage-durable")
                return value

            def append_artifact(
                artifact_ref: str,
                kind: str,
                payload: dict[str, object],
            ) -> str:
                value = original_artifact(artifact_ref, kind, payload)
                if kind == "attempt-observation":
                    events.append("observation-durable")
                elif kind == "objective-judgment":
                    events.append("judgment-durable")
                return value

            def disposition(**kwargs: object) -> str:
                value = original_disposition(**kwargs)
                events.append("disposition-durable")
                return value

            def finalize(final: runner.AttemptFinal) -> str:
                self.assertFalse(runtime_root.exists())
                value = original_finalize(final)
                events.append("final")
                return value

            with (
                mock.patch.object(ledger, "append_stage", side_effect=append_stage),
                mock.patch.object(
                    ledger,
                    "append_artifact_durable",
                    side_effect=append_artifact,
                ),
                mock.patch.object(
                    ledger,
                    "record_terminal_disposition",
                    side_effect=disposition,
                ),
                mock.patch.object(ledger, "finalize", side_effect=finalize),
            ):
                final = asyncio.run(
                    runner._execute_one_attempt(
                        dependencies,
                        scheduled,
                        prompt_plan,
                        arm_context,
                    )
                )
            self.assertFalse(final.observation.proposal_admitted)
            self.assertIsNone(final.observation.clone_lifecycle)
            self.assertEqual(
                events,
                [
                    "stage-durable",
                    "execute",
                    "observation-durable",
                    "evaluator",
                    "judgment-durable",
                    "disposition-durable",
                    "stateless-terminal-durable",
                    "stateless-root-removed",
                    "stateless-cleanup-durable",
                    "final",
                ],
            )
            registry.assert_closed()
            witness = ledger.seal(expected_attempts=1)
            self.assertEqual(witness["attempts"], 1)
            self.assertEqual(witness["finalized"], 1)
            self.assertTrue(ledger._sealed)
            self.assertFalse(ledger._closed)
            self.assertEqual(ledger.counts(), (1, 1))
            with self.assertRaisesRegex(RuntimeError, "sealed"):
                ledger.finalize(final)
            ledger.close()
            self.assertTrue(ledger._closed)

    def test_private_replay_capability_is_confined_to_the_io_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frozen, catalog, items = _qualification_frozen_state(
                Path(directory) / "frozen"
            )
            task = _qualification_development_tasks()[0]
            baselines = frozen.baselines_for(task.replicate_commitment)
            replay = _ordinary_private_replay(
                task=task,
                catalog=catalog,
                items=items,
                source_baseline_seal_ref=baselines[
                    "IO_FULL_ORDINARY_12"
                ].seal_ref,
            )

            class ReplayTarget:
                def __init__(self, batch: AcquisitionRecallBatch) -> None:
                    self.batch = batch
                    self.calls: list[tuple[object, ...]] = []

                def recall_from_hits(
                    self,
                    hits: tuple[AcquisitionReferenceHit, ...],
                    *,
                    limit: int,
                    frozen_origin: bool,
                ) -> AcquisitionRecallBatch:
                    self.calls.append((hits, limit, frozen_origin))
                    return self.batch

            target = ReplayTarget(replay.batch)
            self.assertIs(replay.replay(target), replay.batch)
            self.assertEqual(target.calls, [(replay.hits, 12, True)])

            io_context = runner.ArmExecutionContext(
                task_id=task.task_id,
                arm="IO_FULL_ORDINARY_12",
                preparation_ref=_ref("io-preparation"),
                baseline=baselines["IO_FULL_ORDINARY_12"],
                ordinary_private_replay=replay,
            )
            in_context = runner.ArmExecutionContext(
                task_id=task.task_id,
                arm="IN_FULL_NEUTRAL_12",
                preparation_ref=_ref("in-preparation"),
                baseline=baselines["IN_FULL_NEUTRAL_12"],
                neutral_episode_refs=baselines[
                    "IN_FULL_NEUTRAL_12"
                ].neutral_episode_refs,
            )
            self.assertIs(io_context.ordinary_private_replay, replay)
            self.assertIsNone(in_context.ordinary_private_replay)
            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "IN execution capability differs",
            ):
                runner.ArmExecutionContext(
                    task_id=task.task_id,
                    arm="IN_FULL_NEUTRAL_12",
                    preparation_ref=_ref("contaminated-in-preparation"),
                    baseline=baselines["IN_FULL_NEUTRAL_12"],
                    neutral_episode_refs=baselines[
                        "IN_FULL_NEUTRAL_12"
                    ].neutral_episode_refs,
                    ordinary_private_replay=replay,
                )
            with self.assertRaisesRegex(
                runner.V2InvariantError,
                "stateless arm acquired",
            ):
                runner.ArmExecutionContext(
                    task_id=task.task_id,
                    arm="O0_ORDINARY_FROZEN_12",
                    preparation_ref=_ref("contaminated-o0-preparation"),
                    ordinary_private_replay=replay,
                )

    def test_malformed_in_and_io_are_durably_terminalized_then_disposed(self) -> None:
        for arm in ("IN_FULL_NEUTRAL_12", "IO_FULL_ORDINARY_12"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                frozen, catalog, items = _qualification_frozen_state(
                    root / "frozen"
                )
                task = _qualification_development_tasks()[0]
                baselines = frozen.baselines_for(task.replicate_commitment)
                replay = _ordinary_private_replay(
                    task=task,
                    catalog=catalog,
                    items=items,
                    source_baseline_seal_ref=baselines[
                        "IO_FULL_ORDINARY_12"
                    ].seal_ref,
                )
                prompt_plan = _prompt_plan_for_stateful_arm(
                    task=task,
                    arm=arm,
                    ordinary_capture=(replay.capture if arm == "IO_FULL_ORDINARY_12" else None),
                )
                scheduled = runner.ScheduledAttempt(
                    purpose="qualification",
                    phase="development",
                    arm=arm,
                    task=task.to_canonical(),
                    block_ordinal=0,
                    arm_position=runner.ARMS.index(arm),
                )
                context = runner.ArmExecutionContext(
                    task_id=task.task_id,
                    arm=arm,
                    preparation_ref=_ref(f"malformed-preparation-{arm}"),
                    baseline=baselines[arm],
                    neutral_episode_refs=(
                        baselines[arm].neutral_episode_refs
                        if arm == "IN_FULL_NEUTRAL_12"
                        else ()
                    ),
                    ordinary_private_replay=(
                        replay if arm == "IO_FULL_ORDINARY_12" else None
                    ),
                )

                state_root = root / "state"
                runtime_parent = state_root / "qualification-v1" / "arm-runtime"
                cognee_parent = state_root / "qualification-v1" / "cognee-scopes"
                runtime_parent.mkdir(parents=True, mode=0o700)
                cognee_parent.mkdir(parents=True, mode=0o700)
                runtime_parent.chmod(0o700)
                cognee_parent.chmod(0o700)
                ledger = runner.EvidenceLedger(root / "evidence.sqlite3", create=True)
                with mock.patch.object(runner, "STATE_ROOT", state_root):
                    allocation = runner.DisposablePathAllocation(
                        purpose="qualification",
                        task_id=task.task_id,
                        replicate_commitment=task.replicate_commitment,
                        role=arm,
                        runtime_root=str(runtime_parent / "malformed-runtime"),
                        cognee_scope_roots=(
                            str(cognee_parent / "malformed-scope"),
                        ),
                    )
                    registry = runner.PathAllocationRegistry("qualification", ledger)
                    registry.register(allocation)
                clone_root = Path(allocation.runtime_root)
                scope_root = Path(allocation.cognee_scope_roots[0])
                clone_root.mkdir(mode=0o700)
                scope_root.mkdir(mode=0o700)
                events: list[str] = []
                holder: dict[str, runner.AttemptObservation] = {}

                def judgment_payload(
                    task_id: str,
                    observed_arm: runner.Arm,
                    receipt_ref: str,
                    raw_response: str,
                    proposal_admitted: bool,
                ) -> dict[str, object]:
                    response = {
                        "arm": observed_arm,
                        "attempt_receipt_ref": receipt_ref,
                        "protocol_identity": runner.PROTOCOL_IDENTITY,
                        "raw_response": raw_response,
                        "record_kind": "response",
                        "task_id": task_id,
                    }
                    response_ref = runner.domain_ref(
                        runner.EVALUATOR_RECORD_DOMAIN,
                        response,
                    )
                    without_ref = {
                        "arm": observed_arm,
                        "attempt_receipt_ref": receipt_ref,
                        "disposition": "UNSUCCESSFUL",
                        "protocol_identity": runner.PROTOCOL_IDENTITY,
                        "proposal_admitted": proposal_admitted,
                        "public_task_ref": task.public_task_ref,
                        "raw_response": raw_response,
                        "record_kind": "objective-judgment",
                        "response_commitment": response_ref,
                        "response_conforms": False,
                        "score": 0.0,
                        "success": False,
                        "task_id": task_id,
                    }
                    return {
                        **without_ref,
                        "evaluator_record_ref": runner.domain_ref(
                            runner.EVALUATOR_RECORD_DOMAIN,
                            without_ref,
                        ),
                    }

                class RecordingEvaluator:
                    protocol_identity = runner.PROTOCOL_IDENTITY
                    identity = runner.QUALIFICATION_IDENTITY

                    def record_attempt(
                        self,
                        task_id: str,
                        observed_arm: runner.Arm,
                        receipt_ref: str,
                        raw_response: str,
                        *,
                        proposal_admitted: bool,
                    ) -> dict[str, object]:
                        observation = holder["observation"]
                        ledger.require_artifact(
                            observation.observation_ref,
                            "attempt-observation",
                            observation.to_canonical(),
                        )
                        events.append("evaluator")
                        return judgment_payload(
                            task_id,
                            observed_arm,
                            receipt_ref,
                            raw_response,
                            proposal_admitted,
                        )

                def execute_attempt(
                    _scheduled: runner.ScheduledAttempt,
                    _stage: runner.AttemptStage,
                    plan: runner.PromptPlan,
                    permit: runner.AttemptGenerationPermit,
                    _resources: runner.ResourceLedger,
                    observed_context: runner.ArmExecutionContext | None,
                ) -> runner.AttemptObservation:
                    self.assertIs(observed_context, context)
                    permit.before_proposal_generation()
                    events.append("execute")
                    observation = runner.AttemptObservation(
                        task_id=task.task_id,
                        arm=arm,
                        proposal_admitted=False,
                        proposals=(),
                        selected_index=None,
                        selected_trace=None,
                        raw_response="",
                        proposal_generation_ref=_ref(
                            f"malformed-generation-{arm}"
                        ),
                        proposal_prompt_ref=plan.prompt_ref,
                        proposal_prompt_tokens=len(plan.token_ids),
                        proposal_output_tokens=1,
                        execution_request_ref=None,
                        execution_receipt_ref=None,
                        execution_prompt_ref=None,
                        execution_prompt_tokens=None,
                        execution_output_tokens=None,
                        selected_trace_bound=None,
                        response_bound=None,
                        selected_grammar_class=None,
                        response_grammar_class=None,
                        recalled_record_refs=tuple(
                            item.record_ref
                            for item in plan.history_plan.presentations
                        ),
                    )
                    holder["observation"] = observation
                    return observation

                def forbidden_feedback(*_args: object) -> runner.StatefulFeedbackApplication:
                    raise AssertionError("malformed proposal reached objective feedback")

                def terminalize(
                    _scheduled: runner.ScheduledAttempt,
                    _stage: runner.AttemptStage,
                    observation: runner.AttemptObservation,
                    judgment: dict[str, object],
                    feedback_applied: bool,
                    _resources: runner.ResourceLedger,
                ) -> runner.CloneTerminalEvidence:
                    self.assertFalse(feedback_applied)
                    self.assertIsNone(observation.resolution_ref)
                    ledger.require_artifact(
                        runner.canonical_ref("objective-judgment", judgment),
                        "objective-judgment",
                        judgment,
                    )
                    events.append("terminalize")
                    return runner.CloneTerminalEvidence(
                        task_id=task.task_id,
                        arm=arm,
                        source_root=baselines[arm].root,
                        clone_root=str(clone_root),
                        source_before=baselines[arm].hashes,
                        source_after=baselines[arm].hashes,
                        clone_before=baselines[arm].hashes,
                        clone_terminal=baselines[arm].hashes,
                        quiescent=True,
                        cognee_scope_roots=(str(scope_root),),
                        allocation_ref=allocation.allocation_ref,
                    )

                def dispose(
                    terminal: runner.CloneTerminalEvidence,
                    disposition_ref: str,
                ) -> runner.CloneLifecycleEvidence:
                    row = ledger._connection.execute(
                        "SELECT status, disposition_ref FROM attempts"
                    ).fetchone()
                    self.assertEqual(row, (1, disposition_ref))
                    events.append("dispose")
                    scope_root.rmdir()
                    clone_root.rmdir()
                    lifecycle = runner.CloneLifecycleEvidence(
                        terminal=terminal,
                        terminal_disposition_ref=disposition_ref,
                        disposed=True,
                        cleanup_absence_verified=True,
                    )
                    ledger.append_artifact_durable(
                        lifecycle.lifecycle_ref,
                        "clone-lifecycle",
                        lifecycle.to_canonical(),
                    )
                    return lifecycle

                dependencies = runner.OrchestrationDependencies(
                    mode=runner.RunMode.QUALIFICATION,
                    evaluator=RecordingEvaluator(),
                    ledger=ledger,
                    path_allocations=registry,
                    attempt_budget=runner.AttemptBudget(
                        runner.RunMode.QUALIFICATION
                    ),
                    resources=runner.ResourceLedger(runner.RunMode.QUALIFICATION),
                    prepare_adaptation_attempt=lambda _scheduled: prompt_plan,
                    freeze_adaptation=lambda _attempts: frozen,
                    prepare_task_block=lambda _task, _context: None,
                    dispose_recall_captures=lambda block, _refs: block,
                    execute_attempt=execute_attempt,
                    apply_objective_feedback=forbidden_feedback,
                    terminalize_stateful_attempt=terminalize,
                    dispose_clone=dispose,
                    dispose_stateless_attempt=forbidden_feedback,
                    finalize_persistent_stateful_attempt=forbidden_feedback,
                    verify_frozen_adaptation=lambda _frozen: _ref("freeze-check"),
                    sample_resources=lambda: {
                        "configured_pools": 0,
                        "cuda_allocated_bytes": 0,
                        "cuda_reserved_bytes": 0,
                        "process_count": 0,
                        "rss_bytes": 0,
                        "state_scratch_bytes": 0,
                    },
                    verify_source_seal=lambda: _ref("source-seal-check"),
                    close_runtime=lambda: None,
                )
                original_stage = ledger.append_stage
                original_artifact = ledger.append_artifact_durable
                original_disposition = ledger.record_terminal_disposition
                original_finalize = ledger.finalize

                def append_stage(
                    stage: runner.AttemptStage,
                    plan: runner.PromptPlan,
                ) -> str:
                    value = original_stage(stage, plan)
                    events.append("stage-durable")
                    return value

                def append_artifact(
                    artifact_ref: str,
                    kind: str,
                    payload: dict[str, object],
                ) -> str:
                    value = original_artifact(artifact_ref, kind, payload)
                    if kind == "attempt-observation":
                        events.append("observation-durable")
                    elif kind == "objective-judgment":
                        events.append("judgment-durable")
                    elif kind == "clone-lifecycle":
                        events.append("clone-lifecycle-durable")
                    else:
                        raise AssertionError(
                            f"unexpected durable artifact kind: {kind}"
                        )
                    return value

                def disposition(**kwargs: object) -> str:
                    value = original_disposition(**kwargs)
                    events.append("disposition-durable")
                    return value

                def finalize(final: runner.AttemptFinal) -> str:
                    self.assertFalse(clone_root.exists())
                    self.assertFalse(scope_root.exists())
                    value = original_finalize(final)
                    events.append("final")
                    return value

                with (
                    mock.patch.object(ledger, "append_stage", side_effect=append_stage),
                    mock.patch.object(
                        ledger,
                        "append_artifact_durable",
                        side_effect=append_artifact,
                    ),
                    mock.patch.object(
                        ledger,
                        "record_terminal_disposition",
                        side_effect=disposition,
                    ),
                    mock.patch.object(ledger, "finalize", side_effect=finalize),
                ):
                    final = asyncio.run(
                        runner._execute_one_attempt(
                            dependencies,
                            scheduled,
                            prompt_plan,
                            context,
                        )
                    )
                self.assertFalse(final.observation.proposal_admitted)
                self.assertIsNotNone(final.observation.clone_lifecycle)
                self.assertEqual(
                    events,
                    [
                        "stage-durable",
                        "execute",
                        "observation-durable",
                        "evaluator",
                        "judgment-durable",
                        "terminalize",
                        "disposition-durable",
                        "dispose",
                        "clone-lifecycle-durable",
                        "final",
                    ],
                )
                registry.assert_closed()
                witness = ledger.seal(expected_attempts=1)
                self.assertEqual(witness["attempts"], 1)
                self.assertEqual(witness["finalized"], 1)
                self.assertTrue(ledger._sealed)
                self.assertFalse(ledger._closed)
                self.assertEqual(ledger.counts(), (1, 1))
                with self.assertRaisesRegex(RuntimeError, "sealed"):
                    ledger.finalize(final)
                ledger.close()
                self.assertTrue(ledger._closed)


@dataclass(frozen=True, slots=True)
class _E2EGenesis:
    root: Path

    def validate_artifacts(self) -> None:
        if not self.root.is_dir() or stat.S_IMODE(os.lstat(self.root).st_mode) != 0o700:
            raise AssertionError("fake genesis boundary changed")


@dataclass(frozen=True, slots=True)
class _E2ELoadedFoundation:
    manifest: _OwnerManifest
    io: object
    guard: object


class _InjectedEvaluator:
    """Expose the real in-process evaluator through the injected protocol surface."""

    protocol_identity = runner.PROTOCOL_IDENTITY

    def __init__(self, suite: object) -> None:
        self._suite = suite
        self.identity = getattr(suite, "identity")
        commitments = getattr(suite, "commitments")
        self.commitments = dict(commitments.to_canonical())
        self.closed = False

    def release_phase(self, phase: runner.Phase) -> tuple[object, ...]:
        return tuple(self._suite.release_phase(phase))

    def complete_phase(self, phase: runner.Phase) -> None:
        self._suite.complete_phase(phase)

    def admit_final(self, admission_ref: str) -> None:
        self._suite.admit_final(admission_ref)

    def record_attempt(
        self,
        task_id: str,
        arm: runner.Arm,
        attempt_receipt_ref: str,
        raw_response: str,
        *,
        proposal_admitted: bool,
    ) -> dict[str, object]:
        judgment = self._suite.record_attempt(
            task_id,
            arm,
            attempt_receipt_ref,
            raw_response,
            proposal_admitted=proposal_admitted,
        )
        return dict(judgment.to_canonical())

    def final_metrics(self) -> dict[str, object]:
        return dict(self._suite.final_metrics().to_canonical())

    def close(self) -> None:
        self.closed = True


class _FullFakeProductHarness:
    """Bounded CPU fixture that leaves orchestration and evidence code unmocked."""

    def __init__(
        self,
        *,
        purpose: runner.Purpose,
        root: Path,
        ledger: runner.EvidenceLedger,
        resources: runner.ResourceLedger,
        lineages: runner.V2LineageSetOwner,
        mechanics: runner.V2LineageMechanics,
        foundation: runner.V2FoundationOwner,
        genesis: _E2EGenesis,
    ) -> None:
        self.purpose = purpose
        self.root = root
        self.phase_root = root / f"{purpose}-v1"
        self.arm_runtime_parent = self.phase_root / "arm-runtime"
        self.cognee_scope_parent = self.phase_root / "cognee-scopes"
        self.ledger = ledger
        self.resources = resources
        self.lineages = lineages
        self.mechanics = mechanics
        self.foundation = foundation
        self.genesis = genesis
        self.path_registry = runner.PathAllocationRegistry(purpose, ledger)
        self.events: list[str] = []
        self.frozen: runner.FrozenAdaptationState | None = None
        self.catalog_items: dict[str, tuple[CognitiveAcquisitionItem, ...]] = {}
        self.persistent_scopes: dict[str, Path] = {}
        self.disposable: dict[
            str,
            tuple[runner.DisposablePathAllocation, runner.BaselineSeal | None],
        ] = {}
        self.closed = False
        self.source_seal_ref = _ref(f"full-fake-{purpose}-source-seal")
        self.product = runner.V2ProductRuntimeOwner(
            purpose=purpose,
            foundation=foundation,
            genesis=genesis,
            lineages=lineages,
            ledger=ledger,
            path_registry=self.path_registry,
            resources=resources,
            acquisition_opener=_OwnerAcquisitionOpener(self.events),
            mechanics=mechanics,
            arm_runtime_parent=self.arm_runtime_parent,
            cognee_scope_parent=self.cognee_scope_parent,
        )

    @classmethod
    async def create(
        cls,
        *,
        purpose: runner.Purpose,
        root: Path,
    ) -> "_FullFakeProductHarness":
        phase_root = root / f"{purpose}-v1"
        genesis_root = phase_root / "genesis"
        runtime_parent = phase_root / "arm-runtime"
        scope_parent = phase_root / "cognee-scopes"
        for path in (root, phase_root, genesis_root, runtime_parent, scope_parent):
            path.mkdir(mode=0o700)
            path.chmod(0o700)
        ledger = runner.EvidenceLedger(root / f"{purpose}-evidence.sqlite3", create=True)
        resources = runner.ResourceLedger(runner.RunMode(purpose))
        events: list[str] = []
        mechanics = _owner_mechanics(events, ledger=ledger)
        replicates = (
            (runner.qualification_replicate_commitment(),)
            if purpose == "qualification"
            else _EVALUATION_COMMITMENTS
        )
        lineages = await runner.V2LineageSetOwner.create_from_phase_parents(
            purpose=purpose,
            replicate_commitments=replicates,
            arm_runtime_parent=runtime_parent,
            cognee_scope_parent=scope_parent,
            genesis=object(),
            manifest=_OwnerManifest(),
            ledger=ledger,
            acquisition_opener=_OwnerAcquisitionOpener(events),
            mechanics=mechanics,
            injected_paths=True,
        )

        source_ref = _ref(f"full-fake-{purpose}-source-seal")

        def sample_resources() -> dict[str, int]:
            return {
                "configured_pools": 0,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
                "process_count": 0,
                "rss_bytes": 0,
                "state_scratch_bytes": 0,
            }

        def verify_source_seal() -> str:
            return source_ref

        guard = object.__new__(runner.GuardedLocalQwenBoundary)
        guard.resources = resources
        guard.sample_resources = sample_resources
        guard.verify_source_seal = verify_source_seal
        loaded = _E2ELoadedFoundation(_OwnerManifest(), object(), object())
        foundation = runner.V2FoundationOwner(
            loaded=loaded,
            guard=guard,
            _mirror=None,
            helpers={},
            _construction_capability=(
                runner._V2_FOUNDATION_OWNER_CONSTRUCTION_CAPABILITY
            ),
        )
        harness = cls(
            purpose=purpose,
            root=root,
            ledger=ledger,
            resources=resources,
            lineages=lineages,
            mechanics=mechanics,
            foundation=foundation,
            genesis=_E2EGenesis(genesis_root),
        )
        harness.events.extend(events)
        # Preserve the exact callable objects owned by the guard.
        harness.sample_resources = sample_resources  # type: ignore[method-assign]
        harness.verify_source_seal = verify_source_seal  # type: ignore[method-assign]
        return harness

    @staticmethod
    def _tokenize(prompt: str) -> tuple[int, ...]:
        return tuple(range(1, len(prompt.split()) + 1))

    @staticmethod
    def _task_text(task: object) -> str:
        return runner._render_v2_public_task(task)

    def _adaptation_prompt(self, scheduled: runner.ScheduledAttempt) -> runner.PromptPlan:
        if scheduled.arm == "IN_FULL_NEUTRAL_12":
            history = runner.neutral_history_plan(
                scheduled.task_id,
                scheduled.arm,
                _neutral_episodes(),
            )
        else:
            rows: list[runner.HistoryPresentation] = []
            for slot in range(12):
                source_ref = _ref(
                    f"full-fake-adaptation-history-{scheduled.task_id}-{slot}"
                )
                record = _memory_record(
                    ordinal=slot,
                    source_ref=source_ref,
                    content=f"Public ordinary adaptation evidence {slot:02d}.",
                )
                rows.append(
                    runner.HistoryPresentation(
                        slot=slot,
                        record_ref=record.record_ref,
                        acquisition_ref=_ref(
                            f"full-fake-adaptation-acquisition-{scheduled.task_id}-{slot}"
                        ),
                        source_ref=source_ref,
                        raw_record_sha256=hashlib.sha256(
                            record.canonical_bytes()
                        ).hexdigest(),
                        presented_text=record.content,
                        neutral=False,
                    )
                )
            history = runner.HistoryPlan(
                arm=scheduled.arm,
                task_id=scheduled.task_id,
                presentations=tuple(rows),
            )
        prompt = runner.build_v1_proposal_prompt(
            self._task_text(scheduled.task),
            history.evidence_texts,
        )
        token_ids = self._tokenize(prompt)
        return runner.PromptPlan(
            arm=scheduled.arm,
            task_id=scheduled.task_id,
            history_plan=history,
            prompt=prompt,
            prompt_ref=runner.content_ref({"prompt": prompt}),
            token_ids=token_ids,
            padding_atoms=0,
            target_tokens=len(token_ids),
        )

    def prepare_adaptation_attempt(
        self, scheduled: runner.ScheduledAttempt
    ) -> runner.PromptPlan:
        self.events.append(f"prepare-adaptation:{scheduled.stage_key if hasattr(scheduled, 'stage_key') else scheduled.task_id}:{scheduled.arm}")
        return self._adaptation_prompt(scheduled)

    def _allocation_name(self, stage: runner.AttemptStage, role: str) -> str:
        return runner.canonical_ref(
            "full-fake-disposable-name",
            {"role": role, "stage_ref": stage.stage_ref},
        ).removeprefix("sha256:")[:24]

    def execute_attempt(
        self,
        scheduled: runner.ScheduledAttempt,
        stage: runner.AttemptStage,
        plan: runner.PromptPlan,
        permit: runner.AttemptGenerationPermit,
        resources: runner.ResourceLedger,
        context: runner.ArmExecutionContext | None,
    ) -> runner.AttemptObservation:
        if resources is not self.resources:
            raise AssertionError("aggregate owner received another resource ledger")
        if scheduled.phase == "adaptation":
            if context is not None or scheduled.arm not in runner.STATEFUL_ARMS:
                raise AssertionError("adaptation capability changed")
            scope = self.cognee_scope_parent / (
                "persistent-" + self._allocation_name(stage, scheduled.arm)
            )
            scope.mkdir(mode=0o700)
            scope.chmod(0o700)
            self.persistent_scopes[stage.stage_ref] = scope
        else:
            if context is None:
                raise AssertionError("development/final arm lacks its context")
            name = self._allocation_name(stage, scheduled.arm)
            runtime_root = self.arm_runtime_parent / f"attempt-{name}"
            scope_roots: tuple[str, ...] = ()
            baseline: runner.BaselineSeal | None = None
            if scheduled.arm in runner.STATEFUL_ARMS:
                baseline = context.baseline
                scope_roots = (str(self.cognee_scope_parent / f"scope-{name}"),)
            allocation = runner.DisposablePathAllocation(
                purpose=self.purpose,
                task_id=stage.task_id,
                replicate_commitment=stage.replicate_commitment,
                role=scheduled.arm,
                runtime_root=str(runtime_root),
                cognee_scope_roots=scope_roots,
            )
            self.path_registry.register(allocation)
            runtime_root.mkdir(mode=0o700)
            runtime_root.chmod(0o700)
            for raw_scope in scope_roots:
                scope = Path(raw_scope)
                scope.mkdir(mode=0o700)
                scope.chmod(0o700)
            if scheduled.arm in runner.STATELESS_ARMS:
                journal = runtime_root / "qwen-execution-journal.sqlite3"
                journal.write_bytes(b"full-fake-stateless-journal-v1")
                journal.chmod(0o600)
            elif scheduled.arm == "IO_FULL_ORDINARY_12":
                replay = context.ordinary_private_replay
                if replay is None:
                    raise AssertionError("IO lost its private replay capability")

                class ReplayTarget:
                    def recall_from_hits(
                        self,
                        _hits: tuple[object, ...],
                        *,
                        limit: int,
                        frozen_origin: bool,
                    ) -> object:
                        if limit != 12 or frozen_origin is not True:
                            raise AssertionError("private replay authority changed")
                        return replay.batch

                replay.replay(ReplayTarget())
            self.disposable[stage.stage_ref] = (allocation, baseline)

        permit.before_proposal_generation()
        permit.before_execution_generation()
        proposals = (
            "Use the first bounded public synthetic procedure.",
            "Use the second bounded public synthetic procedure.",
        )
        selected_index = (
            runner.control_selection_index(stage.public_task_ref, len(proposals))
            if scheduled.arm in runner.STATELESS_ARMS
            else 0
        )
        pair_name = {
            "F0_NEUTRAL_FORMATTED_12": "neutral-pair",
            "IN_FULL_NEUTRAL_12": "neutral-pair",
            "O0_ORDINARY_FROZEN_12": "ordinary-pair",
            "IO_FULL_ORDINARY_12": "ordinary-pair",
        }.get(scheduled.arm, scheduled.arm)
        if scheduled.arm in runner.STATEFUL_ARMS:
            resources.record_runtime_embeddings(
                runner.RUNTIME_EMBEDDING_ROWS_PER_STATEFUL_ATTEMPT
            )
        return runner.AttemptObservation(
            task_id=stage.task_id,
            arm=scheduled.arm,
            proposal_admitted=True,
            proposals=proposals,
            selected_index=selected_index,
            selected_trace=proposals[selected_index],
            raw_response="FULL_FAKE_NONCONFORMING_RESPONSE",
            proposal_generation_ref=_ref(
                f"full-fake-generation-{stage.task_id}-{pair_name}"
            ),
            proposal_prompt_ref=plan.prompt_ref,
            proposal_prompt_tokens=len(plan.token_ids),
            proposal_output_tokens=1,
            execution_request_ref=_ref(
                f"full-fake-execution-request-{stage.stage_ref}"
            ),
            execution_receipt_ref=_ref(
                f"full-fake-execution-receipt-{stage.stage_ref}"
            ),
            execution_prompt_ref=_ref(
                f"full-fake-execution-prompt-{stage.stage_ref}"
            ),
            execution_prompt_tokens=1,
            execution_output_tokens=1,
            selected_trace_bound=True,
            response_bound=True,
            selected_grammar_class="PUBLIC_SYNTHETIC",
            response_grammar_class="PUBLIC_SYNTHETIC",
            recalled_record_refs=tuple(
                item.record_ref for item in plan.history_plan.presentations
            ),
        )

    def apply_objective_feedback(
        self,
        scheduled: runner.ScheduledAttempt,
        stage: runner.AttemptStage,
        observation: runner.AttemptObservation,
        judgment: dict[str, object],
        resources: runner.ResourceLedger,
    ) -> runner.StatefulFeedbackApplication:
        if (
            scheduled.arm not in runner.STATEFUL_ARMS
            or not observation.proposal_admitted
            or resources is not self.resources
            or judgment["attempt_receipt_ref"] != stage.attempt_receipt_ref
        ):
            raise AssertionError("objective-feedback gate changed")
        updated = replace(
            observation,
            resolution_ref=_ref(f"full-fake-resolution-{stage.stage_ref}"),
            prospective_reservation_ref=_ref(
                f"full-fake-reservation-{stage.stage_ref}"
            ),
            prospective_batch_ref=_ref(f"full-fake-batch-{stage.stage_ref}"),
        )
        return runner.StatefulFeedbackApplication(updated, None)

    def finalize_persistent_stateful_attempt(
        self,
        scheduled: runner.ScheduledAttempt,
        stage: runner.AttemptStage,
        observation: runner.AttemptObservation,
        _judgment: dict[str, object],
        feedback_applied: bool,
        terminal_disposition_ref: str,
        resources: runner.ResourceLedger,
    ) -> runner.PersistentRuntimeCleanupEvidence:
        if resources is not self.resources or feedback_applied is not True:
            raise AssertionError("persistent finalization gate changed")
        self.ledger.require_terminal_disposition(
            task_id=stage.task_id,
            arm=stage.arm,
            stage_ref=stage.stage_ref,
            attempt_receipt_ref=stage.attempt_receipt_ref,
            terminal_disposition_ref=terminal_disposition_ref,
        )
        owner = self.lineages.owner_for(stage.replicate_commitment, stage.arm)
        scope = self.persistent_scopes.pop(stage.stage_ref)
        terminal = runner.PersistentRuntimeTerminalEvidence(
            task_id=stage.task_id,
            arm=stage.arm,
            replicate_commitment=stage.replicate_commitment,
            stage_ref=stage.stage_ref,
            attempt_receipt_ref=stage.attempt_receipt_ref,
            terminal_disposition_ref=terminal_disposition_ref,
            runtime_root=str(owner.root),
            cognee_scope_root=str(scope),
            allocation_ref=owner.allocation_ref,
            dataset_id=runner.canonical_ref(
                "full-fake-cognee-incarnation",
                {"stage_ref": stage.stage_ref},
            ),
            hashes=owner.initial_hashes,
            feedback_applied=True,
            quiescent=True,
        )
        self.ledger.append_artifact_durable(
            terminal.terminal_ref,
            "persistent-runtime-terminal",
            terminal.to_canonical(),
        )
        scope.rmdir()
        cleanup = runner.PersistentRuntimeCleanupEvidence(terminal, True, True)
        self.ledger.append_artifact_durable(
            cleanup.cleanup_ref,
            "persistent-runtime-cleanup",
            cleanup.to_canonical(),
        )
        if observation.resolution_ref is None:
            raise AssertionError("persistent cleanup preceded objective feedback")
        return cleanup

    def _catalog_for_replicate(
        self,
        replicate_commitment: str,
        attempts: tuple[runner.AttemptFinal, ...],
    ) -> runner.DonorCatalog:
        admitted = tuple(
            item
            for item in attempts
            if item.stage.replicate_commitment == replicate_commitment
            and item.stage.arm == "IO_FULL_ORDINARY_12"
        )
        if len(admitted) != 45:
            raise AssertionError("full fake did not produce all 45 IO donors")
        bootstrap = _bootstrap_episode()
        neutral_refs = (bootstrap.episode_ref, *(item.episode_ref for item in _neutral_episodes()))
        rows: list[CognitiveAcquisitionItem] = []
        predecessor: str | None = None
        for ordinal, source_ref in enumerate(neutral_refs):
            record = _memory_record(
                ordinal=ordinal,
                source_ref=source_ref,
                content=f"Full fake neutral acquisition {ordinal:02d}.",
            )
            acquisition = CognitiveAcquisition(
                ordinal=ordinal,
                predecessor_acquisition_ref=predecessor,
                source_contract="ANG-CTR-COGNITIVE-EPISODE-001@0.1.0",
                source_ref=source_ref,
                record=record,
            )
            rows.append(
                CognitiveAcquisitionItem(
                    ordinal,
                    acquisition.acquisition_ref,
                    record.record_ref,
                    acquisition,
                )
            )
            predecessor = acquisition.acquisition_ref
        next_ordinal = len(rows)
        for final in admitted:
            batch_source_ref = final.observation.prospective_batch_ref
            source_ref = final.observation.resolution_ref
            if batch_source_ref is None or source_ref is None:
                raise AssertionError(
                    "admitted IO attempt lacks dynamics batch or resolution"
                )
            batch_record = _proposed_batch_record(
                ordinal=next_ordinal,
                source_ref=batch_source_ref,
                content=(
                    "Full fake public proposed dynamics batch "
                    f"{final.stage.family} {final.stage.task_ordinal:02d}."
                ),
            )
            batch_acquisition = CognitiveAcquisition(
                ordinal=next_ordinal,
                predecessor_acquisition_ref=predecessor,
                source_contract=PROSPECTIVE_DYNAMICS_BATCH_CONTRACT,
                source_ref=batch_source_ref,
                record=batch_record,
            )
            rows.append(
                CognitiveAcquisitionItem(
                    next_ordinal,
                    batch_acquisition.acquisition_ref,
                    batch_record.record_ref,
                    batch_acquisition,
                )
            )
            predecessor = batch_acquisition.acquisition_ref
            next_ordinal += 1
            record = _memory_record(
                ordinal=next_ordinal,
                source_ref=source_ref,
                content=(
                    "Full fake public observed donor "
                    f"{final.stage.family} {final.stage.task_ordinal:02d}."
                ),
            )
            acquisition = CognitiveAcquisition(
                ordinal=next_ordinal,
                predecessor_acquisition_ref=predecessor,
                source_contract="ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0",
                source_ref=source_ref,
                record=record,
            )
            rows.append(
                CognitiveAcquisitionItem(
                    next_ordinal,
                    acquisition.acquisition_ref,
                    record.record_ref,
                    acquisition,
                )
            )
            predecessor = acquisition.acquisition_ref
            next_ordinal += 1
        items = tuple(rows)
        self.catalog_items[replicate_commitment] = items
        return runner.build_donor_catalog(
            items,
            admitted,
            replicate_commitment=replicate_commitment,
            expected_next_ordinal=len(items),
        )

    def freeze_adaptation(
        self, attempts: tuple[runner.AttemptFinal, ...]
    ) -> runner.FrozenAdaptationState:
        expected = 48 if self.purpose == "qualification" else 270
        if len(attempts) != expected or self.persistent_scopes:
            raise AssertionError("adaptation did not close before freeze")
        neutral = _neutral_episodes()
        bootstrap = _bootstrap_episode()
        chain = runner.validate_neutral_fixture_chain(
            neutral,
            bootstrap_episode_ref=bootstrap.episode_ref,
            competence_state_digest=_STATE_REF,
            model_ref=_MODEL_REF,
            encoder_ref=_ENCODER_REF,
        )
        seals: list[runner.BaselineSeal] = []
        catalogs: list[runner.DonorCatalog] = []
        for replicate in self.lineages.replicate_commitments:
            catalogs.append(self._catalog_for_replicate(replicate, attempts))
            for arm in ("IN_FULL_NEUTRAL_12", "IO_FULL_ORDINARY_12"):
                owner = self.lineages.owner_for(replicate, arm)
                seals.append(
                    runner.BaselineSeal(
                        purpose=self.purpose,
                        replicate_commitment=replicate,
                        arm=arm,
                        root=str(owner.root),
                        hashes=owner.initial_hashes,
                        neutral_episode_refs=tuple(item.episode_ref for item in neutral),
                        neutral_chain_ref=str(chain["chain_ref"]),
                        bootstrap_episode_ref=bootstrap.episode_ref,
                        genesis_competence_digest=_STATE_REF,
                        model_ref=_MODEL_REF,
                        encoder_ref=_ENCODER_REF,
                        quiescent=True,
                    )
                )
        self.frozen = runner.FrozenAdaptationState(
            purpose=self.purpose,
            baseline_seals=tuple(seals),
            donor_catalogs=tuple(catalogs),
        )
        self.product._frozen = self.frozen
        return self.frozen

    def _capture_allocation(
        self,
        *,
        task: dict[str, object],
        kind: str,
    ) -> runner.DisposablePathAllocation:
        task_id = str(task["task_id"])
        replicate = str(task["replicate_commitment"])
        token = runner.canonical_ref(
            "full-fake-capture-name", {"kind": kind, "task_id": task_id}
        ).removeprefix("sha256:")[:24]
        allocation = runner.DisposablePathAllocation(
            purpose=self.purpose,
            task_id=task_id,
            replicate_commitment=replicate,
            role=f"recall-{kind}",
            runtime_root=str(self.arm_runtime_parent / f"capture-{token}"),
            cognee_scope_roots=(
                str(self.cognee_scope_parent / f"capture-{token}"),
            ),
        )
        self.path_registry.register(allocation)
        for target in allocation.targets:
            path = Path(target)
            path.mkdir(mode=0o700)
            path.chmod(0o700)
        return allocation

    def _neutral_batch(self) -> _RecallBatch:
        rows: list[_RecallItem] = []
        for slot, episode in enumerate(_neutral_episodes()):
            record = _memory_record(
                ordinal=slot,
                source_ref=episode.episode_ref,
                content=f"Neutral capture source {slot:02d}.",
            )
            rows.append(
                _RecallItem(
                    record=record,
                    record_ref=record.record_ref,
                    acquisition_ref=_ref(f"full-fake-neutral-acquisition-{slot}"),
                    source_ref=episode.episode_ref,
                    text=record.content,
                    ordinal=slot,
                    source_contract="ANG-CTR-COGNITIVE-EPISODE-001@0.1.0",
                    temporal_marker=f"ordinal:{slot}",
                )
            )
        return _RecallBatch(tuple(rows))

    def _ordinary_replay(
        self,
        *,
        task: dict[str, object],
        catalog: runner.DonorCatalog,
        capture: runner.FrozenRecallCapture,
    ) -> runner.PrivateRecallReplay:
        items = self.catalog_items[str(task["replicate_commitment"])]
        by_record = {item.record_ref: item for item in items}
        entries = catalog.entries[:12]
        hits = tuple(
            AcquisitionReferenceHit(
                record_ref=entry.record_ref,
                score=float(12 - index),
                backend_ref="full-fake-private-replay",
            )
            for index, entry in enumerate(entries)
        )
        recalls = tuple(
            AcquisitionRecall(
                record=by_record[entry.record_ref].acquisition.record,
                acquisition_ref=entry.acquisition_ref,
                source_contract=entry.source_contract,
                source_ref=entry.source_ref,
                position=TemporalPosition(
                    acquired_ordinal=entry.acquired_ordinal,
                    age=0,
                    landmark_relations=(),
                    world_valid_from=None,
                    world_valid_until=None,
                ),
                adjacent_records=(),
                backend_score=float(12 - index),
                backend_ref="full-fake-private-replay",
                world_valid_at_query=None,
            )
            for index, entry in enumerate(entries)
        )
        return runner.PrivateRecallReplay(
            capture,
            hits,
            AcquisitionRecallBatch(items=recalls, rejected=()),
        )

    def prepare_task_block(
        self,
        task: dict[str, object],
        context: runner.TaskPreparationContext,
    ) -> runner.PreparedTaskBlock:
        if self.frozen is None or context.frozen_state_ref != self.frozen.freeze_ref:
            raise AssertionError("task preparation escaped frozen adaptation")
        neutral_allocation = self._capture_allocation(task=task, kind="neutral")
        ordinary_allocation = self._capture_allocation(task=task, kind="ordinary")
        neutral_capture = runner.freeze_recall_capture(
            task=task,
            kind="neutral",
            recall_batch=self._neutral_batch(),
            query_ref=_ref(f"full-fake-neutral-query-{context.task_id}"),
            backend_binding_ref=_ref(
                f"full-fake-neutral-backend-{context.task_id}"
            ),
            source_baseline_seal_ref=context.in_baseline.seal_ref,
            capture_clone_ref=_ref(
                f"full-fake-neutral-clone-{context.task_id}"
            ),
            capture_quiescent=True,
            capture_disposed=False,
            cleanup_absence_verified=False,
            cleanup_paths=neutral_allocation.targets,
            source_root=context.in_baseline.root,
            source_hashes=dict(context.in_baseline.hashes),
            allocation_ref=neutral_allocation.allocation_ref,
        )
        items = self.catalog_items[context.replicate_commitment]
        by_record = {item.record_ref: item for item in items}
        ordinary_entries = context.donor_catalog.entries[:12]
        ordinary_batch = _RecallBatch(
            tuple(
                _RecallItem(
                    record=by_record[entry.record_ref].acquisition.record,
                    record_ref=entry.record_ref,
                    acquisition_ref=entry.acquisition_ref,
                    source_ref=entry.source_ref,
                    text=entry.public_record_text,
                    ordinal=entry.acquired_ordinal,
                    source_contract=entry.source_contract,
                    temporal_marker=f"ordinal:{entry.acquired_ordinal}",
                )
                for entry in ordinary_entries
            )
        )
        ordinary_capture = runner.freeze_recall_capture(
            task=task,
            kind="ordinary",
            recall_batch=ordinary_batch,
            query_ref=_ref(f"full-fake-ordinary-query-{context.task_id}"),
            backend_binding_ref=_ref(
                f"full-fake-ordinary-backend-{context.task_id}"
            ),
            source_baseline_seal_ref=context.io_baseline.seal_ref,
            capture_clone_ref=_ref(
                f"full-fake-ordinary-clone-{context.task_id}"
            ),
            capture_quiescent=True,
            capture_disposed=False,
            cleanup_absence_verified=False,
            cleanup_paths=ordinary_allocation.targets,
            source_root=context.io_baseline.root,
            source_hashes=dict(context.io_baseline.hashes),
            allocation_ref=ordinary_allocation.allocation_ref,
        )
        replay = self._ordinary_replay(
            task=task,
            catalog=context.donor_catalog,
            capture=ordinary_capture,
        )
        embeddings = {
            entry.record_ref: (
                1.0,
                float((entry.acquired_ordinal % 17) + 1),
            )
            for entry in context.donor_catalog.entries
        }
        selections = {
            mode: runner.build_ranked_selection_plan(
                task=task,
                catalog=context.donor_catalog,
                mode=mode,
                task_embedding=(1.0, 0.0),
                record_embeddings=embeddings,
            )
            for mode in ("cross-low", "same-low", "same-high")
        }
        runner.validate_disjoint_same_family_selections(
            selections["same-low"].selected,
            selections["same-high"].selected,
        )
        task_id = context.task_id
        histories: dict[runner.Arm, runner.HistoryPlan] = {
            "F0_NEUTRAL_FORMATTED_12": runner.recall_history_plan(
                task_id,
                "F0_NEUTRAL_FORMATTED_12",
                neutral_capture,
                neutral=True,
            ),
            "IN_FULL_NEUTRAL_12": runner.recall_history_plan(
                task_id,
                "IN_FULL_NEUTRAL_12",
                neutral_capture,
                neutral=True,
            ),
            "O0_ORDINARY_FROZEN_12": runner.recall_history_plan(
                task_id,
                "O0_ORDINARY_FROZEN_12",
                ordinary_capture,
                neutral=False,
            ),
            "IO_FULL_ORDINARY_12": runner.recall_history_plan(
                task_id,
                "IO_FULL_ORDINARY_12",
                ordinary_capture,
                neutral=False,
            ),
            "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6": runner.ranked_control_history_plan(
                task_id,
                "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6",
                selections["cross-low"],
                _neutral_episodes(),
            ),
            "S0_SAME_FAMILY_LOW_6_NEUTRAL_6": runner.ranked_control_history_plan(
                task_id,
                "S0_SAME_FAMILY_LOW_6_NEUTRAL_6",
                selections["same-low"],
                _neutral_episodes(),
            ),
            "S1_SAME_FAMILY_HIGH_6_NEUTRAL_6": runner.ranked_control_history_plan(
                task_id,
                "S1_SAME_FAMILY_HIGH_6_NEUTRAL_6",
                selections["same-high"],
                _neutral_episodes(),
            ),
        }
        plans, equalities = runner.plan_equalized_prompts(
            task_text=self._task_text(task),
            history_plans=histories,
            tokenize_chat_prompt=self._tokenize,
        )
        plans["N0_NATIVE_NO_HISTORY"] = runner.native_prompt_plan(
            task_id=task_id,
            task_text=self._task_text(task),
            tokenize_chat_prompt=self._tokenize,
        )
        return runner.PreparedTaskBlock.create(
            task=task,
            prompt_plans=plans,
            pair_equalities=equalities,
            frozen=self.frozen,
            neutral_recall=neutral_capture,
            ordinary_recall=ordinary_capture,
            ordinary_private_replay=replay,
        )

    def dispose_recall_captures(
        self,
        block: runner.PreparedTaskBlock,
        terminal_refs: tuple[str, str],
    ) -> runner.PreparedTaskBlock:
        captures: list[runner.FrozenRecallCapture] = []
        for capture, terminal_ref in zip(
            (block.neutral_recall, block.ordinary_recall),
            terminal_refs,
            strict=True,
        ):
            self.ledger.require_artifact(
                terminal_ref,
                "recall-capture-terminal",
                capture.terminal_payload,
            )
            for target in capture.cleanup_paths:
                Path(target).rmdir()
            updated = replace(
                capture,
                capture_disposed=True,
                cleanup_absence_verified=True,
            )
            self.ledger.append_artifact_durable(
                updated.cleanup_ref,
                "recall-capture-cleanup",
                updated.cleanup_payload,
            )
            captures.append(updated)
        replay = runner.PrivateRecallReplay(
            captures[1],
            block.ordinary_private_replay.hits,
            block.ordinary_private_replay.batch,
        )
        return replace(
            block,
            neutral_recall=captures[0],
            ordinary_recall=captures[1],
            ordinary_private_replay=replay,
        )

    def terminalize_stateful_attempt(
        self,
        _scheduled: runner.ScheduledAttempt,
        stage: runner.AttemptStage,
        _observation: runner.AttemptObservation,
        _judgment: dict[str, object],
        feedback_applied: bool,
        resources: runner.ResourceLedger,
    ) -> runner.CloneTerminalEvidence:
        if feedback_applied is not True or resources is not self.resources:
            raise AssertionError("clone terminalization preceded feedback")
        allocation, baseline = self.disposable[stage.stage_ref]
        if baseline is None:
            raise AssertionError("stateful clone lost its frozen baseline")
        return runner.CloneTerminalEvidence(
            task_id=stage.task_id,
            arm=stage.arm,
            source_root=baseline.root,
            clone_root=allocation.runtime_root,
            source_before=baseline.hashes,
            source_after=baseline.hashes,
            clone_before=baseline.hashes,
            clone_terminal=baseline.hashes,
            quiescent=True,
            cognee_scope_roots=allocation.cognee_scope_roots,
            allocation_ref=allocation.allocation_ref,
        )

    def dispose_clone(
        self,
        terminal: runner.CloneTerminalEvidence,
        terminal_disposition_ref: str,
    ) -> runner.CloneLifecycleEvidence:
        self.ledger.require_terminal_disposition_payload(
            task_id=terminal.task_id,
            arm=terminal.arm,
            terminal_disposition_ref=terminal_disposition_ref,
            expected_clone_terminal_ref=terminal.terminal_ref,
        )
        for target in (*terminal.cognee_scope_roots, terminal.clone_root):
            Path(target).rmdir()
        lifecycle = runner.CloneLifecycleEvidence(
            terminal,
            terminal_disposition_ref,
            True,
            True,
        )
        self.ledger.append_artifact_durable(
            lifecycle.lifecycle_ref,
            "clone-lifecycle",
            lifecycle.to_canonical(),
        )
        self.disposable.pop(
            next(
                key
                for key, (allocation, _baseline) in self.disposable.items()
                if allocation.allocation_ref == terminal.allocation_ref
            )
        )
        return lifecycle

    def dispose_stateless_attempt(
        self,
        _scheduled: runner.ScheduledAttempt,
        stage: runner.AttemptStage,
        _observation: runner.AttemptObservation,
        terminal_disposition_ref: str,
    ) -> runner.StatelessRuntimeCleanupEvidence:
        self.ledger.require_terminal_disposition(
            task_id=stage.task_id,
            arm=stage.arm,
            stage_ref=stage.stage_ref,
            attempt_receipt_ref=stage.attempt_receipt_ref,
            terminal_disposition_ref=terminal_disposition_ref,
        )
        allocation, baseline = self.disposable.pop(stage.stage_ref)
        if baseline is not None:
            raise AssertionError("stateless runtime acquired a baseline")
        journal = Path(allocation.runtime_root) / "qwen-execution-journal.sqlite3"
        raw = journal.read_bytes()
        terminal = runner.StatelessRuntimeTerminalEvidence(
            task_id=stage.task_id,
            arm=stage.arm,
            runtime_root=allocation.runtime_root,
            journal_path=str(journal),
            journal_sha256=hashlib.sha256(raw).hexdigest(),
            journal_bytes=len(raw),
            allocation_ref=allocation.allocation_ref,
            terminal_disposition_ref=terminal_disposition_ref,
        )
        self.ledger.append_artifact_durable(
            terminal.terminal_ref,
            "stateless-runtime-terminal",
            terminal.to_canonical(),
        )
        journal.unlink()
        Path(allocation.runtime_root).rmdir()
        cleanup = runner.StatelessRuntimeCleanupEvidence(terminal, True, True)
        self.ledger.append_artifact_durable(
            cleanup.cleanup_ref,
            "stateless-runtime-cleanup",
            cleanup.to_canonical(),
        )
        return cleanup

    def verify_frozen_adaptation(
        self, frozen: runner.FrozenAdaptationState
    ) -> str:
        if frozen is not self.frozen:
            raise AssertionError("aggregate owner received another frozen state")
        for seal in frozen.baseline_seals:
            if not Path(seal.root).is_dir():
                raise AssertionError("retained frozen baseline disappeared")
        return frozen.freeze_ref

    def close(self) -> None:
        if self.persistent_scopes or self.disposable:
            raise AssertionError("aggregate owner closed with live attempt state")
        if not self.ledger._sealed or self.ledger._closed:
            raise AssertionError(
                "aggregate owner requires a readable logically sealed ledger"
            )
        self.path_registry.assert_closed()
        if tuple(self.cognee_scope_parent.iterdir()):
            raise AssertionError("aggregate owner retained a disposable Cognee scope")
        self.closed = True
        self.product._closed = True

    def patched_product_methods(self) -> ExitStack:
        harness = self
        functions = {
            "prepare_adaptation_attempt": lambda _owner, scheduled: harness.prepare_adaptation_attempt(scheduled),
            "freeze_adaptation": lambda _owner, attempts: harness.freeze_adaptation(attempts),
            "prepare_task_block": lambda _owner, task, context: harness.prepare_task_block(task, context),
            "dispose_recall_captures": lambda _owner, block, refs: harness.dispose_recall_captures(block, refs),
            "execute_attempt": lambda _owner, scheduled, stage, plan, permit, resources, context: harness.execute_attempt(scheduled, stage, plan, permit, resources, context),
            "apply_objective_feedback": lambda _owner, scheduled, stage, observation, judgment, resources: harness.apply_objective_feedback(scheduled, stage, observation, judgment, resources),
            "finalize_persistent_stateful_attempt": lambda _owner, scheduled, stage, observation, judgment, feedback, disposition, resources: harness.finalize_persistent_stateful_attempt(scheduled, stage, observation, judgment, feedback, disposition, resources),
            "terminalize_stateful_attempt": lambda _owner, scheduled, stage, observation, judgment, feedback, resources: harness.terminalize_stateful_attempt(scheduled, stage, observation, judgment, feedback, resources),
            "dispose_clone": lambda _owner, terminal, disposition: harness.dispose_clone(terminal, disposition),
            "dispose_stateless_attempt": lambda _owner, scheduled, stage, observation, disposition: harness.dispose_stateless_attempt(scheduled, stage, observation, disposition),
            "verify_frozen_adaptation": lambda _owner, frozen: harness.verify_frozen_adaptation(frozen),
            "close": lambda _owner: harness.close(),
        }
        stack = ExitStack()
        for name, function in functions.items():
            stack.enter_context(
                mock.patch.object(runner.V2ProductRuntimeOwner, name, new=function)
            )
        return stack


_SYNTHETIC_WORKER_CODE = (
    "import time\n"
    "while True:\n"
    "    time.sleep(60)\n"
)
_SYNTHETIC_WAITING_ROOT_CODE = (
    "import subprocess, sys\n"
    "child = subprocess.Popen(\n"
    "    [sys.executable, '-c', sys.argv[1]],\n"
    "    stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, close_fds=False,\n"
    ")\n"
    "print(child.pid, flush=True)\n"
    "raise SystemExit(child.wait())\n"
)
_SYNTHETIC_REPARENTING_ROOT_CODE = (
    "import os, subprocess, sys\n"
    "child = subprocess.Popen(\n"
    "    [sys.executable, '-c', sys.argv[1]],\n"
    "    stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, close_fds=False,\n"
    ")\n"
    "print(child.pid, flush=True)\n"
    "os._exit(0)\n"
)
_SYNTHETIC_UNKNOWN_HOLDER_CODE = (
    "import time\n"
    "unknown_holder_marker = True\n"
    "while unknown_holder_marker:\n"
    "    time.sleep(60)\n"
)
_SYNTHETIC_FAILED_START_CODE = (
    "import time\n"
    "failed_start_marker = True\n"
    "while failed_start_marker:\n"
    "    time.sleep(60)\n"
)


def _synthetic_worker_expectation(
    root_argv: tuple[str, ...],
    worker_argv: tuple[str, ...],
) -> runner._V2WorkerGraphExpectation:
    uid_line = tuple(
        line
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines()
        if line.startswith("Uid:")
    )
    if len(uid_line) != 1:
        raise AssertionError("synthetic worker test cannot bind its uid tuple")
    uids = tuple(int(value) for value in uid_line[0].split()[1:])
    executable = str(Path(os.sys.executable).resolve(strict=True))
    return runner._V2WorkerGraphExpectation(
        root_argv=root_argv,
        worker_argv=worker_argv,
        allowed_intermediate_argv=(root_argv,),
        root_executable=executable,
        worker_executable=executable,
        root_uids=uids,
        worker_uids=uids,
        maximum_nodes=2,
    )


async def _wait_for_test_condition(
    predicate: object,
    *,
    label: str,
    timeout_seconds: float = 5.0,
) -> None:
    if not callable(predicate):
        raise TypeError("test condition must be callable")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.01)


def _terminate_test_pidfd(pidfd: int | None) -> None:
    if pidfd is None:
        return
    try:
        if not runner._v2_pidfd_terminal(pidfd):
            signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
    except (OSError, ValueError):
        pass


def _close_test_fd(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


async def _terminate_test_process(
    process: asyncio.subprocess.Process | None,
) -> None:
    if process is None:
        return
    if process.stdin is not None:
        try:
            process.stdin.close()
        except (AttributeError, OSError, RuntimeError):
            pass
    if process.returncode is None:
        try:
            process.kill()
        except (ProcessLookupError, RuntimeError):
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=3.0)
    except (asyncio.TimeoutError, ProcessLookupError, RuntimeError):
        pass


class V2WorkerLifecycleTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("ANGLER_RUN_V2_COGNEE_INCARNATION_SMOKE") == "1",
        "set ANGLER_RUN_V2_COGNEE_INCARNATION_SMOKE=1 inside the approved isolation",
    )
    def test_live_same_scope_uses_fresh_cognee_incarnations_after_removal(
        self,
    ) -> None:
        async def exercise() -> None:
            from angler.memory.cognitive_acquisition import (
                CognitiveGraphProjectionV2,
            )

            runner._validate_v2_inherited_parent_attestation(
                runner._attest_v2_inherited_worker_parent()
            )
            with tempfile.TemporaryDirectory(
                prefix=".v2-cognee-incarnation-smoke-",
                dir="/opt/angler/state/project-angler",
            ) as directory:
                parent = Path(directory).resolve()
                parent.chmod(0o700)
                store = CognitiveTransactionStore(parent / "canonical.sqlite3")
                store.initialize(
                    _STATE_REF,
                    _SNAPSHOT,
                    model_ref=_MODEL_REF,
                    encoder_ref=_ENCODER_REF,
                )
                episode = _bootstrap_episode()
                store.commit_episode(
                    episode,
                    _SNAPSHOT,
                    expected_parent_digest=_STATE_REF,
                )
                item = store.acquisition_items(limit=1)[0]
                projection = CognitiveGraphProjectionV2.from_acquisition(
                    item.acquisition
                )
                query = item.acquisition.record.content
                scope = runner.V2AcquisitionScopeSpec.create(
                    purpose="qualification",
                    replicate_commitment=runner.qualification_replicate_commitment(),
                    arm="IN_FULL_NEUTRAL_12",
                    scope_kind="persistent-adaptation",
                    state_parent=parent,
                )

                async def project_and_recall(
                    session: runner.V2AcquisitionSession,
                ) -> None:
                    backend_ref = await session.backend.project(projection)
                    self.assertIsNotNone(backend_ref)
                    hits = tuple(await session.backend.search(query, limit=1))
                    self.assertEqual(
                        tuple(hit.record_ref for hit in hits),
                        (item.record_ref,),
                    )

                active: runner.V2AcquisitionSession | None = None
                try:
                    first = await runner.open_v2_acquisition_scope_live(scope)
                    active = first
                    first_dataset = first.dataset_id
                    first_user = first.binding.user_id
                    first_tenant = first.binding.tenant_id
                    await project_and_recall(first)
                    await first.close_preserving_state()
                    active = None
                    self.assertTrue(Path(scope.state_root).exists())

                    with self.assertRaises(runner.V2IntegrityStop):
                        await runner.open_v2_acquisition_scope_live(
                            scope,
                            expected_dataset_id=(
                                "00000000-0000-0000-0000-000000000001"
                            ),
                        )
                    runner._assert_no_unresolved_v2_cognee_workers()

                    retained = await runner.open_v2_acquisition_scope_live(
                        scope,
                        expected_dataset_id=first_dataset,
                    )
                    active = retained
                    self.assertEqual(retained.dataset_id, first_dataset)
                    self.assertEqual(retained.binding.user_id, first_user)
                    self.assertEqual(retained.binding.tenant_id, first_tenant)
                    await retained.forget_close_and_remove()
                    active = None
                    self.assertFalse(Path(scope.state_root).exists())

                    second = await runner.open_v2_acquisition_scope_live(scope)
                    active = second
                    self.assertNotEqual(second.dataset_id, first_dataset)
                    self.assertNotEqual(second.binding.user_id, first_user)
                    self.assertNotEqual(second.binding.tenant_id, first_tenant)
                    await project_and_recall(second)
                    await second.forget_close_and_remove()
                    active = None
                    self.assertFalse(Path(scope.state_root).exists())
                    runner._assert_no_unresolved_v2_cognee_workers()
                    self.assertEqual(
                        tuple(path for path in parent.iterdir() if path.is_dir()),
                        (),
                    )
                finally:
                    if active is not None and not active._closed:
                        await active.forget_close_and_remove()
                    await runner._drain_unresolved_v2_cognee_workers()
                    if Path(scope.state_root).exists():
                        runner._remove_exact_private_tree(
                            Path(scope.state_root),
                            expected_parent=parent,
                        )

        asyncio.run(exercise())

    @unittest.skipUnless(
        os.environ.get("ANGLER_RUN_V2_COGNEE_INCARNATION_SMOKE") == "1",
        "set ANGLER_RUN_V2_COGNEE_INCARNATION_SMOKE=1 inside the approved isolation",
    )
    def test_live_thirteen_record_ordinary_recall_reaches_twelve_hits(
        self,
    ) -> None:
        async def exercise() -> None:
            from transformers import AutoTokenizer

            runner._validate_v2_inherited_parent_attestation(
                runner._attest_v2_inherited_worker_parent()
            )
            with tempfile.TemporaryDirectory(
                prefix=".v2-cognee-ordinary-recall-smoke-",
                dir="/opt/angler/state/project-angler",
            ) as directory:
                parent = Path(directory).resolve()
                parent.chmod(0o700)
                store = CognitiveTransactionStore(parent / "canonical.sqlite3")
                store.initialize(
                    _STATE_REF,
                    _SNAPSHOT,
                    model_ref=_MODEL_REF,
                    encoder_ref=_ENCODER_REF,
                )
                bootstrap = _bootstrap_episode()
                store.commit_episode(
                    bootstrap,
                    _SNAPSHOT,
                    expected_parent_digest=_STATE_REF,
                )
                for episode in _neutral_episodes():
                    store.commit_episode(
                        episode,
                        _SNAPSHOT,
                        expected_parent_digest=_STATE_REF,
                    )
                self.assertEqual(store.acquisition_head().next_ordinal, 13)
                pending = tuple(
                    store.pending_acquisition_projections(
                        limit=runner.MAXIMUM_PROJECTION_RETRY
                    )
                )
                self.assertEqual(len(pending), 13)
                for item in pending:
                    store.ack_acquisition_projection(item.projection_ref)
                self.assertEqual(
                    tuple(store.pending_acquisition_projections(limit=1)),
                    (),
                )

                suite = evaluator.make_qualification_evaluator()
                task = suite.release_phase("adaptation")[0]
                task_text = evaluator.render_public_task(task)
                scope = runner.V2AcquisitionScopeSpec.create(
                    purpose="qualification",
                    replicate_commitment=runner.qualification_replicate_commitment(),
                    arm="IO_FULL_ORDINARY_12",
                    scope_kind="persistent-adaptation",
                    state_parent=parent,
                )
                session: runner.V2AcquisitionSession | None = None
                stage = "open"
                try:
                    session = await runner.open_v2_acquisition_scope_live(scope)
                    memory = runner._live_v2_lineage_mechanics().memory_factory(
                        source=store,
                        backend=session.backend,
                    )
                    stage = "rebuild-call"
                    rebuilt = tuple(
                        await memory.rebuild(
                            limit=runner.MAXIMUM_PROJECTION_RETRY
                        )
                    )
                    stage = "rebuild-count"
                    self.assertEqual(len(rebuilt), 13)
                    stage = "retry-pending"
                    self.assertEqual(
                        tuple(
                            await memory.retry_pending_projections(
                                limit=runner.MAXIMUM_PROJECTION_RETRY
                            )
                        ),
                        (),
                    )
                    stage = "synchronization"
                    memory.assert_synchronized(store.acquisition_head())

                    stage = "ordinary-search"
                    hits = runner._validated_reference_hits(
                        await session.backend.search(
                            task_text,
                            limit=runner.RECALL_SLOTS,
                        ),
                        label="V2 live ordinary diagnostic",
                    )
                    self.assertEqual(len(hits), runner.RECALL_SLOTS)
                    self.assertEqual(
                        len({item.record_ref for item in hits}),
                        runner.RECALL_SLOTS,
                    )

                    stage = "ordinary-recall"
                    batch = memory.recall_from_hits(
                        hits,
                        limit=runner.RECALL_SLOTS,
                        frozen_origin=False,
                    )
                    self.assertEqual(batch.rejected, ())
                    self.assertEqual(len(batch.items), runner.RECALL_SLOTS)
                    self.assertEqual(
                        tuple(item.record.record_ref for item in batch.items),
                        tuple(item.record_ref for item in hits),
                    )
                    self.assertTrue(
                        all(item.adjacent_records == () for item in batch.items)
                    )
                    history = runner._history_plan_from_recall_batch(
                        task_id=task.task_id,
                        arm="IO_FULL_ORDINARY_12",
                        batch=batch,
                    )

                    stage = "prompt-tokenization"
                    tokenizer = AutoTokenizer.from_pretrained(
                        runner.MODEL_ROOT,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    prompt = runner._unpadded_prompt_plan(
                        task_text=task_text,
                        history_plan=history,
                        tokenize_chat_prompt=lambda value: (
                            runner.frozen_qwen_chat_token_ids(
                                tokenizer,
                                value,
                                enable_thinking=False,
                            )
                        ),
                    )
                    self.assertLessEqual(
                        prompt.target_tokens,
                        runner.MAXIMUM_INPUT_TOKENS,
                    )
                    stage = "cleanup"
                    await session.forget_close_and_remove()
                    session = None
                    self.assertFalse(Path(scope.state_root).exists())
                    runner._assert_no_unresolved_v2_cognee_workers()
                except BaseException as error:
                    raise AssertionError(
                        "live 13-record ordinary-recall diagnostic failed "
                        f"at substage {stage}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                finally:
                    propagating = os.sys.exc_info()[0] is not None
                    cleanup_failures: list[BaseException] = []
                    if session is not None and not session._closed:
                        try:
                            await session.forget_close_and_remove()
                        except BaseException as error:
                            cleanup_failures.append(error)
                            try:
                                await session.close_preserving_state()
                            except BaseException as close_error:
                                cleanup_failures.append(close_error)
                    try:
                        await runner._drain_unresolved_v2_cognee_workers()
                    except BaseException as error:
                        cleanup_failures.append(error)
                    if (
                        Path(scope.state_root).exists()
                        and (session is None or session._closed)
                    ):
                        runner._remove_exact_private_tree(
                            Path(scope.state_root),
                            expected_parent=parent,
                        )
                    if cleanup_failures and not propagating:
                        raise ExceptionGroup(
                            "live ordinary-recall diagnostic cleanup failed",
                            cleanup_failures,
                        )

        asyncio.run(exercise())

    @unittest.skipUnless(
        os.environ.get("ANGLER_RUN_V2_COGNEE_INCARNATION_SMOKE") == "1",
        "set ANGLER_RUN_V2_COGNEE_INCARNATION_SMOKE=1 inside the approved isolation",
    )
    def test_live_r4_fifteen_record_rejoin_filters_proposed_then_fills_twelve(
        self,
    ) -> None:
        async def exercise() -> None:
            runner._validate_v2_inherited_parent_attestation(
                runner._attest_v2_inherited_worker_parent()
            )
            consumed_store = Path(
                "/opt/angler/state/project-angler/"
                "high-level-multidomain-v2-history-conditioning-isolation-r4/"
                "qualification-v1/arm-runtime/"
                "lineage-48563a9acb5e863d1bffa334/cognitive.sqlite3"
            )
            consumed_raw = runner._stable_regular_bytes(
                consumed_store,
                expected_mode=0o600,
                expected_owner_uid=os.geteuid(),
                expected_owner_gid=os.getegid(),
                maximum_bytes=4 * 1024 * 1024,
                label="consumed R4 IO canonical store",
            )
            self.assertEqual(
                hashlib.sha256(consumed_raw).hexdigest(),
                "eaaf66e6802d6f574fbd3460701bcb98a1955391481140eb098ed9efab39c65c",
            )
            with tempfile.TemporaryDirectory(
                prefix=".v2-r5-cognee-proposed-filter-smoke-",
                dir="/opt/angler/state/project-angler",
            ) as directory:
                parent = Path(directory).resolve()
                parent.chmod(0o700)
                copied_store = parent / "canonical.sqlite3"
                copied_store.write_bytes(consumed_raw)
                copied_store.chmod(0o600)
                store = CognitiveTransactionStore(copied_store)
                store.audit_integrity()
                inventory = tuple(store.acquisition_items(limit=64))
                self.assertEqual(len(inventory), 15)
                self.assertEqual(
                    Counter(
                        item.acquisition.source_contract for item in inventory
                    ),
                    Counter(
                        {
                            LEGACY_COGNITIVE_EPISODE_CONTRACT: 13,
                            PROSPECTIVE_DYNAMICS_BATCH_CONTRACT: 1,
                            PROSPECTIVE_RESOLUTION_CONTRACT: 1,
                        }
                    ),
                )
                proposed_ref = next(
                    item.record_ref
                    for item in inventory
                    if item.acquisition.source_contract
                    == PROSPECTIVE_DYNAMICS_BATCH_CONTRACT
                )
                scope = runner.V2AcquisitionScopeSpec.create(
                    purpose="qualification",
                    replicate_commitment=runner.qualification_replicate_commitment(),
                    arm="IO_FULL_ORDINARY_12",
                    scope_kind="persistent-adaptation",
                    state_parent=parent,
                )
                session: runner.V2AcquisitionSession | None = None
                stage = "open"
                try:
                    session = await runner.open_v2_acquisition_scope_live(scope)
                    memory = runner._live_v2_lineage_mechanics().memory_factory(
                        source=store,
                        backend=session.backend,
                    )
                    stage = "rebuild"
                    self.assertEqual(
                        len(
                            tuple(
                                await memory.rebuild(
                                    limit=runner.MAXIMUM_PROJECTION_RETRY
                                )
                            )
                        ),
                        15,
                    )
                    self.assertEqual(
                        tuple(
                            await memory.retry_pending_projections(
                                limit=runner.MAXIMUM_PROJECTION_RETRY
                            )
                        ),
                        (),
                    )
                    memory.assert_synchronized(store.acquisition_head())
                    task = evaluator.make_qualification_evaluator().release_phase(
                        "adaptation"
                    )[0]
                    task_text = evaluator.render_public_task(task)

                    stage = "full-search"
                    full_hits = tuple(
                        await session.backend.search(task_text, limit=len(inventory))
                    )
                    self.assertEqual(len(full_hits), 15)
                    self.assertIn(
                        proposed_ref,
                        {item.record_ref for item in full_hits},
                    )
                    self.assertTrue(
                        all(
                            float(left.score) <= float(right.score)
                            for left, right in zip(
                                full_hits,
                                full_hits[1:],
                            )
                        )
                    )
                    raw_batch = memory.recall_from_hits(
                        full_hits,
                        limit=len(inventory),
                        frozen_origin=False,
                    )
                    self.assertEqual(
                        raw_batch.rejected,
                        ("REFERENCE_EPISTEMIC_STATUS_DENIED",),
                    )
                    self.assertEqual(len(raw_batch.items), 14)

                    stage = "eligible-fill"
                    selected_hits, batch = await runner._bounded_ordinary_recall(
                        query=task_text,
                        store=store,
                        backend=session.backend,
                        memory=memory,
                        frozen_origin=False,
                        label="V2 live R4 proposed-filter regression",
                    )
                    self.assertEqual(len(selected_hits), runner.RECALL_SLOTS)
                    self.assertEqual(len(batch.items), runner.RECALL_SLOTS)
                    self.assertEqual(batch.rejected, ())
                    self.assertNotIn(
                        proposed_ref,
                        {item.record_ref for item in batch.items},
                    )
                    self.assertTrue(
                        all(
                            item.epistemic_status in ("OBSERVED", "VALIDATED")
                            for item in batch.items
                        )
                    )
                    stage = "cleanup"
                    await session.forget_close_and_remove()
                    session = None
                    self.assertFalse(Path(scope.state_root).exists())
                    runner._assert_no_unresolved_v2_cognee_workers()
                except BaseException as error:
                    raise AssertionError(
                        "live R5 proposed-filter regression failed "
                        f"at substage {stage}: {type(error).__name__}: {error}"
                    ) from error
                finally:
                    propagating = os.sys.exc_info()[0] is not None
                    cleanup_failures: list[BaseException] = []
                    if session is not None and not session._closed:
                        try:
                            await session.forget_close_and_remove()
                        except BaseException as error:
                            cleanup_failures.append(error)
                            try:
                                await session.close_preserving_state()
                            except BaseException as close_error:
                                cleanup_failures.append(close_error)
                    try:
                        await runner._drain_unresolved_v2_cognee_workers()
                    except BaseException as error:
                        cleanup_failures.append(error)
                    if (
                        Path(scope.state_root).exists()
                        and (session is None or session._closed)
                    ):
                        runner._remove_exact_private_tree(
                            Path(scope.state_root),
                            expected_parent=parent,
                        )
                    if cleanup_failures and not propagating:
                        raise ExceptionGroup(
                            "live R5 proposed-filter regression cleanup failed",
                            cleanup_failures,
                        )
            self.assertEqual(
                hashlib.sha256(
                    runner._stable_regular_bytes(
                        consumed_store,
                        expected_mode=0o600,
                        expected_owner_uid=os.geteuid(),
                        expected_owner_gid=os.getegid(),
                        maximum_bytes=4 * 1024 * 1024,
                        label="post-smoke consumed R4 IO canonical store",
                    )
                ).hexdigest(),
                "eaaf66e6802d6f574fbd3460701bcb98a1955391481140eb098ed9efab39c65c",
            )

        asyncio.run(exercise())

    def test_inherited_worker_attestation_and_direct_child_are_exact(self) -> None:
        receipt = _synthetic_inherited_parent_attestation()
        self.assertEqual(
            runner._validate_v2_inherited_parent_attestation(receipt),
            receipt,
        )
        expected_capabilities = tuple(
            (name, runner._V2_ZERO_CAPABILITY_SET)
            for name in runner._V2_CAPABILITY_STATUS_FIELDS
        )
        child = runner._V2WorkerProcessIdentity(
            pid=41_001,
            ppid=os.getpid(),
            starttime=701,
            executable="/synthetic/cognee-python",
            uids=(1000, 1000, 1000, 1000),
            argv=("/synthetic/cognee-python", "-m", "synthetic.worker"),
            depth=0,
            pidfd=91,
            gids=(1000, 1000, 1000, 1000),
            groups=(),
            no_new_privileges=True,
            capabilities=expected_capabilities,
            network_namespace=str(receipt["network_namespace"]),
        )
        runner._validate_v2_inherited_worker_process_graph((child,), receipt)

        child_mutations = (
            replace(child, ppid=os.getpid() + 1),
            replace(child, uids=(1000, 1000, 1000, 0)),
            replace(child, gids=(1000, 1000, 1000, 0)),
            replace(child, groups=(1000,)),
            replace(child, no_new_privileges=False),
            replace(
                child,
                capabilities=(
                    (runner._V2_CAPABILITY_STATUS_FIELDS[0], "0000000000000001"),
                    *expected_capabilities[1:],
                ),
            ),
            replace(child, network_namespace="net:[41003]"),
            replace(child, depth=1),
        )
        for ordinal, changed in enumerate(child_mutations):
            with (
                self.subTest(child_mutation=ordinal),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "inherited Cognee direct child differs",
                ),
            ):
                runner._validate_v2_inherited_worker_process_graph(
                    (changed,), receipt
                )
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "inherited Cognee direct child differs",
        ):
            runner._validate_v2_inherited_worker_process_graph(
                (child, replace(child, pid=41_002, starttime=702)), receipt
            )

        receipt_mutations: list[dict[str, object]] = []
        for key, changed in (
            ("uids", [1000, 1000, 1000, 0]),
            ("gids", [1000, 1000, 1000, 0]),
            ("groups", [1000]),
            ("no_new_privileges", False),
            ("interfaces", [[1, "lo"], [2, "eth0"]]),
            ("ipv4_route_count", 1),
            ("ipv6_route_count", 0),
            ("loopback_operstate", "up"),
            ("network_namespace", "net:[0]"),
        ):
            mutation = json.loads(json.dumps(receipt))
            mutation[key] = changed
            receipt_mutations.append(mutation)
        capability_mutation = json.loads(json.dumps(receipt))
        capability_mutation["capabilities"]["CapEff"] = "0000000000000001"
        receipt_mutations.append(capability_mutation)
        extra_mutation = json.loads(json.dumps(receipt))
        extra_mutation["unexpected"] = True
        receipt_mutations.append(extra_mutation)
        for ordinal, changed in enumerate(receipt_mutations):
            with (
                self.subTest(parent_mutation=ordinal),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "parent attestation differs",
                ),
            ):
                runner._validate_v2_inherited_parent_attestation(changed)
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "parent attestation differs",
        ):
            runner._validate_v2_inherited_parent_attestation(
                receipt,
                expected_network_namespace="net:[41003]",
            )

    def test_live_scope_start_binds_parent_namespace_before_effect(self) -> None:
        async def exercise() -> None:
            from angler.memory import cognee_acquisition_adapter as adapter_module
            from angler.memory.cognee_subprocess_bindings import (
                CogneeSubprocessBindings,
            )

            events: list[tuple[str, object]] = []
            receipt = _synthetic_inherited_parent_attestation()
            namespace = str(receipt["network_namespace"])
            python = os.sys.executable
            expectation = _synthetic_worker_expectation(
                (python, "-c", _SYNTHETIC_FAILED_START_CODE),
                (python, "-c", _SYNTHETIC_WORKER_CODE),
            )
            binding = types.SimpleNamespace(
                close=mock.AsyncMock(),
                dataset_id="synthetic-dataset-id",
                tenant_id="synthetic-tenant-id",
                dataset_name="synthetic-dataset",
                tenant_name="synthetic-tenant",
                node_set_name="synthetic-records",
            )
            backend = types.SimpleNamespace(
                project=mock.AsyncMock(),
                search=mock.AsyncMock(),
                forget_namespace=mock.AsyncMock(),
            )

            def attest(expected: str | None = None) -> dict[str, object]:
                events.append(("attest", expected))
                if expected not in (None, namespace):
                    raise AssertionError("unexpected inherited namespace")
                return json.loads(json.dumps(receipt))

            async def start(**kwargs: object) -> object:
                events.append(
                    ("start", kwargs.get("inherited_network_namespace"))
                )
                self.assertEqual(
                    kwargs["inherited_network_namespace"], namespace
                )
                self.assertEqual(kwargs["timeout_seconds"], 30.0)
                return binding

            def capture(
                observed_binding: object,
                observed_scope: runner.V2AcquisitionScopeSpec,
                observed_parent: object,
            ) -> runner._V2CogneeWorkerHandle:
                events.append(("capture", observed_parent))
                self.assertIs(observed_binding, binding)
                self.assertEqual(observed_parent, receipt)
                return runner._V2CogneeWorkerHandle(
                    binding=binding,
                    transport=object(),
                    process=object(),
                    scope_spec=observed_scope,
                    pid=41_101,
                    process_graph=(),
                    untrusted_processes=(),
                    graph_expectation=expectation,
                    pipe_identities=((1, 1), (1, 2), (1, 3)),
                    pipe_holder_bindings=(),
                    semantic_graph_captured=True,
                    parent_attestation=receipt,
                )

            with tempfile.TemporaryDirectory() as directory:
                scope = runner.V2AcquisitionScopeSpec.create(
                    purpose="qualification",
                    replicate_commitment=_ref("inherited-start-replicate"),
                    arm="IO_FULL_ORDINARY_12",
                    scope_kind="persistent-adaptation",
                    state_parent=Path(directory).resolve(),
                )
                binding.dataset_name = scope.dataset_name
                binding.tenant_name = scope.tenant_name
                binding.node_set_name = scope.node_set_name
                start_mock = mock.AsyncMock(side_effect=start)
                with (
                    mock.patch.object(
                        runner,
                        "_attest_v2_inherited_worker_parent",
                        side_effect=attest,
                    ),
                    mock.patch.object(
                        runner,
                        "_v2_cognee_graph_expectation",
                        return_value=expectation,
                    ),
                    mock.patch.object(
                        runner,
                        "_assert_no_preexisting_v2_scope_process",
                        side_effect=lambda _value: events.append(
                            ("preexisting", None)
                        ),
                    ),
                    mock.patch.object(
                        runner,
                        "_v2_process_key_snapshot",
                        side_effect=lambda: events.append(("snapshot", None)) or {},
                    ),
                    mock.patch.object(
                        runner.V2AcquisitionScopeSpec,
                        "runtime_scope",
                        return_value=mock.sentinel.synthetic_runtime_scope,
                    ),
                    mock.patch.object(
                        CogneeSubprocessBindings, "start", new=start_mock
                    ),
                    mock.patch.object(
                        runner._V2CogneeWorkerHandle,
                        "capture",
                        side_effect=capture,
                    ),
                    mock.patch.object(
                        adapter_module,
                        "CogneeAcquisitionAdapter",
                        side_effect=lambda **_kwargs: events.append(
                            ("adapter", None)
                        )
                        or backend,
                    ),
                ):
                    session = await runner.open_v2_acquisition_scope_live(
                        scope,
                        expected_dataset_id="synthetic-dataset-id",
                    )
                self.assertIs(session.binding, binding)
                self.assertIs(session.backend, backend)
                self.assertEqual(
                    tuple(name for name, _value in events),
                    (
                        "attest",
                        "preexisting",
                        "snapshot",
                        "attest",
                        "start",
                        "capture",
                        "adapter",
                    ),
                )
                self.assertEqual(events[0], ("attest", None))
                self.assertEqual(events[3], ("attest", namespace))
                self.assertEqual(events[4], ("start", namespace))

        asyncio.run(exercise())

    def test_worker_cleanup_rejoins_parent_before_every_graph_scan(self) -> None:
        receipt = _synthetic_inherited_parent_attestation()
        namespace = str(receipt["network_namespace"])
        python = os.sys.executable
        expectation = _synthetic_worker_expectation(
            (python, "-c", _SYNTHETIC_FAILED_START_CODE),
            (python, "-c", _SYNTHETIC_WORKER_CODE),
        )
        handle = runner._V2CogneeWorkerHandle(
            binding=object(),
            transport=object(),
            process=object(),
            scope_spec=object(),
            pid=41_201,
            process_graph=(),
            untrusted_processes=(),
            graph_expectation=expectation,
            pipe_identities=((1, 1), (1, 2), (1, 3)),
            pipe_holder_bindings=(),
            semantic_graph_captured=False,
            parent_attestation=receipt,
        )

        events: list[str] = []

        def attest(expected: str | None = None) -> dict[str, object]:
            self.assertEqual(expected, namespace)
            events.append("attest")
            return json.loads(json.dumps(receipt))

        def stop_rescan(*_args: object, **_kwargs: object) -> object:
            events.append("rescan")
            raise runner.V2IntegrityStop("synthetic rescan stop")

        def stop_holders(*_args: object, **_kwargs: object) -> object:
            events.append("holders")
            raise runner.V2IntegrityStop("synthetic holder stop")

        with (
            mock.patch.object(
                runner,
                "_attest_v2_inherited_worker_parent",
                side_effect=attest,
            ),
            mock.patch.object(
                runner,
                "_capture_v2_terminal_authority",
                side_effect=stop_rescan,
            ),
        ):
            failures = handle._rescan_merge_before_close()
        self.assertEqual(events, ["attest", "rescan"])
        self.assertEqual(
            tuple(str(item) for item in failures),
            ("V2_COGNEE_PROCESS_RESCAN_FAILED",),
        )

        events.clear()
        with (
            mock.patch.object(
                runner,
                "_attest_v2_inherited_worker_parent",
                side_effect=attest,
            ),
            mock.patch.object(
                runner,
                "_capture_v2_pipe_holder_authority",
                side_effect=stop_holders,
            ),
        ):
            failures = handle._merge_pipe_holders()
        self.assertEqual(events, ["attest", "holders"])
        self.assertEqual(
            tuple(str(item) for item in failures),
            ("V2_COGNEE_PIPE_HOLDER_CAPTURE_FAILED",),
        )

        authority = runner._V2FailedStartAuthority(
            scope_ref=_ref("inherited-failed-start-scope"),
            baseline=frozenset(),
            expectation=expectation,
            parent_attestation=receipt,
        )
        events.clear()

        def empty_delta(*_args: object, **_kwargs: object) -> object:
            events.append("delta")
            return (), ()

        with (
            mock.patch.object(
                runner,
                "_attest_v2_inherited_worker_parent",
                side_effect=attest,
            ),
            mock.patch.object(
                runner,
                "_capture_v2_postlaunch_delta",
                side_effect=empty_delta,
            ),
        ):
            self.assertEqual(authority._refresh(), [])
        self.assertEqual(events, ["attest", "delta"])

    def test_exact_graph_separates_and_never_signals_unknown_pipe_holder(self) -> None:
        async def exercise() -> None:
            python = os.sys.executable
            root_argv = (
                python,
                "-c",
                _SYNTHETIC_WAITING_ROOT_CODE,
                _SYNTHETIC_WORKER_CODE,
            )
            worker_argv = (python, "-c", _SYNTHETIC_WORKER_CODE)
            expectation = _synthetic_worker_expectation(root_argv, worker_argv)
            root: asyncio.subprocess.Process | None = None
            unknown: asyncio.subprocess.Process | None = None
            leaf_cleanup_pidfd: int | None = None
            graph: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            untrusted: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            duplicated_pipe: int | None = None
            try:
                root = await asyncio.create_subprocess_exec(
                    *root_argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.assertIsNotNone(root.stdout)
                raw_leaf_pid = await asyncio.wait_for(
                    root.stdout.readline(), timeout=3.0
                )
                leaf_pid = int(raw_leaf_pid.strip())
                leaf_cleanup_pidfd = os.pidfd_open(leaf_pid, 0)
                pipes = runner._v2_asyncio_pipe_identities(root)
                graph = runner._capture_v2_process_graph(root.pid)
                self.assertEqual(
                    tuple((item.pid, item.depth) for item in graph),
                    ((root.pid, 0), (leaf_pid, 1)),
                )
                runner._validate_v2_process_graph(graph, expectation, pipes)

                stdout_transport = root._transport.get_pipe_transport(1)
                stdout_pipe = stdout_transport.get_extra_info("pipe")
                duplicated_pipe = os.dup(stdout_pipe.fileno())
                unknown = await asyncio.create_subprocess_exec(
                    python,
                    "-c",
                    _SYNTHETIC_UNKNOWN_HOLDER_CODE,
                    pass_fds=(duplicated_pipe,),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                os.close(duplicated_pipe)
                duplicated_pipe = None
                await _wait_for_test_condition(
                    lambda: unknown.pid in runner._v2_pipe_holders(pipes),
                    label="unknown duplicated-pipe holder",
                )
                merged, untrusted, bindings, drift = (
                    runner._capture_v2_pipe_holder_authority(
                        pipes, graph, (), expectation
                    )
                )
                graph = merged
                self.assertTrue(drift)
                self.assertEqual(tuple(item.pid for item in untrusted), (unknown.pid,))
                runner._validate_v2_pipe_holder_bindings(
                    (*graph, *untrusted), bindings, pipes
                )

                leaf = next(item for item in graph if item.pid == leaf_pid)
                unknown_identity = untrusted[0]
                signaled: list[int] = []
                original_signal = signal.pidfd_send_signal

                def record_signal(
                    pidfd: int,
                    signal_number: int,
                    siginfo: object,
                    flags: int,
                ) -> None:
                    signaled.append(pidfd)
                    original_signal(pidfd, signal_number, siginfo, flags)

                with mock.patch.object(
                    runner.signal,
                    "pidfd_send_signal",
                    side_effect=record_signal,
                ):
                    self.assertEqual(
                        runner._signal_v2_processes_leaf_first(
                            graph, signal.SIGTERM
                        ),
                        [],
                    )
                self.assertTrue(signaled)
                self.assertEqual(signaled[0], leaf.pidfd)
                self.assertNotIn(unknown_identity.pidfd, signaled)
                self.assertEqual(await runner._wait_v2_pidfds(graph, 5.0), ())
                self.assertFalse(runner._v2_pidfd_terminal(unknown_identity.pidfd))

                await _terminate_test_process(unknown)
                unknown = None
                await asyncio.wait_for(root.wait(), timeout=3.0)
                self.assertEqual(runner._v2_pipe_holders(pipes), ())
            finally:
                _close_test_fd(duplicated_pipe)
                await _terminate_test_process(unknown)
                _terminate_test_pidfd(leaf_cleanup_pidfd)
                await _terminate_test_process(root)
                _close_test_fd(leaf_cleanup_pidfd)
                runner._close_v2_pidfds((*graph, *untrusted))

        asyncio.run(exercise())

    def test_reparented_exact_worker_is_recovered_and_killed_by_pidfd(self) -> None:
        async def exercise() -> None:
            python = os.sys.executable
            root_argv = (
                python,
                "-c",
                _SYNTHETIC_REPARENTING_ROOT_CODE,
                _SYNTHETIC_WORKER_CODE,
            )
            worker_argv = (python, "-c", _SYNTHETIC_WORKER_CODE)
            expectation = _synthetic_worker_expectation(root_argv, worker_argv)
            root: asyncio.subprocess.Process | None = None
            leaf_cleanup_pidfd: int | None = None
            captured: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            untrusted: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            try:
                root = await asyncio.create_subprocess_exec(
                    *root_argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.assertIsNotNone(root.stdout)
                pipes = runner._v2_asyncio_pipe_identities(root)
                leaf_pid = int(
                    (
                        await asyncio.wait_for(
                            root.stdout.readline(), timeout=3.0
                        )
                    ).strip()
                )
                leaf_cleanup_pidfd = os.pidfd_open(leaf_pid, 0)
                await _wait_for_test_condition(
                    lambda: root.returncode is not None,
                    label="synthetic top-process reap",
                )
                captured, untrusted, bindings, semantic = (
                    runner._capture_v2_terminal_authority(
                        root.pid, pipes, expectation
                    )
                )
                self.assertFalse(semantic)
                self.assertEqual(untrusted, ())
                self.assertEqual(
                    tuple((item.pid, item.depth) for item in captured),
                    ((leaf_pid, 1),),
                )
                self.assertEqual(bindings, ((leaf_pid, captured[0].starttime, pipes),))
                self.assertEqual(
                    runner._signal_v2_processes_leaf_first(
                        captured, signal.SIGTERM
                    ),
                    [],
                )
                self.assertEqual(await runner._wait_v2_pidfds(captured, 5.0), ())
                await asyncio.wait_for(root.wait(), timeout=3.0)
                await _wait_for_test_condition(
                    lambda: runner._v2_pipe_holders(pipes) == (),
                    label="reparented worker pipe release",
                )
            finally:
                _terminate_test_pidfd(leaf_cleanup_pidfd)
                await _terminate_test_process(root)
                _close_test_fd(leaf_cleanup_pidfd)
                runner._close_v2_pidfds((*captured, *untrusted))

        asyncio.run(exercise())

    def test_pipe_holder_toctou_never_yields_signal_authority(self) -> None:
        pipes = ((7, 11),)
        cases = (
            ("changed-starttime", "changed", False),
            ("dropped-pipe", "dropped", False),
            ("fd-scan-error", "error", False),
            ("terminal-pidfd", "stable", True),
        )
        for label, holdings_mode, terminal in cases:
            with self.subTest(case=label):
                read_fd, write_fd = os.pipe()
                first = runner._V2WorkerProcessIdentity(
                    pid=4_242,
                    ppid=1,
                    starttime=101,
                    executable="/test/worker",
                    uids=(1_000,) * 4,
                    argv=("test-worker",),
                    depth=1,
                    pidfd=read_fd,
                )
                third = (
                    replace(first, starttime=102)
                    if holdings_mode == "changed"
                    else first
                )
                holdings = (
                    mock.Mock(side_effect=OSError("synthetic fd scan failure"))
                    if holdings_mode == "error"
                    else mock.Mock(
                        return_value=() if holdings_mode == "dropped" else pipes
                    )
                )
                try:
                    with (
                        mock.patch.object(
                            runner.os, "pidfd_open", return_value=read_fd
                        ),
                        mock.patch.object(
                            runner,
                            "_read_v2_process_identity",
                            side_effect=(first, first, third),
                        ),
                        mock.patch.object(
                            runner,
                            "_v2_process_pipe_holdings",
                            holdings,
                        ),
                        mock.patch.object(
                            runner,
                            "_v2_pidfd_terminal",
                            return_value=terminal,
                        ),
                        mock.patch.object(
                            runner.signal, "pidfd_send_signal"
                        ) as send_signal,
                    ):
                        self.assertIsNone(
                            runner._capture_v2_stable_pipe_holder(
                                first.pid,
                                depth=1,
                                expected_held=pipes,
                                pipe_identities=pipes,
                            )
                        )
                        send_signal.assert_not_called()
                    with self.assertRaises(OSError):
                        os.fstat(read_fd)
                finally:
                    _close_test_fd(read_fd)
                    _close_test_fd(write_fd)

        with self.subTest(case="post-baseline-role-only-process-is-untrusted"):
            python = os.sys.executable
            expectation = _synthetic_worker_expectation(
                (python, "-c", _SYNTHETIC_FAILED_START_CODE),
                (python, "-c", _SYNTHETIC_WORKER_CODE),
            )
            root_pid = 31_001
            child_pid = 31_002
            concurrent_pid = 32_000
            process_rows = {
                root_pid: (os.getpid(), 201, expectation.root_argv, 0),
                child_pid: (root_pid, 202, expectation.worker_argv, 1),
                concurrent_pid: (777, 203, expectation.worker_argv, 1),
            }
            current = {
                (pid, starttime): ppid
                for pid, (ppid, starttime, _argv, _depth) in process_rows.items()
            }
            descriptors: dict[int, int] = {}
            write_descriptors: list[int] = []
            authorized: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            untrusted: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            try:
                for pid in process_rows:
                    read_fd, write_fd = os.pipe()
                    descriptors[pid] = read_fd
                    write_descriptors.append(write_fd)

                def read_identity(
                    pid: int,
                    *,
                    depth: int,
                    pidfd: int,
                ) -> runner._V2WorkerProcessIdentity:
                    ppid, starttime, argv, _expected_depth = process_rows[pid]
                    return runner._V2WorkerProcessIdentity(
                        pid=pid,
                        ppid=ppid,
                        starttime=starttime,
                        executable=expectation.root_executable,
                        uids=expectation.root_uids,
                        argv=argv,
                        depth=depth,
                        pidfd=pidfd,
                    )

                def cmdline(pid: int) -> tuple[str, ...] | None:
                    row = process_rows.get(pid)
                    return None if row is None else row[2]

                with (
                    mock.patch.object(
                        runner, "_v2_process_key_snapshot", return_value=current
                    ),
                    mock.patch.object(
                        runner, "_v2_filtered_cmdline", side_effect=cmdline
                    ),
                    mock.patch.object(
                        runner.os,
                        "pidfd_open",
                        side_effect=lambda pid, _flags: descriptors[pid],
                    ),
                    mock.patch.object(
                        runner,
                        "_read_v2_process_identity",
                        side_effect=read_identity,
                    ),
                    mock.patch.object(
                        runner, "_v2_pidfd_terminal", return_value=False
                    ),
                    mock.patch.object(
                        runner.signal, "pidfd_send_signal"
                    ) as send_signal,
                ):
                    authorized, untrusted = runner._capture_v2_postlaunch_delta(
                        frozenset(), expectation
                    )
                    self.assertEqual(
                        tuple(item.pid for item in authorized),
                        (root_pid, child_pid),
                    )
                    self.assertEqual(
                        tuple(item.pid for item in untrusted),
                        (concurrent_pid,),
                    )
                    self.assertEqual(
                        runner._signal_v2_processes_leaf_first(
                            authorized, signal.SIGTERM
                        ),
                        [],
                    )
                    self.assertEqual(
                        tuple(call.args[0] for call in send_signal.call_args_list),
                        (descriptors[child_pid], descriptors[root_pid]),
                    )
                    self.assertNotIn(
                        descriptors[concurrent_pid],
                        tuple(call.args[0] for call in send_signal.call_args_list),
                    )
            finally:
                runner._close_v2_pidfds((*authorized, *untrusted))
                for descriptor in descriptors.values():
                    _close_test_fd(descriptor)
                for descriptor in write_descriptors:
                    _close_test_fd(descriptor)

        ancestry_cases = (
            ("reused-nondirect-root", os.getpid() + 10_000, 33_001),
            ("child-ancestry-drift", os.getpid(), 99_999),
        )
        for label, root_ppid, child_ppid in ancestry_cases:
            with self.subTest(case=label):
                root_pid = 33_001
                child_pid = 33_002
                read_fds: dict[int, int] = {}
                write_fds: list[int] = []
                try:
                    for pid in (root_pid, child_pid):
                        read_fd, write_fd = os.pipe()
                        read_fds[pid] = read_fd
                        write_fds.append(write_fd)
                    identities = {
                        root_pid: runner._V2WorkerProcessIdentity(
                            pid=root_pid,
                            ppid=root_ppid,
                            starttime=301,
                            executable="/synthetic/root",
                            uids=(1_000,) * 4,
                            argv=("synthetic-root",),
                            depth=0,
                            pidfd=read_fds[root_pid],
                        ),
                        child_pid: runner._V2WorkerProcessIdentity(
                            pid=child_pid,
                            ppid=child_ppid,
                            starttime=302,
                            executable="/synthetic/worker",
                            uids=(1_000,) * 4,
                            argv=("synthetic-worker",),
                            depth=1,
                            pidfd=read_fds[child_pid],
                        ),
                    }

                    def read_ancestry_identity(
                        pid: int,
                        *,
                        depth: int,
                        pidfd: int,
                    ) -> runner._V2WorkerProcessIdentity:
                        return replace(identities[pid], depth=depth, pidfd=pidfd)

                    with (
                        mock.patch.object(
                            runner,
                            "_v2_descendant_depths",
                            return_value={root_pid: 0, child_pid: 1},
                        ),
                        mock.patch.object(
                            runner.os,
                            "pidfd_open",
                            side_effect=lambda pid, _flags: read_fds[pid],
                        ),
                        mock.patch.object(
                            runner,
                            "_read_v2_process_identity",
                            side_effect=read_ancestry_identity,
                        ),
                        mock.patch.object(
                            runner, "_v2_pidfd_terminal", return_value=False
                        ),
                        mock.patch.object(
                            runner.signal, "pidfd_send_signal"
                        ) as send_signal,
                        self.assertRaisesRegex(
                            runner.V2IntegrityStop,
                            "captured ancestry",
                        ),
                    ):
                        runner._capture_v2_process_graph(root_pid)
                    send_signal.assert_not_called()
                    for descriptor in read_fds.values():
                        with self.assertRaises(OSError):
                            os.fstat(descriptor)
                finally:
                    for descriptor in read_fds.values():
                        _close_test_fd(descriptor)
                    for descriptor in write_fds:
                        _close_test_fd(descriptor)

        with self.subTest(case="fallback-reused-root-is-untrusted"):
            python = os.sys.executable
            expectation = _synthetic_worker_expectation(
                (python, "-c", _SYNTHETIC_FAILED_START_CODE),
                (python, "-c", _SYNTHETIC_WORKER_CODE),
            )
            read_fd, write_fd = os.pipe()
            reused_root = runner._V2WorkerProcessIdentity(
                pid=34_001,
                ppid=1,
                starttime=401,
                executable=expectation.root_executable,
                uids=expectation.root_uids,
                argv=expectation.root_argv,
                depth=0,
                pidfd=read_fd,
            )
            captured: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            untrusted: tuple[runner._V2WorkerProcessIdentity, ...] = ()
            try:
                with (
                    mock.patch.object(
                        runner,
                        "_capture_v2_process_graph",
                        side_effect=runner.V2IntegrityStop("synthetic graph miss"),
                    ),
                    mock.patch.object(
                        runner, "_v2_pipe_holder_inventory", return_value={}
                    ),
                    mock.patch.object(
                        runner.os, "pidfd_open", return_value=read_fd
                    ),
                    mock.patch.object(
                        runner,
                        "_read_v2_process_identity",
                        side_effect=(reused_root, reused_root),
                    ),
                    mock.patch.object(
                        runner, "_v2_pidfd_terminal", return_value=False
                    ),
                    mock.patch.object(
                        runner.signal, "pidfd_send_signal"
                    ) as send_signal,
                ):
                    captured, untrusted, bindings, semantic = (
                        runner._capture_v2_terminal_authority(
                            reused_root.pid, ((7, 11),), expectation
                        )
                    )
                    self.assertEqual(captured, ())
                    self.assertEqual(untrusted, (reused_root,))
                    self.assertEqual(bindings, ())
                    self.assertFalse(semantic)
                    self.assertEqual(
                        runner._signal_v2_processes_leaf_first(
                            captured, signal.SIGTERM
                        ),
                        [],
                    )
                    send_signal.assert_not_called()
            finally:
                runner._close_v2_pidfds((*captured, *untrusted))
                _close_test_fd(read_fd)
                _close_test_fd(write_fd)

        with self.subTest(case="rescan-root-reuse-demotes-new-authority"):
            python = os.sys.executable
            expectation = _synthetic_worker_expectation(
                (python, "-c", _SYNTHETIC_FAILED_START_CODE),
                (python, "-c", _SYNTHETIC_WORKER_CODE),
            )
            descriptors: list[int] = []
            write_descriptors: list[int] = []
            try:
                for _index in range(4):
                    read_fd, write_fd = os.pipe()
                    descriptors.append(read_fd)
                    write_descriptors.append(write_fd)
                retained_root = runner._V2WorkerProcessIdentity(
                    pid=35_001,
                    ppid=os.getpid(),
                    starttime=501,
                    executable=expectation.root_executable,
                    uids=expectation.root_uids,
                    argv=expectation.root_argv,
                    depth=0,
                    pidfd=descriptors[0],
                )
                retained_child = runner._V2WorkerProcessIdentity(
                    pid=35_002,
                    ppid=retained_root.pid,
                    starttime=502,
                    executable=expectation.worker_executable,
                    uids=expectation.worker_uids,
                    argv=expectation.worker_argv,
                    depth=1,
                    pidfd=descriptors[1],
                )
                reused_root = replace(
                    retained_root,
                    starttime=601,
                    pidfd=descriptors[2],
                )
                replacement_child = replace(
                    retained_child,
                    pid=35_003,
                    ppid=reused_root.pid,
                    starttime=602,
                    pidfd=descriptors[3],
                )
                handle = runner._V2CogneeWorkerHandle(
                    binding=object(),
                    transport=object(),
                    process=object(),
                    scope_spec=object(),
                    pid=retained_root.pid,
                    process_graph=(retained_root, retained_child),
                    untrusted_processes=(),
                    graph_expectation=expectation,
                    pipe_identities=((1, 1), (1, 2), (1, 3)),
                    pipe_holder_bindings=(),
                    semantic_graph_captured=True,
                )
                with (
                    mock.patch.object(
                        runner,
                        "_capture_v2_terminal_authority",
                        return_value=(
                            (reused_root, replacement_child),
                            (),
                            (),
                            True,
                        ),
                    ),
                    mock.patch.object(
                        runner, "_v2_pidfd_terminal", return_value=False
                    ),
                    mock.patch.object(
                        runner.signal, "pidfd_send_signal"
                    ) as send_signal,
                ):
                    failures = handle._rescan_merge_before_close()
                    self.assertEqual(
                        tuple(str(item) for item in failures),
                        (
                            "V2_COGNEE_ROOT_PID_REUSED",
                            "V2_COGNEE_PROCESS_GRAPH_DRIFT",
                        ),
                    )
                    self.assertEqual(
                        handle.process_graph,
                        (retained_root, retained_child),
                    )
                    self.assertEqual(
                        handle.untrusted_processes,
                        (reused_root, replacement_child),
                    )
                    self.assertEqual(
                        runner._signal_v2_processes_leaf_first(
                            handle.process_graph, signal.SIGTERM
                        ),
                        [],
                    )
                    signaled = tuple(
                        call.args[0] for call in send_signal.call_args_list
                    )
                    self.assertEqual(
                        signaled,
                        (retained_child.pidfd, retained_root.pidfd),
                    )
                    self.assertNotIn(reused_root.pidfd, signaled)
                    self.assertNotIn(replacement_child.pidfd, signaled)
            finally:
                if "handle" in locals():
                    runner._close_v2_pidfds(
                        (*handle.process_graph, *handle.untrusted_processes)
                    )
                for descriptor in descriptors:
                    _close_test_fd(descriptor)
                for descriptor in write_descriptors:
                    _close_test_fd(descriptor)

    def test_failed_start_before_binding_return_cleans_owned_delta(self) -> None:
        async def exercise() -> None:
            from angler.memory.cognee_subprocess_bindings import (
                CogneeSubprocessBindings,
            )

            python = os.sys.executable
            root_argv = (python, "-c", _SYNTHETIC_FAILED_START_CODE)
            worker_argv = (python, "-c", _SYNTHETIC_WORKER_CODE)
            expectation = _synthetic_worker_expectation(root_argv, worker_argv)
            started: list[asyncio.subprocess.Process] = []

            async def fail_after_start(
                _owner: object | None = None,
                **kwargs: object,
            ) -> object:
                self.assertEqual(
                    kwargs["inherited_network_namespace"], "net:[41002]"
                )
                process = await asyncio.create_subprocess_exec(
                    *root_argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                started.append(process)
                await asyncio.sleep(0)
                raise RuntimeError("synthetic failure before binding return")

            self.assertEqual(runner._UNRESOLVED_V2_COGNEE_STARTS, {})
            with tempfile.TemporaryDirectory() as directory:
                scope = runner.V2AcquisitionScopeSpec.create(
                    purpose="qualification",
                    replicate_commitment=_ref("failed-start-replicate"),
                    arm="IO_FULL_ORDINARY_12",
                    scope_kind="persistent-adaptation",
                    state_parent=Path(directory).resolve(),
                )
                try:
                    with (
                        mock.patch.object(
                            runner,
                            "_attest_v2_inherited_worker_parent",
                            side_effect=lambda expected=None: (
                                _synthetic_inherited_parent_attestation(
                                    "net:[41002]"
                                )
                                if expected in (None, "net:[41002]")
                                else self.fail(
                                    "unexpected inherited namespace"
                                )
                            ),
                        ),
                        mock.patch.object(
                            runner,
                            "_v2_cognee_graph_expectation",
                            return_value=expectation,
                        ),
                        mock.patch.object(
                            runner.V2AcquisitionScopeSpec,
                            "runtime_scope",
                            return_value=mock.sentinel.synthetic_runtime_scope,
                        ),
                        mock.patch.object(
                            CogneeSubprocessBindings,
                            "start",
                            new=fail_after_start,
                        ),
                        self.assertRaisesRegex(
                            runner.V2IntegrityStop,
                            "V2_COGNEE_BINDING_START_FAILED",
                        ),
                    ):
                        await runner.open_v2_acquisition_scope_live(scope)
                    self.assertEqual(len(started), 1)
                    await asyncio.wait_for(started[0].wait(), timeout=3.0)
                    self.assertIsNotNone(started[0].returncode)
                    self.assertEqual(runner._UNRESOLVED_V2_COGNEE_STARTS, {})
                    runner._assert_no_unresolved_v2_cognee_workers()
                finally:
                    for process in started:
                        await _terminate_test_process(process)
                    runner._UNRESOLVED_V2_COGNEE_STARTS.pop(
                        scope.scope_ref, None
                    )

        asyncio.run(exercise())

    def test_unresolved_registries_drain_or_remain_fail_closed(self) -> None:
        async def exercise() -> None:
            registries = (
                runner._UNRESOLVED_V2_COGNEE_HANDLES,
                runner._UNRESOLVED_V2_COGNEE_BINDINGS,
                runner._UNRESOLVED_V2_COGNEE_STARTS,
            )
            self.assertTrue(all(not value for value in registries))
            python = os.sys.executable
            expectation = _synthetic_worker_expectation(
                (python, "-c", _SYNTHETIC_FAILED_START_CODE),
                (python, "-c", _SYNTHETIC_WORKER_CODE),
            )
            with tempfile.TemporaryDirectory() as directory:
                scope = runner.V2AcquisitionScopeSpec.create(
                    purpose="qualification",
                    replicate_commitment=_ref("retained-worker-replicate"),
                    arm="IO_FULL_ORDINARY_12",
                    scope_kind="persistent-adaptation",
                    state_parent=Path(directory).resolve(),
                )
                binding = object()
                opaque_process = object()
                handle = runner._V2CogneeWorkerHandle(
                    binding=object(),
                    transport=object(),
                    process=opaque_process,
                    scope_spec=scope,
                    pid=7_001,
                    process_graph=(),
                    untrusted_processes=(),
                    graph_expectation=expectation,
                    pipe_identities=((1, 1), (1, 2), (1, 3)),
                    pipe_holder_bindings=(),
                    semantic_graph_captured=False,
                )
                record = runner._V2UnresolvedCogneeBinding(
                    binding=binding,
                    process=opaque_process,
                    pipe_identities=None,
                )
                authority = runner._V2FailedStartAuthority(
                    scope_ref=scope.scope_ref,
                    baseline=frozenset(),
                    expectation=expectation,
                )

                async def close_handle(
                    value: runner._V2CogneeWorkerHandle,
                    *,
                    require_scope: bool = True,
                ) -> None:
                    self.assertFalse(require_scope)
                    value.terminal = True

                async def close_binding(
                    value: object,
                    *,
                    worker_handle: object | None = None,
                ) -> None:
                    self.assertIsNone(worker_handle)
                    runner._UNRESOLVED_V2_COGNEE_BINDINGS.pop(-id(value))

                async def close_start(
                    value: runner._V2FailedStartAuthority,
                ) -> None:
                    value.terminal = True
                    runner._UNRESOLVED_V2_COGNEE_STARTS.pop(value.scope_ref)

                try:
                    runner._UNRESOLVED_V2_COGNEE_HANDLES[handle.pid] = handle
                    runner._UNRESOLVED_V2_COGNEE_BINDINGS[-id(binding)] = record
                    runner._UNRESOLVED_V2_COGNEE_STARTS[scope.scope_ref] = authority
                    with (
                        mock.patch.object(
                            runner._V2CogneeWorkerHandle,
                            "close_terminal",
                            new=close_handle,
                        ),
                        mock.patch.object(
                            runner,
                            "_close_v2_acquisition_binding_preserving_state",
                            new=close_binding,
                        ),
                        mock.patch.object(
                            runner._V2FailedStartAuthority,
                            "close_terminal",
                            new=close_start,
                        ),
                    ):
                        await runner._drain_unresolved_v2_cognee_workers()
                    self.assertTrue(all(not value for value in registries))
                    runner._assert_no_unresolved_v2_cognee_workers()

                    handle.terminal = False
                    runner._UNRESOLVED_V2_COGNEE_HANDLES[handle.pid] = handle

                    async def fail_close(
                        value: runner._V2CogneeWorkerHandle,
                        *,
                        require_scope: bool = True,
                    ) -> None:
                        self.assertIs(value, handle)
                        self.assertFalse(require_scope)
                        raise runner.V2IntegrityStop("synthetic retained close")

                    with (
                        mock.patch.object(
                            runner._V2CogneeWorkerHandle,
                            "close_terminal",
                            new=fail_close,
                        ),
                        self.assertRaises(BaseException),
                    ):
                        await runner._drain_unresolved_v2_cognee_workers()
                    self.assertIs(
                        runner._UNRESOLVED_V2_COGNEE_HANDLES[handle.pid],
                        handle,
                    )
                    with self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "V2_COGNEE_RETAINED_WORKER_REMAINS",
                    ):
                        runner._assert_no_unresolved_v2_cognee_workers()
                finally:
                    for value in registries:
                        value.clear()

        asyncio.run(exercise())


class FullInjectedOrchestrationEvidenceTraversalTests(unittest.TestCase):
    @staticmethod
    def _close_fake_foundation(owner: runner.V2FoundationOwner) -> None:
        owner.guard = None
        owner.loaded = None
        owner._closed = True

    def _failed_preparation_fixture(
        self,
    ) -> tuple[
        _FullFakeProductHarness,
        runner.ScheduledAttempt,
        runner.V2LineageOwner,
    ]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness = asyncio.run(
                _FullFakeProductHarness.create(
                    purpose="qualification",
                    root=state_root,
                )
            )
        harness.product.mechanics = _owner_mechanics(
            harness.events,
            ledger=harness.ledger,
            fail_rebuild_arms=frozenset({"IO_FULL_ORDINARY_12"}),
        )
        scheduled = next(
            item
            for item in runner.adaptation_schedule(
                evaluator.make_qualification_evaluator().release_phase(
                    "adaptation"
                ),
                purpose="qualification",
            )
            if item.arm == "IO_FULL_ORDINARY_12"
        )
        source = harness.lineages.owner_for(
            scheduled.replicate_commitment,
            "IO_FULL_ORDINARY_12",
        )
        return harness, scheduled, source

    async def _frozen_disposable_fixture(
        self,
        state_root: Path,
    ) -> tuple[
        _FullFakeProductHarness,
        runner.FrozenAdaptationState,
        evaluator.PublicEvaluationTask,
        runner.TaskPreparationContext,
    ]:
        harness = await _FullFakeProductHarness.create(
            purpose="qualification",
            root=state_root,
        )
        commitment = runner.qualification_replicate_commitment()
        seals = tuple(owner.persist_and_seal() for owner in harness.lineages.owners)
        catalog, _, _ = _catalog_bundle(
            replicate="qualification-01",
            replicate_commitment=commitment,
            purpose="qualification",
        )
        frozen = runner.FrozenAdaptationState(
            purpose="qualification",
            baseline_seals=seals,
            donor_catalogs=(catalog,),
        )
        harness.product._frozen = frozen
        suite = evaluator.make_qualification_evaluator()
        raw_task = next(
            binding.public
            for binding in suite._bindings.values()
            if binding.public.phase == "development"
        )
        public_ref = evaluator._public_task_ref(
            raw_task.replicate,
            commitment,
            raw_task.family,
            raw_task.phase,
            raw_task.ordinal,
            raw_task.payload_json,
        )
        task = evaluator.PublicEvaluationTask(
            task_id=evaluator._task_id(
                raw_task.replicate,
                raw_task.family,
                raw_task.phase,
                raw_task.ordinal,
                public_ref,
            ),
            replicate=raw_task.replicate,
            replicate_commitment=commitment,
            family=raw_task.family,
            phase=raw_task.phase,
            ordinal=raw_task.ordinal,
            payload_json=raw_task.payload_json,
            public_commitment=public_ref,
        )
        context = runner.TaskPreparationContext.create(task, frozen)
        return harness, frozen, task, context

    @staticmethod
    def _exact_fake_baseline_clone(
        source: _OwnerPaths,
        destination: _OwnerPaths,
        _expected_hashes: dict[str, str],
    ) -> None:
        destination.root.mkdir(mode=0o700)
        destination.root.chmod(0o700)
        for name in ("store", "journal", "learner"):
            target = getattr(destination, name)
            target.write_bytes(getattr(source, name).read_bytes())
            target.chmod(0o600)

    @staticmethod
    async def _install_failed_poststage_owner(
        harness: _FullFakeProductHarness,
        scheduled: runner.ScheduledAttempt,
        *,
        staged: bool = True,
    ) -> runner.V2StatefulAttemptOwner:
        source = harness.lineages.owner_for(
            scheduled.replicate_commitment,
            scheduled.arm,
        )
        session = await runner._open_v2_acquisition_session(
            source.scope_spec,
            opener=harness.product.acquisition_opener,
        )
        runtime = object.__new__(runner.V2AttemptExecutionRuntime)
        runtime.arm = scheduled.arm
        runtime.proposal_adapter = object()
        runtime.relation_adapter = object()
        runtime.executor = object()
        runtime.journal = source.journal
        runtime.guard = harness.foundation.guard
        runtime.cycle = object()
        runtime.learner = source.learner
        runtime.store = source.store
        runtime.memory = object()
        runtime.session = session
        runtime.runtime_root = None
        runtime.allocation_ref = None
        runtime.allocation = None
        runtime.path_registry = None
        runtime._turn = None
        runtime._completed = None
        runtime._last_observation_ref = None
        runtime._journal_identity = None
        owner = object.__new__(runner.V2StatefulAttemptOwner)
        owner.runtime = runtime
        owner.source_owner = source
        owner.ledger = harness.ledger
        owner.quiesce_runtime = lambda **_kwargs: None
        owner.source_seal = None
        owner.clone_paths = None
        owner.clone_before = ()
        owner.allocation = None
        owner.path_registry = None
        owner._terminal = None
        owner._closed = False
        harness.product._adaptation_owners[scheduled.attempt_key] = owner
        if staged:
            stage_ref = _ref(f"synthetic-partial-stage-{scheduled.task_id}")
            harness.product._active_by_stage[stage_ref] = owner
            harness.product._active_cells[scheduled.attempt_key] = stage_ref
        return owner

    def test_r5_shaped_clone_open_failure_is_exactly_retained_and_closable(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness, frozen, task, context = asyncio.run(
                self._frozen_disposable_fixture(state_root)
            )
            helpers = dict(runner._load_approved_r2_low_level_helpers())
            helpers["clone_closed_baseline"] = self._exact_fake_baseline_clone
            harness.product.mechanics = _owner_mechanics(
                harness.events,
                ledger=harness.ledger,
                fail_rebuild_arms=frozenset({"IN_FULL_NEUTRAL_12"}),
            )
            with (
                mock.patch.object(
                    runner,
                    "_load_approved_r2_low_level_helpers",
                    return_value=helpers,
                ),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "V2_DISPOSABLE_CLONE_OPEN_FAILED",
                ),
            ):
                asyncio.run(
                    harness.product._capture_recall(
                        task=task.to_canonical(),
                        context=context,
                        kind="neutral",
                    )
                )

            self.assertFalse(harness.product._capture_owners)
            self.assertFalse(harness.product._pending_disposable_retentions)
            self.assertEqual(len(harness.path_registry._allocations), 1)
            allocation = next(iter(harness.path_registry._allocations.values()))
            evidence = harness.product._retained_disposable_allocations[
                allocation.allocation_ref
            ]
            self.assertEqual(
                (allocation.role, allocation.task_id),
                ("recall-neutral", task.task_id),
            )
            self.assertEqual(
                evidence.source_baseline_seal_ref,
                context.in_baseline.seal_ref,
            )
            self.assertEqual(
                evidence.failure_boundary_code,
                "V2_DISPOSABLE_CLONE_OPEN_FAILED",
            )
            self.assertIsNone(evidence.stage_ref)
            self.assertIsNone(evidence.dataset_id)
            self.assertTrue(evidence.worker_terminal)
            self.assertEqual(evidence.absent_targets, ())
            self.assertEqual(
                tuple(item.target_kind for item in evidence.extant_targets),
                ("runtime-root", "cognee-scope-root"),
            )
            self.assertEqual(
                tuple(item.target_path for item in evidence.extant_targets),
                allocation.targets,
            )
            self.assertTrue(all(Path(item).is_dir() for item in allocation.targets))
            harness.ledger.require_artifact(
                evidence.evidence_ref,
                "retained-disposable-allocation",
                evidence.to_canonical(),
            )
            harness.path_registry.assert_failure_accounted()
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "live targets",
            ):
                harness.path_registry.assert_closed()
            runner._assert_no_unresolved_v2_cognee_workers()
            self.assertIn("close:IN_FULL_NEUTRAL_12", harness.events)
            opened_at = max(
                index
                for index, event in enumerate(harness.events)
                if event == "open:IN_FULL_NEUTRAL_12"
            )
            self.assertNotIn(
                "forget:IN_FULL_NEUTRAL_12",
                harness.events[opened_at:],
            )
            for owner, expected in zip(
                harness.lineages.owners,
                frozen.baseline_seals,
                strict=True,
            ):
                self.assertEqual(owner.verify_seal().seal_ref, expected.seal_ref)
            kinds = {
                row[0]
                for row in harness.ledger._connection.execute(
                    "SELECT kind FROM artifacts"
                )
            }
            self.assertFalse(
                {
                    "recall-capture-terminal",
                    "recall-capture-cleanup",
                    "clone-lifecycle",
                    "attempt-stage",
                    "attempt-final",
                }.intersection(kinds)
            )

            with mock.patch.object(
                runner.V2FoundationOwner,
                "close",
                new=self._close_fake_foundation,
            ):
                asyncio.run(harness.product.close())
            self.assertTrue(harness.product._closed)
            self.assertTrue(harness.lineages._closed)
            self.assertTrue(harness.foundation._closed)
            harness.path_registry.assert_failure_accounted()
            self.assertTrue(all(Path(item).is_dir() for item in allocation.targets))
            self.assertEqual(
                runner._terminal_scope_tree_snapshot(allocation.runtime_root),
                evidence.extant_targets[0].tree,
            )
            self.assertEqual(
                runner._terminal_scope_tree_snapshot(
                    allocation.cognee_scope_roots[0]
                ),
                evidence.extant_targets[1].tree,
            )
            harness.ledger.close()

    def test_stateless_development_before_freeze_has_zero_allocation(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness = asyncio.run(
                _FullFakeProductHarness.create(
                    purpose="qualification",
                    root=state_root,
                )
            )
            self.assertIsNone(harness.product._frozen)
            task = _qualification_development_tasks()[0]
            arm = "N0_NATIVE_NO_HISTORY"
            prompt_plan = runner.native_prompt_plan(
                task_id=task.task_id,
                task_text="One bounded public synthetic qualification task.",
                tokenize_chat_prompt=lambda prompt: tuple(
                    range(1, len(prompt.split()) + 1)
                ),
            )
            scheduled = runner.ScheduledAttempt(
                purpose="qualification",
                phase="development",
                arm=arm,
                task=task.to_canonical(),
                block_ordinal=0,
                arm_position=runner.ARMS.index(arm),
            )
            stage = runner.AttemptStage.from_schedule(scheduled, prompt_plan)
            context = runner.ArmExecutionContext(
                task_id=task.task_id,
                arm=arm,
                preparation_ref=_ref("prefreeze-stateless-preparation"),
            )
            permit = runner.AttemptGenerationPermit(
                runner.AttemptBudget(runner.RunMode.QUALIFICATION),
                task.task_id,
                arm,
            )
            before_runtime = tuple(
                sorted(
                    (
                        item.name,
                        item.stat(follow_symlinks=False).st_dev,
                        item.stat(follow_symlinks=False).st_ino,
                    )
                    for item in os.scandir(harness.arm_runtime_parent)
                )
            )
            before_scope = tuple(
                sorted(item.name for item in os.scandir(harness.cognee_scope_parent))
            )
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "V2 product attempt invocation differs",
            ):
                asyncio.run(
                    harness.product.execute_attempt(
                        scheduled,
                        stage,
                        prompt_plan,
                        permit,
                        harness.resources,
                        context,
                    )
                )
            self.assertFalse(harness.path_registry._allocations)
            self.assertFalse(harness.path_registry._targets)
            self.assertFalse(harness.product._active_by_stage)
            self.assertFalse(harness.product._active_cells)
            self.assertEqual(
                harness.ledger._connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE kind='path-allocation'"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                tuple(
                    sorted(
                        (
                            item.name,
                            item.stat(follow_symlinks=False).st_dev,
                            item.stat(follow_symlinks=False).st_ino,
                        )
                        for item in os.scandir(harness.arm_runtime_parent)
                    )
                ),
                before_runtime,
            )
            self.assertEqual(
                tuple(
                    sorted(
                        item.name
                        for item in os.scandir(harness.cognee_scope_parent)
                    )
                ),
                before_scope,
            )

            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "V2 stateless attempt requires frozen adaptation",
            ):
                asyncio.run(
                    harness.product._open_stateless_attempt(
                        scheduled=scheduled,
                        stage_ref=stage.stage_ref,
                    )
                )
            self.assertFalse(harness.path_registry._allocations)
            expected_root = harness.arm_runtime_parent / (
                harness.product._allocation_slug(
                    task_id=task.task_id,
                    replicate_commitment=task.replicate_commitment,
                    role=arm,
                )
            )
            self.assertFalse(expected_root.exists())
            with mock.patch.object(
                runner.V2FoundationOwner,
                "close",
                new=self._close_fake_foundation,
            ):
                asyncio.run(harness.product.close())
            harness.ledger.close()

    def test_unfrozen_extra_runtime_sibling_fails_then_exact_retry_succeeds(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness = asyncio.run(
                _FullFakeProductHarness.create(
                    purpose="qualification",
                    root=state_root,
                )
            )
            extra = harness.arm_runtime_parent / "unregistered-runtime"
            extra.mkdir(mode=0o700)
            extra.chmod(0o700)
            with (
                mock.patch.object(
                    runner.V2FoundationOwner,
                    "close",
                    new=self._close_fake_foundation,
                ),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "V2_FAILED_LINEAGE_OWNER_CLOSE_FAILED",
                ),
            ):
                asyncio.run(harness.product.close())
            self.assertFalse(harness.product._closed)
            self.assertFalse(harness.lineages._closed)
            self.assertTrue(extra.is_dir())
            self.assertTrue(
                all(item.paths is not None for item in harness.lineages.owners)
            )

            extra.rmdir()
            with mock.patch.object(
                runner.V2FoundationOwner,
                "close",
                new=self._close_fake_foundation,
            ):
                asyncio.run(harness.product.close())
            self.assertTrue(harness.product._closed)
            self.assertTrue(harness.lineages._closed)
            harness.ledger.close()

    def test_partial_retained_target_tamper_and_extra_inventory_fail_closed(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness, _frozen, task, context = asyncio.run(
                self._frozen_disposable_fixture(state_root)
            )
            source = harness.lineages.owner_for(
                context.replicate_commitment,
                "IN_FULL_NEUTRAL_12",
            )
            allocation, scope_spec = harness.product._register_disposable(
                task_id=task.task_id,
                replicate_commitment=context.replicate_commitment,
                role="recall-neutral",
                source_arm="IN_FULL_NEUTRAL_12",
                scope_kind="neutral-recall-capture",
            )
            assert scope_spec is not None
            scope = Path(scope_spec.state_root)
            scope.mkdir(mode=0o700)
            scope.chmod(0o700)
            marker = scope / "failed-before-runtime"
            marker.write_bytes(b"bounded partial disposable state")
            marker.chmod(0o600)
            pending = harness.product._queue_failed_disposable_allocation(
                allocation=allocation,
                source_owner=source,
                source_seal=context.in_baseline,
                failure_boundary_code="V2_DISPOSABLE_CLONE_OPEN_FAILED",
                stage_ref=None,
                session=None,
            )
            evidence = asyncio.run(
                harness.product._retain_failed_disposable_allocation(pending)
            )
            self.assertIsInstance(
                evidence,
                runner.RetainedDisposableAllocationEvidence,
            )
            assert evidence is not None
            self.assertEqual(evidence.absent_targets, (allocation.runtime_root,))
            self.assertEqual(len(evidence.extant_targets), 1)
            self.assertEqual(
                evidence.extant_targets[0].target_kind,
                "cognee-scope-root",
            )
            self.assertEqual(evidence.extant_targets[0].target_path, str(scope))
            harness.path_registry.assert_failure_accounted()

            late = scope / "late-child"
            late.write_bytes(b"unregistered late state")
            late.chmod(0o600)
            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "retained disposable target changed",
            ):
                harness.path_registry.assert_failure_accounted()
            late.unlink()
            harness.path_registry.assert_failure_accounted()

            extra = harness.cognee_scope_parent / "unregistered-extra-scope"
            extra.mkdir(mode=0o700)
            extra.chmod(0o700)
            try:
                with (
                    mock.patch.object(
                        runner.V2FoundationOwner,
                        "close",
                        new=self._close_fake_foundation,
                    ),
                    self.assertRaisesRegex(
                        runner.V2IntegrityStop,
                        "V2_LINEAGE_OWNER_CLOSE_FAILED",
                    ),
                ):
                    asyncio.run(harness.product.close())
                self.assertFalse(harness.lineages._closed)
            finally:
                extra.rmdir()
            with mock.patch.object(
                runner.V2FoundationOwner,
                "close",
                new=self._close_fake_foundation,
            ):
                asyncio.run(harness.product.close())
            self.assertTrue(harness.product._closed)
            self.assertTrue(scope.is_dir())
            harness.path_registry.assert_failure_accounted()
            harness.ledger.close()

    def test_disposable_model_graph_detaches_before_foundation_and_close_retries(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness, _frozen, task, context = asyncio.run(
                self._frozen_disposable_fixture(state_root)
            )
            source = harness.lineages.owner_for(
                context.replicate_commitment,
                "IN_FULL_NEUTRAL_12",
            )
            allocation, scope_spec = harness.product._register_disposable(
                task_id=task.task_id,
                replicate_commitment=context.replicate_commitment,
                role="IN_FULL_NEUTRAL_12",
                source_arm="IN_FULL_NEUTRAL_12",
                scope_kind="disposable-attempt",
            )
            assert scope_spec is not None
            runtime_root = Path(allocation.runtime_root)
            runtime_root.mkdir(mode=0o700)
            runtime_root.chmod(0o700)
            marker = runtime_root / "partial-runtime-state"
            marker.write_bytes(b"bounded model-bearing failure state")
            marker.chmod(0o600)
            session = asyncio.run(
                runner._open_v2_acquisition_session(
                    scope_spec,
                    opener=harness.product.acquisition_opener,
                )
            )

            runtime = object.__new__(runner.V2AttemptExecutionRuntime)
            runtime.arm = "IN_FULL_NEUTRAL_12"
            runtime.proposal_adapter = object()
            runtime.relation_adapter = object()
            runtime.executor = object()
            runtime.journal = object()
            runtime.guard = harness.foundation.guard
            runtime.cycle = object()
            runtime.learner = object()
            runtime.store = object()
            runtime.memory = object()
            runtime.session = session
            runtime.runtime_root = runtime_root
            runtime.allocation_ref = allocation.allocation_ref
            runtime.allocation = None
            runtime.path_registry = None
            runtime._turn = None
            runtime._completed = None
            runtime._last_observation_ref = None
            runtime._journal_identity = None
            attempt_owner = object.__new__(runner.V2StatefulAttemptOwner)
            attempt_owner.runtime = runtime
            attempt_owner.source_owner = source
            attempt_owner.ledger = harness.ledger
            attempt_owner.quiesce_runtime = lambda **_kwargs: None
            attempt_owner.source_seal = context.in_baseline
            attempt_owner.clone_paths = _OwnerPaths(
                runtime_root,
                runtime_root / "cognitive.sqlite3",
                runtime_root / "qwen-execution-journal.sqlite3",
                runtime_root / "learner-state.bin",
            )
            attempt_owner.clone_before = context.in_baseline.hashes
            attempt_owner.allocation = allocation
            attempt_owner.path_registry = harness.path_registry
            attempt_owner._terminal = None
            attempt_owner._closed = False
            stage_ref = _ref("model-bearing-disposable-failed-stage")
            cell = (task.task_id, "IN_FULL_NEUTRAL_12")
            harness.product._active_by_stage[stage_ref] = attempt_owner
            harness.product._active_cells[cell] = stage_ref

            foundation_calls = 0

            def close_foundation(owner: runner.V2FoundationOwner) -> None:
                nonlocal foundation_calls
                foundation_calls += 1
                self.assertIsNone(attempt_owner.runtime)
                self.assertTrue(attempt_owner._closed)
                self.assertFalse(harness.product._active_by_stage)
                self.assertFalse(harness.product._active_cells)
                self.assertFalse(harness.product._capture_owners)
                for value in (
                    runtime.proposal_adapter,
                    runtime.relation_adapter,
                    runtime.executor,
                    runtime.journal,
                    runtime.guard,
                    runtime.cycle,
                    runtime.learner,
                    runtime.store,
                    runtime.memory,
                    runtime.session,
                ):
                    self.assertIsNone(value)
                if foundation_calls == 1:
                    raise RuntimeError("synthetic first foundation close failure")
                self._close_fake_foundation(owner)

            with (
                mock.patch.object(
                    runner.V2FoundationOwner,
                    "close",
                    new=close_foundation,
                ),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "V2_PRODUCT_FOUNDATION_CLOSE_FAILED",
                ),
            ):
                asyncio.run(harness.product.close())
            self.assertTrue(harness.lineages._closed)
            self.assertFalse(harness.product._closed)
            self.assertEqual(foundation_calls, 1)
            self.assertEqual(
                len(harness.product._retained_disposable_allocations),
                1,
            )
            with mock.patch.object(
                runner.V2FoundationOwner,
                "close",
                new=close_foundation,
            ):
                asyncio.run(harness.product.close())
            self.assertEqual(foundation_calls, 2)
            self.assertTrue(harness.product._closed)
            harness.path_registry.assert_failure_accounted()
            count = harness.ledger._connection.execute(
                "SELECT COUNT(*) FROM artifacts WHERE kind=?",
                ("retained-disposable-allocation",),
            ).fetchone()[0]
            self.assertEqual(count, 1)
            harness.ledger.close()

    def test_retained_disposable_append_is_idempotently_retryable(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness, _frozen, task, context = asyncio.run(
                self._frozen_disposable_fixture(state_root)
            )
            allocation, _scope_spec = harness.product._register_disposable(
                task_id=task.task_id,
                replicate_commitment=context.replicate_commitment,
                role="N0_NATIVE_NO_HISTORY",
            )
            runtime_root = Path(allocation.runtime_root)
            runtime_root.mkdir(mode=0o700)
            runtime_root.chmod(0o700)
            marker = runtime_root / "failed-stateless-build"
            marker.write_bytes(b"bounded stateless failure state")
            marker.chmod(0o600)
            pending = harness.product._queue_failed_disposable_allocation(
                allocation=allocation,
                source_owner=None,
                source_seal=None,
                failure_boundary_code="V2_STATELESS_BUILD_FAILED",
                stage_ref=_ref("failed-stateless-build-stage"),
                session=None,
            )
            original = harness.ledger.append_artifact_durable
            injected = False

            def commit_then_fail(
                artifact_ref: str,
                kind: str,
                payload: dict[str, object],
            ) -> str:
                nonlocal injected
                result = original(artifact_ref, kind, payload)
                if kind == "retained-disposable-allocation" and not injected:
                    injected = True
                    raise OSError("synthetic post-commit durability failure")
                return result

            with (
                mock.patch.object(
                    harness.ledger,
                    "append_artifact_durable",
                    side_effect=commit_then_fail,
                ),
                self.assertRaisesRegex(
                    OSError,
                    "post-commit durability failure",
                ),
            ):
                asyncio.run(
                    harness.product._retain_failed_disposable_allocation(pending)
                )
            self.assertTrue(injected)
            self.assertIn(
                allocation.allocation_ref,
                harness.product._pending_disposable_retentions,
            )
            self.assertFalse(harness.product._retained_disposable_allocations)
            self.assertFalse(harness.path_registry._retained)
            evidence = asyncio.run(
                harness.product._retain_failed_disposable_allocation(pending)
            )
            self.assertIsInstance(
                evidence,
                runner.RetainedDisposableAllocationEvidence,
            )
            self.assertFalse(harness.product._pending_disposable_retentions)
            harness.path_registry.assert_failure_accounted()
            count = harness.ledger._connection.execute(
                "SELECT COUNT(*) FROM artifacts WHERE kind=?",
                ("retained-disposable-allocation",),
            ).fetchone()[0]
            self.assertEqual(count, 1)
            with mock.patch.object(
                runner.V2FoundationOwner,
                "close",
                new=self._close_fake_foundation,
            ):
                asyncio.run(harness.product.close())
            self.assertTrue(harness.product._closed)
            harness.ledger.close()

    def test_failed_prestage_scope_is_durably_owned_and_not_disposed(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness = asyncio.run(
                _FullFakeProductHarness.create(
                    purpose="qualification",
                    root=state_root,
                )
            )
            harness.product.mechanics = _owner_mechanics(
                harness.events,
                ledger=harness.ledger,
                fail_rebuild_arms=frozenset({"IO_FULL_ORDINARY_12"}),
            )
            scheduled = next(
                item
                for item in runner.adaptation_schedule(
                    evaluator.make_qualification_evaluator().release_phase(
                        "adaptation"
                    ),
                    purpose="qualification",
                )
                if item.arm == "IO_FULL_ORDINARY_12"
            )
            source = harness.lineages.owner_for(
                scheduled.replicate_commitment,
                "IO_FULL_ORDINARY_12",
            )

            with self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "V2_PERSISTENT_ATTEMPT_PREPARATION_FAILED",
            ):
                asyncio.run(
                    harness.product.prepare_adaptation_attempt(scheduled)
                )

            retained = tuple(
                harness.product._retained_preparation_scopes.values()
            )
            self.assertEqual(len(retained), 1)
            evidence = retained[0]
            self.assertEqual(evidence.task_id, scheduled.task_id)
            self.assertEqual(evidence.allocation_ref, source.allocation_ref)
            self.assertEqual(evidence.scope_ref, source.scope_spec.scope_ref)
            self.assertTrue(Path(evidence.state_root).is_dir())
            harness.ledger.require_artifact(
                evidence.evidence_ref,
                "retained-preparation-scope",
                evidence.to_canonical(),
            )
            failure_open = max(
                index
                for index, event in enumerate(harness.events)
                if event == "open:IO_FULL_ORDINARY_12"
            )
            failure_events = harness.events[failure_open:]
            self.assertIn("close:IO_FULL_ORDINARY_12", failure_events)
            self.assertNotIn("forget:IO_FULL_ORDINARY_12", failure_events)

            def close_foundation(owner: runner.V2FoundationOwner) -> None:
                owner.guard = None
                owner.loaded = None
                owner._closed = True

            with mock.patch.object(
                runner.V2FoundationOwner,
                "close",
                new=close_foundation,
            ):
                asyncio.run(harness.product.close())
            self.assertTrue(harness.product._closed)
            self.assertTrue(harness.lineages._closed)
            self.assertTrue(Path(evidence.state_root).is_dir())
            kinds = tuple(
                row[0]
                for row in harness.ledger._connection.execute(
                    "SELECT kind FROM artifacts ORDER BY kind"
                ).fetchall()
            )
            self.assertIn("retained-preparation-scope", kinds)
            self.assertNotIn("persistent-runtime-cleanup", kinds)
            harness.ledger.close()

    def test_poststage_partial_bytes_are_bound_before_owner_detach(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness = asyncio.run(
                _FullFakeProductHarness.create(
                    purpose="qualification",
                    root=state_root,
                )
            )
        scheduled = next(
            item
            for item in runner.adaptation_schedule(
                evaluator.make_qualification_evaluator().release_phase(
                    "adaptation"
                ),
                purpose="qualification",
            )
            if item.arm == "IN_FULL_NEUTRAL_12"
        )
        owner = asyncio.run(
            self._install_failed_poststage_owner(harness, scheduled)
        )
        expected_stage_ref = harness.product._active_cells[
            scheduled.attempt_key
        ]
        source = owner.source_owner
        before = runner._raw_retained_lineage_state(source)
        source.paths.journal.write_bytes(b"post-stage-partial-mutation")
        source.paths.journal.chmod(0o600)
        after = runner._raw_retained_lineage_state(source)
        self.assertNotEqual(before[1], after[1])
        with mock.patch.object(
            runner.V2FoundationOwner,
            "close",
            new=self._close_fake_foundation,
        ):
            asyncio.run(harness.product.close())
        retained = tuple(harness.product._retained_preparation_scopes.values())
        self.assertEqual(len(retained), 1)
        evidence = retained[0]
        self.assertEqual(
            evidence.reason_code,
            "V2_PERSISTENT_ATTEMPT_POST_STAGE_FAILED",
        )
        self.assertEqual(
            evidence.stage_ref,
            expected_stage_ref,
        )
        self.assertEqual(evidence.lineage_hashes, after[1])
        self.assertEqual(evidence.lineage_tree, after[2])
        self.assertTrue(owner._closed)
        self.assertIsNone(owner.runtime)
        self.assertTrue(Path(evidence.state_root).is_dir())
        harness.ledger.close()

    def test_prepared_but_unstaged_failure_is_not_labeled_poststage(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_root = Path(temporary.name).resolve() / "state"
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness = asyncio.run(
                _FullFakeProductHarness.create(
                    purpose="qualification",
                    root=state_root,
                )
            )
        scheduled = next(
            item
            for item in runner.adaptation_schedule(
                evaluator.make_qualification_evaluator().release_phase(
                    "adaptation"
                ),
                purpose="qualification",
            )
            if item.arm == "IN_FULL_NEUTRAL_12"
        )
        owner = asyncio.run(
            self._install_failed_poststage_owner(
                harness,
                scheduled,
                staged=False,
            )
        )
        with mock.patch.object(
            runner.V2FoundationOwner,
            "close",
            new=self._close_fake_foundation,
        ):
            asyncio.run(harness.product.close())
        evidence = next(iter(harness.product._retained_preparation_scopes.values()))
        self.assertEqual(
            evidence.reason_code,
            "V2_PERSISTENT_ATTEMPT_PREPARATION_FAILED",
        )
        self.assertIsNone(evidence.stage_ref)
        self.assertTrue(owner._closed)
        harness.ledger.close()

    def test_terminal_but_raising_close_retains_and_reports_original(self) -> None:
        harness, scheduled, _source = self._failed_preparation_fixture()
        original = runner.V2AcquisitionSession.close_preserving_state

        async def terminal_then_raise(
            session: runner.V2AcquisitionSession,
        ) -> None:
            await original(session)
            raise runner.V2IntegrityStop("synthetic terminal close failure")

        with (
            mock.patch.object(
                runner.V2AcquisitionSession,
                "close_preserving_state",
                new=terminal_then_raise,
            ),
            self.assertRaises(ExceptionGroup) as raised,
        ):
            asyncio.run(harness.product.prepare_adaptation_attempt(scheduled))
        leaves = tuple(
            str(item)
            for item in raised.exception.exceptions
        )
        self.assertIn("V2_PERSISTENT_ATTEMPT_PREPARATION_FAILED", leaves)
        self.assertIn("V2_PERSISTENT_PREPARATION_WORKER_CLOSE_FAILED", leaves)
        self.assertEqual(len(harness.product._retained_preparation_scopes), 1)
        self.assertFalse(harness.product._pending_scope_retentions)
        with mock.patch.object(
            runner.V2FoundationOwner,
            "close",
            new=self._close_fake_foundation,
        ):
            asyncio.run(harness.product.close())
        self.assertTrue(harness.product._closed)
        harness.ledger.close()

    def test_opener_failure_after_scope_creation_retains_without_dataset(self) -> None:
        harness, scheduled, _source = self._failed_preparation_fixture()

        async def fail_after_scope_creation(
            scope_spec: runner.V2AcquisitionScopeSpec,
            *,
            expected_dataset_id: str | None = None,
        ) -> object:
            self.assertIsNone(expected_dataset_id)
            root = Path(scope_spec.state_root)
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            marker = root / "failed-before-session"
            marker.write_bytes(b"terminal scope without returned session")
            marker.chmod(0o600)
            raise RuntimeError("synthetic opener failure after scope creation")

        harness.product.acquisition_opener = fail_after_scope_creation
        with self.assertRaisesRegex(
            runner.V2IntegrityStop,
            "V2_PERSISTENT_ATTEMPT_PREPARATION_FAILED",
        ):
            asyncio.run(harness.product.prepare_adaptation_attempt(scheduled))
        evidence = next(iter(harness.product._retained_preparation_scopes.values()))
        self.assertIsNone(evidence.dataset_id)
        self.assertTrue(evidence.worker_terminal)
        self.assertFalse(harness.product._pending_scope_retentions)
        with mock.patch.object(
            runner.V2FoundationOwner,
            "close",
            new=self._close_fake_foundation,
        ):
            asyncio.run(harness.product.close())
        self.assertTrue(harness.product._closed)
        harness.ledger.close()

    def test_extra_direct_scope_alias_fails_then_exact_retry_succeeds(self) -> None:
        harness, scheduled, _source = self._failed_preparation_fixture()
        with self.assertRaises(runner.V2IntegrityStop):
            asyncio.run(harness.product.prepare_adaptation_attempt(scheduled))
        evidence = next(iter(harness.product._retained_preparation_scopes.values()))
        alias = harness.cognee_scope_parent / "unexpected-scope-alias"
        alias.symlink_to(Path(evidence.state_root), target_is_directory=True)
        try:
            with (
                mock.patch.object(
                    runner.V2FoundationOwner,
                    "close",
                    new=self._close_fake_foundation,
                ),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "V2_FAILED_LINEAGE_OWNER_CLOSE_FAILED",
                ),
            ):
                asyncio.run(harness.product.close())
            self.assertFalse(harness.lineages._closed)
            self.assertTrue(alias.is_symlink())
        finally:
            alias.unlink(missing_ok=True)
        with mock.patch.object(
            runner.V2FoundationOwner,
            "close",
            new=self._close_fake_foundation,
        ):
            asyncio.run(harness.product.close())
        self.assertTrue(harness.product._closed)
        self.assertTrue(Path(evidence.state_root).is_dir())
        harness.ledger.close()

    def test_scope_tree_enumeration_stops_at_maximum_plus_one(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve() / "bounded-tree"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        consumed: list[str] = []

        class Entry:
            def __init__(self, name: str) -> None:
                self.name = name

            def stat(self, *, follow_symlinks: bool) -> os.stat_result:
                raise AssertionError("over-limit entry reached metadata inspection")

        class Scan:
            def __enter__(self) -> "Scan":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def __iter__(self) -> object:
                for name in ("first", "second"):
                    consumed.append(name)
                    yield Entry(name)
                raise AssertionError("enumeration continued past maximum+1")

        with (
            mock.patch.object(runner, "MAXIMUM_RETAINED_SCOPE_ENTRIES", 2),
            mock.patch.object(runner.os, "scandir", return_value=Scan()),
            self.assertRaisesRegex(
                runner.V2IntegrityStop,
                "retained scope entry ceiling exceeded",
            ),
        ):
            runner._terminal_scope_tree_snapshot_once(root)
        self.assertEqual(consumed, ["first", "second"])

    def test_late_extra_scope_child_is_detected_before_lineage_detach(self) -> None:
        harness, scheduled, _source = self._failed_preparation_fixture()
        with self.assertRaises(runner.V2IntegrityStop):
            asyncio.run(harness.product.prepare_adaptation_attempt(scheduled))
        original = runner._validate_failed_lineage_owner_preserving_state
        validation_count = 0
        late = harness.cognee_scope_parent / "late-unowned-scope"

        def validate_then_insert(
            owner: runner.V2LineageOwner,
            retained: runner.RetainedPreparationScopeEvidence | None,
        ) -> None:
            nonlocal validation_count
            original(owner, retained)
            validation_count += 1
            if validation_count == len(harness.lineages.owners):
                late.mkdir(mode=0o700)
                late.chmod(0o700)

        try:
            with (
                mock.patch.object(
                    runner,
                    "_validate_failed_lineage_owner_preserving_state",
                    side_effect=validate_then_insert,
                ),
                mock.patch.object(
                    runner.V2FoundationOwner,
                    "close",
                    new=self._close_fake_foundation,
                ),
                self.assertRaisesRegex(
                    runner.V2IntegrityStop,
                    "V2_FAILED_LINEAGE_OWNER_CLOSE_FAILED",
                ),
            ):
                asyncio.run(harness.product.close())
            self.assertFalse(harness.lineages._closed)
            self.assertTrue(all(item.paths is not None for item in harness.lineages.owners))
        finally:
            late.rmdir()
        with mock.patch.object(
            runner.V2FoundationOwner,
            "close",
            new=self._close_fake_foundation,
        ):
            asyncio.run(harness.product.close())
        self.assertTrue(harness.product._closed)
        harness.ledger.close()

    def test_committed_retention_append_failure_is_idempotently_retryable(self) -> None:
        harness, scheduled, _source = self._failed_preparation_fixture()
        original = harness.ledger.append_artifact_durable
        failed = False

        def commit_then_fail(
            artifact_ref: str,
            kind: str,
            payload: dict[str, object],
        ) -> str:
            nonlocal failed
            result = original(artifact_ref, kind, payload)
            if kind == "retained-preparation-scope" and not failed:
                failed = True
                raise OSError("synthetic post-commit fsync failure")
            return result

        with (
            mock.patch.object(
                harness.ledger,
                "append_artifact_durable",
                side_effect=commit_then_fail,
            ),
            self.assertRaises(ExceptionGroup),
        ):
            asyncio.run(harness.product.prepare_adaptation_attempt(scheduled))
        self.assertTrue(failed)
        self.assertFalse(harness.ledger._closed)
        self.assertFalse(harness.product._retained_preparation_scopes)
        self.assertEqual(len(harness.product._pending_scope_retentions), 1)
        with mock.patch.object(
            runner.V2FoundationOwner,
            "close",
            new=self._close_fake_foundation,
        ):
            asyncio.run(harness.product.close())
        self.assertTrue(harness.product._closed)
        self.assertEqual(len(harness.product._retained_preparation_scopes), 1)
        self.assertFalse(harness.product._pending_scope_retentions)
        count = harness.ledger._connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind=?",
            ("retained-preparation-scope",),
        ).fetchone()[0]
        self.assertEqual(count, 1)
        harness.ledger.close()

    def _run_full_fake(self, purpose: runner.Purpose) -> tuple[dict[str, object], _FullFakeProductHarness, _InjectedEvaluator]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        literal_state_root = runner.STATE_ROOT
        state_root = Path(temporary.name).resolve() / "state"
        self.assertNotEqual(state_root, literal_state_root)
        with mock.patch.object(runner, "STATE_ROOT", state_root):
            harness = asyncio.run(
                _FullFakeProductHarness.create(purpose=purpose, root=state_root)
            )
            suite = (
                evaluator.make_qualification_evaluator()
                if purpose == "qualification"
                else evaluator.make_evaluation_evaluator(
                    _EVALUATION_SEEDS,
                    _EVALUATION_COMMITMENTS,
                    _ADMISSION_REF,
                )
            )
            evaluator_boundary = _InjectedEvaluator(suite)
            with harness.patched_product_methods():
                dependencies = runner.OrchestrationDependencies(
                    mode=runner.RunMode(purpose),
                    evaluator=evaluator_boundary,
                    ledger=harness.ledger,
                    path_allocations=harness.path_registry,
                    attempt_budget=runner.AttemptBudget(runner.RunMode(purpose)),
                    resources=harness.resources,
                    prepare_adaptation_attempt=harness.product.prepare_adaptation_attempt,
                    freeze_adaptation=harness.product.freeze_adaptation,
                    prepare_task_block=harness.product.prepare_task_block,
                    dispose_recall_captures=harness.product.dispose_recall_captures,
                    execute_attempt=harness.product.execute_attempt,
                    apply_objective_feedback=harness.product.apply_objective_feedback,
                    finalize_persistent_stateful_attempt=(
                        harness.product.finalize_persistent_stateful_attempt
                    ),
                    terminalize_stateful_attempt=(
                        harness.product.terminalize_stateful_attempt
                    ),
                    dispose_clone=harness.product.dispose_clone,
                    dispose_stateless_attempt=(
                        harness.product.dispose_stateless_attempt
                    ),
                    verify_frozen_adaptation=(
                        harness.product.verify_frozen_adaptation
                    ),
                    sample_resources=harness.sample_resources,
                    verify_source_seal=harness.verify_source_seal,
                    close_runtime=harness.product.close,
                )
                for callback in (
                    dependencies.prepare_adaptation_attempt,
                    dependencies.freeze_adaptation,
                    dependencies.prepare_task_block,
                    dependencies.dispose_recall_captures,
                    dependencies.execute_attempt,
                    dependencies.apply_objective_feedback,
                    dependencies.finalize_persistent_stateful_attempt,
                    dependencies.terminalize_stateful_attempt,
                    dependencies.dispose_clone,
                    dependencies.dispose_stateless_attempt,
                    dependencies.verify_frozen_adaptation,
                    dependencies.close_runtime,
                ):
                    self.assertIs(callback.__self__, harness.product)
                result = asyncio.run(
                    runner.run_qualification_injected(
                        dependencies,
                        qualification_release_ref=_ref(
                            "full-fake-qualification-release"
                        ),
                    )
                    if purpose == "qualification"
                    else runner.run_evaluation_injected(
                        dependencies,
                        admission_ref=_ADMISSION_REF,
                    )
                )
            self.assertEqual(runner.validate_run_result(result), result)
            self.assertTrue(harness.ledger._sealed)
            self.assertTrue(harness.ledger._closed)
            self.assertEqual(
                runner.audit_evidence_ledger(
                    harness.ledger.path,
                    expected_attempts=(72 if purpose == "qualification" else 918),
                ),
                result["integrity"]["ledger"],
            )
            for allocation in harness.path_registry._allocations.values():
                self.assertTrue(
                    all(Path(target).is_relative_to(state_root) for target in allocation.targets)
                )
            self.assertTrue(
                all(owner.root.is_relative_to(state_root) for owner in harness.lineages.owners)
            )
            return result, harness, evaluator_boundary

    def test_full_fake_orchestration_and_evidence_close_exact_denominators(self) -> None:
        for purpose, expected in (("qualification", 72), ("evaluation", 918)):
            with self.subTest(purpose=purpose):
                result, harness, evaluator_boundary = self._run_full_fake(purpose)
                self.assertEqual(result["purpose"], purpose)
                self.assertEqual(result["accounting"]["arm_tasks"], expected)
                self.assertEqual(result["attempts"]["attempts"], expected)
                self.assertEqual(result["attempts"]["admitted"], expected)
                self.assertEqual(
                    result["resources"]["runtime_embedding_rows"],
                    (810 if purpose == "qualification" else 6_480),
                )
                self.assertTrue(harness.closed)
                self.assertTrue(evaluator_boundary.closed)
                self.assertEqual(harness.path_registry._closed, set(harness.path_registry._allocations))
                self.assertEqual(tuple(harness.cognee_scope_parent.iterdir()), ())
                self.assertEqual(
                    len(tuple(harness.arm_runtime_parent.iterdir())),
                    2 if purpose == "qualification" else 6,
                )


if __name__ == "__main__":
    unittest.main()
