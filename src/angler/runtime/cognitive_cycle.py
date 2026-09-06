"""One transactional cognitive cycle over Angler's existing adaptive state.

The cycle is deliberately orchestration, not a task solver.  A replaceable
model proposes public procedure traces, the existing ability learner scores
them, and an injected executor performs only the selected trace.  Objective
feedback is accepted after execution and causes one learner update followed by
one atomic canonical commit.  Cognee/Moving-Origin projection happens only
after that commit and is therefore rebuildable rather than authoritative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import inspect
import hashlib
import math
import re
from typing import Any, Literal, Protocol, runtime_checkable

from angler.cognition.contracts import CognitiveEpisode, ProspectiveCommitment
from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
    ObjectiveFeedbackRecord,
    ProspectiveTurnReservation,
)
from angler.cognition.prospective_origin import (
    CognitiveEpisodeV2,
    Perspective,
    ProspectiveBranch,
    ProspectiveDynamicsBatch,
    ProspectiveResolution,
    ProspectiveResourceEnvelope,
    ProspectiveTurnReservationV2,
    RealityMode,
    ResolutionDisposition,
    SituatedContext,
    SituatedStateLineage,
)
from angler.memory import MemoryProjection, RecallBatch
from angler.memory.cognitive_acquisition import CognitiveAcquisition
from angler.memory.cognitive_acquisition_graph import AcquisitionReferenceHit
from angler.episodes.canonical import canonical_bytes
from angler.reasoning.prospective_dynamics import ProspectiveResourceBudget
from angler.runtime.autonomous_apprentice import OutcomeContext
from angler.runtime.durable_ability_bridge import (
    AbilityDecision,
    EncodedProcedureCandidate,
    FrozenProcedureTextAdapter,
    ProcedureRelationAdapter,
)
from angler.runtime.prospective_observation import (
    FrozenObservedStateEncoder,
    QwenReceiptObservedStateEncoderV1,
    SyntheticObservedStateEncoderV1,
)


_MAX_TASK = 16_384
_MAX_TRACE = 1_024
_MAX_RESPONSE = 16_384
_MAX_OBSERVATION = 4_096
_MAX_FEEDBACK = 2_048
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REMOVAL_CONDITIONS = frozenset(
    ("FULL", "PROSPECTIVE_REMOVAL", "FROZEN_ORIGIN", "BACKEND_REMOVAL")
)


def _text(
    value: str,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> str:
    if (
        type(value) is not str
        or len(value) > maximum
        or (not value.strip() and not (allow_empty and value == ""))
    ):
        qualifier = "text" if allow_empty else "non-empty text"
        raise ValueError(f"{label} must be {qualifier} of at most {maximum} characters")
    return value.strip()


def _digest(value: str, label: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _snapshot_sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _content_ref(value: str) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes({"content": value})).hexdigest()


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@runtime_checkable
class CognitiveExecutor(Protocol):
    """Durable external-action boundary with exact receipt recovery.

    Implementations persist one canonical receipt under request.idempotency_key
    before returning an effect and return that receipt from RECORDED recovery.
    """

    def execute(
        self,
        request: CognitiveExecutionRequest,
    ) -> "CognitiveExecution | Any": ...

    def recover(
        self,
        request: CognitiveExecutionRequest,
    ) -> "CognitiveExecutionRecovery | Any": ...


@runtime_checkable
class CognitiveStateOwner(Protocol):
    """The narrow DurableAbilityLearner-compatible state-owner contract."""

    @property
    def pending_decision(self) -> AbilityDecision | None: ...

    def select(
        self,
        *,
        task_id: str,
        request: str,
        candidates: Sequence[EncodedProcedureCandidate],
    ) -> AbilityDecision: ...

    def apply_outcome(self, outcome: OutcomeContext) -> None: ...

    def capture_state(self) -> bytes: ...

    def restore_state(self, state: bytes) -> None: ...

    def state_digest(self) -> str: ...

    def capture_pending_state(self) -> bytes: ...

    def restore_pending_state(self, state: bytes) -> AbilityDecision: ...


@runtime_checkable
class ProspectiveCognitiveStateOwner(CognitiveStateOwner, Protocol):
    """Additional learned-state surface required by the successor lane."""

    @property
    def pending_prospective_material(self) -> object | None: ...

    def select_prospective(
        self,
        *,
        task_id: str,
        request: str,
        candidates: Sequence[EncodedProcedureCandidate],
        recalled_refs: tuple[str, ...],
        context_refs: tuple[str, ...],
        eligibility_rows: tuple[tuple[int, ...], ...],
        resource_budget: object,
    ) -> AbilityDecision: ...

    def bind_pending_prospective_batch(self, batch_ref: str) -> object: ...

    def component_state_integrity(self) -> object: ...


@dataclass(frozen=True, slots=True)
class CognitiveExecution:
    status: Literal["COMPLETED", "CLARIFICATION_REQUIRED", "ERROR"]
    executed_trace: str
    response: str
    observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("COMPLETED", "CLARIFICATION_REQUIRED", "ERROR"):
            raise ValueError("execution status is unsupported")
        _text(self.executed_trace, "executed_trace", _MAX_TRACE)
        _text(self.response, "response", _MAX_RESPONSE, allow_empty=True)
        if type(self.observations) is not tuple:
            raise TypeError("execution observations must be a tuple")
        for item in self.observations:
            _text(item, "execution observation", _MAX_OBSERVATION)


@dataclass(frozen=True, slots=True)
class CognitiveExecutionRecovery:
    """Authoritative executor knowledge for one already-claimed request."""

    status: Literal["RECORDED", "NOT_STARTED", "IN_PROGRESS", "UNKNOWN"]
    receipt: CognitiveExecutionReceipt | None = None

    def __post_init__(self) -> None:
        if self.status not in ("RECORDED", "NOT_STARTED", "IN_PROGRESS", "UNKNOWN"):
            raise ValueError("execution recovery status is unsupported")
        if self.status == "RECORDED":
            if not isinstance(self.receipt, CognitiveExecutionReceipt):
                raise ValueError("RECORDED recovery requires the exact durable receipt")
        elif self.receipt is not None:
            raise ValueError("only RECORDED recovery may contain a receipt")


class ExecutionReconciliationRequired(RuntimeError):
    """Raised when an external effect may have happened but is not yet known."""

    def __init__(self, reservation_ref: str) -> None:
        self.reservation_ref = _digest(reservation_ref, "reservation_ref")
        super().__init__(
            "claimed execution requires authoritative reconciliation before retry: "
            + self.reservation_ref
        )


@dataclass(frozen=True, slots=True)
class PendingCognitiveTurn:
    decision: AbilityDecision
    recalled_refs: tuple[str, ...]
    commitment: ProspectiveCommitment
    parent_state_snapshot: bytes
    parent_head: object
    reservation: ProspectiveTurnReservation
    successor_reservation: ProspectiveTurnReservationV2 | None = None
    parent_lineage: SituatedStateLineage | None = None


@dataclass(frozen=True, slots=True)
class ExecutedCognitiveTurn:
    turn: PendingCognitiveTurn
    execution: CognitiveExecution
    input_observations: tuple[str, ...]
    execution_request: CognitiveExecutionRequest
    execution_receipt: CognitiveExecutionReceipt


@dataclass(frozen=True, slots=True)
class CognitiveCommitResult:
    episode_ref: str
    sequence: int
    head: object
    projection_pending: bool


@dataclass(frozen=True, slots=True)
class CognitiveResolutionResult:
    resolution_ref: str
    acquisition_ref: str
    projection_pending: bool


def _uncertainty(logits: tuple[float, ...]) -> float:
    """Return normalized label-free uncertainty from the public scores."""

    if not 2 <= len(logits) <= 64 or any(not math.isfinite(item) for item in logits):
        raise ValueError("decision must expose 2 through 64 finite logits")
    maximum = max(logits)
    weights = [math.exp(item - maximum) for item in logits]
    total = sum(weights)
    probabilities = [item / total for item in weights]
    entropy = -sum(item * math.log(item) for item in probabilities if item > 0.0)
    return entropy / math.log(float(len(logits)))


def _outbox_fields(item: object) -> tuple[int, str, CognitiveEpisode]:
    if isinstance(item, tuple) and len(item) == 3:
        sequence, episode_ref, episode = item
    else:
        sequence = getattr(item, "sequence")
        episode_ref = getattr(item, "episode_ref")
        episode = getattr(item, "episode")
    return int(sequence), str(episode_ref), episode


class CognitiveCycle:
    """Coordinate one learned procedure turn and one canonical state commit."""

    def __init__(
        self,
        *,
        text_adapter: FrozenProcedureTextAdapter,
        relation_adapter: ProcedureRelationAdapter,
        learner: CognitiveStateOwner,
        memory: object,
        executor: CognitiveExecutor,
        transaction_store: object,
        model_ref: str,
        encoder_ref: str,
        agent_ref: str,
        world_ref: str,
        recall_limit: int = 12,
        proposal_count: int = 4,
        successor_mode: bool = False,
        reality_mode: RealityMode | str | None = None,
        subject_ref: str | None = None,
        scope_ref: str | None = None,
        bootstrap_evidence_refs: tuple[str, ...] = (),
        observation_encoder: FrozenObservedStateEncoder | None = None,
        removal_condition: Literal[
            "FULL",
            "PROSPECTIVE_REMOVAL",
            "FROZEN_ORIGIN",
            "BACKEND_REMOVAL",
        ] = "FULL",
        frozen_recall_hits: tuple[AcquisitionReferenceHit, ...] = (),
    ) -> None:
        if not isinstance(text_adapter, FrozenProcedureTextAdapter):
            raise TypeError("text_adapter does not implement FrozenProcedureTextAdapter")
        if not isinstance(relation_adapter, ProcedureRelationAdapter):
            raise TypeError("relation_adapter does not implement ProcedureRelationAdapter")
        if not isinstance(learner, CognitiveStateOwner):
            raise TypeError("learner does not implement the cognitive state-owner contract")
        if not callable(getattr(memory, "recall", None)):
            raise TypeError("memory must provide recall")
        if type(successor_mode) is not bool:
            raise TypeError("successor_mode must be bool")
        typed_projection = callable(getattr(memory, "remember_episode_item", None))
        legacy_projection = callable(getattr(memory, "remember", None)) and callable(
            getattr(memory, "reproject", None)
        )
        acquisition_projection = all(
            callable(getattr(memory, method, None))
            for method in (
                "retry_pending_projections",
                "assert_synchronized",
                "recall",
            )
        )
        if successor_mode:
            if not acquisition_projection:
                raise TypeError("successor memory must project the acquisition stream")
        elif not (typed_projection or legacy_projection):
            raise TypeError(
                "memory must provide typed episode projection or legacy projection"
            )
        if not isinstance(executor, CognitiveExecutor):
            raise TypeError("executor does not implement CognitiveExecutor")
        for method in (
            "initialize",
            "assert_compatibility",
            "head",
            "load_head_state",
            "get_episode",
        ):
            if not callable(getattr(transaction_store, method, None)):
                raise TypeError(f"transaction_store is missing {method}")
        lane_methods = (
            (
                "active_prospective_turn",
                "prospective_turn_for_episode",
                "reserve_prospective_turn",
                "claim_prospective_turn",
                "record_prospective_execution",
                "stage_prospective_feedback",
                "resolve_prospective_turn",
                "commit_observed_prospective_turn",
                "acquisition_head",
                "pending_acquisition_projections",
            )
            if successor_mode
            else (
                "commit_episode",
                "active_reservation",
                "reserve_turn",
                "cancel_reservation",
                "claim_turn",
                "record_execution",
                "stage_feedback",
                "commit_reserved_episode",
                "resolve_noncompletion",
                "pending_projections",
                "ack_projection",
            )
        )
        for method in lane_methods:
            if not callable(getattr(transaction_store, method, None)):
                raise TypeError(f"transaction_store is missing {method}")
        if typed_projection and not callable(
            getattr(transaction_store, "get_episode_item", None)
        ):
            raise TypeError("typed memory requires transaction_store.get_episode_item")
        if (typed_projection or successor_mode) and getattr(
            memory, "source", None
        ) is not transaction_store:
            raise ValueError("typed memory and cognitive cycle must share one canonical store")
        if typed_projection and not callable(
            getattr(memory, "assert_synchronized", None)
        ):
            raise TypeError("typed memory must provide bounded synchronization validation")
        if type(recall_limit) is not int or recall_limit < 1:
            raise ValueError("recall_limit must be a positive integer")
        if type(proposal_count) is not int or not 2 <= proposal_count <= 64:
            raise ValueError("proposal_count must be an integer from 2 through 64")
        self.text_adapter = text_adapter
        self.relation_adapter = relation_adapter
        self.learner = learner
        self.memory = memory
        self._successor_mode = successor_mode
        self._typed_projection = typed_projection and not successor_mode
        self._acquisition_projection = acquisition_projection and successor_mode
        self.executor = executor
        self.transaction_store = transaction_store
        self.model_ref = _text(model_ref, "model_ref", 512)
        self.encoder_ref = _text(encoder_ref, "encoder_ref", 512)
        self.agent_ref = _text(agent_ref, "agent_ref", 512)
        self.world_ref = _text(world_ref, "world_ref", 512)
        _digest(self.model_ref, "model_ref")
        _digest(self.encoder_ref, "encoder_ref")
        _digest(self.agent_ref, "agent_ref")
        _digest(self.world_ref, "world_ref")
        self.recall_limit = recall_limit
        self.proposal_count = proposal_count
        self._situated_context: SituatedContext | None = None
        self._observation_encoder: FrozenObservedStateEncoder | None = None
        self._current_lineage: SituatedStateLineage | None = None
        self._qwen_genesis_binding: tuple[str, int, str] | None = None
        self._removal_condition = removal_condition
        self._frozen_recall_hits = frozen_recall_hits
        if successor_mode:
            for method in (
                "select_prospective",
                "bind_pending_prospective_batch",
                "component_state_integrity",
            ):
                if not callable(getattr(learner, method, None)):
                    raise TypeError(f"successor learner is missing {method}")
            synthetic_observation = (
                type(observation_encoder) is SyntheticObservedStateEncoderV1
            )
            qwen_observation = (
                type(observation_encoder) is QwenReceiptObservedStateEncoderV1
            )
            if not (synthetic_observation or qwen_observation):
                raise TypeError(
                    "the successor lane requires exact "
                    "SyntheticObservedStateEncoderV1 or exact "
                    "QwenReceiptObservedStateEncoderV1"
                )
            if (
                type(removal_condition) is not str
                or removal_condition not in _REMOVAL_CONDITIONS
            ):
                raise ValueError("successor removal_condition is not recognized")
            prospective_lesion = getattr(learner, "prospective_lesion", None)
            if type(prospective_lesion) is not bool or prospective_lesion is not (
                removal_condition == "PROSPECTIVE_REMOVAL"
            ):
                raise ValueError(
                    "successor prospective removal condition and learner differ"
                )
            if (
                type(frozen_recall_hits) is not tuple
                or len(frozen_recall_hits) > recall_limit
                or any(
                    type(hit) is not AcquisitionReferenceHit
                    for hit in frozen_recall_hits
                )
                or len({hit.record_ref for hit in frozen_recall_hits})
                != len(frozen_recall_hits)
            ):
                raise ValueError(
                    "frozen_recall_hits must be one bounded unique hit manifest"
                )
            if removal_condition != "FULL" and not frozen_recall_hits:
                raise ValueError(
                    "every removal condition requires a preassigned frozen hit manifest"
                )
            if frozen_recall_hits and not callable(
                getattr(memory, "recall_from_hits", None)
            ):
                raise TypeError(
                    "manifest-controlled recall requires canonical hit replay"
                )
            if (
                removal_condition != "BACKEND_REMOVAL"
                and frozen_recall_hits
                and not callable(
                    getattr(getattr(memory, "backend", None), "search", None)
                )
            ):
                raise TypeError(
                    "manifest-controlled recall requires a searchable reference backend"
                )
            if reality_mode is None:
                raise ValueError("successor cycle requires an explicit reality_mode")
            try:
                resolved_reality = RealityMode(reality_mode)
            except (TypeError, ValueError) as exc:
                raise ValueError("reality_mode must be ACTUAL or SIMULATED") from exc
            if resolved_reality not in (RealityMode.ACTUAL, RealityMode.SIMULATED):
                raise ValueError("reality_mode must be ACTUAL or SIMULATED")
            subject = _digest(subject_ref, "subject_ref")  # type: ignore[arg-type]
            scope = _digest(scope_ref, "scope_ref")  # type: ignore[arg-type]
            if (
                type(bootstrap_evidence_refs) is not tuple
                or not bootstrap_evidence_refs
                or bootstrap_evidence_refs != tuple(sorted(bootstrap_evidence_refs))
                or len(set(bootstrap_evidence_refs)) != len(bootstrap_evidence_refs)
            ):
                raise ValueError(
                    "bootstrap_evidence_refs must be a nonempty canonical tuple"
                )
            for evidence_ref in bootstrap_evidence_refs:
                _digest(evidence_ref, "bootstrap evidence ref")
            relation_encoder_ref = _digest(
                getattr(relation_adapter, "encoder_ref", None),
                "relation adapter encoder_ref",
            )
            if not (
                observation_encoder.encoder_ref
                == relation_encoder_ref
                == self.encoder_ref
            ):
                raise ValueError(
                    "cycle, observation, and relation encoder identities must match"
                )
            if qwen_observation:
                # Keep the live model path closed to protocol-shaped substitutes.
                # The import is local so the established synthetic lane does not
                # acquire the Qwen journal/representation implementation.
                from angler.runtime.qwen_cognitive import (
                    FrozenQwenCognitiveExecutorV1,
                    FrozenQwenCycleManifestV1,
                    FrozenQwenProcedureAdapterV1,
                    FrozenQwenProcedureRelationAdapterV1,
                )

                if type(text_adapter) is not FrozenQwenProcedureAdapterV1:
                    raise TypeError(
                        "Qwen successor use requires the exact frozen procedure adapter"
                    )
                if type(relation_adapter) is not FrozenQwenProcedureRelationAdapterV1:
                    raise TypeError(
                        "Qwen successor use requires the exact frozen relation adapter"
                    )
                if type(executor) is not FrozenQwenCognitiveExecutorV1:
                    raise TypeError(
                        "Qwen successor use requires the exact journaled executor"
                    )
                manifest = text_adapter.manifest
                if type(manifest) is not FrozenQwenCycleManifestV1:
                    raise TypeError("Qwen successor manifest class is not exact")
                if not (
                    text_adapter.io is relation_adapter.io
                    and relation_adapter.io is executor.io
                ):
                    raise ValueError(
                        "every Qwen component must share one loaded frozen model"
                    )
                manifest_ref = manifest.manifest_ref
                if not (
                    relation_adapter.manifest_ref
                    == executor.manifest_ref
                    == observation_encoder.manifest_ref
                    == manifest_ref
                ):
                    raise ValueError("Qwen component manifest identities differ")
                if not (
                    self.model_ref
                    == text_adapter.model_ref
                    == relation_adapter.model_ref
                    == executor.model_ref
                    == observation_encoder.model_ref
                    == manifest.model_ref
                ):
                    raise ValueError("Qwen component model identities differ")
                if not (
                    self.encoder_ref
                    == observation_encoder.encoder_ref
                    == relation_adapter.encoder_ref
                    == manifest.encoder_ref
                ):
                    raise ValueError("Qwen component encoder identities differ")
                if observation_encoder.latent_width != manifest.latent_width:
                    raise ValueError("Qwen observation latent width differs")
                if manifest.prospective_lesion is not False:
                    raise ValueError(
                        "the frozen Qwen base manifest must bind an intact learner"
                    )
                integrity = learner.component_state_integrity()
                if (
                    getattr(integrity, "checkpoint_ref", None)
                    != manifest.learner_checkpoint_ref
                    or getattr(integrity, "config_ref", None)
                    != manifest.prospective_config_ref
                ):
                    raise ValueError(
                        "Qwen learner checkpoint or prospective configuration differs"
                    )
                self._qwen_genesis_binding = (
                    manifest.initial_competence_state_digest,
                    manifest.genesis_snapshot_bytes,
                    manifest.genesis_snapshot_sha256,
                )
            self._situated_context = SituatedContext(
                reality_mode=resolved_reality,
                perspective=Perspective.AGENT_SITUATED,
                subject_ref=subject,
                scope_ref=scope,
                world_ref=self.world_ref,
                evidence_refs=bootstrap_evidence_refs,
            )
            self._observation_encoder = observation_encoder
        elif any(
            value is not None
            for value in (reality_mode, subject_ref, scope_ref, observation_encoder)
        ) or bootstrap_evidence_refs or removal_condition != "FULL" or frozen_recall_hits:
            raise ValueError("successor configuration requires successor_mode=True")
        self._beginning = False
        self._pending: PendingCognitiveTurn | None = None
        self._executing = False
        self._executed: ExecutedCognitiveTurn | None = None
        self._claimed_unknown = False
        self._restore_or_initialize()
        if self._successor_mode:
            self._restore_successor_lineage()
            self._restore_active_reservation()
        else:
            self._assert_memory_synchronized()
            self._restore_active_reservation()

    @property
    def pending_turn(self) -> PendingCognitiveTurn | None:
        return self._pending

    @property
    def removal_condition(self) -> str:
        """The construction-fixed evaluation intervention for this cycle."""

        return self._removal_condition

    @property
    def executed_turn(self) -> ExecutedCognitiveTurn | None:
        return self._executed

    def _restore_or_initialize(self) -> None:
        head = self.transaction_store.head()
        if head is None:
            snapshot = self.learner.capture_state()
            if self._qwen_genesis_binding is not None:
                expected_digest, expected_bytes, expected_sha256 = (
                    self._qwen_genesis_binding
                )
                if (
                    self.learner.state_digest() != expected_digest
                    or len(snapshot) != expected_bytes
                    or hashlib.sha256(snapshot).hexdigest() != expected_sha256
                ):
                    raise ValueError(
                        "Qwen learner does not begin from the frozen genesis snapshot"
                    )
            self.transaction_store.initialize(
                self.learner.state_digest(),
                snapshot,
                model_ref=self.model_ref,
                encoder_ref=self.encoder_ref,
            )
            return
        self.transaction_store.assert_compatibility(self.model_ref, self.encoder_ref)
        snapshot = self.transaction_store.load_head_state()
        if not isinstance(snapshot, bytes) or not snapshot:
            raise ValueError("canonical cognitive head has no valid state snapshot")
        self.learner.restore_state(snapshot)
        if self.learner.state_digest() != head.state_digest:
            raise ValueError("restored learner state does not match the canonical head")
        if head.episode_ref is not None:
            episode = self.transaction_store.get_episode(head.episode_ref)
            if episode.model_ref != self.model_ref or episode.encoder_ref != self.encoder_ref:
                raise ValueError("canonical head belongs to another model or encoder identity")

    @property
    def current_lineage(self) -> SituatedStateLineage | None:
        return self._current_lineage

    def _component_integrity(self) -> object:
        value = self.learner.component_state_integrity()  # type: ignore[attr-defined]
        if type(getattr(value, "step", None)) is not int or value.step < 0:
            raise ValueError("successor component integrity has an invalid step")
        for field in (
            "checkpoint_ref",
            "config_ref",
            "world_state_digest",
            "self_state_digest",
            "focus_state_digest",
            "outcome_state_digest",
        ):
            _digest(getattr(value, field, None), f"component {field}")
        return value

    def _lineage_for_current_state(
        self,
        *,
        parent_lineage_ref: str | None,
        state_evidence_refs: tuple[str, ...],
    ) -> SituatedStateLineage:
        if self._situated_context is None:
            raise RuntimeError("successor situated context is absent")
        snapshot = self.learner.capture_state()
        integrity = self._component_integrity()
        return SituatedStateLineage(
            context=self._situated_context,
            agent_ref=self.agent_ref,
            parent_lineage_ref=parent_lineage_ref,
            state_step=integrity.step,
            checkpoint_ref=integrity.checkpoint_ref,
            config_ref=integrity.config_ref,
            competence_state_digest=self.learner.state_digest(),
            snapshot_digest=_snapshot_sha256(snapshot),
            world_state_digest=integrity.world_state_digest,
            self_state_digest=integrity.self_state_digest,
            focus_state_digest=integrity.focus_state_digest,
            outcome_state_digest=integrity.outcome_state_digest,
            state_evidence_refs=state_evidence_refs,
        )

    def _assert_lineage_matches_current(self, lineage: SituatedStateLineage) -> None:
        expected = self._lineage_for_current_state(
            parent_lineage_ref=lineage.parent_lineage_ref,
            state_evidence_refs=lineage.state_evidence_refs,
        )
        if expected != lineage:
            raise ValueError("situated lineage does not match the exact learner state")

    def _restore_successor_lineage(self) -> None:
        """Recover an observed child lineage or adopt one explicit legacy root."""

        if self._situated_context is None:
            raise RuntimeError("successor situated context is absent")
        head = self.transaction_store.head()
        if head is None:
            raise RuntimeError("successor store is not initialized")
        lineage = None
        if head.episode_ref is not None:
            record = self.transaction_store.prospective_turn_for_episode(
                head.episode_ref
            )
            if record is not None:
                episode_v2 = record.episode_v2
                if (
                    record.status != "RESOLVED"
                    or record.resolution_disposition != "OBSERVED"
                    or episode_v2 is None
                    or episode_v2.legacy_episode.episode_ref != head.episode_ref
                    or episode_v2.resolution.child_lineage is None
                ):
                    raise ValueError("current successor episode has no exact child lineage")
                lineage = episode_v2.resolution.child_lineage
        if lineage is None:
            lineage = self._lineage_for_current_state(
                parent_lineage_ref=None,
                state_evidence_refs=self._situated_context.evidence_refs,
            )
        if (
            lineage.context != self._situated_context
            or lineage.agent_ref != self.agent_ref
            or lineage.competence_state_digest != head.state_digest
            or lineage.snapshot_digest != head.snapshot_sha256
        ):
            raise ValueError("restored successor lineage drifted from fixed identity or head")
        self._assert_lineage_matches_current(lineage)
        self._current_lineage = lineage

    def _restore_parent(self, turn: PendingCognitiveTurn) -> None:
        self.learner.restore_state(turn.parent_state_snapshot)
        self._pending = None
        self._executing = False
        self._executed = None
        self._claimed_unknown = False

    @staticmethod
    def _assert_decision_matches_reservation(
        decision: AbilityDecision,
        reservation: ProspectiveTurnReservation,
    ) -> None:
        actual = (
            decision.task_id,
            decision.request,
            decision.proposals,
            decision.selected_index,
            decision.selected_trace,
            tuple(value.hex() for value in decision.logits),
            decision.evidence_ref,
            tuple(sorted(decision.supporting_evidence_refs)),
            decision.parent_state_digest,
        )
        expected = (
            reservation.task_id,
            reservation.request,
            reservation.proposals,
            reservation.selected_index,
            reservation.selected_trace,
            tuple(value.hex() for value in reservation.logits),
            reservation.decision_evidence_ref,
            reservation.supporting_evidence_refs,
            reservation.parent_competence_digest,
        )
        if actual != expected:
            raise ValueError("restored learner decision differs from the reservation")

    @staticmethod
    def _execution_from_receipt(
        receipt: CognitiveExecutionReceipt,
    ) -> CognitiveExecution:
        return CognitiveExecution(
            receipt.status,
            receipt.executed_trace,
            receipt.response,
            receipt.output_observations,
        )

    @staticmethod
    def _receipt_from_execution(
        request: CognitiveExecutionRequest,
        execution: CognitiveExecution,
    ) -> CognitiveExecutionReceipt:
        return CognitiveExecutionReceipt.from_request(
            request,
            status=execution.status,
            executed_trace=execution.executed_trace,
            response=execution.response,
            output_observations=execution.observations,
        )

    def _restore_active_reservation(self) -> None:
        """Rehydrate exactly one pending or executed turn from canonical bytes."""

        if self._successor_mode:
            self._restore_active_prospective_reservation()
            return

        record = self.transaction_store.active_reservation()
        if record is None:
            return
        head = self.transaction_store.head()
        snapshot = self.transaction_store.load_head_state()
        reservation = record.reservation
        if (
            head is None
            or not isinstance(snapshot, bytes)
            or reservation.parent_sequence != head.sequence
            or reservation.parent_event_ref != head.episode_ref
            or reservation.parent_competence_digest != head.state_digest
            or reservation.parent_snapshot_digest != head.snapshot_sha256
            or reservation.agent_ref != self.agent_ref
            or reservation.world_ref != self.world_ref
            or _snapshot_sha256(snapshot) != head.snapshot_sha256
        ):
            raise ValueError("active reservation does not bind the canonical head")
        if record.pending_blob is None:
            raise ValueError("active reservation has no pending learner state")
        reservation.assert_pending_blob(record.pending_blob)
        decision = self.learner.restore_pending_state(record.pending_blob)
        self._assert_decision_matches_reservation(decision, reservation)
        turn = PendingCognitiveTurn(
            decision=decision,
            recalled_refs=reservation.recalled_refs,
            commitment=reservation.commitment,
            parent_state_snapshot=snapshot,
            parent_head=head,
            reservation=reservation,
        )
        self._pending = turn
        self._executing = False
        self._executed = None
        self._claimed_unknown = record.status == "CLAIMED"
        if record.status == "RESERVED":
            if record.execution_request is not None or record.execution_receipt is not None:
                raise ValueError("RESERVED turn contains execution material")
            return
        if record.execution_request is None:
            raise ValueError("claimed reservation has no execution request")
        record.execution_request.assert_reservation(reservation)
        if record.status == "CLAIMED":
            if record.execution_receipt is not None:
                raise ValueError("CLAIMED turn already contains an execution receipt")
            return
        if record.status != "EXECUTION_RECORDED" or record.execution_receipt is None:
            raise ValueError("active reservation lifecycle is unsupported")
        record.execution_receipt.assert_request(record.execution_request)
        if record.execution_receipt.status != "COMPLETED":
            self.transaction_store.resolve_noncompletion(reservation.reservation_ref)
            self._restore_parent(turn)
            return
        execution = self._execution_from_receipt(record.execution_receipt)
        self._executed = ExecutedCognitiveTurn(
            turn,
            execution,
            record.execution_request.input_observations,
            record.execution_request,
            record.execution_receipt,
        )
        self._claimed_unknown = False

    def _restore_active_prospective_reservation(self) -> None:
        """Rehydrate one schema-v3 successor aggregate without projecting it."""

        record = self.transaction_store.active_prospective_turn()
        if record is None:
            return
        head = self.transaction_store.head()
        snapshot = self.transaction_store.load_head_state()
        wrapper = record.reservation
        reservation = wrapper.legacy_reservation
        if (
            head is None
            or not isinstance(snapshot, bytes)
            or self._current_lineage is None
            or wrapper.batch.parent_lineage != self._current_lineage
            or reservation.parent_sequence != head.sequence
            or reservation.parent_event_ref != head.episode_ref
            or reservation.parent_competence_digest != head.state_digest
            or reservation.parent_snapshot_digest != head.snapshot_sha256
            or reservation.agent_ref != self.agent_ref
            or reservation.world_ref != self.world_ref
            or _snapshot_sha256(snapshot) != head.snapshot_sha256
        ):
            raise ValueError("active successor reservation does not bind the canonical head")
        if record.pending_blob is None:
            raise ValueError("active successor reservation has no pending learner state")
        reservation.assert_pending_blob(record.pending_blob)
        decision = self.learner.restore_pending_state(record.pending_blob)
        self._assert_decision_matches_reservation(decision, reservation)
        material = getattr(self.learner, "pending_prospective_material", None)
        if material is None or getattr(material, "batch_ref", None) != wrapper.batch_ref:
            raise ValueError("restored prospective material does not bind the batch")
        turn = PendingCognitiveTurn(
            decision=decision,
            recalled_refs=reservation.recalled_refs,
            commitment=reservation.commitment,
            parent_state_snapshot=snapshot,
            parent_head=head,
            reservation=reservation,
            successor_reservation=wrapper,
            parent_lineage=self._current_lineage,
        )
        self._pending = turn
        self._executing = False
        self._executed = None
        self._claimed_unknown = record.status == "CLAIMED"
        if record.status == "RESERVED":
            if record.execution_request is not None or record.execution_receipt is not None:
                raise ValueError("RESERVED successor turn contains execution material")
            return
        if record.execution_request is None:
            raise ValueError("claimed successor reservation has no execution request")
        record.execution_request.assert_reservation(reservation)
        if record.status == "CLAIMED":
            if record.execution_receipt is not None:
                raise ValueError("CLAIMED successor turn already has an execution receipt")
            return
        if record.status != "EXECUTION_RECORDED" or record.execution_receipt is None:
            raise ValueError("active successor reservation lifecycle is unsupported")
        record.execution_receipt.assert_request(record.execution_request)
        execution = self._execution_from_receipt(record.execution_receipt)
        self._executed = ExecutedCognitiveTurn(
            turn,
            execution,
            record.execution_request.input_observations,
            record.execution_request,
            record.execution_receipt,
        )
        self._claimed_unknown = False
        if record.execution_receipt.status != "COMPLETED":
            self._resolve_successor_lifecycle(
                turn,
                disposition=ResolutionDisposition(record.execution_receipt.status),
                execution_request=record.execution_request,
                execution_receipt=record.execution_receipt,
            )
            self._restore_parent(turn)

    def _rehydrate_active_reservation(self, turn: PendingCognitiveTurn) -> None:
        self.learner.restore_state(turn.parent_state_snapshot)
        self._pending = None
        self._executing = False
        self._executed = None
        self._claimed_unknown = False
        self._restore_active_reservation()

    def _assert_memory_synchronized(self, head: object | None = None) -> None:
        if not self._typed_projection:
            return
        current = self.transaction_store.head() if head is None else head
        if current is None:
            raise RuntimeError("canonical cognitive store is not initialized")
        self.memory.assert_synchronized(current.sequence, current.episode_ref)

    @staticmethod
    def _tensor_row(value: object, index: int, label: str) -> tuple[float, ...]:
        try:
            row = value.detach().cpu()[index].reshape(-1).tolist()
        except Exception as exc:
            raise ValueError(f"{label} is not an aligned tensor") from exc
        result = tuple(float(item) for item in row)
        if not result or any(not math.isfinite(item) for item in result):
            raise ValueError(f"{label} must contain finite values")
        return result

    @staticmethod
    def _tensor_scalar(value: object, index: int, label: str) -> float:
        try:
            result = float(value.detach().cpu().reshape(-1)[index].item())
        except Exception as exc:
            raise ValueError(f"{label} is not an aligned tensor") from exc
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite")
        return result

    def _build_prospective_batch(
        self,
        *,
        task_id: str,
        request: str,
        decision: AbilityDecision,
        candidates: tuple[EncodedProcedureCandidate, ...],
        recalled_refs: tuple[str, ...],
        parent_head: object,
        acquisition_head: object,
        material: object,
    ) -> ProspectiveDynamicsBatch:
        if self._current_lineage is None or self._situated_context is None:
            raise RuntimeError("successor lineage is absent")
        context_refs = (self._situated_context.context_ref,)
        eligibility_rows = tuple((0,) for _ in candidates)
        budget = getattr(material, "resource_budget", None)
        prospective = getattr(material, "prospective_output", None)
        component = getattr(material, "component_state", None)
        if (
            getattr(material, "decision", None) != decision
            or tuple(getattr(material, "candidate_traces", ())) != decision.proposals
            or tuple(getattr(material, "recalled_refs", ())) != recalled_refs
            or tuple(getattr(material, "context_refs", ())) != context_refs
            or tuple(getattr(material, "eligibility_rows", ())) != eligibility_rows
            or getattr(material, "parent_competence_digest", None)
            != decision.parent_state_digest
            or component is None
            or getattr(material, "checkpoint_ref", None) != component.checkpoint_ref
            or getattr(material, "config_ref", None) != component.config_ref
            or component != self._component_integrity()
            or prospective is None
            or budget is None
        ):
            raise ValueError("pending prospective material drifted from the exact turn")
        support_rows = tuple(
            tuple(sorted(row.supporting_evidence_refs)) for row in candidates
        )
        if tuple(getattr(material, "candidate_supporting_evidence_refs", ())) != support_rows:
            raise ValueError("pending prospective support rows drifted")
        if getattr(budget, "world_reads", None) != 1 or getattr(
            budget, "recurrent_steps", None
        ) != 1 or getattr(budget, "branches", None) != len(candidates):
            raise ValueError("successor resource budget must be the one-context lane")
        hypothetical = SituatedContext(
            reality_mode=RealityMode.HYPOTHETICAL,
            perspective=self._situated_context.perspective,
            subject_ref=self._situated_context.subject_ref,
            scope_ref=self._situated_context.scope_ref,
            world_ref=self._situated_context.world_ref,
            evidence_refs=context_refs,
        )
        request_ref = _content_ref(request)
        branches = []
        for index, candidate in enumerate(candidates):
            focus_mask = self._tensor_row(
                prospective.focused_mask, index, "prospective focused mask"
            )
            full_weights = self._tensor_row(
                prospective.focus_weights, index, "prospective focus weights"
            )
            focused = tuple(
                position for position, present in enumerate(focus_mask) if bool(present)
            )
            weights = tuple(full_weights[position] for position in focused)
            branches.append(
                ProspectiveBranch(
                    parent_lineage_ref=self._current_lineage.lineage_ref,
                    task_id=task_id,
                    request_ref=request_ref,
                    candidate_index=index,
                    candidate_trace=candidate.trace,
                    context=hypothetical,
                    predicted_next_latent=self._tensor_row(
                        prospective.future_latents,
                        index,
                        "prospective future latent",
                    ),
                    outcome_logit=self._tensor_scalar(
                        prospective.outcome_logits, index, "prospective outcome logit"
                    ),
                    uncertainty=self._tensor_scalar(
                        prospective.uncertainties, index, "prospective uncertainty"
                    ),
                    selection_residual=self._tensor_scalar(
                        prospective.selection_residuals,
                        index,
                        "prospective selection residual",
                    ),
                    decision_logit=decision.logits[index],
                    horizon=1,
                    recurrent_steps=prospective.recurrent_steps,
                    focused_context_indices=focused,
                    focus_weights=weights,
                    supporting_evidence_refs=support_rows[index],
                )
            )
        latent_width = len(branches[0].predicted_next_latent)
        resources = ProspectiveResourceEnvelope(
            branch_capacity=budget.branches,
            read_capacity=budget.world_reads,
            latent_width=latent_width,
            recurrent_step_capacity=budget.recurrent_steps,
            horizon_capacity=1,
        )
        return ProspectiveDynamicsBatch(
            parent_lineage=self._current_lineage,
            parent_sequence=parent_head.sequence,
            parent_event_ref=parent_head.episode_ref,
            parent_acquisition_ref=acquisition_head.acquisition_ref,
            task_id=task_id,
            request=request,
            model_ref=self.model_ref,
            encoder_ref=self.encoder_ref,
            dynamics_checkpoint_ref=component.checkpoint_ref,
            dynamics_config_ref=component.config_ref,
            decision_evidence_ref=decision.evidence_ref,
            recalled_refs=recalled_refs,
            context_refs=context_refs,
            eligibility_rows=eligibility_rows,
            resources=resources,
            branches=tuple(branches),
            selected_branch_ref=branches[decision.selected_index].branch_ref,
        )

    async def begin_turn(
        self,
        task_id: str,
        request: str,
        *,
        world_time: int | None = None,
    ) -> PendingCognitiveTurn:
        if self._successor_mode:
            return await self._begin_successor_turn(
                task_id,
                request,
                world_time=world_time,
            )
        if (
            self._beginning
            or self._pending is not None
            or self._executing
            or self._executed is not None
            or self.learner.pending_decision is not None
        ):
            raise RuntimeError("one cognitive turn is already pending")
        task_id = _text(task_id, "task_id", 256)
        request = _text(request, "request", _MAX_TASK)
        parent_head = self.transaction_store.head()
        self._assert_memory_synchronized(parent_head)
        parent_snapshot = self.learner.capture_state()
        parent_digest = self.learner.state_digest()
        if (
            parent_head is None
            or parent_head.state_digest != parent_digest
            or parent_head.snapshot_sha256 != _snapshot_sha256(parent_snapshot)
        ):
            raise ValueError("learner state does not exactly match the canonical head")
        reservation_created = False
        self._beginning = True
        try:
            recall = await self.memory.recall(
                request, limit=self.recall_limit, world_time=world_time
            )
            raw = self.text_adapter.propose_procedure_traces(
                request,
                tuple(item.text for item in recall.items),
                count=self.proposal_count,
            )
            proposals = tuple(_text(item, "proposal", _MAX_TRACE) for item in raw)
            if (
                len(proposals) != self.proposal_count
                or len(set(proposals)) != self.proposal_count
            ):
                raise ValueError(
                    "the model must return the requested number of distinct traces"
                )
            candidates = tuple(
                self.relation_adapter.encode_candidates(request, proposals, recall)
            )
            if (
                len(candidates) != self.proposal_count
                or tuple(row.trace for row in candidates) != proposals
            ):
                raise ValueError("relation adapter output does not align with proposals")
            recalled_refs = {item.artifact_ref for item in recall.items}
            if any(
                not set(row.supporting_evidence_refs).issubset(recalled_refs)
                for row in candidates
            ):
                raise ValueError("candidate cites evidence outside validated recall")
            decision = self.learner.select(
                task_id=task_id, request=request, candidates=candidates
            )
            if (
                decision.task_id != task_id
                or decision.request != request
                or decision.proposals != proposals
                or type(decision.selected_index) is not int
                or not 0 <= decision.selected_index < self.proposal_count
                or decision.selected_trace != proposals[decision.selected_index]
                or not set(decision.supporting_evidence_refs).issubset(recalled_refs)
            ):
                raise ValueError("learner decision does not bind the public candidates")
            if (
                decision.parent_state_digest != parent_digest
                or self.learner.state_digest() != parent_digest
                or self.learner.capture_state() != parent_snapshot
            ):
                raise ValueError("decision was not scored from the captured parent state")
            if self.transaction_store.head() != parent_head:
                raise ValueError("canonical head changed while the decision was being scored")
            commitment = ProspectiveCommitment(
                parent_event_ref=parent_head.episode_ref,
                task_id=task_id,
                candidate_index=decision.selected_index,
                candidate_trace=decision.selected_trace,
                predicted_score=decision.logits[decision.selected_index],
                uncertainty=_uncertainty(decision.logits),
                horizon=1,
                competence_state_digest=parent_digest,
            )
            pending_blob = self.learner.capture_pending_state()
            recalled_refs = tuple(item.artifact_ref for item in recall.items)
            reservation = ProspectiveTurnReservation.create(
                pending_blob=pending_blob,
                parent_sequence=parent_head.sequence,
                parent_event_ref=parent_head.episode_ref,
                parent_competence_digest=parent_digest,
                parent_snapshot_digest=parent_head.snapshot_sha256,
                model_ref=self.model_ref,
                encoder_ref=self.encoder_ref,
                agent_ref=self.agent_ref,
                world_ref=self.world_ref,
                task_id=task_id,
                request=request,
                recalled_refs=recalled_refs,
                proposals=proposals,
                selected_index=decision.selected_index,
                selected_trace=decision.selected_trace,
                logits=tuple(decision.logits),
                decision_evidence_ref=decision.evidence_ref,
                supporting_evidence_refs=tuple(
                    sorted(decision.supporting_evidence_refs)
                ),
                commitment=commitment,
            )
            reserved = self.transaction_store.reserve_turn(
                reservation,
                pending_blob,
            )
            reservation_created = reserved.transitioned
            if not reserved.transitioned or reserved.record.reservation != reservation:
                raise ValueError("new cognitive turn did not create its exact reservation")
            turn = PendingCognitiveTurn(
                decision,
                recalled_refs,
                commitment,
                parent_snapshot,
                parent_head,
                reservation,
            )
            self._pending = turn
            return turn
        except Exception:
            self.learner.restore_state(parent_snapshot)
            self._pending = None
            if reservation_created:
                self._restore_active_reservation()
            raise
        finally:
            self._beginning = False

    async def _synchronize_acquisition_memory(self) -> tuple[str, ...]:
        """Drain or rebuild the disposable acquisition view before new recall."""

        try:
            acknowledged = await _await_if_needed(
                self.memory.retry_pending_projections(limit=64)
            )
            self.memory.assert_synchronized(self.transaction_store.acquisition_head())
            return tuple(acknowledged)
        except ValueError:
            # A fresh disposable view can be behind already-acknowledged canonical
            # history. Rebuild from the bounded source rather than inventing gaps.
            await _await_if_needed(self.memory.rebuild(limit=64))
        acknowledged = await _await_if_needed(
            self.memory.retry_pending_projections(limit=64)
        )
        self.memory.assert_synchronized(self.transaction_store.acquisition_head())
        return tuple(acknowledged)

    async def _begin_successor_turn(
        self,
        task_id: str,
        request: str,
        *,
        world_time: int | None,
    ) -> PendingCognitiveTurn:
        if (
            self._beginning
            or self._pending is not None
            or self._executing
            or self._executed is not None
            or self.learner.pending_decision is not None
        ):
            raise RuntimeError("one cognitive turn is already pending")
        task_id = _text(task_id, "task_id", 256)
        request = _text(request, "request", _MAX_TASK)
        await self._synchronize_acquisition_memory()
        parent_head = self.transaction_store.head()
        acquisition_head = self.transaction_store.acquisition_head()
        parent_snapshot = self.learner.capture_state()
        parent_digest = self.learner.state_digest()
        if (
            parent_head is None
            or self._current_lineage is None
            or parent_head.state_digest != parent_digest
            or parent_head.snapshot_sha256 != _snapshot_sha256(parent_snapshot)
            or self._current_lineage.competence_state_digest != parent_digest
            or self._current_lineage.snapshot_digest != parent_head.snapshot_sha256
        ):
            raise ValueError("successor learner and lineage do not match the canonical head")
        self._assert_lineage_matches_current(self._current_lineage)
        self._beginning = True
        try:
            if self.removal_condition == "BACKEND_REMOVAL":
                recall = self.memory.recall_from_hits(
                    self._frozen_recall_hits,
                    limit=self.recall_limit,
                    world_time=world_time,
                    frozen_origin=False,
                )
            elif self._frozen_recall_hits:
                live_hits = tuple(
                    await _await_if_needed(
                        self.memory.backend.search(
                            request,
                            limit=self.recall_limit,
                        )
                    )
                )
                if live_hits != self._frozen_recall_hits:
                    raise ValueError(
                        "live reference hits differ from the frozen evaluation manifest"
                    )
                recall = self.memory.recall_from_hits(
                    self._frozen_recall_hits,
                    limit=self.recall_limit,
                    world_time=world_time,
                    frozen_origin=self.removal_condition == "FROZEN_ORIGIN",
                )
            else:
                recall = await self.memory.recall(
                    request,
                    limit=self.recall_limit,
                    world_time=world_time,
                    frozen_origin=self.removal_condition == "FROZEN_ORIGIN",
                )
            recalled_items = tuple(recall.items)
            if not recalled_items:
                raise ValueError("successor recall has no admissible observed evidence")
            if any(
                item.epistemic_status not in ("OBSERVED", "VALIDATED")
                for item in recalled_items
            ):
                raise ValueError("successor recall contains non-admissible evidence")
            recalled_refs = tuple(item.artifact_ref for item in recalled_items)
            if len(set(recalled_refs)) != len(recalled_refs):
                raise ValueError("successor recall contains duplicate evidence")
            raw = self.text_adapter.propose_procedure_traces(
                request,
                tuple(item.text for item in recalled_items),
                count=self.proposal_count,
            )
            proposals = tuple(_text(item, "proposal", _MAX_TRACE) for item in raw)
            if (
                len(proposals) != self.proposal_count
                or len(set(proposals)) != self.proposal_count
            ):
                raise ValueError(
                    "the model must return the requested number of distinct traces"
                )
            candidates = tuple(
                self.relation_adapter.encode_candidates(request, proposals, recall)
            )
            if (
                len(candidates) != self.proposal_count
                or tuple(row.trace for row in candidates) != proposals
            ):
                raise ValueError("relation adapter output does not align with proposals")
            recalled_set = set(recalled_refs)
            for row in candidates:
                if (
                    not row.supporting_evidence_refs
                    or row.supporting_evidence_refs
                    != tuple(sorted(row.supporting_evidence_refs))
                    or not set(row.supporting_evidence_refs).issubset(recalled_set)
                ):
                    raise ValueError(
                        "every prospective candidate requires canonical recalled support"
                    )
            context_refs = (self._situated_context.context_ref,)  # type: ignore[union-attr]
            eligibility_rows = tuple((0,) for _ in candidates)
            resource_budget = ProspectiveResourceBudget(
                world_reads=1,
                branches=len(candidates),
                recurrent_steps=1,
            )
            decision = self.learner.select_prospective(  # type: ignore[attr-defined]
                task_id=task_id,
                request=request,
                candidates=candidates,
                recalled_refs=recalled_refs,
                context_refs=context_refs,
                eligibility_rows=eligibility_rows,
                resource_budget=resource_budget,
            )
            if (
                decision.task_id != task_id
                or decision.request != request
                or decision.proposals != proposals
                or type(decision.selected_index) is not int
                or not 0 <= decision.selected_index < self.proposal_count
                or decision.selected_trace != proposals[decision.selected_index]
                or not set(decision.supporting_evidence_refs).issubset(recalled_set)
                or decision.parent_state_digest != parent_digest
                or self.learner.state_digest() != parent_digest
                or self.learner.capture_state() != parent_snapshot
            ):
                raise ValueError("prospective decision does not bind the exact parent")
            material = getattr(self.learner, "pending_prospective_material", None)
            if material is None:
                raise ValueError("successor learner did not retain prospective material")
            batch = self._build_prospective_batch(
                task_id=task_id,
                request=request,
                decision=decision,
                candidates=candidates,
                recalled_refs=recalled_refs,
                parent_head=parent_head,
                acquisition_head=acquisition_head,
                material=material,
            )
            selected_branch = batch.selected_branch
            commitment = ProspectiveCommitment(
                parent_event_ref=parent_head.episode_ref,
                task_id=task_id,
                candidate_index=decision.selected_index,
                candidate_trace=decision.selected_trace,
                predicted_score=decision.logits[decision.selected_index],
                uncertainty=selected_branch.uncertainty,
                horizon=selected_branch.horizon,
                competence_state_digest=parent_digest,
            )
            bound = self.learner.bind_pending_prospective_batch(  # type: ignore[attr-defined]
                batch.batch_ref
            )
            if getattr(bound, "batch_ref", None) != batch.batch_ref:
                raise ValueError("learner did not bind the exact prospective batch")
            pending_blob = self.learner.capture_pending_state()
            legacy = ProspectiveTurnReservation.create(
                pending_blob=pending_blob,
                parent_sequence=parent_head.sequence,
                parent_event_ref=parent_head.episode_ref,
                parent_competence_digest=parent_digest,
                parent_snapshot_digest=parent_head.snapshot_sha256,
                model_ref=self.model_ref,
                encoder_ref=self.encoder_ref,
                agent_ref=self.agent_ref,
                world_ref=self.world_ref,
                task_id=task_id,
                request=request,
                recalled_refs=recalled_refs,
                proposals=proposals,
                selected_index=decision.selected_index,
                selected_trace=decision.selected_trace,
                logits=decision.logits,
                decision_evidence_ref=decision.evidence_ref,
                supporting_evidence_refs=tuple(
                    sorted(decision.supporting_evidence_refs)
                ),
                commitment=commitment,
            )
            wrapper = ProspectiveTurnReservationV2(
                legacy_reservation=legacy,
                batch=batch,
            )
            from angler.memory.cognitive_acquisition_graph import (
                prospective_batch_record,
            )

            record = prospective_batch_record(batch, acquisition_head.next_ordinal)
            acquisition = CognitiveAcquisition.from_source(
                batch,
                ordinal=acquisition_head.next_ordinal,
                predecessor_acquisition_ref=acquisition_head.acquisition_ref,
                record=record,
            )
            if (
                self.transaction_store.head() != parent_head
                or self.transaction_store.acquisition_head() != acquisition_head
            ):
                raise ValueError("canonical head changed while scoring the successor turn")
            reserved = self.transaction_store.reserve_prospective_turn(
                wrapper,
                acquisition,
                pending_blob,
            )
            if not reserved.transitioned or reserved.record.reservation != wrapper:
                raise ValueError("new successor turn did not create its exact reservation")
            turn = PendingCognitiveTurn(
                decision=decision,
                recalled_refs=recalled_refs,
                commitment=commitment,
                parent_state_snapshot=parent_snapshot,
                parent_head=parent_head,
                reservation=legacy,
                successor_reservation=wrapper,
                parent_lineage=self._current_lineage,
            )
            self._pending = turn
            return turn
        except Exception:
            self.learner.restore_state(parent_snapshot)
            self._pending = None
            # Another cycle may have atomically won the same canonical store
            # while this instance was scoring.  Always rehydrate that winner;
            # leaving the local learner apparently idle would permit an unsafe
            # second begin against an already-active external-effect key.
            if self.transaction_store.active_prospective_turn() is not None:
                self._restore_active_prospective_reservation()
            raise
        finally:
            self._beginning = False

    async def execute_turn(
        self,
        turn: PendingCognitiveTurn,
        *,
        observations: Sequence[str] = (),
    ) -> ExecutedCognitiveTurn:
        if (
            turn is not self._pending
            or self._executing
            or self._executed is not None
            or self.learner.pending_decision != turn.decision
        ):
            raise ValueError("turn is not the currently pending cognitive turn")
        if self._claimed_unknown:
            raise ExecutionReconciliationRequired(turn.reservation.reservation_ref)
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise TypeError("observations must be a sequence of complete text items")
        bounded = tuple(_text(item, "observation", _MAX_OBSERVATION) for item in observations)
        if (
            self.transaction_store.head() != turn.parent_head
            or self.learner.state_digest() != turn.parent_head.state_digest
            or self.learner.capture_state() != turn.parent_state_snapshot
        ):
            if self._successor_mode:
                self._resolve_successor_lifecycle(
                    turn,
                    disposition=ResolutionDisposition.CANCELLED,
                )
            else:
                cancelled = self.transaction_store.cancel_reservation(
                    turn.reservation.reservation_ref
                )
                if (
                    cancelled.record.status != "RESOLVED"
                    or cancelled.record.resolution_disposition != "CANCELLED"
                ):
                    raise RuntimeError("unclaimed turn was not durably cancelled")
            self._restore_parent(turn)
            raise ValueError("canonical head changed before procedure execution")
        request = CognitiveExecutionRequest.from_reservation(
            turn.reservation,
            input_observations=bounded,
        )
        claimed = (
            self.transaction_store.claim_prospective_turn(request)
            if self._successor_mode
            else self.transaction_store.claim_turn(request)
        )
        if not claimed.transitioned:
            self._claimed_unknown = True
            raise ExecutionReconciliationRequired(turn.reservation.reservation_ref)
        self._executing = True
        try:
            result = await _await_if_needed(self.executor.execute(request))
            return self._record_execution_result(turn, request, result)
        except Exception as exc:
            self._executing = False
            self._claimed_unknown = True
            if isinstance(exc, ExecutionReconciliationRequired):
                raise
            raise ExecutionReconciliationRequired(
                turn.reservation.reservation_ref
            ) from exc

    def _resolve_successor_lifecycle(
        self,
        turn: PendingCognitiveTurn,
        *,
        disposition: ResolutionDisposition,
        execution_request: CognitiveExecutionRequest | None = None,
        execution_receipt: CognitiveExecutionReceipt | None = None,
    ) -> CognitiveResolutionResult:
        wrapper = turn.successor_reservation
        if not self._successor_mode or wrapper is None:
            raise RuntimeError("successor lifecycle requires a successor turn")
        evidence_refs = ()
        if execution_request is not None and execution_receipt is not None:
            evidence_refs = tuple(
                sorted(
                    (
                        execution_request.execution_request_ref,
                        execution_receipt.execution_receipt_ref,
                    )
                )
            )
        resolution = ProspectiveResolution.lifecycle(
            disposition=disposition,
            reservation=wrapper,
            execution_request=execution_request,
            execution_receipt=execution_receipt,
            observation_evidence_refs=evidence_refs,
        )
        from angler.memory.cognitive_acquisition_graph import (
            prospective_resolution_record,
        )

        head = self.transaction_store.acquisition_head()
        record = prospective_resolution_record(resolution, head.next_ordinal)
        acquisition = CognitiveAcquisition.from_source(
            resolution,
            ordinal=head.next_ordinal,
            predecessor_acquisition_ref=head.acquisition_ref,
            record=record,
        )
        transitioned = self.transaction_store.resolve_prospective_turn(
            resolution,
            acquisition,
        )
        if (
            not transitioned.transitioned
            or transitioned.record.resolution != resolution
            or transitioned.record.resolution_acquisition_ref
            != acquisition.acquisition_ref
        ):
            raise ValueError("successor lifecycle did not commit exact resolution bytes")
        return CognitiveResolutionResult(
            resolution_ref=resolution.resolution_ref,
            acquisition_ref=acquisition.acquisition_ref,
            projection_pending=True,
        )

    async def _project_successor_views(self) -> bool:
        try:
            await self._synchronize_acquisition_memory()
            return bool(self.transaction_store.pending_acquisition_projections(limit=1))
        except Exception:
            return True

    async def cancel_pending_turn(
        self, turn: PendingCognitiveTurn
    ) -> CognitiveResolutionResult:
        """Explicitly close one unclaimed successor turn without an outcome."""

        if (
            not self._successor_mode
            or turn is not self._pending
            or self._executing
            or self._executed is not None
            or self._claimed_unknown
        ):
            raise ValueError("only the current unclaimed successor turn can be cancelled")
        result = self._resolve_successor_lifecycle(
            turn,
            disposition=ResolutionDisposition.CANCELLED,
        )
        self._restore_parent(turn)
        pending = await self._project_successor_views()
        return CognitiveResolutionResult(
            result.resolution_ref,
            result.acquisition_ref,
            pending,
        )

    async def resolve_completed_unevaluated(
        self,
        completed: ExecutedCognitiveTurn,
        *,
        reservation_ref: str,
    ) -> CognitiveResolutionResult:
        """Close a completed effect only when the caller declares no feedback exists."""

        turn = completed.turn
        if (
            not self._successor_mode
            or turn is not self._pending
            or completed is not self._executed
            or completed.execution.status != "COMPLETED"
            or reservation_ref != turn.reservation.reservation_ref
        ):
            raise ValueError("completed successor execution is not awaiting disposition")
        record = self.transaction_store.active_prospective_turn()
        if record is None or record.feedback is not None:
            raise ValueError("objective feedback is already staged or the turn is absent")
        result = self._resolve_successor_lifecycle(
            turn,
            disposition=ResolutionDisposition.COMPLETED_UNEVALUATED,
            execution_request=completed.execution_request,
            execution_receipt=completed.execution_receipt,
        )
        self._restore_parent(turn)
        pending = await self._project_successor_views()
        return CognitiveResolutionResult(
            result.resolution_ref,
            result.acquisition_ref,
            pending,
        )

    def _record_execution_result(
        self,
        turn: PendingCognitiveTurn,
        request: CognitiveExecutionRequest,
        result: object,
    ) -> ExecutedCognitiveTurn:
        if not isinstance(result, CognitiveExecution):
            raise TypeError("executor must return CognitiveExecution")
        receipt = self._receipt_from_execution(request, result)
        return self._record_execution_receipt(turn, request, receipt)

    def _record_execution_receipt(
        self,
        turn: PendingCognitiveTurn,
        request: CognitiveExecutionRequest,
        receipt: CognitiveExecutionReceipt,
    ) -> ExecutedCognitiveTurn:
        if not isinstance(receipt, CognitiveExecutionReceipt):
            raise TypeError("executor recovery must supply CognitiveExecutionReceipt")
        receipt.assert_request(request)
        result = self._execution_from_receipt(receipt)
        recorded = (
            self.transaction_store.record_prospective_execution(receipt)
            if self._successor_mode
            else self.transaction_store.record_execution(receipt)
        )
        if not recorded.transitioned:
            existing = recorded.record.execution_receipt
            if existing != receipt:
                raise ValueError("execution recovery differs from the durable receipt")
        completed = ExecutedCognitiveTurn(
            turn,
            result,
            request.input_observations,
            request,
            receipt,
        )
        self._executing = False
        self._claimed_unknown = False
        if result.status != "COMPLETED":
            if self._successor_mode:
                self._resolve_successor_lifecycle(
                    turn,
                    disposition=ResolutionDisposition(result.status),
                    execution_request=request,
                    execution_receipt=receipt,
                )
            else:
                self.transaction_store.resolve_noncompletion(
                    turn.reservation.reservation_ref
                )
            self._restore_parent(turn)
            return completed
        self._executed = completed
        return completed

    async def recover_claimed_execution(self) -> ExecutedCognitiveTurn:
        """Recover an exact claim without guessing whether its effect happened."""

        if self._executing:
            raise ValueError("claimed execution recovery is already in progress")
        record = (
            self.transaction_store.active_prospective_turn()
            if self._successor_mode
            else self.transaction_store.active_reservation()
        )
        if (
            record is None
            or record.status != "CLAIMED"
            or record.execution_request is None
            or self._pending is None
        ):
            raise RuntimeError("no claimed execution is awaiting recovery")
        request = record.execution_request
        reservation = (
            record.reservation.legacy_reservation
            if self._successor_mode
            else record.reservation
        )
        request.assert_reservation(reservation)
        self._executing = True
        try:
            recovery = await _await_if_needed(self.executor.recover(request))
            if not isinstance(recovery, CognitiveExecutionRecovery):
                raise TypeError("executor recovery must return CognitiveExecutionRecovery")
            if recovery.status in ("IN_PROGRESS", "UNKNOWN"):
                raise ExecutionReconciliationRequired(reservation.reservation_ref)
            if recovery.status == "RECORDED":
                return self._record_execution_receipt(
                    self._pending,
                    request,
                    recovery.receipt,
                )
            else:
                result = await _await_if_needed(self.executor.execute(request))
            return self._record_execution_result(self._pending, request, result)
        except Exception as exc:
            self._executing = False
            self._claimed_unknown = True
            if isinstance(exc, ExecutionReconciliationRequired):
                raise
            raise ExecutionReconciliationRequired(reservation.reservation_ref) from exc

    def _build_episode(
        self,
        completed: ExecutedCognitiveTurn,
        *,
        outcome: Literal["success", "failure"],
        feedback_text: str,
        feedback_source_ref: str,
        child_state_digest: str,
    ) -> CognitiveEpisode:
        turn = completed.turn
        decision = turn.decision
        return CognitiveEpisode(
            task_id=decision.task_id,
            request=decision.request,
            recalled_refs=turn.recalled_refs,
            proposals=decision.proposals,
            selected_index=decision.selected_index,
            commitment=turn.commitment,
            response=completed.execution.response,
            observations=(*completed.input_observations, *completed.execution.observations),
            outcome=outcome,
            feedback_text=feedback_text,
            feedback_source_ref=feedback_source_ref,
            parent_state_digest=decision.parent_state_digest,
            child_state_digest=child_state_digest,
            model_ref=self.model_ref,
            encoder_ref=self.encoder_ref,
            supporting_evidence_refs=tuple(sorted(decision.supporting_evidence_refs)),
            visibility="LEARNER_VISIBLE",
        )

    @staticmethod
    def _projection(episode: CognitiveEpisode) -> MemoryProjection:
        text = (
            f"Task: {episode.request}\n"
            f"Selected procedure: {episode.commitment.candidate_trace}\n"
            f"Response: {episode.response}\n"
            f"Objective outcome: {episode.outcome}\n"
            f"Feedback: {episode.feedback_text}"
        )[:4_096].strip()
        return MemoryProjection.from_mapping(
            artifact_ref=episode.episode_ref,
            text=text,
            source_ref=episode.feedback_source_ref,
            visibility=episode.visibility,
            context={
                "child_state_digest": episode.child_state_digest,
                "commitment_ref": episode.commitment.commitment_ref,
                "encoder_ref": episode.encoder_ref,
                "model_ref": episode.model_ref,
                "outcome": episode.outcome,
                "parent_state_digest": episode.parent_state_digest,
            },
        )

    def _projection_input(self, episode: CognitiveEpisode) -> object:
        if self._typed_projection:
            item = self.transaction_store.get_episode_item(episode.episode_ref)
            if item.episode != episode:
                raise ValueError("canonical episode item differs from the committed episode")
            return item
        return self._projection(episode)

    async def _project_committed_input(self, value: object) -> None:
        """Send one already-verified value to the rebuildable memory lane."""

        if self._typed_projection:
            await _await_if_needed(self.memory.remember_episode_item(value))
        else:
            await _await_if_needed(self.memory.remember(value))

    async def record_outcome(
        self,
        completed: ExecutedCognitiveTurn,
        *,
        reservation_ref: str,
        task_id: str,
        outcome: Literal["success", "failure"],
        feedback_text: str,
        feedback_source_ref: str,
    ) -> CognitiveCommitResult:
        turn = completed.turn
        if (
            turn is not self._pending
            or completed is not self._executed
            or self._executing
            or completed.execution.status != "COMPLETED"
        ):
            raise ValueError("completed execution is not awaiting objective feedback")
        if (
            reservation_ref != turn.reservation.reservation_ref
            or task_id != turn.decision.task_id
        ):
            raise ValueError("feedback identity does not match the exact reservation")
        if outcome not in ("success", "failure"):
            raise ValueError("outcome must be objective success or failure")
        feedback_text = _text(feedback_text, "feedback_text", _MAX_FEEDBACK)
        feedback_source_ref = _text(feedback_source_ref, "feedback_source_ref", 512)
        _digest(feedback_source_ref, "feedback_source_ref")
        if (
            self.transaction_store.head() != turn.parent_head
            or self.learner.state_digest() != turn.parent_head.state_digest
            or self.learner.capture_state() != turn.parent_state_snapshot
        ):
            raise ValueError("canonical head changed before objective feedback")
        feedback = ObjectiveFeedbackRecord.from_execution(
            turn.reservation,
            completed.execution_request,
            completed.execution_receipt,
            outcome=outcome,
            feedback_text=feedback_text,
            feedback_source_ref=feedback_source_ref,
        )
        observed_encoding = None
        if self._successor_mode:
            wrapper = turn.successor_reservation
            if wrapper is None or self._observation_encoder is None:
                raise RuntimeError("successor observation boundary is absent")
            observed_encoding = self._observation_encoder.encode(
                completed.execution_request,
                completed.execution_receipt,
                latent_width=wrapper.batch.resources.latent_width,
            )
            observed_encoding.assert_context(
                completed.execution_request,
                completed.execution_receipt,
                encoder_ref=self.encoder_ref,
                latent_width=wrapper.batch.resources.latent_width,
            )
        staged = (
            self.transaction_store.stage_prospective_feedback(feedback)
            if self._successor_mode
            else self.transaction_store.stage_feedback(feedback)
        )
        if staged.record.feedback != feedback:
            raise ValueError("staged objective feedback differs from the exact record")
        return await self._commit_staged_feedback(
            completed,
            feedback,
            observed_encoding=observed_encoding,
        )

    async def resume_staged_outcome(self) -> CognitiveCommitResult:
        """Finish one already-recorded feedback update after a restart/failure."""

        record = (
            self.transaction_store.active_prospective_turn()
            if self._successor_mode
            else self.transaction_store.active_reservation()
        )
        if (
            record is None
            or record.status != "EXECUTION_RECORDED"
            or record.feedback is None
            or self._executed is None
        ):
            raise RuntimeError("no staged objective feedback is awaiting resolution")
        observed_encoding = None
        if self._successor_mode:
            if self._observation_encoder is None:
                raise RuntimeError("successor observation boundary is absent")
            wrapper = record.reservation
            observed_encoding = self._observation_encoder.encode(
                self._executed.execution_request,
                self._executed.execution_receipt,
                latent_width=wrapper.batch.resources.latent_width,
            )
            observed_encoding.assert_context(
                self._executed.execution_request,
                self._executed.execution_receipt,
                encoder_ref=self.encoder_ref,
                latent_width=wrapper.batch.resources.latent_width,
            )
        return await self._commit_staged_feedback(
            self._executed,
            record.feedback,
            observed_encoding=observed_encoding,
        )

    async def _commit_staged_feedback(
        self,
        completed: ExecutedCognitiveTurn,
        feedback: ObjectiveFeedbackRecord,
        *,
        observed_encoding: object | None = None,
    ) -> CognitiveCommitResult:
        if self._successor_mode:
            return await self._commit_staged_prospective_feedback(
                completed,
                feedback,
                observed_encoding=observed_encoding,
            )
        turn = completed.turn
        feedback.assert_context(
            turn.reservation,
            completed.execution_request,
            completed.execution_receipt,
        )
        try:
            # The present DurableAbilityLearner consumes only the objective
            # disposition and verifier identity.  Rich feedback text and the
            # full public trajectory stay in the canonical episode for a
            # successor learner rather than entering through a hidden channel.
            self.learner.apply_outcome(
                OutcomeContext(
                    turn.decision.task_id,
                    turn.decision.request,
                    (),
                    feedback.outcome,
                    feedback.feedback_source_ref,
                )
            )
            if self.learner.pending_decision is not None:
                raise ValueError("learner retained a pending decision after objective feedback")
            child_digest = self.learner.state_digest()
            if child_digest == turn.decision.parent_state_digest:
                raise ValueError("objective feedback did not advance competence state")
            child_snapshot = self.learner.capture_state()
            episode = self._build_episode(
                completed,
                outcome=feedback.outcome,
                feedback_text=feedback.feedback_text,
                feedback_source_ref=feedback.feedback_source_ref,
                child_state_digest=child_digest,
            )
            committed = self.transaction_store.commit_reserved_episode(
                turn.reservation.reservation_ref,
                episode,
                child_snapshot,
                expected_parent_digest=turn.decision.parent_state_digest,
            )
        except Exception:
            self._rehydrate_active_reservation(turn)
            raise

        # Canonical state now owns the child.  Projection failure must not roll
        # competence back beneath the committed episode.
        self._pending = None
        self._executed = None
        # Canonical reload/identity verification is a store-integrity boundary,
        # not a disposable projection operation.  It must surface if corrupt.
        projection_input = self._projection_input(episode)
        projection_pending = True
        try:
            await self._project_committed_input(projection_input)
            self.transaction_store.ack_projection(episode.episode_ref)
            projection_pending = False
        except Exception:
            projection_pending = True
        return CognitiveCommitResult(
            episode_ref=committed.episode_ref,
            sequence=committed.sequence,
            head=committed.head,
            projection_pending=projection_pending,
        )

    async def _commit_staged_prospective_feedback(
        self,
        completed: ExecutedCognitiveTurn,
        feedback: ObjectiveFeedbackRecord,
        *,
        observed_encoding: object | None,
    ) -> CognitiveCommitResult:
        turn = completed.turn
        wrapper = turn.successor_reservation
        parent_lineage = turn.parent_lineage
        if (
            wrapper is None
            or parent_lineage is None
            or observed_encoding is None
            or self._observation_encoder is None
        ):
            raise RuntimeError("observed successor commit lacks its exact material")
        feedback.assert_context(
            turn.reservation,
            completed.execution_request,
            completed.execution_receipt,
        )
        observed_encoding.assert_context(
            completed.execution_request,
            completed.execution_receipt,
            encoder_ref=self.encoder_ref,
            latent_width=wrapper.batch.resources.latent_width,
        )
        if (
            self.transaction_store.head() != turn.parent_head
            or self.learner.state_digest() != turn.parent_head.state_digest
            or self.learner.capture_state() != turn.parent_state_snapshot
            or self._current_lineage != parent_lineage
        ):
            raise ValueError("canonical head changed before successor feedback")
        try:
            self.learner.apply_outcome(
                OutcomeContext(
                    turn.decision.task_id,
                    turn.decision.request,
                    (),
                    feedback.outcome,
                    feedback.feedback_source_ref,
                )
            )
            if self.learner.pending_decision is not None:
                raise ValueError(
                    "learner retained a pending decision after successor feedback"
                )
            child_digest = self.learner.state_digest()
            if child_digest == turn.decision.parent_state_digest:
                raise ValueError("successor feedback did not advance competence state")
            child_snapshot = self.learner.capture_state()
            state_evidence_refs = tuple(
                sorted(
                    {
                        *observed_encoding.observation_evidence_refs,
                        feedback.feedback_ref,
                    }
                )
            )
            child_lineage = self._lineage_for_current_state(
                parent_lineage_ref=parent_lineage.lineage_ref,
                state_evidence_refs=state_evidence_refs,
            )
            parent_lineage.assert_successor(child_lineage)
            resolution = ProspectiveResolution.observed(
                reservation=wrapper,
                execution_request=completed.execution_request,
                execution_receipt=completed.execution_receipt,
                objective_feedback=feedback,
                child_lineage=child_lineage,
                observed_next_latent=observed_encoding.observed_next_latent,
                observation_evidence_refs=(
                    observed_encoding.observation_evidence_refs
                ),
            )
            episode = self._build_episode(
                completed,
                outcome=feedback.outcome,
                feedback_text=feedback.feedback_text,
                feedback_source_ref=feedback.feedback_source_ref,
                child_state_digest=child_digest,
            )
            episode_v2 = CognitiveEpisodeV2(
                legacy_episode=episode,
                resolution=resolution,
            )
            from angler.memory.cognitive_acquisition_graph import (
                prospective_resolution_record,
            )

            acquisition_head = self.transaction_store.acquisition_head()
            record = prospective_resolution_record(
                resolution,
                acquisition_head.next_ordinal,
            )
            acquisition = CognitiveAcquisition.from_source(
                resolution,
                ordinal=acquisition_head.next_ordinal,
                predecessor_acquisition_ref=acquisition_head.acquisition_ref,
                record=record,
            )
            committed = self.transaction_store.commit_observed_prospective_turn(
                resolution,
                episode_v2,
                acquisition,
                child_snapshot,
                turn.decision.parent_state_digest,
            )
        except Exception:
            self._rehydrate_active_reservation(turn)
            raise

        self._current_lineage = child_lineage
        self._pending = None
        self._executed = None
        projection_pending = await self._project_successor_views()
        return CognitiveCommitResult(
            episode_ref=committed.episode_ref,
            sequence=committed.sequence,
            head=committed.head,
            projection_pending=projection_pending,
        )

    async def retry_pending_projections(self, limit: int = 64) -> tuple[str, ...]:
        """Retry one bounded projection-outbox page in canonical order."""

        if self._successor_mode:
            if limit == 64:
                return await self._synchronize_acquisition_memory()
            acknowledged = await _await_if_needed(
                self.memory.retry_pending_projections(limit=limit)
            )
            if not self.transaction_store.pending_acquisition_projections(limit=1):
                self.memory.assert_synchronized(
                    self.transaction_store.acquisition_head()
                )
            return tuple(acknowledged)

        acknowledged: list[str] = []
        for item in self.transaction_store.pending_projections(limit=limit):
            _sequence, episode_ref, episode = _outbox_fields(item)
            if self._typed_projection:
                await _await_if_needed(
                    self.memory.remember_episode_item(
                        self.transaction_store.get_episode_item(episode_ref)
                    )
                )
            else:
                projection = self._projection(episode)
                try:
                    self.memory.origin.anchor(episode_ref)
                except KeyError:
                    await self.memory.remember(projection)
                else:
                    await self.memory.reproject(projection)
            self.transaction_store.ack_projection(episode_ref)
            acknowledged.append(episode_ref)
        return tuple(acknowledged)


__all__ = [
    "CognitiveCommitResult",
    "CognitiveCycle",
    "CognitiveExecution",
    "CognitiveExecutionRecovery",
    "CognitiveExecutor",
    "CognitiveResolutionResult",
    "CognitiveStateOwner",
    "ExecutionReconciliationRequired",
    "ExecutedCognitiveTurn",
    "PendingCognitiveTurn",
    "ProspectiveCognitiveStateOwner",
]
