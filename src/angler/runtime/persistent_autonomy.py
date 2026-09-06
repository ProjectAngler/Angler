"""Crash-recoverable mechanics for Jenny 2.0's one cognitive life loop.

This module owns scheduling, idempotency, permission enforcement, recovery, and
atomic state commits.  It deliberately owns no curiosity, drive, goal,
procedure, affect, or task policy.  Those decisions arrive through one injected
learned-cycle boundary and remain shadow-only unless explicitly qualified.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Literal, Protocol

from .affective_substrate import (
    EventSubstrateObservation,
    project_canonical_episode,
)
from .jenny2_activity import ActivityPhase, Jenny2ActivitySink
from .jenny_genesis import JennyGenesis
from .latency_trace import latency_phase
from .temporal_v2 import TemporalNow, TemporalV2, TrustedClock, semantic_age


SUPERVISOR_STATE_CONTRACT = "ANG-CTR-PERSISTENT-AUTONOMY-STATE-001@0.1.0"
_SCHEMA_VERSION = 2
_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._:/-][a-z0-9]+)*$")
_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_DISPOSITIONS = frozenset(("ACT", "ASK", "WAIT", "STOP"))
CONSEQUENCE_NAMES = (
    "objective_progress",
    "constraint_satisfaction",
    "prediction_error",
    "information_gain",
    "evidence_quality",
    "reuse_value",
    "cost",
    "safety",
    "human_feedback",
)

# A choice is the transaction envelope for several independently bounded
# records.  Its ceiling must therefore be an aggregate ceiling, not the same
# 64 KiB ceiling used by one nested autonomous-target formation input.  Keeping
# this contract here also guarantees that the reserved AffordanceRequest and
# the committed CycleChoice accept exactly the same byte-identical envelope.
CHOICE_CONTEXT_MAX_CHARACTERS = 4 * 65_536


def _validated_mapped_consequence(
    value: object,
) -> tuple[tuple[str, float], ...]:
    if type(value) is not tuple:
        raise TypeError("consequence mapper must return a tuple")
    try:
        names = tuple(item[0] for item in value)
        normalized = tuple((item[0], float(item[1])) for item in value)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("consequence mapper returned malformed values") from exc
    if names != CONSEQUENCE_NAMES:
        raise ValueError("mapped consequence must contain the canonical fields in order")
    if any(not math.isfinite(score) or not -1.0 <= score <= 1.0 for _, score in normalized):
        raise ValueError("mapped consequence values must be finite in [-1, 1]")
    return normalized


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _bounded(value: str, label: str, maximum: int, *, empty: bool = False) -> str:
    if type(value) is not str or len(value) > maximum or (not empty and not value.strip()):
        raise ValueError(f"{label} must be bounded text")
    return value


def _identifier(value: str, label: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError(f"{label} must be a stable lowercase identifier")
    return value


def _reference(value: str, label: str) -> str:
    if type(value) is not str or not _REF.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 reference")
    return value


def _state(value: bytes) -> bytes:
    if type(value) is not bytes or not 1 <= len(value) <= 16 * 1024 * 1024:
        raise ValueError("learned state must contain 1 byte through 16 MiB")
    return value


@dataclass(frozen=True, slots=True)
class Affordance:
    affordance_id: str
    disposition: Literal["ACT", "ASK", "WAIT", "STOP"]
    description: str
    permission_scope: str
    external_effect: bool = False
    authorization_mode: Literal["LOCAL", "HOST_GUARDED"] = "LOCAL"

    def __post_init__(self) -> None:
        _identifier(self.affordance_id, "affordance_id")
        if self.disposition not in _DISPOSITIONS:
            raise ValueError("affordance disposition is unsupported")
        _bounded(self.description, "description", 1024)
        _identifier(self.permission_scope, "permission_scope")
        if type(self.external_effect) is not bool:
            raise TypeError("external_effect must be boolean")
        if self.authorization_mode not in ("LOCAL", "HOST_GUARDED"):
            raise ValueError("authorization_mode must be LOCAL or HOST_GUARDED")
        if self.authorization_mode == "HOST_GUARDED" and not self.external_effect:
            raise ValueError("HOST_GUARDED affordances must declare an external effect")


@dataclass(frozen=True, slots=True)
class AffordanceRequest:
    idempotency_key: str
    trigger_ref: str
    affordance_id: str
    observation_ref: str
    state_head_ref: str
    action_payload: str = ""
    choice_context_json: str = "{}"

    def __post_init__(self) -> None:
        _reference(self.idempotency_key, "idempotency_key")
        _identifier(self.trigger_ref, "trigger_ref")
        _identifier(self.affordance_id, "affordance_id")
        _reference(self.observation_ref, "observation_ref")
        _reference(self.state_head_ref, "state_head_ref")
        _bounded(self.action_payload, "action_payload", 16_384, empty=True)
        _bounded(
            self.choice_context_json,
            "choice_context_json",
            CHOICE_CONTEXT_MAX_CHARACTERS,
        )
        try:
            context = json.loads(self.choice_context_json)
        except json.JSONDecodeError as exc:
            raise ValueError("choice_context_json must be JSON") from exc
        if type(context) is not dict:
            raise ValueError("choice_context_json must encode an object")


@dataclass(frozen=True, slots=True)
class ObservableConsequence:
    """Raw executor observation with no built-in reward or truth judgment."""

    request_ref: str
    source_kind: Literal["TOOL", "TEST", "WORLD"]
    source_ref: str
    observation_json: str
    artifact_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _reference(self.request_ref, "observable request_ref")
        if self.source_kind not in ("TOOL", "TEST", "WORLD"):
            raise ValueError("observable source_kind is unsupported")
        _reference(self.source_ref, "observable source_ref")
        _bounded(self.observation_json, "observable observation_json", 16_384)

        def reject_constant(value: str) -> object:
            raise ValueError(f"non-finite JSON constant is forbidden: {value}")

        def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("observable JSON contains a duplicate key")
                result[key] = value
            return result

        try:
            decoded = json.loads(
                self.observation_json,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("observable observation_json must be JSON") from exc
        if type(decoded) is not dict or not decoded:
            raise ValueError("observable observation_json must encode a non-empty object")
        if _canonical(decoded).decode("utf-8") != self.observation_json:
            raise ValueError("observable observation_json must be canonical JSON")
        for values, label in (
            (self.artifact_refs, "artifact_refs"),
            (self.evidence_refs, "evidence_refs"),
        ):
            if (
                type(values) is not tuple
                or len(values) > 16
                or any(type(reference) is not str for reference in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(f"observable {label} must be a canonical unique tuple")
            for reference in values:
                _reference(reference, f"observable {label} item")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": "jenny.observable-consequence.v1",
            "request_ref": self.request_ref,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "observation_json": self.observation_json,
            "artifact_refs": self.artifact_refs,
            "evidence_refs": self.evidence_refs,
        }

    @property
    def observation_ref(self) -> str:
        return _digest(_canonical(self.canonical_payload()))


@dataclass(frozen=True, slots=True)
class AffordanceReceipt:
    status: Literal["COMPLETED", "COMPLETED_UNEVALUATED", "DENIED", "ERROR"]
    output: str
    consequence: tuple[tuple[str, float], ...]
    observable_consequence: ObservableConsequence | None = None

    def __post_init__(self) -> None:
        if self.status not in ("COMPLETED", "COMPLETED_UNEVALUATED", "DENIED", "ERROR"):
            raise ValueError("receipt status is unsupported")
        _bounded(self.output, "receipt output", 16_384, empty=True)
        if type(self.consequence) is not tuple or len(self.consequence) > 32:
            raise ValueError("consequence must contain at most 32 values")
        names: set[str] = set()
        for name, value in self.consequence:
            _identifier(name, "consequence name")
            if name in names:
                raise ValueError("consequence names must be unique")
            names.add(name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError("consequence values must be finite")
        if self.observable_consequence is not None and not isinstance(
            self.observable_consequence, ObservableConsequence
        ):
            raise TypeError("observable_consequence must be ObservableConsequence or None")
        if self.status == "COMPLETED_UNEVALUATED" and (
            self.consequence or self.observable_consequence is not None
        ):
            raise ValueError("unevaluated receipt cannot carry observed consequence")
        if self.status in ("DENIED", "ERROR") and self.consequence:
            raise ValueError("denied or error receipt cannot carry scalar consequence")

    @property
    def receipt_ref(self) -> str:
        return _digest(_canonical(_receipt_payload(self)))


AffordanceExecutor = Callable[[AffordanceRequest], AffordanceReceipt]
ObservableConsequenceMapper = Callable[
    [ObservableConsequence], tuple[tuple[str, float], ...]
]


class DynamicAffordanceRegistry:
    """Runtime-discovered affordances; registration order has no priority."""

    def __init__(self) -> None:
        self._items: dict[
            str,
            tuple[
                Affordance,
                AffordanceExecutor | None,
                str | None,
                Literal["TOOL", "TEST", "WORLD"] | None,
                ObservableConsequenceMapper | None,
                str | None,
                Literal["SYNC", "DEFERRED"],
            ],
        ] = {}

    def register(
        self,
        affordance: Affordance,
        executor: AffordanceExecutor | None = None,
        *,
        observable_source_ref: str | None = None,
        observable_source_kind: Literal["TOOL", "TEST", "WORLD"] | None = None,
        consequence_mapper: ObservableConsequenceMapper | None = None,
        consequence_mapper_ref: str | None = None,
        execution_mode: Literal["SYNC", "DEFERRED"] = "SYNC",
    ) -> None:
        if not isinstance(affordance, Affordance):
            raise TypeError("affordance must be an Affordance")
        if affordance.affordance_id in self._items:
            raise ValueError("affordance_id is already registered")
        if execution_mode not in ("SYNC", "DEFERRED"):
            raise ValueError("execution_mode must be SYNC or DEFERRED")
        if (
            affordance.disposition == "ACT"
            and executor is None
            and execution_mode != "DEFERRED"
        ):
            raise ValueError("synchronous ACT affordances require an executor")
        if affordance.disposition != "ACT" and executor is not None:
            raise ValueError("only ACT affordances may have executors")
        if execution_mode == "DEFERRED":
            if affordance.disposition != "ACT":
                raise ValueError("only ACT affordances may use deferred execution")
            if executor is not None:
                raise ValueError(
                    "deferred ACT affordances cannot have an in-process executor"
                )
        if (
            affordance.authorization_mode == "HOST_GUARDED"
            and execution_mode != "DEFERRED"
        ):
            raise ValueError("HOST_GUARDED affordances require deferred execution")
        if observable_source_ref is not None:
            _reference(observable_source_ref, "observable_source_ref")
            if affordance.disposition != "ACT":
                raise ValueError("only ACT affordances may bind an observable source")
            if observable_source_kind not in ("TOOL", "TEST", "WORLD"):
                raise ValueError("observable source kind must be TOOL, TEST, or WORLD")
        elif observable_source_kind is not None:
            raise ValueError("observable source kind requires a source reference")
        if (consequence_mapper is None) != (consequence_mapper_ref is None):
            raise ValueError("consequence mapper and reference must be supplied together")
        if consequence_mapper is not None:
            if observable_source_ref is None:
                raise ValueError("consequence mapper requires an observable source")
            if not callable(consequence_mapper):
                raise TypeError("consequence mapper must be callable")
            _reference(consequence_mapper_ref, "consequence_mapper_ref")  # type: ignore[arg-type]
        self._items[affordance.affordance_id] = (
            affordance,
            executor,
            observable_source_ref,
            observable_source_kind,
            consequence_mapper,
            consequence_mapper_ref,
            execution_mode,
        )

    def definition(self, affordance_id: str) -> Affordance:
        try:
            return self._items[affordance_id][0]
        except KeyError as exc:
            raise KeyError("affordance is not registered") from exc

    def executor(self, affordance_id: str) -> AffordanceExecutor:
        try:
            executor = self._items[affordance_id][1]
        except KeyError as exc:
            raise KeyError("affordance is not registered") from exc
        if executor is None:
            raise ValueError("affordance has no executor")
        return executor

    def definitions(self) -> tuple[Affordance, ...]:
        return tuple(self._items[key][0] for key in sorted(self._items))

    def observable_source_ref(self, affordance_id: str) -> str | None:
        try:
            return self._items[affordance_id][2]
        except KeyError as exc:
            raise KeyError("affordance is not registered") from exc

    def observable_source_kind(
        self, affordance_id: str
    ) -> Literal["TOOL", "TEST", "WORLD"] | None:
        try:
            return self._items[affordance_id][3]
        except KeyError as exc:
            raise KeyError("affordance is not registered") from exc

    def consequence_mapper(
        self, affordance_id: str
    ) -> ObservableConsequenceMapper | None:
        try:
            return self._items[affordance_id][4]
        except KeyError as exc:
            raise KeyError("affordance is not registered") from exc

    def consequence_mapper_ref(self, affordance_id: str) -> str | None:
        try:
            return self._items[affordance_id][5]
        except KeyError as exc:
            raise KeyError("affordance is not registered") from exc

    def execution_mode(
        self, affordance_id: str
    ) -> Literal["SYNC", "DEFERRED"]:
        try:
            return self._items[affordance_id][6]
        except KeyError as exc:
            raise KeyError("affordance is not registered") from exc


@dataclass(frozen=True, slots=True)
class CycleObservation:
    trigger_ref: str
    source: Literal["SCHEDULER", "HUMAN", "AGENT", "CONTINUATION"]
    content: str

    def __post_init__(self) -> None:
        _identifier(self.trigger_ref, "trigger_ref")
        if self.source not in ("SCHEDULER", "HUMAN", "AGENT", "CONTINUATION"):
            raise ValueError("observation source is unsupported")
        _bounded(self.content, "observation content", 16_384, empty=self.source == "SCHEDULER")

    @property
    def observation_ref(self) -> str:
        return _digest(_canonical(asdict(self)))


@dataclass(frozen=True, slots=True)
class RankedAffordance:
    affordance_id: str
    score: float

    def __post_init__(self) -> None:
        _identifier(self.affordance_id, "ranked affordance_id")
        if type(self.score) not in (int, float) or not math.isfinite(float(self.score)):
            raise ValueError("ranking score must be finite")


@dataclass(frozen=True, slots=True)
class CycleChoice:
    selected_affordance_id: str
    rankings: tuple[RankedAffordance, ...]
    prediction: str
    uncertainty: float
    wake_after_seconds: float | None = None
    action_payload: str = ""
    context_json: str = "{}"

    def __post_init__(self) -> None:
        _identifier(self.selected_affordance_id, "selected_affordance_id")
        if type(self.rankings) is not tuple or not self.rankings:
            raise ValueError("rankings must be a non-empty tuple")
        ids = [item.affordance_id for item in self.rankings]
        if len(ids) != len(set(ids)) or self.selected_affordance_id not in ids:
            raise ValueError("rankings must uniquely contain the selected affordance")
        _bounded(self.prediction, "prediction", 4096)
        if type(self.uncertainty) not in (int, float) or not math.isfinite(float(self.uncertainty)) or not 0 <= self.uncertainty <= 1:
            raise ValueError("uncertainty must be finite in [0, 1]")
        if self.wake_after_seconds is not None:
            if type(self.wake_after_seconds) not in (int, float) or not math.isfinite(float(self.wake_after_seconds)) or not 0 <= self.wake_after_seconds <= 86_400:
                raise ValueError("wake_after_seconds must be in [0, 86400]")
        _bounded(self.action_payload, "action_payload", 16_384, empty=True)
        _bounded(
            self.context_json,
            "context_json",
            CHOICE_CONTEXT_MAX_CHARACTERS,
        )
        try:
            context = json.loads(self.context_json)
        except json.JSONDecodeError as exc:
            raise ValueError("context_json must be JSON") from exc
        if type(context) is not dict:
            raise ValueError("context_json must encode an object")

    @property
    def choice_ref(self) -> str:
        return _digest(_canonical(_choice_payload(self)))


@dataclass(frozen=True, slots=True)
class CycleUpdate:
    child_state: bytes
    reflection: str
    consolidation_proposal: str

    def __post_init__(self) -> None:
        _state(self.child_state)
        _bounded(self.reflection, "reflection", 8192)
        _bounded(self.consolidation_proposal, "consolidation_proposal", 8192)

    @property
    def child_state_ref(self) -> str:
        return _digest(self.child_state)


class CanonicalLearnedCycle(Protocol):
    """One learned choice/update boundary; the supervisor supplies no policy."""

    @property
    def qualification_ref(self) -> str | None: ...

    def choose(
        self,
        *,
        observation: CycleObservation,
        temporal: TemporalNow,
        affordances: Sequence[Affordance],
        state: bytes,
    ) -> CycleChoice: ...

    def learn(
        self,
        *,
        observation: CycleObservation,
        choice: CycleChoice,
        receipt: AffordanceReceipt,
        temporal: TemporalV2,
        parent_state: bytes,
    ) -> CycleUpdate: ...


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    reason: str


class PermissionGate:
    def __init__(
        self,
        *,
        allowed_internal_scopes: Sequence[str] = ("internal.cognition",),
        allowed_host_guarded_scopes: Sequence[str] = (),
        allowed_external_read_scopes: Sequence[str] = (),
        external_effects_enabled: bool = False,
    ) -> None:
        self.allowed_internal_scopes = frozenset(
            _identifier(item, "allowed internal scope") for item in allowed_internal_scopes
        )
        self.allowed_host_guarded_scopes = frozenset(
            _identifier(item, "allowed host-guarded scope")
            for item in allowed_host_guarded_scopes
        )
        self.allowed_external_read_scopes = frozenset(
            _identifier(item, "allowed external read scope")
            for item in allowed_external_read_scopes
        )
        if type(external_effects_enabled) is not bool:
            raise TypeError("external_effects_enabled must be boolean")
        self.external_effects_enabled = external_effects_enabled

    def authorize(self, affordance: Affordance) -> PermissionDecision:
        if affordance.authorization_mode == "HOST_GUARDED":
            if affordance.permission_scope not in self.allowed_host_guarded_scopes:
                return PermissionDecision(
                    False, "host-guarded permission scope is not allowed"
                )
            return PermissionDecision(
                True, "deferred request allowed for separate guarded host"
            )
        if (
            affordance.external_effect
            and affordance.permission_scope in self.allowed_external_read_scopes
        ):
            return PermissionDecision(
                True, "explicitly allowlisted read-only external observation"
            )
        if affordance.external_effect and not self.external_effects_enabled:
            return PermissionDecision(False, "external effects are disabled")
        if not affordance.external_effect and affordance.permission_scope not in self.allowed_internal_scopes:
            return PermissionDecision(False, "internal permission scope is not allowed")
        return PermissionDecision(True, "bounded scope allowed")


@dataclass(frozen=True, slots=True)
class SupervisorStateHead:
    state_ref: str
    moving_origin_ordinal: int
    last_event_ref: str | None
    scheduler_enabled: bool
    revision: int


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    status: Literal[
        "DISABLED",
        "QUIESCENT",
        "SHADOW",
        "WAITING",
        "ASKING",
        "STOPPED",
        "TOOL_PENDING",
        "COMMITTED",
    ]
    trigger_ref: str
    selected_affordance_id: str | None
    episode_ref: str | None
    pending_ref: str | None
    moving_origin_ordinal: int
    detail: str


@dataclass(frozen=True, slots=True)
class ProjectionItem:
    episode_ref: str
    event_ref: str
    ordinal: int
    payload_json: str

    def __post_init__(self) -> None:
        _reference(self.episode_ref, "episode_ref")
        _reference(self.event_ref, "event_ref")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("projection ordinal must be non-negative")
        _bounded(self.payload_json, "payload_json", 262_144)
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("payload_json must be JSON") from exc
        if type(payload) is not dict:
            raise ValueError("payload_json must encode an object")

    @property
    def projection_ref(self) -> str:
        return _digest(
            _canonical(
                {
                    "episode_ref": self.episode_ref,
                    "event_ref": self.event_ref,
                    "ordinal": self.ordinal,
                    "payload_json": self.payload_json,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class CanonicalAutonomyEpisode:
    episode_ref: str
    event_ref: str
    ordinal: int
    payload_json: str

    def __post_init__(self) -> None:
        _reference(self.episode_ref, "episode_ref")
        _reference(self.event_ref, "event_ref")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("episode ordinal must be non-negative")
        _bounded(self.payload_json, "payload_json", 262_144)
        payload = json.loads(self.payload_json)
        if type(payload) is not dict or payload.get("event_ref") != self.event_ref:
            raise ValueError("episode payload identity differs")


class ConsolidationProjector(Protocol):
    def project(self, item: ProjectionItem) -> str: ...


class AutonomyHeartbeat:
    """Bounded wake/sleep service; it contains no cognitive action policy."""

    def __init__(
        self,
        supervisor: "PersistentAutonomySupervisor",
        *,
        interval_seconds: float = 1.0,
        max_steps_per_session: int = 64,
        projector: ConsolidationProjector | None = None,
        operation_lock: object | None = None,
    ) -> None:
        if not isinstance(supervisor, PersistentAutonomySupervisor):
            raise TypeError("supervisor must be PersistentAutonomySupervisor")
        if type(interval_seconds) not in (int, float) or not math.isfinite(float(interval_seconds)) or not 0.001 <= interval_seconds <= 3600:
            raise ValueError("interval_seconds must be in [0.001, 3600]")
        if type(max_steps_per_session) is not int or not 1 <= max_steps_per_session <= 10_000:
            raise ValueError("max_steps_per_session must be 1 through 10000")
        self.supervisor = supervisor
        self.interval_seconds = float(interval_seconds)
        self.max_steps_per_session = max_steps_per_session
        self.projector = projector
        if operation_lock is not None and not all(
            callable(getattr(operation_lock, name, None))
            for name in ("acquire", "release")
        ):
            raise TypeError("operation_lock must support acquire and release")
        self.operation_lock = operation_lock
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._foreground_activity = threading.Event()
        self._thread: threading.Thread | None = None
        self._results: list[SupervisorResult] = []
        self._error: BaseException | None = None
        self._failed_wakes = 0
        self.max_failed_wakes = 24

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("heartbeat is already running")
        self._stop.clear()
        self._wake.clear()
        self._foreground_activity.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="jenny-2-autonomy-heartbeat", daemon=True
        )
        self._thread.start()

    def defer_for_foreground(self) -> None:
        """Restart idle eligibility when owner-directed work arrives.

        This is scheduling only.  It neither creates cognitive work nor changes
        the learned initiative decision.  Waking the wait lets repeated owner
        turns extend the quiet window without polling.
        """

        self._foreground_activity.set()
        self._wake.set()

    def _run(self) -> None:
        try:
            completed_steps = 0
            while completed_steps < self.max_steps_per_session:
                # Autonomous cognition is bounded idle-time work. Waiting before
                # every step (including the first) leaves the foreground lane
                # available after startup and prevents a restart from immediately
                # monopolizing the one canonical operation/GPU lane.
                self._wake.clear()
                if self._stop.is_set():
                    break
                if self._foreground_activity.is_set():
                    self._foreground_activity.clear()
                    continue
                self._wake.wait(self.interval_seconds)
                if self._stop.is_set():
                    break
                if self._foreground_activity.is_set():
                    self._foreground_activity.clear()
                    continue

                acquired = False
                if self.operation_lock is not None:
                    try:
                        acquired = bool(self.operation_lock.acquire(blocking=False))
                    except TypeError:
                        acquired = bool(self.operation_lock.acquire(False))
                    if not acquired:
                        # A heartbeat is low-priority idle work.  It never queues
                        # behind foreground work and tries again only after a new
                        # full idle interval.
                        continue
                    if self._foreground_activity.is_set():
                        self.operation_lock.release()
                        self._foreground_activity.clear()
                        continue
                try:
                    revision = self.supervisor.state_head().revision
                    result = self.supervisor.scheduler_tick(
                        f"heartbeat:{revision}",
                        preempt_requested=self._foreground_activity.is_set,
                    )
                    self._results.append(result)
                    import sys as _sys

                    print(
                        f"JENNY2_LIFE_TICK status={result.status} "
                        f"ordinal={result.moving_origin_ordinal} "
                        f"detail={str(getattr(result, 'detail', ''))[:160]}",
                        file=_sys.stderr,
                        flush=True,
                    )
                    if result.status == "COMMITTED" and self.projector is not None:
                        self.supervisor.retry_pending_projections(self.projector)
                    completed_steps += 1
                except Exception as exc:  # one failed wake must not end her day
                    import sys as _sys
                    import traceback as _traceback

                    self._error = exc
                    self._failed_wakes += 1
                    print(
                        "JENNY2_LIFE_WAKE_ERROR "
                        f"count={self._failed_wakes} type={type(exc).__name__} "
                        f"detail={str(exc)[:200]}",
                        file=_sys.stderr,
                        flush=True,
                    )
                    _traceback.print_exc(file=_sys.stderr)
                    _sys.stderr.flush()
                    if self._failed_wakes >= self.max_failed_wakes:
                        self._stop.set()
                        self._wake.set()
                finally:
                    if acquired:
                        self.operation_lock.release()
        except BaseException as exc:  # preserve the exact service failure for the owner
            self._error = exc
            self._stop.set()
            self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def join(self, timeout: float | None = None) -> tuple[SupervisorResult, ...]:
        thread = self._thread
        if thread is None:
            return tuple(self._results)
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("heartbeat did not stop within the requested timeout")
        if self._error is not None:
            raise RuntimeError("heartbeat stopped on a supervisor failure") from self._error
        return tuple(self._results)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        """Expose a retained terminal fault without joining or restarting the loop."""

        return self._error


def _choice_payload(choice: CycleChoice) -> dict[str, object]:
    return {
        "selected_affordance_id": choice.selected_affordance_id,
        "rankings": [asdict(item) for item in choice.rankings],
        "prediction": choice.prediction,
        "uncertainty": choice.uncertainty,
        "wake_after_seconds": choice.wake_after_seconds,
        "action_payload": choice.action_payload,
        "context_json": choice.context_json,
    }


def _choice_from_payload(value: Mapping[str, object]) -> CycleChoice:
    rankings = value.get("rankings")
    if type(rankings) is not list:
        raise ValueError("pending choice rankings are malformed")
    return CycleChoice(
        selected_affordance_id=value["selected_affordance_id"],  # type: ignore[arg-type]
        rankings=tuple(RankedAffordance(**item) for item in rankings),  # type: ignore[arg-type]
        prediction=value["prediction"],  # type: ignore[arg-type]
        uncertainty=value["uncertainty"],  # type: ignore[arg-type]
        wake_after_seconds=value.get("wake_after_seconds"),  # type: ignore[arg-type]
        action_payload=value.get("action_payload", ""),  # type: ignore[arg-type]
        context_json=value.get("context_json", "{}"),  # type: ignore[arg-type]
    )


def _receipt_payload(receipt: AffordanceReceipt) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": receipt.status,
        "output": receipt.output,
        "consequence": receipt.consequence,
    }
    if receipt.observable_consequence is not None:
        payload["observable_consequence"] = (
            receipt.observable_consequence.canonical_payload()
        )
    return payload


def _observable_from_payload(value: object) -> ObservableConsequence:
    if type(value) is not dict or set(value) != {
        "version",
        "request_ref",
        "source_kind",
        "source_ref",
        "observation_json",
        "artifact_refs",
        "evidence_refs",
    }:
        raise ValueError("observable consequence payload is malformed")
    if value.get("version") != "jenny.observable-consequence.v1":
        raise ValueError("observable consequence version differs")
    artifacts = value.get("artifact_refs")
    evidence = value.get("evidence_refs")
    if type(artifacts) is not list or type(evidence) is not list:
        raise ValueError("observable consequence references are malformed")
    return ObservableConsequence(
        request_ref=value["request_ref"],  # type: ignore[arg-type]
        source_kind=value["source_kind"],  # type: ignore[arg-type]
        source_ref=value["source_ref"],  # type: ignore[arg-type]
        observation_json=value["observation_json"],  # type: ignore[arg-type]
        artifact_refs=tuple(artifacts),
        evidence_refs=tuple(evidence),
    )


def _receipt_from_payload(value: Mapping[str, object]) -> AffordanceReceipt:
    fields = set(value)
    if fields not in (
        {"status", "output", "consequence"},
        {"status", "output", "consequence", "observable_consequence"},
    ):
        raise ValueError("pending receipt fields differ")
    raw = value.get("consequence")
    if type(raw) is not list:
        raise ValueError("pending receipt consequence is malformed")
    return AffordanceReceipt(
        status=value["status"],  # type: ignore[arg-type]
        output=value["output"],  # type: ignore[arg-type]
        consequence=tuple((item[0], item[1]) for item in raw),  # type: ignore[misc]
        observable_consequence=(
            None
            if "observable_consequence" not in value
            else _observable_from_payload(value["observable_consequence"])
        ),
    )


class PersistentAutonomySupervisor:
    """One durable control surface around one canonical learned cycle."""

    def __init__(
        self,
        path: str | Path,
        *,
        genesis: JennyGenesis,
        initial_state: bytes,
        clock: TrustedClock,
        cycle: CanonicalLearnedCycle,
        affordances: DynamicAffordanceRegistry,
        permission_gate: PermissionGate | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("a durable filesystem path is required")
        if not isinstance(genesis, JennyGenesis):
            raise TypeError("genesis must be JennyGenesis")
        _state(initial_state)
        if not isinstance(clock, TrustedClock):
            raise TypeError("clock must be TrustedClock")
        if not isinstance(affordances, DynamicAffordanceRegistry):
            raise TypeError("affordances must be DynamicAffordanceRegistry")
        if fault_injector is not None and not callable(fault_injector):
            raise TypeError("fault_injector must be callable")
        self.genesis = genesis
        self.clock = clock
        self.cycle = cycle
        self.affordances = affordances
        self.permission_gate = permission_gate or PermissionGate()
        self.fault_injector = fault_injector
        self._activity_sink: Jenny2ActivitySink | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize(initial_state)

    def bind_activity_sink(self, sink: Jenny2ActivitySink) -> None:
        """Attach one non-authoritative mechanical observer before service start."""

        required = ("begin", "phase", "finish", "fail", "committed")
        if any(not callable(getattr(sink, name, None)) for name in required):
            raise TypeError("activity sink does not implement the required hooks")
        if self._activity_sink is not None and self._activity_sink is not sink:
            raise RuntimeError("a different activity sink is already bound")
        self._activity_sink = sink

    def _activity_begin(self, source: str) -> None:
        if self._activity_sink is not None:
            try:
                self._activity_sink.begin(source)
            except Exception:
                pass

    def _activity_phase(self, phase: ActivityPhase) -> None:
        if self._activity_sink is not None:
            try:
                self._activity_sink.phase(phase)
            except Exception:
                pass

    def _activity_finish(self, status: str) -> None:
        if self._activity_sink is not None:
            try:
                self._activity_sink.finish(status)
            except Exception:
                pass

    def _activity_fail(self, error: BaseException) -> None:
        if self._activity_sink is not None:
            try:
                self._activity_sink.fail(error)
            except Exception:
                pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self, initial_state: bytes) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS identity (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            genesis_ref TEXT NOT NULL UNIQUE,
            genesis_bytes BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS supervisor_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            contract TEXT NOT NULL,
            scheduler_enabled INTEGER NOT NULL CHECK(scheduler_enabled IN (0,1)),
            state_ref TEXT NOT NULL,
            state_blob BLOB NOT NULL,
            moving_origin_ordinal INTEGER NOT NULL,
            last_event_ref TEXT,
            pending_json BLOB,
            shadow_json BLOB,
            revision INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS effects (
            idempotency_key TEXT PRIMARY KEY,
            trigger_ref TEXT NOT NULL UNIQUE,
            affordance_id TEXT NOT NULL,
            request_json BLOB NOT NULL,
            receipt_json BLOB
        );
        CREATE TABLE IF NOT EXISTS episodes (
            episode_ref TEXT PRIMARY KEY,
            trigger_ref TEXT NOT NULL UNIQUE,
            ordinal INTEGER NOT NULL UNIQUE,
            event_ref TEXT NOT NULL UNIQUE,
            episode_json BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projection_outbox (
            episode_ref TEXT PRIMARY KEY REFERENCES episodes(episode_ref),
            event_ref TEXT NOT NULL,
            ordinal INTEGER NOT NULL UNIQUE,
            payload_json BLOB NOT NULL,
            backend_ref TEXT
        );
        CREATE TABLE IF NOT EXISTS learned_updates (
            trigger_ref TEXT PRIMARY KEY,
            parent_state_ref TEXT NOT NULL,
            child_state_ref TEXT NOT NULL,
            child_state_blob BLOB NOT NULL,
            update_ref TEXT NOT NULL UNIQUE,
            update_json BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS autonomy_wake_checkpoint (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            state_ref TEXT NOT NULL,
            last_event_ref TEXT,
            checkpoint_ref TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS affective_substrate (
            episode_ref TEXT PRIMARY KEY REFERENCES episodes(episode_ref),
            ordinal INTEGER NOT NULL UNIQUE,
            sample_ref TEXT NOT NULL UNIQUE,
            sample_json BLOB NOT NULL
        );
        """
        with closing(self._connect()) as connection:
            connection.executescript(schema)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, 1, _SCHEMA_VERSION):
                raise RuntimeError("unsupported persistent-autonomy schema version")
            if version < _SCHEMA_VERSION:
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.execute("BEGIN IMMEDIATE")
            identity = connection.execute(
                "SELECT genesis_ref, genesis_bytes FROM identity WHERE singleton=1"
            ).fetchone()
            if identity is None:
                connection.execute(
                    "INSERT INTO identity(singleton, genesis_ref, genesis_bytes) VALUES(1,?,?)",
                    (self.genesis.genesis_ref, self.genesis.canonical_bytes()),
                )
                connection.execute(
                    "INSERT INTO supervisor_state VALUES(1,?,?,?,?,?,?,?,?,?)",
                    (
                        SUPERVISOR_STATE_CONTRACT,
                        1,
                        _digest(initial_state),
                        initial_state,
                        -1,
                        None,
                        None,
                        None,
                        0,
                    ),
                )
            elif tuple(identity) != (
                self.genesis.genesis_ref,
                self.genesis.canonical_bytes(),
            ):
                raise RuntimeError("Jenny genesis identity does not match the durable store")
            # Schema-v1 stores predate the silent event substrate.  Their
            # canonical episode bytes are immutable, so a purely mechanical
            # projection can be backfilled without inventing a state or
            # changing learned history.  Existing rows must be byte exact.
            for episode_ref, ordinal, event_ref, encoded_episode in connection.execute(
                "SELECT episode_ref,ordinal,event_ref,episode_json FROM episodes "
                "ORDER BY ordinal"
            ).fetchall():
                episode_payload = json.loads(bytes(encoded_episode))
                sample = project_canonical_episode(
                    episode_ref=str(episode_ref), episode_payload=episode_payload
                )
                if sample.ordinal != int(ordinal) or sample.event_ref != str(event_ref):
                    raise RuntimeError("event substrate backfill identity differs")
                existing_sample = connection.execute(
                    "SELECT ordinal,sample_ref,sample_json FROM affective_substrate "
                    "WHERE episode_ref=?",
                    (episode_ref,),
                ).fetchone()
                expected_sample = (
                    sample.ordinal,
                    sample.sample_ref,
                    sample.payload_json.encode("utf-8"),
                )
                if existing_sample is None:
                    connection.execute(
                        "INSERT INTO affective_substrate VALUES(?,?,?,?)",
                        (episode_ref, *expected_sample),
                    )
                elif tuple(existing_sample) != expected_sample:
                    raise RuntimeError("event substrate backfill bytes differ")
            connection.commit()
        self.audit_integrity()

    def audit_integrity(self) -> SupervisorStateHead:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT contract,scheduler_enabled,state_ref,state_blob,"
                "moving_origin_ordinal,last_event_ref,revision FROM supervisor_state "
                "WHERE singleton=1"
            ).fetchone()
            if row is None or row[0] != SUPERVISOR_STATE_CONTRACT:
                raise RuntimeError("persistent-autonomy state is absent or incompatible")
            if row[2] != _digest(bytes(row[3])):
                raise RuntimeError("persistent learned-state digest mismatch")
            count = int(connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
            outbox_count = int(
                connection.execute("SELECT COUNT(*) FROM projection_outbox").fetchone()[0]
            )
            if outbox_count != count:
                raise RuntimeError("canonical episodes and projection outbox diverged")
            substrate_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM affective_substrate"
                ).fetchone()[0]
            )
            if substrate_count != count:
                raise RuntimeError("canonical episodes and event substrate diverged")
            mismatch = connection.execute(
                "SELECT 1 FROM episodes e JOIN projection_outbox p USING(episode_ref) "
                "WHERE e.event_ref != p.event_ref OR e.ordinal != p.ordinal LIMIT 1"
            ).fetchone()
            if mismatch is not None:
                raise RuntimeError("projection outbox identity differs from canonical episode")
            substrate_mismatch = connection.execute(
                "SELECT 1 FROM episodes e JOIN affective_substrate s USING(episode_ref) "
                "WHERE e.ordinal != s.ordinal LIMIT 1"
            ).fetchone()
            if substrate_mismatch is not None:
                raise RuntimeError("event substrate ordinal differs from canonical episode")
            maximum = connection.execute("SELECT MAX(ordinal) FROM episodes").fetchone()[0]
            ordinal = int(row[4])
            if (count == 0 and (ordinal != -1 or maximum is not None)) or (
                count > 0 and (maximum != ordinal or count != ordinal + 1)
            ):
                raise RuntimeError("Moving Origin ordinal and episode history diverged")
            if ordinal == -1 and row[5] is not None:
                raise RuntimeError("empty history cannot have a last event")
            if ordinal >= 0:
                event = connection.execute(
                    "SELECT event_ref FROM episodes WHERE ordinal=?", (ordinal,)
                ).fetchone()
                if event is None or event[0] != row[5]:
                    raise RuntimeError("last event does not match Moving Origin head")
            checkpoint = connection.execute(
                "SELECT state_ref,last_event_ref,checkpoint_ref "
                "FROM autonomy_wake_checkpoint WHERE singleton=1"
            ).fetchone()
            if checkpoint is not None:
                state_ref, last_event_ref, checkpoint_ref = checkpoint
                _reference(str(state_ref), "autonomy wake checkpoint state_ref")
                if last_event_ref is not None:
                    _reference(
                        str(last_event_ref),
                        "autonomy wake checkpoint last_event_ref",
                    )
                expected_checkpoint_ref = self._autonomy_wake_checkpoint_ref(
                    str(state_ref),
                    None if last_event_ref is None else str(last_event_ref),
                )
                if checkpoint_ref != expected_checkpoint_ref:
                    raise RuntimeError("autonomy wake checkpoint digest mismatch")
            return SupervisorStateHead(
                state_ref=row[2],
                moving_origin_ordinal=ordinal,
                last_event_ref=row[5],
                scheduler_enabled=bool(row[1]),
                revision=int(row[6]),
            )

    def _state_row(self, connection: sqlite3.Connection) -> tuple[object, ...]:
        row = connection.execute(
            "SELECT scheduler_enabled,state_ref,state_blob,moving_origin_ordinal,"
            "last_event_ref,pending_json,shadow_json,revision FROM supervisor_state "
            "WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("persistent-autonomy state is absent")
        return row

    def state_head(self) -> SupervisorStateHead:
        return self.audit_integrity()

    def state_bytes(self) -> bytes:
        with closing(self._connect()) as connection:
            row = self._state_row(connection)
            return bytes(row[2])

    @staticmethod
    def _autonomy_wake_checkpoint_ref(
        state_ref: str, last_event_ref: str | None
    ) -> str:
        return _digest(
            _canonical(
                {
                    "version": "jenny.autonomy-wake-checkpoint.v1",
                    "state_ref": state_ref,
                    "last_event_ref": last_event_ref,
                }
            )
        )

    def _autonomy_wake_was_inspected(self, head: SupervisorStateHead) -> bool:
        """Return whether this exact canonical state/event pair was inspected."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state_ref,last_event_ref,checkpoint_ref "
                "FROM autonomy_wake_checkpoint WHERE singleton=1"
            ).fetchone()
        if row is None:
            return False
        expected = self._autonomy_wake_checkpoint_ref(
            head.state_ref, head.last_event_ref
        )
        return tuple(row) == (head.state_ref, head.last_event_ref, expected)

    def _record_autonomy_wake_inspection(self, head: SupervisorStateHead) -> bool:
        """Checkpoint one successful inspection if its canonical input is current.

        The write occurs only after the learned boundary returned successfully.
        A model/recall/selection failure therefore leaves the state/event pair
        eligible for retry.  If foreground work advanced the head concurrently,
        the obsolete inspection is not recorded.
        """

        checkpoint_ref = self._autonomy_wake_checkpoint_ref(
            head.state_ref, head.last_event_ref
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._state_row(connection)
            if current[1] != head.state_ref or current[4] != head.last_event_ref:
                connection.commit()
                return False
            connection.execute(
                "INSERT INTO autonomy_wake_checkpoint"
                "(singleton,state_ref,last_event_ref,checkpoint_ref) VALUES(1,?,?,?) "
                "ON CONFLICT(singleton) DO UPDATE SET "
                "state_ref=excluded.state_ref,last_event_ref=excluded.last_event_ref,"
                "checkpoint_ref=excluded.checkpoint_ref",
                (head.state_ref, head.last_event_ref, checkpoint_ref),
            )
            connection.commit()
        return True

    def state_bytes_for_ref(self, state_ref: str) -> bytes:
        """Read hash-verified historical state without making it a second head."""

        _reference(state_ref, "state_ref")
        with closing(self._connect()) as connection:
            current = self._state_row(connection)
            if current[1] == state_ref:
                candidates = (bytes(current[2]),)
            else:
                rows = connection.execute(
                    "SELECT child_state_blob FROM learned_updates "
                    "WHERE child_state_ref=?",
                    (state_ref,),
                ).fetchall()
                candidates = tuple(bytes(row[0]) for row in rows)
        if not candidates:
            raise KeyError("state_ref is absent from canonical learned updates")
        canonical = candidates[0]
        if any(value != canonical for value in candidates[1:]):
            raise RuntimeError("one state reference resolves to different canonical bytes")
        if _digest(canonical) != state_ref:
            raise RuntimeError("historical learned-state digest mismatch")
        return canonical

    def pending_bytes(self) -> bytes | None:
        with closing(self._connect()) as connection:
            value = self._state_row(connection)[5]
            return None if value is None else bytes(value)

    def shadow_bytes(self) -> bytes | None:
        with closing(self._connect()) as connection:
            value = self._state_row(connection)[6]
            return None if value is None else bytes(value)

    def set_scheduler_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE supervisor_state SET scheduler_enabled=?, revision=revision+1 "
                "WHERE singleton=1",
                (int(enabled),),
            )
            connection.commit()

    def semantic_age(self, acquired_ordinal: int) -> int:
        return semantic_age(self.state_head().moving_origin_ordinal, acquired_ordinal)

    def _existing_episode(self, trigger_ref: str) -> SupervisorResult | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT episode_ref,ordinal,episode_json FROM episodes WHERE trigger_ref=?",
                (trigger_ref,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(bytes(row[2]))
        return SupervisorResult(
            "COMMITTED",
            trigger_ref,
            payload["choice"]["selected_affordance_id"],
            row[0],
            None,
            int(row[1]),
            "existing idempotent episode",
        )

    def _preempt_wait_for_human(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            if row[5] is None:
                connection.commit()
                return
            pending = json.loads(bytes(row[5]))
            if pending.get("phase") not in ("WAIT", "ASK"):
                raise RuntimeError("human ingress cannot preempt an unresolved effect")
            connection.execute(
                "UPDATE supervisor_state SET pending_json=NULL, revision=revision+1 "
                "WHERE singleton=1"
            )
            connection.commit()

    def _validate_choice(self, choice: CycleChoice) -> Affordance:
        if not isinstance(choice, CycleChoice):
            raise TypeError("learned cycle must return a CycleChoice")
        registered = {item.affordance_id for item in self.affordances.definitions()}
        ranked = {item.affordance_id for item in choice.rankings}
        if not ranked <= registered:
            raise ValueError("learned choice ranked an unregistered affordance")
        return self.affordances.definition(choice.selected_affordance_id)

    def _store_shadow(
        self, observation: CycleObservation, temporal: TemporalNow, choice: CycleChoice
    ) -> None:
        payload = _canonical(
            {
                "observation": asdict(observation),
                "temporal": asdict(temporal),
                "choice": _choice_payload(choice),
                "qualification_ref": None,
            }
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE supervisor_state SET shadow_json=?, revision=revision+1 "
                "WHERE singleton=1",
                (payload,),
            )
            connection.commit()

    def _store_control_pending(
        self,
        *,
        phase: Literal["WAIT", "ASK"],
        observation: CycleObservation,
        temporal: TemporalNow,
        choice: CycleChoice,
    ) -> str:
        payload = {
            "version": "jenny.autonomy.pending.v1",
            "phase": phase,
            "observation": asdict(observation),
            "temporal": asdict(temporal),
            "choice": _choice_payload(choice),
            "parent_state_ref": self.state_head().state_ref,
            "qualification_ref": self.cycle.qualification_ref,
        }
        encoded = _canonical(payload)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            if row[5] is not None:
                raise RuntimeError("one autonomy operation is already pending")
            connection.execute(
                "UPDATE supervisor_state SET pending_json=?, revision=revision+1 "
                "WHERE singleton=1",
                (encoded,),
            )
            connection.commit()
        return _digest(encoded)

    def _validate_v2_reservation_binding(
        self,
        *,
        pending: Mapping[str, object],
        request: AffordanceRequest,
    ) -> Literal["SYNC", "DEFERRED"]:
        pending_request = pending.get("request")
        if type(pending_request) is not dict or pending_request != asdict(request):
            raise RuntimeError("effect request differs from its exact reservation")
        pending_affordance = pending.get("affordance")
        if type(pending_affordance) is not dict:
            raise RuntimeError("pending affordance definition is malformed")
        try:
            normalized_pending_affordance = Affordance(
                **pending_affordance  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("pending affordance definition is malformed") from exc
        current_affordance = self.affordances.definition(request.affordance_id)
        if normalized_pending_affordance != current_affordance:
            raise RuntimeError("affordance definition changed after reservation")
        expected_source_ref = pending.get("observable_source_ref")
        expected_source_kind = pending.get("observable_source_kind")
        expected_mapper_ref = pending.get("consequence_mapper_ref")
        expected_execution_mode = pending.get("execution_mode", "SYNC")
        if expected_execution_mode not in ("SYNC", "DEFERRED"):
            raise RuntimeError("pending execution mode is malformed")
        current_source_ref = self.affordances.observable_source_ref(
            request.affordance_id
        )
        current_source_kind = self.affordances.observable_source_kind(
            request.affordance_id
        )
        current_mapper_ref = self.affordances.consequence_mapper_ref(
            request.affordance_id
        )
        current_execution_mode = self.affordances.execution_mode(
            request.affordance_id
        )
        if (
            current_source_ref != expected_source_ref
            or current_source_kind != expected_source_kind
            or current_mapper_ref != expected_mapper_ref
            or current_execution_mode != expected_execution_mode
        ):
            raise RuntimeError("execution or observable binding changed after reservation")
        return expected_execution_mode  # type: ignore[return-value]

    def _validate_v2_receipt_binding(
        self,
        *,
        pending: Mapping[str, object],
        request: AffordanceRequest,
        receipt: AffordanceReceipt,
    ) -> None:
        self._validate_v2_reservation_binding(pending=pending, request=request)
        expected_source_ref = pending.get("observable_source_ref")
        expected_source_kind = pending.get("observable_source_kind")
        expected_mapper_ref = pending.get("consequence_mapper_ref")
        observed = receipt.observable_consequence
        if expected_source_ref is None:
            if observed is not None:
                raise ValueError("unbound affordance returned an observed consequence")
            return
        if expected_source_kind not in ("TOOL", "TEST", "WORLD"):
            raise RuntimeError("pending observable source kind is malformed")
        if receipt.status == "COMPLETED" and observed is None:
            raise ValueError("source-bound completion has no observed consequence")
        if observed is not None and (
            observed.request_ref != request.idempotency_key
            or observed.source_ref != expected_source_ref
            or observed.source_kind != expected_source_kind
        ):
            raise ValueError("observed consequence binding differs")
        if receipt.consequence:
            if expected_mapper_ref is None:
                raise ValueError("source observation has scalar credit without a mapper")
            _validated_mapped_consequence(receipt.consequence)

    def _apply_consequence_mapper(
        self, request: AffordanceRequest, receipt: AffordanceReceipt
    ) -> AffordanceReceipt:
        source_ref = self.affordances.observable_source_ref(request.affordance_id)
        if source_ref is None:
            return receipt
        # The observer reports only what happened.  It cannot grade its own
        # observation; scalar credit is produced, if at all, by another bound
        # component with a separate content identity.
        if receipt.consequence:
            raise ValueError("source-bound executor cannot supply scalar consequence")
        observed = receipt.observable_consequence
        if receipt.status == "COMPLETED" and observed is None:
            raise ValueError("source-bound completion has no observed consequence")
        if observed is not None and (
            observed.request_ref != request.idempotency_key
            or observed.source_ref != source_ref
            or observed.source_kind
            != self.affordances.observable_source_kind(request.affordance_id)
        ):
            raise ValueError("observed consequence binding differs")
        mapper = self.affordances.consequence_mapper(request.affordance_id)
        if receipt.status != "COMPLETED" or mapper is None:
            return receipt
        if observed is None:  # guarded above, retained for type narrowing
            raise RuntimeError("completed observation disappeared before mapping")
        mapped = _validated_mapped_consequence(mapper(observed))
        return AffordanceReceipt(
            receipt.status,
            receipt.output,
            mapped,
            observed,
        )

    def _reserve_effect(
        self,
        *,
        observation: CycleObservation,
        temporal: TemporalNow,
        choice: CycleChoice,
        affordance: Affordance,
    ) -> tuple[AffordanceRequest, bytes]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            if row[5] is not None:
                raise RuntimeError("one autonomy operation is already pending")
            state_ref = str(row[1])
            observable_source_ref = self.affordances.observable_source_ref(
                affordance.affordance_id
            )
            observable_source_kind = self.affordances.observable_source_kind(
                affordance.affordance_id
            )
            consequence_mapper_ref = self.affordances.consequence_mapper_ref(
                affordance.affordance_id
            )
            execution_mode = self.affordances.execution_mode(
                affordance.affordance_id
            )
            idempotency_key = _digest(
                _canonical(
                    {
                        "genesis_ref": self.genesis.genesis_ref,
                        "trigger_ref": observation.trigger_ref,
                        "choice_ref": choice.choice_ref,
                        "state_ref": state_ref,
                        "affordance": asdict(affordance),
                        "observable_source_ref": observable_source_ref,
                        "observable_source_kind": observable_source_kind,
                        "consequence_mapper_ref": consequence_mapper_ref,
                        "execution_mode": execution_mode,
                    }
                )
            )
            request = AffordanceRequest(
                idempotency_key=idempotency_key,
                trigger_ref=observation.trigger_ref,
                affordance_id=affordance.affordance_id,
                observation_ref=observation.observation_ref,
                state_head_ref=state_ref,
                action_payload=choice.action_payload,
                choice_context_json=choice.context_json,
            )
            pending = {
                "version": "jenny.autonomy.pending.v2",
                "phase": "RESERVED",
                "observation": asdict(observation),
                "temporal": asdict(temporal),
                "choice": _choice_payload(choice),
                "affordance": asdict(affordance),
                "observable_source_ref": observable_source_ref,
                "observable_source_kind": observable_source_kind,
                "consequence_mapper_ref": consequence_mapper_ref,
                "execution_mode": execution_mode,
                "request": asdict(request),
                "parent_state_ref": state_ref,
                "qualification_ref": self.cycle.qualification_ref,
            }
            encoded = _canonical(pending)
            connection.execute(
                "INSERT INTO effects(idempotency_key,trigger_ref,affordance_id,request_json) "
                "VALUES(?,?,?,?)",
                (
                    idempotency_key,
                    observation.trigger_ref,
                    affordance.affordance_id,
                    _canonical(asdict(request)),
                ),
            )
            connection.execute(
                "UPDATE supervisor_state SET pending_json=?, revision=revision+1 "
                "WHERE singleton=1",
                (encoded,),
            )
            connection.commit()
            return request, bytes(row[2])

    def _deferred_effect_from_pending(
        self,
        pending_bytes: bytes,
        *,
        phases: tuple[str, ...],
    ) -> tuple[dict[str, object], Affordance, AffordanceRequest]:
        pending = json.loads(pending_bytes)
        if pending.get("version") != "jenny.autonomy.pending.v2":
            raise RuntimeError("deferred execution requires a v2 reservation")
        if pending.get("phase") not in phases:
            raise RuntimeError("pending operation is not at the required deferred phase")
        raw_request = pending.get("request")
        if type(raw_request) is not dict:
            raise RuntimeError("pending deferred operation has no exact request")
        request = AffordanceRequest(**raw_request)  # type: ignore[arg-type]
        mode = self._validate_v2_reservation_binding(
            pending=pending,
            request=request,
        )
        if mode != "DEFERRED":
            raise RuntimeError("unresolved synchronous reservation requires reconciliation")
        raw_affordance = pending.get("affordance")
        if type(raw_affordance) is not dict:
            raise RuntimeError("pending deferred operation has no exact affordance")
        affordance = Affordance(**raw_affordance)  # type: ignore[arg-type]
        if affordance.disposition != "ACT":
            raise RuntimeError("only ACT affordances may be completed asynchronously")
        raw_observation = pending.get("observation")
        raw_choice = pending.get("choice")
        if type(raw_observation) is not dict or type(raw_choice) is not dict:
            raise RuntimeError("pending deferred cognition is malformed")
        observation = CycleObservation(**raw_observation)  # type: ignore[arg-type]
        choice = _choice_from_payload(raw_choice)
        if (
            request.trigger_ref != observation.trigger_ref
            or request.observation_ref != observation.observation_ref
            or request.affordance_id != affordance.affordance_id
            or choice.selected_affordance_id != affordance.affordance_id
            or request.action_payload != choice.action_payload
            or request.choice_context_json != choice.context_json
        ):
            raise RuntimeError("deferred request differs from reserved cognition")
        head = self.state_head()
        if (
            pending.get("parent_state_ref") != head.state_ref
            or request.state_head_ref != head.state_ref
        ):
            raise RuntimeError("pending deferred operation parent state drifted")
        if pending.get("qualification_ref") != self.cycle.qualification_ref:
            raise RuntimeError("learned selector qualification changed during the operation")
        permission = self.permission_gate.authorize(affordance)
        if not permission.allowed:
            raise PermissionError(
                f"deferred operation is no longer authorized: {permission.reason}"
            )
        with closing(self._connect()) as connection:
            effect = connection.execute(
                "SELECT trigger_ref,affordance_id,request_json,receipt_json "
                "FROM effects WHERE idempotency_key=?",
                (request.idempotency_key,),
            ).fetchone()
        if effect is None:
            raise RuntimeError("deferred effect reservation is absent")
        if tuple(effect[:3]) != (
            request.trigger_ref,
            request.affordance_id,
            _canonical(asdict(request)),
        ):
            raise RuntimeError("deferred effect bytes differ from reservation")
        has_receipt = effect[3] is not None
        if (pending.get("phase") == "RESERVED") == has_receipt:
            raise RuntimeError("deferred receipt phase and effect record diverged")
        return pending, affordance, request

    def pending_deferred_effect(
        self,
    ) -> tuple[Affordance, AffordanceRequest] | None:
        """Return the exact allowed deferred reservation awaiting execution."""

        pending_bytes = self.pending_bytes()
        if pending_bytes is None:
            return None
        pending = json.loads(pending_bytes)
        if pending.get("phase") != "RESERVED":
            return None
        _, affordance, request = self._deferred_effect_from_pending(
            pending_bytes,
            phases=("RESERVED",),
        )
        return affordance, request

    def pending_effect_request(self) -> AffordanceRequest | None:
        """Return only the request portion of ``pending_deferred_effect``."""

        pending = self.pending_deferred_effect()
        return None if pending is None else pending[1]

    def reserved_effect_request(self, idempotency_key: str) -> AffordanceRequest | None:
        """Resolve an exact durable request for completion-recovery only.

        The returned record is inert: this lookup neither authorizes nor executes
        the effect.  It permits a transport coordinator that durably staged its
        final receipt to close the tiny crash window between the canonical episode
        commit and its own transport-level ``COMMITTED`` marker.
        """

        _reference(idempotency_key, "idempotency_key")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT request_json FROM effects WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(bytes(row[0]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("stored effect request is malformed") from exc
        if type(payload) is not dict:
            raise RuntimeError("stored effect request is not an object")
        request = AffordanceRequest(**payload)  # type: ignore[arg-type]
        if request.idempotency_key != idempotency_key:
            raise RuntimeError("stored effect request identity differs")
        if _canonical(asdict(request)) != bytes(row[0]):
            raise RuntimeError("stored effect request failed canonical validation")
        return request

    def reserved_effect_request_for_trigger(
        self, trigger_ref: str
    ) -> AffordanceRequest | None:
        """Resolve the immutable effect request bound to one ingress identity."""

        _identifier(trigger_ref, "trigger_ref")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT idempotency_key FROM effects WHERE trigger_ref=?",
                (trigger_ref,),
            ).fetchone()
        if row is None:
            return None
        return self.reserved_effect_request(str(row[0]))

    def _raw_receipt_for_stored_mapped_receipt(
        self,
        request: AffordanceRequest,
        stored: AffordanceReceipt,
    ) -> AffordanceReceipt:
        mapper = self.affordances.consequence_mapper(request.affordance_id)
        if stored.status == "COMPLETED" and mapper is not None:
            return AffordanceReceipt(
                stored.status,
                stored.output,
                (),
                stored.observable_consequence,
            )
        return stored

    def _completed_deferred_replay(
        self,
        request: AffordanceRequest,
        raw_receipt: AffordanceReceipt,
    ) -> SupervisorResult:
        if self.affordances.execution_mode(request.affordance_id) != "DEFERRED":
            raise RuntimeError("synchronous effects cannot use deferred completion")
        affordance = self.affordances.definition(request.affordance_id)
        permission = self.permission_gate.authorize(affordance)
        if not permission.allowed:
            raise PermissionError(
                f"deferred operation is no longer authorized: {permission.reason}"
            )
        with closing(self._connect()) as connection:
            effect = connection.execute(
                "SELECT request_json,receipt_json FROM effects WHERE idempotency_key=?",
                (request.idempotency_key,),
            ).fetchone()
        if effect is None or bytes(effect[0]) != _canonical(asdict(request)):
            raise RuntimeError("deferred request differs from its exact reservation")
        if effect[1] is None:
            raise RuntimeError("deferred reservation lost its pending state")
        stored = _receipt_from_payload(json.loads(bytes(effect[1])))
        if raw_receipt != self._raw_receipt_for_stored_mapped_receipt(request, stored):
            raise RuntimeError("deferred completion conflicts with the stored receipt")
        existing = self._existing_episode(request.trigger_ref)
        if existing is None:
            raise RuntimeError("completed deferred effect has no canonical episode")
        return existing

    def complete_deferred_effect(
        self,
        request: AffordanceRequest,
        raw_receipt: AffordanceReceipt,
    ) -> SupervisorResult:
        """Accept one external observation and resume the canonical commit path."""

        if not isinstance(request, AffordanceRequest):
            raise TypeError("request must be an AffordanceRequest")
        if not isinstance(raw_receipt, AffordanceReceipt):
            raise TypeError("raw_receipt must be an AffordanceReceipt")
        pending_bytes = self.pending_bytes()
        if pending_bytes is None:
            return self._completed_deferred_replay(request, raw_receipt)
        pending, _, expected_request = self._deferred_effect_from_pending(
            pending_bytes,
            phases=("RESERVED", "RECEIPT", "UPDATE"),
        )
        if request != expected_request:
            raise RuntimeError("deferred request differs from its exact reservation")
        if pending["phase"] == "RESERVED":
            receipt = self._apply_consequence_mapper(request, raw_receipt)
            self._persist_receipt(request, receipt)
            return self._resume_receipt()
        stored = _receipt_from_payload(pending["receipt"])  # type: ignore[arg-type]
        if raw_receipt != self._raw_receipt_for_stored_mapped_receipt(request, stored):
            raise RuntimeError("deferred completion conflicts with the stored receipt")
        return self.resume_pending()

    def _persist_receipt(
        self, request: AffordanceRequest, receipt: AffordanceReceipt
    ) -> None:
        encoded_receipt = _canonical(_receipt_payload(receipt))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            if row[5] is None:
                raise RuntimeError("effect receipt has no pending reservation")
            pending = json.loads(bytes(row[5]))
            pending_request = pending.get("request")
            if type(pending_request) is not dict or pending_request != asdict(request):
                raise RuntimeError("effect request differs from its exact reservation")
            version = pending.get("version")
            if version == "jenny.autonomy.pending.v2":
                self._validate_v2_receipt_binding(
                    pending=pending, request=request, receipt=receipt
                )
            elif version == "jenny.autonomy.pending.v1":
                if receipt.observable_consequence is not None:
                    raise ValueError("legacy pending receipt cannot carry an observation")
            else:
                raise RuntimeError("pending operation version is unsupported")
            existing = connection.execute(
                "SELECT request_json,receipt_json FROM effects WHERE idempotency_key=?",
                (request.idempotency_key,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("effect reservation is absent")
            if bytes(existing[0]) != _canonical(asdict(request)):
                raise RuntimeError("effect request bytes differ from reservation")
            if existing[1] is not None and bytes(existing[1]) != encoded_receipt:
                raise RuntimeError("idempotency key has a different receipt")
            pending["phase"] = "RECEIPT"
            pending["receipt"] = _receipt_payload(receipt)
            connection.execute(
                "UPDATE effects SET receipt_json=? WHERE idempotency_key=?",
                (encoded_receipt, request.idempotency_key),
            )
            connection.execute(
                "UPDATE supervisor_state SET pending_json=?, revision=revision+1 "
                "WHERE singleton=1",
                (_canonical(pending),),
            )
            connection.commit()
        if self.fault_injector is not None:
            self.fault_injector("after_receipt_persisted")

    def _resume_receipt(self) -> SupervisorResult:
        self._activity_phase("REFLECT_CONSOLIDATE")
        try:
            with latency_phase("supervisor.reflect_learn_commit"):
                return self._resume_receipt_active()
        except BaseException as exc:
            self._activity_fail(exc)
            raise

    def _resume_receipt_active(self) -> SupervisorResult:
        with closing(self._connect()) as connection:
            row = self._state_row(connection)
            if row[5] is None:
                raise RuntimeError("no pending receipt to resume")
            pending_bytes = bytes(row[5])
            pending = json.loads(pending_bytes)
            if pending.get("phase") != "RECEIPT":
                raise RuntimeError("pending operation has no durable receipt")
            parent_state = bytes(row[2])
            parent_ref = str(row[1])
            current_ordinal = int(row[3])
        if pending.get("parent_state_ref") != parent_ref:
            raise RuntimeError("pending operation parent state drifted")
        if pending.get("qualification_ref") != self.cycle.qualification_ref:
            raise RuntimeError("learned selector qualification changed during the operation")
        observation = CycleObservation(**pending["observation"])
        choice = _choice_from_payload(pending["choice"])
        receipt = _receipt_from_payload(pending["receipt"])
        version = pending.get("version")
        if version == "jenny.autonomy.pending.v2":
            raw_request = pending.get("request")
            if type(raw_request) is not dict:
                raise RuntimeError("pending v2 receipt has no exact request")
            self._validate_v2_receipt_binding(
                pending=pending,
                request=AffordanceRequest(**raw_request),  # type: ignore[arg-type]
                receipt=receipt,
            )
        elif version == "jenny.autonomy.pending.v1":
            if receipt.observable_consequence is not None:
                raise ValueError("legacy pending receipt cannot carry an observation")
        else:
            raise RuntimeError("pending operation version is unsupported")
        event_sample = TemporalNow(**pending["temporal"])
        acquired = self.clock.sample(current_ordinal)
        recorded = self.clock.sample(current_ordinal)
        verified = acquired if receipt.status == "COMPLETED" else None
        temporal = TemporalV2.from_samples(
            ordinal=current_ordinal + 1,
            event=event_sample,
            acquired=acquired,
            recorded=recorded,
            verified=verified,
            source="jenny.persistent-autonomy.v1",
        )
        with latency_phase("supervisor.learn"):
            update = self.cycle.learn(
                observation=observation,
                choice=choice,
                receipt=receipt,
                temporal=temporal,
                parent_state=parent_state,
            )
        if not isinstance(update, CycleUpdate):
            raise TypeError("learned cycle must return CycleUpdate")
        update_payload = {
            "trigger_ref": observation.trigger_ref,
            "parent_state_ref": parent_ref,
            "child_state_ref": update.child_state_ref,
            "reflection": update.reflection,
            "consolidation_proposal": update.consolidation_proposal,
            "temporal": asdict(temporal),
        }
        update_ref = _digest(_canonical(update_payload))
        update_payload["update_ref"] = update_ref
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            if row[5] is None or bytes(row[5]) != pending_bytes:
                raise RuntimeError("pending operation changed before update staging")
            if str(row[1]) != parent_ref or bytes(row[2]) != parent_state:
                raise RuntimeError("parent learned state changed before update staging")
            existing = connection.execute(
                "SELECT parent_state_ref,child_state_ref,child_state_blob,update_ref,"
                "update_json FROM learned_updates WHERE trigger_ref=?",
                (observation.trigger_ref,),
            ).fetchone()
            expected = (
                parent_ref,
                update.child_state_ref,
                update.child_state,
                update_ref,
                _canonical(update_payload),
            )
            if existing is None:
                connection.execute(
                    "INSERT INTO learned_updates VALUES(?,?,?,?,?,?)",
                    (observation.trigger_ref, *expected),
                )
            elif tuple(existing) != expected:
                raise RuntimeError("trigger has a different staged learned update")
            pending["phase"] = "UPDATE"
            pending["update_ref"] = update_ref
            connection.execute(
                "UPDATE supervisor_state SET pending_json=?, revision=revision+1 "
                "WHERE singleton=1",
                (_canonical(pending),),
            )
            connection.commit()
        if self.fault_injector is not None:
            self.fault_injector("after_update_persisted")
        return self._commit_staged_update()

    def _commit_staged_update(self) -> SupervisorResult:
        self._activity_phase("COMMIT")
        try:
            with latency_phase("supervisor.atomic_commit"):
                return self._commit_staged_update_active()
        except BaseException as exc:
            self._activity_fail(exc)
            raise

    def _commit_staged_update_active(self) -> SupervisorResult:
        with closing(self._connect()) as connection:
            row = self._state_row(connection)
            if row[5] is None:
                raise RuntimeError("no staged learned update is pending")
            pending_bytes = bytes(row[5])
            pending = json.loads(pending_bytes)
            if pending.get("phase") != "UPDATE":
                raise RuntimeError("pending operation has no staged learned update")
            parent_state = bytes(row[2])
            parent_ref = str(row[1])
            current_ordinal = int(row[3])
            previous_event_ref = row[4]
            update_row = connection.execute(
                "SELECT parent_state_ref,child_state_ref,child_state_blob,update_ref,"
                "update_json FROM learned_updates WHERE trigger_ref=?",
                (pending["observation"]["trigger_ref"],),
            ).fetchone()
        if update_row is None:
            raise RuntimeError("staged learned update bytes are absent")
        if update_row[0] != parent_ref or bytes(update_row[2]) == b"":
            raise RuntimeError("staged learned update parent or bytes differ")
        child_state = bytes(update_row[2])
        if update_row[1] != _digest(child_state):
            raise RuntimeError("staged learned update digest differs")
        update_payload = json.loads(bytes(update_row[4]))
        if update_payload.get("update_ref") != update_row[3]:
            raise RuntimeError("staged learned update identity differs")
        without_ref = dict(update_payload)
        without_ref.pop("update_ref", None)
        if _digest(_canonical(without_ref)) != update_row[3]:
            raise RuntimeError("staged learned update content differs")
        if pending.get("update_ref") != update_row[3]:
            raise RuntimeError("pending update reference differs")
        if pending.get("qualification_ref") != self.cycle.qualification_ref:
            raise RuntimeError("learned selector qualification changed during the operation")
        observation = CycleObservation(**pending["observation"])
        choice = _choice_from_payload(pending["choice"])
        receipt = _receipt_from_payload(pending["receipt"])
        version = pending.get("version")
        if version == "jenny.autonomy.pending.v2":
            raw_request = pending.get("request")
            if type(raw_request) is not dict:
                raise RuntimeError("pending v2 update has no exact request")
            self._validate_v2_receipt_binding(
                pending=pending,
                request=AffordanceRequest(**raw_request),  # type: ignore[arg-type]
                receipt=receipt,
            )
        elif version == "jenny.autonomy.pending.v1":
            if receipt.observable_consequence is not None:
                raise ValueError("legacy pending update cannot carry an observation")
        else:
            raise RuntimeError("pending operation version is unsupported")
        temporal = TemporalV2(**update_payload["temporal"])
        if temporal.moving_origin_ordinal != current_ordinal + 1:
            raise RuntimeError("staged temporal ordinal differs from state head")
        episode_payload = {
            "contract": SUPERVISOR_STATE_CONTRACT,
            "genesis_ref": self.genesis.genesis_ref,
            "qualification_ref": pending["qualification_ref"],
            "observation": asdict(observation),
            "choice": _choice_payload(choice),
            "receipt": _receipt_payload(receipt),
            "temporal": asdict(temporal),
            "parent_state_ref": parent_ref,
            "child_state_ref": update_row[1],
            "reflection": update_payload["reflection"],
            "consolidation_proposal": update_payload["consolidation_proposal"],
            "update_ref": update_row[3],
            "previous_event_ref": previous_event_ref,
        }
        if "feedback" in pending:
            episode_payload["feedback"] = pending["feedback"]
        event_ref = _digest(_canonical(episode_payload))
        episode_payload["event_ref"] = event_ref
        episode_ref = _digest(_canonical(episode_payload))
        encoded_episode = _canonical(episode_payload)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            if row[5] is None or bytes(row[5]) != pending_bytes:
                raise RuntimeError("pending operation changed before atomic commit")
            if str(row[1]) != parent_ref or bytes(row[2]) != parent_state:
                raise RuntimeError("parent learned state changed before atomic commit")
            ordinal = int(row[3]) + 1
            if ordinal != temporal.moving_origin_ordinal:
                raise RuntimeError("temporal ordinal changed before atomic commit")
            connection.execute(
                "INSERT INTO episodes VALUES(?,?,?,?,?)",
                (episode_ref, observation.trigger_ref, ordinal, event_ref, encoded_episode),
            )
            connection.execute(
                "INSERT INTO projection_outbox(episode_ref,event_ref,ordinal,payload_json) "
                "VALUES(?,?,?,?)",
                (episode_ref, event_ref, ordinal, encoded_episode),
            )
            substrate = project_canonical_episode(
                episode_ref=episode_ref, episode_payload=episode_payload
            )
            if substrate.ordinal != ordinal or substrate.event_ref != event_ref:
                raise RuntimeError("event substrate commit identity differs")
            connection.execute(
                "INSERT INTO affective_substrate VALUES(?,?,?,?)",
                (
                    episode_ref,
                    substrate.ordinal,
                    substrate.sample_ref,
                    substrate.payload_json.encode("utf-8"),
                ),
            )
            connection.execute(
                "UPDATE supervisor_state SET state_ref=?,state_blob=?,"
                "moving_origin_ordinal=?,last_event_ref=?,pending_json=NULL,"
                "revision=revision+1 WHERE singleton=1",
                (update_row[1], child_state, ordinal, event_ref),
            )
            connection.commit()
        self.audit_integrity()
        if self._activity_sink is not None:
            try:
                self._activity_sink.committed(
                    episode_ref=episode_ref,
                    event_ref=event_ref,
                    ordinal=temporal.moving_origin_ordinal,
                    payload_json=encoded_episode.decode("utf-8"),
                )
            except Exception:
                pass
        return SupervisorResult(
            "COMMITTED",
            observation.trigger_ref,
            choice.selected_affordance_id,
            episode_ref,
            None,
            temporal.moving_origin_ordinal,
            "receipt, learned update, episode, and state head committed",
        )

    def pending_projections(self, limit: int = 64) -> tuple[ProjectionItem, ...]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("projection limit must be 1 through 256")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT episode_ref,event_ref,ordinal,payload_json "
                "FROM projection_outbox WHERE backend_ref IS NULL "
                "ORDER BY ordinal LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(
            ProjectionItem(row[0], row[1], int(row[2]), bytes(row[3]).decode("utf-8"))
            for row in rows
        )

    def episode_items(
        self, *, after_ordinal: int = -1, limit: int = 256
    ) -> tuple[CanonicalAutonomyEpisode, ...]:
        if type(after_ordinal) is not int or after_ordinal < -1:
            raise ValueError("after_ordinal must be at least -1")
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("episode limit must be 1 through 256")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT episode_ref,event_ref,ordinal,episode_json FROM episodes "
                "WHERE ordinal>? ORDER BY ordinal LIMIT ?",
                (after_ordinal, limit),
            ).fetchall()
        return tuple(
            CanonicalAutonomyEpisode(
                row[0], row[1], int(row[2]), bytes(row[3]).decode("utf-8")
            )
            for row in rows
        )

    def event_substrate_items(
        self, *, after_ordinal: int = -1, limit: int = 256
    ) -> tuple[EventSubstrateObservation, ...]:
        """Read hash-validated, unlabelled substrate observations in time order."""

        if type(after_ordinal) is not int or after_ordinal < -1:
            raise ValueError("after_ordinal must be at least -1")
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("substrate limit must be 1 through 256")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT episode_ref,ordinal,sample_ref,sample_json "
                "FROM affective_substrate WHERE ordinal>? ORDER BY ordinal LIMIT ?",
                (after_ordinal, limit),
            ).fetchall()
        items: list[EventSubstrateObservation] = []
        for episode_ref, ordinal, sample_ref, sample_json in rows:
            payload_json = bytes(sample_json).decode("utf-8")
            payload = json.loads(payload_json)
            item = EventSubstrateObservation(
                episode_ref=str(episode_ref),
                event_ref=payload.get("event_ref"),
                ordinal=int(ordinal),
                sample_ref=str(sample_ref),
                payload_json=payload_json,
            )
            items.append(item)
        return tuple(items)

    def episode_item(self, episode_ref: str) -> CanonicalAutonomyEpisode:
        """Read one canonical episode by its content identity."""

        _reference(episode_ref, "episode_ref")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT episode_ref,event_ref,ordinal,episode_json FROM episodes "
                "WHERE episode_ref=?",
                (episode_ref,),
            ).fetchone()
        if row is None:
            raise KeyError("episode_ref is absent from canonical history")
        return CanonicalAutonomyEpisode(
            row[0], row[1], int(row[2]), bytes(row[3]).decode("utf-8")
        )

    def episode_item_at_ordinal(self, ordinal: int) -> CanonicalAutonomyEpisode:
        """Read one canonical episode at an exact Moving-Origin position."""

        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("episode ordinal must be non-negative")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT episode_ref,event_ref,ordinal,episode_json FROM episodes "
                "WHERE ordinal=?",
                (ordinal,),
            ).fetchone()
        if row is None:
            raise KeyError("ordinal is absent from canonical history")
        return CanonicalAutonomyEpisode(
            row[0], row[1], int(row[2]), bytes(row[3]).decode("utf-8")
        )

    def acknowledge_projection(self, projection_ref: str, backend_ref: str) -> None:
        _reference(projection_ref, "projection_ref")
        _bounded(backend_ref, "backend_ref", 512)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT episode_ref,event_ref,ordinal,payload_json,backend_ref "
                "FROM projection_outbox ORDER BY ordinal"
            ).fetchall()
            selected = None
            for row in rows:
                item = ProjectionItem(
                    row[0], row[1], int(row[2]), bytes(row[3]).decode("utf-8")
                )
                if item.projection_ref == projection_ref:
                    selected = row
                    break
            if selected is None:
                raise KeyError("projection_ref is unknown")
            if selected[4] is not None:
                if selected[4] != backend_ref:
                    raise RuntimeError("projection already has a different backend reference")
                connection.commit()
                return
            connection.execute(
                "UPDATE projection_outbox SET backend_ref=? WHERE episode_ref=?",
                (backend_ref, selected[0]),
            )
            connection.commit()

    def rebuild_projections(self) -> int:
        """Clear every projection mark so the rebuildable memory projection is
        regenerated from the canonical record. Returns the episode count."""

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE projection_outbox SET backend_ref=NULL WHERE backend_ref IS NOT NULL"
            )
            count = int(cursor.rowcount)
            connection.commit()
        return count

    def retry_pending_projections(
        self, projector: ConsolidationProjector, *, limit: int = 64
    ) -> tuple[str, ...]:
        projected: list[str] = []
        for item in self.pending_projections(limit=limit):
            backend_ref = projector.project(item)
            _bounded(backend_ref, "projector backend_ref", 512)
            self.acknowledge_projection(item.projection_ref, backend_ref)
            projected.append(item.projection_ref)
        return tuple(projected)

    def reconcile_interrupted_synchronous(self) -> SupervisorResult | None:
        """Close one crash-stranded synchronous reservation as an ERROR.

        A synchronous executor cannot be safely replayed after process loss: it
        may have completed an effect before its receipt was persisted. Deferred
        host operations retain their existing explicit continuation protocol.
        """

        pending_bytes = self.pending_bytes()
        if pending_bytes is None:
            return None
        pending = json.loads(pending_bytes)
        if (
            pending.get("version") != "jenny.autonomy.pending.v2"
            or pending.get("phase") != "RESERVED"
        ):
            return None
        raw_request = pending.get("request")
        if type(raw_request) is not dict:
            raise RuntimeError("pending synchronous operation has no exact request")
        request = AffordanceRequest(**raw_request)  # type: ignore[arg-type]
        mode = self._validate_v2_reservation_binding(
            pending=pending, request=request
        )
        if mode != "SYNC":
            return None
        receipt = AffordanceReceipt(
            "ERROR",
            "Synchronous executor was interrupted before a durable receipt; "
            "the operation was not replayed.",
            (),
        )
        self._persist_receipt(request, receipt)
        return self._resume_receipt()

    def resume_pending(self) -> SupervisorResult:
        with closing(self._connect()) as connection:
            value = self._state_row(connection)[5]
        if value is None:
            raise RuntimeError("no autonomy operation is pending")
        pending = json.loads(bytes(value))
        version = pending.get("version")
        if version not in (
            "jenny.autonomy.pending.v1",
            "jenny.autonomy.pending.v2",
        ):
            raise RuntimeError("pending operation version is unsupported")
        if pending.get("phase") == "RECEIPT":
            return self._resume_receipt()
        if pending.get("phase") == "UPDATE":
            return self._commit_staged_update()
        phase = pending.get("phase")
        if phase in ("WAIT", "ASK"):
            if version != "jenny.autonomy.pending.v1":
                raise RuntimeError("control pending operation version is unsupported")
            head = self.state_head()
            return SupervisorResult(
                "WAITING" if phase == "WAIT" else "ASKING",
                pending["observation"]["trigger_ref"],
                pending["choice"]["selected_affordance_id"],
                None,
                _digest(bytes(value)),
                head.moving_origin_ordinal,
                "exact control pending state restored",
            )
        if phase == "RESERVED":
            _, affordance, request = self._deferred_effect_from_pending(
                bytes(value),
                phases=("RESERVED",),
            )
            head = self.state_head()
            return SupervisorResult(
                "TOOL_PENDING",
                request.trigger_ref,
                affordance.affordance_id,
                None,
                _digest(bytes(value)),
                head.moving_origin_ordinal,
                "exact deferred tool reservation restored without execution",
            )
        raise RuntimeError("reserved effect has no receipt and requires executor reconciliation")

    def _step(
        self,
        observation: CycleObservation,
        *,
        preempt_requested: Callable[[], bool] | None = None,
    ) -> SupervisorResult:
        if preempt_requested is not None and not callable(preempt_requested):
            raise TypeError("preempt_requested must be callable or None")
        self._activity_begin(observation.source)
        try:
            result = self._step_active(
                observation, preempt_requested=preempt_requested
            )
        except BaseException as exc:
            self._activity_fail(exc)
            raise
        self._activity_finish(result.status)
        return result

    def _step_active(
        self,
        observation: CycleObservation,
        *,
        preempt_requested: Callable[[], bool] | None = None,
    ) -> SupervisorResult:
        with latency_phase("supervisor.idempotency_lookup"):
            existing = self._existing_episode(observation.trigger_ref)
        if existing is not None:
            return existing
        with latency_phase("supervisor.prepare_context"):
            head = self.state_head()
            self._activity_phase("SAMPLE_TIME")
            temporal = self.clock.sample(head.moving_origin_ordinal)
            definitions = self.affordances.definitions()
        if not definitions:
            raise RuntimeError("no dynamic affordances are registered")
        self._activity_phase("DELIBERATE")
        with latency_phase("supervisor.choose"):
            choice = self.cycle.choose(
                observation=observation,
                temporal=temporal,
                affordances=definitions,
                state=self.state_bytes(),
            )
            affordance = self._validate_choice(choice)
        if (
            observation.source == "SCHEDULER"
            and preempt_requested is not None
            and bool(preempt_requested())
        ):
            return SupervisorResult(
                "QUIESCENT",
                observation.trigger_ref,
                None,
                None,
                None,
                head.moving_origin_ordinal,
                "foreground ingress preempted autonomous work after selection "
                "and before permission, execution, or state commit",
            )
        qualification_ref = self.cycle.qualification_ref
        if qualification_ref is None:
            self._store_shadow(observation, temporal, choice)
            return SupervisorResult(
                "SHADOW",
                observation.trigger_ref,
                choice.selected_affordance_id,
                None,
                None,
                head.moving_origin_ordinal,
                "learned selector is not causally qualified; no action or episode occurred",
            )
        _reference(qualification_ref, "qualification_ref")
        if affordance.disposition in ("WAIT", "ASK"):
            pending_ref = self._store_control_pending(
                phase=affordance.disposition,
                observation=observation,
                temporal=temporal,
                choice=choice,
            )
            return SupervisorResult(
                "WAITING" if affordance.disposition == "WAIT" else "ASKING",
                observation.trigger_ref,
                affordance.affordance_id,
                None,
                pending_ref,
                head.moving_origin_ordinal,
                "learned control disposition persisted without advancing semantic time",
            )
        if affordance.disposition == "STOP":
            self.set_scheduler_enabled(False)
            return SupervisorResult(
                "STOPPED",
                observation.trigger_ref,
                affordance.affordance_id,
                None,
                None,
                head.moving_origin_ordinal,
                "scheduler stopped; human ingress remains available",
            )
        self._activity_phase("RESERVE")
        with latency_phase("supervisor.reserve_effect"):
            request, _ = self._reserve_effect(
                observation=observation,
                temporal=temporal,
                choice=choice,
                affordance=affordance,
            )
        self._activity_phase("PERMISSION_GATE")
        with latency_phase("supervisor.permission_gate"):
            permission = self.permission_gate.authorize(affordance)
        if permission.allowed:
            if self.affordances.execution_mode(affordance.affordance_id) == "DEFERRED":
                pending = self.pending_bytes()
                if pending is None:
                    raise RuntimeError("deferred reservation disappeared after persistence")
                return SupervisorResult(
                    "TOOL_PENDING",
                    observation.trigger_ref,
                    affordance.affordance_id,
                    None,
                    _digest(pending),
                    head.moving_origin_ordinal,
                    "deferred tool reservation persisted without execution or learning",
                )
            self._activity_phase("EXECUTE")
            with latency_phase("supervisor.execute_affordance"):
                try:
                    raw_receipt = self.affordances.executor(
                        affordance.affordance_id
                    )(request)
                except Exception as exc:
                    # The reservation is already durable. Convert an operational
                    # failure into a durable ERROR receipt so one bad tool payload
                    # cannot strand the whole supervisor in RESERVED forever.
                    detail = (
                        f"{affordance.affordance_id} executor failed: "
                        f"{type(exc).__name__}: {exc}"
                    )[:16_384]
                    receipt = AffordanceReceipt(
                        "ERROR",
                        detail,
                        (),
                        None,
                    )
                else:
                    # Validate the returned boundary outside the executor-error
                    # handler. A raised tool exception is an operational failure;
                    # a returned forged binding or self-assigned scalar credit is
                    # an integrity violation and must remain fail-closed rather
                    # than being normalized into an ordinary tool error.
                    if not isinstance(raw_receipt, AffordanceReceipt):
                        raise TypeError(
                            "affordance executor must return AffordanceReceipt"
                        )
                    receipt = self._apply_consequence_mapper(request, raw_receipt)
        else:
            receipt = AffordanceReceipt(
                "DENIED", permission.reason, ()
            )
        with latency_phase("supervisor.persist_receipt"):
            self._persist_receipt(request, receipt)
        return self._resume_receipt()

    def scheduler_tick(
        self,
        trigger_ref: str,
        *,
        preempt_requested: Callable[[], bool] | None = None,
    ) -> SupervisorResult:
        _identifier(trigger_ref, "trigger_ref")
        if preempt_requested is not None and not callable(preempt_requested):
            raise TypeError("preempt_requested must be callable or None")
        self._activity_begin("SCHEDULER")
        try:
            result = self._scheduler_tick_active(
                trigger_ref, preempt_requested=preempt_requested
            )
        except BaseException as exc:
            self._activity_fail(exc)
            raise
        self._activity_finish(result.status)
        return result

    def _preempt_formed_autonomous_target(
        self,
        *,
        trigger_ref: str,
        head: SupervisorStateHead,
        preempt_requested: Callable[[], bool] | None,
    ) -> SupervisorResult | None:
        """Discard one in-memory target when foreground work takes priority.

        This checkpoint runs only after target formation returns and before the
        canonical cycle can select, reserve, authorize, execute, or commit.  It
        performs no cognitive selection: the cycle owns clearing its ephemeral
        handoff and the supervisor merely cancels that handoff.
        """

        if preempt_requested is None or not bool(preempt_requested()):
            return None
        cancel = getattr(self.cycle, "cancel_pending_autonomous_request", None)
        if not callable(cancel):
            raise RuntimeError(
                "autonomous target preemption requires a cycle-owned cancel hook"
            )
        cancel()
        return SupervisorResult(
            "QUIESCENT",
            trigger_ref,
            None,
            None,
            None,
            head.moving_origin_ordinal,
            "foreground ingress preempted the formed autonomous target before "
            "selection, permission, execution, or state commit",
        )

    def _scheduler_tick_active(
        self,
        trigger_ref: str,
        *,
        preempt_requested: Callable[[], bool] | None = None,
    ) -> SupervisorResult:
        head = self.state_head()
        if not head.scheduler_enabled:
            return SupervisorResult(
                "DISABLED",
                trigger_ref,
                None,
                None,
                None,
                head.moving_origin_ordinal,
                "scheduler disabled",
            )
        pending = self.pending_bytes()
        if pending is not None:
            restored = self.resume_pending()
            if restored.status in ("WAITING", "ASKING"):
                return restored
            return restored
        readiness = getattr(self.cycle, "has_autonomous_intent", None)
        if callable(readiness):
            if self._autonomy_wake_was_inspected(head):
                return SupervisorResult(
                    "QUIESCENT",
                    trigger_ref,
                    None,
                    None,
                    None,
                    head.moving_origin_ordinal,
                    "no canonical state or event change since the last initiative inspection",
                )
            self._activity_phase("DELIBERATE")
            if not readiness(state=self.state_bytes()):
                initiative = getattr(self.cycle, "propose_autonomous_request", None)
                request = None
                if callable(initiative):
                    request = initiative(
                        state=self.state_bytes(),
                        temporal=self.clock.sample(head.moving_origin_ordinal),
                        affordances=self.affordances.definitions(),
                    )
                    preempted = self._preempt_formed_autonomous_target(
                        trigger_ref=trigger_ref,
                        head=head,
                        preempt_requested=preempt_requested,
                    )
                    if preempted is not None:
                        return preempted
                    if request is not None and (
                        type(request) is not str
                        or not request.strip()
                        or len(request) > 2_048
                    ):
                        raise ValueError(
                            "model-authored autonomous request must be bounded text or None"
                        )
                if request is None:
                    self._record_autonomy_wake_inspection(head)
                    return SupervisorResult(
                        "QUIESCENT",
                        trigger_ref,
                        None,
                        None,
                        None,
                        head.moving_origin_ordinal,
                        "the learned model found no state-supported operation for this wake",
                    )
                result = self._step(
                    CycleObservation(trigger_ref, "SCHEDULER", request.strip()),
                    preempt_requested=preempt_requested,
                )
                self._record_autonomy_wake_inspection(head)
                return result
            result = self._step(
                CycleObservation(trigger_ref, "SCHEDULER", ""),
                preempt_requested=preempt_requested,
            )
            self._record_autonomy_wake_inspection(head)
            return result
        self._activity_phase("DELIBERATE")
        return self._step(
            CycleObservation(trigger_ref, "SCHEDULER", ""),
            preempt_requested=preempt_requested,
        )

    def human_ingress(self, trigger_ref: str, content: str) -> SupervisorResult:
        observation = CycleObservation(trigger_ref, "HUMAN", content)
        if self.pending_bytes() is not None:
            self._settle_pending_for_human()
        return self._step(observation)

    def _settle_pending_for_human(self) -> None:
        """A person is speaking. An unresolved effect is resumed first; if the
        resume fails it is quarantined with its reason so the person is never
        refused. WAIT/ASK control states are preempted as before."""

        raw = self.pending_bytes()
        if raw is None:
            return
        pending = json.loads(bytes(raw))
        if pending.get("phase") in ("WAIT", "ASK"):
            self._preempt_wait_for_human()
            return
        try:
            self.resume_pending()
        except Exception as exc:  # noqa: BLE001 — quarantined, never a refusal
            self._quarantine_pending(bytes(raw), f"{type(exc).__name__}: {str(exc)[:300]}")
        if self.pending_bytes() is not None:
            self._quarantine_pending(bytes(self.pending_bytes()), "unresolved after resume")

    def _quarantine_pending(self, raw: bytes, reason: str) -> None:
        import os as _os
        import sys as _sys
        import time as _time

        target_dir = _os.environ.get(
            "JENNY2_QUARANTINE_DIR",
            _os.path.join(_os.path.dirname(str(self.path)), "quarantined-effects"),
        )
        try:
            _os.makedirs(target_dir, exist_ok=True)
            name = _os.path.join(target_dir, f"{int(_time.time())}-{_digest(raw)[7:23]}.json")
            with open(name, "wb") as handle:
                handle.write(raw)
            with open(name + ".reason.txt", "w", encoding="utf-8") as handle:
                handle.write(reason + "\n")
        except OSError:
            name = "(sidecar write failed)"
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE supervisor_state SET pending_json=NULL, revision=revision+1 "
                "WHERE singleton=1"
            )
            connection.commit()
        print(
            f"JENNY2_PENDING_QUARANTINED file={name} reason={reason[:200]}",
            file=_sys.stderr,
            flush=True,
        )

    def agent_ingress(self, trigger_ref: str, content: str) -> SupervisorResult:
        """Submit a human-originated agent goal without treating it as chat prose."""

        observation = CycleObservation(trigger_ref, "AGENT", content)
        if self.pending_bytes() is not None:
            self._preempt_wait_for_human()
        return self._step(observation)

    def continuation_ingress(self, trigger_ref: str, content: str) -> SupervisorResult:
        """Resume model-authored work through the same all-affordance cycle."""

        observation = CycleObservation(trigger_ref, "CONTINUATION", content)
        if self.pending_bytes() is not None:
            self._preempt_wait_for_human()
        return self._step(observation)

    def submit_feedback(
        self,
        trigger_ref: str,
        *,
        target_episode_ref: str,
        feedback_text: str,
        feedback_source_ref: str,
        consequence: tuple[tuple[str, float], ...],
    ) -> SupervisorResult:
        """Append real late feedback and learn once from an unevaluated episode.

        Feedback is a new canonical event.  The completed output and original
        choice remain immutable; no external action is rerun.
        """

        observation = CycleObservation(trigger_ref, "HUMAN", feedback_text)
        _reference(target_episode_ref, "target_episode_ref")
        _reference(feedback_source_ref, "feedback_source_ref")
        receipt_template = AffordanceReceipt("COMPLETED", "", consequence)
        existing = self._existing_episode(trigger_ref)
        if existing is not None:
            return existing
        if self.pending_bytes() is not None:
            self._preempt_wait_for_human()
        qualification_ref = self.cycle.qualification_ref
        if qualification_ref is None:
            raise RuntimeError("late feedback cannot update an unqualified learned cycle")
        _reference(qualification_ref, "qualification_ref")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT episode_json FROM episodes WHERE episode_ref=?",
                (target_episode_ref,),
            ).fetchone()
            if row is None:
                raise KeyError("feedback target episode is absent")
            target = json.loads(bytes(row[0]))
            if target.get("receipt", {}).get("status") != "COMPLETED_UNEVALUATED":
                raise ValueError("late feedback requires a completed unevaluated episode")
            for candidate in connection.execute("SELECT episode_json FROM episodes"):
                payload = json.loads(bytes(candidate[0]))
                if payload.get("feedback", {}).get("target_episode_ref") == target_episode_ref:
                    raise ValueError("the target episode already has canonical feedback")
            state_row = self._state_row(connection)
            if state_row[5] is not None:
                raise RuntimeError("one autonomy operation is already pending")
            parent_ref = str(state_row[1])
            current_ordinal = int(state_row[3])
        choice = _choice_from_payload(target["choice"])
        target_receipt = _receipt_from_payload(target["receipt"])
        context = json.loads(choice.context_json)
        context["late_feedback"] = {
            "feedback_text": feedback_text,
            "feedback_source_ref": feedback_source_ref,
            "target_episode_ref": target_episode_ref,
            "target_choice_ref": choice.choice_ref,
            "target_receipt_ref": target_receipt.receipt_ref,
        }
        choice = CycleChoice(
            choice.selected_affordance_id,
            choice.rankings,
            choice.prediction,
            choice.uncertainty,
            choice.wake_after_seconds,
            choice.action_payload,
            json.dumps(context, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        )
        receipt = AffordanceReceipt(
            "COMPLETED",
            target["receipt"]["output"],
            receipt_template.consequence,
        )
        temporal = self.clock.sample(current_ordinal)
        pending = {
            "version": "jenny.autonomy.pending.v1",
            "phase": "RECEIPT",
            "observation": asdict(observation),
            "temporal": asdict(temporal),
            "choice": _choice_payload(choice),
            "receipt": _receipt_payload(receipt),
            "parent_state_ref": parent_ref,
            "qualification_ref": qualification_ref,
            "feedback": {
                "target_episode_ref": target_episode_ref,
                "feedback_source_ref": feedback_source_ref,
                "feedback_text": feedback_text,
                "consequence": [list(item) for item in consequence],
            },
        }
        encoded = _canonical(pending)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            if row[5] is not None or str(row[1]) != parent_ref:
                raise RuntimeError("state changed before late feedback staging")
            connection.execute(
                "UPDATE supervisor_state SET pending_json=?, revision=revision+1 "
                "WHERE singleton=1",
                (encoded,),
            )
            connection.commit()
        if self.fault_injector is not None:
            self.fault_injector("after_feedback_persisted")
        return self._resume_receipt()

    def run_internal(self, run_ref: str, *, max_steps: int) -> tuple[SupervisorResult, ...]:
        _identifier(run_ref, "run_ref")
        if type(max_steps) is not int or not 1 <= max_steps <= 64:
            raise ValueError("max_steps must be 1 through 64")
        results: list[SupervisorResult] = []
        for index in range(max_steps):
            result = self.scheduler_tick(f"{run_ref}:{index}")
            results.append(result)
            if result.status != "COMMITTED":
                break
        return tuple(results)
