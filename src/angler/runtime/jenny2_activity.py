"""Non-authoritative, nonblocking observability for Jenny 2.0 runtime work.

The activity view is deliberately smaller than the canonical episode.  It
reports mechanical execution progress and a strict summary of the last
committed cycle; it never exposes observations, prompts, memories, action
payloads, alternative candidates, or model scratch reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import threading
from time import monotonic
from typing import Literal, Protocol


ActivityPhase = Literal[
    "IDLE",
    "OBSERVE",
    "SAMPLE_TIME",
    "DELIBERATE",
    "RESERVE",
    "PERMISSION_GATE",
    "EXECUTE",
    "REFLECT_CONSOLIDATE",
    "COMMIT",
    "ERROR",
]

_PHASES = frozenset(
    (
        "IDLE",
        "OBSERVE",
        "SAMPLE_TIME",
        "DELIBERATE",
        "RESERVE",
        "PERMISSION_GATE",
        "EXECUTE",
        "REFLECT_CONSOLIDATE",
        "COMMIT",
        "ERROR",
    )
)
_SOURCES = frozenset(("HUMAN", "SCHEDULER", "AGENT", "CONTINUATION"))
_REFLECTION_FIELDS = (
    "epistemic_status",
    "analysis",
    "revision",
    "retained_principle",
    "reasoned_judgment",
    "uncertainty",
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _bounded_text(value: object, maximum: int = 2_048) -> str | None:
    if type(value) is not str or not value.strip():
        return None
    return value[:maximum]


def _finite_unit_interval(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        return None
    return normalized


def _object(value: object) -> dict[str, object]:
    return value if type(value) is dict else {}


def _reflection_summary(value: object) -> dict[str, object]:
    text = _bounded_text(value, 8_192)
    if text is None:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text[:2_048]}
    if type(decoded) is not dict:
        return {"summary": text[:2_048]}
    summary: dict[str, object] = {}
    for name in _REFLECTION_FIELDS:
        item = decoded.get(name)
        if name == "uncertainty":
            normalized = _finite_unit_interval(item)
            if normalized is not None:
                summary[name] = normalized
            continue
        bounded = _bounded_text(item)
        if bounded is not None:
            summary[name] = bounded
    return summary


def summarize_committed_episode(
    *,
    episode_ref: str,
    event_ref: str,
    ordinal: int,
    payload_json: str,
) -> dict[str, object]:
    """Return only the explicit public activity fields from one episode."""

    try:
        payload = _object(json.loads(payload_json))
    except (TypeError, json.JSONDecodeError):
        payload = {}
    choice = _object(payload.get("choice"))
    receipt = _object(payload.get("receipt"))
    try:
        context = _object(json.loads(str(choice.get("context_json", "{}"))))
    except json.JSONDecodeError:
        context = {}
    intent = _object(context.get("intent_proposal"))
    consequence = receipt.get("consequence")
    return {
        "episode_ref": episode_ref,
        "event_ref": event_ref,
        "moving_origin_ordinal": ordinal,
        "selected_affordance_id": _bounded_text(
            choice.get("selected_affordance_id"), 256
        ),
        "rationale": {
            "epistemic_status": _bounded_text(
                intent.get("epistemic_status"), 256
            ),
            "state_assessment": _bounded_text(intent.get("state_assessment")),
            "resolution_target": _bounded_text(intent.get("resolution_target")),
            "desired_state_change": _bounded_text(
                intent.get("desired_state_change")
            ),
            "selection_basis": _bounded_text(intent.get("selection_basis")),
        },
        "prediction": {
            "expected_state_delta": _bounded_text(choice.get("prediction"), 4_096),
            "uncertainty": _finite_unit_interval(choice.get("uncertainty")),
        },
        "receipt": {
            "status": _bounded_text(receipt.get("status"), 128),
            "has_consequence": type(consequence) is list and bool(consequence),
            "has_observed_consequence": type(
                receipt.get("observable_consequence")
            ) is dict,
        },
        "reflection": _reflection_summary(payload.get("reflection")),
    }


@dataclass(frozen=True, slots=True)
class Jenny2ActivitySnapshot:
    version: str
    sequence: int
    phase: ActivityPhase
    phase_started_at_utc: str
    phase_elapsed_seconds: float
    operation_source: str | None
    current_status: str
    error_code: str | None
    last_completed: dict[str, object] | None


class Jenny2ActivitySink(Protocol):
    """Small mechanical hook consumed by the persistent supervisor."""

    def begin(self, source: str) -> None: ...

    def phase(self, phase: ActivityPhase) -> None: ...

    def finish(self, status: str) -> None: ...

    def fail(self, error: BaseException) -> None: ...

    def committed(
        self,
        *,
        episode_ref: str,
        event_ref: str,
        ordinal: int,
        payload_json: str,
    ) -> None: ...


class Jenny2ActivityTracker:
    """A lock-isolated cache; reads never acquire the cognitive operation lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._phase: ActivityPhase = "IDLE"
        self._phase_started_at_utc = _utc_now()
        self._phase_started_monotonic = monotonic()
        self._operation_source: str | None = None
        self._current_status = "IDLE"
        self._error_code: str | None = None
        self._last_completed_json: str | None = None

    def _transition(
        self,
        phase: ActivityPhase,
        *,
        source: str | None,
        status: str,
        error_code: str | None,
    ) -> None:
        if phase not in _PHASES:
            raise ValueError("activity phase is unsupported")
        if source is not None and source not in _SOURCES:
            raise ValueError("activity source is unsupported")
        if type(status) is not str or not status or len(status) > 128:
            raise ValueError("activity status must be bounded text")
        if error_code is not None and (
            type(error_code) is not str
            or not error_code
            or len(error_code) > 128
        ):
            raise ValueError("activity error code must be bounded text")
        with self._lock:
            self._sequence += 1
            self._phase = phase
            self._phase_started_at_utc = _utc_now()
            self._phase_started_monotonic = monotonic()
            self._operation_source = source
            self._current_status = status
            self._error_code = error_code

    def begin(self, source: str) -> None:
        self._transition(
            "OBSERVE", source=source, status="RUNNING", error_code=None
        )

    def phase(self, phase: ActivityPhase) -> None:
        if phase in ("IDLE", "ERROR"):
            raise ValueError("terminal activity phases require finish or fail")
        with self._lock:
            source = self._operation_source
        self._transition(phase, source=source, status="RUNNING", error_code=None)

    def finish(self, status: str) -> None:
        self._transition("IDLE", source=None, status=status, error_code=None)

    def fail(self, error: BaseException) -> None:
        self._transition(
            "ERROR",
            source=None,
            status="ERROR",
            error_code=type(error).__name__,
        )

    def committed(
        self,
        *,
        episode_ref: str,
        event_ref: str,
        ordinal: int,
        payload_json: str,
    ) -> None:
        summary = summarize_committed_episode(
            episode_ref=episode_ref,
            event_ref=event_ref,
            ordinal=ordinal,
            payload_json=payload_json,
        )
        encoded = json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock:
            self._last_completed_json = encoded
        self.finish("COMMITTED")

    def snapshot(self) -> Jenny2ActivitySnapshot:
        with self._lock:
            elapsed = max(0.0, monotonic() - self._phase_started_monotonic)
            last_completed = (
                None
                if self._last_completed_json is None
                else json.loads(self._last_completed_json)
            )
            return Jenny2ActivitySnapshot(
                version="jenny.runtime-activity.v1",
                sequence=self._sequence,
                phase=self._phase,
                phase_started_at_utc=self._phase_started_at_utc,
                phase_elapsed_seconds=round(elapsed, 6),
                operation_source=self._operation_source,
                current_status=self._current_status,
                error_code=self._error_code,
                last_completed=last_completed,
            )
