"""One closed, attributable cognitive cycle over higher-level components.

This module is orchestration and evidence plumbing.  It contains no task
solver, answer table, verifier, or model-specific prompt.  Concrete adapters
provide semantic memory, temporal state, experience generation, and cortex
execution; objective consequence enters only after a durable receipt exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Mapping, Protocol, Sequence


# Compatibility vocabulary retained for replaying older evidence. New
# structured experience may describe an operation in ordinary language.
PROCESS_ACTIONS = frozenset(
    {"DECOMPOSE", "RETRIEVE", "PREDICT", "TRY", "VERIFY", "REVISE", "ASK", "ACT", "STOP"}
)
REMOVALS = frozenset(
    {"FULL", "NO_EXPERIENCE", "NO_UTILITY", "NO_REFLECTION", "NO_TEMPORAL", "SHUFFLED_CONSEQUENCE"}
)


def _text(value: object, label: str, maximum: int, *, empty: bool = False) -> str:
    if type(value) is not str or len(value) > maximum or (not empty and not value.strip()):
        raise ValueError(f"{label} must be bounded text")
    return value


def _finite(value: object, label: str, *, low: float = -1.0, high: float = 1.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not low <= result <= high:
        raise ValueError(f"{label} must be in [{low}, {high}]")
    return result


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def content_ref(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ConsequenceVector:
    objective_progress: float
    constraint_satisfaction: float
    prediction_error: float
    information_gain: float
    evidence_quality: float
    reuse_value: float
    cost: float
    safety: float
    human_feedback: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _finite(value, name))

    @property
    def learning_utility(self) -> float:
        positive = (
            self.objective_progress
            + self.constraint_satisfaction
            + self.information_gain
            + self.evidence_quality
            + self.reuse_value
            + self.safety
            + self.human_feedback
        ) / 7.0
        penalty = (abs(self.prediction_error) + max(0.0, self.cost)) / 2.0
        return max(-1.0, min(1.0, positive - 0.35 * penalty))


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    record_ref: str
    content: str
    semantic_distance: float
    utility: float = 0.0
    acquired_ordinal: int = 0
    event_time_utc: str | None = None
    acquired_time_utc: str | None = None
    recorded_time_utc: str | None = None
    verified_time_utc: str | None = None
    valid_from_utc: str | None = None
    valid_until_utc: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        _text(self.record_ref, "record_ref", 96)
        _text(self.content, "content", 4096)
        if type(self.semantic_distance) not in (int, float) or not math.isfinite(float(self.semantic_distance)):
            raise ValueError("semantic_distance must be finite")
        _finite(self.utility, "utility")
        if type(self.acquired_ordinal) is not int or self.acquired_ordinal < 0:
            raise ValueError("acquired_ordinal must be non-negative")
        for name in (
            "event_time_utc", "acquired_time_utc", "recorded_time_utc",
            "verified_time_utc", "valid_from_utc", "valid_until_utc", "timezone",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"{name} must be non-empty text or None")


@dataclass(frozen=True, slots=True)
class TemporalContext:
    now: int
    last_event_ref: str | None
    landmark_relations: tuple[tuple[str, str], ...] = ()
    trusted_utc: str | None = None
    local_time: str | None = None
    local_timezone: str | None = None
    local_utc_offset_seconds: int | None = None
    clock_uncertainty_ms: float | None = None
    clock_jump_detected: bool = False

    def __post_init__(self) -> None:
        if type(self.now) is not int or self.now < -1:
            raise ValueError("temporal now must be at least -1")
        for name in ("trusted_utc", "local_time", "local_timezone"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"{name} must be non-empty text or None")
        if self.local_utc_offset_seconds is not None and type(
            self.local_utc_offset_seconds
        ) is not int:
            raise TypeError("local_utc_offset_seconds must be an integer or None")
        if self.clock_uncertainty_ms is not None and (
            type(self.clock_uncertainty_ms) not in (int, float)
            or not math.isfinite(float(self.clock_uncertainty_ms))
            or self.clock_uncertainty_ms < 0
        ):
            raise ValueError("clock_uncertainty_ms must be finite and non-negative")
        if type(self.clock_jump_detected) is not bool:
            raise TypeError("clock_jump_detected must be boolean")


@dataclass(frozen=True, slots=True)
class StructuredExperience:
    interpretation: str
    process_action: str
    strategy: str
    predicted_consequence: str
    checks: tuple[str, ...]
    uncertainty: float
    world_model: str = "No situated world model supplied."
    self_model: str = "No situated self model supplied."
    focus: str = "Address the current request."
    unfinished_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.interpretation, "interpretation", 16384)
        _text(self.process_action, "process_action", 4096)
        _text(self.strategy, "strategy", 16384)
        _text(self.predicted_consequence, "predicted_consequence", 16384)
        if type(self.checks) is not tuple or not 1 <= len(self.checks) <= 8:
            raise ValueError("checks must contain 1 through 8 entries")
        for check in self.checks:
            _text(check, "check", 2048)
        object.__setattr__(self, "uncertainty", _finite(self.uncertainty, "uncertainty", low=0.0))
        _text(self.world_model, "world_model", 16384)
        _text(self.self_model, "self_model", 16384)
        _text(self.focus, "focus", 8192)
        if type(self.unfinished_patterns) is not tuple or len(self.unfinished_patterns) > 8:
            raise ValueError("unfinished_patterns must contain at most 8 entries")
        for pattern in self.unfinished_patterns:
            _text(pattern, "unfinished_pattern", 2048)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "StructuredExperience":
        base = {"interpretation", "process_action", "strategy", "predicted_consequence", "checks", "uncertainty"}
        situated = {"world_model", "self_model", "focus", "unfinished_patterns"}
        if set(value) not in (base, base | situated) or type(value["checks"]) is not list:
            raise ValueError("structured experience schema differs")
        if situated <= set(value) and type(value["unfinished_patterns"]) is not list:
            raise ValueError("unfinished_patterns must be a list")

        def text(field: str, default: str, limit: int) -> str:
            # Shape normalization: a list becomes joined text, an empty value
            # becomes the default, and over-long text is trimmed to its bound.
            raw = value.get(field, default)
            if type(raw) is list:
                raw = " ".join(str(item) for item in raw)
            if type(raw) is not str or not raw.strip():
                raw = default
            return raw[:limit]

        checks = [str(item)[:2048] for item in value["checks"] if str(item).strip()][:8] or ["No explicit check supplied."]
        patterns = [str(item)[:2048] for item in value.get("unfinished_patterns", ()) if str(item).strip()][:8]
        return cls(
            interpretation=text("interpretation", "(none given)", 16384),
            # Meaning, not shape: what she intends to do must be her own word.
            process_action=(
                value["process_action"][:4096] if type(value.get("process_action")) is str else value.get("process_action")
            ),  # type: ignore[arg-type]
            strategy=text("strategy", "(none given)", 16384),
            predicted_consequence=text("predicted_consequence", "(none given)", 16384),
            checks=tuple(checks),
            uncertainty=value["uncertainty"],  # type: ignore[arg-type]
            world_model=text("world_model", "No situated world model supplied.", 16384),
            self_model=text("self_model", "No situated self model supplied.", 16384),
            focus=text("focus", "Address the current request.", 8192),
            unfinished_patterns=tuple(patterns),
        )


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    request_ref: str
    response: str
    status: str

    def __post_init__(self) -> None:
        _text(self.request_ref, "request_ref", 96)
        _text(self.response, "response", 16384, empty=True)
        if self.status not in {"COMPLETED", "CLARIFICATION_REQUIRED", "ERROR"}:
            raise ValueError("receipt status is unsupported")

    @property
    def receipt_ref(self) -> str:
        return content_ref(asdict(self))


@dataclass(frozen=True, slots=True)
class TraceSpan:
    stage: str
    input_refs: tuple[str, ...]
    output_ref: str
    parent_ref: str | None

    @property
    def span_ref(self) -> str:
        return content_ref(asdict(self))


class TraceLedger:
    def __init__(self, maximum: int = 64) -> None:
        if not 1 <= maximum <= 256:
            raise ValueError("trace maximum must be 1 through 256")
        self.maximum = maximum
        self._spans: list[TraceSpan] = []

    def append(self, stage: str, inputs: Sequence[str], output: object) -> str:
        if len(self._spans) >= self.maximum:
            raise RuntimeError("trace ceiling reached")
        output_ref = content_ref(output)
        parent = self._spans[-1].span_ref if self._spans else None
        span = TraceSpan(stage, tuple(inputs), output_ref, parent)
        self._spans.append(span)
        return span.span_ref

    @property
    def spans(self) -> tuple[TraceSpan, ...]:
        return tuple(self._spans)

    @property
    def ledger_ref(self) -> str:
        return content_ref([asdict(item) for item in self._spans])


class UtilityMemory:
    """MemRL-style utility learned after semantic retrieval, never instead of it."""

    def __init__(self, *, alpha: float = 0.25) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._values: dict[str, float] = {}

    def rank(self, candidates: Sequence[MemoryCandidate], *, removal: str = "FULL") -> tuple[MemoryCandidate, ...]:
        if removal not in REMOVALS:
            raise ValueError("unknown removal")
        valued = tuple(replace(item, utility=0.0 if removal == "NO_UTILITY" else self._values.get(item.record_ref, 0.0)) for item in candidates)
        return tuple(sorted(valued, key=lambda item: (-item.utility, item.semantic_distance, item.record_ref)))

    def update(self, record_refs: Sequence[str], consequence: ConsequenceVector, *, removal: str = "FULL") -> None:
        if removal in {"NO_UTILITY", "SHUFFLED_CONSEQUENCE"}:
            return
        target = consequence.learning_utility
        for record_ref in record_refs:
            old = self._values.get(record_ref, 0.0)
            self._values[record_ref] = old + self.alpha * (target - old)

    def snapshot(self) -> dict[str, float]:
        return dict(sorted(self._values.items()))


class SemanticMemory(Protocol):
    def recall(self, request: str, *, limit: int) -> Sequence[MemoryCandidate]: ...
    def store(self, content: str, provenance_refs: Sequence[str]) -> str: ...


class TemporalMemory(Protocol):
    def context(self) -> TemporalContext: ...
    def advance(self, event_ref: str) -> TemporalContext: ...


class ExperienceModel(Protocol):
    def generate_experience(self, request: str, memories: Sequence[MemoryCandidate], temporal: TemporalContext) -> StructuredExperience: ...
    def reflect(self, request: str, receipt: ExecutionReceipt, consequence: ConsequenceVector, experience: StructuredExperience) -> str: ...
    def consolidate(self, request: str, experience: StructuredExperience, reflection: str, consequence: ConsequenceVector) -> str: ...


class Cortex(Protocol):
    def execute(self, request: str, experience: StructuredExperience, memories: Sequence[MemoryCandidate]) -> ExecutionReceipt: ...


@dataclass(frozen=True, slots=True)
class PreparedCycle:
    cycle_ref: str
    request: str
    removal: str
    temporal: TemporalContext
    semantic_candidates: tuple[MemoryCandidate, ...]
    selected_memories: tuple[MemoryCandidate, ...]
    experience: StructuredExperience
    receipt: ExecutionReceipt
    trace_ref_before_feedback: str


@dataclass(frozen=True, slots=True)
class CompletedCycle:
    prepared: PreparedCycle
    consequence: ConsequenceVector
    reflection: str
    consolidation: str
    consolidation_ref: str
    next_temporal: TemporalContext
    utility_state: Mapping[str, float]
    trace_ref: str
    spans: tuple[TraceSpan, ...]


class HigherLevelExperienceCycle:
    def __init__(self, *, memory: SemanticMemory, temporal: TemporalMemory, utility: UtilityMemory, experience_model: ExperienceModel, cortex: Cortex, recall_limit: int = 4) -> None:
        if not 1 <= recall_limit <= 32:
            raise ValueError("recall_limit must be 1 through 32")
        self.memory, self.temporal, self.utility = memory, temporal, utility
        self.experience_model, self.cortex, self.recall_limit = experience_model, cortex, recall_limit
        self.trace = TraceLedger()

    def prepare(self, request: str, *, removal: str = "FULL") -> PreparedCycle:
        _text(request, "request", 16384)
        if removal not in REMOVALS:
            raise ValueError("unknown removal")
        temporal = TemporalContext(-1, None) if removal == "NO_TEMPORAL" else self.temporal.context()
        semantic = tuple(self.memory.recall(request, limit=self.recall_limit))
        self.trace.append("SEMANTIC_RECALL", (content_ref(request),), [asdict(x) for x in semantic])
        selected = self.utility.rank(semantic, removal=removal)
        self.trace.append("UTILITY_SELECTION", tuple(x.record_ref for x in semantic), [asdict(x) for x in selected])
        if removal == "NO_EXPERIENCE":
            experience = StructuredExperience("No generated experience supplied.", "TRY", "Solve from the request and cited evidence.", "Unknown until observed.", ("Verify the public result.",), 1.0)
        else:
            experience = self.experience_model.generate_experience(request, selected, temporal)
        self.trace.append("EXPERIENCE_GENERATION", tuple(x.record_ref for x in selected), asdict(experience))
        receipt = self.cortex.execute(request, experience, selected)
        self.trace.append("CORTEX_EXECUTION", (content_ref(request), content_ref(asdict(experience))), asdict(receipt))
        cycle_ref = content_ref({"request": request, "removal": removal, "receipt_ref": receipt.receipt_ref})
        return PreparedCycle(cycle_ref, request, removal, temporal, semantic, selected, experience, receipt, self.trace.ledger_ref)

    def observe(self, prepared: PreparedCycle, consequence: ConsequenceVector) -> CompletedCycle:
        if prepared.receipt.status != "COMPLETED":
            raise RuntimeError("only completed execution may receive learning consequence")
        self.trace.append("OBJECTIVE_CONSEQUENCE", (prepared.receipt.receipt_ref,), asdict(consequence))
        if prepared.removal == "NO_REFLECTION":
            reflection, consolidation = "REMOVED", "REMOVED"
        else:
            reflection = self.experience_model.reflect(prepared.request, prepared.receipt, consequence, prepared.experience)
            _text(reflection, "reflection", 8192)
            self.trace.append("REFLECTION", (prepared.receipt.receipt_ref, content_ref(asdict(consequence))), reflection)
            consolidation = self.experience_model.consolidate(prepared.request, prepared.experience, reflection, consequence)
            _text(consolidation, "consolidation", 8192)
        consolidation_ref = self.memory.store(consolidation, (prepared.cycle_ref, prepared.receipt.receipt_ref))
        self.trace.append("CONSOLIDATION", (prepared.cycle_ref,), {"record_ref": consolidation_ref, "content": consolidation})
        self.utility.update((item.record_ref for item in prepared.selected_memories), consequence, removal=prepared.removal)
        self.trace.append("UTILITY_UPDATE", tuple(item.record_ref for item in prepared.selected_memories), self.utility.snapshot())
        next_temporal = self.temporal.advance(consolidation_ref)
        self.trace.append("TEMPORAL_ADVANCE", (consolidation_ref,), asdict(next_temporal))
        return CompletedCycle(prepared, consequence, reflection, consolidation, consolidation_ref, next_temporal, self.utility.snapshot(), self.trace.ledger_ref, self.trace.spans)
