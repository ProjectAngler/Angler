"""Phase-gated public evaluation over three frozen synthetic task families.

This module owns presentation aliases, response parsing, and objective judging.
It deliberately exposes no answer-producing operation.  Evaluation replicate
seeds are injected as 32-byte values by the runner; only domain-separated
commitments leave the evaluator boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import hmac
import itertools
import json
import re
from typing import Literal, TypeAlias

from angler.procedures.records import ActionSchema, GroundAction, Record
from experiments.evaluators import causal_operator_suite as causal
from experiments.evaluators import glyph_machine_trace_suite as glyph
from experiments.evaluators import symbolic_procedure_transfer_suite as symbolic


SUITE_SCHEMA = "angler.high-level-multidomain.v1"
QUALIFICATION_IDENTITY = f"{SUITE_SCHEMA}-harness-qualification"
EVALUATION_IDENTITY = f"{SUITE_SCHEMA}-evaluation"
QUALIFICATION_SEED = 2_026_083_190

Family = Literal[
    "symbolic-demonstration-transfer",
    "glyph-machine",
    "causal-operator",
]
Phase = Literal["adaptation", "development", "final"]
Role = Literal[
    "adaptation",
    "development",
    "final",
    "surface",
    "random-feedback",
]
Purpose = Literal["qualification", "evaluation"]
SeedFamily: TypeAlias = Family | Literal["random-feedback"]
Arm = Literal[
    "FULL",
    "QWEN_ONLY",
    "RETRIEVAL_ONLY",
    "FROZEN_ORIGIN",
    "PROSPECTIVE_REMOVAL",
    "BACKEND_REMOVAL",
    "RANDOM_FEEDBACK",
    "QUALIFICATION",
]

FAMILIES: tuple[Family, ...] = (
    "symbolic-demonstration-transfer",
    "glyph-machine",
    "causal-operator",
)
SEED_FAMILIES: tuple[SeedFamily, ...] = (*FAMILIES, "random-feedback")
PHASES: tuple[Phase, ...] = ("adaptation", "development", "final")
EVALUATION_ARMS: tuple[Arm, ...] = (
    "FULL",
    "QWEN_ONLY",
    "RETRIEVAL_ONLY",
    "FROZEN_ORIGIN",
    "PROSPECTIVE_REMOVAL",
    "BACKEND_REMOVAL",
    "RANDOM_FEEDBACK",
)
QUALIFICATION_ARM: Arm = "QUALIFICATION"
ROLES: tuple[Role, ...] = (
    "adaptation",
    "development",
    "final",
    "surface",
    "random-feedback",
)
_SEED_ROLES: dict[SeedFamily, tuple[Role, ...]] = {
    "symbolic-demonstration-transfer": (
        "adaptation",
        "development",
        "final",
    ),
    "glyph-machine": ("adaptation", "development", "final", "surface"),
    "causal-operator": ("adaptation", "development", "final"),
    "random-feedback": ("random-feedback",),
}

_COUNTS: dict[Family, dict[Phase, int]] = {
    "symbolic-demonstration-transfer": {
        "adaptation": 4,
        "development": 2,
        "final": 4,
    },
    "glyph-machine": {
        "adaptation": 2,
        "development": 2,
        "final": 4,
    },
    "causal-operator": {
        "adaptation": 6,
        "development": 6,
        "final": 12,
    },
}

_SEED_BYTES = 32
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GLYPH_TOKEN = re.compile(r"^A_[0-9a-f]{16}$")
_CAUSAL_ACTION = re.compile(
    r"^ACT\((E[1-9][0-9]*),(L[1-9][0-9]*),(L[1-9][0-9]*)\)$"
)
_SEED_COMMITMENT_DOMAIN = (
    b"angler.high-level-multidomain.v1\x00replicate-seed\x00"
)
_PRIVATE_COMMITMENT_DOMAIN = (
    b"angler.high-level-multidomain.v1\x00private-task\x00"
)
_RANDOM_FEEDBACK_COMMITMENT_DOMAIN = (
    b"angler.high-level-multidomain.v1\x00random-feedback-schedule\x00"
)


@dataclass(frozen=True, slots=True)
class PublicEvaluationTask:
    """One immutable model-facing task with no raw seed or process-private binding."""

    task_id: str
    replicate: str
    replicate_commitment: str
    family: Family
    phase: Phase
    ordinal: int
    payload_json: str
    public_commitment: str

    def __post_init__(self) -> None:
        _require_digest(self.task_id, "task_id")
        _require_digest(self.replicate_commitment, "replicate_commitment")
        _require_digest(self.public_commitment, "public_commitment")
        if not isinstance(self.replicate, str) or not re.fullmatch(
            r"(?:qualification|replicate)-[0-9]{2}", self.replicate
        ):
            raise ValueError("replicate must be a canonical public label")
        _validate_family(self.family)
        _validate_phase(self.phase)
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("ordinal must be a non-negative integer")
        payload = _decode_canonical_json(self.payload_json, "payload_json")
        if payload.get("family") != self.family:
            raise ValueError("public payload family does not match task metadata")
        expected_public = _public_commitment(
            self.replicate,
            self.replicate_commitment,
            self.family,
            self.phase,
            self.ordinal,
            self.payload_json,
        )
        if self.public_commitment != expected_public:
            raise ValueError("public_commitment does not bind this task")
        expected_id = _task_id(
            self.replicate,
            self.family,
            self.phase,
            self.ordinal,
            self.public_commitment,
        )
        if self.task_id != expected_id:
            raise ValueError("task_id does not bind this task")

    def to_canonical(self) -> dict[str, object]:
        return {
            "family": self.family,
            "ordinal": self.ordinal,
            "payload": json.loads(self.payload_json),
            "phase": self.phase,
            "public_commitment": self.public_commitment,
            "replicate": self.replicate,
            "replicate_commitment": self.replicate_commitment,
            "schema": SUITE_SCHEMA,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class ParsedPublicResponse:
    """A strict parse result; invalid text is never normalized or repaired."""

    valid: bool
    tokens: tuple[str, ...]
    error: str | None

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise TypeError("valid must be bool")
        if type(self.tokens) is not tuple or any(
            not isinstance(item, str) for item in self.tokens
        ):
            raise TypeError("tokens must be an immutable text tuple")
        if self.valid:
            if self.error is not None:
                raise ValueError("a valid parse cannot expose an error")
        elif self.tokens or self.error != "MALFORMED_RESPONSE":
            raise ValueError("an invalid parse must expose only the generic error")


@dataclass(frozen=True, slots=True)
class ObjectiveJudgment:
    """Scalar-only objective feedback plus preserved raw submission evidence."""

    task_id: str
    arm: Arm
    attempt_receipt_ref: str
    raw_response: str
    response_commitment: str
    score: float
    disposition: Literal["SUCCESS", "UNSUCCESSFUL"]

    def __post_init__(self) -> None:
        _require_digest(self.task_id, "judgment task_id")
        _validate_arm(self.arm)
        _require_digest(self.attempt_receipt_ref, "attempt_receipt_ref")
        _require_digest(self.response_commitment, "response_commitment")
        if not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be text")
        if self.response_commitment != _response_commitment(
            self.task_id,
            self.arm,
            self.attempt_receipt_ref,
            self.raw_response,
        ):
            raise ValueError("response_commitment does not bind the judgment")
        if type(self.score) is not float or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be a finite scalar from zero to one")
        if self.score not in (0.0, 1.0):
            raise ValueError("primary score must be exact binary success")
        expected = "SUCCESS" if self.score == 1.0 else "UNSUCCESSFUL"
        if self.disposition != expected:
            raise ValueError("disposition must match the primary score")

    def to_canonical(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "attempt_receipt_ref": self.attempt_receipt_ref,
            "disposition": self.disposition,
            "raw_response": self.raw_response,
            "response_commitment": self.response_commitment,
            "score": self.score,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """One post-completion aggregate; never an individual-task measurement."""

    phase: Phase
    arm: Arm
    family: Family
    attempts: int
    binary_success_total: int
    pairwise_agreement_total: float | None

    def __post_init__(self) -> None:
        _validate_phase(self.phase)
        _validate_arm(self.arm)
        _validate_family(self.family)
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts <= 0
        ):
            raise ValueError("attempts must be a positive integer")
        if (
            isinstance(self.binary_success_total, bool)
            or not isinstance(self.binary_success_total, int)
            or not 0 <= self.binary_success_total <= self.attempts
        ):
            raise ValueError("binary_success_total is outside the attempt count")
        if self.family == "symbolic-demonstration-transfer":
            value = self.pairwise_agreement_total
            if (
                type(value) is not float
                or not 0.0 <= value <= float(self.attempts)
            ):
                raise ValueError("symbolic aggregate requires bounded pairwise total")
        elif self.pairwise_agreement_total is not None:
            raise ValueError("pairwise total belongs only to the symbolic family")

    def to_canonical(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "attempts": self.attempts,
            "binary_success_total": self.binary_success_total,
            "family": self.family,
            "pairwise_agreement_total": self.pairwise_agreement_total,
            "phase": self.phase,
        }


@dataclass(frozen=True, slots=True)
class FinalMetrics:
    """Aggregate-only metric snapshot available after the full suite closes."""

    identity: str
    purpose: Purpose
    aggregates: tuple[MetricAggregate, ...]

    def __post_init__(self) -> None:
        expected_identity = (
            QUALIFICATION_IDENTITY
            if self.purpose == "qualification"
            else EVALUATION_IDENTITY
        )
        if self.identity != expected_identity:
            raise ValueError("metrics identity does not match purpose")
        if type(self.aggregates) is not tuple or not self.aggregates or any(
            not isinstance(item, MetricAggregate) for item in self.aggregates
        ):
            raise TypeError("aggregates must contain metric records")
        keys = tuple(
            (item.phase, item.arm, item.family) for item in self.aggregates
        )
        if len(set(keys)) != len(keys):
            raise ValueError("metric aggregate keys must be unique")

    @property
    def digest(self) -> str:
        return _digest("final-metrics", self.to_canonical())

    def to_canonical(self) -> dict[str, object]:
        return {
            "aggregates": [item.to_canonical() for item in self.aggregates],
            "identity": self.identity,
            "purpose": self.purpose,
            "schema": SUITE_SCHEMA,
        }


@dataclass(frozen=True, slots=True)
class TaskCommitment:
    """Public and keyed-private bindings without task or answer payloads."""

    task_id: str
    replicate_commitment: str
    family: Family
    phase: Phase
    ordinal: int
    public_commitment: str
    private_commitment: str

    def __post_init__(self) -> None:
        for label, value in (
            ("task_id", self.task_id),
            ("replicate_commitment", self.replicate_commitment),
            ("public_commitment", self.public_commitment),
            ("private_commitment", self.private_commitment),
        ):
            _require_digest(value, label)
        _validate_family(self.family)
        _validate_phase(self.phase)
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("ordinal must be a non-negative integer")

    def to_canonical(self) -> dict[str, object]:
        return {
            "family": self.family,
            "ordinal": self.ordinal,
            "phase": self.phase,
            "private_commitment": self.private_commitment,
            "public_commitment": self.public_commitment,
            "replicate_commitment": self.replicate_commitment,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class RandomFeedbackCommitment:
    """Manifest-safe binding of one balanced process-private schedule."""

    replicate_commitment: str
    schedule_commitment: str

    def __post_init__(self) -> None:
        _require_digest(self.replicate_commitment, "replicate_commitment")
        _require_digest(self.schedule_commitment, "schedule_commitment")

    def to_canonical(self) -> dict[str, str]:
        return {
            "replicate_commitment": self.replicate_commitment,
            "schedule_commitment": self.schedule_commitment,
        }


@dataclass(frozen=True, slots=True)
class EvaluatorCommitments:
    """Manifest-safe task metadata for one evaluator construction."""

    identity: str
    purpose: Purpose
    qualification_seed: int | None
    replicate_commitments: tuple[str, ...]
    tasks: tuple[TaskCommitment, ...]
    random_feedback: tuple[RandomFeedbackCommitment, ...]

    def __post_init__(self) -> None:
        expected_identity = (
            QUALIFICATION_IDENTITY
            if self.purpose == "qualification"
            else EVALUATION_IDENTITY
        )
        if self.identity != expected_identity:
            raise ValueError("identity does not match evaluator purpose")
        if self.purpose == "qualification":
            if self.qualification_seed != QUALIFICATION_SEED:
                raise ValueError("qualification seed is not the frozen identity")
            expected_replicates = 1
        elif self.purpose == "evaluation":
            if self.qualification_seed is not None:
                raise ValueError("evaluation commitments cannot expose raw seeds")
            expected_replicates = 2
        else:
            raise ValueError("purpose must be qualification or evaluation")
        if type(self.replicate_commitments) is not tuple or len(
            self.replicate_commitments
        ) != expected_replicates:
            raise ValueError("replicate commitment count is inconsistent")
        if len(set(self.replicate_commitments)) != expected_replicates:
            raise ValueError("replicate commitments must be distinct")
        for value in self.replicate_commitments:
            _require_digest(value, "replicate commitment")
        if type(self.tasks) is not tuple or any(
            not isinstance(item, TaskCommitment) for item in self.tasks
        ):
            raise TypeError("tasks must be an immutable TaskCommitment tuple")
        task_ids = tuple(item.task_id for item in self.tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task commitments contain duplicate identities")
        public_values = tuple(item.public_commitment for item in self.tasks)
        private_values = tuple(item.private_commitment for item in self.tasks)
        if len(set(public_values)) != len(public_values) or len(
            set(private_values)
        ) != len(private_values):
            raise ValueError("public and private task commitments must be unique")
        available_phases: tuple[Phase, ...] = (
            ("adaptation", "development")
            if self.purpose == "qualification"
            else PHASES
        )
        for replicate_commitment in self.replicate_commitments:
            for family in FAMILIES:
                for phase in available_phases:
                    selected = tuple(
                        item
                        for item in self.tasks
                        if item.replicate_commitment == replicate_commitment
                        and item.family == family
                        and item.phase == phase
                    )
                    expected = _COUNTS[family][phase]
                    if len(selected) != expected or {
                        item.ordinal for item in selected
                    } != set(range(expected)):
                        raise ValueError(
                            "task commitments do not match frozen family counts"
                        )
        if any(
            item.replicate_commitment not in self.replicate_commitments
            or item.phase not in available_phases
            for item in self.tasks
        ):
            raise ValueError("task commitment is outside the declared evaluator")
        if type(self.random_feedback) is not tuple or len(
            self.random_feedback
        ) != expected_replicates:
            raise ValueError("random-feedback commitment count is inconsistent")
        if any(
            not isinstance(item, RandomFeedbackCommitment)
            for item in self.random_feedback
        ):
            raise TypeError("random_feedback must contain commitment records")
        if tuple(
            item.replicate_commitment for item in self.random_feedback
        ) != self.replicate_commitments:
            raise ValueError("random-feedback commitments must follow replicate order")

    @property
    def digest(self) -> str:
        return _digest("evaluator-commitments", self.to_canonical())

    def to_canonical(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "purpose": self.purpose,
            "qualification_seed": self.qualification_seed,
            "random_feedback": [
                item.to_canonical() for item in self.random_feedback
            ],
            "replicate_commitments": list(self.replicate_commitments),
            "schema": SUITE_SCHEMA,
            "tasks": [item.to_canonical() for item in self.tasks],
        }


@dataclass(frozen=True, slots=True, repr=False)
class _TaskBinding:
    public: PublicEvaluationTask
    source: object
    aliases: tuple[tuple[str, object], ...]
    private_commitment: str


@dataclass(frozen=True, slots=True, repr=False)
class _RandomFeedbackBinding:
    replicate_commitment: str
    assignments: tuple[tuple[str, float], ...]
    schedule_commitment: str


@dataclass(frozen=True, slots=True, repr=False)
class _JudgmentBinding:
    judgment: ObjectiveJudgment
    pairwise_agreement: float | None


class HighLevelMultidomainEvaluator:
    """Stateful release boundary with process-private objective judges."""

    def __init__(
        self,
        purpose: Purpose,
        replicate_keys: Sequence[bytes],
        expected_replicate_commitments: Sequence[str],
        expected_final_admission: str | None,
    ) -> None:
        if purpose not in ("qualification", "evaluation"):
            raise ValueError("purpose must be qualification or evaluation")
        keys = tuple(replicate_keys)
        commitments = tuple(expected_replicate_commitments)
        expected_count = 1 if purpose == "qualification" else 2
        if len(keys) != expected_count or len(commitments) != expected_count:
            raise ValueError("replicate count does not match evaluator purpose")
        for key in keys:
            _validate_replicate_key(key)
        if len(set(keys)) != len(keys):
            raise ValueError("raw replicate seeds must be distinct")
        computed = tuple(commit_replicate_seed(key) for key in keys)
        if commitments != computed:
            raise ValueError("injected replicate seed commitments do not match")
        if purpose == "qualification":
            if expected_final_admission is not None:
                raise ValueError("qualification cannot bind a final admission")
        else:
            if expected_final_admission is None:
                raise ValueError("evaluation requires a final admission digest")
            _require_digest(expected_final_admission, "expected_final_admission")

        self._purpose = purpose
        self._identity = (
            QUALIFICATION_IDENTITY
            if purpose == "qualification"
            else EVALUATION_IDENTITY
        )
        phases: tuple[Phase, ...] = (
            ("adaptation", "development")
            if purpose == "qualification"
            else PHASES
        )
        symbolic_mechanisms, glyph_mechanisms = _select_replicate_mechanisms(
            purpose,
            keys,
        )
        bindings: list[_TaskBinding] = []
        random_feedback: list[_RandomFeedbackBinding] = []
        for zero_index, (
            key,
            commitment,
            symbolic_mechanism,
            glyph_mechanism,
        ) in enumerate(
            zip(
                keys,
                commitments,
                symbolic_mechanisms,
                glyph_mechanisms,
                strict=True,
            )
        ):
            index = zero_index + 1
            replicate = (
                f"qualification-{index:02d}"
                if purpose == "qualification"
                else f"replicate-{index:02d}"
            )
            replicate_bindings = _make_replicate_bindings(
                purpose=purpose,
                replicate=replicate,
                replicate_key=key,
                replicate_commitment=commitment,
                replicate_index=zero_index,
                phases=phases,
                symbolic_mechanism=symbolic_mechanism,
                glyph_mechanism=glyph_mechanism,
            )
            bindings.extend(replicate_bindings)
            random_feedback.append(
                _make_random_feedback_binding(
                    key,
                    commitment,
                    tuple(
                        item.public.task_id
                        for item in replicate_bindings
                        if item.public.phase == "adaptation"
                    ),
                )
            )
        by_id = {item.public.task_id: item for item in bindings}
        if len(by_id) != len(bindings):
            raise RuntimeError("generated task identities overlap")
        public_commitments = [item.public.public_commitment for item in bindings]
        if len(set(public_commitments)) != len(public_commitments):
            raise RuntimeError("generated public task commitments overlap")
        public_payloads = [item.public.payload_json for item in bindings]
        if len(set(public_payloads)) != len(public_payloads):
            raise RuntimeError("generated public task payloads overlap")

        self._bindings = by_id
        self._ordered_ids = tuple(item.public.task_id for item in bindings)
        self._released: set[Phase] = set()
        self._completed: set[Phase] = set()
        self._expected_final_admission = expected_final_admission
        self._final_admission: str | None = None
        self._consumed_pairs: set[tuple[str, Arm]] = set()
        self._consumed_attempt_receipts: set[str] = set()
        self._judgments: dict[tuple[str, Arm], _JudgmentBinding] = {}
        self._random_feedback = tuple(random_feedback)
        random_feedback_by_task = {
            task_id: value
            for schedule in random_feedback
            for task_id, value in schedule.assignments
        }
        if len(random_feedback_by_task) != 12 * expected_count:
            raise RuntimeError("random-feedback task identities overlap")
        self._random_feedback_by_task = random_feedback_by_task
        task_commitments = tuple(
            TaskCommitment(
                task_id=item.public.task_id,
                replicate_commitment=item.public.replicate_commitment,
                family=item.public.family,
                phase=item.public.phase,
                ordinal=item.public.ordinal,
                public_commitment=item.public.public_commitment,
                private_commitment=item.private_commitment,
            )
            for item in bindings
        )
        self._commitments = EvaluatorCommitments(
            identity=self._identity,
            purpose=purpose,
            qualification_seed=(
                QUALIFICATION_SEED if purpose == "qualification" else None
            ),
            replicate_commitments=commitments,
            tasks=task_commitments,
            random_feedback=tuple(
                RandomFeedbackCommitment(
                    item.replicate_commitment,
                    item.schedule_commitment,
                )
                for item in random_feedback
            ),
        )

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def purpose(self) -> Purpose:
        return self._purpose

    @property
    def commitments(self) -> EvaluatorCommitments:
        return self._commitments

    @property
    def released_phases(self) -> tuple[Phase, ...]:
        return tuple(phase for phase in PHASES if phase in self._released)

    @property
    def completed_phases(self) -> tuple[Phase, ...]:
        return tuple(phase for phase in PHASES if phase in self._completed)

    def random_feedback_schedule(
        self,
        replicate_commitment: str,
    ) -> tuple[tuple[str, float], ...]:
        """Return one released replicate's balanced task-ID control schedule."""

        _require_digest(replicate_commitment, "replicate_commitment")
        if "adaptation" not in self._released:
            raise RuntimeError("random feedback is unavailable before adaptation release")
        matches = tuple(
            item
            for item in self._random_feedback
            if item.replicate_commitment == replicate_commitment
        )
        if len(matches) != 1:
            raise ValueError("replicate_commitment is not owned by this evaluator")
        return matches[0].assignments

    def random_feedback_for(self, task_id: str) -> float:
        """Look up one released adaptation task's predeclared control scalar."""

        if not isinstance(task_id, str):
            raise TypeError("task_id must be text")
        if "adaptation" not in self._released:
            raise RuntimeError("random feedback is unavailable before adaptation release")
        try:
            return self._random_feedback_by_task[task_id]
        except KeyError as error:
            raise ValueError("task_id is not an adaptation task owned here") from error

    def release_phase(self, phase: Phase) -> tuple[PublicEvaluationTask, ...]:
        """Release one phase only after every preceding phase is complete."""

        _validate_phase(phase)
        available = (
            ("adaptation", "development")
            if self._purpose == "qualification"
            else PHASES
        )
        if phase not in available:
            raise ValueError("qualification cannot materialize final tasks")
        position = available.index(phase)
        predecessors = available[:position]
        if any(item not in self._completed for item in predecessors):
            raise RuntimeError("the preceding phase is not complete")
        if phase == "final" and self._final_admission is None:
            raise RuntimeError("final release requires an explicit admission digest")
        self._released.add(phase)
        return tuple(
            self._bindings[task_id].public
            for task_id in self._ordered_ids
            if self._bindings[task_id].public.phase == phase
        )

    def complete_phase(self, phase: Phase) -> None:
        """Advance only after every exact task/arm judgment pair exists."""

        _validate_phase(phase)
        if phase not in self._released:
            raise RuntimeError("a phase must be released before it is completed")
        task_ids = tuple(
            task_id
            for task_id in self._ordered_ids
            if self._bindings[task_id].public.phase == phase
        )
        expected = {
            (task_id, arm)
            for task_id in task_ids
            for arm in _expected_arms(self._purpose, phase)
        }
        actual = {
            key
            for key in self._judgments
            if self._bindings[key[0]].public.phase == phase
        }
        if actual != expected:
            raise RuntimeError(
                "phase completion requires every exact task/arm judgment pair"
            )
        self._completed.add(phase)

    def admit_final(self, admission_digest: str) -> None:
        """Record runner-side manifest/source admission without inspecting it."""

        if self._purpose != "evaluation":
            raise ValueError("qualification has no final admission")
        _require_digest(admission_digest, "admission_digest")
        if "development" not in self._completed:
            raise RuntimeError("development must complete before final admission")
        if admission_digest != self._expected_final_admission:
            raise ValueError("admission_digest does not match construction binding")
        if self._final_admission is not None and self._final_admission != admission_digest:
            raise RuntimeError("final admission is immutable")
        self._final_admission = admission_digest

    def final_metrics(self) -> FinalMetrics:
        """Return aggregate metrics only after every available phase completes."""

        available: tuple[Phase, ...] = (
            ("adaptation", "development")
            if self._purpose == "qualification"
            else PHASES
        )
        if any(phase not in self._completed for phase in available):
            raise RuntimeError("final metrics are unavailable before suite completion")
        aggregates: list[MetricAggregate] = []
        for phase in available:
            for arm in _expected_arms(self._purpose, phase):
                for family in FAMILIES:
                    selected = tuple(
                        stored
                        for (task_id, stored_arm), stored in self._judgments.items()
                        if stored_arm == arm
                        and self._bindings[task_id].public.phase == phase
                        and self._bindings[task_id].public.family == family
                    )
                    if not selected:
                        raise RuntimeError("metric aggregation found an empty cell")
                    pairwise_total = (
                        float(
                            sum(
                                _require_pairwise(item.pairwise_agreement)
                                for item in selected
                            )
                        )
                        if family == "symbolic-demonstration-transfer"
                        else None
                    )
                    aggregates.append(
                        MetricAggregate(
                            phase=phase,
                            arm=arm,
                            family=family,
                            attempts=len(selected),
                            binary_success_total=sum(
                                int(item.judgment.score) for item in selected
                            ),
                            pairwise_agreement_total=pairwise_total,
                        )
                    )
        return FinalMetrics(
            identity=self._identity,
            purpose=self._purpose,
            aggregates=tuple(aggregates),
        )

    def judge_response(
        self,
        task_id: str,
        arm: Arm,
        attempt_receipt_ref: str,
        raw_response: str,
    ) -> ObjectiveJudgment:
        """Parse and judge one released task while returning scalar feedback only."""

        if not isinstance(task_id, str):
            raise TypeError("task_id must be text")
        _validate_arm(arm)
        _require_digest(attempt_receipt_ref, "attempt_receipt_ref")
        if not isinstance(raw_response, str):
            raise TypeError("raw_response must be text")
        try:
            binding = self._bindings[task_id]
        except KeyError as error:
            raise ValueError("task_id is not owned by this evaluator") from error
        if binding.public.phase not in self._released:
            raise RuntimeError("unreleased task cannot be judged before phase release")
        if arm not in _expected_arms(self._purpose, binding.public.phase):
            raise ValueError("arm is not admitted for this purpose and phase")
        judgment_key = (task_id, arm)
        if judgment_key in self._consumed_pairs:
            raise RuntimeError("task/arm judgment is already consumed")
        if attempt_receipt_ref in self._consumed_attempt_receipts:
            raise RuntimeError("attempt receipt is already consumed")
        self._consumed_pairs.add(judgment_key)
        self._consumed_attempt_receipts.add(attempt_receipt_ref)

        parsed = parse_public_response(binding.public, raw_response)
        if not parsed.valid:
            primary = 0.0
            pairwise = (
                0.0
                if binding.public.family == "symbolic-demonstration-transfer"
                else None
            )
        elif binding.public.family == "symbolic-demonstration-transfer":
            pair = binding.source
            if not isinstance(pair, symbolic.GeneratedDemonstrationProcedureTask):
                raise RuntimeError("symbolic evaluator binding is inconsistent")
            pairwise = float(
                symbolic.score_demonstration_procedure_answer(
                    pair.learner,
                    pair.hidden,
                    parsed.tokens,
                )
            )
            primary = float(pairwise == 1.0)
        elif binding.public.family == "glyph-machine":
            pair = binding.source
            if not isinstance(pair, glyph.GeneratedGlyphMachineTask):
                raise RuntimeError("glyph evaluator binding is inconsistent")
            aliases = dict(binding.aliases)
            actions = tuple(
                _require_action_schema(aliases[token]).ground()
                for token in parsed.tokens
            )
            committed = glyph.commit_glyph_procedure(
                pair.learner,
                actions,
                stopped=True,
            )
            primary = glyph.judge_glyph_procedure_attempt(pair, committed)
            pairwise = None
        else:
            challenge = binding.source
            if not isinstance(challenge, causal.OperatorChallenge):
                raise RuntimeError("causal evaluator binding is inconsistent")
            aliases = dict(binding.aliases)
            actions: list[GroundAction] = []
            schema = challenge.allowed_action_schemas[0]
            for token in parsed.tokens:
                match = _CAUSAL_ACTION.fullmatch(token)
                if match is None:
                    raise RuntimeError("validated causal parse became inconsistent")
                entity, source, destination = (
                    aliases[item] for item in match.groups()
                )
                if not all(isinstance(item, str) for item in (entity, source, destination)):
                    raise RuntimeError("causal alias binding is inconsistent")
                actions.append(schema.ground(entity, source, destination))
            committed = causal.commit_action_sequence(challenge, tuple(actions))
            result = causal.evaluate_committed_sequence(challenge, committed)
            primary = float(result.success)
            pairwise = None

        judgment = ObjectiveJudgment(
            task_id=task_id,
            arm=arm,
            attempt_receipt_ref=attempt_receipt_ref,
            raw_response=raw_response,
            response_commitment=_response_commitment(
                task_id,
                arm,
                attempt_receipt_ref,
                raw_response,
            ),
            score=float(primary),
            disposition=("SUCCESS" if primary == 1.0 else "UNSUCCESSFUL"),
        )
        self._judgments[judgment_key] = _JudgmentBinding(judgment, pairwise)
        return judgment


def make_qualification_evaluator() -> HighLevelMultidomainEvaluator:
    """Construct the fixed public-seed qualification without any final phase."""

    key = QUALIFICATION_SEED.to_bytes(_SEED_BYTES, "big")
    commitment = commit_replicate_seed(key)
    return HighLevelMultidomainEvaluator(
        "qualification",
        (key,),
        (commitment,),
        None,
    )


def make_evaluation_evaluator(
    raw_replicate_seeds: Sequence[bytes],
    expected_seed_commitments: Sequence[str],
    expected_final_admission: str,
) -> HighLevelMultidomainEvaluator:
    """Construct the final evaluator from two process-private persisted seeds."""

    return HighLevelMultidomainEvaluator(
        "evaluation",
        raw_replicate_seeds,
        expected_seed_commitments,
        expected_final_admission,
    )


def commit_replicate_seed(raw_seed: bytes) -> str:
    """Return the sole public projection of one injected evaluation seed."""

    _validate_replicate_key(raw_seed)
    return "sha256:" + hashlib.sha256(
        _SEED_COMMITMENT_DOMAIN + raw_seed
    ).hexdigest()


def derive_family_seed(raw_seed: bytes, family: SeedFamily, role: Role) -> int:
    """Derive the frozen 64-bit family/role seed using the leaf's HMAC rule."""

    _validate_replicate_key(raw_seed)
    _validate_seed_family(family)
    if role not in ROLES:
        raise ValueError("role is not declared by the suite")
    if role not in _SEED_ROLES[family]:
        raise ValueError("role is not applicable to this seed family")
    message = (
        f"{SUITE_SCHEMA}\x00{family}\x00{role}"
    ).encode("utf-8")
    return int.from_bytes(hmac.new(raw_seed, message, hashlib.sha256).digest()[:8], "big")


def expected_phase_count(purpose: Purpose, phase: Phase) -> int:
    """Return the preregistered public-task denominator for one phase."""

    if purpose not in ("qualification", "evaluation"):
        raise ValueError("purpose must be qualification or evaluation")
    _validate_phase(phase)
    if purpose == "qualification" and phase == "final":
        return 0
    replicate_count = 1 if purpose == "qualification" else 2
    return replicate_count * sum(_COUNTS[family][phase] for family in FAMILIES)


def render_public_task(task: PublicEvaluationTask) -> str:
    """Return the canonical model-facing JSON without adding prompt context."""

    if not isinstance(task, PublicEvaluationTask):
        raise TypeError("task must be a PublicEvaluationTask")
    return task.payload_json


def parse_public_response(
    task: PublicEvaluationTask,
    raw_response: str,
) -> ParsedPublicResponse:
    """Strictly parse the declared grammar; never strip, infer, or repair text."""

    if not isinstance(task, PublicEvaluationTask):
        raise TypeError("task must be a PublicEvaluationTask")
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be text")
    payload = json.loads(task.payload_json)
    if not raw_response or raw_response != raw_response.strip():
        return _malformed()

    if task.family == "symbolic-demonstration-transfer":
        parts = tuple(raw_response.split(","))
        allowed = tuple(payload["query"])
        if (
            len(parts) != 5
            or len(set(parts)) != 5
            or any(item not in allowed for item in parts)
            or ",".join(parts) != raw_response
        ):
            return _malformed()
        return ParsedPublicResponse(True, parts, None)

    if task.family == "glyph-machine":
        parts = tuple(raw_response.split(","))
        maximum = int(payload["maximum_steps"])
        allowed = set(payload["actions"])
        if (
            len(parts) < 1
            or parts[-1] != "STOP"
            or parts.count("STOP") != 1
            or len(parts[:-1]) > maximum
            or any(_GLYPH_TOKEN.fullmatch(item) is None for item in parts[:-1])
            or any(item not in allowed for item in parts[:-1])
            or ",".join(parts) != raw_response
        ):
            return _malformed()
        return ParsedPublicResponse(True, parts[:-1], None)

    parts = tuple(raw_response.split(";"))
    maximum = int(payload["maximum_steps"])
    entity_aliases = set(payload["entities"])
    location_aliases = set(payload["locations"])
    if not 1 <= len(parts) <= maximum or ";".join(parts) != raw_response:
        return _malformed()
    for item in parts:
        match = _CAUSAL_ACTION.fullmatch(item)
        if match is None:
            return _malformed()
        entity, source, destination = match.groups()
        if entity not in entity_aliases or not {source, destination} <= location_aliases:
            return _malformed()
    return ParsedPublicResponse(True, parts, None)


def _select_replicate_mechanisms(
    purpose: Purpose,
    replicate_keys: tuple[bytes, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    partition: symbolic.PermutationPartition = (
        "development" if purpose == "qualification" else "final"
    )
    remaining_symbolic = list(
        symbolic.demonstration_permutation_partition(partition)
    )
    remaining_glyph = list(glyph.glyph_machine_mechanism_partition(partition))
    selected_symbolic: list[tuple[int, ...]] = []
    selected_glyph: list[str] = []
    for replicate_key in replicate_keys:
        symbolic_index = derive_family_seed(
            replicate_key,
            "symbolic-demonstration-transfer",
            "adaptation",
        ) % len(remaining_symbolic)
        selected_symbolic.append(remaining_symbolic.pop(symbolic_index))
        glyph_index = derive_family_seed(
            replicate_key,
            "glyph-machine",
            "adaptation",
        ) % len(remaining_glyph)
        selected_glyph.append(remaining_glyph.pop(glyph_index))
    return tuple(selected_symbolic), tuple(selected_glyph)


def _make_replicate_bindings(
    *,
    purpose: Purpose,
    replicate: str,
    replicate_key: bytes,
    replicate_commitment: str,
    replicate_index: int,
    phases: tuple[Phase, ...],
    symbolic_mechanism: tuple[int, ...],
    glyph_mechanism: str,
) -> tuple[_TaskBinding, ...]:
    symbolic_partition: symbolic.PermutationPartition = (
        "development" if purpose == "qualification" else "final"
    )
    symbolic_adaptation_seed = derive_family_seed(
        replicate_key,
        "symbolic-demonstration-transfer",
        "adaptation",
    )
    query_count = 2 if purpose == "qualification" else 6
    symbolic_stream = symbolic.make_demonstration_procedure_transfer_stream(
        symbolic_adaptation_seed,
        supports_per_procedure=4,
        queries_per_procedure=query_count,
        position_permutation=symbolic_mechanism,
        mechanism_partition=symbolic_partition,
        expose_transform_demonstrations=True,
    )
    symbolic_by_phase: dict[
        Phase, tuple[symbolic.GeneratedDemonstrationProcedureTask, ...]
    ] = {
        "adaptation": tuple(
            pair
            for pair in symbolic_stream.supports
            if pair.learner.demonstrations_visible
        ),
        "development": symbolic_stream.queries[:2],
        "final": (
            () if purpose == "qualification" else symbolic_stream.queries[2:]
        ),
    }

    glyph_partition: glyph.GlyphMachinePartition = (
        "development" if purpose == "qualification" else "final"
    )
    glyph_adaptation_seed = derive_family_seed(
        replicate_key, "glyph-machine", "adaptation"
    )
    surface_seed = derive_family_seed(replicate_key, "glyph-machine", "surface")
    glyph_stream = glyph.make_glyph_machine_trace_stream(
        glyph_adaptation_seed,
        surface_seed=surface_seed,
        supports=2,
        queries=query_count,
        observations_per_support=2,
        maximum_steps=4,
        mechanism_commitment=glyph_mechanism,
        mechanism_partition=glyph_partition,
    )
    glyph_by_phase: dict[Phase, tuple[glyph.GeneratedGlyphMachineTask, ...]] = {
        "adaptation": glyph_stream.supports,
        "development": glyph_stream.queries[:2],
        "final": (() if purpose == "qualification" else glyph_stream.queries[2:]),
    }

    bindings: list[_TaskBinding] = []
    for phase in phases:
        symbolic_pairs = symbolic_by_phase[phase]
        glyph_pairs = glyph_by_phase[phase]
        causal_count = _COUNTS["causal-operator"][phase]
        cases_per_domain = causal_count // len(causal.SUPPORTED_DOMAINS)
        challenges = causal.make_heldout_operator_suite(
            derive_family_seed(replicate_key, "causal-operator", phase),
            cases_per_domain=cases_per_domain,
        )
        if (
            len(symbolic_pairs)
            != _COUNTS["symbolic-demonstration-transfer"][phase]
            or len(glyph_pairs) != _COUNTS["glyph-machine"][phase]
            or len(challenges) != causal_count
        ):
            raise RuntimeError("one or more family task counts are inconsistent")

        symbolic_bindings = tuple(
            _make_symbolic_binding(
                pair,
                replicate=replicate,
                replicate_key=replicate_key,
                replicate_commitment=replicate_commitment,
                phase=phase,
                ordinal=ordinal,
            )
            for ordinal, pair in enumerate(symbolic_pairs)
        )
        glyph_bindings = tuple(
            _make_glyph_binding(
                pair,
                replicate=replicate,
                replicate_key=replicate_key,
                replicate_commitment=replicate_commitment,
                phase=phase,
                ordinal=ordinal,
            )
            for ordinal, pair in enumerate(glyph_pairs)
        )
        causal_bindings = tuple(
            _make_causal_binding(
                challenge,
                replicate=replicate,
                replicate_key=replicate_key,
                replicate_commitment=replicate_commitment,
                phase=phase,
                ordinal=ordinal,
                surface_slot=(
                    replicate_index * 4
                    + _causal_phase_slot(phase, ordinal, cases_per_domain)
                ),
            )
            for ordinal, challenge in enumerate(challenges)
        )
        bindings.extend(
            _round_robin((symbolic_bindings, glyph_bindings, causal_bindings))
        )
    return tuple(bindings)


def _make_symbolic_binding(
    pair: symbolic.GeneratedDemonstrationProcedureTask,
    *,
    replicate: str,
    replicate_key: bytes,
    replicate_commitment: str,
    phase: Phase,
    ordinal: int,
) -> _TaskBinding:
    learner = pair.learner
    payload = {
        "demonstrations": [
            {
                "input": list(item.input_symbols),
                "output": list(item.output_symbols),
            }
            for item in learner.demonstrations
        ],
        "family": "symbolic-demonstration-transfer",
        "instruction": (
            "Return exactly the five query symbols in the acquired order, "
            "comma-separated with no spaces or other text."
        ),
        "procedure": "P1",
        "query": [item.symbol for item in learner.items],
    }
    private = {
        "public": learner.to_canonical(),
        "source_instance_id": pair.hidden.source_instance_id,
        "transform": list(pair.hidden.source_solution.position_permutation),
    }
    return _make_binding(
        source=pair,
        aliases=(),
        payload=payload,
        private=private,
        replicate=replicate,
        replicate_key=replicate_key,
        replicate_commitment=replicate_commitment,
        family="symbolic-demonstration-transfer",
        phase=phase,
        ordinal=ordinal,
    )


def _make_glyph_binding(
    pair: glyph.GeneratedGlyphMachineTask,
    *,
    replicate: str,
    replicate_key: bytes,
    replicate_commitment: str,
    phase: Phase,
    ordinal: int,
) -> _TaskBinding:
    learner = pair.learner
    state_aliases = {
        state.digest: _glyph_alias("S", state.digest) for state in learner.states
    }
    action_aliases = {
        action.digest: _glyph_alias("A", action.digest) for action in learner.actions
    }
    if len(set(state_aliases.values())) != len(state_aliases) or len(
        set(action_aliases.values())
    ) != len(action_aliases):
        raise RuntimeError("glyph public alias collision")
    goal_state = next(
        state for state in learner.states if state.records == learner.goal.required
    )
    observations = []
    for trace in learner.observations:
        observations.append(
            {
                "origin": state_aliases[trace.initial.digest],
                "transitions": [
                    {
                        "action": action_aliases[item.action.schema.digest],
                        "after": state_aliases[item.after.digest],
                        "before": state_aliases[item.before.digest],
                    }
                    for item in trace.transitions
                ],
            }
        )
    payload = {
        "actions": list(action_aliases.values()),
        "family": "glyph-machine",
        "goal": state_aliases[goal_state.digest],
        "instruction": (
            "Return a comma-separated action-alias sequence ending in STOP, "
            "with no spaces or other text."
        ),
        "maximum_steps": learner.max_steps,
        "observations": observations,
        "origin": state_aliases[learner.origin.digest],
        "states": list(state_aliases.values()),
    }
    private = {
        "action_digests": list(pair.hidden.action_digests),
        "mechanism_commitment": pair.hidden.mechanism_commitment,
        "public": learner.to_canonical(),
        "state_digests": list(pair.hidden.state_digests),
        "transition_rows": [list(row) for row in pair.hidden.transition_rows],
    }
    aliases: tuple[tuple[str, object], ...] = tuple(
        (action_aliases[action.digest], action) for action in learner.actions
    )
    return _make_binding(
        source=pair,
        aliases=aliases,
        payload=payload,
        private=private,
        replicate=replicate,
        replicate_key=replicate_key,
        replicate_commitment=replicate_commitment,
        family="glyph-machine",
        phase=phase,
        ordinal=ordinal,
    )


def _make_causal_binding(
    challenge: causal.OperatorChallenge,
    *,
    replicate: str,
    replicate_key: bytes,
    replicate_commitment: str,
    phase: Phase,
    ordinal: int,
    surface_slot: int,
) -> _TaskBinding:
    records = challenge.origin.records + challenge.goal.required + challenge.goal.forbidden
    locations = tuple(
        sorted(
            {
                argument
                for record in records
                for argument in record.arguments
                if argument.startswith("position_")
            },
            key=lambda value: int(value.removeprefix("position_")),
        )
    )
    entities = tuple(
        sorted(
            {
                argument
                for record in records
                for argument in record.arguments
                if not argument.startswith(("position_", "limit_"))
            }
        )
    )
    canonical_entity_aliases = tuple(
        f"E{index}" for index in range(1, len(entities) + 1)
    )
    canonical_location_aliases = tuple(
        f"L{index}" for index in range(1, len(locations) + 1)
    )
    surface_options = itertools.product(
        itertools.permutations(canonical_entity_aliases),
        itertools.permutations(canonical_location_aliases),
    )
    try:
        entity_surface, location_surface = next(
            itertools.islice(surface_options, surface_slot, None)
        )
    except StopIteration as error:
        raise RuntimeError("causal surface slot exceeds permutation space") from error
    entity_aliases = dict(zip(entities, entity_surface, strict=True))
    location_aliases = dict(zip(locations, location_surface, strict=True))
    payload = {
        "action_schema": "ACT(entity,source,destination)",
        "domain": challenge.domain,
        "entities": list(entity_aliases.values()),
        "family": "causal-operator",
        "goal": [
            _render_causal_record(record, entity_aliases, location_aliases)
            for record in challenge.goal.required
        ],
        "instruction": (
            "Return one or more ACT calls separated by semicolons, up to the "
            "maximum_steps ceiling, with no spaces or other text."
        ),
        "locations": list(location_aliases.values()),
        "maximum_steps": challenge.maximum_steps,
        "origin": [
            _render_causal_record(record, entity_aliases, location_aliases)
            for record in challenge.origin.records
        ],
    }
    private = {
        "allowed_action_schemas": [
            item.digest for item in challenge.allowed_action_schemas
        ],
        "case_id": challenge.case_id,
        "goal_digest": challenge.goal.digest,
        "origin_digest": challenge.origin.digest,
    }
    aliases = tuple(
        [(alias, value) for value, alias in entity_aliases.items()]
        + [(alias, value) for value, alias in location_aliases.items()]
    )
    return _make_binding(
        source=challenge,
        aliases=aliases,
        payload=payload,
        private=private,
        replicate=replicate,
        replicate_key=replicate_key,
        replicate_commitment=replicate_commitment,
        family="causal-operator",
        phase=phase,
        ordinal=ordinal,
    )


def _make_binding(
    *,
    source: object,
    aliases: tuple[tuple[str, object], ...],
    payload: dict[str, object],
    private: dict[str, object],
    replicate: str,
    replicate_key: bytes,
    replicate_commitment: str,
    family: Family,
    phase: Phase,
    ordinal: int,
) -> _TaskBinding:
    payload_json = _canonical_json(payload)
    public_commitment = _public_commitment(
        replicate,
        replicate_commitment,
        family,
        phase,
        ordinal,
        payload_json,
    )
    task_id = _task_id(replicate, family, phase, ordinal, public_commitment)
    task = PublicEvaluationTask(
        task_id=task_id,
        replicate=replicate,
        replicate_commitment=replicate_commitment,
        family=family,
        phase=phase,
        ordinal=ordinal,
        payload_json=payload_json,
        public_commitment=public_commitment,
    )
    private_bytes = _canonical_json(
        {
            "private": private,
            "public_commitment": public_commitment,
            "task_id": task_id,
        }
    ).encode("utf-8")
    private_commitment = "sha256:" + hmac.new(
        replicate_key,
        _PRIVATE_COMMITMENT_DOMAIN + private_bytes,
        hashlib.sha256,
    ).hexdigest()
    return _TaskBinding(task, source, aliases, private_commitment)


def _render_causal_record(
    record: Record,
    entity_aliases: dict[str, str],
    location_aliases: dict[str, str],
) -> str:
    predicate = {
        "token_in": "AT",
        "file_at": "AT",
        "item_in": "AT",
        "position_adjacent": "ADJACENT",
        "directory_link": "LINK",
        "container_capacity": "CAPACITY",
    }.get(record.predicate.rsplit(".", maxsplit=1)[-1])
    if predicate is None:
        raise ValueError("causal public record uses an unsupported predicate")
    arguments = []
    for argument in record.arguments:
        if argument in entity_aliases:
            arguments.append(entity_aliases[argument])
        elif argument in location_aliases:
            arguments.append(location_aliases[argument])
        elif argument.startswith("limit_"):
            arguments.append(argument.removeprefix("limit_"))
        else:
            raise ValueError("causal public record contains an unbound argument")
    return f"{predicate}({','.join(arguments)})"


def _glyph_alias(prefix: Literal["S", "A"], public_digest: str) -> str:
    _require_digest(public_digest, "glyph public digest")
    suffix = hashlib.sha256(
        b"angler.high-level-multidomain.v1\x00glyph-alias\x00"
        + prefix.encode("ascii")
        + b"\x00"
        + public_digest.encode("ascii")
    ).hexdigest()[:16]
    return f"{prefix}_{suffix}"


def _causal_phase_slot(
    phase: Phase,
    ordinal: int,
    cases_per_domain: int,
) -> int:
    if phase == "adaptation":
        return 0
    if phase == "development":
        return 1
    if cases_per_domain != 4:
        raise RuntimeError("final causal surface requires four cases per domain")
    occurrence = (ordinal % cases_per_domain) // 2
    if occurrence not in (0, 1):
        raise RuntimeError("final causal occurrence is outside the frozen slots")
    return 2 + occurrence


def _round_robin(
    groups: tuple[tuple[_TaskBinding, ...], ...],
) -> tuple[_TaskBinding, ...]:
    """Interleave declared family groups without changing order inside a family."""

    maximum = max((len(group) for group in groups), default=0)
    return tuple(
        group[index]
        for index in range(maximum)
        for group in groups
        if index < len(group)
    )


def _make_random_feedback_binding(
    replicate_key: bytes,
    replicate_commitment: str,
    adaptation_task_ids: tuple[str, ...],
) -> _RandomFeedbackBinding:
    if len(adaptation_task_ids) != 12 or len(set(adaptation_task_ids)) != 12:
        raise RuntimeError("random feedback requires twelve unique adaptation tasks")
    seed = derive_family_seed(
        replicate_key,
        "random-feedback",
        "random-feedback",
    )
    seed_bytes = seed.to_bytes(8, "big")
    ordered = tuple(
        sorted(
            adaptation_task_ids,
            key=lambda task_id: (
                hashlib.sha256(seed_bytes + task_id.encode("ascii")).digest(),
                task_id,
            ),
        )
    )
    by_task = {
        task_id: (0.0 if index < 6 else 1.0)
        for index, task_id in enumerate(ordered)
    }
    assignments = tuple(
        (task_id, by_task[task_id]) for task_id in adaptation_task_ids
    )
    assignment_bytes = _canonical_json(
        {
            "assignments": [
                {"score": score, "task_id": task_id}
                for task_id, score in assignments
            ],
            "replicate_commitment": replicate_commitment,
            "schema": SUITE_SCHEMA,
        }
    ).encode("utf-8")
    commitment = "sha256:" + hmac.new(
        replicate_key,
        _RANDOM_FEEDBACK_COMMITMENT_DOMAIN + assignment_bytes,
        hashlib.sha256,
    ).hexdigest()
    return _RandomFeedbackBinding(
        replicate_commitment,
        assignments,
        commitment,
    )


def _public_commitment(
    replicate: str,
    replicate_commitment: str,
    family: Family,
    phase: Phase,
    ordinal: int,
    payload_json: str,
) -> str:
    return _digest(
        "public-task",
        {
            "family": family,
            "ordinal": ordinal,
            "payload_json": payload_json,
            "phase": phase,
            "replicate": replicate,
            "replicate_commitment": replicate_commitment,
        },
    )


def _response_commitment(
    task_id: str,
    arm: Arm,
    attempt_receipt_ref: str,
    raw_response: str,
) -> str:
    return _digest(
        "response",
        {
            "arm": arm,
            "attempt_receipt_ref": attempt_receipt_ref,
            "raw_response": raw_response,
            "task_id": task_id,
        },
    )


def _task_id(
    replicate: str,
    family: Family,
    phase: Phase,
    ordinal: int,
    public_commitment: str,
) -> str:
    return _digest(
        "task-id",
        {
            "family": family,
            "ordinal": ordinal,
            "phase": phase,
            "public_commitment": public_commitment,
            "replicate": replicate,
        },
    )


def _digest(kind: str, payload: dict[str, object]) -> str:
    encoded = _canonical_json(
        {"kind": kind, "payload": payload, "schema": SUITE_SCHEMA}
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_canonical_json(value: str, label: str) -> dict[str, object]:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain valid JSON") from error
    if not isinstance(payload, dict) or _canonical_json(payload) != value:
        raise ValueError(f"{label} must contain a canonical JSON object")
    return payload


def _malformed() -> ParsedPublicResponse:
    return ParsedPublicResponse(False, (), "MALFORMED_RESPONSE")


def _require_action_schema(value: object) -> ActionSchema:
    if not isinstance(value, ActionSchema):
        raise RuntimeError("glyph alias is not bound to an action schema")
    return value


def _require_pairwise(value: float | None) -> float:
    if type(value) is not float or not 0.0 <= value <= 1.0:
        raise RuntimeError("internal pairwise agreement is invalid")
    return value


def _expected_arms(purpose: Purpose, phase: Phase) -> tuple[Arm, ...]:
    if purpose == "qualification":
        return (QUALIFICATION_ARM,)
    if phase == "adaptation":
        return ("FULL", "RANDOM_FEEDBACK")
    return EVALUATION_ARMS


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")


def _validate_replicate_key(value: bytes) -> None:
    if type(value) is not bytes or len(value) != _SEED_BYTES:
        raise ValueError("raw replicate seed must be exactly 32 bytes")


def _validate_family(value: str) -> None:
    if value not in FAMILIES:
        raise ValueError("family is not declared by the suite")


def _validate_seed_family(value: str) -> None:
    if value not in SEED_FAMILIES:
        raise ValueError("family is not declared by the seed domain")


def _validate_phase(value: str) -> None:
    if value not in PHASES:
        raise ValueError("phase is not declared by the suite")


def _validate_arm(value: str) -> None:
    if value not in (*EVALUATION_ARMS, QUALIFICATION_ARM):
        raise ValueError("arm is not declared by the suite")


__all__ = [
    "Arm",
    "EVALUATION_ARMS",
    "EVALUATION_IDENTITY",
    "EvaluatorCommitments",
    "FAMILIES",
    "FinalMetrics",
    "HighLevelMultidomainEvaluator",
    "MetricAggregate",
    "ObjectiveJudgment",
    "PHASES",
    "ParsedPublicResponse",
    "PublicEvaluationTask",
    "QUALIFICATION_IDENTITY",
    "QUALIFICATION_ARM",
    "QUALIFICATION_SEED",
    "RandomFeedbackCommitment",
    "ROLES",
    "SEED_FAMILIES",
    "SUITE_SCHEMA",
    "TaskCommitment",
    "commit_replicate_seed",
    "derive_family_seed",
    "expected_phase_count",
    "make_evaluation_evaluator",
    "make_qualification_evaluator",
    "parse_public_response",
    "render_public_task",
]
