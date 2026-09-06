"""Sealed V2 evaluator for the history-conditioning isolation experiment.

This module owns fresh synthetic task construction, public presentation,
strict response parsing, phase release, and objective outcome judging.  Raw
evaluation seeds and generator-private solutions stay behind the evaluator
boundary.  There is deliberately no answer-, route-, or repair-producing API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import hmac
import itertools
import json
import math
import re
from typing import Literal

from angler.procedures.records import ActionSchema, GroundAction, Record
from experiments.evaluators import causal_operator_suite as causal
from experiments.evaluators import glyph_machine_trace_suite as glyph
from experiments.evaluators import symbolic_procedure_transfer_suite as symbolic


PROTOCOL_IDENTITY = (
    "angler.high-level-multidomain.v2-history-conditioning-isolation.v3"
)
SUITE_SCHEMA = PROTOCOL_IDENTITY
QUALIFICATION_IDENTITY = (
    "angler.high-level-multidomain.v2-history-conditioning-isolation."
    "qualification.v6"
)
EVALUATION_IDENTITY = (
    "angler.high-level-multidomain.v2-history-conditioning-isolation."
    "evaluation.v3"
)
QUALIFICATION_SEED = 2_026_090_220

Family = Literal[
    "symbolic-demonstration-transfer",
    "glyph-machine",
    "causal-operator",
]
Phase = Literal["adaptation", "development", "final"]
Role = Literal["adaptation", "development", "final", "surface"]
Purpose = Literal["qualification", "evaluation"]
Arm = Literal[
    "N0_NATIVE_NO_HISTORY",
    "F0_NEUTRAL_FORMATTED_12",
    "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6",
    "S0_SAME_FAMILY_LOW_6_NEUTRAL_6",
    "S1_SAME_FAMILY_HIGH_6_NEUTRAL_6",
    "O0_ORDINARY_FROZEN_12",
    "IN_FULL_NEUTRAL_12",
    "IO_FULL_ORDINARY_12",
]
Disposition = Literal["SUCCESS", "UNSUCCESSFUL"]

FAMILIES: tuple[Family, ...] = (
    "symbolic-demonstration-transfer",
    "glyph-machine",
    "causal-operator",
)
PHASES: tuple[Phase, ...] = ("adaptation", "development", "final")
EVALUATION_ARMS: tuple[Arm, ...] = (
    "N0_NATIVE_NO_HISTORY",
    "F0_NEUTRAL_FORMATTED_12",
    "X0_CROSS_FAMILY_LOW_6_NEUTRAL_6",
    "S0_SAME_FAMILY_LOW_6_NEUTRAL_6",
    "S1_SAME_FAMILY_HIGH_6_NEUTRAL_6",
    "O0_ORDINARY_FROZEN_12",
    "IN_FULL_NEUTRAL_12",
    "IO_FULL_ORDINARY_12",
)
ADAPTATION_ARMS: tuple[Arm, ...] = (
    "IN_FULL_NEUTRAL_12",
    "IO_FULL_ORDINARY_12",
)
ROLES: tuple[Role, ...] = ("adaptation", "development", "final", "surface")
_SEED_ROLES: dict[Family, tuple[Role, ...]] = {
    "symbolic-demonstration-transfer": ("adaptation",),
    "glyph-machine": ("adaptation", "surface"),
    "causal-operator": ("adaptation", "development", "final"),
}

_EVALUATION_COUNTS: dict[Family, dict[Phase, int]] = {
    family: {"adaptation": 15, "development": 3, "final": 6}
    for family in FAMILIES
}
_QUALIFICATION_COUNTS: dict[Family, dict[Phase, int]] = {
    family: {"adaptation": 15, "development": 1, "final": 0}
    for family in FAMILIES
}
_SEED_BYTES = 32
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GLYPH_TOKEN = re.compile(r"^A_[0-9a-f]{16}$")
_CAUSAL_ACTION = re.compile(
    r"^ACT\((E[1-9][0-9]*),(L[1-9][0-9]*),(L[1-9][0-9]*)\)$"
)

_IDENTITY_PREFIX = "angler.high-level-multidomain.v2-history-conditioning-isolation"
_REPLICATE_SEED_DOMAIN = (_IDENTITY_PREFIX + "\x00replicate-seed\x00").encode()
_PRIVATE_TASK_DOMAIN = (_IDENTITY_PREFIX + "\x00private-task\x00").encode()
_PUBLIC_TASK_DOMAIN = (_IDENTITY_PREFIX + "\x00public-task\x00").encode()
_TASK_ID_DOMAIN = (_IDENTITY_PREFIX + "\x00task-id\x00").encode()
_EVALUATOR_RECORD_DOMAIN = (_IDENTITY_PREFIX + "\x00evaluator-record\x00").encode()
CONTROL_INDEX_DOMAIN = (_IDENTITY_PREFIX + "\x00control-index\x00").encode()
_CAUSAL_SURFACE_DOMAIN = (PROTOCOL_IDENTITY + "\x00causal-surface\x00").encode()

GLYPH_DEVELOPMENT_CANDIDATE_COMMITMENT = (
    "sha256:f4261dcf9227bf7f08e7191264fd5f089688658ccf995f1b9f3ae2e724a83d1b"
)
GLYPH_FINAL_CANDIDATE_COMMITMENT = (
    "sha256:5cbdd4f602e332c6fd0581d29a6b990f36726d3b7bfd8464c4b44d8731e8d7ab"
)
CONSUMED_REPLICATE_COMMITMENTS: frozenset[str] = frozenset(
    {
        "sha256:86383b64d268bc944abffa1481d81cb730b75766d7d9f0f06d153ca9837cf4f9",
        "sha256:4ec98ef2fed6da44b2593301876d59ee051112e28d9c9caabc243f76481d7c4d",
        "sha256:869675ad3b43bdd300f6ed58b8bf0af2b87520318b42e36d721565a7b7e9d9f5",
        "sha256:44ea52eb9a4085e0e803a965b53ad0fa6ac58cd642fc82ae130a583371b9316f",
        "sha256:a6b8f252b7b8482d0e6648204899dff9d35f4a7c20160d948854bc271d293c8f",
        "sha256:a7327638bef71c33166f1cce2b6285fefcf1c496d3d4b2708346ee534d2965ae",
        "sha256:07d4cbc1ed6fc94466622677bdc1092c96ebfc3024d6080d483a2dfd0c876670",
        "sha256:980d453d2e54799283429d38bdffc6efc0f11264b4ff500bccdcb7a3979e69ff",
        "sha256:646c8207543482e6cee8255fe2e81697a0c2fffa9eb94ecb56644a0c3ba5788b",
        "sha256:613957a33dee19a992d06e7854b9c8435226c559ce7cdf0b734b02654d90a4e0",
        "sha256:2382474330f7e2d63e7e02bbe4d02112eecd5ce6d8c5be810b8b3671b9e030af",
    }
)


@dataclass(frozen=True, slots=True)
class PublicEvaluationTask:
    """Immutable model-facing task projection with no private binding."""

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
        _validate_replicate_label(self.replicate)
        _validate_family(self.family)
        _validate_phase(self.phase)
        _validate_ordinal(self.ordinal)
        payload = _decode_canonical_json(self.payload_json, "payload_json")
        if payload.get("family") != self.family:
            raise ValueError("public payload family does not match task metadata")
        if self.public_commitment != _public_task_ref(
            self.replicate,
            self.replicate_commitment,
            self.family,
            self.phase,
            self.ordinal,
            self.payload_json,
        ):
            raise ValueError("public_commitment does not bind this task")
        if self.task_id != _task_id(
            self.replicate,
            self.family,
            self.phase,
            self.ordinal,
            self.public_commitment,
        ):
            raise ValueError("task_id does not bind this task")

    @property
    def protocol_identity(self) -> str:
        return PROTOCOL_IDENTITY

    @property
    def public_task_ref(self) -> str:
        return self.public_commitment

    @property
    def public_prompt(self) -> str:
        return self.payload_json

    @property
    def payload(self) -> dict[str, object]:
        return _decode_canonical_json(self.payload_json, "payload_json")

    def to_canonical(self) -> dict[str, object]:
        return {
            "family": self.family,
            "ordinal": self.ordinal,
            "payload": json.loads(self.payload_json),
            "phase": self.phase,
            "protocol_identity": PROTOCOL_IDENTITY,
            "public_task_ref": self.public_commitment,
            "replicate": self.replicate,
            "replicate_commitment": self.replicate_commitment,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class ParsedPublicResponse:
    """Strict parse result; invalid text is never normalized or repaired."""

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
        if self.valid and self.error is not None:
            raise ValueError("a valid parse cannot expose an error")
        if not self.valid and (self.tokens or self.error != "MALFORMED_RESPONSE"):
            raise ValueError("an invalid parse must expose only the generic error")


@dataclass(frozen=True, slots=True)
class ObjectiveJudgment:
    """Outcome-only judgment with conformance and immutable evidence refs."""

    task_id: str
    public_task_ref: str
    arm: Arm
    attempt_receipt_ref: str
    raw_response: str
    response_commitment: str
    proposal_admitted: bool
    response_conforms: bool
    success: bool
    score: float
    disposition: Disposition
    evaluator_record_ref: str

    def __post_init__(self) -> None:
        for label, value in (
            ("task_id", self.task_id),
            ("public_task_ref", self.public_task_ref),
            ("attempt_receipt_ref", self.attempt_receipt_ref),
            ("response_commitment", self.response_commitment),
            ("evaluator_record_ref", self.evaluator_record_ref),
        ):
            _require_digest(value, label)
        _validate_arm(self.arm)
        if not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be text")
        if (
            type(self.proposal_admitted) is not bool
            or type(self.response_conforms) is not bool
            or type(self.success) is not bool
        ):
            raise TypeError("proposal_admitted, response_conforms, and success must be bool")
        if not self.proposal_admitted and (
            self.raw_response != "" or self.response_conforms or self.success
        ):
            raise ValueError("an unadmitted proposal has no execution response")
        if type(self.score) is not float or self.score not in (0.0, 1.0):
            raise ValueError("score must be exact binary float")
        if self.success != (self.score == 1.0):
            raise ValueError("success must match score")
        expected_disposition: Disposition = (
            "SUCCESS" if self.success else "UNSUCCESSFUL"
        )
        if self.disposition != expected_disposition:
            raise ValueError("disposition must match success")
        if self.response_commitment != _response_commitment(
            self.task_id,
            self.arm,
            self.attempt_receipt_ref,
            self.raw_response,
        ):
            raise ValueError("response_commitment does not bind the response")
        if self.evaluator_record_ref != _evaluator_record_ref(
            self._canonical_without_ref()
        ):
            raise ValueError("evaluator_record_ref does not bind the judgment")

    def _canonical_without_ref(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "attempt_receipt_ref": self.attempt_receipt_ref,
            "disposition": self.disposition,
            "protocol_identity": PROTOCOL_IDENTITY,
            "proposal_admitted": self.proposal_admitted,
            "public_task_ref": self.public_task_ref,
            "raw_response": self.raw_response,
            "record_kind": "objective-judgment",
            "response_commitment": self.response_commitment,
            "response_conforms": self.response_conforms,
            "score": self.score,
            "success": self.success,
            "task_id": self.task_id,
        }

    def to_canonical(self) -> dict[str, object]:
        return {
            **self._canonical_without_ref(),
            "evaluator_record_ref": self.evaluator_record_ref,
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
            if (
                type(self.pairwise_agreement_total) is not float
                or not 0.0
                <= self.pairwise_agreement_total
                <= float(self.attempts)
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
    """Aggregate-only snapshot available after the evaluator closes."""

    identity: str
    purpose: Purpose
    aggregates: tuple[MetricAggregate, ...]

    def __post_init__(self) -> None:
        if self.identity != _identity_for(self.purpose):
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
        return _evaluator_record_ref(
            {
                "protocol_identity": PROTOCOL_IDENTITY,
                "record_kind": "final-metrics",
                **self.to_canonical(),
            }
        )

    def to_canonical(self) -> dict[str, object]:
        return {
            "aggregates": [item.to_canonical() for item in self.aggregates],
            "identity": self.identity,
            "protocol_identity": PROTOCOL_IDENTITY,
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class TaskCommitment:
    """Manifest-safe public and keyed-private task bindings."""

    task_id: str
    replicate: str
    replicate_commitment: str
    family: Family
    phase: Phase
    ordinal: int
    public_task_ref: str
    private_commitment: str

    def __post_init__(self) -> None:
        for label, value in (
            ("task_id", self.task_id),
            ("replicate_commitment", self.replicate_commitment),
            ("public_task_ref", self.public_task_ref),
            ("private_commitment", self.private_commitment),
        ):
            _require_digest(value, label)
        _validate_replicate_label(self.replicate)
        _validate_family(self.family)
        _validate_phase(self.phase)
        _validate_ordinal(self.ordinal)

    @property
    def public_commitment(self) -> str:
        """Compatibility spelling for the immutable public task ref."""

        return self.public_task_ref

    def to_canonical(self) -> dict[str, object]:
        return {
            "family": self.family,
            "ordinal": self.ordinal,
            "phase": self.phase,
            "private_commitment": self.private_commitment,
            "public_task_ref": self.public_task_ref,
            "replicate": self.replicate,
            "replicate_commitment": self.replicate_commitment,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class EvaluatorCommitments:
    """Manifest-safe metadata for one evaluator construction."""

    identity: str
    purpose: Purpose
    qualification_seed: int | None
    replicate_labels: tuple[str, ...]
    replicate_commitments: tuple[str, ...]
    glyph_candidate_commitment: str
    tasks: tuple[TaskCommitment, ...]

    def __post_init__(self) -> None:
        if self.identity != _identity_for(self.purpose):
            raise ValueError("identity does not match evaluator purpose")
        expected_replicates = 1 if self.purpose == "qualification" else 3
        if self.purpose == "qualification":
            if self.qualification_seed != QUALIFICATION_SEED:
                raise ValueError("qualification seed is not frozen")
            expected_labels = ("qualification-01",)
            expected_glyph = GLYPH_DEVELOPMENT_CANDIDATE_COMMITMENT
        else:
            if self.qualification_seed is not None:
                raise ValueError("evaluation metadata cannot expose raw seeds")
            expected_labels = ("replicate-01", "replicate-02", "replicate-03")
            expected_glyph = GLYPH_FINAL_CANDIDATE_COMMITMENT
        if self.replicate_labels != expected_labels:
            raise ValueError("replicate labels do not match purpose")
        if (
            type(self.replicate_commitments) is not tuple
            or len(self.replicate_commitments) != expected_replicates
            or len(set(self.replicate_commitments)) != expected_replicates
        ):
            raise ValueError("replicate commitments must be exact and distinct")
        for value in self.replicate_commitments:
            _require_digest(value, "replicate commitment")
        if self.glyph_candidate_commitment != expected_glyph:
            raise ValueError("glyph candidate commitment does not match purpose")
        if type(self.tasks) is not tuple or any(
            not isinstance(item, TaskCommitment) for item in self.tasks
        ):
            raise TypeError("tasks must be an immutable TaskCommitment tuple")
        task_ids = tuple(item.task_id for item in self.tasks)
        public_refs = tuple(item.public_task_ref for item in self.tasks)
        private_refs = tuple(item.private_commitment for item in self.tasks)
        if any(
            len(set(values)) != len(values)
            for values in (task_ids, public_refs, private_refs)
        ):
            raise ValueError("task commitments must be globally unique")
        counts = _counts_for(self.purpose)
        phases = _phases_for(self.purpose)
        for label, commitment in zip(
            self.replicate_labels,
            self.replicate_commitments,
            strict=True,
        ):
            for family in FAMILIES:
                for phase in phases:
                    selected = tuple(
                        item
                        for item in self.tasks
                        if item.replicate == label
                        and item.replicate_commitment == commitment
                        and item.family == family
                        and item.phase == phase
                    )
                    expected = counts[family][phase]
                    if len(selected) != expected or {
                        item.ordinal for item in selected
                    } != set(range(expected)):
                        raise ValueError(
                            "task commitments do not match frozen cardinalities"
                        )
        if any(
            item.replicate not in self.replicate_labels
            or item.replicate_commitment not in self.replicate_commitments
            or item.phase not in phases
            for item in self.tasks
        ):
            raise ValueError("task commitment is outside this evaluator")

    @property
    def digest(self) -> str:
        return _evaluator_record_ref(
            {
                "protocol_identity": PROTOCOL_IDENTITY,
                "record_kind": "evaluator-commitments",
                **self.to_canonical(),
            }
        )

    def to_canonical(self) -> dict[str, object]:
        return {
            "glyph_candidate_commitment": self.glyph_candidate_commitment,
            "identity": self.identity,
            "protocol_identity": PROTOCOL_IDENTITY,
            "purpose": self.purpose,
            "qualification_seed": self.qualification_seed,
            "replicate_commitments": list(self.replicate_commitments),
            "replicate_labels": list(self.replicate_labels),
            "tasks": [item.to_canonical() for item in self.tasks],
        }


@dataclass(frozen=True, slots=True, repr=False)
class _TaskBinding:
    public: PublicEvaluationTask
    source: object
    aliases: tuple[tuple[str, object], ...]
    private_commitment: str


@dataclass(frozen=True, slots=True, repr=False)
class _JudgmentBinding:
    judgment: ObjectiveJudgment
    pairwise_agreement: float | None


class HighLevelMultidomainEvaluator:
    """Stateful, phase-gated V2 task release and objective-judge boundary."""

    def __init__(
        self,
        purpose: Purpose,
        replicate_keys: Sequence[bytes],
        expected_replicate_commitments: Sequence[str],
        expected_final_admission: str | None,
    ) -> None:
        _validate_purpose(purpose)
        keys = tuple(replicate_keys)
        commitments = tuple(expected_replicate_commitments)
        expected_count = 1 if purpose == "qualification" else 3
        if len(keys) != expected_count or len(commitments) != expected_count:
            raise ValueError("replicate count does not match evaluator purpose")
        for key in keys:
            _validate_replicate_key(key)
        if len(set(keys)) != expected_count:
            raise ValueError("raw replicate seeds must be distinct")
        computed = tuple(commit_replicate_seed(key) for key in keys)
        if commitments != computed:
            raise ValueError("injected replicate seed commitments do not match")
        qualification_commitment = commit_replicate_seed(_qualification_key())
        if purpose == "qualification":
            if keys != (_qualification_key(),):
                raise ValueError("qualification key is not the frozen public seed")
            if expected_final_admission is not None:
                raise ValueError("qualification cannot bind a final admission")
        else:
            if expected_final_admission is None:
                raise ValueError("evaluation requires a final admission digest")
            _require_digest(expected_final_admission, "expected_final_admission")
            if qualification_commitment in commitments:
                raise ValueError("evaluation cannot reuse the qualification seed")
            reused = set(commitments) & CONSUMED_REPLICATE_COMMITMENTS
            if reused:
                raise ValueError("evaluation seed commitment was already consumed")

        self._purpose = purpose
        self._identity = _identity_for(purpose)
        self._replicate_labels = _replicate_labels_for(purpose)
        self._expected_final_admission = expected_final_admission
        self._final_admission: str | None = None
        self._released: set[Phase] = set()
        self._completed: set[Phase] = set()
        self._consumed_pairs: set[tuple[str, Arm]] = set()
        self._consumed_attempt_receipts: set[str] = set()
        self._judgments: dict[tuple[str, Arm], _JudgmentBinding] = {}

        symbolic_mechanisms, glyph_mechanisms = _select_replicate_mechanisms(
            purpose,
            keys,
        )
        bindings: list[_TaskBinding] = []
        for replicate_index, (
            key,
            commitment,
            label,
            symbolic_mechanism,
            glyph_mechanism,
        ) in enumerate(
            zip(
                keys,
                commitments,
                self._replicate_labels,
                symbolic_mechanisms,
                glyph_mechanisms,
                strict=True,
            )
        ):
            bindings.extend(
                _make_replicate_bindings(
                    purpose=purpose,
                    replicate=label,
                    replicate_key=key,
                    replicate_commitment=commitment,
                    replicate_index=replicate_index,
                    symbolic_mechanism=symbolic_mechanism,
                    glyph_mechanism=glyph_mechanism,
                )
            )

        _validate_generated_bindings(bindings)
        self._bindings = {item.public.task_id: item for item in bindings}
        self._ordered_ids = tuple(item.public.task_id for item in bindings)
        task_commitments = tuple(
            TaskCommitment(
                task_id=item.public.task_id,
                replicate=item.public.replicate,
                replicate_commitment=item.public.replicate_commitment,
                family=item.public.family,
                phase=item.public.phase,
                ordinal=item.public.ordinal,
                public_task_ref=item.public.public_task_ref,
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
            replicate_labels=self._replicate_labels,
            replicate_commitments=commitments,
            glyph_candidate_commitment=(
                GLYPH_DEVELOPMENT_CANDIDATE_COMMITMENT
                if purpose == "qualification"
                else GLYPH_FINAL_CANDIDATE_COMMITMENT
            ),
            tasks=task_commitments,
        )

    @property
    def protocol_identity(self) -> str:
        return PROTOCOL_IDENTITY

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def purpose(self) -> Purpose:
        return self._purpose

    @property
    def replicate_labels(self) -> tuple[str, ...]:
        return self._replicate_labels

    @property
    def commitments(self) -> EvaluatorCommitments:
        return self._commitments

    @property
    def released_phases(self) -> tuple[Phase, ...]:
        return tuple(phase for phase in PHASES if phase in self._released)

    @property
    def completed_phases(self) -> tuple[Phase, ...]:
        return tuple(phase for phase in PHASES if phase in self._completed)

    def expected_arms_for(self, task_id: str) -> tuple[Arm, ...]:
        """Return the exact predeclared arm cell for one owned public task."""

        try:
            task = self._bindings[task_id].public
        except KeyError as error:
            raise ValueError("task_id is not owned by this evaluator") from error
        return expected_arms_for_task(self._purpose, task)

    def release_phase(self, phase: Phase) -> tuple[PublicEvaluationTask, ...]:
        """Release a phase only after all predecessor cells are complete."""

        _validate_phase(phase)
        available = _phases_for(self._purpose)
        if phase not in available:
            raise ValueError("qualification cannot materialize final tasks")
        if any(item not in self._completed for item in available[: available.index(phase)]):
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
        """Close a phase only after every exact task/arm judgment exists."""

        _validate_phase(phase)
        if phase not in self._released:
            raise RuntimeError("a phase must be released before completion")
        task_ids = tuple(
            task_id
            for task_id in self._ordered_ids
            if self._bindings[task_id].public.phase == phase
        )
        expected = {
            (task_id, arm)
            for task_id in task_ids
            for arm in self.expected_arms_for(task_id)
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
        """Bind the runner's independently reviewed final-release admission."""

        if self._purpose != "evaluation":
            raise ValueError("qualification has no final admission")
        _require_digest(admission_digest, "admission_digest")
        if "development" not in self._completed:
            raise RuntimeError("development must complete before final admission")
        if admission_digest != self._expected_final_admission:
            raise ValueError("admission_digest does not match construction binding")
        if self._final_admission not in (None, admission_digest):
            raise RuntimeError("final admission is immutable")
        self._final_admission = admission_digest

    def record_attempt(
        self,
        task_id: str,
        arm: Arm,
        attempt_receipt_ref: str,
        raw_response: str,
        *,
        proposal_admitted: bool,
    ) -> ObjectiveJudgment:
        """Consume and objectively finalize one exact task/arm attempt.

        An unadmitted proposal has no execution response.  It is finalized as
        generic unsuccessful evidence without parsing or invoking a private
        family judge.  Admitted responses are parsed once and never repaired.
        """

        if not isinstance(task_id, str):
            raise TypeError("task_id must be text")
        _validate_arm(arm)
        _require_digest(attempt_receipt_ref, "attempt_receipt_ref")
        if not isinstance(raw_response, str):
            raise TypeError("raw_response must be text")
        if type(proposal_admitted) is not bool:
            raise TypeError("proposal_admitted must be bool")
        if not proposal_admitted and raw_response != "":
            raise ValueError("unadmitted proposal must have an empty raw_response")
        try:
            binding = self._bindings[task_id]
        except KeyError as error:
            raise ValueError("task_id is not owned by this evaluator") from error
        if binding.public.phase not in self._released:
            raise RuntimeError("unreleased task cannot be judged")
        if arm not in self.expected_arms_for(task_id):
            raise ValueError("arm is not admitted for this task")
        judgment_key = (task_id, arm)
        if judgment_key in self._consumed_pairs:
            raise RuntimeError("task/arm judgment is already consumed")
        if attempt_receipt_ref in self._consumed_attempt_receipts:
            raise RuntimeError("attempt receipt is already consumed")

        # Consumption precedes all private judging so an internal failure can
        # never be converted into a retry under the same scientific identity.
        self._consumed_pairs.add(judgment_key)
        self._consumed_attempt_receipts.add(attempt_receipt_ref)

        pairwise: float | None
        if not proposal_admitted:
            response_conforms = False
            primary = 0.0
            pairwise = (
                0.0
                if binding.public.family == "symbolic-demonstration-transfer"
                else None
            )
        else:
            parsed = parse_public_response(binding.public, raw_response)
            response_conforms = parsed.valid
            if not parsed.valid:
                primary = 0.0
                pairwise = (
                    0.0
                    if binding.public.family == "symbolic-demonstration-transfer"
                    else None
                )
            elif binding.public.family == "symbolic-demonstration-transfer":
                pair = binding.source
                if not isinstance(
                    pair,
                    symbolic.GeneratedDemonstrationProcedureTask,
                ):
                    raise RuntimeError("symbolic private binding is inconsistent")
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
                    raise RuntimeError("glyph private binding is inconsistent")
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
                primary = float(
                    glyph.judge_glyph_procedure_attempt(pair, committed)
                )
                pairwise = None
            else:
                challenge = binding.source
                if not isinstance(challenge, causal.OperatorChallenge):
                    raise RuntimeError("causal private binding is inconsistent")
                aliases = dict(binding.aliases)
                schema = challenge.allowed_action_schemas[0]
                actions: list[GroundAction] = []
                for token in parsed.tokens:
                    match = _CAUSAL_ACTION.fullmatch(token)
                    if match is None:
                        raise RuntimeError("validated causal parse became inconsistent")
                    values = tuple(aliases[item] for item in match.groups())
                    if not all(isinstance(item, str) for item in values):
                        raise RuntimeError("causal alias binding is inconsistent")
                    entity, source, destination = values
                    actions.append(schema.ground(entity, source, destination))
                committed = causal.commit_action_sequence(
                    challenge,
                    tuple(actions),
                )
                result = causal.evaluate_committed_sequence(challenge, committed)
                primary = float(result.success)
                pairwise = None

        success = primary == 1.0
        response_ref = _response_commitment(
            task_id,
            arm,
            attempt_receipt_ref,
            raw_response,
        )
        without_ref = {
            "arm": arm,
            "attempt_receipt_ref": attempt_receipt_ref,
            "disposition": "SUCCESS" if success else "UNSUCCESSFUL",
            "protocol_identity": PROTOCOL_IDENTITY,
            "proposal_admitted": proposal_admitted,
            "public_task_ref": binding.public.public_task_ref,
            "raw_response": raw_response,
            "record_kind": "objective-judgment",
            "response_commitment": response_ref,
            "response_conforms": response_conforms,
            "score": float(primary),
            "success": success,
            "task_id": task_id,
        }
        judgment = ObjectiveJudgment(
            task_id=task_id,
            public_task_ref=binding.public.public_task_ref,
            arm=arm,
            attempt_receipt_ref=attempt_receipt_ref,
            raw_response=raw_response,
            response_commitment=response_ref,
            proposal_admitted=proposal_admitted,
            response_conforms=response_conforms,
            success=success,
            score=float(primary),
            disposition="SUCCESS" if success else "UNSUCCESSFUL",
            evaluator_record_ref=_evaluator_record_ref(without_ref),
        )
        self._judgments[judgment_key] = _JudgmentBinding(judgment, pairwise)
        return judgment

    def judge_response(
        self,
        task_id: str,
        arm: Arm,
        attempt_receipt_ref: str,
        raw_response: str,
    ) -> ObjectiveJudgment:
        """Compatibility entry point for an already-admitted response."""

        return self.record_attempt(
            task_id,
            arm,
            attempt_receipt_ref,
            raw_response,
            proposal_admitted=True,
        )

    def final_metrics(self) -> FinalMetrics:
        """Return aggregate metrics only after every available phase closes."""

        available = _phases_for(self._purpose)
        if any(phase not in self._completed for phase in available):
            raise RuntimeError("final metrics are unavailable before suite completion")
        aggregates: list[MetricAggregate] = []
        for phase in available:
            phase_arms = _phase_arm_union(self._purpose, phase)
            for arm in phase_arms:
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
                                int(item.judgment.success) for item in selected
                            ),
                            pairwise_agreement_total=pairwise_total,
                        )
                    )
        return FinalMetrics(
            identity=self._identity,
            purpose=self._purpose,
            aggregates=tuple(aggregates),
        )


def make_qualification_evaluator() -> HighLevelMultidomainEvaluator:
    """Construct the fixed public-seed technical qualification evaluator."""

    key = _qualification_key()
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
    """Construct evaluation from exactly three process-private fresh seeds."""

    return HighLevelMultidomainEvaluator(
        "evaluation",
        raw_replicate_seeds,
        expected_seed_commitments,
        expected_final_admission,
    )


def commit_replicate_seed(raw_seed: bytes) -> str:
    """Return the sole public projection of one injected replicate seed."""

    _validate_replicate_key(raw_seed)
    return "sha256:" + hashlib.sha256(_REPLICATE_SEED_DOMAIN + raw_seed).hexdigest()


def derive_family_seed(raw_seed: bytes, family: Family, role: Role) -> int:
    """Derive the exact unsigned 64-bit V2 family/role seed."""

    _validate_replicate_key(raw_seed)
    _validate_family(family)
    if role not in ROLES or role not in _SEED_ROLES[family]:
        raise ValueError("role is not applicable to this family")
    message = f"{PROTOCOL_IDENTITY}\x00{family}\x00{role}".encode("utf-8")
    return int.from_bytes(
        hmac.new(raw_seed, message, hashlib.sha256).digest()[:8],
        "big",
    )


def expected_phase_count(purpose: Purpose, phase: Phase) -> int:
    """Return the exact public-task count across all purpose replicates."""

    _validate_purpose(purpose)
    _validate_phase(phase)
    if phase not in _phases_for(purpose):
        return 0
    replicate_count = 1 if purpose == "qualification" else 3
    return replicate_count * sum(
        _counts_for(purpose)[family][phase] for family in FAMILIES
    )


def expected_attempt_count(purpose: Purpose, phase: Phase) -> int:
    """Return the exact finalized task/arm cell count for one phase."""

    _validate_purpose(purpose)
    _validate_phase(phase)
    if phase not in _phases_for(purpose):
        return 0
    if purpose == "qualification" and phase == "adaptation":
        return 45 + len(FAMILIES)
    return expected_phase_count(purpose, phase) * len(
        ADAPTATION_ARMS if phase == "adaptation" else EVALUATION_ARMS
    )


def expected_arms_for_task(
    purpose: Purpose,
    task: PublicEvaluationTask,
) -> tuple[Arm, ...]:
    """Return the frozen arm cell for one public task and purpose."""

    _validate_purpose(purpose)
    if not isinstance(task, PublicEvaluationTask):
        raise TypeError("task must be a PublicEvaluationTask")
    if purpose == "qualification":
        if task.replicate != "qualification-01":
            raise ValueError("task does not belong to qualification")
        if task.phase == "final":
            raise ValueError("qualification has no final task")
        if task.phase == "adaptation":
            return (
                ADAPTATION_ARMS
                if task.ordinal == 0
                else ("IO_FULL_ORDINARY_12",)
            )
        return EVALUATION_ARMS
    if not task.replicate.startswith("replicate-"):
        raise ValueError("task does not belong to evaluation")
    return ADAPTATION_ARMS if task.phase == "adaptation" else EVALUATION_ARMS


def render_public_task(task: PublicEvaluationTask) -> str:
    """Return canonical model-facing JSON without extra prompt context."""

    if not isinstance(task, PublicEvaluationTask):
        raise TypeError("task must be a PublicEvaluationTask")
    return task.payload_json


def parse_public_response(
    task: PublicEvaluationTask,
    raw_response: str,
) -> ParsedPublicResponse:
    """Strictly parse the public grammar without normalization or repair."""

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


def glyph_candidate_commitments(purpose: Purpose) -> tuple[str, ...]:
    """Return the frozen four-state public mechanism candidate list."""

    _validate_purpose(purpose)
    partition: glyph.GlyphMachinePartition = (
        "development" if purpose == "qualification" else "final"
    )
    candidates = tuple(glyph.glyph_machine_mechanism_partition(partition)[2:16])
    if len(candidates) != 14:
        raise RuntimeError("glyph four-state candidate count drifted")
    expected = (
        GLYPH_DEVELOPMENT_CANDIDATE_COMMITMENT
        if purpose == "qualification"
        else GLYPH_FINAL_CANDIDATE_COMMITMENT
    )
    observed = "sha256:" + hashlib.sha256(
        _canonical_json(list(candidates)).encode("utf-8")
    ).hexdigest()
    if observed != expected:
        raise RuntimeError("glyph four-state candidate commitment drifted")
    return candidates


def _select_replicate_mechanisms(
    purpose: Purpose,
    replicate_keys: tuple[bytes, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    symbolic_partition: symbolic.PermutationPartition = (
        "development" if purpose == "qualification" else "final"
    )
    remaining_symbolic = list(
        symbolic.demonstration_permutation_partition(symbolic_partition)
    )
    remaining_glyph = list(glyph_candidate_commitments(purpose))
    selected_symbolic: list[tuple[int, ...]] = []
    selected_glyph: list[str] = []
    for key in replicate_keys:
        symbolic_index = derive_family_seed(
            key,
            "symbolic-demonstration-transfer",
            "adaptation",
        ) % len(remaining_symbolic)
        selected_symbolic.append(remaining_symbolic.pop(symbolic_index))
        glyph_index = derive_family_seed(
            key,
            "glyph-machine",
            "adaptation",
        ) % len(remaining_glyph)
        selected_glyph.append(remaining_glyph.pop(glyph_index))
    if purpose == "qualification":
        complete = glyph.glyph_machine_mechanism_partition("development")
        if len(selected_glyph) != 1 or complete.index(selected_glyph[0]) != 13:
            raise RuntimeError(
                "fixed qualification did not select glyph partition index 13"
            )
    return tuple(selected_symbolic), tuple(selected_glyph)


def _make_replicate_bindings(
    *,
    purpose: Purpose,
    replicate: str,
    replicate_key: bytes,
    replicate_commitment: str,
    replicate_index: int,
    symbolic_mechanism: tuple[int, ...],
    glyph_mechanism: str,
) -> tuple[_TaskBinding, ...]:
    partition: Literal["development", "final"] = (
        "development" if purpose == "qualification" else "final"
    )
    symbolic_seed = derive_family_seed(
        replicate_key,
        "symbolic-demonstration-transfer",
        "adaptation",
    )
    symbolic_stream = symbolic.make_demonstration_procedure_transfer_stream(
        symbolic_seed,
        supports_per_procedure=15,
        queries_per_procedure=9,
        position_permutation=symbolic_mechanism,
        mechanism_partition=partition,
        expose_transform_demonstrations=True,
    )
    symbolic_adaptation = tuple(
        pair
        for pair in symbolic_stream.supports
        if pair.learner.demonstrations_visible
    )
    if len(symbolic_stream.supports) != 30 or len(symbolic_adaptation) != 15:
        raise RuntimeError("symbolic support construction drifted")
    symbolic_by_phase: dict[
        Phase,
        tuple[symbolic.GeneratedDemonstrationProcedureTask, ...],
    ] = {
        "adaptation": symbolic_adaptation,
        "development": (
            symbolic_stream.queries[:1]
            if purpose == "qualification"
            else symbolic_stream.queries[:3]
        ),
        "final": (
            ()
            if purpose == "qualification"
            else symbolic_stream.queries[3:9]
        ),
    }

    glyph_seed = derive_family_seed(replicate_key, "glyph-machine", "adaptation")
    surface_seed = derive_family_seed(replicate_key, "glyph-machine", "surface")
    glyph_stream = glyph.make_glyph_machine_trace_stream(
        glyph_seed,
        surface_seed=surface_seed,
        supports=15,
        queries=9,
        observations_per_support=2,
        maximum_steps=4,
        mechanism_commitment=glyph_mechanism,
        mechanism_partition=partition,
    )
    if len(glyph_stream.supports) != 15 or len(glyph_stream.queries) != 9:
        raise RuntimeError("glyph stream cardinality drifted")
    glyph_by_phase: dict[Phase, tuple[glyph.GeneratedGlyphMachineTask, ...]] = {
        "adaptation": glyph_stream.supports,
        "development": (
            glyph_stream.queries[:1]
            if purpose == "qualification"
            else glyph_stream.queries[:3]
        ),
        "final": (
            () if purpose == "qualification" else glyph_stream.queries[3:9]
        ),
    }

    causal_development = _causal_development_subset(
        causal.make_heldout_operator_suite(
            derive_family_seed(
                replicate_key,
                "causal-operator",
                "development",
            ),
            cases_per_domain=2,
        )
    )
    causal_by_phase: dict[Phase, tuple[causal.OperatorChallenge, ...]] = {
        "adaptation": causal.make_heldout_operator_suite(
            derive_family_seed(replicate_key, "causal-operator", "adaptation"),
            cases_per_domain=5,
        ),
        "development": (
            causal_development[:1]
            if purpose == "qualification"
            else causal_development
        ),
        "final": (
            ()
            if purpose == "qualification"
            else causal.make_heldout_operator_suite(
                derive_family_seed(replicate_key, "causal-operator", "final"),
                cases_per_domain=2,
            )
        ),
    }

    counts = _counts_for(purpose)
    bindings: list[_TaskBinding] = []
    for phase in _phases_for(purpose):
        symbolic_pairs = symbolic_by_phase[phase]
        glyph_pairs = glyph_by_phase[phase]
        causal_challenges = causal_by_phase[phase]
        if (
            len(symbolic_pairs) != counts["symbolic-demonstration-transfer"][phase]
            or len(glyph_pairs) != counts["glyph-machine"][phase]
            or len(causal_challenges) != counts["causal-operator"][phase]
        ):
            raise RuntimeError("one or more V2 family counts are inconsistent")

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
                purpose=purpose,
                replicate=replicate,
                replicate_key=replicate_key,
                replicate_commitment=replicate_commitment,
                replicate_index=replicate_index,
                phase=phase,
                ordinal=ordinal,
            )
            for ordinal, challenge in enumerate(causal_challenges)
        )
        bindings.extend(
            _round_robin((symbolic_bindings, glyph_bindings, causal_bindings))
        )
    return tuple(bindings)


def _causal_development_subset(
    candidates: tuple[causal.OperatorChallenge, ...],
) -> tuple[causal.OperatorChallenge, ...]:
    if len(candidates) != 2 * len(causal.SUPPORTED_DOMAINS):
        raise RuntimeError("causal development candidate pool must contain six")
    selected: list[causal.OperatorChallenge] = []
    for domain in causal.SUPPORTED_DOMAINS:
        matches = tuple(
            item
            for item in candidates
            if item.domain == domain
            and item.maximum_steps == causal.SINGLE_OPERATOR_STEPS
        )
        if len(matches) != 1:
            raise RuntimeError(
                "causal development subset requires one two-step case per domain"
            )
        selected.append(matches[0])
    return tuple(selected)


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
    observations = [
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
        for trace in learner.observations
    ]
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
    purpose: Purpose,
    replicate: str,
    replicate_key: bytes,
    replicate_commitment: str,
    replicate_index: int,
    phase: Phase,
    ordinal: int,
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
    phase_offset = {"adaptation": 0, "development": 15, "final": 18}[phase]
    task_slot = (
        (0 if purpose == "qualification" else 72)
        + 24 * replicate_index
        + phase_offset
        + ordinal
    )
    canonical_entity_aliases = tuple(
        f"E{8 * task_slot + index + 1}" for index in range(len(entities))
    )
    canonical_location_aliases = tuple(
        f"L{8 * task_slot + index + 1}" for index in range(len(locations))
    )
    surface_material = (
        _CAUSAL_SURFACE_DOMAIN
        + purpose.encode("ascii")
        + b"\x00"
        + replicate.encode("ascii")
        + b"\x00"
        + phase.encode("ascii")
        + b"\x00"
        + str(ordinal).encode("ascii")
    )
    option_count = math.factorial(len(entities)) * math.factorial(len(locations))
    option_index = int.from_bytes(
        hashlib.sha256(surface_material).digest(),
        "big",
    ) % option_count
    surface_options = itertools.product(
        itertools.permutations(canonical_entity_aliases),
        itertools.permutations(canonical_location_aliases),
    )
    try:
        entity_surface, location_surface = next(
            itertools.islice(surface_options, option_index, None)
        )
    except StopIteration as error:
        raise RuntimeError("causal surface selection exceeded permutation space") from error
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
        "surface_option_index": option_index,
        "task_slot": task_slot,
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
    public_task_ref = _public_task_ref(
        replicate,
        replicate_commitment,
        family,
        phase,
        ordinal,
        payload_json,
    )
    task_id = _task_id(
        replicate,
        family,
        phase,
        ordinal,
        public_task_ref,
    )
    task = PublicEvaluationTask(
        task_id=task_id,
        replicate=replicate,
        replicate_commitment=replicate_commitment,
        family=family,
        phase=phase,
        ordinal=ordinal,
        payload_json=payload_json,
        public_commitment=public_task_ref,
    )
    private_bytes = _canonical_json(
        {
            "private": private,
            "protocol_identity": PROTOCOL_IDENTITY,
            "public_task_ref": public_task_ref,
            "task_id": task_id,
        }
    ).encode("utf-8")
    private_commitment = "sha256:" + hmac.new(
        replicate_key,
        _PRIVATE_TASK_DOMAIN + private_bytes,
        hashlib.sha256,
    ).hexdigest()
    return _TaskBinding(task, source, aliases, private_commitment)


def _public_task_ref(
    replicate: str,
    replicate_commitment: str,
    family: Family,
    phase: Phase,
    ordinal: int,
    payload_json: str,
) -> str:
    payload = {
        "family": family,
        "ordinal": ordinal,
        "payload": _decode_canonical_json(payload_json, "payload_json"),
        "phase": phase,
        "protocol_identity": PROTOCOL_IDENTITY,
        "replicate": replicate,
        "replicate_commitment": replicate_commitment,
    }
    return _domain_digest(_PUBLIC_TASK_DOMAIN, payload)


def _task_id(
    replicate: str,
    family: Family,
    phase: Phase,
    ordinal: int,
    public_task_ref: str,
) -> str:
    return _domain_digest(
        _TASK_ID_DOMAIN,
        {
            "family": family,
            "ordinal": ordinal,
            "phase": phase,
            "protocol_identity": PROTOCOL_IDENTITY,
            "public_task_ref": public_task_ref,
            "replicate": replicate,
        },
    )


def _response_commitment(
    task_id: str,
    arm: Arm,
    attempt_receipt_ref: str,
    raw_response: str,
) -> str:
    return _evaluator_record_ref(
        {
            "arm": arm,
            "attempt_receipt_ref": attempt_receipt_ref,
            "protocol_identity": PROTOCOL_IDENTITY,
            "raw_response": raw_response,
            "record_kind": "response",
            "task_id": task_id,
        }
    )


def _evaluator_record_ref(payload: dict[str, object]) -> str:
    return _domain_digest(_EVALUATOR_RECORD_DOMAIN, payload)


def _domain_digest(domain: bytes, payload: object) -> str:
    return "sha256:" + hashlib.sha256(
        domain + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


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
    arguments: list[str] = []
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
        _PUBLIC_TASK_DOMAIN
        + _canonical_json(
            {
                "prefix": prefix,
                "protocol_identity": PROTOCOL_IDENTITY,
                "public_digest": public_digest,
                "record_kind": "glyph-alias",
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{suffix}"


def _round_robin(
    groups: tuple[tuple[_TaskBinding, ...], ...],
) -> tuple[_TaskBinding, ...]:
    maximum = max((len(group) for group in groups), default=0)
    return tuple(
        group[index]
        for index in range(maximum)
        for group in groups
        if index < len(group)
    )


def _validate_generated_bindings(bindings: list[_TaskBinding]) -> None:
    if not bindings:
        raise RuntimeError("evaluator generated no tasks")
    task_ids = tuple(item.public.task_id for item in bindings)
    public_refs = tuple(item.public.public_task_ref for item in bindings)
    private_refs = tuple(item.private_commitment for item in bindings)
    payloads = tuple(item.public.payload_json for item in bindings)
    if any(
        len(set(values)) != len(values)
        for values in (task_ids, public_refs, private_refs, payloads)
    ):
        raise RuntimeError("generated task binding collision")

    causal_bindings = tuple(
        item
        for item in bindings
        if item.public.family == "causal-operator"
    )
    case_ids = tuple(
        item.source.case_id
        for item in causal_bindings
        if isinstance(item.source, causal.OperatorChallenge)
    )
    if len(case_ids) != len(causal_bindings) or len(set(case_ids)) != len(case_ids):
        raise RuntimeError("causal case identity collision")
    seen_entity_aliases: set[str] = set()
    seen_location_aliases: set[str] = set()
    for binding in causal_bindings:
        payload = json.loads(binding.public.payload_json)
        entities = set(payload["entities"])
        locations = set(payload["locations"])
        if seen_entity_aliases & entities or seen_location_aliases & locations:
            raise RuntimeError("task-indexed causal alias collision")
        seen_entity_aliases.update(entities)
        seen_location_aliases.update(locations)


def _qualification_key() -> bytes:
    return QUALIFICATION_SEED.to_bytes(_SEED_BYTES, "big")


def _counts_for(
    purpose: Purpose,
) -> dict[Family, dict[Phase, int]]:
    _validate_purpose(purpose)
    return (
        _QUALIFICATION_COUNTS
        if purpose == "qualification"
        else _EVALUATION_COUNTS
    )


def _phases_for(purpose: Purpose) -> tuple[Phase, ...]:
    _validate_purpose(purpose)
    return (
        ("adaptation", "development")
        if purpose == "qualification"
        else PHASES
    )


def _replicate_labels_for(purpose: Purpose) -> tuple[str, ...]:
    _validate_purpose(purpose)
    return (
        ("qualification-01",)
        if purpose == "qualification"
        else ("replicate-01", "replicate-02", "replicate-03")
    )


def _identity_for(purpose: Purpose) -> str:
    _validate_purpose(purpose)
    return (
        QUALIFICATION_IDENTITY
        if purpose == "qualification"
        else EVALUATION_IDENTITY
    )


def _phase_arm_union(purpose: Purpose, phase: Phase) -> tuple[Arm, ...]:
    _validate_purpose(purpose)
    _validate_phase(phase)
    if phase == "adaptation":
        return ADAPTATION_ARMS
    return EVALUATION_ARMS


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


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")


def _validate_replicate_key(value: bytes) -> None:
    if type(value) is not bytes or len(value) != _SEED_BYTES:
        raise ValueError("raw replicate seed must be exactly 32 bytes")


def _validate_replicate_label(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(
        r"(?:qualification|replicate)-[0-9]{2}",
        value,
    ) is None:
        raise ValueError("replicate must be a canonical public label")


def _validate_ordinal(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("ordinal must be a non-negative integer")


def _validate_family(value: str) -> None:
    if value not in FAMILIES:
        raise ValueError("family is not declared by the evaluator")


def _validate_phase(value: str) -> None:
    if value not in PHASES:
        raise ValueError("phase is not declared by the evaluator")


def _validate_arm(value: str) -> None:
    if value not in EVALUATION_ARMS:
        raise ValueError("arm is not declared by the evaluator")


def _validate_purpose(value: str) -> None:
    if value not in ("qualification", "evaluation"):
        raise ValueError("purpose must be qualification or evaluation")


__all__ = [
    "ADAPTATION_ARMS",
    "Arm",
    "CONSUMED_REPLICATE_COMMITMENTS",
    "CONTROL_INDEX_DOMAIN",
    "EVALUATION_ARMS",
    "EVALUATION_IDENTITY",
    "EvaluatorCommitments",
    "FAMILIES",
    "FinalMetrics",
    "GLYPH_DEVELOPMENT_CANDIDATE_COMMITMENT",
    "GLYPH_FINAL_CANDIDATE_COMMITMENT",
    "HighLevelMultidomainEvaluator",
    "MetricAggregate",
    "ObjectiveJudgment",
    "PHASES",
    "PROTOCOL_IDENTITY",
    "ParsedPublicResponse",
    "PublicEvaluationTask",
    "QUALIFICATION_IDENTITY",
    "QUALIFICATION_SEED",
    "ROLES",
    "SUITE_SCHEMA",
    "TaskCommitment",
    "commit_replicate_seed",
    "derive_family_seed",
    "expected_arms_for_task",
    "expected_attempt_count",
    "expected_phase_count",
    "glyph_candidate_commitments",
    "make_evaluation_evaluator",
    "make_qualification_evaluator",
    "parse_public_response",
    "render_public_task",
]
