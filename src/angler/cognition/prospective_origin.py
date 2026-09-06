"""Canonical successor contracts for situated prospective cognition.

These records preserve learned hypotheses before execution and bind later
objective feedback without reinterpreting any existing ``001`` contract.  They
canonicalize and validate supplied values; they do not choose candidates,
create evidence, infer truth, or authorize an external effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Any, ClassVar, Mapping, Self

from angler.cognition.contracts import (
    CognitiveEpisode,
    _CanonicalContract,
    _content_ref,
    _digest,
    _digest_tuple,
    _exact_contract,
    _exact_keys,
    _hex_float,
    _identity,
    _integer,
    _payload_list,
    _payload_tuple,
    _text,
    _tuple,
)
from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
    ObjectiveFeedbackRecord,
    ProspectiveTurnReservation,
)
from angler.episodes.canonical import canonical_bytes


MAX_BRANCH_CAPACITY = 4_096
MAX_CONTEXTS = 256
MAX_LATENT_WIDTH = 4_096
MAX_RECURRENT_STEPS = 256
MAX_HORIZON = 256
FOCUS_SUM_TOLERANCE = 1e-6
PREDICTION_ERROR_METRIC = "binary-outcome-softplus-argument-v1"


class RealityMode(str, Enum):
    ACTUAL = "ACTUAL"
    SIMULATED = "SIMULATED"
    HYPOTHETICAL = "HYPOTHETICAL"


class Perspective(str, Enum):
    AGENT_SITUATED = "AGENT_SITUATED"


class ResolutionDisposition(str, Enum):
    OBSERVED = "OBSERVED"
    COMPLETED_UNEVALUATED = "COMPLETED_UNEVALUATED"
    CANCELLED = "CANCELLED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    ERROR = "ERROR"


class BranchStatus(str, Enum):
    OPEN = "OPEN"
    OBSERVED = "OBSERVED"
    COMPLETED_UNEVALUATED = "COMPLETED_UNEVALUATED"
    CANCELLED = "CANCELLED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    ERROR = "ERROR"


def _enum_value(value: Any, enum_type: type[Enum], label: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not registered") from exc


def _finite_float(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _bounded_digests(
    value: Any,
    label: str,
    *,
    maximum: int = MAX_CONTEXTS,
    sorted_required: bool,
    nonempty: bool = False,
) -> tuple[str, ...]:
    values = _digest_tuple(
        value,
        label,
        sorted_required=sorted_required,
        nonempty=nonempty,
    )
    if len(values) > maximum:
        raise ValueError(f"{label} exceeds the {maximum}-item ceiling")
    return values


def _bounded_float_tuple(
    value: Any,
    label: str,
    *,
    minimum: int = 1,
    maximum: int = MAX_LATENT_WIDTH,
) -> tuple[float, ...]:
    values = _tuple(value, label)
    if not minimum <= len(values) <= maximum:
        raise ValueError(f"{label} must contain {minimum} through {maximum} values")
    for item in values:
        _finite_float(item, f"{label} item")
    return values


def _bounded_index_tuple(
    value: Any,
    label: str,
    *,
    maximum_items: int = MAX_CONTEXTS,
    nonempty: bool = True,
) -> tuple[int, ...]:
    values = _tuple(value, label)
    if (nonempty and not values) or len(values) > maximum_items:
        qualifier = "one or more" if nonempty else "zero or more"
        raise ValueError(
            f"{label} must contain {qualifier} values within the {maximum_items}-item ceiling"
        )
    for item in values:
        _integer(item, f"{label} item")
    if len(set(values)) != len(values) or values != tuple(sorted(values)):
        raise ValueError(f"{label} must be distinct and canonically sorted")
    return values


def _domain_ref(domain: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(domain.encode("ascii") + b"\0")
    digest.update(canonical_bytes(payload))
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SituatedContext:
    """An operational context label, never a personhood or truth claim."""

    reality_mode: RealityMode
    perspective: Perspective
    subject_ref: str
    scope_ref: str
    world_ref: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reality_mode",
            _enum_value(self.reality_mode, RealityMode, "reality_mode"),
        )
        object.__setattr__(
            self,
            "perspective",
            _enum_value(self.perspective, Perspective, "perspective"),
        )
        for label, value in (
            ("subject_ref", self.subject_ref),
            ("scope_ref", self.scope_ref),
            ("world_ref", self.world_ref),
        ):
            _digest(value, label)
        _bounded_digests(
            self.evidence_refs,
            "context evidence_refs",
            sorted_required=True,
            nonempty=True,
        )

    @property
    def context_ref(self) -> str:
        return _domain_ref("angler.situated-context.v1", self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "evidence_refs": list(self.evidence_refs),
            "perspective": self.perspective.value,
            "reality_mode": self.reality_mode.value,
            "scope_ref": self.scope_ref,
            "subject_ref": self.subject_ref,
            "world_ref": self.world_ref,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SituatedContext":
        _exact_keys(
            payload,
            {
                "evidence_refs",
                "perspective",
                "reality_mode",
                "scope_ref",
                "subject_ref",
                "world_ref",
            },
            "situated context",
        )
        return cls(
            reality_mode=payload["reality_mode"],
            perspective=payload["perspective"],
            subject_ref=payload["subject_ref"],
            scope_ref=payload["scope_ref"],
            world_ref=payload["world_ref"],
            evidence_refs=_payload_tuple(payload["evidence_refs"], "evidence_refs"),
        )


@dataclass(frozen=True, slots=True)
class SituatedStateLineage(_CanonicalContract):
    """Evidence-bound identity for one learned situated-state snapshot."""

    CONTRACT: ClassVar[str] = "ANG-CTR-SITUATED-STATE-LINEAGE-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "lineage_ref"

    context: SituatedContext
    agent_ref: str
    parent_lineage_ref: str | None
    state_step: int
    checkpoint_ref: str
    config_ref: str
    competence_state_digest: str
    snapshot_digest: str
    world_state_digest: str
    self_state_digest: str
    focus_state_digest: str
    outcome_state_digest: str
    state_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, SituatedContext):
            raise TypeError("context must be a SituatedContext")
        if self.context.reality_mode not in {RealityMode.ACTUAL, RealityMode.SIMULATED}:
            raise ValueError("persisted state lineage must be ACTUAL or SIMULATED")
        _digest(self.agent_ref, "agent_ref")
        _digest(self.parent_lineage_ref, "parent_lineage_ref", optional=True)
        _integer(self.state_step, "state_step")
        if self.parent_lineage_ref is not None and self.state_step == 0:
            raise ValueError("a state-step-zero lineage cannot have a parent")
        for label, value in (
            ("checkpoint_ref", self.checkpoint_ref),
            ("config_ref", self.config_ref),
            ("competence_state_digest", self.competence_state_digest),
            ("snapshot_digest", self.snapshot_digest),
            ("world_state_digest", self.world_state_digest),
            ("self_state_digest", self.self_state_digest),
            ("focus_state_digest", self.focus_state_digest),
            ("outcome_state_digest", self.outcome_state_digest),
        ):
            _digest(value, label)
        _bounded_digests(
            self.state_evidence_refs,
            "state_evidence_refs",
            sorted_required=True,
            nonempty=True,
        )

    @property
    def lineage_ref(self) -> str:
        return self.content_ref

    def assert_successor(self, child: "SituatedStateLineage") -> None:
        if not isinstance(child, SituatedStateLineage):
            raise TypeError("child must be a SituatedStateLineage")
        if child.parent_lineage_ref != self.lineage_ref:
            raise ValueError("child lineage does not name the exact parent")
        if child.state_step != self.state_step + 1:
            raise ValueError("child lineage must advance exactly one state step")
        parent_identity = (
            self.agent_ref,
            self.context.reality_mode,
            self.context.perspective,
            self.context.subject_ref,
            self.context.scope_ref,
            self.context.world_ref,
            self.checkpoint_ref,
            self.config_ref,
        )
        child_identity = (
            child.agent_ref,
            child.context.reality_mode,
            child.context.perspective,
            child.context.subject_ref,
            child.context.scope_ref,
            child.context.world_ref,
            child.checkpoint_ref,
            child.config_ref,
        )
        if child_identity != parent_identity:
            raise ValueError("child lineage drifted from the parent identity boundary")

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent_ref": self.agent_ref,
            "checkpoint_ref": self.checkpoint_ref,
            "competence_state_digest": self.competence_state_digest,
            "config_ref": self.config_ref,
            "context": self.context.to_payload(),
            "contract": self.CONTRACT,
            "focus_state_digest": self.focus_state_digest,
            "outcome_state_digest": self.outcome_state_digest,
            "parent_lineage_ref": self.parent_lineage_ref,
            "self_state_digest": self.self_state_digest,
            "snapshot_digest": self.snapshot_digest,
            "state_evidence_refs": list(self.state_evidence_refs),
            "state_step": self.state_step,
            "world_state_digest": self.world_state_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "agent_ref",
            "checkpoint_ref",
            "competence_state_digest",
            "config_ref",
            "context",
            "contract",
            "focus_state_digest",
            "outcome_state_digest",
            "parent_lineage_ref",
            "self_state_digest",
            "snapshot_digest",
            "state_evidence_refs",
            "state_step",
            "world_state_digest",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        return cls(
            context=SituatedContext.from_payload(payload["context"]),
            agent_ref=payload["agent_ref"],
            parent_lineage_ref=payload["parent_lineage_ref"],
            state_step=payload["state_step"],
            checkpoint_ref=payload["checkpoint_ref"],
            config_ref=payload["config_ref"],
            competence_state_digest=payload["competence_state_digest"],
            snapshot_digest=payload["snapshot_digest"],
            world_state_digest=payload["world_state_digest"],
            self_state_digest=payload["self_state_digest"],
            focus_state_digest=payload["focus_state_digest"],
            outcome_state_digest=payload["outcome_state_digest"],
            state_evidence_refs=_payload_tuple(
                payload["state_evidence_refs"], "state_evidence_refs"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProspectiveResourceEnvelope:
    branch_capacity: int
    read_capacity: int
    latent_width: int
    recurrent_step_capacity: int
    horizon_capacity: int

    def __post_init__(self) -> None:
        _integer(self.branch_capacity, "branch_capacity", minimum=2)
        if self.branch_capacity > MAX_BRANCH_CAPACITY:
            raise ValueError("branch_capacity exceeds the 4096-item resource ceiling")
        _integer(self.read_capacity, "read_capacity", minimum=1)
        if self.read_capacity > MAX_CONTEXTS:
            raise ValueError("read_capacity exceeds the 256-item resource ceiling")
        _integer(self.latent_width, "latent_width", minimum=1)
        if self.latent_width > MAX_LATENT_WIDTH:
            raise ValueError("latent_width exceeds the 4096-value resource ceiling")
        _integer(self.recurrent_step_capacity, "recurrent_step_capacity", minimum=1)
        if self.recurrent_step_capacity > MAX_RECURRENT_STEPS:
            raise ValueError("recurrent_step_capacity exceeds the 256-step ceiling")
        _integer(self.horizon_capacity, "horizon_capacity", minimum=1)
        if self.horizon_capacity > MAX_HORIZON:
            raise ValueError("horizon_capacity exceeds the 256-step ceiling")

    def to_payload(self) -> dict[str, int]:
        return {
            "branch_capacity": self.branch_capacity,
            "horizon_capacity": self.horizon_capacity,
            "latent_width": self.latent_width,
            "read_capacity": self.read_capacity,
            "recurrent_step_capacity": self.recurrent_step_capacity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProspectiveResourceEnvelope":
        _exact_keys(
            payload,
            {
                "branch_capacity",
                "horizon_capacity",
                "latent_width",
                "read_capacity",
                "recurrent_step_capacity",
            },
            "prospective resource envelope",
        )
        return cls(
            branch_capacity=payload["branch_capacity"],
            read_capacity=payload["read_capacity"],
            latent_width=payload["latent_width"],
            recurrent_step_capacity=payload["recurrent_step_capacity"],
            horizon_capacity=payload["horizon_capacity"],
        )


@dataclass(frozen=True, slots=True)
class ProspectiveBranch:
    """One learned row in a jointly committed sibling batch."""

    parent_lineage_ref: str
    task_id: str
    request_ref: str
    candidate_index: int
    candidate_trace: str
    context: SituatedContext
    predicted_next_latent: tuple[float, ...]
    outcome_logit: float
    uncertainty: float
    selection_residual: float
    decision_logit: float
    horizon: int
    recurrent_steps: int
    focused_context_indices: tuple[int, ...]
    focus_weights: tuple[float, ...]
    supporting_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.parent_lineage_ref, "branch parent_lineage_ref")
        _identity(self.task_id, "branch task_id")
        _digest(self.request_ref, "branch request_ref")
        _integer(self.candidate_index, "candidate_index")
        _text(self.candidate_trace, "candidate_trace", 8_192)
        if not isinstance(self.context, SituatedContext):
            raise TypeError("branch context must be a SituatedContext")
        if self.context.reality_mode is not RealityMode.HYPOTHETICAL:
            raise ValueError("prospective branch context must be HYPOTHETICAL")
        _bounded_float_tuple(self.predicted_next_latent, "predicted_next_latent")
        _finite_float(self.outcome_logit, "outcome_logit")
        _finite_float(self.uncertainty, "uncertainty", nonnegative=True)
        _finite_float(self.selection_residual, "selection_residual")
        _finite_float(self.decision_logit, "decision_logit")
        _integer(self.horizon, "horizon", minimum=1)
        _integer(self.recurrent_steps, "recurrent_steps", minimum=1)
        focused = _bounded_index_tuple(
            self.focused_context_indices,
            "focused_context_indices",
        )
        weights = _tuple(self.focus_weights, "focus_weights")
        if len(weights) != len(focused):
            raise ValueError("focus_weights must align one-to-one with focused indices")
        for value in weights:
            _finite_float(value, "focus weight", nonnegative=True)
        if not math.isclose(
            math.fsum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=FOCUS_SUM_TOLERANCE,
        ):
            raise ValueError("focus weights must sum to one within the fixed codec tolerance")
        _bounded_digests(
            self.supporting_evidence_refs,
            "branch supporting_evidence_refs",
            sorted_required=True,
            nonempty=True,
        )

    @property
    def trace_ref(self) -> str:
        return _content_ref(self.candidate_trace)

    @property
    def candidate_ref(self) -> str:
        return _domain_ref(
            "angler.prospective-candidate.v1",
            {
                "candidate_index": self.candidate_index,
                "task_id": self.task_id,
                "trace_ref": self.trace_ref,
            },
        )

    @property
    def branch_ref(self) -> str:
        return _domain_ref("angler.prospective-branch.v1", self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "candidate_trace": self.candidate_trace,
            "context": self.context.to_payload(),
            "decision_logit_hex": self.decision_logit.hex(),
            "focus_weights_hex": [item.hex() for item in self.focus_weights],
            "focused_context_indices": list(self.focused_context_indices),
            "horizon": self.horizon,
            "outcome_logit_hex": self.outcome_logit.hex(),
            "parent_lineage_ref": self.parent_lineage_ref,
            "predicted_next_latent_hex": [
                item.hex() for item in self.predicted_next_latent
            ],
            "recurrent_steps": self.recurrent_steps,
            "request_ref": self.request_ref,
            "selection_residual_hex": self.selection_residual.hex(),
            "supporting_evidence_refs": list(self.supporting_evidence_refs),
            "task_id": self.task_id,
            "uncertainty_hex": self.uncertainty.hex(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProspectiveBranch":
        keys = {
            "candidate_index",
            "candidate_trace",
            "context",
            "decision_logit_hex",
            "focus_weights_hex",
            "focused_context_indices",
            "horizon",
            "outcome_logit_hex",
            "parent_lineage_ref",
            "predicted_next_latent_hex",
            "recurrent_steps",
            "request_ref",
            "selection_residual_hex",
            "supporting_evidence_refs",
            "task_id",
            "uncertainty_hex",
        }
        _exact_keys(payload, keys, "prospective branch")
        latent = _payload_tuple(
            payload["predicted_next_latent_hex"], "predicted_next_latent_hex"
        )
        weights = _payload_tuple(payload["focus_weights_hex"], "focus_weights_hex")
        return cls(
            parent_lineage_ref=payload["parent_lineage_ref"],
            task_id=payload["task_id"],
            request_ref=payload["request_ref"],
            candidate_index=payload["candidate_index"],
            candidate_trace=payload["candidate_trace"],
            context=SituatedContext.from_payload(payload["context"]),
            predicted_next_latent=tuple(
                _hex_float(item, f"predicted_next_latent_hex[{index}]")
                for index, item in enumerate(latent)
            ),
            outcome_logit=_hex_float(payload["outcome_logit_hex"], "outcome_logit_hex"),
            uncertainty=_hex_float(payload["uncertainty_hex"], "uncertainty_hex"),
            selection_residual=_hex_float(
                payload["selection_residual_hex"], "selection_residual_hex"
            ),
            decision_logit=_hex_float(
                payload["decision_logit_hex"], "decision_logit_hex"
            ),
            horizon=payload["horizon"],
            recurrent_steps=payload["recurrent_steps"],
            focused_context_indices=_payload_tuple(
                payload["focused_context_indices"], "focused_context_indices"
            ),
            focus_weights=tuple(
                _hex_float(item, f"focus_weights_hex[{index}]")
                for index, item in enumerate(weights)
            ),
            supporting_evidence_refs=_payload_tuple(
                payload["supporting_evidence_refs"], "supporting_evidence_refs"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProspectiveDynamicsBatch(_CanonicalContract):
    """All learned sibling hypotheses committed under one parent lineage."""

    CONTRACT: ClassVar[str] = "ANG-CTR-PROSPECTIVE-DYNAMICS-BATCH-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "batch_ref"

    parent_lineage: SituatedStateLineage
    parent_sequence: int
    parent_event_ref: str | None
    parent_acquisition_ref: str | None
    task_id: str
    request: str
    model_ref: str
    encoder_ref: str
    dynamics_checkpoint_ref: str
    dynamics_config_ref: str
    decision_evidence_ref: str
    recalled_refs: tuple[str, ...]
    context_refs: tuple[str, ...]
    eligibility_rows: tuple[tuple[int, ...], ...]
    resources: ProspectiveResourceEnvelope
    branches: tuple[ProspectiveBranch, ...]
    selected_branch_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_lineage, SituatedStateLineage):
            raise TypeError("parent_lineage must be a SituatedStateLineage")
        _integer(self.parent_sequence, "parent_sequence")
        _digest(self.parent_event_ref, "parent_event_ref", optional=True)
        if (self.parent_sequence == 0) != (self.parent_event_ref is None):
            raise ValueError("parent_event_ref must be absent only at sequence zero")
        _digest(
            self.parent_acquisition_ref,
            "parent_acquisition_ref",
            optional=True,
        )
        _identity(self.task_id, "task_id")
        _text(self.request, "request", 16_384)
        for label, value in (
            ("model_ref", self.model_ref),
            ("encoder_ref", self.encoder_ref),
            ("dynamics_checkpoint_ref", self.dynamics_checkpoint_ref),
            ("dynamics_config_ref", self.dynamics_config_ref),
            ("decision_evidence_ref", self.decision_evidence_ref),
            ("selected_branch_ref", self.selected_branch_ref),
        ):
            _digest(value, label)
        if self.dynamics_checkpoint_ref != self.parent_lineage.checkpoint_ref:
            raise ValueError("batch checkpoint does not match parent lineage")
        if self.dynamics_config_ref != self.parent_lineage.config_ref:
            raise ValueError("batch config does not match parent lineage")
        _bounded_digests(
            self.recalled_refs,
            "recalled_refs",
            sorted_required=False,
        )
        context_refs = _bounded_digests(
            self.context_refs,
            "context_refs",
            sorted_required=False,
            nonempty=True,
        )
        if not isinstance(self.resources, ProspectiveResourceEnvelope):
            raise TypeError("resources must be a ProspectiveResourceEnvelope")
        branches = _tuple(self.branches, "branches")
        if not 2 <= len(branches) <= self.resources.branch_capacity:
            raise ValueError("branches do not fit the declared branch capacity")
        if any(not isinstance(item, ProspectiveBranch) for item in branches):
            raise TypeError("branches must contain ProspectiveBranch values")
        if tuple(item.candidate_index for item in branches) != tuple(range(len(branches))):
            raise ValueError("branch candidate indices must be exactly 0 through N-1")
        for label, values in (
            ("branch refs", tuple(item.branch_ref for item in branches)),
            ("candidate refs", tuple(item.candidate_ref for item in branches)),
            ("trace refs", tuple(item.trace_ref for item in branches)),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
        rows = _tuple(self.eligibility_rows, "eligibility_rows")
        if len(rows) != len(branches):
            raise ValueError("eligibility_rows must align one-to-one with branches")
        request_ref = self.request_ref
        parent_context = self.parent_lineage.context
        recalled = set(self.recalled_refs)
        context_ref_set = set(context_refs)
        for index, (branch, raw_row) in enumerate(zip(branches, rows, strict=True)):
            row = _bounded_index_tuple(raw_row, f"eligibility_rows[{index}]")
            if row and row[-1] >= len(context_refs):
                raise ValueError("eligibility row references an absent context")
            expected_context = (
                RealityMode.HYPOTHETICAL,
                parent_context.perspective,
                parent_context.subject_ref,
                parent_context.scope_ref,
                parent_context.world_ref,
            )
            actual_context = (
                branch.context.reality_mode,
                branch.context.perspective,
                branch.context.subject_ref,
                branch.context.scope_ref,
                branch.context.world_ref,
            )
            if actual_context != expected_context:
                raise ValueError("branch context drifted from the parent lineage")
            if branch.parent_lineage_ref != self.parent_lineage.lineage_ref:
                raise ValueError("branch does not name the exact parent lineage")
            if branch.task_id != self.task_id or branch.request_ref != request_ref:
                raise ValueError("branch task/request identity drifted from the batch")
            if len(branch.predicted_next_latent) != self.resources.latent_width:
                raise ValueError("branch latent width does not match the resource envelope")
            if branch.horizon > self.resources.horizon_capacity:
                raise ValueError("branch horizon exceeds the resource envelope")
            if branch.recurrent_steps > self.resources.recurrent_step_capacity:
                raise ValueError("branch recurrence exceeds the resource envelope")
            if len(branch.focused_context_indices) > self.resources.read_capacity:
                raise ValueError("branch focus exceeds the declared read capacity")
            if not set(branch.focused_context_indices).issubset(row):
                raise ValueError("learned focus escaped caller-declared eligibility")
            if not set(branch.context.evidence_refs).issubset(context_ref_set):
                raise ValueError("branch context cites evidence outside the context set")
            if not set(branch.supporting_evidence_refs).issubset(recalled):
                raise ValueError("branch support is not present in recalled evidence")
        if sum(item.branch_ref == self.selected_branch_ref for item in branches) != 1:
            raise ValueError("selected_branch_ref must occur exactly once in the batch")

    @property
    def request_ref(self) -> str:
        return _content_ref(self.request)

    @property
    def batch_ref(self) -> str:
        return self.content_ref

    @property
    def selected_branch(self) -> ProspectiveBranch:
        return next(item for item in self.branches if item.branch_ref == self.selected_branch_ref)

    @property
    def selected_index(self) -> int:
        return self.selected_branch.candidate_index

    def to_payload(self) -> dict[str, Any]:
        return {
            "branches": [item.to_payload() for item in self.branches],
            "context_refs": list(self.context_refs),
            "contract": self.CONTRACT,
            "decision_evidence_ref": self.decision_evidence_ref,
            "dynamics_checkpoint_ref": self.dynamics_checkpoint_ref,
            "dynamics_config_ref": self.dynamics_config_ref,
            "eligibility_rows": [list(row) for row in self.eligibility_rows],
            "encoder_ref": self.encoder_ref,
            "model_ref": self.model_ref,
            "parent_acquisition_ref": self.parent_acquisition_ref,
            "parent_event_ref": self.parent_event_ref,
            "parent_lineage": self.parent_lineage.to_payload(),
            "parent_sequence": self.parent_sequence,
            "recalled_refs": list(self.recalled_refs),
            "request": self.request,
            "resources": self.resources.to_payload(),
            "selected_branch_ref": self.selected_branch_ref,
            "task_id": self.task_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "branches",
            "context_refs",
            "contract",
            "decision_evidence_ref",
            "dynamics_checkpoint_ref",
            "dynamics_config_ref",
            "eligibility_rows",
            "encoder_ref",
            "model_ref",
            "parent_acquisition_ref",
            "parent_event_ref",
            "parent_lineage",
            "parent_sequence",
            "recalled_refs",
            "request",
            "resources",
            "selected_branch_ref",
            "task_id",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        branch_payloads = _payload_list(payload["branches"], "branches")
        raw_rows = _payload_list(payload["eligibility_rows"], "eligibility_rows")
        return cls(
            parent_lineage=SituatedStateLineage.from_payload(payload["parent_lineage"]),
            parent_sequence=payload["parent_sequence"],
            parent_event_ref=payload["parent_event_ref"],
            parent_acquisition_ref=payload["parent_acquisition_ref"],
            task_id=payload["task_id"],
            request=payload["request"],
            model_ref=payload["model_ref"],
            encoder_ref=payload["encoder_ref"],
            dynamics_checkpoint_ref=payload["dynamics_checkpoint_ref"],
            dynamics_config_ref=payload["dynamics_config_ref"],
            decision_evidence_ref=payload["decision_evidence_ref"],
            recalled_refs=_payload_tuple(payload["recalled_refs"], "recalled_refs"),
            context_refs=_payload_tuple(payload["context_refs"], "context_refs"),
            eligibility_rows=tuple(
                _payload_tuple(row, f"eligibility_rows[{index}]")
                for index, row in enumerate(raw_rows)
            ),
            resources=ProspectiveResourceEnvelope.from_payload(payload["resources"]),
            branches=tuple(ProspectiveBranch.from_payload(item) for item in branch_payloads),
            selected_branch_ref=payload["selected_branch_ref"],
        )


@dataclass(frozen=True, slots=True)
class ProspectiveTurnReservationV2(_CanonicalContract):
    """Compatibility wrapper; external idempotency remains the legacy ref."""

    CONTRACT: ClassVar[str] = "ANG-CTR-PROSPECTIVE-TURN-RESERVATION-002@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "reservation_ref"

    legacy_reservation: ProspectiveTurnReservation
    batch: ProspectiveDynamicsBatch

    def __post_init__(self) -> None:
        if not isinstance(self.legacy_reservation, ProspectiveTurnReservation):
            raise TypeError("legacy_reservation must be a ProspectiveTurnReservation")
        if not isinstance(self.batch, ProspectiveDynamicsBatch):
            raise TypeError("batch must be a ProspectiveDynamicsBatch")
        legacy = self.legacy_reservation
        batch = self.batch
        parent = batch.parent_lineage
        common_legacy = (
            legacy.parent_sequence,
            legacy.parent_event_ref,
            legacy.parent_competence_digest,
            legacy.parent_snapshot_digest,
            legacy.model_ref,
            legacy.encoder_ref,
            legacy.agent_ref,
            legacy.world_ref,
            legacy.task_id,
            legacy.request,
            legacy.recalled_refs,
            legacy.decision_evidence_ref,
        )
        common_batch = (
            batch.parent_sequence,
            batch.parent_event_ref,
            parent.competence_state_digest,
            parent.snapshot_digest,
            batch.model_ref,
            batch.encoder_ref,
            parent.agent_ref,
            parent.context.world_ref,
            batch.task_id,
            batch.request,
            batch.recalled_refs,
            batch.decision_evidence_ref,
        )
        if common_batch != common_legacy:
            raise ValueError("legacy reservation and prospective batch context differ")
        if len(legacy.proposals) != len(batch.branches):
            raise ValueError("legacy proposals must biject to every batch branch")
        for index, (proposal, logit, branch) in enumerate(
            zip(legacy.proposals, legacy.logits, batch.branches, strict=True)
        ):
            if (
                branch.candidate_index != index
                or branch.candidate_trace != proposal
                or branch.decision_logit.hex() != logit.hex()
            ):
                raise ValueError("legacy proposal/logit does not match its batch branch")
        selected = batch.selected_branch
        commitment = legacy.commitment
        if batch.selected_index != legacy.selected_index:
            raise ValueError("batch selection does not match the legacy reservation")
        selected_compatibility = (
            selected.candidate_index,
            selected.candidate_trace,
            selected.decision_logit.hex(),
            selected.uncertainty.hex(),
            selected.horizon,
            selected.supporting_evidence_refs,
            selected.parent_lineage_ref,
        )
        commitment_compatibility = (
            commitment.candidate_index,
            commitment.candidate_trace,
            commitment.predicted_score.hex(),
            None if commitment.uncertainty is None else commitment.uncertainty.hex(),
            commitment.horizon,
            legacy.supporting_evidence_refs,
            parent.lineage_ref,
        )
        if selected_compatibility != commitment_compatibility:
            raise ValueError("selected branch does not reproduce the legacy commitment")

    @property
    def reservation_ref(self) -> str:
        return self.content_ref

    @property
    def legacy_reservation_ref(self) -> str:
        return self.legacy_reservation.reservation_ref

    @property
    def batch_ref(self) -> str:
        return self.batch.batch_ref

    @property
    def parent_lineage_ref(self) -> str:
        return self.batch.parent_lineage.lineage_ref

    @property
    def selected_branch_ref(self) -> str:
        return self.batch.selected_branch_ref

    @property
    def commitment_ref(self) -> str:
        return self.legacy_reservation.commitment_ref

    @property
    def idempotency_key(self) -> str:
        return self.legacy_reservation.reservation_ref

    def to_payload(self) -> dict[str, Any]:
        return {
            "batch": self.batch.to_payload(),
            "contract": self.CONTRACT,
            "legacy_reservation": self.legacy_reservation.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_contract(
            payload,
            {"batch", "contract", "legacy_reservation"},
            cls.CONTRACT,
        )
        return cls(
            legacy_reservation=ProspectiveTurnReservation.from_payload(
                payload["legacy_reservation"]
            ),
            batch=ProspectiveDynamicsBatch.from_payload(payload["batch"]),
        )


@dataclass(frozen=True, slots=True)
class BranchResolution:
    branch_ref: str
    status: BranchStatus

    def __post_init__(self) -> None:
        _digest(self.branch_ref, "branch resolution ref")
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, BranchStatus, "branch status"),
        )

    def to_payload(self) -> dict[str, str]:
        return {"branch_ref": self.branch_ref, "status": self.status.value}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BranchResolution":
        _exact_keys(payload, {"branch_ref", "status"}, "branch resolution")
        return cls(branch_ref=payload["branch_ref"], status=payload["status"])


@dataclass(frozen=True, slots=True)
class ProspectiveResolution(_CanonicalContract):
    """Append-only lifecycle resolution; never derives outcome from a receipt."""

    CONTRACT: ClassVar[str] = "ANG-CTR-PROSPECTIVE-RESOLUTION-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "resolution_ref"

    disposition: ResolutionDisposition
    reservation: ProspectiveTurnReservationV2
    execution_request: CognitiveExecutionRequest | None
    execution_receipt: CognitiveExecutionReceipt | None
    objective_feedback: ObjectiveFeedbackRecord | None
    child_lineage: SituatedStateLineage | None
    observed_next_latent: tuple[float, ...] | None
    observed_outcome: str | None
    prediction_error_metric: str | None
    prediction_error: float | None
    feedback_source_ref: str | None
    observation_evidence_refs: tuple[str, ...]
    branch_statuses: tuple[BranchResolution, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum_value(
                self.disposition,
                ResolutionDisposition,
                "resolution disposition",
            ),
        )
        if not isinstance(self.reservation, ProspectiveTurnReservationV2):
            raise TypeError("reservation must be a ProspectiveTurnReservationV2")
        evidence = _bounded_digests(
            self.observation_evidence_refs,
            "observation_evidence_refs",
            sorted_required=True,
        )
        statuses = _tuple(self.branch_statuses, "branch_statuses")
        if any(not isinstance(item, BranchResolution) for item in statuses):
            raise TypeError("branch_statuses must contain BranchResolution values")
        branches = self.reservation.batch.branches
        if tuple(item.branch_ref for item in statuses) != tuple(
            item.branch_ref for item in branches
        ):
            raise ValueError("branch_statuses must cover every batch branch in order")
        selected_index = self.reservation.batch.selected_index
        expected_selected_status = BranchStatus(self.disposition.value)
        for index, status in enumerate(statuses):
            expected = expected_selected_status if index == selected_index else BranchStatus.OPEN
            if status.status is not expected:
                raise ValueError("only the selected branch may carry the lifecycle status")

        request = self.execution_request
        receipt = self.execution_receipt
        feedback = self.objective_feedback
        if request is not None:
            if not isinstance(request, CognitiveExecutionRequest):
                raise TypeError("execution_request has the wrong contract type")
            request.assert_reservation(self.reservation.legacy_reservation)
        if receipt is not None:
            if request is None or not isinstance(receipt, CognitiveExecutionReceipt):
                raise TypeError("execution_receipt requires an exact execution request")
            receipt.assert_request(request)
        if feedback is not None:
            if request is None or receipt is None or not isinstance(
                feedback, ObjectiveFeedbackRecord
            ):
                raise TypeError("objective_feedback requires the exact execution context")
            feedback.assert_context(self.reservation.legacy_reservation, request, receipt)

        if self.disposition is ResolutionDisposition.OBSERVED:
            if request is None or receipt is None or feedback is None:
                raise ValueError("OBSERVED resolution requires request, receipt, and feedback")
            if receipt.status != "COMPLETED":
                raise ValueError("OBSERVED resolution requires a completed receipt")
            if not isinstance(self.child_lineage, SituatedStateLineage):
                raise TypeError("OBSERVED resolution requires a child lineage")
            self.reservation.batch.parent_lineage.assert_successor(self.child_lineage)
            if self.observed_next_latent is None:
                raise ValueError("OBSERVED resolution requires an observed next latent")
            latent = _bounded_float_tuple(
                self.observed_next_latent,
                "observed_next_latent",
            )
            if len(latent) != self.reservation.batch.resources.latent_width:
                raise ValueError("observed next latent width differs from the batch")
            if not evidence:
                raise ValueError("OBSERVED resolution requires cited observation evidence")
            if self.observed_outcome != feedback.outcome:
                raise ValueError("observed outcome differs from objective feedback")
            if self.feedback_source_ref != feedback.feedback_source_ref:
                raise ValueError("feedback source differs from objective feedback")
            if self.prediction_error_metric != PREDICTION_ERROR_METRIC:
                raise ValueError("prediction error metric is not registered")
            _finite_float(self.prediction_error, "prediction_error")
            label = 1.0 if feedback.outcome == "success" else -1.0
            expected_error = -label * self.reservation.batch.selected_branch.outcome_logit
            if self.prediction_error.hex() != expected_error.hex():
                raise ValueError("prediction_error does not match the declared measurement")
        else:
            if any(
                item is not None
                for item in (
                    feedback,
                    self.child_lineage,
                    self.observed_next_latent,
                    self.observed_outcome,
                    self.prediction_error_metric,
                    self.prediction_error,
                    self.feedback_source_ref,
                )
            ):
                raise ValueError("non-observed lifecycle cannot carry outcome or child state")
            if self.disposition is ResolutionDisposition.CANCELLED:
                if request is not None or receipt is not None:
                    raise ValueError("CANCELLED resolution cannot carry execution material")
            else:
                if request is None or receipt is None:
                    raise ValueError("recorded execution lifecycle requires request and receipt")
                expected_receipt = {
                    ResolutionDisposition.COMPLETED_UNEVALUATED: "COMPLETED",
                    ResolutionDisposition.CLARIFICATION_REQUIRED: "CLARIFICATION_REQUIRED",
                    ResolutionDisposition.ERROR: "ERROR",
                }[self.disposition]
                if receipt.status != expected_receipt:
                    raise ValueError("receipt status differs from the lifecycle disposition")

    @property
    def resolution_ref(self) -> str:
        return self.content_ref

    @property
    def batch_ref(self) -> str:
        return self.reservation.batch_ref

    @property
    def selected_branch_ref(self) -> str:
        return self.reservation.selected_branch_ref

    @property
    def parent_lineage(self) -> SituatedStateLineage:
        return self.reservation.batch.parent_lineage

    @property
    def parent_lineage_ref(self) -> str:
        return self.parent_lineage.lineage_ref

    @classmethod
    def observed(
        cls,
        *,
        reservation: ProspectiveTurnReservationV2,
        execution_request: CognitiveExecutionRequest,
        execution_receipt: CognitiveExecutionReceipt,
        objective_feedback: ObjectiveFeedbackRecord,
        child_lineage: SituatedStateLineage,
        observed_next_latent: tuple[float, ...],
        observation_evidence_refs: tuple[str, ...],
    ) -> Self:
        selected = reservation.batch.selected_branch
        label = 1.0 if objective_feedback.outcome == "success" else -1.0
        return cls(
            disposition=ResolutionDisposition.OBSERVED,
            reservation=reservation,
            execution_request=execution_request,
            execution_receipt=execution_receipt,
            objective_feedback=objective_feedback,
            child_lineage=child_lineage,
            observed_next_latent=observed_next_latent,
            observed_outcome=objective_feedback.outcome,
            prediction_error_metric=PREDICTION_ERROR_METRIC,
            prediction_error=-label * selected.outcome_logit,
            feedback_source_ref=objective_feedback.feedback_source_ref,
            observation_evidence_refs=observation_evidence_refs,
            branch_statuses=tuple(
                BranchResolution(
                    branch_ref=branch.branch_ref,
                    status=(
                        BranchStatus.OBSERVED
                        if branch.candidate_index == reservation.batch.selected_index
                        else BranchStatus.OPEN
                    ),
                )
                for branch in reservation.batch.branches
            ),
        )

    @classmethod
    def lifecycle(
        cls,
        *,
        disposition: ResolutionDisposition,
        reservation: ProspectiveTurnReservationV2,
        execution_request: CognitiveExecutionRequest | None = None,
        execution_receipt: CognitiveExecutionReceipt | None = None,
        observation_evidence_refs: tuple[str, ...] = (),
    ) -> Self:
        disposition = _enum_value(
            disposition,
            ResolutionDisposition,
            "resolution disposition",
        )
        if disposition is ResolutionDisposition.OBSERVED:
            raise ValueError("use observed() for an objective outcome")
        selected_status = BranchStatus(disposition.value)
        return cls(
            disposition=disposition,
            reservation=reservation,
            execution_request=execution_request,
            execution_receipt=execution_receipt,
            objective_feedback=None,
            child_lineage=None,
            observed_next_latent=None,
            observed_outcome=None,
            prediction_error_metric=None,
            prediction_error=None,
            feedback_source_ref=None,
            observation_evidence_refs=observation_evidence_refs,
            branch_statuses=tuple(
                BranchResolution(
                    branch_ref=branch.branch_ref,
                    status=(
                        selected_status
                        if branch.candidate_index == reservation.batch.selected_index
                        else BranchStatus.OPEN
                    ),
                )
                for branch in reservation.batch.branches
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "branch_statuses": [item.to_payload() for item in self.branch_statuses],
            "child_lineage": (
                None if self.child_lineage is None else self.child_lineage.to_payload()
            ),
            "contract": self.CONTRACT,
            "disposition": self.disposition.value,
            "execution_receipt": (
                None
                if self.execution_receipt is None
                else self.execution_receipt.to_payload()
            ),
            "execution_request": (
                None
                if self.execution_request is None
                else self.execution_request.to_payload()
            ),
            "feedback_source_ref": self.feedback_source_ref,
            "objective_feedback": (
                None
                if self.objective_feedback is None
                else self.objective_feedback.to_payload()
            ),
            "observation_evidence_refs": list(self.observation_evidence_refs),
            "observed_next_latent_hex": (
                None
                if self.observed_next_latent is None
                else [item.hex() for item in self.observed_next_latent]
            ),
            "observed_outcome": self.observed_outcome,
            "prediction_error_hex": (
                None if self.prediction_error is None else self.prediction_error.hex()
            ),
            "prediction_error_metric": self.prediction_error_metric,
            "reservation": self.reservation.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "branch_statuses",
            "child_lineage",
            "contract",
            "disposition",
            "execution_receipt",
            "execution_request",
            "feedback_source_ref",
            "objective_feedback",
            "observation_evidence_refs",
            "observed_next_latent_hex",
            "observed_outcome",
            "prediction_error_hex",
            "prediction_error_metric",
            "reservation",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        raw_latent = payload["observed_next_latent_hex"]
        latent = None
        if raw_latent is not None:
            values = _payload_tuple(raw_latent, "observed_next_latent_hex")
            latent = tuple(
                _hex_float(item, f"observed_next_latent_hex[{index}]")
                for index, item in enumerate(values)
            )
        raw_error = payload["prediction_error_hex"]
        return cls(
            disposition=payload["disposition"],
            reservation=ProspectiveTurnReservationV2.from_payload(payload["reservation"]),
            execution_request=(
                None
                if payload["execution_request"] is None
                else CognitiveExecutionRequest.from_payload(payload["execution_request"])
            ),
            execution_receipt=(
                None
                if payload["execution_receipt"] is None
                else CognitiveExecutionReceipt.from_payload(payload["execution_receipt"])
            ),
            objective_feedback=(
                None
                if payload["objective_feedback"] is None
                else ObjectiveFeedbackRecord.from_payload(payload["objective_feedback"])
            ),
            child_lineage=(
                None
                if payload["child_lineage"] is None
                else SituatedStateLineage.from_payload(payload["child_lineage"])
            ),
            observed_next_latent=latent,
            observed_outcome=payload["observed_outcome"],
            prediction_error_metric=payload["prediction_error_metric"],
            prediction_error=(
                None
                if raw_error is None
                else _hex_float(raw_error, "prediction_error_hex")
            ),
            feedback_source_ref=payload["feedback_source_ref"],
            observation_evidence_refs=_payload_tuple(
                payload["observation_evidence_refs"], "observation_evidence_refs"
            ),
            branch_statuses=tuple(
                BranchResolution.from_payload(item)
                for item in _payload_list(payload["branch_statuses"], "branch_statuses")
            ),
        )


@dataclass(frozen=True, slots=True)
class CognitiveEpisodeV2(_CanonicalContract):
    """Observed-only successor episode pointing one way to its resolution."""

    CONTRACT: ClassVar[str] = "ANG-CTR-COGNITIVE-EPISODE-002@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "episode_ref"

    legacy_episode: CognitiveEpisode
    resolution: ProspectiveResolution

    def __post_init__(self) -> None:
        if not isinstance(self.legacy_episode, CognitiveEpisode):
            raise TypeError("legacy_episode must be a CognitiveEpisode")
        if not isinstance(self.resolution, ProspectiveResolution):
            raise TypeError("resolution must be a ProspectiveResolution")
        if self.resolution.disposition is not ResolutionDisposition.OBSERVED:
            raise ValueError("CognitiveEpisodeV2 requires an OBSERVED resolution")
        request = self.resolution.execution_request
        receipt = self.resolution.execution_receipt
        feedback = self.resolution.objective_feedback
        child = self.resolution.child_lineage
        if request is None or receipt is None or feedback is None or child is None:
            raise ValueError("observed resolution is missing its exact compatibility view")
        legacy = self.resolution.reservation.legacy_reservation
        episode = self.legacy_episode
        expected = (
            legacy.task_id,
            legacy.request,
            legacy.recalled_refs,
            legacy.proposals,
            legacy.selected_index,
            legacy.commitment,
            receipt.response,
            (*request.input_observations, *receipt.output_observations),
            feedback.outcome,
            feedback.feedback_text,
            feedback.feedback_source_ref,
            self.resolution.parent_lineage.competence_state_digest,
            child.competence_state_digest,
            legacy.model_ref,
            legacy.encoder_ref,
            legacy.supporting_evidence_refs,
        )
        actual = (
            episode.task_id,
            episode.request,
            episode.recalled_refs,
            episode.proposals,
            episode.selected_index,
            episode.commitment,
            episode.response,
            episode.observations,
            episode.outcome,
            episode.feedback_text,
            episode.feedback_source_ref,
            episode.parent_state_digest,
            episode.child_state_digest,
            episode.model_ref,
            episode.encoder_ref,
            episode.supporting_evidence_refs,
        )
        if actual != expected:
            raise ValueError("legacy episode does not match the observed resolution")

    @property
    def episode_ref(self) -> str:
        return self.content_ref

    @property
    def resolution_ref(self) -> str:
        return self.resolution.resolution_ref

    @property
    def reservation_ref(self) -> str:
        return self.resolution.reservation.reservation_ref

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "legacy_episode": self.legacy_episode.to_payload(),
            "resolution": self.resolution.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_contract(
            payload,
            {"contract", "legacy_episode", "resolution"},
            cls.CONTRACT,
        )
        return cls(
            legacy_episode=CognitiveEpisode.from_payload(payload["legacy_episode"]),
            resolution=ProspectiveResolution.from_payload(payload["resolution"]),
        )


__all__ = [
    "BranchResolution",
    "BranchStatus",
    "CognitiveEpisodeV2",
    "FOCUS_SUM_TOLERANCE",
    "MAX_BRANCH_CAPACITY",
    "MAX_CONTEXTS",
    "MAX_HORIZON",
    "MAX_LATENT_WIDTH",
    "MAX_RECURRENT_STEPS",
    "PREDICTION_ERROR_METRIC",
    "Perspective",
    "ProspectiveBranch",
    "ProspectiveDynamicsBatch",
    "ProspectiveResourceEnvelope",
    "ProspectiveResolution",
    "ProspectiveTurnReservationV2",
    "RealityMode",
    "ResolutionDisposition",
    "SituatedContext",
    "SituatedStateLineage",
]
