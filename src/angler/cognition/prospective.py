"""Canonical records for durable prospective execution.

These records make a selected procedure and its execution boundary durable.
They do not choose a procedure, infer whether an external effect occurred, or
turn feedback into a solution.  Every identity is the SHA-256 of strict
canonical JSON under Angler's existing cognition contract profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, ClassVar, Mapping, Self

from angler.cognition.contracts import (
    ProspectiveCommitment,
    _CanonicalContract,
    _digest,
    _digest_tuple,
    _exact_contract,
    _hex_float,
    _identity,
    _integer,
    _payload_tuple,
    _text,
    _tuple,
)


MAX_PENDING_BLOB_BYTES = 16 * 1024 * 1024
_MIN_PROPOSALS = 2
_MAX_PROPOSALS = 64
_MAX_OBSERVATIONS = 64
_MAX_OBSERVATION_CHARS = 8_192
_EXECUTION_STATUSES = frozenset(
    {"COMPLETED", "CLARIFICATION_REQUIRED", "ERROR"}
)
_OUTCOMES = frozenset({"success", "failure"})


def _blob_identity(value: Any) -> tuple[str, int]:
    if type(value) is not bytes:
        raise TypeError("pending_blob must be exact bytes")
    size = len(value)
    if not 1 <= size <= MAX_PENDING_BLOB_BYTES:
        raise ValueError(
            "pending_blob must contain between 1 byte and 16 MiB inclusive"
        )
    return "sha256:" + hashlib.sha256(value).hexdigest(), size


def _bounded_observations(value: Any, label: str) -> tuple[str, ...]:
    observations = _tuple(value, label)
    if len(observations) > _MAX_OBSERVATIONS:
        raise ValueError(f"{label} exceeds the {_MAX_OBSERVATIONS}-item ceiling")
    total = 0
    for observation in observations:
        _text(observation, f"{label} item", _MAX_OBSERVATION_CHARS)
        total += len(observation)
    if total > _MAX_OBSERVATIONS * _MAX_OBSERVATION_CHARS:
        raise ValueError(f"{label} exceeds its aggregate character ceiling")
    return observations


@dataclass(frozen=True, slots=True)
class ProspectiveTurnReservation(_CanonicalContract):
    """One selected public decision durably reserved before execution."""

    CONTRACT: ClassVar[str] = "ANG-CTR-PROSPECTIVE-TURN-RESERVATION-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "reservation_ref"

    parent_sequence: int
    parent_event_ref: str | None
    parent_competence_digest: str
    parent_snapshot_digest: str
    model_ref: str
    encoder_ref: str
    agent_ref: str
    world_ref: str
    task_id: str
    request: str
    recalled_refs: tuple[str, ...]
    proposals: tuple[str, ...]
    selected_index: int
    selected_trace: str
    logits: tuple[float, ...]
    decision_evidence_ref: str
    supporting_evidence_refs: tuple[str, ...]
    commitment: ProspectiveCommitment
    pending_blob_sha256: str
    pending_blob_size: int

    def __post_init__(self) -> None:
        _integer(self.parent_sequence, "parent_sequence")
        _digest(self.parent_event_ref, "parent_event_ref", optional=True)
        if (self.parent_sequence == 0) != (self.parent_event_ref is None):
            raise ValueError(
                "parent_event_ref must be absent only for parent sequence zero"
            )
        for label, value in (
            ("parent_competence_digest", self.parent_competence_digest),
            ("parent_snapshot_digest", self.parent_snapshot_digest),
            ("model_ref", self.model_ref),
            ("encoder_ref", self.encoder_ref),
            ("agent_ref", self.agent_ref),
            ("world_ref", self.world_ref),
            ("decision_evidence_ref", self.decision_evidence_ref),
            ("pending_blob_sha256", self.pending_blob_sha256),
        ):
            _digest(value, label)
        _identity(self.task_id, "task_id")
        _text(self.request, "request", 16_384)
        _digest_tuple(self.recalled_refs, "recalled_refs", sorted_required=False)

        proposals = _tuple(self.proposals, "proposals")
        if not _MIN_PROPOSALS <= len(proposals) <= _MAX_PROPOSALS:
            raise ValueError("proposals must contain between 2 and 64 public proposals")
        for proposal in proposals:
            _text(proposal, "proposal", 8_192)
        if len(set(proposals)) != len(proposals):
            raise ValueError("public proposals must be distinct")

        _integer(self.selected_index, "selected_index")
        if self.selected_index >= len(proposals):
            raise ValueError("selected_index is outside the proposal set")
        _text(self.selected_trace, "selected_trace", 8_192)
        if self.selected_trace != proposals[self.selected_index]:
            raise ValueError("selected_trace does not match the selected public proposal")

        logits = _tuple(self.logits, "logits")
        if len(logits) != len(proposals):
            raise ValueError("logits must align one-to-one with proposals")
        if any(type(item) is not float or not math.isfinite(item) for item in logits):
            raise ValueError("logits must contain only finite floats")

        _digest_tuple(
            self.supporting_evidence_refs,
            "supporting_evidence_refs",
            sorted_required=True,
        )
        if not set(self.supporting_evidence_refs).issubset(self.recalled_refs):
            raise ValueError("supporting_evidence_refs must be a subset of recalled_refs")

        if not isinstance(self.commitment, ProspectiveCommitment):
            raise TypeError("commitment must be a ProspectiveCommitment")
        commitment = self.commitment
        if commitment.parent_event_ref != self.parent_event_ref:
            raise ValueError("commitment parent_event_ref does not match the reservation")
        if commitment.competence_state_digest != self.parent_competence_digest:
            raise ValueError("commitment competence state does not match the reservation")
        if commitment.task_id != self.task_id:
            raise ValueError("commitment task_id does not match the reservation")
        if commitment.candidate_index != self.selected_index:
            raise ValueError("commitment candidate_index does not match the reservation")
        if commitment.candidate_trace != self.selected_trace:
            raise ValueError("commitment candidate_trace does not match the reservation")
        if commitment.predicted_score.hex() != logits[self.selected_index].hex():
            raise ValueError("commitment score does not match the selected decision logit")

        _integer(self.pending_blob_size, "pending_blob_size", minimum=1)
        if self.pending_blob_size > MAX_PENDING_BLOB_BYTES:
            raise ValueError("pending_blob_size exceeds the 16 MiB ceiling")

    @classmethod
    def create(cls, *, pending_blob: bytes, **fields: Any) -> Self:
        """Bind exact opaque learner bytes without interpreting them."""

        digest, size = _blob_identity(pending_blob)
        return cls(
            pending_blob_sha256=digest,
            pending_blob_size=size,
            **fields,
        )

    @property
    def reservation_ref(self) -> str:
        return self.content_ref

    @property
    def commitment_ref(self) -> str:
        return self.commitment.commitment_ref

    @property
    def idempotency_key(self) -> str:
        """The sole external idempotency identity mandated by the leaf."""

        # A commitment identifies the selected prediction but intentionally
        # predates the complete execution context.  The reservation additionally
        # binds request, acting agent, world, recalled evidence, and pending
        # learner bytes, so only it is collision-safe as an external effect key.
        return self.reservation_ref

    def assert_pending_blob(self, pending_blob: bytes) -> None:
        digest, size = _blob_identity(pending_blob)
        if size != self.pending_blob_size or digest != self.pending_blob_sha256:
            raise ValueError("pending_blob does not match the reservation")

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent_ref": self.agent_ref,
            "commitment": self.commitment.to_payload(),
            "contract": self.CONTRACT,
            "decision_evidence_ref": self.decision_evidence_ref,
            "encoder_ref": self.encoder_ref,
            "logits_hex": [item.hex() for item in self.logits],
            "model_ref": self.model_ref,
            "parent_competence_digest": self.parent_competence_digest,
            "parent_event_ref": self.parent_event_ref,
            "parent_sequence": self.parent_sequence,
            "parent_snapshot_digest": self.parent_snapshot_digest,
            "pending_blob_sha256": self.pending_blob_sha256,
            "pending_blob_size": self.pending_blob_size,
            "proposals": list(self.proposals),
            "recalled_refs": list(self.recalled_refs),
            "request": self.request,
            "selected_index": self.selected_index,
            "selected_trace": self.selected_trace,
            "supporting_evidence_refs": list(self.supporting_evidence_refs),
            "task_id": self.task_id,
            "world_ref": self.world_ref,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "agent_ref",
            "commitment",
            "contract",
            "decision_evidence_ref",
            "encoder_ref",
            "logits_hex",
            "model_ref",
            "parent_competence_digest",
            "parent_event_ref",
            "parent_sequence",
            "parent_snapshot_digest",
            "pending_blob_sha256",
            "pending_blob_size",
            "proposals",
            "recalled_refs",
            "request",
            "selected_index",
            "selected_trace",
            "supporting_evidence_refs",
            "task_id",
            "world_ref",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        raw_logits = _payload_tuple(payload["logits_hex"], "logits_hex")
        logits = tuple(
            _hex_float(item, f"logits_hex[{index}]")
            for index, item in enumerate(raw_logits)
        )
        return cls(
            parent_sequence=payload["parent_sequence"],
            parent_event_ref=payload["parent_event_ref"],
            parent_competence_digest=payload["parent_competence_digest"],
            parent_snapshot_digest=payload["parent_snapshot_digest"],
            model_ref=payload["model_ref"],
            encoder_ref=payload["encoder_ref"],
            agent_ref=payload["agent_ref"],
            world_ref=payload["world_ref"],
            task_id=payload["task_id"],
            request=payload["request"],
            recalled_refs=_payload_tuple(payload["recalled_refs"], "recalled_refs"),
            proposals=_payload_tuple(payload["proposals"], "proposals"),
            selected_index=payload["selected_index"],
            selected_trace=payload["selected_trace"],
            logits=logits,
            decision_evidence_ref=payload["decision_evidence_ref"],
            supporting_evidence_refs=_payload_tuple(
                payload["supporting_evidence_refs"], "supporting_evidence_refs"
            ),
            commitment=ProspectiveCommitment.from_payload(payload["commitment"]),
            pending_blob_sha256=payload["pending_blob_sha256"],
            pending_blob_size=payload["pending_blob_size"],
        )


@dataclass(frozen=True, slots=True)
class CognitiveExecutionRequest(_CanonicalContract):
    """The exact request claimed before an executor may begin an effect."""

    CONTRACT: ClassVar[str] = "ANG-CTR-COGNITIVE-EXECUTION-REQUEST-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "execution_request_ref"

    reservation_ref: str
    commitment_ref: str
    idempotency_key: str
    task_id: str
    request: str
    selected_trace: str
    input_observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("reservation_ref", self.reservation_ref),
            ("commitment_ref", self.commitment_ref),
            ("idempotency_key", self.idempotency_key),
        ):
            _digest(value, label)
        if self.idempotency_key != self.reservation_ref:
            raise ValueError("idempotency_key must be the exact reservation_ref")
        _identity(self.task_id, "task_id")
        _text(self.request, "request", 16_384)
        _text(self.selected_trace, "selected_trace", 8_192)
        _bounded_observations(self.input_observations, "input_observations")

    @classmethod
    def from_reservation(
        cls,
        reservation: ProspectiveTurnReservation,
        *,
        input_observations: tuple[str, ...] = (),
    ) -> Self:
        if not isinstance(reservation, ProspectiveTurnReservation):
            raise TypeError("reservation must be a ProspectiveTurnReservation")
        return cls(
            reservation_ref=reservation.reservation_ref,
            commitment_ref=reservation.commitment_ref,
            idempotency_key=reservation.idempotency_key,
            task_id=reservation.task_id,
            request=reservation.request,
            selected_trace=reservation.selected_trace,
            input_observations=input_observations,
        )

    @property
    def execution_request_ref(self) -> str:
        return self.content_ref

    def assert_reservation(self, reservation: ProspectiveTurnReservation) -> None:
        if not isinstance(reservation, ProspectiveTurnReservation):
            raise TypeError("reservation must be a ProspectiveTurnReservation")
        expected = (
            reservation.reservation_ref,
            reservation.commitment_ref,
            reservation.idempotency_key,
            reservation.task_id,
            reservation.request,
            reservation.selected_trace,
        )
        actual = (
            self.reservation_ref,
            self.commitment_ref,
            self.idempotency_key,
            self.task_id,
            self.request,
            self.selected_trace,
        )
        if actual != expected:
            raise ValueError("execution request does not match the reservation")

    def to_payload(self) -> dict[str, Any]:
        return {
            "commitment_ref": self.commitment_ref,
            "contract": self.CONTRACT,
            "idempotency_key": self.idempotency_key,
            "input_observations": list(self.input_observations),
            "request": self.request,
            "reservation_ref": self.reservation_ref,
            "selected_trace": self.selected_trace,
            "task_id": self.task_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "commitment_ref",
            "contract",
            "idempotency_key",
            "input_observations",
            "request",
            "reservation_ref",
            "selected_trace",
            "task_id",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        return cls(
            reservation_ref=payload["reservation_ref"],
            commitment_ref=payload["commitment_ref"],
            idempotency_key=payload["idempotency_key"],
            task_id=payload["task_id"],
            request=payload["request"],
            selected_trace=payload["selected_trace"],
            input_observations=_payload_tuple(
                payload["input_observations"], "input_observations"
            ),
        )


@dataclass(frozen=True, slots=True)
class CognitiveExecutionReceipt(_CanonicalContract):
    """A public executor result bound to one exact claimed request."""

    CONTRACT: ClassVar[str] = "ANG-CTR-COGNITIVE-EXECUTION-RECEIPT-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "execution_receipt_ref"

    execution_request_ref: str
    status: str
    executed_trace: str
    response: str
    output_observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.execution_request_ref, "execution_request_ref")
        if self.status not in _EXECUTION_STATUSES:
            raise ValueError("execution status is not registered")
        _text(self.executed_trace, "executed_trace", 8_192)
        _text(self.response, "response", 16_384, allow_empty=True)
        _bounded_observations(self.output_observations, "output_observations")

    @classmethod
    def from_request(
        cls,
        request: CognitiveExecutionRequest,
        *,
        status: str,
        executed_trace: str,
        response: str,
        output_observations: tuple[str, ...] = (),
    ) -> Self:
        if not isinstance(request, CognitiveExecutionRequest):
            raise TypeError("request must be a CognitiveExecutionRequest")
        receipt = cls(
            execution_request_ref=request.execution_request_ref,
            status=status,
            executed_trace=executed_trace,
            response=response,
            output_observations=output_observations,
        )
        receipt.assert_request(request)
        return receipt

    @property
    def execution_receipt_ref(self) -> str:
        return self.content_ref

    def assert_request(self, request: CognitiveExecutionRequest) -> None:
        if not isinstance(request, CognitiveExecutionRequest):
            raise TypeError("request must be a CognitiveExecutionRequest")
        if (
            self.execution_request_ref != request.execution_request_ref
            or self.executed_trace != request.selected_trace
        ):
            raise ValueError("execution receipt does not match the exact request")

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "executed_trace": self.executed_trace,
            "execution_request_ref": self.execution_request_ref,
            "output_observations": list(self.output_observations),
            "response": self.response,
            "status": self.status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "contract",
            "executed_trace",
            "execution_request_ref",
            "output_observations",
            "response",
            "status",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        return cls(
            execution_request_ref=payload["execution_request_ref"],
            status=payload["status"],
            executed_trace=payload["executed_trace"],
            response=payload["response"],
            output_observations=_payload_tuple(
                payload["output_observations"], "output_observations"
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveFeedbackRecord(_CanonicalContract):
    """Objective feedback bound to the completed commitment before learning."""

    CONTRACT: ClassVar[str] = "ANG-CTR-OBJECTIVE-FEEDBACK-001@0.1.0"
    REF_ATTRIBUTE: ClassVar[str] = "feedback_ref"

    reservation_ref: str
    execution_receipt_ref: str
    commitment_ref: str
    task_id: str
    outcome: str
    feedback_text: str
    feedback_source_ref: str

    def __post_init__(self) -> None:
        for label, value in (
            ("reservation_ref", self.reservation_ref),
            ("execution_receipt_ref", self.execution_receipt_ref),
            ("commitment_ref", self.commitment_ref),
            ("feedback_source_ref", self.feedback_source_ref),
        ):
            _digest(value, label)
        _identity(self.task_id, "task_id")
        if self.outcome not in _OUTCOMES:
            raise ValueError("outcome must be objective success or failure")
        _text(self.feedback_text, "feedback_text", 8_192)

    @classmethod
    def from_execution(
        cls,
        reservation: ProspectiveTurnReservation,
        request: CognitiveExecutionRequest,
        receipt: CognitiveExecutionReceipt,
        *,
        outcome: str,
        feedback_text: str,
        feedback_source_ref: str,
    ) -> Self:
        if not isinstance(receipt, CognitiveExecutionReceipt):
            raise TypeError("receipt must be a CognitiveExecutionReceipt")
        if receipt.status != "COMPLETED":
            raise ValueError("objective feedback requires a completed execution")
        request.assert_reservation(reservation)
        receipt.assert_request(request)
        return cls(
            reservation_ref=reservation.reservation_ref,
            execution_receipt_ref=receipt.execution_receipt_ref,
            commitment_ref=reservation.commitment_ref,
            task_id=reservation.task_id,
            outcome=outcome,
            feedback_text=feedback_text,
            feedback_source_ref=feedback_source_ref,
        )

    @property
    def feedback_ref(self) -> str:
        return self.content_ref

    @property
    def objective_feedback_ref(self) -> str:
        return self.content_ref

    def assert_context(
        self,
        reservation: ProspectiveTurnReservation,
        request: CognitiveExecutionRequest,
        receipt: CognitiveExecutionReceipt,
    ) -> None:
        request.assert_reservation(reservation)
        receipt.assert_request(request)
        if receipt.status != "COMPLETED":
            raise ValueError("objective feedback requires a completed execution")
        if (
            self.reservation_ref != reservation.reservation_ref
            or self.execution_receipt_ref != receipt.execution_receipt_ref
            or self.commitment_ref != reservation.commitment_ref
            or self.task_id != reservation.task_id
        ):
            raise ValueError("objective feedback does not match the execution context")

    def to_payload(self) -> dict[str, Any]:
        return {
            "commitment_ref": self.commitment_ref,
            "contract": self.CONTRACT,
            "execution_receipt_ref": self.execution_receipt_ref,
            "feedback_source_ref": self.feedback_source_ref,
            "feedback_text": self.feedback_text,
            "outcome": self.outcome,
            "reservation_ref": self.reservation_ref,
            "task_id": self.task_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        keys = {
            "commitment_ref",
            "contract",
            "execution_receipt_ref",
            "feedback_source_ref",
            "feedback_text",
            "outcome",
            "reservation_ref",
            "task_id",
        }
        _exact_contract(payload, keys, cls.CONTRACT)
        return cls(
            reservation_ref=payload["reservation_ref"],
            execution_receipt_ref=payload["execution_receipt_ref"],
            commitment_ref=payload["commitment_ref"],
            task_id=payload["task_id"],
            outcome=payload["outcome"],
            feedback_text=payload["feedback_text"],
            feedback_source_ref=payload["feedback_source_ref"],
        )


__all__ = [
    "CognitiveExecutionReceipt",
    "CognitiveExecutionRequest",
    "MAX_PENDING_BLOB_BYTES",
    "ObjectiveFeedbackRecord",
    "ProspectiveTurnReservation",
]
