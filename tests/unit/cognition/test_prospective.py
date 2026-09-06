from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import unittest

from angler.cognition import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
    ObjectiveFeedbackRecord,
    ProspectiveCommitment,
    ProspectiveTurnReservation,
)
from angler.cognition.prospective import MAX_PENDING_BLOB_BYTES


def ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def commitment(
    *,
    parent_event_ref: str | None = ref("parent-event"),
    task_id: str = "task-001",
    candidate_index: int = 2,
    candidate_trace: str = "trace c",
    predicted_score: float = 0.75,
    competence_state_digest: str = ref("parent-competence"),
) -> ProspectiveCommitment:
    return ProspectiveCommitment(
        parent_event_ref=parent_event_ref,
        task_id=task_id,
        candidate_index=candidate_index,
        candidate_trace=candidate_trace,
        predicted_score=predicted_score,
        uncertainty=0.125,
        horizon=1,
        competence_state_digest=competence_state_digest,
    )


def reservation(
    *, pending_blob: bytes = b"opaque learner checkpoint"
) -> ProspectiveTurnReservation:
    return ProspectiveTurnReservation.create(
        parent_sequence=7,
        parent_event_ref=ref("parent-event"),
        parent_competence_digest=ref("parent-competence"),
        parent_snapshot_digest=ref("parent-snapshot"),
        model_ref=ref("model"),
        encoder_ref=ref("encoder"),
        agent_ref=ref("stable-angler-agent"),
        world_ref=ref("synthetic-test-world"),
        task_id="task-001",
        request="Solve the public task.",
        recalled_refs=(ref("recall-b"), ref("recall-a"), ref("recall-c")),
        proposals=("trace a", "trace b", "trace c", "trace d"),
        selected_index=2,
        selected_trace="trace c",
        logits=(-0.5, 0.25, 0.75, -1.0),
        decision_evidence_ref=ref("decision-evidence"),
        supporting_evidence_refs=tuple(sorted((ref("recall-a"), ref("recall-c")))),
        commitment=commitment(),
        pending_blob=pending_blob,
    )


def request(
    value: ProspectiveTurnReservation | None = None,
) -> CognitiveExecutionRequest:
    return CognitiveExecutionRequest.from_reservation(
        reservation() if value is None else value,
        input_observations=("public input one", "public input two"),
    )


def receipt(
    value: CognitiveExecutionRequest | None = None,
    *,
    status: str = "COMPLETED",
) -> CognitiveExecutionReceipt:
    value = request() if value is None else value
    return CognitiveExecutionReceipt.from_request(
        value,
        status=status,
        executed_trace=value.selected_trace,
        response="Public executor response.",
        output_observations=("public output",),
    )


def feedback(
    reserved: ProspectiveTurnReservation | None = None,
    claimed: CognitiveExecutionRequest | None = None,
    executed: CognitiveExecutionReceipt | None = None,
) -> ObjectiveFeedbackRecord:
    reserved = reservation() if reserved is None else reserved
    claimed = request(reserved) if claimed is None else claimed
    executed = receipt(claimed) if executed is None else executed
    return ObjectiveFeedbackRecord.from_execution(
        reserved,
        claimed,
        executed,
        outcome="success",
        feedback_text="Objective evaluator accepted the result.",
        feedback_source_ref=ref("objective-evaluator"),
    )


class ProspectiveReservationContractTests(unittest.TestCase):
    def test_all_records_round_trip_with_exact_content_identity(self) -> None:
        reserved = reservation()
        claimed = request(reserved)
        executed = receipt(claimed)
        observed = feedback(reserved, claimed, executed)
        values = (
            (reserved, ProspectiveTurnReservation, "reservation_ref"),
            (claimed, CognitiveExecutionRequest, "execution_request_ref"),
            (executed, CognitiveExecutionReceipt, "execution_receipt_ref"),
            (observed, ObjectiveFeedbackRecord, "feedback_ref"),
        )
        for value, contract, attribute in values:
            with self.subTest(contract=contract.__name__):
                encoded = value.canonical_bytes()
                restored = contract.from_json(encoded)
                self.assertEqual(restored, value)
                self.assertEqual(restored.canonical_bytes(), encoded)
                self.assertEqual(getattr(restored, attribute), value.content_ref)
                self.assertRegex(value.content_ref, r"^sha256:[0-9a-f]{64}$")
                with self.assertRaises(ValueError):
                    contract.from_json(encoded + b"\n")

    def test_reservation_binds_opaque_pending_bytes_and_is_immutable(self) -> None:
        blob = b"exact pending learner state"
        value = reservation(pending_blob=blob)
        value.assert_pending_blob(blob)
        self.assertEqual(value.pending_blob_size, len(blob))
        self.assertEqual(
            value.pending_blob_sha256,
            "sha256:" + hashlib.sha256(blob).hexdigest(),
        )
        self.assertEqual(value.idempotency_key, value.reservation_ref)
        with self.assertRaises(ValueError):
            value.assert_pending_blob(blob + b"!")
        with self.assertRaises(FrozenInstanceError):
            value.task_id = "changed"  # type: ignore[misc]

    def test_agent_and_world_are_required_content_addressed_bindings(self) -> None:
        value = reservation()
        self.assertEqual(value.agent_ref, ref("stable-angler-agent"))
        self.assertEqual(value.world_ref, ref("synthetic-test-world"))
        alternate_world = replace(value, world_ref=ref("alternate-world"))
        alternate_agent = replace(value, agent_ref=ref("alternate-agent"))
        alternate_request = replace(value, request="A distinct public request.")
        for alternate in (alternate_world, alternate_agent, alternate_request):
            with self.subTest(alternate=alternate):
                self.assertEqual(
                    value.commitment.commitment_ref,
                    alternate.commitment.commitment_ref,
                )
                self.assertNotEqual(value.reservation_ref, alternate.reservation_ref)
                self.assertNotEqual(value.idempotency_key, alternate.idempotency_key)
        self.assertEqual(
            request(value).reservation_ref,
            value.reservation_ref,
        )

        payload = value.to_payload()
        for field in ("agent_ref", "world_ref"):
            with self.subTest(field=field, condition="missing"):
                missing = dict(payload)
                del missing[field]
                with self.assertRaises(ValueError):
                    ProspectiveTurnReservation.from_payload(missing)
            with self.subTest(field=field, condition="not-a-digest"):
                with self.assertRaises(ValueError):
                    ProspectiveTurnReservation.from_payload(
                        {**payload, field: "ANG-NAMED-IDENTITY"}
                    )
            with self.subTest(field=field, condition="uppercase"):
                with self.assertRaises(ValueError):
                    ProspectiveTurnReservation.from_payload(
                        {**payload, field: payload[field].upper()}
                    )

    def test_pending_blob_type_and_numeric_ceiling_are_strict(self) -> None:
        with self.assertRaises(TypeError):
            reservation(pending_blob=bytearray(b"mutable"))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            reservation(pending_blob=b"")
        base = reservation().to_payload()
        with self.assertRaises(ValueError):
            ProspectiveTurnReservation.from_payload(
                {**base, "pending_blob_size": MAX_PENDING_BLOB_BYTES + 1}
            )

    def test_reservation_rejects_commitment_cross_binding_mismatches(self) -> None:
        value = reservation()
        base = value.to_payload()
        cases = (
            {"parent_event_ref": ref("wrong-parent-event")},
            {"competence_state_digest": ref("wrong-competence")},
            {"task_id": "different-task"},
            {"candidate_index": 1},
            {"candidate_trace": "trace b"},
            {"predicted_score_hex": float(0.5).hex()},
        )
        for change in cases:
            with self.subTest(change=change):
                altered_commitment = {**base["commitment"], **change}
                with self.assertRaises(ValueError):
                    ProspectiveTurnReservation.from_payload(
                        {**base, "commitment": altered_commitment}
                    )

    def test_reservation_rejects_invalid_decision_shape_order_and_floats(self) -> None:
        base = reservation().to_payload()
        cases = (
            {"proposals": ["only one"]},
            {
                "proposals": [f"trace {index}" for index in range(65)],
                "logits_hex": [base["logits_hex"][2] for _ in range(65)],
            },
            {"proposals": ["a", "b", "c", "c"]},
            {"selected_index": 4},
            {"selected_trace": "trace d"},
            {"logits_hex": ["0x0.0p+0", "0x0.0p+0"]},
            {"logits_hex": [0.0, 0.0, 0.0, 0.0]},
            {"logits_hex": ["0x0.0p+0", "0x0.0p+0", "inf", "0x0.0p+0"]},
            {
                "supporting_evidence_refs": list(
                    reversed(base["supporting_evidence_refs"])
                )
            },
            {"supporting_evidence_refs": [ref("not-recalled")]},
            {"recalled_refs": [ref("same"), ref("same")]},
        )
        for change in cases:
            with self.subTest(change=change):
                with self.assertRaises((TypeError, ValueError)):
                    ProspectiveTurnReservation.from_payload({**base, **change})

    def test_candidate_cardinality_is_variable_from_two_through_sixty_four(self) -> None:
        for count, selected_index in ((2, 1), (7, 5), (64, 63)):
            with self.subTest(count=count, selected_index=selected_index):
                proposals = tuple(f"trace {index}" for index in range(count))
                logits = tuple(float(index) / 8.0 for index in range(count))
                selected_trace = proposals[selected_index]
                value = ProspectiveTurnReservation.create(
                    parent_sequence=7,
                    parent_event_ref=ref("parent-event"),
                    parent_competence_digest=ref("parent-competence"),
                    parent_snapshot_digest=ref("parent-snapshot"),
                    model_ref=ref("model"),
                    encoder_ref=ref("encoder"),
                    agent_ref=ref("stable-angler-agent"),
                    world_ref=ref("synthetic-test-world"),
                    task_id="task-001",
                    request="Solve the public task.",
                    recalled_refs=(),
                    proposals=proposals,
                    selected_index=selected_index,
                    selected_trace=selected_trace,
                    logits=logits,
                    decision_evidence_ref=ref("decision-evidence"),
                    supporting_evidence_refs=(),
                    commitment=commitment(
                        candidate_index=selected_index,
                        candidate_trace=selected_trace,
                        predicted_score=logits[selected_index],
                    ),
                    pending_blob=b"opaque state",
                )
                restored = ProspectiveTurnReservation.from_json(
                    value.canonical_bytes()
                )
                self.assertEqual(restored, value)
                self.assertEqual(len(restored.proposals), count)

    def test_initial_parent_identity_is_exact(self) -> None:
        base = reservation().to_payload()
        with self.assertRaises(ValueError):
            ProspectiveTurnReservation.from_payload(
                {**base, "parent_sequence": 0}
            )
        initial_commitment = commitment(parent_event_ref=None)
        initial = ProspectiveTurnReservation.create(
            parent_sequence=0,
            parent_event_ref=None,
            parent_competence_digest=ref("parent-competence"),
            parent_snapshot_digest=ref("parent-snapshot"),
            model_ref=ref("model"),
            encoder_ref=ref("encoder"),
            agent_ref=ref("stable-angler-agent"),
            world_ref=ref("synthetic-test-world"),
            task_id="task-001",
            request="Initial task.",
            recalled_refs=(),
            proposals=("trace a", "trace b", "trace c", "trace d"),
            selected_index=2,
            selected_trace="trace c",
            logits=(-0.5, 0.25, 0.75, -1.0),
            decision_evidence_ref=ref("decision-evidence"),
            supporting_evidence_refs=(),
            commitment=initial_commitment,
            pending_blob=b"pending",
        )
        self.assertIsNone(initial.parent_event_ref)


class ExecutionAndFeedbackBindingTests(unittest.TestCase):
    def test_request_derives_only_from_exact_reservation(self) -> None:
        reserved = reservation()
        claimed = request(reserved)
        claimed.assert_reservation(reserved)
        self.assertEqual(claimed.idempotency_key, reserved.reservation_ref)
        altered = replace(reserved, request="Different public request.")
        with self.assertRaises(ValueError):
            claimed.assert_reservation(altered)
        with self.assertRaises(ValueError):
            replace(claimed, idempotency_key=ref("another-key"))

    def test_request_observations_are_immutable_and_bounded(self) -> None:
        reserved = reservation()
        with self.assertRaises(TypeError):
            CognitiveExecutionRequest.from_reservation(
                reserved,
                input_observations=["mutable list"],  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            CognitiveExecutionRequest.from_reservation(
                reserved,
                input_observations=tuple("item" for _ in range(65)),
            )

    def test_receipt_status_and_exact_request_are_enforced(self) -> None:
        claimed = request()
        for status in ("COMPLETED", "CLARIFICATION_REQUIRED", "ERROR"):
            with self.subTest(status=status):
                value = receipt(claimed, status=status)
                value.assert_request(claimed)
        with self.assertRaises(ValueError):
            receipt(claimed, status="UNKNOWN")
        with self.assertRaises(ValueError):
            CognitiveExecutionReceipt.from_request(
                claimed,
                status="COMPLETED",
                executed_trace="different trace",
                response="response",
            )
        forged_ref = replace(
            receipt(claimed), execution_request_ref=ref("other-request")
        )
        with self.assertRaises(ValueError):
            forged_ref.assert_request(claimed)

    def test_feedback_binds_the_completed_execution_chain(self) -> None:
        reserved = reservation()
        claimed = request(reserved)
        executed = receipt(claimed)
        value = feedback(reserved, claimed, executed)
        value.assert_context(reserved, claimed, executed)
        self.assertEqual(value.objective_feedback_ref, value.feedback_ref)
        with self.assertRaises(ValueError):
            feedback(
                reserved,
                claimed,
                receipt(claimed, status="CLARIFICATION_REQUIRED"),
            )
        with self.assertRaises(ValueError):
            replace(value, task_id="wrong-task").assert_context(
                reserved, claimed, executed
            )
        with self.assertRaises(ValueError):
            replace(value, execution_receipt_ref=ref("other-receipt")).assert_context(
                reserved, claimed, executed
            )

    def test_payloads_reject_unknown_fields_and_noncanonical_arrays(self) -> None:
        values = (
            (reservation(), ProspectiveTurnReservation),
            (request(), CognitiveExecutionRequest),
            (receipt(), CognitiveExecutionReceipt),
            (feedback(), ObjectiveFeedbackRecord),
        )
        for value, contract in values:
            with self.subTest(contract=contract.__name__):
                with self.assertRaises(ValueError):
                    contract.from_payload({**value.to_payload(), "extra": True})
        base = request().to_payload()
        with self.assertRaises(TypeError):
            CognitiveExecutionRequest.from_payload(
                {**base, "input_observations": ("not", "json")}
            )


if __name__ == "__main__":
    unittest.main()
