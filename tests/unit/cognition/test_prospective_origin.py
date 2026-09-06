from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import unittest

from angler.cognition.contracts import CognitiveEpisode, ProspectiveCommitment
from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
    ObjectiveFeedbackRecord,
    ProspectiveTurnReservation,
)
from angler.cognition.prospective_origin import (
    BranchResolution,
    BranchStatus,
    CognitiveEpisodeV2,
    PREDICTION_ERROR_METRIC,
    Perspective,
    ProspectiveBranch,
    ProspectiveDynamicsBatch,
    ProspectiveResourceEnvelope,
    ProspectiveResolution,
    ProspectiveTurnReservationV2,
    RealityMode,
    ResolutionDisposition,
    SituatedContext,
    SituatedStateLineage,
)
from angler.episodes.canonical import canonical_bytes


def ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def context(
    mode: RealityMode = RealityMode.ACTUAL,
    *,
    subject_ref: str | None = None,
    scope_ref: str | None = None,
    world_ref: str | None = None,
    evidence_refs: tuple[str, ...] | None = None,
) -> SituatedContext:
    return SituatedContext(
        reality_mode=mode,
        perspective=Perspective.AGENT_SITUATED,
        subject_ref=ref("subject") if subject_ref is None else subject_ref,
        scope_ref=ref("scope") if scope_ref is None else scope_ref,
        world_ref=ref("world") if world_ref is None else world_ref,
        evidence_refs=(
            tuple(sorted((ref("context-a"), ref("context-b"))))
            if evidence_refs is None
            else evidence_refs
        ),
    )


def lineage(
    *,
    parent: SituatedStateLineage | None = None,
    mode: RealityMode = RealityMode.ACTUAL,
    context_value: SituatedContext | None = None,
) -> SituatedStateLineage:
    step = 0 if parent is None else parent.state_step + 1
    return SituatedStateLineage(
        context=context(mode) if context_value is None else context_value,
        agent_ref=ref("agent"),
        parent_lineage_ref=None if parent is None else parent.lineage_ref,
        state_step=step,
        checkpoint_ref=ref("checkpoint"),
        config_ref=ref("config"),
        competence_state_digest=ref(f"competence-{step}"),
        snapshot_digest=ref(f"snapshot-{step}"),
        world_state_digest=ref(f"world-state-{step}"),
        self_state_digest=ref(f"self-state-{step}"),
        focus_state_digest=ref(f"focus-state-{step}"),
        outcome_state_digest=ref(f"outcome-state-{step}"),
        state_evidence_refs=tuple(
            sorted((ref(f"state-evidence-{step}"), ref("state-evidence-common")))
        ),
    )


def branches_for(
    parent: SituatedStateLineage,
    *,
    count: int,
    request: str,
    latent_width: int,
    support: tuple[str, ...],
    context_refs: tuple[str, ...],
) -> tuple[ProspectiveBranch, ...]:
    hypothetical = context(
        RealityMode.HYPOTHETICAL,
        subject_ref=parent.context.subject_ref,
        scope_ref=parent.context.scope_ref,
        world_ref=parent.context.world_ref,
        evidence_refs=tuple(sorted(context_refs)),
    )
    request_ref = CognitiveEpisode.proposal_ref(request)
    return tuple(
        ProspectiveBranch(
            parent_lineage_ref=parent.lineage_ref,
            task_id="task-origin",
            request_ref=request_ref,
            candidate_index=index,
            candidate_trace=f"proposal {index}",
            context=hypothetical,
            predicted_next_latent=tuple(
                float(index + offset + 1) / float(count + latent_width)
                for offset in range(latent_width)
            ),
            outcome_logit=(float(index) - float(count) / 2.0) / float(count),
            uncertainty=0.125 + float(index) / float(count + 1),
            selection_residual=float(index) / float(10 * count),
            decision_logit=float(index) / float(count),
            horizon=1 + index % 3,
            recurrent_steps=1 + index % 2,
            focused_context_indices=(0, 1),
            focus_weights=(0.25, 0.75),
            supporting_evidence_refs=support,
        )
        for index in range(count)
    )


def batch(
    *,
    count: int = 7,
    latent_width: int = 3,
    branch_capacity: int | None = None,
) -> ProspectiveDynamicsBatch:
    parent = lineage()
    request = "Solve the bounded public task."
    support = (ref("support"),)
    recalled = (ref("recall"), *support)
    context_refs = (ref("world-row-a"), ref("world-row-b"))
    branch_values = branches_for(
        parent,
        count=count,
        request=request,
        latent_width=latent_width,
        support=support,
        context_refs=context_refs,
    )
    selected_index = count // 2
    return ProspectiveDynamicsBatch(
        parent_lineage=parent,
        parent_sequence=0,
        parent_event_ref=None,
        parent_acquisition_ref=None,
        task_id="task-origin",
        request=request,
        model_ref=ref("model"),
        encoder_ref=ref("encoder"),
        dynamics_checkpoint_ref=parent.checkpoint_ref,
        dynamics_config_ref=parent.config_ref,
        decision_evidence_ref=ref("decision"),
        recalled_refs=recalled,
        context_refs=context_refs,
        eligibility_rows=tuple((0, 1) for _ in branch_values),
        resources=ProspectiveResourceEnvelope(
            branch_capacity=(count if branch_capacity is None else branch_capacity),
            read_capacity=2,
            latent_width=latent_width,
            recurrent_step_capacity=2,
            horizon_capacity=3,
        ),
        branches=branch_values,
        selected_branch_ref=branch_values[selected_index].branch_ref,
    )


def wrapper(*, count: int = 7, latent_width: int = 3) -> ProspectiveTurnReservationV2:
    value = batch(count=count, latent_width=latent_width)
    selected = value.selected_branch
    commitment = ProspectiveCommitment(
        parent_event_ref=value.parent_event_ref,
        task_id=value.task_id,
        candidate_index=selected.candidate_index,
        candidate_trace=selected.candidate_trace,
        predicted_score=selected.decision_logit,
        uncertainty=selected.uncertainty,
        horizon=selected.horizon,
        competence_state_digest=value.parent_lineage.competence_state_digest,
    )
    legacy = ProspectiveTurnReservation.create(
        pending_blob=b"exact prospective pending state",
        parent_sequence=value.parent_sequence,
        parent_event_ref=value.parent_event_ref,
        parent_competence_digest=value.parent_lineage.competence_state_digest,
        parent_snapshot_digest=value.parent_lineage.snapshot_digest,
        model_ref=value.model_ref,
        encoder_ref=value.encoder_ref,
        agent_ref=value.parent_lineage.agent_ref,
        world_ref=value.parent_lineage.context.world_ref,
        task_id=value.task_id,
        request=value.request,
        recalled_refs=value.recalled_refs,
        proposals=tuple(item.candidate_trace for item in value.branches),
        selected_index=value.selected_index,
        selected_trace=selected.candidate_trace,
        logits=tuple(item.decision_logit for item in value.branches),
        decision_evidence_ref=value.decision_evidence_ref,
        supporting_evidence_refs=selected.supporting_evidence_refs,
        commitment=commitment,
    )
    return ProspectiveTurnReservationV2(legacy_reservation=legacy, batch=value)


def observed_resolution() -> tuple[
    ProspectiveResolution,
    CognitiveExecutionRequest,
    CognitiveExecutionReceipt,
    ObjectiveFeedbackRecord,
]:
    reservation = wrapper()
    request = CognitiveExecutionRequest.from_reservation(
        reservation.legacy_reservation,
        input_observations=("public input",),
    )
    receipt = CognitiveExecutionReceipt.from_request(
        request,
        status="COMPLETED",
        executed_trace=request.selected_trace,
        response="Observable response",
        output_observations=("public output",),
    )
    feedback = ObjectiveFeedbackRecord.from_execution(
        reservation.legacy_reservation,
        request,
        receipt,
        outcome="success",
        feedback_text="The external objective check passed.",
        feedback_source_ref=ref("feedback-source"),
    )
    child = lineage(
        parent=reservation.batch.parent_lineage,
        context_value=reservation.batch.parent_lineage.context,
    )
    resolution = ProspectiveResolution.observed(
        reservation=reservation,
        execution_request=request,
        execution_receipt=receipt,
        objective_feedback=feedback,
        child_lineage=child,
        observed_next_latent=tuple(
            0.5 + index / 10.0
            for index in range(reservation.batch.resources.latent_width)
        ),
        observation_evidence_refs=(ref("observed-next-state"),),
    )
    return resolution, request, receipt, feedback


def legacy_episode(resolution: ProspectiveResolution) -> CognitiveEpisode:
    legacy = resolution.reservation.legacy_reservation
    request = resolution.execution_request
    receipt = resolution.execution_receipt
    feedback = resolution.objective_feedback
    child = resolution.child_lineage
    assert request is not None and receipt is not None and feedback is not None
    assert child is not None
    return CognitiveEpisode(
        task_id=legacy.task_id,
        request=legacy.request,
        recalled_refs=legacy.recalled_refs,
        proposals=legacy.proposals,
        selected_index=legacy.selected_index,
        commitment=legacy.commitment,
        response=receipt.response,
        observations=(*request.input_observations, *receipt.output_observations),
        outcome=feedback.outcome,
        feedback_text=feedback.feedback_text,
        feedback_source_ref=feedback.feedback_source_ref,
        parent_state_digest=resolution.parent_lineage.competence_state_digest,
        child_state_digest=child.competence_state_digest,
        model_ref=legacy.model_ref,
        encoder_ref=legacy.encoder_ref,
        supporting_evidence_refs=legacy.supporting_evidence_refs,
    )


class SituatedStateLineageTests(unittest.TestCase):
    def test_lineage_round_trip_and_exact_successor(self) -> None:
        parent = lineage(mode=RealityMode.SIMULATED)
        child = lineage(parent=parent, mode=RealityMode.SIMULATED, context_value=parent.context)
        parent.assert_successor(child)
        for value in (parent, child):
            encoded = value.canonical_bytes()
            restored = SituatedStateLineage.from_json(encoded)
            self.assertEqual(restored, value)
            self.assertEqual(restored.canonical_bytes(), encoded)
            self.assertRegex(value.lineage_ref, r"^sha256:[0-9a-f]{64}$")
        with self.assertRaises(FrozenInstanceError):
            parent.state_step = 9  # type: ignore[misc]

    def test_context_and_successor_identity_drift_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            lineage(mode=RealityMode.HYPOTHETICAL)
        with self.assertRaises(ValueError):
            context("UNREGISTERED")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SituatedContext(
                reality_mode=RealityMode.ACTUAL,
                perspective="EXTERNAL_OBSERVER",  # type: ignore[arg-type]
                subject_ref=ref("subject"),
                scope_ref=ref("scope"),
                world_ref=ref("world"),
                evidence_refs=(ref("evidence"),),
            )
        with self.assertRaises(ValueError):
            replace(lineage(), state_evidence_refs=())

        parent = lineage()
        child = lineage(parent=parent, context_value=parent.context)
        drifted_context = replace(child.context, scope_ref=ref("other-scope"))
        with self.assertRaises(ValueError):
            parent.assert_successor(replace(child, context=drifted_context))
        with self.assertRaises(ValueError):
            parent.assert_successor(replace(child, checkpoint_ref=ref("other-checkpoint")))
        with self.assertRaises(ValueError):
            parent.assert_successor(replace(child, state_step=3))


class ProspectiveDynamicsBatchTests(unittest.TestCase):
    def test_dynamic_branch_counts_and_latent_widths_round_trip(self) -> None:
        for count, width, capacity in ((2, 1, 2), (7, 3, 16), (64, 5, 64), (65, 2, 96)):
            with self.subTest(count=count, width=width):
                value = batch(count=count, latent_width=width, branch_capacity=capacity)
                encoded = value.canonical_bytes()
                restored = ProspectiveDynamicsBatch.from_json(encoded)
                self.assertEqual(restored, value)
                self.assertEqual(restored.canonical_bytes(), encoded)
                self.assertEqual(len(value.branches), count)
                self.assertEqual(len(value.branches[-1].predicted_next_latent), width)
        with self.assertRaises(ValueError):
            wrapper(count=65)

    def test_focus_eligibility_context_and_resource_tampering_fail(self) -> None:
        value = batch()
        first = value.branches[0]
        escaped = replace(first, focused_context_indices=(0, 2), focus_weights=(0.5, 0.5))
        with self.assertRaises(ValueError):
            replace(value, branches=(escaped, *value.branches[1:]))
        with self.assertRaises(ValueError):
            replace(value, eligibility_rows=((0,), *value.eligibility_rows[1:]))
        with self.assertRaises(ValueError):
            replace(
                value,
                branches=(
                    replace(first, predicted_next_latent=(0.0,)),
                    *value.branches[1:],
                ),
            )
        drifted_context = replace(first.context, subject_ref=ref("other-subject"))
        with self.assertRaises(ValueError):
            replace(value, branches=(replace(first, context=drifted_context), *value.branches[1:]))
        with self.assertRaises(ValueError):
            replace(first, focus_weights=(0.2, 0.2))
        with self.assertRaises(ValueError):
            replace(first, uncertainty=-0.1)

    def test_branch_order_evidence_and_selection_are_exact(self) -> None:
        value = batch()
        with self.assertRaises(ValueError):
            replace(value, branches=tuple(reversed(value.branches)))
        with self.assertRaises(ValueError):
            replace(value, selected_branch_ref=ref("absent-branch"))
        with self.assertRaises(ValueError):
            replace(
                value,
                branches=(
                    replace(value.branches[0], supporting_evidence_refs=(ref("hidden"),)),
                    *value.branches[1:],
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                value,
                branches=(
                    replace(
                        value.branches[0],
                        context=replace(
                            value.branches[0].context,
                            evidence_refs=(ref("outside-context-set"),),
                        ),
                    ),
                    *value.branches[1:],
                ),
            )


class SuccessorReservationTests(unittest.TestCase):
    def test_wrapper_bijects_every_legacy_proposal_and_keeps_legacy_effect_key(self) -> None:
        value = wrapper()
        encoded = value.canonical_bytes()
        self.assertEqual(ProspectiveTurnReservationV2.from_json(encoded), value)
        self.assertEqual(value.idempotency_key, value.legacy_reservation.reservation_ref)
        self.assertNotEqual(value.reservation_ref, value.idempotency_key)
        for index, branch in enumerate(value.batch.branches):
            self.assertEqual(branch.candidate_trace, value.legacy_reservation.proposals[index])
            self.assertEqual(branch.decision_logit.hex(), value.legacy_reservation.logits[index].hex())

    def test_wrapper_rejects_nonselected_proposal_and_logit_drift(self) -> None:
        value = wrapper()
        legacy = value.legacy_reservation
        proposals = list(legacy.proposals)
        proposals[0] = "different public proposal"
        with self.assertRaises(ValueError):
            ProspectiveTurnReservationV2(
                legacy_reservation=replace(legacy, proposals=tuple(proposals)),
                batch=value.batch,
            )
        logits = list(legacy.logits)
        logits[0] += 0.25
        with self.assertRaises(ValueError):
            ProspectiveTurnReservationV2(
                legacy_reservation=replace(legacy, logits=tuple(logits)),
                batch=value.batch,
            )


class ProspectiveResolutionTests(unittest.TestCase):
    def test_observed_resolution_requires_exact_feedback_and_measurement(self) -> None:
        value, _request, _receipt, feedback = observed_resolution()
        encoded = value.canonical_bytes()
        restored = ProspectiveResolution.from_json(encoded)
        self.assertEqual(restored, value)
        self.assertEqual(restored.canonical_bytes(), encoded)
        self.assertEqual(value.observed_outcome, feedback.outcome)
        self.assertEqual(value.prediction_error_metric, PREDICTION_ERROR_METRIC)
        self.assertTrue(all(
            item.status is (BranchStatus.OBSERVED if index == value.reservation.batch.selected_index else BranchStatus.OPEN)
            for index, item in enumerate(value.branch_statuses)
        ))
        with self.assertRaises(ValueError):
            replace(value, objective_feedback=None)
        with self.assertRaises(ValueError):
            replace(value, prediction_error=value.prediction_error + 0.125)  # type: ignore[operator]
        with self.assertRaises(ValueError):
            replace(value, observed_outcome="failure")
        with self.assertRaises(ValueError):
            replace(value, observation_evidence_refs=())
        assert value.child_lineage is not None
        with self.assertRaises(ValueError):
            replace(value, child_lineage=replace(value.child_lineage, parent_lineage_ref=ref("wrong")))

    def test_nonobjective_lifecycle_never_carries_outcome_or_child_update(self) -> None:
        reservation = wrapper()
        request = CognitiveExecutionRequest.from_reservation(reservation.legacy_reservation)
        receipts = {
            ResolutionDisposition.COMPLETED_UNEVALUATED: CognitiveExecutionReceipt.from_request(
                request,
                status="COMPLETED",
                executed_trace=request.selected_trace,
                response="Unscored response",
            ),
            ResolutionDisposition.CLARIFICATION_REQUIRED: CognitiveExecutionReceipt.from_request(
                request,
                status="CLARIFICATION_REQUIRED",
                executed_trace=request.selected_trace,
                response="Need a public clarification.",
            ),
            ResolutionDisposition.ERROR: CognitiveExecutionReceipt.from_request(
                request,
                status="ERROR",
                executed_trace=request.selected_trace,
                response="",
            ),
        }
        cancelled = ProspectiveResolution.lifecycle(
            disposition=ResolutionDisposition.CANCELLED,
            reservation=reservation,
        )
        self.assertIsNone(cancelled.execution_request)
        for disposition, receipt in receipts.items():
            with self.subTest(disposition=disposition):
                value = ProspectiveResolution.lifecycle(
                    disposition=disposition,
                    reservation=reservation,
                    execution_request=request,
                    execution_receipt=receipt,
                )
                self.assertEqual(ProspectiveResolution.from_json(value.canonical_bytes()), value)
                self.assertIsNone(value.observed_outcome)
                self.assertIsNone(value.child_lineage)
                self.assertIsNone(value.objective_feedback)
        with self.assertRaises(ValueError):
            replace(cancelled, observed_outcome="success")
        with self.assertRaises(ValueError):
            ProspectiveResolution.lifecycle(
                disposition=ResolutionDisposition.ERROR,
                reservation=reservation,
                execution_request=request,
                execution_receipt=receipts[ResolutionDisposition.CLARIFICATION_REQUIRED],
            )

    def test_branch_status_coverage_and_closed_vocab_fail_closed(self) -> None:
        value, _request, _receipt, _feedback = observed_resolution()
        with self.assertRaises(ValueError):
            replace(value, branch_statuses=value.branch_statuses[:-1])
        wrong = list(value.branch_statuses)
        wrong[0] = BranchResolution(wrong[0].branch_ref, BranchStatus.ERROR)
        with self.assertRaises(ValueError):
            replace(value, branch_statuses=tuple(wrong))
        with self.assertRaises(ValueError):
            replace(value, disposition="UNREGISTERED")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            replace(value, prediction_error_metric="adaptive-after-results-v2")


class CognitiveEpisodeV2Tests(unittest.TestCase):
    def test_episode_is_observed_only_and_content_graph_is_one_way(self) -> None:
        resolution, _request, _receipt, _feedback = observed_resolution()
        value = CognitiveEpisodeV2(
            legacy_episode=legacy_episode(resolution),
            resolution=resolution,
        )
        encoded = value.canonical_bytes()
        self.assertEqual(CognitiveEpisodeV2.from_json(encoded), value)
        self.assertEqual(value.resolution_ref, resolution.resolution_ref)
        self.assertNotIn("episode_ref", {item.name for item in fields(ProspectiveResolution)})
        self.assertNotIn("episode_ref", resolution.to_payload())

        cancelled = ProspectiveResolution.lifecycle(
            disposition=ResolutionDisposition.CANCELLED,
            reservation=resolution.reservation,
        )
        with self.assertRaises(ValueError):
            CognitiveEpisodeV2(legacy_episode=value.legacy_episode, resolution=cancelled)
        with self.assertRaises(ValueError):
            replace(value, legacy_episode=replace(value.legacy_episode, outcome="failure"))


class CanonicalStrictnessTests(unittest.TestCase):
    def test_every_top_level_contract_is_strict_and_content_addressed(self) -> None:
        resolution, _request, _receipt, _feedback = observed_resolution()
        values = (
            resolution.parent_lineage,
            resolution.reservation.batch,
            resolution.reservation,
            resolution,
            CognitiveEpisodeV2(legacy_episode(resolution), resolution),
        )
        for value in values:
            with self.subTest(contract=value.CONTRACT):
                encoded = value.canonical_bytes()
                self.assertEqual(type(value).from_json(encoded), value)
                self.assertEqual(type(value).from_json(encoded).content_ref, value.content_ref)
                payload = value.to_payload()
                with self.assertRaises(ValueError):
                    type(value).from_json(canonical_bytes({**payload, "extra": True}))
                with self.assertRaises(ValueError):
                    type(value).from_json(
                        canonical_bytes({**payload, "contract": value.CONTRACT + "-unknown"})
                    )
                missing = dict(payload)
                missing.pop(next(key for key in missing if key != "contract"))
                with self.assertRaises(ValueError):
                    type(value).from_json(canonical_bytes(missing))
                with self.assertRaises(ValueError):
                    type(value).from_json(encoded + b" ")

    def test_schema_has_no_task_solution_or_fixed_candidate_axis_fields(self) -> None:
        names = {item.name for item in fields(ProspectiveBranch)}
        self.assertTrue({"predicted_next_latent", "outcome_logit", "focus_weights"} <= names)
        self.assertFalse({"answer", "correct_answer", "verifier", "task_solution"} & names)
        self.assertGreater(batch(count=65, branch_capacity=128).resources.branch_capacity, 64)


if __name__ == "__main__":
    unittest.main()
