"""Evaluator-owned counterfactual certificates for operator aliases.

The caller supplies only frozen symbolic mirrors, bindings, challenges, and
commitments.  This module independently reconstructs each mirror proposal,
executes each commitment exactly once through the held-out evaluator, and
derives the certificate result from observed transitions.  It accepts no
caller-provided success bit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from angler.procedures.alignment import (
    CounterfactualExecutionCertificate,
    StructuralIsomorphismCandidate,
    find_structural_isomorphisms,
)
from angler.procedures.grounding import (
    GroundedOperatorPrediction,
    StateOperatorBinding,
    instantiate_operator,
)
from angler.procedures.operators import LearnedOperator
from experiments.evaluators.causal_operator_suite import (
    CommittedActionSequence,
    OperatorCaseResult,
    OperatorChallenge,
    evaluate_committed_sequence,
)


EVALUATOR_ISSUER = "angler.evaluator.operator-alias-certificate.v1"
_EVALUATOR_VERSION = "angler.operator-alias-certificate.v1"


class AliasCertificateEvaluationError(ValueError):
    """Raised when certificate inputs are malformed or structurally tampered."""


def certify_counterfactual_alignment(
    candidate: StructuralIsomorphismCandidate,
    source_operator: LearnedOperator,
    target_operator: LearnedOperator,
    source_binding: StateOperatorBinding,
    target_binding: StateOperatorBinding,
    source_challenge: OperatorChallenge,
    target_challenge: OperatorChallenge,
    source_commitment: CommittedActionSequence,
    target_commitment: CommittedActionSequence,
) -> CounterfactualExecutionCertificate:
    """Execute both frozen proposals once and issue a derived certificate.

    Structural mismatches raise before either world is touched.  Well-formed
    proposals that block, apply only partially, miss an exact goal, or disagree
    with their mirror's complete observed delta receive a ``fail`` certificate.
    """

    source_prediction, target_prediction = _validate_and_instantiate(
        candidate,
        source_operator,
        target_operator,
        source_binding,
        target_binding,
        source_challenge,
        target_challenge,
        source_commitment,
        target_commitment,
    )
    execution_digest = _digest(
        "counterfactual_execution",
        {
            "candidate_digest": candidate.digest,
            "source": {
                "binding_digest": source_binding.digest,
                "challenge_id": source_challenge.case_id,
                "commitment_digest": source_commitment.digest,
                "operator_digest": source_operator.digest,
                "prediction_digest": source_prediction.digest,
            },
            "target": {
                "binding_digest": target_binding.digest,
                "challenge_id": target_challenge.case_id,
                "commitment_digest": target_commitment.digest,
                "operator_digest": target_operator.digest,
                "prediction_digest": target_prediction.digest,
            },
        },
    )

    # These are the only two execution calls in this evaluator.  Results are
    # retained and inspected; a failure never triggers a hidden retry.
    source_result = evaluate_committed_sequence(
        source_challenge,
        source_commitment,
    )
    target_result = evaluate_committed_sequence(
        target_challenge,
        target_commitment,
    )
    checks = {
        "source_actions_applied": _all_actions_applied(
            source_result,
            source_commitment,
        ),
        "source_delta_matches": _delta_matches(
            source_challenge,
            source_result,
            source_prediction,
        ),
        "source_exact_goal": _exact_goal_succeeded(
            source_challenge,
            source_result,
        ),
        "source_trace_matches": _trace_matches_commitment(
            source_result,
            source_commitment,
        ),
        "target_actions_applied": _all_actions_applied(
            target_result,
            target_commitment,
        ),
        "target_delta_matches": _delta_matches(
            target_challenge,
            target_result,
            target_prediction,
        ),
        "target_exact_goal": _exact_goal_succeeded(
            target_challenge,
            target_result,
        ),
        "target_trace_matches": _trace_matches_commitment(
            target_result,
            target_commitment,
        ),
    }
    outcome = "pass" if all(checks.values()) else "fail"
    result_digest = _digest(
        "counterfactual_result",
        {
            "candidate_digest": candidate.digest,
            "checks": checks,
            "execution_digest": execution_digest,
            "outcome": outcome,
            "source_result_digest": source_result.digest,
            "target_result_digest": target_result.digest,
        },
    )
    return CounterfactualExecutionCertificate(
        candidate_digest=candidate.digest,
        execution_digest=execution_digest,
        result_digest=result_digest,
        result=outcome,
        issued_by=EVALUATOR_ISSUER,
    )


def _validate_and_instantiate(
    candidate: StructuralIsomorphismCandidate,
    source_operator: LearnedOperator,
    target_operator: LearnedOperator,
    source_binding: StateOperatorBinding,
    target_binding: StateOperatorBinding,
    source_challenge: OperatorChallenge,
    target_challenge: OperatorChallenge,
    source_commitment: CommittedActionSequence,
    target_commitment: CommittedActionSequence,
) -> tuple[GroundedOperatorPrediction, GroundedOperatorPrediction]:
    if not isinstance(candidate, StructuralIsomorphismCandidate):
        raise TypeError("candidate must be a StructuralIsomorphismCandidate")
    if not isinstance(source_operator, LearnedOperator) or not isinstance(
        target_operator,
        LearnedOperator,
    ):
        raise TypeError("source and target operators must be LearnedOperator values")
    if not isinstance(source_binding, StateOperatorBinding) or not isinstance(
        target_binding,
        StateOperatorBinding,
    ):
        raise TypeError("source and target bindings must be StateOperatorBinding values")
    if not isinstance(source_challenge, OperatorChallenge) or not isinstance(
        target_challenge,
        OperatorChallenge,
    ):
        raise TypeError("source and target challenges must be OperatorChallenge values")
    if not isinstance(source_commitment, CommittedActionSequence) or not isinstance(
        target_commitment,
        CommittedActionSequence,
    ):
        raise TypeError("source and target commitments must be frozen sequences")

    if candidate.source_operator_digest != source_operator.digest or (
        candidate.target_operator_digest != target_operator.digest
    ):
        raise AliasCertificateEvaluationError(
            "candidate is not bound to the supplied operator revisions"
        )
    recomputed = find_structural_isomorphisms(
        source_operator,
        target_operator,
        maximum_candidates=1024,
    )
    if candidate.digest not in {item.digest for item in recomputed}:
        raise AliasCertificateEvaluationError(
            "candidate structure is not reproduced from the supplied operators"
        )
    for operator, binding, challenge, commitment, label in (
        (
            source_operator,
            source_binding,
            source_challenge,
            source_commitment,
            "source",
        ),
        (
            target_operator,
            target_binding,
            target_challenge,
            target_commitment,
            "target",
        ),
    ):
        if operator.namespace != challenge.origin.namespace or (
            challenge.goal.namespace != operator.namespace
        ):
            raise AliasCertificateEvaluationError(
                f"{label} operator and challenge namespaces differ"
            )
        if challenge.goal.exact is not True:
            raise AliasCertificateEvaluationError(
                f"{label} certificate challenge must have an exact goal"
            )
        if binding.operator_digest != operator.digest or (
            binding.namespace != operator.namespace
        ):
            raise AliasCertificateEvaluationError(
                f"{label} binding belongs to another operator revision"
            )
        if commitment.challenge_id != challenge.case_id:
            raise AliasCertificateEvaluationError(
                f"{label} commitment belongs to another challenge"
            )
        if len(commitment.actions) > challenge.maximum_steps:
            raise AliasCertificateEvaluationError(
                f"{label} commitment exceeds its action ceiling"
            )
        if any(
            action.schema not in set(challenge.allowed_action_schemas)
            for action in commitment.actions
        ):
            raise AliasCertificateEvaluationError(
                f"{label} commitment uses a disallowed action schema"
            )

    source_prediction = instantiate_operator(source_operator, source_binding)
    target_prediction = instantiate_operator(target_operator, target_binding)
    if source_prediction.actions != source_commitment.actions:
        raise AliasCertificateEvaluationError(
            "source commitment does not equal the instantiated mirror actions"
        )
    if target_prediction.actions != target_commitment.actions:
        raise AliasCertificateEvaluationError(
            "target commitment does not equal the instantiated mirror actions"
        )
    return source_prediction, target_prediction


def _trace_matches_commitment(
    result: OperatorCaseResult,
    commitment: CommittedActionSequence,
) -> bool:
    return tuple(item.action for item in result.trace.transitions) == commitment.actions


def _all_actions_applied(
    result: OperatorCaseResult,
    commitment: CommittedActionSequence,
) -> bool:
    return (
        len(result.trace.transitions) == len(commitment.actions)
        and result.applied_actions == len(commitment.actions)
        and all(item.applied for item in result.trace.transitions)
    )


def _exact_goal_succeeded(
    challenge: OperatorChallenge,
    result: OperatorCaseResult,
) -> bool:
    return (
        challenge.goal.exact is True
        and result.success
        and result.final_state.namespace == challenge.goal.namespace
        and result.final_state.records == challenge.goal.required
    )


def _delta_matches(
    challenge: OperatorChallenge,
    result: OperatorCaseResult,
    prediction: GroundedOperatorPrediction,
) -> bool:
    before = set(challenge.origin.records)
    after = set(result.final_state.records)
    observed_additions = tuple(sorted(after - before))
    observed_deletions = tuple(sorted(before - after))
    return (
        observed_additions == prediction.predicted_additions
        and observed_deletions == prediction.predicted_deletions
    )


def _digest(kind: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {
            "evaluator": _EVALUATOR_VERSION,
            "kind": kind,
            "payload": payload,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AliasCertificateEvaluationError",
    "EVALUATOR_ISSUER",
    "certify_counterfactual_alignment",
]
