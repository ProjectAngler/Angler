from __future__ import annotations

import hashlib
import json
import unittest

from angler.runtime.grounded_appraisal import (
    GROUNDED_APPRAISAL_CONTRACT,
    GROUNDED_APPRAISAL_STATE_KEY,
    MAX_APPRAISAL_CONTEXT_CHARACTERS,
    event_appraisal_signals,
    grounded_appraisal_context,
    update_grounded_appraisal,
)


def _ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def _event(
    *, uncertainty: float, information_gain: float | None, ordinal: int
) -> tuple[dict[str, object], ...]:
    choice = {
        "selected_affordance_id": "cortex.respond",
        "uncertainty": uncertainty,
        "rankings": [
            {"affordance_id": "cortex.respond", "score": 1.0 - uncertainty},
            {"affordance_id": "control.wait", "score": uncertainty},
        ],
    }
    consequence = (
        []
        if information_gain is None
        else [["information_gain", information_gain]]
    )
    receipt = {
        "status": (
            "COMPLETED_UNEVALUATED"
            if information_gain is None
            else "COMPLETED"
        ),
        "consequence": consequence,
        "observable_consequence": (
            None if information_gain is None else {"source_kind": "TEST"}
        ),
    }
    context = {
        "intent_proposal": {
            "uncertainty": uncertainty,
            "alternatives": [{"candidate_id": "wait"}],
        },
        "memories": [
            {
                "record_ref": _ref(f"memory-{ordinal}"),
                "semantic_distance": uncertainty,
            }
        ],
    }
    temporal = {
        "moving_origin_ordinal": ordinal,
        "uncertainty_ms": 1.0,
    }
    return choice, receipt, context, temporal


class GroundedAppraisalTests(unittest.TestCase):
    def test_projects_only_observed_numeric_signals(self) -> None:
        choice, receipt, context, temporal = _event(
            uncertainty=0.25, information_gain=0.5, ordinal=1
        )
        signals = event_appraisal_signals(
            choice=choice, receipt=receipt, context=context, temporal=temporal
        )
        self.assertEqual(signals["choice.uncertainty"], 0.25)
        self.assertEqual(signals["retrieval.mean_distance"], 0.25)
        self.assertEqual(signals["consequence.information_gain"], 0.5)
        self.assertNotIn("emotion", signals)
        self.assertNotIn("utility", signals)
        self.assertNotIn("valence", signals)

    def test_unevaluated_turn_updates_dynamics_without_outcome_credit(self) -> None:
        choice, receipt, context, temporal = _event(
            uncertainty=0.4, information_gain=None, ordinal=2
        )
        state: dict[str, object] = {}
        appraisal = update_grounded_appraisal(
            state,
            choice=choice,
            receipt=receipt,
            context=context,
            temporal=temporal,
            observation_source="HUMAN",
            choice_ref=_ref("choice-2"),
            receipt_ref=_ref("receipt-2"),
        )
        latest = appraisal["latest"]
        self.assertEqual(latest["metadata"]["receipt_status"], "COMPLETED_UNEVALUATED")
        self.assertFalse(latest["metadata"]["observable_consequence_present"])
        self.assertFalse(
            any(name.startswith("consequence.") for name in latest["signal_values"])
        )

    def test_online_moments_and_relations_are_derived_from_history(self) -> None:
        state: dict[str, object] = {}
        for ordinal, (uncertainty, gain) in enumerate(
            ((0.0, -1.0), (0.5, 0.0), (1.0, 1.0)), start=1
        ):
            choice, receipt, context, temporal = _event(
                uncertainty=uncertainty,
                information_gain=gain,
                ordinal=ordinal,
            )
            choice["rankings"] = [
                {"affordance_id": "cortex.respond", "score": 0.5},
                {"affordance_id": "control.wait", "score": 0.5},
            ]
            context["intent_proposal"]["uncertainty"] = 0.2
            context["memories"][0]["semantic_distance"] = 0.3
            update_grounded_appraisal(
                state,
                choice=choice,
                receipt=receipt,
                context=context,
                temporal=temporal,
                observation_source="AGENT",
                choice_ref=_ref(f"choice-{ordinal}"),
                receipt_ref=_ref(f"receipt-{ordinal}"),
            )

        appraisal = state[GROUNDED_APPRAISAL_STATE_KEY]
        moment = appraisal["feature_moments"]["choice.uncertainty"]
        self.assertEqual(moment["count"], 3)
        self.assertAlmostEqual(moment["mean"], 0.5)
        self.assertAlmostEqual(moment["m2"], 0.5)
        context_view = grounded_appraisal_context(state)
        assert context_view is not None
        relation = next(
            item
            for item in context_view["strongest_empirical_relations"]
            if {
                item["left_signal"], item["right_signal"]
            }
            == {"choice.uncertainty", "consequence.information_gain"}
        )
        self.assertAlmostEqual(relation["correlation"], 1.0)
        self.assertEqual(relation["coobservations"], 3)

    def test_latest_innovation_uses_only_prior_observations(self) -> None:
        state: dict[str, object] = {}
        for ordinal, uncertainty in enumerate((0.0, 1.0, 2.0), start=1):
            choice, receipt, context, temporal = _event(
                uncertainty=min(uncertainty, 1.0),
                information_gain=0.0,
                ordinal=ordinal,
            )
            if ordinal == 3:
                choice["uncertainty"] = 1.0
                context["intent_proposal"]["uncertainty"] = 1.0
            update_grounded_appraisal(
                state,
                choice=choice,
                receipt=receipt,
                context=context,
                temporal=temporal,
                observation_source="SCHEDULER",
                choice_ref=_ref(f"innovation-choice-{ordinal}"),
                receipt_ref=_ref(f"innovation-receipt-{ordinal}"),
            )
        appraisal = state[GROUNDED_APPRAISAL_STATE_KEY]
        innovation = appraisal["latest"]["innovations"]["choice.uncertainty"]
        self.assertEqual(innovation["basis_count"], 2)
        self.assertAlmostEqual(innovation["prior_mean"], 0.5)
        self.assertAlmostEqual(innovation["standardized_innovation"], 2 ** -0.5)

    def test_context_is_bounded_and_keeps_provenance(self) -> None:
        state: dict[str, object] = {}
        choice, receipt, context, temporal = _event(
            uncertainty=0.2, information_gain=0.4, ordinal=7
        )
        update_grounded_appraisal(
            state,
            choice=choice,
            receipt=receipt,
            context=context,
            temporal=temporal,
            observation_source="CONTINUATION",
            choice_ref=_ref("bounded-choice"),
            receipt_ref=_ref("bounded-receipt"),
        )
        view = grounded_appraisal_context(state)
        assert view is not None
        self.assertEqual(view["contract"], GROUNDED_APPRAISAL_CONTRACT)
        self.assertEqual(
            state[GROUNDED_APPRAISAL_STATE_KEY]["latest"]["metadata"]["choice_ref"],
            _ref("bounded-choice"),
        )
        self.assertTrue(
            view["latest"]["metadata"]["event_signal_ref"].startswith("sha256:")
        )
        self.assertLessEqual(
            len(json.dumps(view, separators=(",", ":"), sort_keys=True)),
            MAX_APPRAISAL_CONTEXT_CHARACTERS,
        )

    def test_existing_state_contract_mismatch_fails_closed(self) -> None:
        state = {GROUNDED_APPRAISAL_STATE_KEY: {"contract": "wrong"}}
        choice, receipt, context, temporal = _event(
            uncertainty=0.2, information_gain=0.4, ordinal=7
        )
        with self.assertRaisesRegex(ValueError, "identity differs"):
            update_grounded_appraisal(
                state,
                choice=choice,
                receipt=receipt,
                context=context,
                temporal=temporal,
                observation_source="HUMAN",
                choice_ref=_ref("bad-choice"),
                receipt_ref=_ref("bad-receipt"),
            )


if __name__ == "__main__":
    unittest.main()
