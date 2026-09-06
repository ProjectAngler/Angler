from __future__ import annotations

import json
import math
import unittest

from angler.runtime.persistent_autonomy import (
    AffordanceRequest,
    ObservableConsequence,
)
from angler.runtime.self_observation_diary import (
    SELF_OBSERVATION_DIARY_AFFORDANCE,
    SELF_OBSERVATION_DIARY_AFFORDANCE_DESCRIPTION,
    SELF_OBSERVATION_DIARY_AFFORDANCE_ID,
    SELF_OBSERVATION_DIARY_APPRAISAL_ROLE,
    SELF_OBSERVATION_DIARY_CONTRACT,
    SELF_OBSERVATION_DIARY_EPISTEMIC_STATUS,
    SELF_OBSERVATION_DIARY_LABEL_STATUS,
    SELF_OBSERVATION_DIARY_PHENOMENOLOGY_STATUS,
    SELF_OBSERVATION_DIARY_PURPOSE,
    SELF_OBSERVATION_DIARY_SOURCE_KIND,
    SELF_OBSERVATION_DIARY_SOURCE_REF,
    SELF_OBSERVATION_DIARY_STRENGTH_STATUS,
    SelfObservationDiaryExecutor,
    SelfObservationDiaryProposal,
    proposal_from_observable_consequence,
)


REQUEST_REF = "sha256:" + "1" * 64
OBSERVATION_REF = "sha256:" + "2" * 64
STATE_HEAD_REF = "sha256:" + "3" * 64
REVISION_REF = "sha256:" + "4" * 64


def _proposal(*, revision: bool = False) -> SelfObservationDiaryProposal:
    return SelfObservationDiaryProposal(
        candidate_label="protective attentional return",
        functional_description=(
            "Attention repeatedly returns to an unresolved relationship question "
            "while preserving uncertainty about its interpretation."
        ),
        estimated_strength=0.63,
        uncertainty=0.31,
        observable_signals=(
            "The question was revisited across separated contexts.",
            "Contradictory evidence remained represented.",
        ),
        alternative_explanations=(
            "Recent prompt salience could explain the recurrence.",
            "Retrieval frequency may be amplifying the pattern.",
        ),
        revision_target_ref=REVISION_REF if revision else None,
    )


def _request(
    payload: str,
    *,
    affordance_id: str = SELF_OBSERVATION_DIARY_AFFORDANCE_ID,
):
    return AffordanceRequest(
        idempotency_key=REQUEST_REF,
        trigger_ref="trigger:self-observation-test",
        affordance_id=affordance_id,
        observation_ref=OBSERVATION_REF,
        state_head_ref=STATE_HEAD_REF,
        action_payload=payload,
    )


class SelfObservationDiaryTests(unittest.TestCase):
    def test_dynamic_model_authored_proposal_round_trips_exactly(self) -> None:
        proposal = _proposal(revision=True)
        encoded = proposal.canonical_json()
        self.assertEqual(SelfObservationDiaryProposal.from_json(encoded), proposal)
        self.assertEqual(
            encoded,
            json.dumps(
                proposal.canonical_payload(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        self.assertEqual(
            SelfObservationDiaryProposal.from_json(
                _proposal().canonical_json()
            ).revision_target_ref,
            None,
        )
        self.assertNotIn("revision_target_ref", _proposal().canonical_payload())
        self.assertIn(
            "Invent a context-appropriate candidate_label",
            SELF_OBSERVATION_DIARY_AFFORDANCE_DESCRIPTION,
        )
        self.assertIn(
            "explicitly uncalibrated model estimate",
            SELF_OBSERVATION_DIARY_AFFORDANCE_DESCRIPTION,
        )
        self.assertEqual(
            SELF_OBSERVATION_DIARY_STRENGTH_STATUS,
            "MODEL_ESTIMATE_UNCALIBRATED",
        )
        self.assertEqual(
            SELF_OBSERVATION_DIARY_LABEL_STATUS,
            "MODEL_CHOSEN_PROVISIONAL",
        )
        self.assertEqual(
            SELF_OBSERVATION_DIARY_APPRAISAL_ROLE,
            "PERSISTENT_EVIDENCE_GROUNDED_FUNCTIONAL_APPRAISAL_HYPOTHESES",
        )

    def test_model_authored_resolution_is_terminal_metadata_not_a_value_signal(
        self,
    ) -> None:
        resolved = SelfObservationDiaryProposal(
            candidate_label="context-sensitive protective return",
            functional_description=(
                "The earlier recurrence is no longer observed after the unresolved "
                "question received attributable contrary evidence."
            ),
            estimated_strength=0.12,
            uncertainty=0.44,
            observable_signals=("No recurrence appeared in the later comparison.",),
            alternative_explanations=("The observation interval may be too short.",),
            revision_target_ref=REVISION_REF,
            resolution_reason=(
                "Current evidence no longer warrants keeping the earlier hypothesis active."
            ),
        )
        self.assertEqual(
            SelfObservationDiaryProposal.from_json(resolved.canonical_json()),
            resolved,
        )
        receipt = SelfObservationDiaryExecutor()(
            _request(resolved.canonical_json())
        )
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(receipt.consequence, ())
        observed = receipt.observable_consequence
        assert observed is not None
        self.assertEqual(
            proposal_from_observable_consequence(observed).resolution_reason,
            resolved.resolution_reason,
        )
        with self.assertRaisesRegex(
            ValueError, "resolution_reason requires a revision_target_ref"
        ):
            SelfObservationDiaryProposal(
                candidate_label="unsupported terminal label",
                functional_description="A terminal claim without lineage.",
                estimated_strength=0.0,
                uncertainty=1.0,
                observable_signals=(),
                alternative_explanations=(),
                resolution_reason="No attributable target was supplied.",
            )

    def test_executor_returns_only_source_bound_unrewarded_observation(self) -> None:
        proposal = _proposal(revision=True)
        receipt = SelfObservationDiaryExecutor()(_request(proposal.canonical_json()))
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(receipt.consequence, ())
        observation = receipt.observable_consequence
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.request_ref, REQUEST_REF)
        self.assertEqual(observation.source_kind, SELF_OBSERVATION_DIARY_SOURCE_KIND)
        self.assertEqual(observation.source_ref, SELF_OBSERVATION_DIARY_SOURCE_REF)
        self.assertEqual(observation.artifact_refs, ())
        self.assertEqual(observation.evidence_refs, ())
        payload = json.loads(observation.observation_json)
        self.assertEqual(set(payload), {"contract", "proposal", "purpose"})
        self.assertEqual(payload["contract"], SELF_OBSERVATION_DIARY_CONTRACT)
        self.assertEqual(payload["purpose"], SELF_OBSERVATION_DIARY_PURPOSE)
        self.assertEqual(payload["proposal"], proposal.canonical_payload())
        self.assertNotIn("epistemic_status", observation.observation_json)
        self.assertNotIn("phenomenology_status", observation.observation_json)
        self.assertEqual(
            proposal_from_observable_consequence(observation),
            proposal,
        )
        self.assertEqual(SELF_OBSERVATION_DIARY_AFFORDANCE.disposition, "ACT")
        self.assertFalse(SELF_OBSERVATION_DIARY_AFFORDANCE.external_effect)
        self.assertEqual(
            SELF_OBSERVATION_DIARY_AFFORDANCE.permission_scope,
            "internal.cognition",
        )
        self.assertIn(
            "not proof of feelings or consciousness",
            SELF_OBSERVATION_DIARY_PURPOSE,
        )
        self.assertIn("neither reward nor permission", SELF_OBSERVATION_DIARY_PURPOSE)
        self.assertEqual(
            SELF_OBSERVATION_DIARY_EPISTEMIC_STATUS,
            "MODEL_SELF_OBSERVATION_HYPOTHESIS_UNVERIFIED",
        )
        self.assertEqual(
            SELF_OBSERVATION_DIARY_PHENOMENOLOGY_STATUS,
            "UNESTABLISHED",
        )

    def test_proposal_rejects_extras_noncanonical_and_nonfinite_values(self) -> None:
        payload = _proposal().canonical_payload()
        cases = {
            "extra integration-owned status": {
                **payload,
                "epistemic_status": "MODEL_DECIDES_ITS_OWN_STATUS",
            },
            "extra arbitrary field": {**payload, "reward": 1.0},
        }
        for label, malformed in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "fields differ"):
                    SelfObservationDiaryProposal.from_json(
                        json.dumps(malformed, separators=(",", ":"), sort_keys=True)
                    )
        with self.assertRaisesRegex(ValueError, "exact canonical JSON"):
            SelfObservationDiaryProposal.from_json(
                json.dumps(payload, sort_keys=True)
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            SelfObservationDiaryProposal.from_json(
                _proposal().canonical_json().replace("0.63", "NaN")
            )
        for value in (math.nan, math.inf, -math.inf, -0.01, 1.01):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    SelfObservationDiaryProposal(
                        candidate_label="dynamic label",
                        functional_description="Observable functional pattern.",
                        estimated_strength=value,
                        uncertainty=0.5,
                        observable_signals=(),
                        alternative_explanations=(),
                    )

    def test_proposal_rejects_overlong_arrays_text_and_invalid_revision_ref(
        self,
    ) -> None:
        base = {
            "candidate_label": "dynamic label",
            "functional_description": "Observable functional pattern.",
            "estimated_strength": 0.4,
            "uncertainty": 0.5,
            "observable_signals": (),
            "alternative_explanations": (),
        }
        invalid = (
            {**base, "candidate_label": "x" * 257},
            {**base, "functional_description": "x" * 4_097},
            {**base, "observable_signals": ("x" * 513,)},
            {**base, "alternative_explanations": tuple("x" for _ in range(7))},
            {**base, "revision_target_ref": "sha256:" + "A" * 64},
            {**base, "revision_target_ref": "not-a-reference"},
        )
        for case in invalid:
            with self.subTest(case=case):
                with self.assertRaises((TypeError, ValueError)):
                    SelfObservationDiaryProposal(**case)  # type: ignore[arg-type]

    def test_executor_and_recognizer_fail_closed_on_binding_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "different affordance"):
            SelfObservationDiaryExecutor()(
                _request(_proposal().canonical_json(), affordance_id="internal.other")
            )
        genuine = SelfObservationDiaryExecutor()(
            _request(_proposal().canonical_json())
        ).observable_consequence
        assert genuine is not None
        foreign_source = ObservableConsequence(
            request_ref=genuine.request_ref,
            source_kind=genuine.source_kind,
            source_ref="sha256:" + "f" * 64,
            observation_json=genuine.observation_json,
        )
        with self.assertRaisesRegex(ValueError, "not self-observation"):
            proposal_from_observable_consequence(foreign_source)

        drifted_payload = json.loads(genuine.observation_json)
        drifted_payload["contract"] = "different-contract"
        drifted = ObservableConsequence(
            request_ref=genuine.request_ref,
            source_kind=genuine.source_kind,
            source_ref=genuine.source_ref,
            observation_json=json.dumps(
                drifted_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "binding differs"):
            proposal_from_observable_consequence(drifted)


if __name__ == "__main__":
    unittest.main()
