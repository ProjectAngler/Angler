import hashlib
import json
import unittest

from angler.runtime.procedural_experience import (
    MAX_PROCEDURAL_MEMORY_CHARACTERS,
    PROCEDURAL_EXPERIENCE_CONTRACT,
    PROCEDURAL_OUTCOME_PROFILE_STATE_KEY,
    procedural_case_content,
    procedural_case_from_episode,
    procedural_projection_metadata,
    record_procedural_outcome,
)


def _ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def _episode(
    *,
    status="COMPLETED",
    consequence=None,
    observed=True,
    request="Find and verify the current fact.",
    observation="Owner feedback text.",
    late_feedback=None,
    capabilities=(),
):
    modules = [
        {
            "capability_key": key,
            "active_capability_ref": reference,
            "revision_index": revision,
        }
        for key, reference, revision in capabilities
    ]
    context = {
        "request": request,
        "intent_proposal": {
            "resolution_target": "Return a grounded current answer.",
            "predicted_consequence": "The requested fact will be verified.",
            "selected_capability_keys": [item[0] for item in capabilities],
        },
        "cognitive_state": {"capability_modules": modules},
    }
    if late_feedback is not None:
        context["late_feedback"] = late_feedback
    receipt = {
        "status": status,
        "output": "The observed operation completed with this result.",
        "consequence": [] if consequence is None else consequence,
    }
    if observed:
        receipt["observable_consequence"] = {
            "version": "jenny.observable-consequence.v1",
            "request_ref": _ref("request"),
            "source_kind": "TOOL",
            "source_ref": _ref("source"),
            "observation_json": json.dumps(
                {"fact": "verified", "source": "fixture"},
                separators=(",", ":"),
                sort_keys=True,
            ),
            "artifact_refs": [],
            "evidence_refs": [_ref("evidence")],
        }
    return {
        "event_ref": _ref("event"),
        "parent_state_ref": _ref("parent"),
        "child_state_ref": _ref("child"),
        "observation": {"source": "HUMAN", "content": observation},
        "choice": {
            "selected_affordance_id": "tool.web.read",
            "rankings": [],
            "prediction": "The requested fact will be verified.",
            "uncertainty": 0.2,
            "wake_after_seconds": None,
            "action_payload": '{"url":"https://example.test"}',
            "context_json": json.dumps(
                context, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ),
        },
        "receipt": receipt,
        "temporal": {
            "moving_origin_ordinal": 17,
            "event_time_utc": "2026-09-04T10:00:00Z",
            "acquired_time_utc": "2026-09-04T10:00:01Z",
            "recorded_time_utc": "2026-09-04T10:00:02Z",
            "verified_time_utc": "2026-09-04T10:00:03Z",
            "valid_from_utc": None,
            "valid_until_utc": None,
            "timezone": "America/New_York",
        },
    }


class ProceduralExperienceTests(unittest.TestCase):
    def test_observed_episode_becomes_grounded_reusable_case(self):
        capability_ref = _ref("capability")
        payload = _episode(
            consequence=[["objective_progress", 1.0], ["cost", -0.2]],
            capabilities=(("verify.current.fact", capability_ref, 3),),
        )

        case = procedural_case_from_episode(payload)

        self.assertIsNotNone(case)
        assert case is not None
        self.assertEqual(case["contract"], PROCEDURAL_EXPERIENCE_CONTRACT)
        self.assertEqual(case["condition"], "Find and verify the current fact.")
        self.assertEqual(case["temporal"]["moving_origin_ordinal"], 17)
        self.assertEqual(
            case["procedure"]["selected_capabilities"][0]["capability_ref"],
            capability_ref,
        )
        self.assertEqual(
            case["outcome"]["consequence_vector"]["objective_progress"], 1.0
        )
        self.assertEqual(case, procedural_case_from_episode(payload))
        self.assertLessEqual(
            len(procedural_case_content(payload)), MAX_PROCEDURAL_MEMORY_CHARACTERS
        )

    def test_unevaluated_output_never_becomes_a_procedure(self):
        payload = _episode(status="COMPLETED_UNEVALUATED", observed=False)
        self.assertIsNone(procedural_case_from_episode(payload))
        self.assertIsNone(procedural_case_content(payload))

    def test_legacy_completion_without_outcome_is_not_promoted(self):
        payload = {
            "receipt": {
                "status": "COMPLETED",
                "output": "unverified",
                "consequence": [],
            }
        }
        self.assertIsNone(procedural_case_from_episode(payload))

    def test_late_feedback_binds_original_request_not_feedback_prompt(self):
        feedback = {
            "feedback_source_ref": _ref("feedback"),
            "feedback_text": "Check the initiating request before explaining why.",
            "target_episode_ref": _ref("episode"),
            "target_choice_ref": _ref("target-choice"),
            "target_receipt_ref": _ref("target-receipt"),
        }
        payload = _episode(
            observed=False,
            observation=feedback["feedback_text"],
            request="What made you decide to search?",
            consequence=[["human_feedback", -1.0]],
            late_feedback=feedback,
        )

        case = procedural_case_from_episode(payload)

        assert case is not None
        self.assertEqual(case["condition"], "What made you decide to search?")
        self.assertEqual(case["feedback"]["feedback_text"], feedback["feedback_text"])
        self.assertEqual(case["feedback"]["target_episode_ref"], _ref("episode"))

    def test_operational_failure_is_retrievable_evidence(self):
        payload = _episode(status="ERROR", observed=False)
        case = procedural_case_from_episode(payload)
        assert case is not None
        self.assertEqual(case["epistemic_status"], "OBSERVED_PROCEDURAL_FAILURE")
        self.assertEqual(case["outcome"]["receipt_status"], "ERROR")

    def test_projection_metadata_preserves_type_and_feedback_lineage(self):
        target_episode_ref = _ref("episode")
        payload = _episode(
            observed=False,
            consequence=[["human_feedback", -1.0]],
            late_feedback={
                "feedback_source_ref": _ref("feedback"),
                "feedback_text": "Find the initiating request before explaining why.",
                "target_episode_ref": target_episode_ref,
                "target_choice_ref": _ref("target-choice"),
                "target_receipt_ref": _ref("target-receipt"),
            },
        )
        content = procedural_case_content(payload)
        assert content is not None

        metadata = procedural_projection_metadata(content)

        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.memory_kind, "PROCEDURAL")
        self.assertEqual(
            metadata.epistemic_status, "OBSERVED_PROCEDURAL_OUTCOME"
        )
        self.assertEqual(metadata.adjacent_episode_refs, (target_episode_ref,))
        self.assertIsNone(procedural_projection_metadata("ordinary episodic text"))

        tampered = json.loads(content)
        tampered["condition"] = "A different unbound condition."
        with self.assertRaisesRegex(ValueError, "case_ref differs"):
            procedural_projection_metadata(
                json.dumps(tampered, separators=(",", ":"), sort_keys=True)
            )

    def test_online_vector_moments_accumulate_without_scalarization(self):
        state = {}
        choice_ref = _ref("choice")
        first_receipt = _ref("receipt-one")
        second_receipt = _ref("receipt-two")
        procedure_ids = ["affordance:cortex.respond", "capability:" + _ref("cap")]
        record_procedural_outcome(
            state,
            procedure_ids=procedure_ids,
            consequence={"progress": 1.0, "conflict": -1.0},
            choice_ref=choice_ref,
            receipt_ref=first_receipt,
            moving_origin_ordinal=4,
        )
        record_procedural_outcome(
            state,
            procedure_ids=procedure_ids,
            consequence={"progress": -1.0, "conflict": 1.0},
            choice_ref=choice_ref,
            receipt_ref=second_receipt,
            moving_origin_ordinal=5,
        )

        profile = state[PROCEDURAL_OUTCOME_PROFILE_STATE_KEY][procedure_ids[1]]
        self.assertEqual(profile["count"], 2)
        self.assertEqual(profile["mean"], {"progress": 0.0, "conflict": 0.0})
        self.assertEqual(profile["m2"], {"progress": 2.0, "conflict": 2.0})
        self.assertEqual(
            profile["recent_evidence_refs"], [first_receipt, second_receipt]
        )
        self.assertNotIn("utility", profile)
        self.assertNotIn("emotion", profile)


if __name__ == "__main__":
    unittest.main()
