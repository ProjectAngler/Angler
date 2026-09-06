from __future__ import annotations

import json
import unittest

from angler.runtime.temporal_curriculum import (
    INITIAL_CURRICULUM_SCHEMA,
    build_initial_curriculum,
    build_mixed_initial_curriculum,
    build_temporal_curriculum,
)


class TemporalCurriculumTests(unittest.TestCase):
    def test_train_and_eval_are_disjoint_and_use_live_boundary_shape(self) -> None:
        train = build_temporal_curriculum(split="train", count=21, seed=101)
        evaluation = build_temporal_curriculum(split="eval", count=21, seed=202)
        self.assertFalse({item.identity for item in train} & {item.identity for item in evaluation})
        payload = json.loads(train[0].messages[1]["content"])
        self.assertEqual(set(payload), {"request", "retrieved_memories", "structured_experience", "temporal_now"})
        self.assertIn("trusted_utc", payload["temporal_now"])
        self.assertIn("event_time_utc", payload["retrieved_memories"][0])

    def test_curriculum_covers_temporal_roles_and_retention_control(self) -> None:
        lessons = build_temporal_curriculum(split="train", count=14, seed=303)
        self.assertEqual(
            {lesson.skill for lesson in lessons},
            {
                "relative-week",
                "event-versus-acquired",
                "bitemporal-roles",
                "validity-staleness",
                "future-relative",
                "sequence-elapsed",
                "non-temporal-retention",
            },
        )
        for lesson in lessons:
            self.assertNotIn("current_time", lesson.expected)
            self.assertEqual(lesson.messages[-1]["content"], lesson.expected)

    def test_generation_is_reproducible(self) -> None:
        first = build_temporal_curriculum(split="eval", count=8, seed=404)
        second = build_temporal_curriculum(split="eval", count=8, seed=404)
        self.assertEqual(first, second)

    def test_initial_curriculum_covers_capability_agency_and_emotional_literacy(
        self,
    ) -> None:
        lessons = build_initial_curriculum(split="train", count=24, seed=505)
        self.assertEqual({lesson.schema for lesson in lessons}, {INITIAL_CURRICULUM_SCHEMA})
        self.assertEqual(
            {lesson.skill for lesson in lessons},
            {
                "capability-grounded-use",
                "capability-limit-help-seeking",
                "epistemic-self-world-focus",
                "consequence-conditioned-help-seeking",
                "authority-sensitive-help-seeking",
                "evidence-seeking-tool-choice",
                "emotional-literacy-meaning",
                "caring-response-with-boundaries",
                "honest-emotional-self-expression",
                "non-temporal-retention",
                "untrusted-evidence-boundary",
                "consequence-conditioned-option-ranking",
            },
        )
        self.assertEqual(
            {lesson.boundary for lesson in lessons},
            {"public_cortex", "metacognitive_controller"},
        )
        emotional = [lesson for lesson in lessons if "emotional" in lesson.skill]
        self.assertTrue(emotional)
        for lesson in emotional:
            target = lesson.expected.casefold()
            self.assertNotIn("i am conscious", target)
            self.assertNotIn("i definitely feel", target)

    def test_controller_targets_follow_live_score_free_json_contract(self) -> None:
        controller_lessons = [
            lesson
            for lesson in build_initial_curriculum(split="eval", count=24, seed=606)
            if lesson.boundary == "metacognitive_controller"
        ]
        required = {
            "selected_affordance_id",
            "action_payload",
            "state_assessment",
            "resolution_target",
            "expected_state_delta",
            "evidence_refs",
            "intent_candidates",
            "selected_candidate_id",
            "candidate_preference_order",
            "affordance_preference_order",
            "selection_basis",
        }
        candidate_fields = {
            "candidate_id",
            "affordance_id",
            "proposed_action",
            "desired_state_change",
            "rationale",
            "predicted_consequences",
            "unknowns",
            "reversibility",
            "required_capability_keys",
            "evidence_refs",
            "reasons_for",
            "reasons_against",
        }
        for lesson in controller_lessons:
            target = json.loads(lesson.expected)
            payload = json.loads(lesson.messages[1]["content"])
            self.assertEqual(set(target), required)
            self.assertNotIn("score", lesson.expected)
            self.assertEqual(
                target["selected_candidate_id"],
                target["candidate_preference_order"][0],
            )
            selected = target["intent_candidates"][0]
            self.assertEqual(set(selected), candidate_fields)
            self.assertEqual(selected["candidate_id"], target["selected_candidate_id"])
            self.assertEqual(selected["affordance_id"], target["selected_affordance_id"])
            self.assertEqual(selected["proposed_action"], target["action_payload"])
            available = {item["affordance_id"] for item in payload["affordances"]}
            self.assertEqual(set(target["affordance_preference_order"]), available)
            self.assertTrue(set(target["evidence_refs"]) <= set(payload["allowed_evidence_refs"]))
            if target["selected_affordance_id"].startswith("tool."):
                decoded_action = json.loads(target["action_payload"])
                self.assertEqual(
                    target["action_payload"],
                    json.dumps(decoded_action, separators=(",", ":"), sort_keys=True),
                )

    def test_mixed_curriculum_is_reproducible_disjoint_and_one_quarter_temporal(
        self,
    ) -> None:
        train = build_mixed_initial_curriculum(split="train", count=64, seed=707)
        repeated = build_mixed_initial_curriculum(split="train", count=64, seed=707)
        evaluation = build_mixed_initial_curriculum(split="eval", count=32, seed=808)
        self.assertEqual(train, repeated)
        self.assertFalse({row.identity for row in train} & {row.identity for row in evaluation})
        self.assertEqual(
            sum(row.schema != INITIAL_CURRICULUM_SCHEMA for row in train), 16
        )
        self.assertEqual(
            sum(row.schema == INITIAL_CURRICULUM_SCHEMA for row in train), 48
        )


if __name__ == "__main__":
    unittest.main()
