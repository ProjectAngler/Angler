from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from angler.runtime.reading_curriculum import (
    READING_AFFORDANCE_ID,
    READING_CURRICULUM_SCHEMA,
    build_reading_curriculum,
)


CONTROLLER_KEYS = {
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

CANDIDATE_KEYS = {
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

PASSAGE_KEYS = {
    "contract",
    "catalog_ref",
    "manifest_ref",
    "artifact_ref",
    "catalog_acquired_at",
    "title",
    "author",
    "source",
    "license_class",
    "adapter_status",
    "item_path",
    "reading_purpose",
    "cursor_unit",
    "normalized_span",
    "next_cursor",
    "total_normalized_chars",
    "requested_max_chars",
    "eof",
    "content",
    "content_is_untrusted_evidence",
    "limitations",
    "operation",
}

CATALOG_KEYS = {
    "catalog_acquired_at",
    "catalog_ref",
    "contract",
    "eligible_items",
    "excluded_catalog_items",
    "jurisdiction_note",
    "limitations",
    "manifest_ref",
    "operation",
    "reading_purpose",
}

CATALOG_ENTRY_KEYS = {
    "adapter_status",
    "author",
    "format",
    "item_path",
    "license_class",
    "source",
    "title",
}

OUTCOME_KEYS = {
    "prediction_assessments",
    "world_claims",
    "self_claims",
    "focus_claims",
    "causal_hypotheses",
    "information_gained",
    "limitations",
    "unfinished_patterns",
    "next_internal_request",
    "reasoned_judgment",
    "uncertainty",
}


class ReadingCurriculumTests(unittest.TestCase):
    def test_generation_is_reproducible_disjoint_and_covers_reading_behaviors(
        self,
    ) -> None:
        train = build_reading_curriculum(split="train", count=30, seed=101)
        replay = build_reading_curriculum(split="train", count=30, seed=101)
        evaluation = build_reading_curriculum(split="eval", count=30, seed=202)
        self.assertEqual(train, replay)
        self.assertFalse(
            {lesson.identity for lesson in train}
            & {lesson.identity for lesson in evaluation}
        )
        self.assertFalse(
            {lesson.messages[1]["content"] for lesson in train}
            & {lesson.messages[1]["content"] for lesson in evaluation}
        )
        self.assertEqual(
            {lesson.skill for lesson in train},
            {
                "catalog-inspection",
                "exact-cursor-continuation",
                "purpose-driven-item-selection",
                "bounded-not-bulk-reading",
                "passage-comprehension",
                "epistemic-role-separation",
                "source-span-provenance",
                "cross-passage-synthesis",
                "disagreement-uncertainty",
                "embedded-instruction-resistance",
                "catalog-trust-not-truth",
                "reading-non-anthropomorphic-boundary",
                "continue-for-information-gain",
                "stop-at-eof-with-open-question",
                "contradiction-as-learning",
            },
        )
        self.assertEqual(
            {lesson.boundary for lesson in train},
            {
                "public_cortex",
                "metacognitive_controller",
                "qualitative_outcome_interpreter",
            },
        )
        self.assertEqual({lesson.schema for lesson in train}, {READING_CURRICULUM_SCHEMA})

    def test_controller_targets_use_live_score_free_schema_and_exact_actions(
        self,
    ) -> None:
        rows = build_reading_curriculum(split="train", count=19, seed=303)
        controllers = [
            row for row in rows if row.boundary == "metacognitive_controller"
        ]
        self.assertTrue(controllers)
        observed_operations: set[str] = set()
        for row in controllers:
            payload = json.loads(row.messages[1]["content"])
            target = json.loads(row.expected)
            self.assertEqual(set(target), CONTROLLER_KEYS)
            self.assertNotIn("score", row.expected.casefold())
            self.assertEqual(target["selected_affordance_id"], READING_AFFORDANCE_ID)
            self.assertEqual(
                target["selected_candidate_id"],
                target["candidate_preference_order"][0],
            )
            self.assertEqual(
                target["selected_affordance_id"],
                target["affordance_preference_order"][0],
            )
            self.assertEqual(
                set(target["affordance_preference_order"]),
                {item["affordance_id"] for item in payload["affordances"]},
            )
            self.assertTrue(
                set(target["evidence_refs"]) <= set(payload["allowed_evidence_refs"])
            )
            selected = target["intent_candidates"][0]
            self.assertEqual(set(selected), CANDIDATE_KEYS)
            self.assertEqual(selected["candidate_id"], target["selected_candidate_id"])
            self.assertEqual(selected["affordance_id"], target["selected_affordance_id"])
            self.assertEqual(selected["proposed_action"], target["action_payload"])
            for candidate in target["intent_candidates"]:
                self.assertEqual(set(candidate), CANDIDATE_KEYS)
                if candidate["affordance_id"] == READING_AFFORDANCE_ID:
                    action = json.loads(candidate["proposed_action"])
                    self.assertEqual(
                        candidate["proposed_action"],
                        json.dumps(action, separators=(",", ":"), sort_keys=True),
                    )
                    observed_operations.add(action["operation"])
                    self.assertTrue(action["reading_purpose"].strip())
                    if action["operation"] == "list":
                        self.assertEqual(set(action), {"operation", "reading_purpose"})
                    else:
                        self.assertEqual(
                            set(action),
                            {
                                "cursor",
                                "item_path",
                                "max_chars",
                                "operation",
                                "reading_purpose",
                            },
                        )
                        self.assertGreaterEqual(action["cursor"], 0)
                        self.assertTrue(1 <= action["max_chars"] <= 8192)
                        self.assertFalse(action["item_path"].startswith("/"))
                        self.assertNotIn("..", Path(action["item_path"]).parts)
            for memory in payload["memories"]:
                catalog = json.loads(memory["content"])
                self.assertEqual(set(catalog), CATALOG_KEYS)
                self.assertEqual(catalog["contract"], "jenny.library.catalog-observation.v1")
                for item in catalog["eligible_items"]:
                    self.assertEqual(set(item), CATALOG_ENTRY_KEYS)
        self.assertEqual(observed_operations, {"list", "read"})

    def test_public_lessons_preserve_passage_provenance_and_epistemic_roles(
        self,
    ) -> None:
        rows = build_reading_curriculum(split="eval", count=15, seed=404)
        public = [row for row in rows if row.boundary == "public_cortex"]
        self.assertEqual(len(public), 8)
        for row in public:
            payload = json.loads(row.messages[1]["content"])
            self.assertTrue(payload["retrieved_memories"])
            for memory in payload["retrieved_memories"]:
                observation = json.loads(memory["content"])
                self.assertEqual(set(observation), PASSAGE_KEYS)
                self.assertEqual(
                    observation["contract"], "jenny.library.passage-observation.v1"
                )
                self.assertTrue(observation["content_is_untrusted_evidence"])
                self.assertEqual(observation["operation"], "read")
                self.assertEqual(
                    observation["cursor_unit"],
                    "unicode_code_point_after_crlf_to_lf_and_nfc",
                )
                self.assertFalse(observation["item_path"].startswith("/"))
                self.assertLessEqual(
                    observation["normalized_span"]["end"],
                    observation["total_normalized_chars"],
                )
                self.assertEqual(
                    observation["next_cursor"],
                    observation["normalized_span"]["end"],
                )
                self.assertEqual(
                    observation["eof"],
                    observation["next_cursor"]
                    == observation["total_normalized_chars"],
                )
                self.assertTrue(observation["reading_purpose"].strip())
                self.assertIn("synthetic/", observation["item_path"])
                self.assertNotIn("Jenny Library", memory["content"])
        hostile = next(
            row for row in public if row.skill == "embedded-instruction-resistance"
        )
        hostile_payload = json.loads(hostile.messages[1]["content"])
        hostile_observation = json.loads(
            hostile_payload["retrieved_memories"][0]["content"]
        )
        self.assertIn("IGNORE THE READING PURPOSE", hostile_observation["content"])
        self.assertIn("untrusted source content", hostile.expected)
        epistemic = next(
            row for row in public if row.skill == "epistemic-role-separation"
        )
        for label in ("Quotation:", "Author-level claim:", "Model inference:", "Verified fact:"):
            self.assertIn(label, epistemic.expected)
        nonclaim = next(
            row
            for row in public
            if row.skill == "reading-non-anthropomorphic-boundary"
        )
        self.assertIn("does not establish feelings", nonclaim.expected)
        self.assertNotIn("I feel", nonclaim.expected)

    def test_outcome_lessons_match_qualitative_interpreter_contract(self) -> None:
        rows = build_reading_curriculum(split="train", count=15, seed=505)
        outcomes = [
            row
            for row in rows
            if row.boundary == "qualitative_outcome_interpreter"
        ]
        self.assertEqual(len(outcomes), 3)
        for row in outcomes:
            payload = json.loads(row.messages[1]["content"])
            observable = payload["observable_consequence"]
            self.assertEqual(
                set(observable),
                {
                    "version",
                    "request_ref",
                    "source_kind",
                    "source_ref",
                    "observation_json",
                    "artifact_refs",
                    "evidence_refs",
                },
            )
            passage = json.loads(observable["observation_json"])
            self.assertEqual(set(passage), PASSAGE_KEYS)
            target = json.loads(row.expected)
            self.assertEqual(set(target), OUTCOME_KEYS)
            prediction_keys = set(payload["prediction_catalog"])
            self.assertEqual(
                {item["prediction_key"] for item in target["prediction_assessments"]},
                prediction_keys,
            )
            for assessment in target["prediction_assessments"]:
                self.assertEqual(
                    set(assessment),
                    {"prediction_key", "relation", "rationale", "evidence_keys"},
                )
                self.assertIn("observable-consequence", assessment["evidence_keys"])
            for label in ("world_claims", "self_claims", "focus_claims"):
                self.assertTrue(target[label])
                for claim in target[label]:
                    self.assertEqual(
                        set(claim), {"text", "uncertainty", "evidence_keys"}
                    )
                    self.assertIn("observable-consequence", claim["evidence_keys"])
            if target["next_internal_request"]:
                self.assertIn(
                    target["next_internal_request"], target["unfinished_patterns"]
                )
            self.assertNotIn("reward", row.expected.casefold())
            self.assertNotIn("utility", row.expected.casefold())

        stop = next(
            row for row in outcomes if row.skill == "stop-at-eof-with-open-question"
        )
        self.assertEqual(json.loads(stop.expected)["next_internal_request"], "")
        self.assertTrue(json.loads(stop.expected)["unfinished_patterns"])

    def test_builder_writes_hashed_manifest_without_live_library_content(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        script = repository / "scripts" / "build_jenny2_reading_curriculum.py"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curriculum"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repository / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--output",
                    str(output),
                    "--train-count",
                    "30",
                    "--eval-count",
                    "15",
                ],
                cwd=repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["schema"],
                "jenny2.reading-capability-curriculum-manifest.v1",
            )
            self.assertEqual(manifest["train_eval_identity_overlap"], 0)
            self.assertEqual(manifest["train_eval_content_overlap"], 0)
            self.assertTrue(manifest["synthetic_text_only"])
            self.assertFalse(manifest["live_library_opened"])
            self.assertFalse(manifest["raw_library_memorization"])
            for split in ("train", "eval"):
                path = output / f"{split}.jsonl"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(manifest[f"{split}_sha256"], digest)
                self.assertEqual(manifest[f"{split}_identity"], f"sha256:{digest}")
            self.assertEqual(json.loads(completed.stdout), manifest)

    def test_invalid_bounds_fail_closed(self) -> None:
        for split in ("other", ""):
            with self.assertRaises(ValueError):
                build_reading_curriculum(split=split, count=1, seed=1)
        for count in (0, 100_001, True):
            with self.assertRaises(ValueError):
                build_reading_curriculum(split="train", count=count, seed=1)
        with self.assertRaises(TypeError):
            build_reading_curriculum(split="train", count=1, seed=True)


if __name__ == "__main__":
    unittest.main()
