from __future__ import annotations

import unittest

from experiments.evaluators.software_pipeline_reconstruction_suite import (
    make_software_pipeline_stream,
)
from experiments.runners.scaled_procedural_software_v1 import (
    DEVELOPMENT_MECHANISMS,
    FINAL_MECHANISMS,
    IDENTITY,
    META_EPISODES,
    SLOW_UPDATES,
    TRAIN_MECHANISMS,
    build_prompt,
    parse_response,
    serialize_public_task,
    support_evidence,
    visible_procedure,
)


class ScaledProceduralSoftwareV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = make_software_pipeline_stream(9137, mechanism_partition="train")

    def test_frozen_scale_and_identity(self) -> None:
        self.assertEqual(IDENTITY, "angler.scaled-procedural-software-reconstruction.v1")
        self.assertEqual(TRAIN_MECHANISMS, 64)
        self.assertEqual(DEVELOPMENT_MECHANISMS, 16)
        self.assertEqual(FINAL_MECHANISMS, 16)
        self.assertEqual(SLOW_UPDATES, 256)
        self.assertEqual(META_EPISODES, 32)

    def test_serialization_preserves_structure_without_opaque_namespace(self) -> None:
        task = self.stream.supports[0].learner
        text = serialize_public_task(task)
        self.assertIn("components:", text)
        self.assertIn("graph=", text)
        self.assertIn("origin=", text)
        self.assertNotIn(task.origin.namespace, text)
        for action in task.grounded_candidates:
            self.assertNotIn(action.schema.name, text)

    def test_visible_target_comes_only_from_public_observation(self) -> None:
        task = self.stream.supports[0].learner
        target = visible_procedure(task)
        self.assertTrue(target.endswith("STOP"))
        evidence = support_evidence(task)
        self.assertIn(f"successful procedure={target}", evidence)

    def test_parser_commits_declared_labels_without_repair(self) -> None:
        task = self.stream.queries[0].learner
        pipeline = parse_response(task, "A C STOP")
        self.assertEqual(pipeline.actions, (task.grounded_candidates[0], task.grounded_candidates[2]))
        self.assertTrue(pipeline.stopped)
        invalid = parse_response(task, "Z")
        self.assertEqual(invalid.actions, ())
        self.assertTrue(invalid.stopped)

    def test_prompt_keeps_evidence_explicit(self) -> None:
        prompt = build_prompt("task", ("first", "second"))
        self.assertIn("task", prompt)
        self.assertIn("experience 1:\nfirst", prompt)
        self.assertIn("experience 2:\nsecond", prompt)


if __name__ == "__main__":
    unittest.main()
