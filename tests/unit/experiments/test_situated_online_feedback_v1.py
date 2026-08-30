from __future__ import annotations

import unittest

from experiments.runners.situated_online_feedback_v1 import (
    ACTIONS,
    PROBE_COUNT,
    STREAM_COUNT,
    TARGETS,
    build_stream,
    parse_action,
)


class SituatedOnlineFeedbackV1Tests(unittest.TestCase):
    def test_stream_is_unique_recurrent_and_probe_disjoint(self) -> None:
        stream, probes = build_stream()
        self.assertEqual(len(stream), STREAM_COUNT)
        self.assertEqual(len(probes), PROBE_COUNT)
        identities = {sample.identity for sample in stream}
        self.assertTrue(identities.isdisjoint(sample.identity for sample in probes))
        self.assertEqual(
            [sum(sample.family_index == family for sample in stream) for family in range(4)],
            [24, 24, 24, 24],
        )
        self.assertEqual(len({sample.query for sample in stream + probes}), 128)

    def test_targets_are_evaluator_metadata_not_text(self) -> None:
        stream, _ = build_stream()
        self.assertEqual(len(TARGETS), len(ACTIONS))
        for sample in stream:
            target = ACTIONS[sample.target_action]
            self.assertNotIn("target", sample.query.lower())
            # Every action is represented exactly once; no candidate is marked correct.
            joined = "\n".join(item.text for item in sample.recall.items)
            self.assertEqual(sum(target in item.text for item in sample.recall.items), 1)
            self.assertNotIn("success", joined.lower())
            self.assertNotIn("failure", joined.lower())

    def test_action_parser_requires_one_unique_procedure(self) -> None:
        self.assertEqual(parse_action("stage"), "stage")
        self.assertEqual(parse_action("Stage. Stage."), "stage")
        self.assertIsNone(parse_action("stage then trace"))
        self.assertIsNone(parse_action("unknown"))


if __name__ == "__main__":
    unittest.main()
