from __future__ import annotations

import unittest

from experiments.runners.situated_memory_cognee_evaluate_v2 import _recall, _target_index


class LiveCogneeEvaluationTests(unittest.TestCase):
    def test_json_recall_is_reconstituted_without_losing_temporal_coordinates(self) -> None:
        row = {
            "items": [
                {
                    "artifact_ref": "sha256:event",
                    "text": "evidence",
                    "source_ref": "sha256:source",
                    "context": [["family", "test"]],
                    "visibility": "LEARNER_VISIBLE",
                    "acquired_ordinal": 2,
                    "age": 4,
                    "landmark_relations": [["current_regime", "AT"]],
                    "world_valid_from": None,
                    "world_valid_until": None,
                    "world_valid_at_query": None,
                    "backend_score": None,
                    "backend_ref": "chunk-1",
                }
            ],
            "rejected": [],
        }

        recall = _recall(row)

        self.assertEqual(recall.items[0].context, (("family", "test"),))
        self.assertEqual(
            recall.items[0].landmark_relations,
            (("current_regime", "AT"),),
        )
        self.assertEqual(recall.items[0].age, 4)

    def test_live_integer_action_indices_are_preserved(self) -> None:
        self.assertEqual(_target_index(2), 2)
        self.assertEqual(_target_index("jade"), 2)
        with self.assertRaises(RuntimeError):
            _target_index(4)


if __name__ == "__main__":
    unittest.main()
