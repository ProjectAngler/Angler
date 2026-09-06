from __future__ import annotations

import unittest

from experiments.runners.compositional_procedure_v4 import (
    EPOCHS,
    IDENTITY,
    _public_training_streams,
    supervised_composed_target,
)
from experiments.runners.scaled_procedural_software_v1 import prepare_corpus


class CompositionalProcedureV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.train, cls.development, _ = prepare_corpus()

    def test_identity_and_fixed_passes(self) -> None:
        self.assertEqual(IDENTITY, "angler.compositional-procedure.v4")
        self.assertEqual(EPOCHS, 8)

    def test_train_partition_exposes_four_action_compositions(self) -> None:
        streams = _public_training_streams(self.train[:1])
        self.assertEqual(len(streams[0].queries), 2)
        for example in streams[0].queries:
            tokens = example.target_text.split()
            self.assertEqual(len(tokens), 4)
            self.assertNotIn("STOP", tokens)

    def test_composed_target_rejects_non_training_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "train partition"):
            supervised_composed_target(self.development[0].queries[0])

    def test_composed_labels_map_to_distinct_public_candidates(self) -> None:
        example = self.train[0].queries[0]
        labels = supervised_composed_target(example).split()
        indices = [ord(value) - ord("A") for value in labels]
        self.assertTrue(all(0 <= value < len(example.action_texts) for value in indices))
        self.assertEqual(len(indices), 4)


if __name__ == "__main__":
    unittest.main()
