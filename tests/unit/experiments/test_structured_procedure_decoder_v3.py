from __future__ import annotations

import unittest

from experiments.evaluators.software_pipeline_reconstruction_suite import make_software_pipeline_stream
from experiments.runners.structured_procedure_decoder_v3 import (
    EPOCHS,
    IDENTITY,
    MAXIMUM_ACTIONS,
    MAXIMUM_STEPS,
    _targets,
)
from experiments.runners.scaled_procedural_software_v1 import (
    PreparedExample,
    prepare_corpus,
    serialize_action_candidates,
)


class StructuredProcedureDecoderV3Tests(unittest.TestCase):
    def test_identity_and_training_scale_are_frozen(self) -> None:
        self.assertEqual(IDENTITY, "angler.structured-procedure-decoder.v3")
        self.assertEqual(EPOCHS, 4)
        self.assertEqual(MAXIMUM_ACTIONS, 6)
        self.assertEqual(MAXIMUM_STEPS, 4)

    def test_action_candidates_are_public_local_component_rows(self) -> None:
        stream = make_software_pipeline_stream(8173, mechanism_partition="train")
        task = stream.queries[0].learner
        rows = serialize_action_candidates(task)
        self.assertEqual(len(rows), len(task.grounded_candidates))
        self.assertTrue(rows[0].startswith("A in="))
        self.assertTrue(rows[-1].startswith("F in="))
        self.assertNotIn(task.origin.namespace, "\n".join(rows))

    def test_visible_target_maps_to_candidate_indices_and_stop(self) -> None:
        _, train, _, _ = prepare_corpus()
        example: PreparedExample = train[0].supports[0]
        target = _targets(example).cpu().tolist()[0]
        self.assertEqual(target[2], MAXIMUM_ACTIONS)
        self.assertEqual(target[3], -100)


if __name__ == "__main__":
    unittest.main()
