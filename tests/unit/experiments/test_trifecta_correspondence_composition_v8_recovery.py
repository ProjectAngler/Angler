import unittest

import torch

from experiments.evaluators.trifecta_correspondence_composition_v8_recovery import (
    restore_plastic_state,
)


class TrifectaV8RecoveryTests(unittest.TestCase):
    def test_restore_plastic_state_is_exact(self) -> None:
        record = {
            "keys": torch.randn(1, 3, 4),
            "values": torch.randn(1, 3, 4),
            "strengths": torch.rand(1, 3),
            "step": 7,
        }
        state = restore_plastic_state(record, torch.device("cpu"))
        self.assertTrue(torch.equal(state.keys, record["keys"]))
        self.assertTrue(torch.equal(state.values, record["values"]))
        self.assertTrue(torch.equal(state.strengths, record["strengths"]))
        self.assertEqual(state.step, 7)


if __name__ == "__main__":
    unittest.main()

