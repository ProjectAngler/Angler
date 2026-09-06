from __future__ import annotations

import unittest

import torch

from experiments.runners.trace_conditioned_procedure_v5 import (
    EPOCHS,
    IDENTITY,
    _procedure_trace,
)
from experiments.runners.compositional_procedure_v4_r1 import _public_training_streams
from experiments.runners.scaled_procedural_software_v1 import prepare_corpus


class TraceConditionedProcedureV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence, raw, _, _ = prepare_corpus()
        cls.train = _public_training_streams(raw)

    def test_identity_budget_and_hidden_pair_isolation(self) -> None:
        self.assertEqual(IDENTITY, "angler.trace-conditioned-procedure.v5")
        self.assertEqual(EPOCHS, 8)
        for stream in self.train:
            for query in stream.queries:
                self.assertIsNone(query.pair)

    def test_visible_trace_preserves_executed_order_and_excludes_stop(self) -> None:
        example = self.train[0].supports[0]
        width = 8
        embeddings = {
            text: torch.full((width,), float(index + 1))
            for index, text in enumerate(example.action_texts)
        }
        trace, mask = _procedure_trace(example, embeddings)
        targets = [token for token in example.target_text.split() if token != "STOP"]
        self.assertEqual(int(mask.sum()), len(targets))
        for step, token in enumerate(targets):
            index = ord(token) - ord("A")
            self.assertTrue(
                torch.equal(
                    trace[0, step].cpu(),
                    embeddings[example.action_texts[index]],
                )
            )
        self.assertTrue(torch.equal(trace[0, len(targets):], torch.zeros_like(trace[0, len(targets):])))


if __name__ == "__main__":
    unittest.main()
