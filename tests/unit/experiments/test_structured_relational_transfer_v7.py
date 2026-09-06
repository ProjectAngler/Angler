from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from angler.reasoning import SemanticRelationalFusion, StructuredRelationalEncoder
from experiments.runners import structured_relational_transfer_v7 as v7


class StructuredRelationalTransferV7Tests(unittest.TestCase):
    def test_identity_and_frozen_parent(self) -> None:
        self.assertEqual(v7.IDENTITY, "angler.structured-relational-transfer.v7")
        self.assertEqual(
            v7.PARENT_CHECKPOINT_SHA256,
            "08a966424ca678e9742a8d80b985ba48fc0c0ff8e3df4bc86c0262b9c53fad7a",
        )

    def test_action_features_align_to_frozen_six_candidate_interface(self) -> None:
        query = "origin=S0; goal=S2; forbidden=none; budget=4\ncomponents:"
        actions = (
            "A in=T0 out=T1 reads=S0 writes=S1 graph=0>1,1>2",
            "B in=T1 out=T2 reads=S1 writes=S2 graph=0>2,2>1",
        )
        example = SimpleNamespace(query_text=query, action_texts=actions)
        embeddings = {text: torch.randn(64) for text in actions}
        encoder = StructuredRelationalEncoder(width=32, reasoning_steps=2, workspace_slots=2)
        fusion = SemanticRelationalFusion(
            semantic_width=64,
            relational_width=32,
            hidden_width=48,
        )
        features, mask = v7._action_features(example, embeddings, encoder, fusion)
        self.assertEqual(features.shape, (1, 6, 64))
        self.assertEqual(mask.tolist(), [[True, True, False, False, False, False]])

    def test_trace_excludes_stop_and_padding(self) -> None:
        features = torch.randn(1, 6, 32)
        target = torch.tensor([[1, 0, 6, -100]])
        trace, mask = v7._trace_from_actions(features, target)
        self.assertTrue(torch.equal(trace[0, 0], features[0, 1]))
        self.assertTrue(torch.equal(trace[0, 1], features[0, 0]))
        self.assertEqual(mask.tolist(), [[True, True, False, False]])

    def test_structure_is_a_required_causal_margin(self) -> None:
        self.assertIn("structured_removed", v7.CAUSAL_MARGIN_ARMS)
        self.assertIn("topology_removed", v7.DIAGNOSTIC_ARMS)
        self.assertIn("action_semantics_removed", v7.DIAGNOSTIC_ARMS)


if __name__ == "__main__":
    unittest.main()
