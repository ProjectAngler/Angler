import unittest

import torch

from angler.reasoning import EdgeAwareActionTraceDecoder


class EdgeAwareActionTraceDecoderTests(unittest.TestCase):
    def _decoder(self):
        return EdgeAwareActionTraceDecoder(
            content_width=16,
            procedure_width=12,
            hidden_width=24,
            heads=4,
            maximum_actions=6,
            maximum_steps=4,
            relation_count=8,
            message_rounds=2,
        )

    def test_shapes_gradients_and_without_replacement(self) -> None:
        decoder = self._decoder()
        slots = torch.randn(1, 3, 12)
        actions = torch.randn(1, 6, 16)
        mask = torch.ones(1, 6, dtype=torch.bool)
        edges = torch.randn(1, 6, 6, 8)
        donor = torch.linspace(2.0, -2.0, 6).unsqueeze(0)
        memory = torch.randn(5, 12)
        memory_mask = torch.ones(5, dtype=torch.bool)
        output = decoder(
            slots,
            actions,
            mask,
            edge_features=edges,
            donor_scores=donor,
            memory_values=memory,
            memory_mask=memory_mask,
        )
        self.assertEqual(output.logits.shape, (1, 4, 7))
        selected = [value for value in output.selected_indices[0].tolist() if value < 6]
        self.assertEqual(len(selected), len(set(selected)))
        output.logits[torch.isfinite(output.logits)].square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in decoder.parameters()))

    def test_donor_prior_is_causally_removable(self) -> None:
        decoder = self._decoder().eval()
        arguments = dict(
            procedure_slots=torch.zeros(1, 2, 12),
            action_features=torch.zeros(1, 6, 16),
            action_mask=torch.ones(1, 6, dtype=torch.bool),
            edge_features=torch.zeros(1, 6, 6, 8),
            donor_scores=torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            memory_values=torch.zeros(1, 12),
            memory_mask=torch.zeros(1, dtype=torch.bool),
            edge_reasoning=False,
        )
        with_prior = decoder(**arguments, donor_prior=True).logits
        without_prior = decoder(**arguments, donor_prior=False).logits
        self.assertGreater(
            float((with_prior - without_prior)[0, 0, 0].detach()), 1.0
        )


if __name__ == "__main__":
    unittest.main()
