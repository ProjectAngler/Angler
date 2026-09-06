from __future__ import annotations

import unittest

import torch

from angler.reasoning import CandidateProcedureDecoder, candidate_procedure_loss


class CandidateProcedureDecoderTests(unittest.TestCase):
    def _decoder(self) -> CandidateProcedureDecoder:
        return CandidateProcedureDecoder(
            content_width=16,
            procedure_width=12,
            hidden_width=24,
            heads=4,
            maximum_actions=6,
            maximum_steps=4,
        )

    def test_teacher_forcing_is_finite_and_masks_absent_candidates(self) -> None:
        torch.manual_seed(5)
        decoder = self._decoder()
        slots = torch.randn(2, 5, 12)
        actions = torch.randn(2, 6, 16)
        mask = torch.tensor(
            [[True, True, True, False, False, False], [True] * 6]
        )
        targets = torch.tensor([[0, 2, 6, -100], [5, 1, 3, 6]])
        decoded = decoder(slots, actions, mask, teacher_actions=targets)
        loss = candidate_procedure_loss(decoded, targets)
        loss.backward()
        self.assertEqual(decoded.logits.shape, (2, 4, 7))
        self.assertTrue(torch.isneginf(decoded.logits[0, :, 3:6]).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in decoder.parameters()))

    def test_candidate_meaning_changes_pointer_logits(self) -> None:
        torch.manual_seed(7)
        decoder = self._decoder()
        slots = torch.randn(1, 5, 12)
        actions = torch.randn(1, 6, 16)
        mask = torch.ones(1, 6, dtype=torch.bool)
        first = decoder(slots, actions, mask).logits
        changed = actions.clone()
        changed[:, 0] = changed[:, 0] + 4.0
        second = decoder(slots, changed, mask).logits
        self.assertFalse(torch.equal(first, second))

    def test_greedy_decode_emits_bounded_candidate_or_stop_indices(self) -> None:
        decoder = self._decoder()
        decoded = decoder(
            torch.randn(1, 3, 12),
            torch.randn(1, 6, 16),
            torch.tensor([[True, True, True, False, False, False]]),
        )
        self.assertEqual(decoded.selected_indices.shape, (1, 4))
        self.assertTrue(bool(((decoded.selected_indices <= 2) | (decoded.selected_indices == 6)).all()))

    def test_frozen_inference_slots_support_decoder_weight_gradients(self) -> None:
        decoder = self._decoder()
        with torch.inference_mode():
            frozen_slots = torch.randn(1, 3, 12)
        self.assertTrue(torch.is_inference(frozen_slots))
        decoded = decoder(
            frozen_slots,
            torch.randn(1, 6, 16),
            torch.ones(1, 6, dtype=torch.bool),
            teacher_actions=torch.tensor([[0, 1, 6, -100]]),
        )
        candidate_procedure_loss(
            decoded,
            torch.tensor([[0, 1, 6, -100]]),
        ).backward()
        self.assertIsNotNone(decoder.procedure_projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
