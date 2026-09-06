from __future__ import annotations

import unittest

import torch

from angler.reasoning import DncAllocatedFactorizedCausalMemoryCore


class DncAllocatedFactorizedCausalMemoryTests(unittest.TestCase):
    def _core(self) -> DncAllocatedFactorizedCausalMemoryCore:
        torch.manual_seed(151)
        return DncAllocatedFactorizedCausalMemoryCore(
            content_width=6,
            temporal_width=3,
        )

    @staticmethod
    def _event(step: int, outcome: float) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(152 + step)
        return (
            torch.randn(1, 6, generator=generator),
            torch.randn(1, 1, 6, generator=generator),
            torch.randn(1, 1, 3, generator=generator),
            torch.tensor([[outcome]], dtype=torch.float32),
            torch.ones(1, 1, dtype=torch.bool),
            torch.randn(1, 1, generator=generator),
        )

    @staticmethod
    def _challenge(step: int = 1000) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(152 + step)
        return (
            torch.randn(1, 6, generator=generator),
            torch.randn(1, 4, 6, generator=generator),
            torch.randn(1, 4, 3, generator=generator),
            torch.zeros(1, 4, dtype=torch.float32),
            torch.ones(1, 4, dtype=torch.bool),
            torch.randn(1, 4, generator=generator),
        )

    @staticmethod
    def _update(core, state, event):
        output = core(*event, memory_state=state)
        state = core.apply_mixed_outcome_update(
            state,
            output,
            event[3],
            event[4],
            detach_state=True,
        )
        return state, output

    def test_exact_zero_initial_memory_has_no_learned_slot_identity(self) -> None:
        core = self._core()
        state = core.initial_memory_state()

        self.assertEqual(state.keys.shape, (64, 32))
        self.assertEqual(state.values.shape, (64, 32))
        self.assertEqual(state.usage.shape, (64,))
        self.assertEqual(state.bytes, 16640)
        self.assertTrue(torch.equal(state.keys, torch.zeros_like(state.keys)))
        self.assertTrue(torch.equal(state.values, torch.zeros_like(state.values)))
        self.assertTrue(torch.equal(state.usage, torch.zeros_like(state.usage)))
        self.assertFalse(hasattr(core, "slot_address_anchors"))
        self.assertFalse(hasattr(core, "allocation_logits"))
        self.assertNotIn("slot_address_anchors", dict(core.named_parameters()))
        self.assertNotIn("allocation_logits", dict(core.named_parameters()))

    def test_first_forty_eight_writes_allocate_once_to_distinct_slots(self) -> None:
        core = self._core()
        state = core.initial_memory_state()
        winners = []

        for step in range(48):
            event = self._event(step, 1.0 if step % 2 == 0 else -1.0)
            state, output = self._update(core, state, event)
            actual = output.write_weights[0, 0]
            nonzero = torch.nonzero(actual > 0.0, as_tuple=False).flatten()
            self.assertEqual(nonzero.numel(), 1)
            winners.append(int(nonzero.item()))
            self.assertLessEqual(
                int((output.read_weights[0, 0] > 0.0).sum().item()),
                2,
            )

        self.assertEqual(len(set(winners)), 48)
        self.assertEqual(len(set(range(64)).difference(winners)), 16)
        self.assertEqual(int((state.usage > 0.0).sum().item()), 48)
        self.assertEqual(state.step, 48)

    def test_outcome_shuffle_preserves_addresses_and_changes_only_values(self) -> None:
        core = self._core()
        true_state = core.initial_memory_state()
        shuffled_state = core.initial_memory_state()
        true_outcomes = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        shuffled_outcomes = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

        for step, (true_outcome, shuffled_outcome) in enumerate(
            zip(true_outcomes, shuffled_outcomes)
        ):
            true_event = self._event(step, true_outcome)
            shuffled_event = (*true_event[:3], torch.tensor([[shuffled_outcome]]), *true_event[4:])
            true_state, true_output = self._update(core, true_state, true_event)
            shuffled_state, shuffled_output = self._update(
                core,
                shuffled_state,
                shuffled_event,
            )

            torch.testing.assert_close(
                true_output.write_keys,
                shuffled_output.write_keys,
                rtol=0.0,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                true_output.write_weights,
                shuffled_output.write_weights,
                rtol=0.0,
                atol=1.0e-6,
            )

        torch.testing.assert_close(
            true_state.keys,
            shuffled_state.keys,
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            true_state.usage,
            shuffled_state.usage,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertGreater(
            float((true_state.values - shuffled_state.values).abs().max().item()),
            1.0e-6,
        )

    def test_slot_permutation_preserves_read_values_and_scores(self) -> None:
        core = self._core()
        state = core.initial_memory_state()
        for step in range(24):
            state, _ = self._update(
                core,
                state,
                self._event(step, 1.0 if step % 2 == 0 else -1.0),
            )

        permutation = torch.randperm(64, generator=torch.Generator().manual_seed(177))
        permuted = type(state)(
            keys=state.keys[permutation].clone(),
            values=state.values[permutation].clone(),
            usage=state.usage[permutation].clone(),
            step=state.step,
        )
        held = self._challenge()
        original_output = core(*held, memory_state=state)
        permuted_output = core(*held, memory_state=permuted)

        torch.testing.assert_close(
            original_output.read_values,
            permuted_output.read_values,
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            original_output.residuals,
            permuted_output.residuals,
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            original_output.scores,
            permuted_output.scores,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertTrue(
            bool(((original_output.read_weights > 0.0).sum(dim=-1) <= 2).all().item())
        )
        self.assertTrue(
            bool(((permuted_output.read_weights > 0.0).sum(dim=-1) <= 2).all().item())
        )

    def test_more_than_sixty_four_writes_remain_finite_and_bounded(self) -> None:
        core = self._core()
        state = core.initial_memory_state()

        for step in range(72):
            state, output = self._update(
                core,
                state,
                self._event(step, 1.0 if step % 2 == 0 else -1.0),
            )
            self.assertEqual(int((output.write_weights[0, 0] > 0.0).sum().item()), 1)
            self.assertLessEqual(
                int((output.read_weights[0, 0] > 0.0).sum().item()),
                2,
            )

        self.assertEqual(state.step, 72)
        self.assertTrue(bool(torch.isfinite(state.keys).all().item()))
        self.assertTrue(bool(torch.isfinite(state.values).all().item()))
        self.assertTrue(bool(torch.isfinite(state.usage).all().item()))
        self.assertLessEqual(float(state.keys.abs().max().item()), 1.0)
        self.assertLessEqual(float(state.values.abs().max().item()), 1.0)
        self.assertGreaterEqual(float(state.usage.min().item()), 0.0)
        self.assertLessEqual(float(state.usage.max().item()), 1.0)


if __name__ == "__main__":
    unittest.main()
