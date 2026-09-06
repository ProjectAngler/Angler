from __future__ import annotations

import unittest

import torch

from angler.reasoning.content_addressed_causal_memory import (
    ContentAddressedCausalMemoryCore,
    ContentAddressedCausalMemoryState,
)


class ContentAddressedCausalMemoryTests(unittest.TestCase):
    def _core(self) -> ContentAddressedCausalMemoryCore:
        torch.manual_seed(91)
        return ContentAddressedCausalMemoryCore(content_width=6, temporal_width=3)

    @staticmethod
    def _inputs(*, outcomes=None, requires_grad=False):
        generator = torch.Generator().manual_seed(92)
        query = torch.randn(1, 6, generator=generator, requires_grad=requires_grad)
        candidates = torch.randn(1, 3, 6, generator=generator, requires_grad=requires_grad)
        temporal = torch.randn(1, 3, 3, generator=generator, requires_grad=requires_grad)
        outcomes = torch.tensor([[1.0, -1.0, 0.0]]) if outcomes is None else outcomes
        outcomes = outcomes.detach().clone().requires_grad_(requires_grad)
        mask = torch.tensor([[True, True, True]])
        base = torch.zeros(1, 3, requires_grad=requires_grad)
        return query, candidates, temporal, outcomes, mask, base

    @staticmethod
    def _update(core, values, state, *, detach_state):
        output = core(*values, memory_state=state)
        return core.apply_mixed_outcome_update(
            state,
            output,
            values[3],
            values[4],
            detach_state=detach_state,
        ), output

    def test_exact_memory_shape_and_long_update_bounds(self) -> None:
        core = self._core()
        state = core.initial_memory_state()
        self.assertEqual(state.keys.shape, (16, 32))
        self.assertEqual(state.values.shape, (16, 32))
        self.assertEqual(state.usage.shape, (16,))
        self.assertGreater(float(state.keys.detach().var(dim=0).sum()), 0.0)

        values = self._inputs()
        for _ in range(128):
            state, _ = self._update(core, values, state, detach_state=True)
        self.assertEqual(state.step, 128)
        self.assertLessEqual(float(state.keys.abs().max()), 1.0)
        self.assertLessEqual(float(state.values.abs().max()), 1.0)
        self.assertGreaterEqual(float(state.usage.min()), 0.0)
        self.assertLessEqual(float(state.usage.max()), 1.0)
        self.assertTrue(torch.isfinite(state.keys).all())

    def test_soft_slot_addresses_break_symmetry_across_distinct_events(self) -> None:
        core = self._core()
        initial = core.initial_memory_state()
        first = self._inputs()
        second = list(self._inputs())
        second[0] = -second[0]
        second[1] = second[1].roll(1, dims=1)
        state, first_output = self._update(core, first, initial, detach_state=True)
        state, second_output = self._update(core, second, state, detach_state=True)

        self.assertGreater(float(first_output.write_weights.detach().var(dim=-1).sum()), 0.0)
        self.assertGreater(float(second_output.write_weights.detach().var(dim=-1).sum()), 0.0)
        self.assertGreater(int((state.usage > 0.01).sum()), 1)
        self.assertGreater(float(state.keys.var(dim=0).sum()), 0.0)

    def test_content_read_concentrates_on_matching_slot(self) -> None:
        core = self._core()
        query = torch.zeros(1, 1, 32)
        query[..., 0] = 1.0
        keys = torch.zeros(16, 32)
        keys[:, 1] = 1.0
        keys[7].zero_()
        keys[7, 0] = 1.0
        weights = core._content_weights(
            query,
            keys,
            torch.tensor([[12.0]]),
            usage=torch.ones(16),
        )
        self.assertEqual(int(weights.argmax(dim=-1).item()), 7)
        self.assertGreater(float(weights[0, 0, 7]), 0.9)

    def test_counterfactual_history_changes_state_and_heldout_behavior(self) -> None:
        core = self._core()
        initial = core.initial_memory_state()
        true_values = self._inputs()
        false_values = self._inputs(outcomes=torch.tensor([[-1.0, 1.0, 0.0]]))
        true_state, _ = self._update(core, true_values, initial, detach_state=False)
        false_state, _ = self._update(core, false_values, initial, detach_state=False)
        held = self._inputs(outcomes=torch.zeros(1, 3))
        true_held = core(*held, memory_state=true_state)
        false_held = core(*held, memory_state=false_state)

        self.assertFalse(torch.equal(true_state.keys, false_state.keys))
        self.assertFalse(torch.equal(true_state.values, false_state.values))
        self.assertFalse(torch.equal(true_held.weights, false_held.weights))

    def test_causal_contrast_gradients_reach_read_write_outcome_and_modulator(self) -> None:
        core = self._core()
        initial = core.initial_memory_state()
        true_values = self._inputs(requires_grad=True)
        false_values = self._inputs(
            outcomes=torch.tensor([[-1.0, 1.0, 0.0]]),
            requires_grad=True,
        )
        true_state, _ = self._update(core, true_values, initial, detach_state=False)
        false_state, _ = self._update(core, false_values, initial, detach_state=False)
        held = self._inputs(outcomes=torch.zeros(1, 3), requires_grad=True)
        true_held = core(*held, memory_state=true_state)
        false_held = core(*held, memory_state=false_state)
        loss = -(true_held.weights[0, 0] - false_held.weights[0, 0])
        loss.backward()

        for observation in (
            *true_values[:4], true_values[5],
            *false_values[:4], false_values[5],
            *held[:4], held[5],
        ):
            self.assertIsNone(observation.grad)
        self.assertGreater(float(core.outcome_embedding.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.read_key_network[1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.writer[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.neuromodulator[-1].weight.grad.abs().sum()), 0.0)

    def test_masks_and_invalid_state_fail_closed(self) -> None:
        core = self._core()
        values = list(self._inputs())
        values[4] = torch.tensor([[True, False, True]])
        output = core(*values)
        self.assertEqual(output.weights[0, 1].item(), 0.0)
        self.assertTrue(torch.equal(output.read_weights[0, 1], torch.zeros(16)))
        self.assertTrue(torch.equal(output.write_weights[0, 1], torch.zeros(16)))
        with self.assertRaisesRegex(ValueError, "expose one candidate"):
            core(
                values[0],
                values[1],
                values[2],
                values[3],
                torch.zeros_like(values[4]),
                values[5],
            )

        invalid = ContentAddressedCausalMemoryState(
            keys=torch.full((16, 32), 1.01),
            values=torch.zeros(16, 32),
            usage=torch.zeros(16),
            step=0,
        )
        with self.assertRaisesRegex(ValueError, "bound"):
            core(*values, memory_state=invalid)
        values[3] = torch.tensor([[1.0, 0.25, 0.0]])
        with self.assertRaisesRegex(ValueError, r"only -1, 0, or \+1"):
            core(*values)


if __name__ == "__main__":
    unittest.main()
