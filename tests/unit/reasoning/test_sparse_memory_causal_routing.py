from __future__ import annotations

import unittest

import torch

from angler.reasoning.sparse_memory_causal_routing import SparseMemoryCausalRoutingCore


class SparseMemoryCausalRoutingTests(unittest.TestCase):
    def _core(self):
        torch.manual_seed(101)
        return SparseMemoryCausalRoutingCore(content_width=6, temporal_width=3)

    @staticmethod
    def _inputs(outcomes=None):
        generator = torch.Generator().manual_seed(102)
        query = torch.randn(1, 6, generator=generator)
        candidates = torch.randn(1, 4, 6, generator=generator)
        temporal = torch.randn(1, 4, 3, generator=generator)
        outcomes = torch.zeros(1, 4) if outcomes is None else outcomes
        return query, candidates, temporal, outcomes, torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4)

    def test_empty_memory_has_exactly_zero_residual(self):
        core = self._core()
        output = core(*self._inputs(), memory_state=core.initial_memory_state())
        self.assertTrue(torch.equal(output.residuals, torch.zeros_like(output.residuals)))
        self.assertTrue(torch.equal(output.scores, output.normalized_base_scores))

    def test_read_and_actual_write_touch_at_most_two_slots(self):
        core = self._core()
        state = core.initial_memory_state()
        event = list(self._inputs(torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
        event[4] = torch.tensor([[True, False, False, False]])
        output = core(*event, memory_state=state)
        self.assertLessEqual(int((output.read_weights[0, 0] > 0).sum()), 2)
        self.assertLessEqual(int((output.write_weights[0, 0] > 0).sum()), 2)
        self.assertAlmostEqual(float(output.write_weights[0, 0].sum()), float(output.write_strengths[0, 0]), places=6)

    def test_nonempty_memory_changes_scores_and_backpropagates_selected_routes(self):
        core = self._core()
        state = core.initial_memory_state()
        event = list(self._inputs(torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
        event[4] = torch.tensor([[True, False, False, False]])
        support = core(*event, memory_state=state)
        state = core.apply_mixed_outcome_update(state, support, event[3], event[4], detach_state=False)
        held = core(*self._inputs(), memory_state=state)
        self.assertGreater(float(held.residuals.abs().sum()), 0.0)
        (-torch.log(held.weights[0, 0].clamp_min(1.0e-8))).backward()
        self.assertGreater(float(core.writer[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.score_network.second.weight.grad.abs().sum()), 0.0)
        self.assertLessEqual(float(state.keys.abs().max()), 1.0)
        self.assertLessEqual(float(state.values.abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()

