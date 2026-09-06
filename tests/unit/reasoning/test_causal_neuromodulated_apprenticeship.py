from __future__ import annotations

import unittest

import torch

from angler.reasoning.causal_neuromodulated_apprenticeship import (
    CausalNeuromodulatedApprenticeshipCore,
)


class CausalNeuromodulatedApprenticeshipTests(unittest.TestCase):
    def _core(self) -> CausalNeuromodulatedApprenticeshipCore:
        torch.manual_seed(81)
        return CausalNeuromodulatedApprenticeshipCore(
            content_width=6,
            temporal_width=3,
            rank=5,
            maximum_update=0.2,
        )

    @staticmethod
    def _inputs(*, outcomes: torch.Tensor | None = None, requires_grad: bool = False):
        generator = torch.Generator().manual_seed(82)
        query = torch.randn(1, 6, generator=generator, requires_grad=requires_grad)
        candidates = torch.randn(1, 3, 6, generator=generator, requires_grad=requires_grad)
        temporal = torch.randn(1, 3, 3, generator=generator, requires_grad=requires_grad)
        if outcomes is None:
            outcomes = torch.tensor([[1.0, -1.0, 0.0]])
        outcomes = outcomes.detach().clone().requires_grad_(requires_grad)
        mask = torch.tensor([[True, True, True]])
        base = torch.zeros(1, 3, requires_grad=requires_grad)
        return query, candidates, temporal, outcomes, mask, base

    def _adapt(self, core, values, *, detach_state: bool):
        state = core.initial_plastic_state()
        output = core(*values, plastic_state=state)
        return core.apply_mixed_outcome_update(
            state,
            output,
            values[3],
            values[4],
            detach_state=detach_state,
        ), output

    def test_vector_feature_and_write_gates_are_bounded(self) -> None:
        core = self._core()
        values = list(self._inputs())
        values[4] = torch.tensor([[True, False, True]])
        output = core(*values)

        self.assertEqual(output.feature_gates.shape, (1, 3, 5))
        self.assertEqual(output.write_gates.shape, (1, 3, 5))
        self.assertTrue(((output.feature_gates[values[4]] >= 0.0) & (output.feature_gates[values[4]] <= 1.0)).all())
        self.assertTrue(((output.write_gates[values[4]] >= 0.0) & (output.write_gates[values[4]] <= 1.0)).all())
        self.assertTrue(torch.equal(output.feature_gates[0, 1], torch.zeros(5)))
        self.assertEqual(output.weights[0, 1].item(), 0.0)

    def test_counterfactual_outcomes_produce_distinct_functional_states(self) -> None:
        core = self._core()
        true_values = self._inputs()
        counterfactual = self._inputs(outcomes=torch.tensor([[-1.0, 1.0, 0.0]]))
        true_state, _ = self._adapt(core, true_values, detach_state=False)
        counterfactual_state, _ = self._adapt(core, counterfactual, detach_state=False)

        self.assertEqual(true_state.step, 1)
        self.assertFalse(torch.equal(true_state.fast_weight, counterfactual_state.fast_weight))
        self.assertFalse(torch.equal(true_state.fast_bias, counterfactual_state.fast_bias))

    def test_heldout_causal_contrast_reaches_outcome_and_neuromodulator(self) -> None:
        core = self._core()
        true_values = self._inputs(requires_grad=True)
        counterfactual = self._inputs(
            outcomes=torch.tensor([[-1.0, 1.0, 0.0]]),
            requires_grad=True,
        )
        true_state, _ = self._adapt(core, true_values, detach_state=False)
        counterfactual_state, _ = self._adapt(core, counterfactual, detach_state=False)

        held_values = list(self._inputs(requires_grad=True))
        held_values[3] = torch.zeros_like(held_values[3], requires_grad=True)
        true_held = core(*held_values, plastic_state=true_state)
        counterfactual_held = core(*held_values, plastic_state=counterfactual_state)
        contrast = -(true_held.weights[0, 0] - counterfactual_held.weights[0, 0])
        contrast.backward()

        for observation in (
            *true_values[:4], true_values[5],
            *counterfactual[:4], counterfactual[5],
            *held_values[:4], held_values[5],
        ):
            self.assertIsNone(observation.grad)
        self.assertGreater(float(core.outcome_embedding.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.neuromodulator[-1].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
