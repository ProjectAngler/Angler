from __future__ import annotations

import unittest

import torch

from angler.reasoning.outcome_aware_apprenticeship import (
    OutcomeAwareApprenticeshipCore,
    OutcomeAwarePlasticState,
)


class OutcomeAwareApprenticeshipTests(unittest.TestCase):
    def _core(self) -> OutcomeAwareApprenticeshipCore:
        torch.manual_seed(71)
        return OutcomeAwareApprenticeshipCore(
            content_width=6,
            temporal_width=3,
            rank=5,
            maximum_update=0.2,
        )

    @staticmethod
    def _inputs(*, requires_grad: bool = False):
        generator = torch.Generator().manual_seed(72)
        query = torch.randn(1, 6, generator=generator, requires_grad=requires_grad)
        candidates = torch.randn(1, 3, 6, generator=generator, requires_grad=requires_grad)
        temporal = torch.randn(1, 3, 3, generator=generator, requires_grad=requires_grad)
        outcomes = torch.tensor(
            [[1.0, -1.0, 0.0]], requires_grad=requires_grad
        )
        mask = torch.tensor([[True, True, True]])
        base = torch.zeros(1, 3, requires_grad=requires_grad)
        return query, candidates, temporal, outcomes, mask, base

    def test_mixed_outcomes_share_one_learned_writer_and_change_state(self) -> None:
        core = self._core()
        values = self._inputs()
        state = core.initial_plastic_state(detach=True)
        output = core(*values, plastic_state=state)
        updated = core.apply_mixed_outcome_update(
            state, output, values[3], values[4]
        )
        swapped_outcomes = values[3].flip(1)
        swapped_output = core(
            values[0], values[1], values[2], swapped_outcomes, values[4], values[5],
            plastic_state=state,
        )
        swapped = core.apply_mixed_outcome_update(
            state, swapped_output, swapped_outcomes, values[4]
        )

        self.assertEqual(updated.step, 1)
        self.assertTrue(torch.isfinite(updated.fast_weight).all())
        self.assertFalse(torch.equal(updated.fast_weight, state.fast_weight))
        self.assertFalse(torch.equal(updated.fast_weight, swapped.fast_weight))

    def test_functional_inner_update_supports_outer_gradients_and_detaches_observations(self) -> None:
        core = self._core()
        values = self._inputs(requires_grad=True)
        state = core.initial_plastic_state()
        support = core(*values, plastic_state=state)
        adapted = core.apply_mixed_outcome_update(
            state,
            support,
            values[3],
            values[4],
            detach_state=False,
        )
        query = values[0].detach().clone().requires_grad_(True)
        candidates = values[1].detach().clone().requires_grad_(True)
        temporal = values[2].detach().clone().requires_grad_(True)
        outcomes = values[3].detach().clone().requires_grad_(True)
        base = values[5].detach().clone().requires_grad_(True)
        held_out = core(
            query,
            candidates,
            temporal,
            outcomes,
            values[4],
            base,
            plastic_state=adapted,
        )
        loss = -torch.log(held_out.weights[0, 0].clamp_min(1.0e-8))
        loss.backward()

        for observation in (*values[:4], values[5], query, candidates, temporal, outcomes, base):
            self.assertIsNone(observation.grad)
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in core.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)
        self.assertGreater(float(core.state_writer[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.outcome_embedding.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.query_projection.weight.grad.abs().sum()), 0.0)

    def test_masking_and_state_validation_fail_closed(self) -> None:
        core = self._core()
        values = list(self._inputs())
        values[4] = torch.tensor([[True, False, True]])
        output = core(*values)
        self.assertEqual(output.weights[0, 1].item(), 0.0)
        self.assertTrue(torch.isneginf(output.scores[0, 1]))

        invalid = OutcomeAwarePlasticState(
            fast_weight=torch.full((5,), float("nan")),
            fast_bias=torch.zeros(()),
            step=0,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            core(*values, plastic_state=invalid)
        values[3] = torch.tensor([[1.0, 0.5, 0.0]])
        with self.assertRaisesRegex(ValueError, r"only -1, 0, or \+1"):
            core(*values)


if __name__ == "__main__":
    unittest.main()
