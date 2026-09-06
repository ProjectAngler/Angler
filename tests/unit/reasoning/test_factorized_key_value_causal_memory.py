from __future__ import annotations

import unittest

import torch

from angler.reasoning import FactorizedKeyValueCausalMemoryCore


class FactorizedKeyValueCausalMemoryTests(unittest.TestCase):
    def _core(self) -> FactorizedKeyValueCausalMemoryCore:
        torch.manual_seed(131)
        return FactorizedKeyValueCausalMemoryCore(
            content_width=6,
            temporal_width=3,
        )

    @staticmethod
    def _inputs(
        *,
        step: int = 0,
        outcomes: torch.Tensor | None = None,
        candidates: int = 3,
    ) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(132 + step)
        query = torch.randn(1, 6, generator=generator)
        candidate_values = torch.randn(1, candidates, 6, generator=generator)
        temporal = torch.randn(1, candidates, 3, generator=generator)
        observed = (
            torch.zeros(1, candidates, dtype=torch.float32)
            if outcomes is None
            else outcomes.detach().clone().to(dtype=torch.float32)
        )
        mask = torch.ones(1, candidates, dtype=torch.bool)
        base_scores = torch.randn(1, candidates, generator=generator)
        return query, candidate_values, temporal, observed, mask, base_scores

    @staticmethod
    def _update(
        core: FactorizedKeyValueCausalMemoryCore,
        inputs: tuple[torch.Tensor, ...],
        state,
        *,
        detach_state: bool,
    ):
        output = core(*inputs, memory_state=state)
        updated = core.apply_mixed_outcome_update(
            state,
            output,
            inputs[3],
            inputs[4],
            detach_state=detach_state,
        )
        return updated, output

    def test_empty_and_all_neutral_memory_have_exactly_zero_residual(self) -> None:
        core = self._core()
        state = core.initial_memory_state()
        neutral = self._inputs(outcomes=torch.zeros(1, 3))

        empty_output = core(*neutral, memory_state=state)
        self.assertTrue(
            torch.equal(empty_output.residuals, torch.zeros_like(empty_output.residuals))
        )
        self.assertTrue(
            torch.equal(empty_output.scores, empty_output.normalized_base_scores)
        )

        for step in range(6):
            neutral_step = self._inputs(
                step=step,
                outcomes=torch.zeros(1, 3),
            )
            state, write_output = self._update(
                core,
                neutral_step,
                state,
                detach_state=True,
            )
            self.assertTrue(
                torch.equal(
                    write_output.write_values,
                    torch.zeros_like(write_output.write_values),
                )
            )
            self.assertTrue(torch.equal(state.values, torch.zeros_like(state.values)))

        held = core(*self._inputs(step=20, outcomes=torch.zeros(1, 3)), memory_state=state)
        self.assertTrue(torch.equal(held.residuals, torch.zeros_like(held.residuals)))
        self.assertTrue(torch.equal(held.scores, held.normalized_base_scores))

    def test_outcome_permutation_changes_values_but_not_addresses_or_usage(self) -> None:
        core = self._core()
        true_state = core.initial_memory_state()
        shuffled_state = core.initial_memory_state()
        true_outcomes = (1.0, -1.0, 1.0, -1.0)
        shuffled_outcomes = (-1.0, 1.0, -1.0, 1.0)

        for step, (true_outcome, shuffled_outcome) in enumerate(
            zip(true_outcomes, shuffled_outcomes)
        ):
            common = self._inputs(step=step, candidates=1)
            true_inputs = (
                *common[:3],
                torch.tensor([[true_outcome]]),
                *common[4:],
            )
            shuffled_inputs = (
                *common[:3],
                torch.tensor([[shuffled_outcome]]),
                *common[4:],
            )
            true_state, true_output = self._update(
                core,
                true_inputs,
                true_state,
                detach_state=True,
            )
            shuffled_state, shuffled_output = self._update(
                core,
                shuffled_inputs,
                shuffled_state,
                detach_state=True,
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
            float((true_state.values - shuffled_state.values).abs().max()),
            1.0e-6,
        )
        challenge = self._inputs(step=40, outcomes=torch.zeros(1, 3))
        true_read = core(*challenge, memory_state=true_state)
        shuffled_read = core(*challenge, memory_state=shuffled_state)
        torch.testing.assert_close(
            true_read.read_weights,
            shuffled_read.read_weights,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_reads_and_writes_are_top_two_and_state_remains_bounded(self) -> None:
        core = self._core()
        state = core.initial_memory_state()
        last_output = None
        for step in range(32):
            outcomes = torch.tensor(
                [[1.0, -1.0, 0.0]] if step % 2 == 0 else [[-1.0, 1.0, 0.0]]
            )
            inputs = self._inputs(step=step, outcomes=outcomes)
            state, last_output = self._update(
                core,
                inputs,
                state,
                detach_state=True,
            )
            self.assertTrue(
                bool(((last_output.read_weights > 0).sum(dim=-1) <= 2).all().item())
            )
            self.assertTrue(
                bool(((last_output.write_weights > 0).sum(dim=-1) <= 2).all().item())
            )

        self.assertIsNotNone(last_output)
        self.assertEqual(state.keys.shape, (16, 32))
        self.assertEqual(state.values.shape, (16, 32))
        self.assertEqual(state.usage.shape, (16,))
        self.assertTrue(bool(torch.isfinite(state.keys).all().item()))
        self.assertTrue(bool(torch.isfinite(state.values).all().item()))
        self.assertTrue(bool(torch.isfinite(state.usage).all().item()))
        self.assertLessEqual(float(state.keys.abs().max()), 1.0)
        self.assertLessEqual(float(state.values.abs().max()), 1.0)
        self.assertGreaterEqual(float(state.usage.min()), 0.0)
        self.assertLessEqual(float(state.usage.max()), 1.0)

    def test_nonneutral_history_backpropagates_through_value_read_and_score_paths(self) -> None:
        core = self._core()
        state = core.initial_memory_state()
        support = self._inputs(outcomes=torch.tensor([[1.0, -1.0, 0.0]]))
        state, _ = self._update(
            core,
            support,
            state,
            detach_state=False,
        )
        held = core(
            *self._inputs(step=50, outcomes=torch.zeros(1, 3)),
            memory_state=state,
        )
        loss = -torch.log(held.weights[0, 0].clamp_min(1.0e-8))
        loss.backward()

        parameters = {
            "outcome_embedding": core.outcome_embedding.weight,
            "writer": core.writer[-1].weight,
            "read_key": core.read_key_network[1].weight,
            "neuromodulator": core.neuromodulator[-1].weight,
            "score": core.score_network.second.weight,
        }
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all().item()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
