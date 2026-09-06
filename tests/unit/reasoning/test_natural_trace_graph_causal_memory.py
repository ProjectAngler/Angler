from __future__ import annotations

import unittest

import torch

from angler.reasoning.natural_trace_graph_causal_memory import (
    NaturalLanguageTraceGraphEncoder,
    NaturalTraceGraphCausalMemoryCore,
    parse_step_trace,
)


class NaturalTraceGraphCausalMemoryTests(unittest.TestCase):
    def _core(self) -> NaturalTraceGraphCausalMemoryCore:
        torch.manual_seed(211)
        return NaturalTraceGraphCausalMemoryCore(
            content_width=6,
            temporal_width=3,
            step_width=6,
            relational_width=16,
            maximum_trace_steps=6,
        )

    @staticmethod
    def _inputs(
        *,
        seed: int,
        outcomes: torch.Tensor,
        candidates: int = 1,
        steps: int = 4,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        query = torch.randn(1, 6, generator=generator)
        candidate = torch.randn(1, candidates, 6, generator=generator)
        temporal = torch.randn(1, candidates, 3, generator=generator)
        mask = torch.ones(1, candidates, dtype=torch.bool)
        base = torch.randn(1, candidates, generator=generator)
        step_features = torch.randn(1, candidates, steps, 6, generator=generator)
        step_mask = torch.ones(1, candidates, steps, dtype=torch.bool)
        return (
            (query, candidate, temporal, outcomes, mask, base),
            step_features,
            step_mask,
        )

    @staticmethod
    def _forward(core, inputs, steps, step_mask, state=None, **kwargs):
        return core(
            *inputs,
            candidate_step_features=steps,
            candidate_step_mask=step_mask,
            memory_state=state,
            **kwargs,
        )

    def test_parser_preserves_raw_public_sentences_and_fails_closed(self) -> None:
        self.assertEqual(
            parse_step_trace(
                "Step 1: Inspect the workspace. Step 2: Apply the change. "
                "Step 3: Verify the result."
            ),
            (
                "Inspect the workspace.",
                "Apply the change.",
                "Verify the result.",
            ),
        )
        invalid = (
            "",
            "Inspect first. Step 1: Verify.",
            "Step 0: Inspect.",
            "Step 2: Inspect.",
            "Step 1: Inspect. Step 3: Verify.",
            "Step 1: Inspect. Step 1: Verify.",
            "Step 1:   Step 2: Verify.",
        )
        for trace in invalid:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                parse_step_trace(trace)

    def test_padding_is_invariant_and_mask_holes_fail_closed(self) -> None:
        torch.manual_seed(212)
        encoder = NaturalLanguageTraceGraphEncoder(step_width=6, relational_width=16)
        encoder.eval()
        visible = torch.randn(1, 1, 3, 6)
        left = torch.cat((visible, torch.zeros(1, 1, 2, 6)), dim=2)
        right = left.clone()
        right[..., 3:, :] = torch.tensor(float("nan"))
        mask = torch.tensor([[[True, True, True, False, False]]])
        with torch.inference_mode():
            left_value = encoder(left, mask)
            right_value = encoder(right, mask)
        torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)

        hole = torch.tensor([[[True, False, True, False, False]]])
        with self.assertRaises(ValueError):
            encoder(left, hole)

    def test_reorder_is_visible_and_repeated_evaluation_is_exact(self) -> None:
        torch.manual_seed(213)
        encoder = NaturalLanguageTraceGraphEncoder(step_width=6, relational_width=16)
        encoder.eval()
        original = torch.randn(1, 1, 4, 6)
        reordered = original[:, :, (0, 2, 1, 3), :]
        mask = torch.ones(1, 1, 4, dtype=torch.bool)
        with torch.inference_mode():
            first = encoder(original, mask)
            repeated = encoder(original, mask)
            changed = encoder(reordered, mask)
        self.assertTrue(torch.equal(first, repeated))
        self.assertGreater(float((first - changed).abs().max().item()), 1.0e-6)

    def test_graph_disabled_is_exactly_zero_and_pooled_semantics_are_untouched(self) -> None:
        core = self._core()
        inputs, steps, mask = self._inputs(
            seed=214,
            outcomes=torch.zeros(1, 3),
            candidates=3,
        )
        state = core.initial_memory_state()
        self.assertEqual(state.bytes, 26240)
        output = self._forward(
            core,
            inputs,
            steps,
            mask,
            state,
            include_graph_address=False,
        )
        self.assertTrue(
            torch.equal(
                output.graph_address_logits,
                torch.zeros_like(output.graph_address_logits),
            )
        )
        expected_candidates = core.candidate_projection(inputs[1].detach())
        expected_query = core.query_projection(inputs[0].detach()).unsqueeze(1).expand_as(
            expected_candidates
        )
        expected_temporal = core.temporal_projection(inputs[2].detach())
        expected_semantic = core.semantic_network(
            torch.cat(
                (
                    expected_candidates,
                    expected_query,
                    expected_candidates * expected_query,
                    expected_temporal,
                ),
                dim=-1,
            )
        )
        torch.testing.assert_close(
            output.pair_features,
            expected_semantic,
            rtol=0.0,
            atol=0.0,
        )

    def test_gradients_reach_graph_address_value_and_scorer_paths(self) -> None:
        core = self._core()
        support, support_steps, support_mask = self._inputs(
            seed=215,
            outcomes=torch.tensor([[1.0]]),
        )
        state = core.initial_memory_state()
        support_output = self._forward(
            core,
            support,
            support_steps,
            support_mask,
            state,
        )
        state = core.apply_mixed_outcome_update(
            state,
            support_output,
            support[3],
            support[4],
            detach_state=False,
        )
        challenge, challenge_steps, challenge_mask = self._inputs(
            seed=216,
            outcomes=torch.zeros(1, 3),
            candidates=3,
        )
        held = self._forward(
            core,
            challenge,
            challenge_steps,
            challenge_mask,
            state,
        )
        loss = -torch.log(held.weights[0, 0].clamp_min(1.0e-8))
        loss.backward()

        parameters = {
            "step_projection": core.trace_graph_encoder.step_projection.weight,
            "previous_message": core.trace_graph_encoder.previous_message.weight,
            "pooling": core.trace_graph_encoder.pooling[1].weight,
            "graph_comparator": core.graph_comparator[-1].weight,
            "address": core.read_key_network[1].weight,
            "value": core.writer[-1].weight,
            "scorer": core.score_network.second.weight,
        }
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all().item()))
                self.assertGreater(float(parameter.grad.abs().sum().item()), 0.0)

    def test_outcome_shuffle_preserves_graph_address_keys_allocation_and_sidecars(self) -> None:
        core = self._core()
        true_state = core.initial_memory_state()
        shuffled_state = core.initial_memory_state()
        true_outcomes = (1.0, -1.0, 1.0, -1.0)
        shuffled_outcomes = (-1.0, 1.0, -1.0, 1.0)

        for index, (true_outcome, shuffled_outcome) in enumerate(
            zip(true_outcomes, shuffled_outcomes)
        ):
            common, steps, step_mask = self._inputs(
                seed=217 + index,
                outcomes=torch.tensor([[true_outcome]]),
            )
            shuffled = (*common[:3], torch.tensor([[shuffled_outcome]]), *common[4:])
            true_output = self._forward(core, common, steps, step_mask, true_state)
            shuffled_output = self._forward(
                core,
                shuffled,
                steps,
                step_mask,
                shuffled_state,
            )
            for name in (
                "read_keys",
                "write_keys",
                "write_weights",
                "graph_address_logits",
            ):
                torch.testing.assert_close(
                    getattr(true_output, name),
                    getattr(shuffled_output, name),
                    rtol=0.0,
                    atol=1.0e-6,
                )
            true_state = core.apply_mixed_outcome_update(
                true_state,
                true_output,
                common[3],
                common[4],
            )
            shuffled_state = core.apply_mixed_outcome_update(
                shuffled_state,
                shuffled_output,
                shuffled[3],
                shuffled[4],
            )

        for name in ("keys", "usage", "graph_step_features", "graph_step_mask"):
            torch.testing.assert_close(
                getattr(true_state, name),
                getattr(shuffled_state, name),
                rtol=0.0,
                atol=1.0e-6,
            )
        self.assertGreater(
            float((true_state.values - shuffled_state.values).abs().max().item()),
            1.0e-6,
        )

    def test_neutral_feedback_keeps_values_and_residual_exactly_zero(self) -> None:
        core = self._core()
        state = core.initial_memory_state()
        neutral, steps, step_mask = self._inputs(
            seed=222,
            outcomes=torch.zeros(1, 1),
        )
        output = self._forward(core, neutral, steps, step_mask, state)
        self.assertTrue(torch.equal(output.write_values, torch.zeros_like(output.write_values)))
        state = core.apply_mixed_outcome_update(
            state,
            output,
            neutral[3],
            neutral[4],
        )
        self.assertTrue(torch.equal(state.values, torch.zeros_like(state.values)))

        challenge, challenge_steps, challenge_mask = self._inputs(
            seed=223,
            outcomes=torch.zeros(1, 3),
            candidates=3,
        )
        held = self._forward(
            core,
            challenge,
            challenge_steps,
            challenge_mask,
            state,
        )
        self.assertTrue(torch.equal(held.residuals, torch.zeros_like(held.residuals)))
        self.assertTrue(torch.equal(held.scores, held.normalized_base_scores))


if __name__ == "__main__":
    unittest.main()
