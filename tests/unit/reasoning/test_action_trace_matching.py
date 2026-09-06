from __future__ import annotations

import unittest

import torch

from angler.reasoning import (
    ActionTraceMatcherDecoder,
    ActionTraceProceduralCore,
    procedural_core_config,
)


class ActionTraceMatchingTests(unittest.TestCase):
    def _core(self):
        return ActionTraceProceduralCore(
            procedural_core_config("compact", content_width=32, temporal_width=7),
            maximum_trace_steps=4,
        )

    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(51)
        return (
            torch.randn(1, 32, generator=generator),
            torch.randn(1, 5, 32, generator=generator),
            torch.randn(1, 5, 7, generator=generator),
            torch.ones(1, 5, dtype=torch.bool),
        )

    def test_each_action_contributes_to_fixed_capacity_memory(self) -> None:
        torch.manual_seed(52)
        core = self._core()
        state = core.initial_plastic_state()
        output = core(*self._inputs(), plastic_state=state)
        trace = torch.randn(1, 4, 32)
        mask = torch.tensor([[True, True, True, False]])
        updated = core.apply_action_trace_feedback(state, output, trace, mask, torch.ones(1))
        reverse = core.apply_action_trace_feedback(
            state,
            output,
            trace.flip(1),
            torch.tensor([[True, True, True, True]]),
            torch.ones(1),
        )
        self.assertEqual(updated.step, 1)
        self.assertEqual(updated.bytes, state.bytes)
        self.assertGreater(float((updated.values - state.values).abs().sum()), 0.0)
        self.assertFalse(torch.equal(updated.values, reverse.values))

    def test_matcher_is_parent_exact_at_initialization_then_learns_residual(self) -> None:
        torch.manual_seed(53)
        decoder = ActionTraceMatcherDecoder(
            content_width=32,
            procedure_width=256,
            hidden_width=256,
            heads=8,
            maximum_actions=6,
            maximum_steps=4,
        )
        slots = torch.randn(1, 8, 256)
        actions = torch.randn(1, 6, 32)
        action_mask = torch.ones(1, 6, dtype=torch.bool)
        memory = torch.randn(16, 256)
        memory_mask = torch.ones(16, dtype=torch.bool)
        baseline = decoder(
            slots,
            actions,
            action_mask,
            memory_values=memory,
            memory_mask=memory_mask,
            correspondence=False,
        )
        matched = decoder(
            slots,
            actions,
            action_mask,
            memory_values=memory,
            memory_mask=memory_mask,
        )
        self.assertTrue(torch.equal(baseline.logits, matched.logits))
        matched.logits.square().mean().backward()
        self.assertGreater(float(decoder.matcher_update[-1].weight.grad.abs().sum()), 0.0)

    def test_memory_correspondence_changes_after_one_matcher_update(self) -> None:
        torch.manual_seed(54)
        decoder = ActionTraceMatcherDecoder(
            content_width=32,
            procedure_width=256,
            hidden_width=256,
            heads=8,
            maximum_actions=6,
            maximum_steps=4,
        )
        with torch.no_grad():
            decoder.matcher_update[-1].weight.normal_(std=0.01)
        slots = torch.randn(1, 8, 256)
        actions = torch.randn(1, 6, 32)
        mask = torch.ones(1, 6, dtype=torch.bool)
        first = decoder(
            slots,
            actions,
            mask,
            memory_values=torch.randn(16, 256),
            memory_mask=torch.ones(16, dtype=torch.bool),
        )
        second = decoder(
            slots,
            actions,
            mask,
            memory_values=torch.randn(16, 256),
            memory_mask=torch.ones(16, dtype=torch.bool),
        )
        self.assertFalse(torch.equal(first.logits, second.logits))


if __name__ == "__main__":
    unittest.main()
