from __future__ import annotations

import unittest

import torch

from angler.reasoning import (
    TraceConditionedProceduralCore,
    procedural_core_config,
)


class TraceConditionedProceduralCoreTests(unittest.TestCase):
    def _core(self) -> TraceConditionedProceduralCore:
        return TraceConditionedProceduralCore(
            procedural_core_config("compact", content_width=32, temporal_width=7),
            maximum_trace_steps=4,
        )

    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(41)
        return (
            torch.randn(1, 32, generator=generator),
            torch.randn(1, 5, 32, generator=generator),
            torch.randn(1, 5, 7, generator=generator),
            torch.ones(1, 5, dtype=torch.bool),
        )

    @staticmethod
    def _trace():
        generator = torch.Generator().manual_seed(42)
        return (
            torch.randn(1, 4, 32, generator=generator),
            torch.tensor([[True, True, True, False]]),
        )

    def test_ordered_trace_changes_the_written_procedure(self) -> None:
        torch.manual_seed(43)
        core = self._core()
        state = core.initial_plastic_state()
        output = core(*self._inputs(), plastic_state=state)
        actions, mask = self._trace()
        forward = core.apply_procedural_feedback(
            state, output, actions, mask, torch.ones(1)
        )
        reversed_actions = actions.clone()
        reversed_actions[:, :3] = actions[:, :3].flip(1)
        reverse = core.apply_procedural_feedback(
            state, output, reversed_actions, mask, torch.ones(1)
        )
        self.assertEqual(forward.step, 1)
        self.assertEqual(forward.bytes, state.bytes)
        self.assertFalse(torch.equal(forward.keys, reverse.keys))
        self.assertFalse(torch.equal(forward.values, reverse.values))

    def test_later_loss_reaches_trace_encoder_and_writer(self) -> None:
        torch.manual_seed(44)
        core = self._core()
        state = core.initial_plastic_state()
        first = core(*self._inputs(), plastic_state=state)
        actions, mask = self._trace()
        state = core.apply_procedural_feedback(
            state,
            first,
            actions,
            mask,
            torch.ones(1),
            detach_state=False,
        )
        later = core(*self._inputs(), plastic_state=state)
        later.procedure_slots.square().mean().backward()
        for parameter in (
            core.trace_projection[1].weight,
            core.trace_recurrent.weight_ih_l0,
            core.trace_write_key[0].weight,
            core.trace_write_value[0].weight,
            core.trace_write_gate[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_trace_validation_rejects_gaps_and_wrong_width(self) -> None:
        core = self._core()
        actions, _ = self._trace()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            core.encode_procedure_trace(
                actions,
                torch.tensor([[True, False, True, False]]),
            )
        with self.assertRaisesRegex(ValueError, "maximum_trace_steps"):
            core.encode_procedure_trace(
                actions[:, :3],
                torch.ones(1, 3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
