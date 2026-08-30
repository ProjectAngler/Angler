from __future__ import annotations

import unittest

import torch

from angler.reasoning import (
    ScalableProceduralCore,
    plastic_state_digest,
    procedural_core_config,
    select_procedural_core_tier,
)


class ScalableProceduralCoreTests(unittest.TestCase):
    def _core(self):
        config = procedural_core_config("compact", content_width=32, temporal_width=7)
        return ScalableProceduralCore(config)

    def _inputs(self):
        torch.manual_seed(3)
        query = torch.randn(2, 32)
        candidates = torch.randn(2, 5, 32)
        temporal = torch.randn(2, 5, 7)
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        return query, candidates, temporal, mask

    def test_forward_masks_candidates_and_emits_qwen_width_prefix(self) -> None:
        torch.manual_seed(4)
        core = self._core()
        output = core(*self._inputs())
        self.assertEqual(output.procedure_slots.shape, (2, 8, 256))
        self.assertEqual(output.qwen_prefix.shape, (2, 8, 32))
        self.assertEqual(output.candidate_weights.shape, (2, 5))
        self.assertTrue(torch.equal(output.candidate_weights[0, 3:], torch.zeros(2)))
        self.assertTrue(torch.isfinite(output.qwen_prefix).all())
        output.qwen_prefix.square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in core.parameters()
        ))

    def test_feedback_changes_fixed_capacity_state_and_reset_restores_identity(self) -> None:
        torch.manual_seed(5)
        core = self._core()
        query, candidates, temporal, mask = self._inputs()
        initial = core.initial_plastic_state()
        initial_digest = plastic_state_digest(initial)
        output = core(query[:1], candidates[:1], temporal[:1], mask[:1], plastic_state=initial)
        updated = core.apply_feedback(initial, output, torch.tensor([1.0]))

        self.assertEqual(updated.step, 1)
        self.assertEqual(updated.keys.shape, initial.keys.shape)
        self.assertEqual(updated.bytes, initial.bytes)
        self.assertNotEqual(plastic_state_digest(updated), initial_digest)
        with torch.inference_mode():
            live = core(query[:1], candidates[:1], temporal[:1], mask[:1], plastic_state=updated)
            reset = core(query[:1], candidates[:1], temporal[:1], mask[:1], plastic_state=initial)
        self.assertFalse(torch.equal(live.procedure_slots, reset.procedure_slots))
        self.assertEqual(plastic_state_digest(initial.detached_clone()), initial_digest)

    def test_tiers_scale_all_core_dimensions_and_parameters(self) -> None:
        configs = [
            procedural_core_config(name, content_width=32, temporal_width=7)
            for name in ("compact", "workstation", "dedicated")
        ]
        counts = [ScalableProceduralCore(config).parameter_count for config in configs]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual([config.model_width for config in configs], [256, 512, 768])
        self.assertEqual([config.procedure_tokens for config in configs], [8, 16, 32])
        self.assertEqual([config.plastic_slots for config in configs], [16, 64, 128])

    def test_resource_selection_uses_declared_headroom(self) -> None:
        compact = select_procedural_core_tier(
            200_000_000,
            content_width=32,
            temporal_width=7,
            training=True,
        )
        large = select_procedural_core_tier(
            8_000_000_000,
            content_width=32,
            temporal_width=7,
            training=False,
        )
        self.assertEqual(compact.tier, "compact")
        self.assertEqual(large.tier, "dedicated")


if __name__ == "__main__":
    unittest.main()
