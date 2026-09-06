from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from angler.reasoning.oml_natural_trace_representation import (
    OMLNaturalTraceRepresentation,
    representation_metrics,
)
from experiments.runners.phase6_cross_variation_plasticity_v16 import (
    AdamWSlot,
    functional_adamw_step,
)


def _episodes(
    *,
    mechanisms: int = 2,
    steps: int = 5,
    width: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20_260_831)
    features = torch.randn(
        mechanisms,
        6,
        steps,
        width,
        generator=generator,
    )
    mask = torch.ones(mechanisms, 6, steps, dtype=torch.bool)
    patterns = torch.tensor(
        (
            (-1, 1, -1, 1, 1, -1),
            (1, -1, 1, -1, -1, 1),
        ),
        dtype=features.dtype,
    )
    outcomes = torch.stack(
        tuple(patterns[index % len(patterns)] for index in range(mechanisms))
    )
    return features, mask, outcomes


def _fast_state(weight: torch.Tensor) -> tuple[AdamWSlot, ...]:
    zero = torch.zeros_like(weight)
    return (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)


def _updated_fast(
    model: OMLNaturalTraceRepresentation,
    features: torch.Tensor,
    mask: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], torch.Tensor]:
    fast = model.fresh_fast_weight()
    loss = model.functional_loss(features, mask, outcomes, fast)
    gradient = torch.autograd.grad(
        loss,
        (fast,),
        create_graph=second_order,
        retain_graph=second_order,
    )[0]
    used = gradient if second_order else gradient.detach()
    updated, state = functional_adamw_step(
        (fast,),
        (used,),
        _fast_state(fast),
        (1.0e-3,),
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        weight_decay=0.0,
    )
    return updated[0], state, fast


class OMLNaturalTraceRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20_260_831)
        self.model = OMLNaturalTraceRepresentation(step_width=12)
        self.features, self.mask, self.outcomes = _episodes()

    def test_shapes_balance_and_parameter_partition(self) -> None:
        codes = self.model.encode(self.features, self.mask)
        hidden = self.model.rln_features(self.features, self.mask)
        logits = self.model.functional_logits(self.features, self.mask)
        loss = self.model.functional_loss(
            self.features,
            self.mask,
            self.outcomes,
        )
        self.assertEqual(codes.shape, (2, 6, 32))
        self.assertEqual(hidden.shape, (2, 6, 64))
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(loss.shape, ())
        report = self.model.parameter_partition_report()
        self.assertEqual(report["fast_initial_shape"], (1, 64))
        self.assertEqual(report["fast_initial_count"], 64)
        self.assertFalse(report["fast_initial_outer_owned"])
        self.assertNotIn(
            "fast_initial_weight",
            dict(self.model.named_parameters()),
        )
        self.assertEqual(
            set(report["rln_parameter_names"]),
            set(dict(self.model.named_parameters())),
        )
        invalid = self.outcomes.clone()
        invalid[0, 0] = 1
        with self.assertRaises(ValueError):
            self.model.functional_loss(self.features, self.mask, invalid)

    def test_padding_repeat_and_reorder_behavior(self) -> None:
        code = self.model.encode(self.features, self.mask)
        repeat = self.model.encode(self.features.clone(), self.mask.clone())
        self.assertTrue(torch.equal(code, repeat))

        padded = torch.cat(
            (self.features, torch.randn(2, 6, 3, 12)),
            dim=2,
        )
        padded_mask = torch.cat(
            (self.mask, torch.zeros(2, 6, 3, dtype=torch.bool)),
            dim=2,
        )
        padded_code = self.model.encode(padded, padded_mask)
        torch.testing.assert_close(code, padded_code, rtol=0.0, atol=0.0)

        reordered = self.features.clone()
        reordered[:, :, [1, 2]] = reordered[:, :, [2, 1]]
        reordered_code = self.model.encode(reordered, self.mask)
        self.assertGreater(float((code - reordered_code).abs().max().item()), 0.0)

    def test_fast_functional_linear_parity(self) -> None:
        hidden = self.model.rln_features(self.features, self.mask)
        fast = self.model.fresh_fast_weight()
        linear = nn.Linear(64, 1, bias=False)
        with torch.no_grad():
            linear.weight.copy_(fast)
        expected = linear(hidden).squeeze(-1)
        actual = self.model.functional_logits(
            self.features,
            self.mask,
            fast,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_second_order_inner_path_reaches_rln(self) -> None:
        updated, _, _ = _updated_fast(
            self.model,
            self.features,
            self.mask,
            self.outcomes,
            second_order=True,
        )
        gradients = torch.autograd.grad(
            updated.square().sum(),
            tuple(self.model.parameters()),
            allow_unused=True,
        )
        self.assertTrue(
            any(
                value is not None and float(value.abs().sum().item()) > 0.0
                for value in gradients
            )
        )

    def test_first_order_detaches_hessian_but_preserves_fast_identity(self) -> None:
        updated, _, fast = _updated_fast(
            self.model,
            self.features,
            self.mask,
            self.outcomes,
            second_order=False,
        )
        rln_gradients = torch.autograd.grad(
            updated.sum(),
            tuple(self.model.parameters()),
            allow_unused=True,
            retain_graph=True,
        )
        self.assertTrue(all(value is None for value in rln_gradients))
        identity = torch.autograd.grad(updated.sum(), fast)[0]
        self.assertGreater(float(identity.abs().sum().item()), 0.0)

    def test_inner_update_changes_only_functional_head_and_moments(self) -> None:
        before = copy.deepcopy(self.model.state_dict())
        updated, state, _ = _updated_fast(
            self.model,
            self.features,
            self.mask,
            self.outcomes,
            second_order=True,
        )
        self.assertFalse(torch.equal(updated.detach(), self.model.fast_initial_weight))
        self.assertEqual(state[0].step, 1)
        self.assertGreater(float(state[0].exp_avg.abs().sum().item()), 0.0)
        self.assertGreater(float(state[0].exp_avg_sq.abs().sum().item()), 0.0)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_outer_optimizer_owns_only_rln(self) -> None:
        fixed_before = self.model.fast_initial_weight.detach().clone()
        parameters_before = {
            name: value.detach().clone()
            for name, value in self.model.named_parameters()
        }
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3.0e-4,
            weight_decay=0.0,
        )
        loss = self.model.functional_loss(
            self.features,
            self.mask,
            self.outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(self.model.fast_initial_weight, fixed_before))
        self.assertTrue(
            any(
                not torch.equal(value.detach(), parameters_before[name])
                for name, value in self.model.named_parameters()
            )
        )

    def test_direction_and_semantics_are_causal_removal_controls(self) -> None:
        full = self.model.encode(self.features, self.mask)
        direction_removed = self.model.encode(
            self.features,
            self.mask,
            include_direction=False,
        )
        semantics_removed = self.model.encode(
            self.features,
            self.mask,
            include_step_semantics=False,
        )
        self.assertGreater(float((full - direction_removed).abs().max().item()), 0.0)
        self.assertGreater(float((full - semantics_removed).abs().max().item()), 0.0)
        unrelated = torch.randn_like(self.features)
        unrelated_removed = self.model.encode(
            unrelated,
            self.mask,
            include_step_semantics=False,
        )
        torch.testing.assert_close(
            semantics_removed,
            unrelated_removed,
            rtol=0.0,
            atol=0.0,
        )

    def test_representation_metrics_are_finite_and_detect_collapse(self) -> None:
        codes = self.model.encode(self.features, self.mask)
        metrics = representation_metrics(codes)
        self.assertTrue(metrics["all_finite"])
        self.assertGreater(metrics["effective_rank"], 1.0)
        self.assertGreater(metrics["finite_nonzero_variance_dimensions"], 0)
        collapsed = representation_metrics(torch.ones(12, 32))
        self.assertEqual(collapsed["effective_rank"], 0.0)
        self.assertAlmostEqual(
            collapsed["mean_off_diagonal_cosine"],
            1.0,
            places=12,
        )

    def test_outcomes_never_enter_encoder(self) -> None:
        before = self.model.encode(self.features, self.mask)
        flipped = -self.outcomes
        loss_true = self.model.functional_loss(
            self.features,
            self.mask,
            self.outcomes,
        )
        loss_flipped = self.model.functional_loss(
            self.features,
            self.mask,
            flipped,
        )
        after = self.model.encode(self.features, self.mask)
        self.assertTrue(torch.equal(before, after))
        self.assertNotEqual(float(loss_true.item()), float(loss_flipped.item()))
        self.assertFalse(self.model.parameter_partition_report()["outcome_encoder_inputs"])


if __name__ == "__main__":
    unittest.main()
