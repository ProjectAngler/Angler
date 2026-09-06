from __future__ import annotations

import copy
import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from angler.reasoning.structure_protected_oml import StructureProtectedOMLCore
from experiments.runners.phase6_cross_variation_plasticity_v16 import (
    AdamWSlot,
    functional_adamw_step,
)


def _paired_rows(
    *,
    mechanisms: int = 2,
    reference_steps: int = 5,
    attempt_steps: int = 5,
    width: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20_260_831_13)
    reference = torch.randn(
        mechanisms, 6, reference_steps, width, generator=generator
    )
    attempt = torch.randn(
        mechanisms, 6, attempt_steps, width, generator=generator
    )
    reference_mask = torch.ones(mechanisms, 6, reference_steps, dtype=torch.bool)
    attempt_mask = torch.ones(mechanisms, 6, attempt_steps, dtype=torch.bool)
    patterns = torch.tensor(
        ((-1, 1, -1, 1, 1, -1), (1, -1, 1, -1, -1, 1)),
        dtype=reference.dtype,
    )
    outcomes = torch.stack(
        tuple(patterns[index % len(patterns)] for index in range(mechanisms))
    )
    return reference, reference_mask, attempt, attempt_mask, outcomes


def _fast_state(weight: torch.Tensor) -> tuple[AdamWSlot, ...]:
    zero = torch.zeros_like(weight)
    return (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)


def _inner_update(
    model: StructureProtectedOMLCore,
    rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], torch.Tensor, torch.Tensor]:
    reference, reference_mask, attempt, attempt_mask, outcomes = rows
    fast = model.fresh_fast_weight()
    loss = model.functional_loss(
        reference, reference_mask, attempt, attempt_mask, outcomes, fast
    )
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
    return updated[0], state, fast, gradient


class StructureProtectedOMLCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20_260_831_13)
        self.model = StructureProtectedOMLCore(step_width=12)
        self.rows = _paired_rows()

    def test_shapes_functional_parity_and_balanced_loss(self) -> None:
        reference, reference_mask, attempt, attempt_mask, outcomes = self.rows
        reference_codes, attempt_codes = self.model.encode_views(
            reference, reference_mask, attempt, attempt_mask
        )
        hidden = self.model.relation_features(
            reference, reference_mask, attempt, attempt_mask
        )
        fast = self.model.fresh_fast_weight()
        logits = self.model.functional_logits(
            reference, reference_mask, attempt, attempt_mask, fast
        )
        loss = self.model.functional_loss(
            reference, reference_mask, attempt, attempt_mask, outcomes, fast
        )
        self.assertEqual(reference_codes.shape, (2, 6, 32))
        self.assertEqual(attempt_codes.shape, (2, 6, 32))
        self.assertEqual(hidden.shape, (2, 6, 64))
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(loss.shape, ())
        linear = nn.Linear(64, 1, bias=False)
        with torch.no_grad():
            linear.weight.copy_(fast)
        torch.testing.assert_close(
            logits, linear(hidden).squeeze(-1), rtol=0.0, atol=0.0
        )
        invalid = outcomes.clone()
        invalid[0, 0] = 1
        with self.assertRaises(ValueError):
            self.model.functional_loss(
                reference, reference_mask, attempt, attempt_mask, invalid, fast
            )

    def test_padding_repeat_and_order_sensitivity(self) -> None:
        reference, reference_mask, attempt, attempt_mask, _ = self.rows
        original = self.model.encode_views(
            reference, reference_mask, attempt, attempt_mask
        )
        repeated = self.model.encode_views(
            reference.clone(), reference_mask.clone(), attempt.clone(), attempt_mask.clone()
        )
        for left, right in zip(original, repeated, strict=True):
            self.assertTrue(torch.equal(left, right))

        padded_reference = torch.cat(
            (reference, torch.randn(2, 6, 2, 12)), dim=2
        )
        padded_attempt = torch.cat((attempt, torch.randn(2, 6, 3, 12)), dim=2)
        padded_reference_mask = torch.cat(
            (reference_mask, torch.zeros(2, 6, 2, dtype=torch.bool)), dim=2
        )
        padded_attempt_mask = torch.cat(
            (attempt_mask, torch.zeros(2, 6, 3, dtype=torch.bool)), dim=2
        )
        padded = self.model.encode_views(
            padded_reference,
            padded_reference_mask,
            padded_attempt,
            padded_attempt_mask,
        )
        for left, right in zip(original, padded, strict=True):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

        reordered = attempt.clone()
        reordered[:, :, [1, 2]] = reordered[:, :, [2, 1]]
        reordered_code = self.model.encode_views(
            reference, reference_mask, reordered, attempt_mask
        )[1]
        self.assertGreater(
            float((original[1] - reordered_code).abs().max().item()), 0.0
        )

    def test_shared_encoder_and_pair_roles_are_learned_and_sensitive(self) -> None:
        reference, reference_mask, attempt, attempt_mask, _ = self.rows
        reference_code, attempt_code = self.model.encode_views(
            reference, reference_mask, attempt, attempt_mask
        )
        reference_direct = self.model.trace_graph_encoder(reference, reference_mask)
        attempt_direct = self.model.trace_graph_encoder(attempt, attempt_mask)
        torch.testing.assert_close(reference_code, reference_direct, rtol=0.0, atol=0.0)
        torch.testing.assert_close(attempt_code, attempt_direct, rtol=0.0, atol=0.0)

        forward = self.model.relation_features(
            reference, reference_mask, attempt, attempt_mask
        )
        reversed_roles = self.model.relation_features(
            attempt, attempt_mask, reference, reference_mask
        )
        self.assertGreater(float((forward - reversed_roles).abs().max().item()), 0.0)
        self.assertFalse(hasattr(self.model, "reference_encoder"))
        self.assertFalse(hasattr(self.model, "attempt_encoder"))

    def test_second_order_connects_inner_update_to_trunk(self) -> None:
        updated, _, _, _ = _inner_update(
            self.model, self.rows, second_order=True
        )
        trunk = tuple(value for _, value in self.model.trunk_named_parameters())
        gradients = torch.autograd.grad(updated.square().sum(), trunk, allow_unused=True)
        self.assertTrue(
            any(
                value is not None and float(value.abs().sum().item()) > 0.0
                for value in gradients
            )
        )

    def test_detached_first_order_removes_hessian_but_keeps_identity(self) -> None:
        updated, _, fast, _ = _inner_update(
            self.model, self.rows, second_order=False
        )
        trunk = tuple(value for _, value in self.model.trunk_named_parameters())
        gradients = torch.autograd.grad(
            updated.sum(), trunk, allow_unused=True, retain_graph=True
        )
        self.assertTrue(all(value is None for value in gradients))
        identity = torch.autograd.grad(updated.sum(), fast)[0]
        self.assertGreater(float(identity.abs().sum().item()), 0.0)

    def test_first_and_second_inner_updates_are_numerically_paired(self) -> None:
        second, second_state, _, second_gradient = _inner_update(
            self.model, self.rows, second_order=True
        )
        first, first_state, _, first_gradient = _inner_update(
            self.model, self.rows, second_order=False
        )
        torch.testing.assert_close(second, first, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second_gradient, first_gradient, rtol=0.0, atol=0.0)
        self.assertEqual(second_state[0].step, first_state[0].step)
        torch.testing.assert_close(
            second_state[0].exp_avg, first_state[0].exp_avg, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            second_state[0].exp_avg_sq,
            first_state[0].exp_avg_sq,
            rtol=0.0,
            atol=0.0,
        )

    def test_parameter_owners_are_disjoint_exhaustive_and_do_not_cross_mutate(self) -> None:
        report = self.model.parameter_partition_report()
        encoder = self.model.encoder_named_parameters()
        trunk = self.model.trunk_named_parameters()
        self.assertTrue(report["partitions_disjoint"])
        self.assertTrue(report["all_trainable_parameters_owned"])
        self.assertFalse(report["fast_initial_outer_owned"])
        self.assertEqual(set(report["encoder_parameter_names"]), {n for n, _ in encoder})
        self.assertEqual(set(report["trunk_parameter_names"]), {n for n, _ in trunk})
        self.assertFalse({id(v) for _, v in encoder} & {id(v) for _, v in trunk})
        self.assertEqual(
            {n for n, _ in (*encoder, *trunk)}, set(dict(self.model.named_parameters()))
        )

        reference, reference_mask, attempt, attempt_mask, outcomes = self.rows
        fixed_before = self.model.fast_initial_weight.detach().clone()
        encoder_before = {name: value.detach().clone() for name, value in encoder}
        trunk_optimizer = torch.optim.AdamW(
            tuple(value for _, value in trunk), lr=3.0e-4, weight_decay=0.0
        )
        loss = self.model.functional_loss(
            reference, reference_mask, attempt, attempt_mask, outcomes
        )
        trunk_gradients = torch.autograd.grad(loss, tuple(value for _, value in trunk))
        trunk_optimizer.zero_grad(set_to_none=True)
        for (_, parameter), gradient in zip(trunk, trunk_gradients, strict=True):
            parameter.grad = gradient
        trunk_optimizer.step()
        for name, value in self.model.encoder_named_parameters():
            self.assertTrue(torch.equal(value, encoder_before[name]), name)
        self.assertTrue(torch.equal(self.model.fast_initial_weight, fixed_before))

    def test_direct_contrastive_codes_reach_encoder_only(self) -> None:
        reference, reference_mask, attempt, attempt_mask, _ = self.rows
        reference_code, attempt_code = self.model.encode_views(
            reference, reference_mask, attempt, attempt_mask
        )
        similarity = (
            F.normalize(reference_code.reshape(-1, 32), dim=-1)
            * F.normalize(attempt_code.reshape(-1, 32), dim=-1)
        ).sum(dim=-1).mean()
        encoder = tuple(value for _, value in self.model.encoder_named_parameters())
        trunk = tuple(value for _, value in self.model.trunk_named_parameters())
        encoder_gradients = torch.autograd.grad(
            similarity, encoder, retain_graph=True, allow_unused=True
        )
        trunk_gradients = torch.autograd.grad(
            similarity, trunk, allow_unused=True
        )
        self.assertTrue(
            any(
                value is not None and float(value.abs().sum().item()) > 0.0
                for value in encoder_gradients
            )
        )
        self.assertTrue(all(value is None for value in trunk_gradients))

    def test_outcomes_and_metadata_cannot_enter_encoder_api(self) -> None:
        parameters = inspect.signature(self.model.encode_views).parameters
        self.assertNotIn("outcomes", parameters)
        self.assertNotIn("family_id", parameters)
        self.assertNotIn("transition", parameters)
        report = self.model.parameter_partition_report()
        self.assertFalse(report["outcome_encoder_inputs"])
        self.assertFalse(report["metadata_encoder_inputs"])

    def test_semantics_removal_is_input_invariant_and_direction_is_causal(self) -> None:
        reference, reference_mask, attempt, attempt_mask, _ = self.rows
        full = self.model.encode_views(
            reference, reference_mask, attempt, attempt_mask
        )
        direction_removed = self.model.encode_views(
            reference,
            reference_mask,
            attempt,
            attempt_mask,
            include_direction=False,
        )
        semantics_removed = self.model.encode_views(
            reference,
            reference_mask,
            attempt,
            attempt_mask,
            include_step_semantics=False,
        )
        self.assertGreater(
            float((full[1] - direction_removed[1]).abs().max().item()), 0.0
        )
        self.assertGreater(
            float((full[1] - semantics_removed[1]).abs().max().item()), 0.0
        )
        unrelated = self.model.encode_views(
            torch.randn_like(reference),
            reference_mask,
            torch.randn_like(attempt),
            attempt_mask,
            include_step_semantics=False,
        )
        for left, right in zip(semantics_removed, unrelated, strict=True):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_inner_update_does_not_mutate_module_state(self) -> None:
        before = copy.deepcopy(self.model.state_dict())
        updated, state, _, _ = _inner_update(
            self.model, self.rows, second_order=True
        )
        self.assertFalse(torch.equal(updated.detach(), self.model.fast_initial_weight))
        self.assertEqual(state[0].step, 1)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()
