from __future__ import annotations

from dataclasses import fields
import copy
import unittest

import torch

from angler.reasoning import (
    AdaptiveFeedbackContext,
    AdaptiveReasoningCore,
    AdaptiveReasoningTrajectory,
    ReasoningCoreConfig,
    self_referential_state_digest,
    snapshot_self_referential_state,
)


class AdaptiveReasoningCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9107)
        self.config = ReasoningCoreConfig(
            knowledge_width=12,
            core_width=16,
            workspace_slots=4,
            attention_heads=4,
            feedforward_width=32,
            reasoning_steps=2,
            maximum_reasoning_steps=4,
            maximum_entities=5,
        )
        self.model = AdaptiveReasoningCore(self.config)
        self.fact_features = torch.randn(2, 3, 12)
        self.fact_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        self.entity_features = torch.randn(2, 4, 12)
        self.entity_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        self.mention_features = torch.randn(2, 6, 12)
        self.mention_mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, False, False],
            ]
        )
        self.mention_fact_indices = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 0, 0]],
            dtype=torch.long,
        )
        self.mention_entity_indices = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [0, 1, 1, 2, 0, 0]],
            dtype=torch.long,
        )

    def inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.fact_features,
            self.fact_mask,
            self.entity_features,
            self.entity_mask,
            self.mention_features,
            self.mention_mask,
            self.mention_fact_indices,
            self.mention_entity_indices,
        )

    def fresh_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.fact_features + 0.37,
            self.fact_mask,
            self.entity_features.roll(1, dims=1) - 0.19,
            self.entity_mask,
            self.mention_features * 0.83 + 0.11,
            self.mention_mask,
            self.mention_fact_indices,
            self.mention_entity_indices,
        )

    def feedback(self) -> dict[str, torch.Tensor]:
        return {
            "reward": torch.tensor([1.0, 0.25]),
        }

    def test_action_is_read_only_and_feedback_is_the_only_write(self) -> None:
        state = self.model.initial_state(2)
        before = self_referential_state_digest(state)
        snapshot = snapshot_self_referential_state(state)
        trajectory = self.model.act(*self.inputs(), state=state, greedy=True)

        self.assertIsInstance(trajectory, AdaptiveReasoningTrajectory)
        self.assertEqual(trajectory.action.order_indices.shape, (2, 1, 4))
        self.assertEqual(trajectory.plastic_context.shape, (2, 16))
        self.assertEqual(self_referential_state_digest(state), before)
        for name, tensor in snapshot.items():
            self.assertTrue(torch.equal(getattr(state, name), tensor))

        slow_before = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        write = self.model.incorporate_feedback(
            trajectory.feedback_context,
            state=state,
            **self.feedback(),
        )
        self.assertNotEqual(self_referential_state_digest(write.state), before)
        self.assertGreater(float(write.delta_norm.item()), 0.0)
        self.assertEqual(write.event.shape, (2, 16))
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, slow_before[name]))

    def test_feedback_state_changes_later_policy_reset_erases_and_swap_transfers(self) -> None:
        state = self.model.initial_state(2)
        first = self.model.act(*self.inputs(), state=state, greedy=True)
        write = self.model.incorporate_feedback(
            first.feedback_context,
            state=state,
            **self.feedback(),
        )
        adapted = self.model.act(
            *self.fresh_inputs(),
            state=write.state,
            greedy=True,
        )
        reset = self.model.act(
            *self.fresh_inputs(),
            state=self.model.initial_state(2),
            greedy=True,
        )

        self.assertFalse(
            torch.equal(adapted.plastic_context, reset.plastic_context)
        )
        self.assertFalse(
            torch.equal(
                adapted.action.log_probability,
                reset.action.log_probability,
            )
        )

        receiver = copy.deepcopy(self.model)
        swapped = receiver.act(
            *self.fresh_inputs(),
            state=write.state,
            greedy=True,
        )
        self.assertTrue(
            torch.equal(swapped.plastic_context, adapted.plastic_context)
        )
        self.assertTrue(
            torch.equal(
                swapped.action.log_probability,
                adapted.action.log_probability,
            )
        )

    def test_later_query_loss_trains_earlier_feedback_update(self) -> None:
        model = AdaptiveReasoningCore(self.config).double()
        support_inputs = tuple(
            value.double() if value.is_floating_point() else value
            for value in self.inputs()
        )
        query_inputs = tuple(
            value.double() if value.is_floating_point() else value
            for value in self.fresh_inputs()
        )
        state = model.initial_state(2)
        support = model.act(*support_inputs, state=state, greedy=True)
        write = model.incorporate_feedback(
            support.feedback_context,
            state=state,
            **self.feedback(),
        )
        prescribed = torch.tensor(
            [[3, 2, 1, 0], [2, 1, 0, -1]],
            dtype=torch.long,
        )
        state_before_query = self_referential_state_digest(write.state)
        later = model.score_training_order(
            *query_inputs,
            prescribed,
            state=write.state,
        )
        self.assertTrue(torch.equal(later.order_indices[:, 0], prescribed))
        self.assertEqual(
            self_referential_state_digest(write.state),
            state_before_query,
        )
        outer_loss = -later.log_probability.mean()
        parameters = (
            model.memory.base_q,
            model.memory.base_k,
            model.memory.base_beta,
            model.feedback_encoder[-1].weight,
        )
        gradients = torch.autograd.grad(outer_loss, parameters)

        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all().item()))
            self.assertGreater(float(gradient.abs().sum().item()), 1e-12)

    def test_action_summary_identifies_different_attempted_orders(self) -> None:
        entities, _ = self.model.core.encode(*self.inputs())
        forward = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, -1]],
            dtype=torch.long,
        )
        reverse = torch.tensor(
            [[3, 2, 1, 0], [2, 1, 0, -1]],
            dtype=torch.long,
        )

        forward_summary = self.model._summarize_action(entities, forward)
        reverse_summary = self.model._summarize_action(entities, reverse)
        self.assertFalse(
            torch.allclose(
                forward_summary,
                reverse_summary,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_public_feedback_surface_has_no_routing_or_solution_channel(self) -> None:
        self.assertEqual(
            [field.name for field in fields(AdaptiveFeedbackContext)],
            ["observation_summary", "action_summary"],
        )
        self.assertEqual(
            tuple(self.feedback()),
            ("reward",),
        )

    def test_invalid_feedback_is_rejected_before_state_write(self) -> None:
        state = self.model.initial_state(2)
        trajectory = self.model.act(*self.inputs(), state=state, greedy=True)
        with self.assertRaisesRegex(ValueError, "reward.*zero and one"):
            self.model.incorporate_feedback(
                trajectory.feedback_context,
                reward=torch.tensor([1.1, 0.0]),
                state=state,
            )


if __name__ == "__main__":
    unittest.main()
