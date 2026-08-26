from __future__ import annotations

import unittest

import torch

from angler.reasoning import (
    ReasoningCoreConfig,
    RecurrentReasoningCore,
    reasoning_state_digest,
    restore_reasoning_state,
    snapshot_reasoning_state,
)


class RecurrentReasoningCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(71)
        self.config = ReasoningCoreConfig(
            knowledge_width=12,
            core_width=16,
            workspace_slots=4,
            attention_heads=4,
            feedforward_width=32,
            reasoning_steps=3,
            maximum_reasoning_steps=4,
            maximum_entities=5,
        )
        self.core = RecurrentReasoningCore(self.config)
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

    def test_sampled_policy_is_a_masked_permutation_and_receives_gradients(self) -> None:
        trajectory = self.core.act(*self.inputs(), samples_per_task=3)

        self.assertEqual(trajectory.order_indices.shape, (2, 3, 4))
        self.assertEqual(trajectory.log_probability.shape, (2, 3))
        self.assertEqual(trajectory.entropy.shape, (2, 3))
        self.assertEqual(trajectory.value.shape, (2,))
        for sample in trajectory.order_indices[0]:
            self.assertEqual(set(sample.tolist()), {0, 1, 2, 3})
        for sample in trajectory.order_indices[1]:
            self.assertEqual(set(sample[:3].tolist()), {0, 1, 2})
            self.assertEqual(int(sample[3].item()), -1)

        rewards = torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]])
        advantages = rewards - trajectory.value.detach().unsqueeze(1)
        policy_loss = -(advantages * trajectory.log_probability).mean()
        value_loss = 0.5 * (
            trajectory.value - rewards.mean(dim=1)
        ).square().mean()
        (policy_loss + value_loss - 0.01 * trajectory.entropy.mean()).backward()
        gradients = [
            parameter.grad
            for parameter in self.core.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
        )
        self.assertIsNone(self.fact_features.grad)
        self.assertIsNone(self.entity_features.grad)
        self.assertIsNone(self.mention_features.grad)
        self.assertIsNotNone(
            self.core.reasoning_step.edge_context[0].weight.grad
        )
        self.assertIsNotNone(
            self.core.reasoning_step.edge_to_fact.weight.grad
        )
        self.assertIsNotNone(
            self.core.reasoning_step.edge_to_entity.weight.grad
        )

    def test_public_encoded_decoder_preserves_ordinary_greedy_action(self) -> None:
        ordinary = self.core.act(*self.inputs(), greedy=True)
        entities, _ = self.core.encode(*self.inputs())
        extracted = self.core.act_encoded(
            entities,
            self.entity_mask,
            greedy=True,
        )

        self.assertTrue(
            torch.equal(ordinary.order_indices, extracted.order_indices)
        )
        self.assertTrue(
            torch.equal(ordinary.log_probability, extracted.log_probability)
        )
        self.assertTrue(torch.equal(ordinary.entropy, extracted.entropy))
        self.assertTrue(torch.equal(ordinary.value, extracted.value))

    def test_zero_one_and_multiple_steps_have_distinct_information_paths(self) -> None:
        entities_without_steps, slots_without_steps = self.core.encode(
            *self.inputs(),
            reasoning_steps=0,
        )
        changed_inputs = list(self.inputs())
        changed_inputs[0] = changed_inputs[0] + 100.0
        changed_inputs[4] = changed_inputs[4] - 100.0
        changed_zero_entities, changed_zero_slots = self.core.encode(
            *changed_inputs,
            reasoning_steps=0,
        )
        entities_with_one_step, slots_with_one_step = self.core.encode(
            *self.inputs(),
            reasoning_steps=1,
        )
        entities_with_multiple_steps, slots_with_multiple_steps = self.core.encode(
            *self.inputs(),
            reasoning_steps=3,
        )

        self.assertTrue(torch.equal(entities_without_steps, changed_zero_entities))
        self.assertTrue(torch.equal(slots_without_steps, changed_zero_slots))
        self.assertFalse(torch.equal(entities_without_steps, entities_with_one_step))
        self.assertFalse(
            torch.equal(entities_with_one_step, entities_with_multiple_steps)
        )
        self.assertFalse(torch.equal(slots_without_steps, slots_with_one_step))
        self.assertFalse(
            torch.equal(slots_with_one_step, slots_with_multiple_steps)
        )
        self.assertTrue(torch.equal(self.fact_features, self.fact_features.clone()))

    def test_public_incidence_changes_recurrent_processing(self) -> None:
        ordinary_entities, ordinary_slots = self.core.encode(
            *self.inputs(),
            reasoning_steps=3,
        )
        changed_inputs = list(self.inputs())
        changed_fact_indices = self.mention_fact_indices.clone()
        changed_fact_indices[0] = torch.tensor([1, 1, 2, 2, 0, 0])
        changed_inputs[6] = changed_fact_indices
        changed_entities, changed_slots = self.core.encode(
            *changed_inputs,
            reasoning_steps=3,
        )

        self.assertFalse(torch.equal(ordinary_entities, changed_entities))
        self.assertFalse(torch.equal(ordinary_slots, changed_slots))

    def test_temperature_is_bounded_and_does_not_change_greedy_argmax(self) -> None:
        cold = self.core.act(
            *self.inputs(),
            greedy=True,
            temperature=0.5,
        )
        warm = self.core.act(
            *self.inputs(),
            greedy=True,
            temperature=2.0,
        )
        self.assertTrue(torch.equal(cold.order_indices, warm.order_indices))
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.core.act(*self.inputs(), temperature=0.0)

    def test_snapshot_digest_mutation_and_exact_restore(self) -> None:
        parent_digest = reasoning_state_digest(self.core)
        snapshot = snapshot_reasoning_state(self.core)
        with torch.no_grad():
            next(self.core.parameters()).add_(0.25)

        self.assertNotEqual(reasoning_state_digest(self.core), parent_digest)
        restore_reasoning_state(self.core, snapshot)
        self.assertEqual(reasoning_state_digest(self.core), parent_digest)

    def test_mention_cannot_reference_a_padded_entity(self) -> None:
        invalid_indices = self.mention_entity_indices.clone()
        invalid_indices[1, 0] = 3
        with self.assertRaisesRegex(ValueError, "padded entity"):
            self.core.encode(
                self.fact_features,
                self.fact_mask,
                self.entity_features,
                self.entity_mask,
                self.mention_features,
                self.mention_mask,
                self.mention_fact_indices,
                invalid_indices,
            )

    def test_mention_cannot_reference_a_padded_fact(self) -> None:
        invalid_indices = self.mention_fact_indices.clone()
        invalid_indices[1, 0] = 2
        with self.assertRaisesRegex(ValueError, "padded fact"):
            self.core.encode(
                self.fact_features,
                self.fact_mask,
                self.entity_features,
                self.entity_mask,
                self.mention_features,
                self.mention_mask,
                invalid_indices,
                self.mention_entity_indices,
            )

    def test_incidence_indices_require_integer_identity_tensors(self) -> None:
        changed_inputs = list(self.inputs())
        changed_inputs[6] = self.mention_fact_indices.to(dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "mention_fact_indices.*torch.long"):
            self.core.encode(*changed_inputs)


if __name__ == "__main__":
    unittest.main()
