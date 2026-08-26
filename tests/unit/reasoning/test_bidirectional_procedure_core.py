from __future__ import annotations

import itertools
import unittest

import torch

from angler.reasoning.bidirectional_procedure_core import (
    BidirectionalProcedureConfig,
    BidirectionalProcedureCore,
    ProcedureLearningBatch,
    permutations_to_tensor,
    procedure_core_digest,
)


def _swap(state: tuple[int, ...], action: int) -> tuple[int, ...]:
    values = list(state)
    values[action], values[action + 1] = values[action + 1], values[action]
    return tuple(values)


def _execute(
    origin: tuple[int, ...],
    actions: tuple[int, ...],
) -> tuple[int, ...]:
    state = origin
    for action in actions:
        state = _swap(state, action)
    return state


def _complete_transition_batch(item_count: int) -> ProcedureLearningBatch:
    states: list[tuple[int, ...]] = []
    actions: list[int] = []
    next_states: list[tuple[int, ...]] = []
    for state in itertools.permutations(range(item_count)):
        for action in range(item_count - 1):
            states.append(state)
            actions.append(action)
            next_states.append(_swap(state, action))
    encoded_states = permutations_to_tensor(states)
    encoded_next = permutations_to_tensor(next_states)
    action_tensor = torch.tensor(actions, dtype=torch.long)
    horizons = torch.ones(len(states), dtype=torch.long)
    return ProcedureLearningBatch(
        states=encoded_states,
        actions=action_tensor,
        next_states=encoded_next,
        origins=encoded_states,
        goals=encoded_next,
        horizons=horizons,
        first_actions=action_tensor,
        last_actions=action_tensor,
        midpoints=encoded_next,
    )


class BidirectionalProcedureCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8127)
        torch.set_num_threads(1)

    def test_every_learning_head_receives_gradient(self) -> None:
        model = BidirectionalProcedureCore(
            BidirectionalProcedureConfig(
                item_count=4,
                hidden_width=32,
                action_width=12,
                maximum_horizon=6,
            )
        )
        losses = model.learning_losses(_complete_transition_batch(4))
        losses["total"].backward()
        expected = (
            "forward_dynamics",
            "backward_dynamics",
            "inverse_action",
            "forward_policy",
            "backward_policy",
            "distance",
            "midpoint",
        )
        for prefix in expected:
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(gradients, prefix)
            self.assertTrue(any(gradient is not None for gradient in gradients), prefix)
            self.assertTrue(
                all(
                    gradient is None or bool(torch.isfinite(gradient).all().item())
                    for gradient in gradients
                ),
                prefix,
            )

    def test_learned_dynamics_construct_and_external_execution_verifies(self) -> None:
        model = BidirectionalProcedureCore(
            BidirectionalProcedureConfig(
                item_count=4,
                hidden_width=64,
                action_width=16,
                maximum_horizon=6,
            )
        )
        batch = _complete_transition_batch(4)
        before = procedure_core_digest(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.0)
        for _ in range(450):
            optimizer.zero_grad(set_to_none=True)
            loss = model.learning_losses(batch)["total"]
            loss.backward()
            optimizer.step()
        self.assertNotEqual(procedure_core_digest(model), before)

        with torch.no_grad():
            forward = model.forward_state_logits(batch.states, batch.actions).argmax(-1)
            backward = model.backward_state_logits(
                batch.next_states,
                batch.actions,
            ).argmax(-1)
        self.assertTrue(torch.equal(forward, batch.next_states.argmax(-1)))
        self.assertTrue(torch.equal(backward, batch.states.argmax(-1)))

        origin = (3, 2, 1, 0)
        goal = (0, 1, 2, 3)
        plan = model.construct_procedure(
            origin,
            goal,
            maximum_steps=6,
            maximum_expansions=200,
        )
        self.assertTrue(plan.found, plan)
        self.assertTrue(plan.exact_frontier_join)
        self.assertLessEqual(len(plan.actions), 6)
        self.assertEqual(_execute(origin, plan.actions), goal)

    def test_goal_is_a_causal_choke_point_not_a_task_side_channel(self) -> None:
        model = BidirectionalProcedureCore(
            BidirectionalProcedureConfig(
                item_count=4,
                hidden_width=64,
                action_width=16,
                maximum_horizon=6,
            )
        )
        batch = _complete_transition_batch(4)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(400):
            optimizer.zero_grad(set_to_none=True)
            model.learning_losses(batch)["total"].backward()
            optimizer.step()

        origin = (0, 1, 2, 3)
        first_goal = (3, 2, 1, 0)
        swapped_goal = (1, 0, 3, 2)
        first = model.construct_procedure(
            origin,
            first_goal,
            maximum_steps=6,
            maximum_expansions=200,
        )
        swapped = model.construct_procedure(
            origin,
            swapped_goal,
            maximum_steps=6,
            maximum_expansions=200,
        )
        self.assertTrue(first.found)
        self.assertTrue(swapped.found)
        self.assertEqual(_execute(origin, first.actions), first_goal)
        self.assertEqual(_execute(origin, swapped.actions), swapped_goal)
        self.assertNotEqual(first.actions, swapped.actions)

    def test_invalid_state_and_ablation_permutations_fail_closed(self) -> None:
        model = BidirectionalProcedureCore(
            BidirectionalProcedureConfig(item_count=4, maximum_horizon=6)
        )
        with self.assertRaisesRegex(ValueError, "complete permutation"):
            model.construct_procedure(
                (0, 0, 2, 3),
                (0, 1, 2, 3),
                maximum_steps=4,
                maximum_expansions=10,
            )
        with self.assertRaisesRegex(ValueError, "must be a permutation"):
            model.construct_procedure(
                (0, 1, 2, 3),
                (3, 2, 1, 0),
                maximum_steps=4,
                maximum_expansions=10,
                backward_action_permutation=(0, 0, 2),
            )
        with self.assertRaisesRegex(ValueError, "must be a permutation"):
            model.construct_procedure(
                (0, 1, 2, 3),
                (3, 2, 1, 0),
                maximum_steps=4,
                maximum_expansions=10,
                policy_action_permutation=(0, 0, 2),
            )


if __name__ == "__main__":
    unittest.main()
