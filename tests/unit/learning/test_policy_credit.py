from __future__ import annotations

import unittest

import torch

from angler.learning.policy_credit import leave_one_out_advantages


class LeaveOneOutCreditTests(unittest.TestCase):
    def test_each_action_is_compared_only_with_other_actions(self) -> None:
        rewards = torch.tensor([0.0, 1.0, 2.0, 5.0])

        advantages = leave_one_out_advantages(rewards)

        expected = torch.tensor([-8.0 / 3.0, -4.0 / 3.0, 0.0, 4.0])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertAlmostEqual(float(advantages.sum().item()), 0.0, places=6)

    def test_credit_is_invariant_to_task_reward_offset(self) -> None:
        rewards = torch.tensor(
            [[0.0, 0.5, 1.0], [10.0, 10.5, 11.0]],
            dtype=torch.float64,
        )

        advantages = leave_one_out_advantages(rewards)

        self.assertTrue(torch.allclose(advantages[0], advantages[1]))

    def test_constant_outcomes_have_zero_policy_credit(self) -> None:
        rewards = torch.full((2, 8), 0.75)

        advantages = leave_one_out_advantages(rewards)

        self.assertTrue(torch.equal(advantages, torch.zeros_like(rewards)))

    def test_invalid_feedback_is_rejected(self) -> None:
        invalid = (
            torch.tensor([1.0]),
            torch.tensor([1, 2]),
            torch.tensor([1.0, float("nan")]),
            torch.tensor([1.0, 2.0], requires_grad=True),
        )
        for rewards in invalid:
            with self.subTest(rewards=rewards):
                with self.assertRaises(ValueError):
                    leave_one_out_advantages(rewards)


if __name__ == "__main__":
    unittest.main()
