from __future__ import annotations

import unittest

import torch

from angler.reasoning import (
    OMLSituatedFeedbackPolicy,
    SituatedFeedbackPolicy,
    situated_outcome_loss,
)


class SituatedFeedbackPolicyTests(unittest.TestCase):
    def _inputs(self):
        query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        candidates = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
        temporal = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.tensor([[True, True]])
        base_scores = torch.tensor([[2.0, 1.0]], requires_grad=True)
        return query, candidates, temporal, mask, base_scores

    def test_zero_initialization_preserves_base_argmax_and_detaches_base(self) -> None:
        torch.manual_seed(11)
        policy = SituatedFeedbackPolicy(content_width=4, temporal_width=2, rank=4)
        values = self._inputs()
        output = policy(*values)

        self.assertEqual(output.scores.argmax(dim=1).item(), 0)
        self.assertTrue(torch.equal(output.residuals, torch.zeros_like(output.residuals)))
        loss = situated_outcome_loss(
            output,
            torch.tensor([0]),
            torch.tensor([-1.0]),
        )
        loss.backward()
        self.assertIsNone(values[-1].grad)

    def test_failure_suppresses_and_success_reinforces_selected_evidence(self) -> None:
        torch.manual_seed(12)
        policy = SituatedFeedbackPolicy(content_width=4, temporal_width=2, rank=4)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
        values = self._inputs()
        before = policy(*values).weights.detach().clone()
        initial_state = policy.capture_plastic_state()

        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            output = policy(*values)
            loss = situated_outcome_loss(output, torch.tensor([0]), torch.tensor([-1.0]))
            loss.backward()
            optimizer.step()
        after_failure = policy(*values).weights.detach().clone()
        self.assertLess(after_failure[0, 0].item(), before[0, 0].item())

        policy.restore_plastic_state(initial_state)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            output = policy(*values)
            loss = situated_outcome_loss(output, torch.tensor([1]), torch.tensor([1.0]))
            loss.backward()
            optimizer.step()
        after_success = policy(*values).weights.detach()
        self.assertGreater(after_success[0, 1].item(), before[0, 1].item())

    def test_state_is_constant_bounded_and_exactly_restorable(self) -> None:
        torch.manual_seed(13)
        policy = SituatedFeedbackPolicy(
            content_width=4,
            temporal_width=2,
            rank=4,
            maximum_residual=0.25,
        )
        count = policy.plastic_parameter_count
        state = policy.capture_plastic_state()
        digest = policy.plastic_state_digest()
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(10.0)
        output = policy(*self._inputs())
        self.assertLessEqual(output.residuals.abs().max().item(), 0.25)
        policy.restore_plastic_state(state)

        self.assertEqual(policy.plastic_parameter_count, count)
        self.assertEqual(policy.plastic_state_digest(), digest)

    def test_oml_policy_separates_slow_representation_from_33_value_fast_state(self) -> None:
        torch.manual_seed(14)
        policy = OMLSituatedFeedbackPolicy(content_width=4, temporal_width=2, rank=32)
        output = policy(*self._inputs())
        self.assertEqual(output.scores.argmax(dim=1).item(), 0)
        self.assertEqual(policy.plastic_parameter_count, 33)
        self.assertGreater(policy.slow_parameter_count, policy.plastic_parameter_count)
        initial = policy.capture_plastic_state()

        policy.freeze_slow()
        trainable = [name for name, value in policy.named_parameters() if value.requires_grad]
        self.assertEqual(trainable, ["fast_weight", "fast_bias"])
        optimizer = torch.optim.SGD((policy.fast_weight, policy.fast_bias), lr=0.5)
        optimizer.zero_grad(set_to_none=True)
        output = policy(*self._inputs())
        loss = situated_outcome_loss(output, torch.tensor([0]), torch.tensor([-1.0]))
        loss.backward()
        optimizer.step()
        self.assertNotEqual(policy.plastic_state_digest(), OMLSituatedFeedbackPolicy(
            content_width=4, temporal_width=2, rank=32
        ).plastic_state_digest())
        policy.restore_plastic_state(initial)
        self.assertTrue(torch.equal(policy.fast_weight, torch.zeros_like(policy.fast_weight)))


if __name__ == "__main__":
    unittest.main()
