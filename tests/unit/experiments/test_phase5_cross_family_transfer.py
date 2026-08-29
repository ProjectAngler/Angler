from __future__ import annotations

import unittest

import torch

from experiments.evaluators.relational_procedure_transfer_suite import (
    make_relational_procedure_transfer_stream,
)
from experiments.runners.phase5_cross_family_transfer import (
    SharedPublicFactAdapter,
)
from experiments.evaluators.skill_memory_suite import (
    make_skill_memory_meta_partition,
)
from experiments.runners import phase5_skill_memory_stream as phase5


class SharedPublicFactAdapterTests(unittest.TestCase):
    def test_zero_initialization_is_bit_exact_identity(self) -> None:
        torch.manual_seed(93_001)
        adapter = SharedPublicFactAdapter()
        public = torch.randn(5, 14)

        self.assertTrue(torch.equal(adapter(public), public))

    def test_encoder_is_item_permutation_equivariant(self) -> None:
        torch.manual_seed(93_003)
        adapter = SharedPublicFactAdapter()
        with torch.no_grad():
            adapter.output_projection.weight.normal_(mean=0.0, std=0.05)
        public = torch.randn(5, 14)
        permutation = torch.tensor((3, 0, 4, 1, 2))

        expected = adapter(public)[permutation]
        actual = adapter(public[permutation])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_zero_adapter_preserves_policy_logits_exactly(self) -> None:
        torch.manual_seed(93_007)
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        pair = make_relational_procedure_transfer_stream(
            93_007,
            supports_per_procedure=1,
            queries_per_procedure=1,
        ).supports[0]
        state = policy.initial_state(1)
        before = policy.score_task(pair.learner, state).logits.detach().clone()

        policy.public_fact_adapter = SharedPublicFactAdapter()
        after = policy.score_task(pair.learner, state).logits.detach().clone()
        self.assertTrue(torch.equal(before, after))

    def test_typed_adapter_cannot_change_native_attribute_tasks(self) -> None:
        torch.manual_seed(93_011)
        policy = phase5.SkillMemoryPolicy(phase5._PROFILES["smoke"])
        pair = make_skill_memory_meta_partition(
            93_011,
            instances_per_program=8,
        ).tasks[0]
        state = policy.initial_state(1)
        before = policy.score_task(pair.learner, state).logits.detach().clone()
        adapter = SharedPublicFactAdapter()
        with torch.no_grad():
            adapter.output_projection.weight.normal_(mean=0.0, std=1.0)
        policy.public_fact_adapter = adapter

        after = policy.score_task(pair.learner, state).logits.detach().clone()
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
