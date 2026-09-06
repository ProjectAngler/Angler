from __future__ import annotations

import unittest

import torch

from angler.runtime.plastic_working_set import (
    DEFAULT_ACTIVE_BUDGET_BYTES,
    LearnedPlasticWorkingSetCoordinator,
    PlasticExpert,
    PlasticityVramBudget,
)


def _ref(number: int) -> str:
    return "sha256:" + f"{number:064x}"


class PlasticWorkingSetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260904)

    def test_learns_numeric_contribution_without_semantic_router_inputs(self) -> None:
        coordinator = LearnedPlasticWorkingSetCoordinator(
            context_width=4, expert_width=4, hidden_width=24
        )
        optimizer = torch.optim.AdamW(coordinator.parameters(), lr=0.02)
        contexts = torch.tensor(
            [[1.0, 0.0, 0.2, 0.1], [0.0, 1.0, 0.1, 0.2]] * 16
        )
        experts = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]] * 32
        )
        contributions = torch.tensor([[1.0, -1.0], [-1.0, 1.0]] * 16)
        novelty = torch.zeros(32)
        with torch.no_grad():
            initial = float(
                coordinator.learning_loss(contexts, experts, contributions, novelty)
            )
        for _ in range(120):
            loss = coordinator.learning_loss(
                contexts, experts, contributions, novelty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final = float(
            coordinator.learning_loss(
                contexts, experts, contributions, novelty
            ).detach()
        )
        self.assertLess(final, initial * 0.2)
        profiles = (
            PlasticExpert(_ref(1), _ref(101), 600_000_000, 8, (1, 0, 0, 0)),
            PlasticExpert(_ref(2), _ref(102), 600_000_000, 8, (0, 1, 0, 0)),
        )
        first = coordinator.choose_working_set(
            context=(1, 0, 0.2, 0.1), experts=profiles
        )
        second = coordinator.choose_working_set(
            context=(0, 1, 0.1, 0.2), experts=profiles
        )
        self.assertEqual(first.selected_expert_refs, (_ref(1),))
        self.assertEqual(second.selected_expert_refs, (_ref(2),))
        self.assertLessEqual(first.total_accounted_bytes, DEFAULT_ACTIVE_BUDGET_BYTES)
        self.assertLessEqual(second.total_accounted_bytes, DEFAULT_ACTIVE_BUDGET_BYTES)

    def test_mandatory_shared_state_is_retained_or_plan_fails(self) -> None:
        coordinator = LearnedPlasticWorkingSetCoordinator(
            context_width=2, expert_width=2, hidden_width=8
        )
        mandatory = PlasticExpert(
            _ref(3), _ref(103), 900_000_000, 16, (1, 0), mandatory=True
        )
        optional = PlasticExpert(_ref(4), _ref(104), 300_000_000, 8, (0, 1))
        decision = coordinator.choose_working_set(
            context=(1, 0), experts=(mandatory, optional)
        )
        self.assertIn(mandatory.expert_ref, decision.selected_expert_refs)
        self.assertNotIn(optional.expert_ref, decision.selected_expert_refs)
        with self.assertRaisesRegex(ValueError, "mandatory plastic state"):
            coordinator.choose_working_set(
                context=(1, 0),
                experts=(mandatory,),
                vram_budget=PlasticityVramBudget(total_bytes=899_999_999),
            )

    def test_backend_and_cross_gpu_reserves_reduce_adapter_capacity(self) -> None:
        coordinator = LearnedPlasticWorkingSetCoordinator(
            context_width=2, expert_width=2, hidden_width=8
        )
        budget = PlasticityVramBudget(
            backend_pool_bytes=120_000_000,
            router_bytes=4_000_000,
            communication_bytes=32_000_000,
            staging_bytes=128_000_000,
        )
        mandatory = PlasticExpert(
            _ref(7), _ref(107), 700_000_000, 16, (1, 0), mandatory=True
        )
        decision = coordinator.choose_working_set(
            context=(1, 0), experts=(mandatory,), vram_budget=budget
        )
        self.assertEqual(decision.reserved_bytes, 284_000_000)
        self.assertEqual(
            decision.adapter_capacity_bytes,
            DEFAULT_ACTIVE_BUDGET_BYTES - 284_000_000,
        )
        self.assertEqual(decision.total_accounted_bytes, 984_000_000)
        self.assertLessEqual(
            decision.total_accounted_bytes, decision.total_budget_bytes
        )

    def test_decision_is_reproducible_and_semantic_text_is_rejected(self) -> None:
        coordinator = LearnedPlasticWorkingSetCoordinator(
            context_width=2, expert_width=2, hidden_width=8
        )
        profiles = (
            PlasticExpert(_ref(5), _ref(105), 1_000, 4, (0.1, 0.2), True),
            PlasticExpert(_ref(6), _ref(106), 1_000, 4, (0.2, 0.1)),
        )
        first = coordinator.choose_working_set(context=(0.3, 0.4), experts=profiles)
        second = coordinator.choose_working_set(context=(0.3, 0.4), experts=profiles)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "numeric values"):
            coordinator.choose_working_set(context="library request", experts=profiles)


if __name__ == "__main__":
    unittest.main()
