import copy
from collections import Counter
from types import SimpleNamespace
import unittest

import torch

from experiments.runners import causal_neuromodulated_apprenticeship_v6 as v6


def _passing_development_metrics() -> dict[str, object]:
    return {
        "mechanism_count": 8,
        "successes": {
            "full": 6,
            "state_reset": 4,
            "outcome_shuffled": 4,
            "outcome_zeroed": 4,
            "memory_disabled": 4,
            "fair_semantic_retrieval": 4,
            "moving_origin_removed": 4,
            "equivalent_state_swap": 6,
        },
        "full_failed_to_pass_repairs": 1,
        "full_negative_feedback_reduced_failed_probability": True,
        "full_target_probability_non_decreasing": True,
        "equivalent_state_target_probability_retention": 0.90,
        "sequential_target_probability_retention": 0.90,
        "outcome_causality": {
            "minimum_target_probability_advantage": 0.10,
            "minimum_failed_probability_reduction": 0.10,
        },
    }


class CausalNeuromodulatedApprenticeshipV6MechanicsTests(unittest.TestCase):
    def test_pair_schedule_is_exact_balanced_and_has_no_self_pairs(self) -> None:
        schedule = v6._pair_schedule()
        expected = tuple(
            (anchor, (anchor + offset) % 64)
            for offset in (5, 17, 29, 43)
            for anchor in range(64)
        )

        self.assertIsInstance(schedule, tuple)
        self.assertEqual(schedule, expected)
        self.assertEqual(len(schedule), 256)
        self.assertTrue(
            all(
                isinstance(pair, tuple)
                and len(pair) == 2
                and all(type(index) is int for index in pair)
                for pair in schedule
            )
        )
        self.assertTrue(
            all(0 <= anchor < 64 and 0 <= current < 64 for anchor, current in schedule)
        )
        self.assertTrue(all(anchor != current for anchor, current in schedule))
        self.assertEqual(
            Counter(anchor for anchor, _ in schedule),
            Counter({index: 4 for index in range(64)}),
        )
        self.assertEqual(
            Counter(current for _, current in schedule),
            Counter({index: 4 for index in range(64)}),
        )

    def test_development_gate_accepts_the_exact_frozen_boundary(self) -> None:
        self.assertTrue(v6._development_gate(_passing_development_metrics(), True))

    def test_development_gate_rejects_each_failed_operand(self) -> None:
        cases: dict[str, tuple[dict[str, object], bool]] = {}

        wrong_count = copy.deepcopy(_passing_development_metrics())
        wrong_count["mechanism_count"] = 7
        cases["mechanism_count"] = (wrong_count, True)

        too_few_full = copy.deepcopy(_passing_development_metrics())
        too_few_full["successes"]["full"] = 5
        cases["full_successes"] = (too_few_full, True)

        for control in v6.REQUIRED_CONTROLS:
            insufficient_advantage = copy.deepcopy(_passing_development_metrics())
            insufficient_advantage["successes"][control] = 5
            cases[f"advantage_over_{control}"] = (insufficient_advantage, True)

        no_repair = copy.deepcopy(_passing_development_metrics())
        no_repair["full_failed_to_pass_repairs"] = 0
        cases["failed_to_pass_repair"] = (no_repair, True)

        no_failure_reduction = copy.deepcopy(_passing_development_metrics())
        no_failure_reduction["full_negative_feedback_reduced_failed_probability"] = False
        cases["negative_feedback"] = (no_failure_reduction, True)

        decreasing_target = copy.deepcopy(_passing_development_metrics())
        decreasing_target["full_target_probability_non_decreasing"] = False
        cases["target_non_decreasing"] = (decreasing_target, True)

        weak_swap = copy.deepcopy(_passing_development_metrics())
        weak_swap["successes"]["equivalent_state_swap"] = 5
        cases["equivalent_swap_successes"] = (weak_swap, True)

        weak_swap_retention = copy.deepcopy(_passing_development_metrics())
        weak_swap_retention["equivalent_state_target_probability_retention"] = 0.899999
        cases["equivalent_swap_retention"] = (weak_swap_retention, True)

        weak_sequential_retention = copy.deepcopy(_passing_development_metrics())
        weak_sequential_retention["sequential_target_probability_retention"] = 0.899999
        cases["sequential_retention"] = (weak_sequential_retention, True)

        weak_target_causality = copy.deepcopy(_passing_development_metrics())
        weak_target_causality["outcome_causality"][
            "minimum_target_probability_advantage"
        ] = 0.099999
        cases["causal_target_margin"] = (weak_target_causality, True)

        weak_failure_causality = copy.deepcopy(_passing_development_metrics())
        weak_failure_causality["outcome_causality"][
            "minimum_failed_probability_reduction"
        ] = 0.099999
        cases["causal_failure_margin"] = (weak_failure_causality, True)

        cases["identity"] = (copy.deepcopy(_passing_development_metrics()), False)

        for label, (metrics, identities_exact) in cases.items():
            with self.subTest(label=label):
                self.assertFalse(v6._development_gate(metrics, identities_exact))

    def test_development_gate_rejects_missing_operands(self) -> None:
        missing_cases = []
        for control in v6.REQUIRED_CONTROLS:
            missing = copy.deepcopy(_passing_development_metrics())
            del missing["successes"][control]
            missing_cases.append(missing)
        for key in (
            "full_failed_to_pass_repairs",
            "full_negative_feedback_reduced_failed_probability",
            "full_target_probability_non_decreasing",
            "equivalent_state_target_probability_retention",
            "sequential_target_probability_retention",
            "outcome_causality",
        ):
            missing = copy.deepcopy(_passing_development_metrics())
            del missing[key]
            missing_cases.append(missing)

        for metrics in missing_cases:
            with self.subTest(metrics=metrics):
                self.assertFalse(v6._development_gate(metrics, True))

    def test_smooth_margin_is_finite_and_has_nonzero_gradient(self) -> None:
        value = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        loss = v6._smooth_margin(value)

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(value.grad)
        self.assertTrue(bool(torch.isfinite(value.grad).item()))
        self.assertGreater(float(value.grad.abs().item()), 0.0)

    def test_episode_query_excludes_diagnostic_and_outcome_words(self) -> None:
        episode = SimpleNamespace(
            task_text="Reconcile the synthetic ledger.",
            request_text="Choose a valid action trace.",
            objective_diagnostic_text=(
                "SUCCESS and FAILURE are outcome-only diagnostic words."
            ),
            outcome_label="SUCCESS",
        )
        mechanism = SimpleNamespace(
            public=SimpleNamespace(episodes=(episode,)),
        )

        query = v6._episode_query_text(mechanism, 0)

        self.assertEqual(
            query,
            "Task: Reconcile the synthetic ledger.\n"
            "Request: Choose a valid action trace.",
        )
        self.assertNotIn(episode.objective_diagnostic_text, query)
        self.assertNotIn("success", query.lower())
        self.assertNotIn("failure", query.lower())


if __name__ == "__main__":
    unittest.main()
