import copy
from collections import Counter
import unittest

from experiments.runners import outcome_aware_apprenticeship_v5 as v5


REQUIRED_CONTROLS = (
    "state_reset",
    "outcome_shuffled",
    "memory_disabled",
    "fair_semantic_retrieval",
    "moving_origin_removed",
    "state_zero",
)


def _passing_development_metrics() -> dict[str, object]:
    return {
        "mechanism_count": 8,
        "successes": {
            "full": 6,
            "state_reset": 4,
            "outcome_shuffled": 4,
            "memory_disabled": 4,
            "fair_semantic_retrieval": 4,
            "moving_origin_removed": 4,
            "state_zero": 4,
            "equivalent_state_swap": 6,
        },
        "equivalent_state_target_probability_retention": 0.90,
    }


class OutcomeAwareApprenticeshipV5MechanicsTests(unittest.TestCase):
    def test_pair_schedule_is_balanced_and_frozen(self) -> None:
        schedule = v5._pair_schedule()
        self.assertIsInstance(schedule, tuple)
        self.assertEqual(len(schedule), 128)
        self.assertTrue(
            all(
                isinstance(pair, tuple)
                and len(pair) == 2
                and all(type(index) is int for index in pair)
                for pair in schedule
            )
        )
        self.assertTrue(
            all(0 <= anchor < 32 and 0 <= current < 32 for anchor, current in schedule)
        )
        self.assertTrue(all(anchor != current for anchor, current in schedule))
        self.assertEqual(
            Counter(anchor for anchor, _ in schedule),
            Counter({index: 4 for index in range(32)}),
        )
        self.assertEqual(
            Counter(current for _, current in schedule),
            Counter({index: 4 for index in range(32)}),
        )
        expected = tuple(
            (anchor, (anchor + offset) % 32)
            for offset in (1, 7, 13, 19)
            for anchor in range(32)
        )
        self.assertEqual(schedule, expected)

    def test_outcome_derangement_changes_every_label_and_preserves_balance(self) -> None:
        outcomes = (1, -1, 1, -1, 1, -1)
        shuffled = v5._deranged_outcomes(outcomes)
        self.assertIsInstance(shuffled, tuple)
        self.assertEqual(shuffled, (-1, 1, -1, 1, -1, 1))
        self.assertEqual(len(shuffled), len(outcomes))
        self.assertTrue(all(left != right for left, right in zip(outcomes, shuffled)))
        self.assertEqual(Counter(shuffled), Counter({-1: 3, 1: 3}))
        with self.assertRaises(ValueError):
            v5._deranged_outcomes((1, 1, 1, -1, -1, -1))

    def test_development_gate_accepts_exact_boundary(self) -> None:
        self.assertTrue(v5._development_gate(_passing_development_metrics()))

    def test_development_gate_rejects_each_failed_operand(self) -> None:
        cases: dict[str, dict[str, object]] = {}

        too_few_full = _passing_development_metrics()
        too_few_full["successes"]["full"] = 5
        cases["full_below_six_of_eight"] = too_few_full

        for control in REQUIRED_CONTROLS:
            insufficient_advantage = _passing_development_metrics()
            insufficient_advantage["successes"][control] = 5
            cases[f"insufficient_advantage_over_{control}"] = insufficient_advantage

        weak_equivalent_swap = _passing_development_metrics()
        weak_equivalent_swap["successes"]["equivalent_state_swap"] = 5
        cases["equivalent_swap_below_full"] = weak_equivalent_swap

        weak_swap_advantage = _passing_development_metrics()
        weak_swap_advantage["successes"]["state_reset"] = 5
        cases["equivalent_swap_not_two_over_reset"] = weak_swap_advantage

        low_retention = _passing_development_metrics()
        low_retention["equivalent_state_target_probability_retention"] = 0.899999
        cases["equivalent_state_retention_below_ninety_percent"] = low_retention

        wrong_count = _passing_development_metrics()
        wrong_count["mechanism_count"] = 7
        cases["development_mechanism_count_changed"] = wrong_count

        for label, metrics in cases.items():
            with self.subTest(label=label):
                self.assertFalse(v5._development_gate(metrics))

    def test_development_gate_rejects_missing_or_malformed_metrics(self) -> None:
        passing = _passing_development_metrics()
        malformed = []
        for control in REQUIRED_CONTROLS:
            missing = copy.deepcopy(passing)
            del missing["successes"][control]
            malformed.append(missing)
        malformed.extend(
            (
                {},
                {**passing, "successes": None},
                {**passing, "mechanism_count": True},
                {
                    **passing,
                    "equivalent_state_target_probability_retention": float("nan"),
                },
            )
        )
        for metrics in malformed:
            with self.subTest(metrics=metrics):
                self.assertFalse(v5._development_gate(metrics))

    def test_final_authorization_refuses_unpassed_development(self) -> None:
        with self.assertRaises(RuntimeError):
            v5._validate_final_authorization(False)
        self.assertIsNone(v5._validate_final_authorization(True))


if __name__ == "__main__":
    unittest.main()
