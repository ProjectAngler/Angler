import copy
from collections import Counter
from types import SimpleNamespace
import unittest

import torch

from angler.reasoning import ContentAddressedCausalMemoryState
from experiments.runners import content_addressed_causal_memory_v7 as v7


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
        "memory_structure": {
            "all_bounds_valid": True,
            "aggregate_effective_write_slots": 4.0,
            "aggregate_maximum_write_share": 0.50,
        },
    }


def _write_outputs(weights: torch.Tensor, count: int = 12) -> tuple[object, ...]:
    shaped = weights.reshape(1, 1, v7.MEMORY_SLOTS)
    return tuple(SimpleNamespace(write_weights=shaped.clone()) for _ in range(count))


class ContentAddressedCausalMemoryV7RunnerTests(unittest.TestCase):
    def test_inherited_pair_schedule_remains_exact_and_balanced(self) -> None:
        schedule = v7.v6._pair_schedule()
        expected = tuple(
            (anchor, (anchor + offset) % 64)
            for offset in (5, 17, 29, 43)
            for anchor in range(64)
        )

        self.assertEqual(schedule, expected)
        self.assertEqual(len(schedule), 256)
        self.assertTrue(all(anchor != current for anchor, current in schedule))
        self.assertEqual(
            Counter(anchor for anchor, _ in schedule),
            Counter({index: 4 for index in range(64)}),
        )
        self.assertEqual(
            Counter(current for _, current in schedule),
            Counter({index: 4 for index in range(64)}),
        )

    def test_uniform_aggregate_write_has_sixteen_effective_slots_and_zero_loss(self) -> None:
        uniform = torch.full(
            (v7.MEMORY_SLOTS,),
            1.0 / v7.MEMORY_SLOTS,
            dtype=torch.float32,
            requires_grad=True,
        )

        loss, metrics = v7._occupancy_loss(_write_outputs(uniform))

        self.assertAlmostEqual(float(loss.item()), 0.0, places=8)
        self.assertAlmostEqual(metrics["aggregate_effective_write_slots"], 16.0, places=5)
        self.assertAlmostEqual(metrics["aggregate_maximum_write_share"], 1.0 / 16.0, places=6)
        self.assertAlmostEqual(metrics["aggregate_write_mass"], 12.0, places=5)

    def test_collapsed_one_slot_write_has_positive_occupancy_loss(self) -> None:
        collapsed = torch.zeros(v7.MEMORY_SLOTS, dtype=torch.float32)
        collapsed[0] = 1.0

        loss, metrics = v7._occupancy_loss(_write_outputs(collapsed))

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertLess(metrics["aggregate_effective_write_slots"], 1.001)
        self.assertGreater(metrics["aggregate_maximum_write_share"], 0.999)

    def test_development_gate_accepts_exact_v6_and_memory_boundaries(self) -> None:
        self.assertTrue(v7._development_gate(_passing_development_metrics(), True))

    def test_development_gate_rejects_each_memory_structure_operand(self) -> None:
        cases = {}

        invalid_bounds = copy.deepcopy(_passing_development_metrics())
        invalid_bounds["memory_structure"]["all_bounds_valid"] = False
        cases["bounds"] = invalid_bounds

        insufficient_occupancy = copy.deepcopy(_passing_development_metrics())
        insufficient_occupancy["memory_structure"][
            "aggregate_effective_write_slots"
        ] = 3.999999
        cases["effective_slots"] = insufficient_occupancy

        excessive_concentration = copy.deepcopy(_passing_development_metrics())
        excessive_concentration["memory_structure"][
            "aggregate_maximum_write_share"
        ] = 0.500001
        cases["maximum_share"] = excessive_concentration

        missing_structure = copy.deepcopy(_passing_development_metrics())
        del missing_structure["memory_structure"]
        cases["missing_structure"] = missing_structure

        for label, metrics in cases.items():
            with self.subTest(label=label):
                self.assertFalse(v7._development_gate(metrics, True))

        self.assertFalse(v7._development_gate(_passing_development_metrics(), False))

    def test_state_metrics_report_valid_bounded_sixteen_by_thirty_two_state(self) -> None:
        keys = torch.linspace(-1.0, 1.0, v7.MEMORY_SLOTS * v7.RANK).reshape(
            v7.MEMORY_SLOTS, v7.RANK
        )
        values = torch.full((v7.MEMORY_SLOTS, v7.RANK), 0.5)
        usage = torch.full((v7.MEMORY_SLOTS,), 0.5)
        state = ContentAddressedCausalMemoryState(
            keys=keys,
            values=values,
            usage=usage,
            step=7,
        )

        metrics = v7._state_metrics(state)

        self.assertTrue(metrics["bounds_valid"])
        self.assertEqual(metrics["step"], 7)
        self.assertEqual(metrics["occupied_slots"], 16)
        self.assertAlmostEqual(metrics["key_min"], -1.0, places=7)
        self.assertAlmostEqual(metrics["key_max"], 1.0, places=7)
        self.assertAlmostEqual(metrics["value_min"], 0.5, places=7)
        self.assertAlmostEqual(metrics["value_max"], 0.5, places=7)
        self.assertAlmostEqual(metrics["usage_min"], 0.5, places=7)
        self.assertAlmostEqual(metrics["usage_max"], 0.5, places=7)
        self.assertAlmostEqual(metrics["effective_usage_slots"], 16.0, places=5)
        self.assertAlmostEqual(metrics["maximum_usage_share"], 1.0 / 16.0, places=7)
        self.assertEqual(state.bytes, 4160)


if __name__ == "__main__":
    unittest.main()
