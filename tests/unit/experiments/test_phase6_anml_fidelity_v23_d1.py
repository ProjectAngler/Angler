from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.runners import phase6_anml_fidelity_v23_d1 as d1
from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


class V23D1PlanTests(unittest.TestCase):
    def test_configuration_factorization_is_exact(self) -> None:
        self.assertEqual(
            tuple((item.name, item.fast_optimizer, item.inner_steps) for item in d1.CONFIGS),
            (
                ("adamw_8", "adamw", 8),
                ("sgd_8", "sgd", 8),
                ("adamw_20", "adamw", 20),
                ("sgd_20", "sgd", 20),
            ),
        )

    def test_meta_records_are_unique_and_share_the_twenty_step_prefix(self) -> None:
        records = d1.meta_records(17)
        self.assertEqual(len(records["inner"]), 20)
        self.assertEqual(len(records["outer"]), 8)
        identities = {
            (record["topology_seed"], record["surface_seed"])
            for record in records["inner"] + records["outer"]
        }
        self.assertEqual(len(identities), 28)
        self.assertTrue(
            all(record["commitment_index"] == 8 + 17 % 24 for record in records["inner"])
        )

    def test_lifetime_orders_are_balanced_and_distinct(self) -> None:
        blocked = d1.lifetime_order(0)
        interleaved = d1.lifetime_order(1)
        self.assertEqual(len(blocked), 512)
        self.assertEqual(len(interleaved), 512)
        self.assertNotEqual(blocked, interleaved)
        for commitment in d1.PILOT_COMMITMENTS:
            self.assertEqual(blocked.count(commitment), 32)
            self.assertEqual(interleaved.count(commitment), 32)

    def test_auc_uses_all_three_frozen_boundaries(self) -> None:
        self.assertAlmostEqual(d1._auc({0: 1.0, 128: 0.5, 512: 0.25}), 0.46875)


class V23D1MathTests(unittest.TestCase):
    def test_gradient_comparison_reports_indirect_component(self) -> None:
        full = (torch.tensor([3.0, 4.0]),)
        direct = (torch.tensor([0.0, 4.0]),)
        report = d1._gradient_comparison(full, direct)
        self.assertAlmostEqual(report["full_norm"], 5.0)
        self.assertAlmostEqual(report["direct_norm"], 4.0)
        self.assertAlmostEqual(report["indirect_difference_norm"], 3.0)
        self.assertAlmostEqual(report["indirect_over_full"], 0.6)
        self.assertAlmostEqual(report["full_direct_cosine"], 0.8)

    def test_cpu_preflight_exercises_both_fast_optimizers(self) -> None:
        report = d1.synthetic_preflight("cpu")
        self.assertTrue(report["passed"])
        self.assertEqual(report["configurations"], tuple(item.name for item in d1.CONFIGS))
        self.assertTrue(report["orders_balanced"])

    def test_relative_improvement_rejects_nonpositive_baseline(self) -> None:
        with self.assertRaises(RuntimeError):
            d1._relative_improvement(0.0, 0.0)


class V23D1ClassificationTests(unittest.TestCase):
    @staticmethod
    def _evaluation(eligible: tuple[str, ...] = ()) -> dict[str, dict[str, object]]:
        return {
            config.name: {
                "full_v23_eligible": config.name in eligible,
                "minimum_second_vs_first_auc": {
                    "adamw_8": 0.01,
                    "sgd_8": 0.03,
                    "adamw_20": 0.02,
                    "sgd_20": 0.04,
                }[config.name],
            }
            for config in d1.CONFIGS
        }

    @staticmethod
    def _d0(adamw: float, sgd: float) -> dict[str, object]:
        return {
            "updates": {
                "adamw": {"live_open_update_separation_fraction": adamw},
                "sgd": {"live_open_update_separation_fraction": sgd},
            }
        }

    def test_best_eligible_configuration_wins(self) -> None:
        result = d1.classify_result(
            self._d0(0.01, 0.20), self._evaluation(("sgd_8", "sgd_20"))
        )
        self.assertEqual(result["classification"], "FULL_V23_ELIGIBLE")
        self.assertEqual(result["selected_configuration"], "sgd_20")

    def test_optimizer_mismatch_requires_ratio_and_absolute_gap(self) -> None:
        supported = d1.classify_result(self._d0(0.02, 0.10), self._evaluation())
        self.assertEqual(supported["classification"], "OPTIMIZER_MISMATCH_SUPPORTED")
        ratio_only = d1.classify_result(self._d0(0.01, 0.03), self._evaluation())
        self.assertEqual(ratio_only["classification"], "FIDELITY_HYPOTHESES_NOT_SUPPORTED")
        gap_only = d1.classify_result(self._d0(0.10, 0.16), self._evaluation())
        self.assertEqual(gap_only["classification"], "FIDELITY_HYPOTHESES_NOT_SUPPORTED")

    def test_atomic_json_rejects_tensors(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            v22._validate_json_value({"bad": torch.ones(1)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            v22.atomic_write_json(path, {"ok": True, "value": math.pi})
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
