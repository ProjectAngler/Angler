from __future__ import annotations

import math
import unittest
from unittest import mock

from experiments.runners import phase6_anml_fidelity_v23_d1 as source
from experiments.runners import phase6_anml_fidelity_v23_d1_r1 as recovery
from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


class Phase6ANMLFidelityV23D1R1Tests(unittest.TestCase):
    def test_integer_milestone_keys_are_stringified_recursively(self) -> None:
        value = {"evaluation": {0: {"loss": 1.0}, 512: {"loss": 0.5}}}
        self.assertEqual(
            recovery.json_ready(value),
            {"evaluation": {"0": {"loss": 1.0}, "512": {"loss": 0.5}}},
        )

    def test_key_stringification_collision_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "collision"):
            recovery.json_ready({1: "integer", "1": "string"})

    def test_non_finite_value_fails_closed(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                recovery.json_ready({"value": value})

    def test_wrapper_restores_validator_after_success(self) -> None:
        original = v22._validate_json_value
        source_output = {
            "artifact_schema": "source",
            "protocol_id": source.PROTOCOL_ID,
            "classification": "FIDELITY_HYPOTHESES_NOT_SUPPORTED",
            "evaluation": {0: {"loss": 0.25}},
        }
        with mock.patch.object(source, "run", return_value=source_output) as delegated:
            result = recovery.recover_source_run("cpu")
        delegated.assert_called_once_with("cpu")
        self.assertIs(v22._validate_json_value, original)
        self.assertEqual(result["evaluation"], {"0": {"loss": 0.25}})
        self.assertEqual(result["protocol_id"], recovery.PROTOCOL_ID)
        self.assertFalse(result["recovery"]["scientific_configuration_changed"])

    def test_wrapper_restores_validator_after_injected_error(self) -> None:
        original = v22._validate_json_value
        with mock.patch.object(source, "run", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                recovery.recover_source_run("cpu")
        self.assertIs(v22._validate_json_value, original)

    def test_frozen_scientific_objects_are_reused(self) -> None:
        self.assertIs(recovery.CONFIGS, source.CONFIGS)
        self.assertIs(recovery.classify_result, source.classify_result)
        self.assertEqual(recovery.OUTER_UPDATES, source.OUTER_UPDATES)
        self.assertEqual(recovery.LIFETIME_UPDATES, source.LIFETIME_UPDATES)
        self.assertEqual(recovery.PROBE_MILESTONES, source.PROBE_MILESTONES)

    def test_cpu_preflight_is_non_semantic_and_json_safe(self) -> None:
        result = recovery.synthetic_preflight("cpu")
        self.assertTrue(result["passed"])
        self.assertTrue(result["scientific_configuration_inherited"])
        v22._validate_json_value(result)


if __name__ == "__main__":
    unittest.main()
