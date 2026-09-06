from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from angler.reasoning import (
    FactorizedKeyValueCausalMemoryCore,
    SparseMemoryCausalRoutingCore,
)
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    FinalPartitionSealedError,
)
from experiments.runners import factorized_key_value_causal_memory_v9 as v9
from experiments.runners import sparse_memory_causal_routing_v8 as v8


_DONOR_NAMES = (
    "query_projection.weight",
    "candidate_projection.weight",
    "temporal_projection.weight",
    "semantic_network.0.weight",
    "semantic_network.0.bias",
    "semantic_network.1.weight",
    "semantic_network.1.bias",
)


def _passing_factorization_metrics() -> dict[str, object]:
    return {
        "factorization": {
            "mechanism_count": 8,
            "maximum_write_key_difference_under_shuffle": 1.0e-6,
            "maximum_write_weight_difference_under_shuffle": 1.0e-6,
            "maximum_final_key_difference_under_shuffle": 1.0e-6,
            "maximum_final_usage_difference_under_shuffle": 1.0e-6,
            "maximum_challenge_read_weight_difference_under_shuffle": 1.0e-6,
            "minimum_final_value_difference_under_shuffle": 1.000001e-6,
            "maximum_neutral_value": 0.0,
            "maximum_neutral_residual": 0.0,
            "maximum_nonzero_read_slots": 2,
            "maximum_nonzero_write_slots": 2,
        }
    }


class FactorizedKeyValueCausalMemoryV9RunnerTests(unittest.TestCase):
    def test_protocol_and_compute_are_inherited_from_v8(self) -> None:
        self.assertNotEqual(v9.IDENTITY, v8.IDENTITY)
        self.assertIs(v9._train_core, v8._train_core)
        self.assertIs(v9._evaluate_all, v8._evaluate_all)
        self.assertEqual(v9.RANK, v8.RANK)
        self.assertEqual(v9.MEMORY_SLOTS, v8.MEMORY_SLOTS)
        self.assertEqual(v9.TEMPORAL_WIDTH, v8.TEMPORAL_WIDTH)
        self.assertEqual(len(v8.v7.v6._pair_schedule()), 256)
        self.assertTrue(callable(v9._factorization_metrics))

    def test_donor_loads_only_seven_outcome_blind_projection_tensors(self) -> None:
        torch.manual_seed(901)
        donor = SparseMemoryCausalRoutingCore(content_width=6, temporal_width=8)
        donor_state = donor.state_dict()
        payload = {"identity": v8.IDENTITY, "core_state_dict": donor_state}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v8.pt"
            torch.save(payload, path)
            torch.manual_seed(902)
            recipient = FactorizedKeyValueCausalMemoryCore(
                content_width=6,
                temporal_width=8,
            )
            outcome_before = recipient.outcome_embedding.weight.detach().clone()
            record = v9._load_semantic_donor(recipient, path)

        self.assertEqual(tuple(record["mapping"]), _DONOR_NAMES)
        for name in _DONOR_NAMES:
            with self.subTest(parameter=name):
                self.assertTrue(
                    torch.equal(recipient.state_dict()[name], donor_state[name])
                )
        self.assertTrue(
            torch.equal(recipient.outcome_embedding.weight, outcome_before)
        )

    def test_gate_rejects_each_factorization_violation(self) -> None:
        cases: dict[str, dict[str, object]] = {}
        for key in (
            "maximum_write_key_difference_under_shuffle",
            "maximum_write_weight_difference_under_shuffle",
            "maximum_final_key_difference_under_shuffle",
            "maximum_final_usage_difference_under_shuffle",
            "maximum_challenge_read_weight_difference_under_shuffle",
        ):
            metrics = copy.deepcopy(_passing_factorization_metrics())
            metrics["factorization"][key] = 1.000001e-6
            cases[key] = metrics

        no_value_separation = copy.deepcopy(_passing_factorization_metrics())
        no_value_separation["factorization"][
            "minimum_final_value_difference_under_shuffle"
        ] = 1.0e-6
        cases["minimum_final_value_difference_under_shuffle"] = no_value_separation

        neutral_value = copy.deepcopy(_passing_factorization_metrics())
        neutral_value["factorization"]["maximum_neutral_value"] = 1.0e-12
        cases["maximum_neutral_value"] = neutral_value

        neutral_residual = copy.deepcopy(_passing_factorization_metrics())
        neutral_residual["factorization"]["maximum_neutral_residual"] = 1.0e-12
        cases["maximum_neutral_residual"] = neutral_residual

        too_many_read_slots = copy.deepcopy(_passing_factorization_metrics())
        too_many_read_slots["factorization"]["maximum_nonzero_read_slots"] = 3
        cases["maximum_nonzero_read_slots"] = too_many_read_slots

        too_many_write_slots = copy.deepcopy(_passing_factorization_metrics())
        too_many_write_slots["factorization"]["maximum_nonzero_write_slots"] = 3
        cases["maximum_nonzero_write_slots"] = too_many_write_slots

        cases["missing_factorization"] = {}

        with mock.patch.object(v9.v8, "_development_gate", return_value=True):
            self.assertTrue(v9._development_gate(_passing_factorization_metrics(), True))
            for label, metrics in cases.items():
                with self.subTest(label=label):
                    self.assertFalse(v9._development_gate(metrics, True))

    def test_final_partition_remains_sealed_by_default(self) -> None:
        corpus = v9.build_causal_neuromodulated_apprenticeship_v6()
        with self.assertRaises(FinalPartitionSealedError):
            _ = corpus.final


if __name__ == "__main__":
    unittest.main()
