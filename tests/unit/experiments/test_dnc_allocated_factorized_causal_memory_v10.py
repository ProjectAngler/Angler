from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from angler.reasoning import (
    DncAllocatedFactorizedCausalMemoryCore,
    FactorizedKeyValueCausalMemoryCore,
)
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    FinalPartitionSealedError,
)
from experiments.runners import dnc_allocated_factorized_causal_memory_v10 as v10


_DONOR_NAMES = (
    "query_projection.weight",
    "candidate_projection.weight",
    "temporal_projection.weight",
    "semantic_network.0.weight",
    "semantic_network.0.bias",
    "semantic_network.1.weight",
    "semantic_network.1.bias",
)


def _passing_allocation_metrics() -> dict[str, object]:
    return {
        "allocation": {
            "event_count": 48,
            "persistent_unique_winning_slots": 48,
            "minimum_local_unique_winning_slots": 6,
            "no_reuse_before_capacity": True,
            "unused_slots_after_persistent": 16,
            "maximum_nonzero_write_slots": 1,
            "aggregate_write": {
                "effective_slots": 32.0,
                "maximum_share": 0.05,
            },
            "slot_permutation_read_value_difference": 1.0e-6,
            "slot_permutation_residual_difference": 1.0e-6,
            "slot_permutation_score_difference": 1.0e-6,
            "stress_step": 72,
            "stress_bounds_valid": True,
        }
    }


def _valid_development_record(source_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "identity": v10.IDENTITY,
        "phase": "train-development",
        "classification": "DEVELOPMENT_GATE_PASSED",
        "development_authorized": True,
        "source_hashes": source_hashes,
        "checkpoint_sha256": "checkpoint-digest",
        "development_metrics": {"synthetic": "valid"},
        "identity_checks": {
            "exact": True,
            "foundation_before": "foundation-digest",
            "foundation_after": "foundation-digest",
            "core_before_evaluation": "core-digest",
            "core_after_evaluation": "core-digest",
        },
    }


def _fake_rows() -> tuple[object, ...]:
    rows = []
    outcomes = torch.tensor((-1.0, 1.0, -1.0, 1.0, 1.0, -1.0))
    for index in range(8):
        generator = torch.Generator().manual_seed(1700 + index)
        target = index % 4
        rows.append(
            SimpleNamespace(
                mechanism_ref=f"synthetic-v10-{index}",
                generator_family=f"family-{index // 2}",
                episode_queries=torch.randn(6, 6, generator=generator),
                episode_candidates=torch.randn(6, 6, generator=generator),
                episode_temporal=torch.randn(6, 8, generator=generator),
                episode_outcomes=outcomes.clone(),
                challenge_query=torch.randn(6, generator=generator),
                challenge_candidates=torch.randn(4, 6, generator=generator),
                challenge_temporal=torch.randn(4, 8, generator=generator),
                target_index=target,
                failed_indices=tuple(i for i in range(4) if i != target),
            )
        )
    return tuple(rows)


class DncAllocatedFactorizedCausalMemoryV10RunnerTests(unittest.TestCase):
    def tearDown(self) -> None:
        # A failed assertion must not contaminate another inherited runner.
        v10.v7.MEMORY_SLOTS = 16

    def test_slot_scope_restores_v7_width_after_success_and_exception(self) -> None:
        self.assertEqual(v10.v7.MEMORY_SLOTS, 16)
        with v10._v10_slot_scope():
            self.assertEqual(v10.v7.MEMORY_SLOTS, 64)
        self.assertEqual(v10.v7.MEMORY_SLOTS, 16)

        with self.assertRaisesRegex(RuntimeError, "injected"):
            with v10._v10_slot_scope():
                self.assertEqual(v10.v7.MEMORY_SLOTS, 64)
                raise RuntimeError("injected")
        self.assertEqual(v10.v7.MEMORY_SLOTS, 16)

    def test_wrapped_train_restores_v7_width_when_donor_raises(self) -> None:
        def fail_inside_scope(*_args, **_kwargs):
            self.assertEqual(v10.v7.MEMORY_SLOTS, 64)
            raise RuntimeError("injected training failure")

        with mock.patch.object(v10.v9, "_train_core", side_effect=fail_inside_scope):
            with self.assertRaisesRegex(RuntimeError, "injected training failure"):
                v10._train_core(object(), ())
        self.assertEqual(v10.v7.MEMORY_SLOTS, 16)

    def test_protocol_thresholds_schedule_and_donor_scope_are_frozen(self) -> None:
        self.assertNotEqual(v10.IDENTITY, v10.v9.IDENTITY)
        self.assertEqual(v10.RANK, v10.v9.RANK)
        self.assertEqual(v10.TEMPORAL_WIDTH, v10.v9.TEMPORAL_WIDTH)
        self.assertEqual(v10.MEMORY_SLOTS, 64)
        self.assertEqual(v10.MINIMUM_EFFECTIVE_WRITE_SLOTS, 32.0)
        self.assertEqual(v10.MAXIMUM_AGGREGATE_WRITE_SHARE, 0.05)
        self.assertEqual(len(v10.v7.v6._pair_schedule()), 256)

        torch.manual_seed(1801)
        donor = FactorizedKeyValueCausalMemoryCore(
            content_width=6,
            temporal_width=8,
        )
        donor_state = donor.state_dict()
        payload = {"identity": v10.v9.IDENTITY, "core_state_dict": donor_state}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v9.pt"
            torch.save(payload, path)
            torch.manual_seed(1802)
            recipient = DncAllocatedFactorizedCausalMemoryCore(
                content_width=6,
                temporal_width=8,
            )
            outcome_before = recipient.outcome_embedding.weight.detach().clone()
            record = v10._load_semantic_donor(recipient, path)

        self.assertEqual(tuple(record["mapping"]), _DONOR_NAMES)
        for name in _DONOR_NAMES:
            with self.subTest(parameter=name):
                self.assertTrue(
                    torch.equal(recipient.state_dict()[name], donor_state[name])
                )
        self.assertTrue(
            torch.equal(recipient.outcome_embedding.weight, outcome_before)
        )

    def test_development_gate_rejects_every_allocation_operand(self) -> None:
        cases: dict[str, dict[str, object]] = {}
        replacements = {
            "event_count": 47,
            "persistent_unique_winning_slots": 47,
            "minimum_local_unique_winning_slots": 5,
            "no_reuse_before_capacity": False,
            "unused_slots_after_persistent": 15,
            "maximum_nonzero_write_slots": 2,
            "stress_step": 71,
            "stress_bounds_valid": False,
        }
        for key, value in replacements.items():
            metrics = copy.deepcopy(_passing_allocation_metrics())
            metrics["allocation"][key] = value
            cases[key] = metrics

        low_effective = copy.deepcopy(_passing_allocation_metrics())
        low_effective["allocation"]["aggregate_write"]["effective_slots"] = 31.999999
        cases["effective_slots"] = low_effective

        concentrated = copy.deepcopy(_passing_allocation_metrics())
        concentrated["allocation"]["aggregate_write"]["maximum_share"] = 0.050001
        cases["maximum_share"] = concentrated

        for key in (
            "slot_permutation_read_value_difference",
            "slot_permutation_residual_difference",
            "slot_permutation_score_difference",
        ):
            metrics = copy.deepcopy(_passing_allocation_metrics())
            metrics["allocation"][key] = 1.000001e-6
            cases[key] = metrics
        cases["missing_allocation"] = {}

        with mock.patch.object(v10.v9, "_development_gate", return_value=True):
            self.assertTrue(v10._development_gate(_passing_allocation_metrics(), True))
            for label, metrics in cases.items():
                with self.subTest(label=label):
                    self.assertFalse(v10._development_gate(metrics, True))

    def test_allocation_metrics_exercise_the_complete_forty_eight_event_horizon(self) -> None:
        torch.manual_seed(1901)
        core = DncAllocatedFactorizedCausalMemoryCore(
            content_width=6,
            temporal_width=8,
        )

        metrics = v10._allocation_metrics(core, _fake_rows())

        self.assertEqual(metrics["event_count"], 48)
        self.assertEqual(metrics["persistent_unique_winning_slots"], 48)
        self.assertEqual(metrics["minimum_local_unique_winning_slots"], 6)
        self.assertIs(metrics["no_reuse_before_capacity"], True)
        self.assertEqual(metrics["unused_slots_after_persistent"], 16)
        self.assertEqual(metrics["maximum_nonzero_write_slots"], 1)
        self.assertLessEqual(
            metrics["slot_permutation_read_value_difference"],
            1.0e-6,
        )
        self.assertLessEqual(
            metrics["slot_permutation_residual_difference"],
            1.0e-6,
        )
        self.assertLessEqual(
            metrics["slot_permutation_score_difference"],
            1.0e-6,
        )
        self.assertEqual(metrics["stress_step"], 72)
        self.assertIs(metrics["stress_bounds_valid"], True)
        self.assertEqual(len(metrics["evaluator_only_read_attribution"]), 8)
        self.assertEqual(v10.v7.MEMORY_SLOTS, 16)

    def test_source_hashes_cover_every_inherited_executable_dependency(self) -> None:
        expected = {
            "runner",
            "core",
            "corpus",
            "v9_runner",
            "v8_runner",
            "v7_runner",
            "v6_runner",
            "v5_runner",
            "v9_core",
            "v8_core",
            "v7_core",
            "situated_feedback",
            "v5_corpus",
            "qwen_runtime",
        }

        hashes = v10._source_hashes()

        self.assertEqual(set(hashes), expected)
        self.assertTrue(
            all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in hashes.values()
            )
        )

    def test_development_authorization_validates_every_bound_identity(self) -> None:
        source_hashes = {"runner": "source-digest"}
        valid = _valid_development_record(source_hashes)
        checkpoint = Path("synthetic-v10-checkpoint.pt")

        with (
            mock.patch.object(v10, "_source_hashes", return_value=source_hashes),
            mock.patch.object(
                v10.v7.v6.v5,
                "_sha256",
                return_value="checkpoint-digest",
            ),
            mock.patch.object(v10, "_development_gate", return_value=True),
        ):
            v10._validate_development_authorization(valid, checkpoint)

            cases = {}
            wrong_identity = copy.deepcopy(valid)
            wrong_identity["identity"] = v10.v9.IDENTITY
            cases["identity"] = wrong_identity

            wrong_phase = copy.deepcopy(valid)
            wrong_phase["phase"] = "final"
            cases["phase"] = wrong_phase

            wrong_classification = copy.deepcopy(valid)
            wrong_classification["classification"] = "DEVELOPMENT_NOT_SUPPORTED"
            cases["classification"] = wrong_classification

            wrong_source = copy.deepcopy(valid)
            wrong_source["source_hashes"] = {"runner": "different-source"}
            cases["source_hashes"] = wrong_source

            not_authorized = copy.deepcopy(valid)
            not_authorized["development_authorized"] = False
            cases["development_authorized"] = not_authorized

            wrong_checkpoint = copy.deepcopy(valid)
            wrong_checkpoint["checkpoint_sha256"] = "different-checkpoint"
            cases["checkpoint_sha256"] = wrong_checkpoint

            inexact = copy.deepcopy(valid)
            inexact["identity_checks"]["exact"] = False
            cases["identity_exact"] = inexact

            changed_foundation = copy.deepcopy(valid)
            changed_foundation["identity_checks"]["foundation_after"] = "changed"
            cases["foundation_identity"] = changed_foundation

            changed_core = copy.deepcopy(valid)
            changed_core["identity_checks"]["core_after_evaluation"] = "changed"
            cases["core_identity"] = changed_core

            missing_metrics = copy.deepcopy(valid)
            missing_metrics["development_metrics"] = None
            cases["development_metrics"] = missing_metrics

            for label, record in cases.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, "final partition remains sealed"):
                        v10._validate_development_authorization(record, checkpoint)

        with (
            mock.patch.object(v10, "_source_hashes", return_value=source_hashes),
            mock.patch.object(
                v10.v7.v6.v5,
                "_sha256",
                return_value="checkpoint-digest",
            ),
            mock.patch.object(v10, "_development_gate", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "final partition remains sealed"):
                v10._validate_development_authorization(valid, checkpoint)

    def test_final_partition_remains_sealed_by_default(self) -> None:
        corpus = v10.build_causal_neuromodulated_apprenticeship_v6()
        with self.assertRaises(FinalPartitionSealedError):
            _ = corpus.final


if __name__ == "__main__":
    unittest.main()
