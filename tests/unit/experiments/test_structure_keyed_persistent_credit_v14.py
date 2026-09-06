from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import torch

from angler.reasoning.structure_keyed_credit_memory import (
    StructureKeyedCreditMemoryCore,
)
from experiments.runners import structure_keyed_persistent_credit_v14 as v14


def _structure_pass() -> dict[str, object]:
    full = {
        "correct": 210,
        "count": 240,
        "accuracy": 0.875,
        "renderer_pair_marginals": {
            str(index): {"accuracy": 0.75} for index in range(8)
        },
        "target_position_marginals": {
            str(index): {"accuracy": 0.75} for index in range(5)
        },
        "paired_order_win_rate": 0.80,
        "paired_order_mean_margin": 0.01,
        "corruptions": {
            name: {"win_rate": 0.75, "mean_distance_margin": 0.05}
            for name in v14.CORRUPTIONS
        },
        "geometry": {
            "effective_rank": 8.0,
            "all_finite": True,
            "finite_nonzero_variance_dimensions": 32,
            "mean_off_diagonal_cosine": 0.95,
            "fraction_distinct_pairs_above_0_999": 0.10,
        },
    }
    return {
        "full": full,
        "direction_removed": {"paired_order_win_rate": 0.69},
        "step_semantics_removed": {"accuracy": 0.67},
    }


def _credit_pass() -> dict[str, object]:
    def arm(balanced: float, auc: float) -> dict[str, object]:
        return {
            "balanced_accuracy": balanced,
            "online_loss_auc": auc,
            "finite_scores": True,
            "read_weight_min": 0.0,
            "read_weight_max": 1.0,
            "write_strength_min": 0.0,
            "write_strength_max": 1.0,
        }
    return {
        "mechanism_count": 48,
        "event_count": 288,
        "arms": {
            "true": arm(0.70, 0.490),
            "shuffled": arm(0.64, 0.495),
            "zero_no_read": arm(0.64, 0.495),
            "per_mechanism_reset": arm(0.64, 0.500),
        },
        "improved_family_count": 9,
        "transition_improved_family_counts": {
            "insertion": 3,
            "removal": 2,
            "reorder": 2,
            "replacement": 2,
        },
        "terminal_removal_and_swaps": {
            "true_balanced_accuracy": 0.75,
            "zero_balanced_accuracy": 0.69,
            "gain": 0.06,
            "matched_state_digest_exact": True,
            "matched_gain_retention": 0.90,
            "unrelated_balanced_accuracy": 0.69,
            "key_shuffled_balanced_accuracy": 0.75,
        },
        "replay": {"exact": True, "maximum_logit_error": 1.0e-6},
        "retention": {"drop": 0.05},
        "candidate_order": {"changed_fraction": 0.10},
        "state_metrics": {
            "finite": True,
            "step": 288,
            "keys_max_abs": 1.0,
            "values_max_abs": 1.0,
            "usage_min": 0.0,
            "usage_max": 1.0,
            "acquisition_min": 0.0,
            "acquisition_max": 1.0,
        },
        "feedback_integrity": {
            "same_visible_inputs": True,
            "same_true_evaluation_labels": True,
            "balanced_shuffled_feedback": True,
        },
        "protocol_invariants": {
            "exact_schedule": True,
            "all_training_gradients_finite": True,
            "train_dev_refs_disjoint": True,
            "final_sealed": True,
            "only_v14_core_trainable": True,
        },
    }


class ScheduleAndBoundaryTests(unittest.TestCase):
    def test_coprime_schedule_exposes_every_mechanism_exactly_twice(self) -> None:
        schedule = v14._training_schedule()
        self.assertEqual(len(schedule), 96)
        self.assertTrue(all(len(row) == 8 and len(set(row)) == 8 for row in schedule))
        counts = {index: 0 for index in range(384)}
        for row in schedule:
            for index in row:
                counts[index] += 1
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(schedule[0], tuple((index * 49) % 384 for index in range(8)))

    def test_temporal_features_exclude_outcome_and_metadata(self) -> None:
        @dataclass
        class Coordinates:
            acquired_ordinal: int = 3
            age: int = 2

        @dataclass
        class Episode:
            temporal: Coordinates
            outcome_label: str
            secret_family: str

        left = Episode(Coordinates(), "SUCCESS", "one")
        right = Episode(Coordinates(), "FAILURE", "two")
        self.assertEqual(v14._episode_temporal(left, 4), v14._episode_temporal(right, 4))
        self.assertEqual(len(v14._episode_temporal(left, 4)), v14.TEMPORAL_WIDTH)

    def test_structure_gate_accepts_boundary_and_rejects_each_clause(self) -> None:
        passing = _structure_pass()
        self.assertTrue(v14._structure_gate(passing, True))
        failures = []
        value = copy.deepcopy(passing); value["full"]["correct"] = 199; failures.append(value)
        value = copy.deepcopy(passing); value["full"]["renderer_pair_marginals"]["0"]["accuracy"] = .749; failures.append(value)
        value = copy.deepcopy(passing); value["full"]["target_position_marginals"]["0"]["accuracy"] = .749; failures.append(value)
        value = copy.deepcopy(passing); value["full"]["corruptions"]["insertion"]["mean_distance_margin"] = .049; failures.append(value)
        value = copy.deepcopy(passing); value["full"]["geometry"]["effective_rank"] = 7.99; failures.append(value)
        value = copy.deepcopy(passing); value["direction_removed"]["paired_order_win_rate"] = .71; failures.append(value)
        value = copy.deepcopy(passing); value["step_semantics_removed"]["accuracy"] = .68; failures.append(value)
        failures.append(passing)
        for index, metrics in enumerate(failures):
            with self.subTest(index=index):
                identity = index != len(failures) - 1
                self.assertFalse(v14._structure_gate(metrics, identity))

    def test_persistent_gate_accepts_boundary_and_rejects_each_clause(self) -> None:
        passing = _credit_pass()
        self.assertTrue(v14._persistent_gate(passing, True))
        mutations = []
        value = copy.deepcopy(passing); value["arms"]["true"]["balanced_accuracy"] = .699; mutations.append(value)
        value = copy.deepcopy(passing); value["arms"]["shuffled"]["balanced_accuracy"] = .70; value["arms"]["shuffled"]["online_loss_auc"] = .49; mutations.append(value)
        value = copy.deepcopy(passing); value["improved_family_count"] = 8; mutations.append(value)
        value = copy.deepcopy(passing); value["transition_improved_family_counts"]["reorder"] = 1; mutations.append(value)
        value = copy.deepcopy(passing); value["arms"]["per_mechanism_reset"]["balanced_accuracy"] = .651; mutations.append(value)
        value = copy.deepcopy(passing); value["terminal_removal_and_swaps"]["gain"] = .049; mutations.append(value)
        value = copy.deepcopy(passing); value["replay"]["exact"] = False; mutations.append(value)
        value = copy.deepcopy(passing); value["terminal_removal_and_swaps"]["matched_gain_retention"] = .899; mutations.append(value)
        value = copy.deepcopy(passing); value["terminal_removal_and_swaps"]["unrelated_balanced_accuracy"] = .75; value["terminal_removal_and_swaps"]["key_shuffled_balanced_accuracy"] = .75; mutations.append(value)
        value = copy.deepcopy(passing); value["retention"]["drop"] = .051; mutations.append(value)
        value = copy.deepcopy(passing); value["candidate_order"]["changed_fraction"] = .099; mutations.append(value)
        value = copy.deepcopy(passing); value["state_metrics"]["usage_max"] = 1.001; mutations.append(value)
        value = copy.deepcopy(passing); value["feedback_integrity"]["same_visible_inputs"] = False; mutations.append(value)
        mutations.append(passing)
        for index, metrics in enumerate(mutations):
            with self.subTest(index=index):
                identity = index != len(mutations) - 1
                self.assertFalse(v14._persistent_gate(metrics, identity))

    def test_independent_classifications(self) -> None:
        self.assertEqual(v14._classification(True, True), "DEVELOPMENT_GATE_PASSED")
        self.assertEqual(v14._classification(True, False), "STRUCTURE_REPLICATED_NOT_ADAPTIVE")
        self.assertEqual(v14._classification(False, True), "PERSISTENT_CREDIT_WITHOUT_STRUCTURE")
        self.assertEqual(v14._classification(False, False), "DEVELOPMENT_NOT_SUPPORTED")


class MemoryControlTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.core = StructureKeyedCreditMemoryCore(temporal_width=4)

    def test_predict_precedes_feedback_and_replay_is_exact(self) -> None:
        state = self.core.initial_state()
        events = []
        logits = []
        for index in range(6):
            relation = torch.randn(1, 64)
            temporal = torch.tensor([[index / 5, (5 - index) / 5, 0.0, 1.0]])
            base = torch.tensor([(-1.0) ** index * .2])
            outcome = torch.tensor([1.0 if index % 2 == 0 else -1.0])
            output = self.core.predict(relation, temporal, base, state=state)
            self.assertEqual(output.state_step, index)
            logits.append(output.logits.detach())
            state, event = self.core.apply_feedback(
                state, output, outcome, evidence_refs=f"test:{index}"
            )
            events.append(event)
        replayed, outputs = self.core.replay(events)
        self.assertEqual(self.core.state_digest(state), self.core.state_digest(replayed))
        self.assertTrue(all(torch.allclose(left, right.logits, atol=1e-6, rtol=0) for left, right in zip(logits, outputs, strict=True)))

    def test_key_shuffle_preserves_values_and_breaks_only_key_binding(self) -> None:
        state = self.core.initial_state()
        output = self.core.predict(torch.randn(1, 64), torch.zeros(1, 4), torch.zeros(1), state=state)
        state, _ = self.core.apply_feedback(state, output, torch.ones(1), evidence_refs="x")
        shuffled = v14._key_shuffled_state(state)
        self.assertTrue(torch.equal(shuffled.values, state.values))
        self.assertTrue(torch.equal(shuffled.usage, state.usage))
        self.assertTrue(torch.equal(shuffled.acquisition, state.acquisition))
        self.assertTrue(torch.equal(shuffled.keys, state.keys.flip(0)))

    def test_atomic_output_refuses_identity_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            v14._atomic_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                v14._atomic_json(path, {"ok": False})

    def test_final_authorization_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"not-authorized")
            with self.assertRaises(RuntimeError):
                v14._validate_final_authorization({}, checkpoint)


if __name__ == "__main__":
    unittest.main()
