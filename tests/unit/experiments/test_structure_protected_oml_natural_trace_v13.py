from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from angler.reasoning.structure_protected_oml import StructureProtectedOMLCore
from experiments.corpora.structure_protected_oml_natural_trace_v13 import (
    FinalPartitionSealedError,
    build_structure_protected_oml_natural_trace_v13,
)
from experiments.runners import structure_protected_oml_natural_trace_v13 as v13


class _FakeQwen:
    def __init__(self, width: int = 8) -> None:
        self.width = width
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts) -> torch.Tensor:
        values = tuple(texts)
        self.calls.append(values)
        return torch.tensor(
            [
                [
                    float((sum(ord(char) for char in text[index:: self.width]) + index) % 97)
                    / 97.0
                    for index in range(self.width)
                ]
                for text in values
            ],
            dtype=torch.float32,
        )


def _row(index: int) -> v13.EncodedPairMechanism:
    generator = torch.Generator().manual_seed(13_000 + index)
    outcomes = torch.tensor((-1.0, 1.0, -1.0, 1.0, 1.0, -1.0))
    mask = torch.ones(6, 4, dtype=torch.bool)
    structural = torch.randn(6, 4, 8, generator=generator)
    structural[1] = structural[0] + 0.01
    return v13.EncodedPairMechanism(
        mechanism_ref=f"row-{index}",
        generator_family=f"family-{index // 4:02d}",
        transition_group=v13.TRANSITIONS[(index // 12) % 4],
        renderer_position_strata=tuple(f"r{value}" for value in range(6)),
        corruption_types=("reorder", "paraphrase", "replacement", "paraphrase", "paraphrase", "omission"),
        reference_features=torch.randn(6, 4, 8, generator=generator),
        reference_mask=mask.clone(),
        attempt_features=torch.randn(6, 4, 8, generator=generator),
        attempt_mask=mask.clone(),
        outcomes=outcomes,
        structural_features=structural,
        structural_mask=mask.clone(),
    )


def _retrieval_report(accuracy: float, hard_margin: float) -> dict[str, object]:
    return {
        "correct": 43,
        "count": 48,
        "accuracy": accuracy,
        "paired_order_win_rate": 0.80,
        "mean_cosine_score_margin": hard_margin,
        "corruption_distance_margins": {
            name: hard_margin for name in v13.CORRUPTIONS
        },
        "by_transition": {
            name: {"correct": 10, "count": 12, "accuracy": 10 / 12}
            for name in v13.TRANSITIONS
        },
        "by_renderer_position": {
            "stratum-a": {"correct": 12, "count": 12, "accuracy": 1.0},
            "stratum-b": {"correct": 12, "count": 12, "accuracy": 1.0},
        },
        "geometry": {
            "effective_rank": 8.0,
            "finite_nonzero_variance_dimensions": 32,
            "minimum_dimension_variance": 1.0e-9,
            "mean_off_diagonal_cosine": 0.95,
            "fraction_distinct_pairs_above_0_999": 0.10,
        },
    }


def _passing_metrics() -> dict[str, object]:
    arms = {
        v13.ARM_PROTECTED_SECOND: {
            "online_loss_auc": 0.50,
            "balanced_accuracy": 0.80,
            "same_length_balanced_accuracy": 0.70,
            "reorder_accuracy": 0.80,
        },
        v13.ARM_PROTECTED_FIRST: {"online_loss_auc": 0.60},
        v13.ARM_PROTECTED_SECOND_NO: {"online_loss_auc": 0.65, "balanced_accuracy": 0.80},
        v13.ARM_SOURCE: {"online_loss_auc": 0.70},
        v13.ARM_SOURCE_NO: {"balanced_accuracy": 0.65},
        v13.ARM_SHUFFLED: {"online_loss_auc": 0.51, "balanced_accuracy": 0.70},
        v13.ARM_DIRECTION_REMOVED: {"online_loss_auc": 0.50, "reorder_accuracy": 0.70},
        v13.ARM_SEMANTICS_REMOVED: {
            "online_loss_auc": 0.50,
            "same_length_balanced_accuracy": 0.60,
        },
    }
    split = {"objective_absolute_delta": 1.0e-6, "maximum_gradient_absolute_delta": 1.0e-6}
    return {
        "arms": arms,
        "improved_panel_count": 3,
        "all_panels_nonregressed": True,
        "improved_family_count": 9,
        "transition_improved_family_counts": {name: 2 for name in v13.TRANSITIONS},
        "behavior_family_success_count": 9,
        "behavior_transition_success_counts": {name: 2 for name in v13.TRANSITIONS},
        "length_only_baseline_balanced_accuracy": 0.50,
        "retrieval": {
            "protected": _retrieval_report(0.90, 0.10),
            "outcome_only": _retrieval_report(0.60, 0.00),
            "source": _retrieval_report(0.60, 0.00),
        },
        "feedback_integrity": {
            "same_support_inputs": True,
            "same_probe_labels": True,
            "all_support_associations_changed": True,
            "balanced_multiset_preserved": True,
        },
        "protocol_invariants": {
            "corpus_audit": {"passed": True},
            "protected_encoders_exact": True,
            "protected_encoder_optimizers_exact": True,
            "source_unchanged": True,
            "final_sealed": True,
            "ownership_exact": True,
            "padding_repeat": {"padding_exact": True, "repeat_exact": True},
            "preflight": {
                "connectivity": {
                    "paired_inputs_exact": True,
                    "raw_loss_exact": True,
                    "raw_fast_gradient_exact": True,
                    "updated_fast_exact": True,
                    "moments_exact": True,
                    "second_trunk_hessian_nonzero": True,
                    "first_trunk_hessian_absent": True,
                },
                "structural_encoder_gradient_nonzero": True,
                "direct_outer_trunk_gradient_nonzero": True,
                "structure_cross_owner_exact": True,
                "trunk_cross_owner_exact": True,
                "infonce_serialization_invariant": True,
                "protected_second_full_split": copy.deepcopy(split),
                "protected_first_full_split": copy.deepcopy(split),
                "outcome_control_full_split": copy.deepcopy(split),
            },
        },
    }


class StructureProtectedOmlNaturalTraceV13RunnerTests(unittest.TestCase):
    def test_frozen_schedule_counts_and_final_seal(self) -> None:
        corpus = build_structure_protected_oml_natural_trace_v13()
        index = {
            mechanism.metadata.mechanism_ref: position
            for position, mechanism in enumerate(corpus.train_inner)
        }
        schedule = tuple(
            tuple(index[value.metadata.mechanism_ref] for value in corpus.train_inner_for_update(update))
            for update in range(v13.OUTER_UPDATES)
        )
        uses = v13._validate_schedule(schedule)
        self.assertEqual(set(uses), {4})
        self.assertTrue(all(len(set(update)) == 8 for update in schedule))
        outer = tuple(
            corpus.train_outer(update, slot)
            for update in range(v13.OUTER_UPDATES)
            for slot in range(v13.OUTER_MECHANISMS)
        )
        self.assertTrue(v13._corpus_audit(corpus.train_inner, outer, corpus.development)["passed"])
        with self.assertRaises(FinalPartitionSealedError):
            _ = corpus.final

    def test_public_encoder_excludes_hidden_metadata_and_diagnostics(self) -> None:
        corpus = build_structure_protected_oml_natural_trace_v13()
        mechanism = corpus.development[0]
        qwen = _FakeQwen()
        row = v13._encode_mechanisms(qwen, (mechanism,))[0]
        encoded_text = "\n".join(text for call in qwen.calls for text in call)
        self.assertNotIn(mechanism.metadata.mechanism_ref, encoded_text)
        self.assertNotIn(mechanism.metadata.generator_family, encoded_text)
        self.assertEqual(row.reference_features.shape, (6, 8, 8))
        self.assertEqual(row.attempt_features.shape, (6, 8, 8))
        self.assertEqual(row.structural_features.shape, (6, 8, 8))

    def test_symmetric_infonce_is_serialization_invariant_and_differentiable(self) -> None:
        generator = torch.Generator().manual_seed(13)
        codes = torch.randn(8, 6, 32, generator=generator, requires_grad=True)
        with torch.no_grad():
            codes[:, 1].copy_(codes[:, 0] + 0.01)
        loss = v13._symmetric_infonce_from_codes(codes)
        gradient = torch.autograd.grad(loss, codes)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(v13._serialization_invariance(codes.detach()))

    def test_owner_preflight_proves_partition_and_meta_gradient_mechanics(self) -> None:
        torch.manual_seed(v13.SEED)
        system = v13._build_system(8, torch.device("cpu"))
        inner = tuple(_row(index) for index in range(8))
        outer = tuple(_row(index + 8) for index in range(8))
        report = v13._owner_preflight(system, inner, outer)
        self.assertTrue(all(report["connectivity"].values()))
        self.assertTrue(report["structure_cross_owner_exact"])
        self.assertTrue(report["trunk_cross_owner_exact"])
        self.assertTrue(report["infonce_serialization_invariant"])
        for key in (
            "protected_second_full_split",
            "protected_first_full_split",
            "outcome_control_full_split",
        ):
            self.assertLessEqual(report[key]["maximum_gradient_absolute_delta"], 1.0e-6)

    def test_shuffle_is_balanced_nonidentity_permutation(self) -> None:
        outcomes = torch.tensor((-1.0, 1.0, -1.0, 1.0, 1.0, -1.0))
        shuffled = v13._shuffled_outcomes(outcomes)
        self.assertFalse(torch.equal(shuffled, outcomes))
        self.assertEqual(sorted(shuffled.tolist()), sorted(outcomes.tolist()))
        torch.testing.assert_close(shuffled, outcomes[list(v13.SHUFFLE_PERMUTATION)])

    def test_gate_accepts_boundaries_and_rejects_each_new_family(self) -> None:
        passing = _passing_metrics()
        self.assertTrue(v13._structural_gate(passing))
        self.assertTrue(v13._oml_gate(passing))
        self.assertTrue(v13._development_gate(passing, True))
        cases = (
            (("retrieval", "protected", "correct"), 39),
            (("retrieval", "protected", "paired_order_win_rate"), 0.749),
            (("retrieval", "protected", "geometry", "effective_rank"), 7.99),
            (("arms", v13.ARM_PROTECTED_SECOND, "same_length_balanced_accuracy"), 0.649),
            (("arms", v13.ARM_PROTECTED_SECOND_NO, "balanced_accuracy"), 0.749),
            (("improved_panel_count",), 2),
            (("feedback_integrity", "same_probe_labels"), False),
            (("protocol_invariants", "protected_encoders_exact"), False),
        )
        for path, value in cases:
            candidate = copy.deepcopy(passing)
            cursor = candidate
            for name in path[:-1]:
                cursor = cursor[name]
            cursor[path[-1]] = value
            with self.subTest(path=path):
                self.assertFalse(v13._development_gate(candidate, True))
        self.assertFalse(v13._development_gate(passing, False))

    def test_classification_tree_and_final_authorization_fail_closed(self) -> None:
        passing = _passing_metrics()
        self.assertEqual(
            v13._classification(passing, authorized=True), "DEVELOPMENT_GATE_PASSED"
        )
        oml_failed = copy.deepcopy(passing)
        oml_failed["arms"][v13.ARM_PROTECTED_SECOND_NO]["balanced_accuracy"] = 0.5
        self.assertEqual(
            v13._classification(oml_failed, authorized=False),
            "STRUCTURAL_REPRESENTATION_SUPPORTED_NOT_INTEGRATED",
        )
        structural_failed = copy.deepcopy(passing)
        structural_failed["retrieval"]["protected"]["correct"] = 1
        self.assertEqual(
            v13._classification(structural_failed, authorized=False),
            "DEVELOPMENT_SHORTCUT_CODE_COLLAPSE",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            incomplete = {
                "identity": v13.IDENTITY,
                "phase": "train-development",
                "classification": "DEVELOPMENT_GATE_PASSED",
                "development_authorized": True,
                "source_hashes": {"runner": "same"},
                "checkpoint_sha256": "same",
                "identity_checks": {"exact": True},
                "development_metrics": {},
            }
            with (
                mock.patch.object(v13, "_source_hashes", return_value={"runner": "same"}),
                mock.patch.object(v13, "_sha256", return_value="same"),
                self.assertRaises(RuntimeError),
            ):
                v13._validate_final_authorization(incomplete, checkpoint)

    def test_atomic_outputs_and_source_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            v13._atomic_json(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text())["value"], 1)
            self.assertFalse(any(value.name.startswith("result.json.tmp") for value in path.parent.iterdir()))
            with self.assertRaises(FileExistsError):
                v13._atomic_json(path, {"value": 2})
        hashes = v13._source_hashes()
        self.assertIn("runner", hashes)
        self.assertIn("core", hashes)
        self.assertIn("corpus", hashes)
        self.assertIn("v12_corpus", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
