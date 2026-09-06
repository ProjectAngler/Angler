from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from angler.reasoning.oml_natural_trace_representation import (
    OMLNaturalTraceRepresentation,
)
from experiments.corpora.scaled_oml_natural_trace_v12 import (
    FinalPartitionSealedError,
    build_scaled_oml_natural_trace_v12,
)
from experiments.runners import scaled_oml_natural_trace_representation_v12 as v12


class _FakeQwen:
    def __init__(self, width: int = 8) -> None:
        self.width = width
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts) -> torch.Tensor:
        texts = tuple(texts)
        self.calls.append(texts)
        return torch.tensor(
            [
                [
                    float((sum(ord(char) for char in text[index:: self.width]) + index) % 101)
                    / 101.0
                    for index in range(self.width)
                ]
                for text in texts
            ],
            dtype=torch.float32,
        )


def _encoded_row(index: int, *, family: str | None = None) -> v12.EncodedMechanism:
    generator = torch.Generator().manual_seed(12_000 + index)
    mask = torch.ones(6, 4, dtype=torch.bool)
    outcomes = torch.tensor((-1.0, 1.0, -1.0, 1.0, 1.0, -1.0))
    return v12.EncodedMechanism(
        mechanism_ref=f"v12-row-{index}",
        generator_family=family or f"family-{index // 4:02d}",
        transition_type=("insertion", "removal", "reorder", "replacement")[(index // 12) % 4],
        step_features=torch.randn(6, 4, 8, generator=generator),
        step_mask=mask,
        outcomes=outcomes,
    )


def _passing_metrics() -> dict[str, object]:
    arms = {
        v12.ARM_SECOND_ONLINE: {"online_loss_auc": 0.50, "balanced_accuracy": 0.80},
        v12.ARM_FIRST_ONLINE: {"online_loss_auc": 0.60, "balanced_accuracy": 0.75},
        v12.ARM_SOURCE_ONLINE: {"online_loss_auc": 0.70, "balanced_accuracy": 0.65},
        v12.ARM_SECOND_NO_UPDATE: {"online_loss_auc": 0.65, "balanced_accuracy": 0.80},
        v12.ARM_FIRST_NO_UPDATE: {"online_loss_auc": 0.70, "balanced_accuracy": 0.70},
        v12.ARM_SOURCE_NO_UPDATE: {"online_loss_auc": 0.75, "balanced_accuracy": 0.65},
        v12.ARM_SHUFFLED: {"online_loss_auc": 0.60, "balanced_accuracy": 0.70},
        v12.ARM_DIRECTION_REMOVED: {"online_loss_auc": 0.55, "balanced_accuracy": 0.80},
        v12.ARM_SEMANTICS_REMOVED: {"online_loss_auc": 0.55, "balanced_accuracy": 0.80},
    }
    return {
        "mechanism_count": 48,
        "arms": arms,
        "improved_panel_count": 3,
        "all_panels_nonregressed": True,
        "improved_family_count": 9,
        "transition_improved_family_counts": {
            "insertion": 3,
            "removal": 2,
            "reorder": 2,
            "replacement": 2,
        },
        "representation": {
            "effective_rank": 8.0,
            "mean_off_diagonal_cosine": 0.95,
            "fraction_distinct_pairs_above_0_999": 0.10,
            "finite_nonzero_variance_dimensions": 32,
            "minimum_dimension_variance": 1.0e-12,
            "reorder_minus_paraphrase_distance": 0.05,
            "padding_exact": True,
            "repeat_exact": True,
        },
        "protocol_invariants": {
            "paired_starts_exact": True,
            "paired_data_and_exposure_exact": True,
            "fast_head_only_inner_mutation": True,
            "rln_only_outer_mutation": True,
            "source_unchanged": True,
            "final_sealed_during_development": True,
            "learner_boundary_exact": True,
            "public_overlap_audit_passed": True,
            "inner_connectivity": {
                "raw_loss_equal": True,
                "raw_fast_gradient_equal": True,
                "updated_fast_weight_equal": True,
                "adamw_moments_equal": True,
                "second_order_hessian_path_nonzero": True,
                "first_order_hessian_path_absent": True,
            },
            "direct_outer_gradients_nonzero": True,
            "full_split_maximum_delta": 1.0e-6,
        },
        "shuffled_control_integrity": {
            "same_trace_order": True,
            "same_probe_labels": True,
            "all_support_associations_changed": True,
            "balanced_multiset_preserved": True,
        },
    }


class ScaledOmlNaturalTraceRepresentationV12RunnerTests(unittest.TestCase):
    def test_constants_robust_objective_and_frozen_schedule(self) -> None:
        self.assertEqual(v12.OUTER_UPDATES, 192)
        self.assertEqual(v12.INNER_STEPS, 8)
        self.assertEqual(v12.OUTER_MECHANISMS, 8)
        losses = torch.tensor((0.2, 0.4, 0.1, 0.8))
        expected = 0.5 * losses.mean() + 0.5 * 0.05 * (
            torch.logsumexp(losses / 0.05, dim=0) - torch.log(torch.tensor(4.0))
        )
        torch.testing.assert_close(v12._robust_objective(losses, 4), expected)

        corpus = build_scaled_oml_natural_trace_v12()
        index = {
            mechanism.metadata.mechanism_ref: position
            for position, mechanism in enumerate(corpus.train_inner)
        }
        schedule = tuple(
            tuple(index[row.metadata.mechanism_ref] for row in corpus.train_inner_for_update(update))
            for update in range(192)
        )
        exposures = v12._validate_inner_schedule(schedule)
        self.assertEqual(set(exposures), {4})
        self.assertTrue(
            all(
                len({corpus.train_inner[position].metadata.generator_family for position in update}) == 8
                for update in schedule
            )
        )
        with self.assertRaises(FinalPartitionSealedError):
            _ = corpus.final

    def test_encoder_batches_only_raw_parsed_step_sentences(self) -> None:
        episodes = []
        diagnostics = []
        for index, outcome in enumerate((-1, 1, -1, 1, 1, -1)):
            diagnostic = f"HIDDEN DIAGNOSTIC {index}"
            diagnostics.append(diagnostic)
            episodes.append(
                SimpleNamespace(
                    action_trace_text=(
                        f"Step 1: inspect public item {index}.\n"
                        f"Step 2: verify public item {index}."
                    ),
                    outcome_value=outcome,
                    objective_diagnostic_text=diagnostic,
                )
            )
        mechanism = SimpleNamespace(
            public=SimpleNamespace(episodes=tuple(episodes)),
            metadata=SimpleNamespace(
                mechanism_ref="HIDDEN-MECHANISM-ID",
                generator_family="HIDDEN-FAMILY-ID",
                heldout_variant="reorder",
            ),
        )
        qwen = _FakeQwen()
        row = v12._encode_mechanisms(qwen, (mechanism,))[0]
        flattened = qwen.calls[0]
        self.assertEqual(len(flattened), 12)
        self.assertTrue(all(text.startswith(("inspect", "verify")) for text in flattened))
        self.assertTrue(all(value not in flattened for value in diagnostics))
        self.assertNotIn("HIDDEN-MECHANISM-ID", flattened)
        self.assertNotIn("HIDDEN-FAMILY-ID", flattened)
        self.assertEqual(row.step_features.shape, (6, 8, 8))
        self.assertEqual(row.step_mask.sum(dim=-1).tolist(), [2] * 6)

    def test_paired_connectivity_and_full_split_gradients(self) -> None:
        torch.manual_seed(v12.SEED)
        second = OMLNaturalTraceRepresentation(step_width=8)
        first = copy.deepcopy(second)
        rows = tuple(_encoded_row(index) for index in range(16))

        connectivity = v12._inner_connectivity_preflight(second, first, rows[0])
        self.assertTrue(all(connectivity.values()))
        for second_order in (True, False):
            with self.subTest(second_order=second_order):
                report = v12._full_split_equivalence(
                    second if second_order else first,
                    rows[:8],
                    rows[8:16],
                    second_order=second_order,
                )
                self.assertLessEqual(report["objective_absolute_delta"], 1.0e-6)
                self.assertLessEqual(report["maximum_gradient_absolute_delta"], 1.0e-6)

    def test_panel_layout_has_same_and_cross_transition_pairs(self) -> None:
        rows = tuple(_encoded_row(index) for index in range(48))
        panels = v12._panel_pairs(rows)
        self.assertEqual(len(panels), 4)
        self.assertTrue(all(len(panel) == 12 for panel in panels))
        self.assertTrue(
            all(left.generator_family == right.generator_family for panel in panels[:2] for left, right in panel)
        )
        self.assertTrue(
            all(left.transition_type != right.transition_type for panel in panels[2:] for left, right in panel)
        )
        self.assertTrue(all(left.mechanism_ref != right.mechanism_ref for panel in panels for left, right in panel))

    def test_shuffle_is_a_nonidentity_label_permutation_only(self) -> None:
        outcomes = torch.tensor((-1.0, 1.0, -1.0, 1.0, 1.0, -1.0))
        shuffled = v12._shuffled_outcomes(outcomes)
        self.assertFalse(torch.equal(outcomes, shuffled))
        self.assertEqual(sorted(outcomes.tolist()), sorted(shuffled.tolist()))
        self.assertTrue(torch.equal(shuffled, outcomes[list(v12.SHUFFLE_PERMUTATION)]))

    def test_development_gate_accepts_boundaries_and_rejects_new_operands(self) -> None:
        passing = _passing_metrics()
        self.assertTrue(v12._development_gate(passing, True))
        self.assertEqual(
            v12._development_classification(passing, authorized=True),
            "DEVELOPMENT_GATE_PASSED",
        )
        cases = []
        for path, value in (
            (("arms", v12.ARM_SECOND_ONLINE, "online_loss_auc"), 0.57),
            (("improved_panel_count",), 2),
            (("all_panels_nonregressed",), False),
            (("representation", "effective_rank"), 7.999),
            (("representation", "mean_off_diagonal_cosine"), 0.951),
            (("representation", "fraction_distinct_pairs_above_0_999"), 0.101),
            (("representation", "finite_nonzero_variance_dimensions"), 31),
            (("representation", "reorder_minus_paraphrase_distance"), 0.0499),
            (("improved_family_count",), 8),
            (("protocol_invariants", "full_split_maximum_delta"), 1.0001e-6),
            (("shuffled_control_integrity", "same_trace_order"), False),
        ):
            candidate = copy.deepcopy(passing)
            cursor = candidate
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            cases.append((path, candidate))
        for path, candidate in cases:
            with self.subTest(path=path):
                self.assertFalse(v12._development_gate(candidate, True))
        self.assertFalse(v12._development_gate(passing, False))
        collapsed = copy.deepcopy(passing)
        collapsed["representation"]["effective_rank"] = 1.0
        self.assertEqual(
            v12._development_classification(collapsed, authorized=False),
            "DEVELOPMENT_SHORTCUT_CODE_COLLAPSE",
        )

    def test_atomic_outputs_and_final_authorization_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            checkpoint_path = root / "checkpoint.pt"
            v12._atomic_json(json_path, {"value": 1})
            v12._atomic_torch_save(checkpoint_path, {"value": torch.tensor(2)})
            self.assertEqual(json.loads(json_path.read_text())["value"], 1)
            self.assertEqual(torch.load(checkpoint_path, weights_only=True)["value"].item(), 2)
            self.assertFalse(any(path.name.startswith("result.json.tmp") for path in root.iterdir()))
            with self.assertRaises(FileExistsError):
                v12._atomic_json(json_path, {"value": 2})

            incomplete = {
                "identity": v12.IDENTITY,
                "phase": "train-development",
                "classification": "DEVELOPMENT_GATE_PASSED",
                "development_authorized": True,
                "source_hashes": {"runner": "source"},
                "checkpoint_sha256": "checkpoint",
                "development_metrics": {},
                "identity_checks": {"exact": True},
            }
            with (
                mock.patch.object(v12, "_source_hashes", return_value={"runner": "source"}),
                mock.patch.object(v12, "_sha256", return_value="checkpoint"),
                self.assertRaises(RuntimeError),
            ):
                v12._validate_final_authorization(incomplete, checkpoint_path)

    def test_source_hashes_cover_new_and_borrowed_executable_chain(self) -> None:
        hashes = v12._source_hashes()
        self.assertEqual(
            set(hashes),
            {
                "runner",
                "core",
                "trace_graph_core",
                "corpus",
                "functional_adamw",
                "adamw_slot",
                "qwen_runtime",
                "qwen_loader",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
