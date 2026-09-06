from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from angler.reasoning.shared_semantic_metric_credit_memory import (
    SharedSemanticMetricCreditMemoryCore,
)
from experiments.corpora.paired_latent_contingency_credit_v15 import (
    build_paired_latent_contingency_credit_v15,
)
from experiments.runners import paired_latent_contingency_credit_v15 as v15
from experiments.runners import shared_semantic_metric_credit_v17 as v17
from experiments.runners import strict_key_value_credit_v16 as v16


def _fake_encoded(pair: object, seed: int) -> v15.EncodedTwinPair:
    generator = torch.Generator().manual_seed(30_000 + seed)
    alpha = torch.randn(64, generator=generator)
    beta = torch.randn(64, generator=generator)
    relation = torch.stack((alpha, beta, alpha, beta, alpha, beta))
    if pair.first.supervision.relation_classes[0] == "beta":
        relation = torch.stack((beta, alpha, beta, alpha, beta, alpha))
    base = torch.randn(6, generator=generator) * 0.1
    temporal = torch.tensor([
        v15._temporal(event, index)
        for index, event in enumerate(pair.first.public.events)
    ], dtype=torch.float32)

    def episode(value: object) -> v15.EncodedContingencyEpisode:
        return v15.EncodedContingencyEpisode(
            episode_ref=value.metadata.episode_ref,
            pair_ref=value.metadata.pair_ref,
            twin_ref=value.metadata.twin_ref,
            generator_family=value.metadata.generator_family,
            transition_group=value.metadata.transition_group,
            relation_features=relation.clone(),
            base_logits=base.clone(),
            temporal_features=temporal.clone(),
            outcomes=torch.tensor(value.supervision.outcome_values, dtype=torch.float32),
            relation_classes=tuple(value.supervision.relation_classes),
            public_payload_sha256=f"PAYLOAD-{value.metadata.pair_ref}",
            input_tensor_sha256=f"INPUT-{value.metadata.pair_ref}",
        )

    return v15.EncodedTwinPair(pair.pair_ref, episode(pair.first), episode(pair.second))


def _arm(ba: float, nll: float) -> dict[str, object]:
    return {
        "balanced_accuracy": ba, "mean_probe_nll": nll, "finite": True,
        "read_weight_min": 0.0, "read_weight_max": 1.0,
        "write_strength_min": 0.0, "write_strength_max": 1.0,
    }


def _strata(count: int) -> dict[str, object]:
    return {
        "count": count, "top1_accuracy": 0.65,
        "mean_matching_mass": 0.30,
        "mean_largest_registered_competitor_mass": 0.20,
        "mean_matching_minus_competitor_margin": 0.10,
    }


def _passing_metrics() -> dict[str, object]:
    return {
        "pair_count": 48,
        "arms": {
            "true": _arm(0.71, 0.20), "deranged": _arm(0.60, 0.23),
            "zero_no_read": _arm(0.60, 0.23),
            "post_acquisition_reset": _arm(0.60, 0.30),
            "outcome_blind_writer": _arm(0.60, 0.30),
        },
        "affine_calibrator": {"balanced_accuracy": 0.60},
        "outcome_blind_integrity": {
            "write_count_exact": True, "allocation_exact": True,
            "write_gate_exact": True, "erase_gate_exact": True,
            "write_keys_exact": True, "usage_exact": True,
            "acquisition_exact": True, "stored_values_differ": True,
            "shared_value_token_outcome_invariant": True,
            "only_stored_outcome_value_content_changed": True,
        },
        "paired_twin": {
            "directional_accuracy": 0.75,
            "mean_outcome_directed_logit_margin": 0.20,
        },
        "state_controls": {
            "matched_balanced_accuracy": 0.71,
            "zero_balanced_accuracy": 0.60,
            "unrelated_balanced_accuracy": 0.60,
            "key_value_mismatch_balanced_accuracy": 0.60,
            "full_causal_gain": 0.11,
            "opposite_twin_swap_reversal_fraction": 0.70,
            "opposite_twin_swap_gain_retention": 0.80,
            "mean_nll": {
                "matched": 0.20, "zero": 0.23, "unrelated": 0.23,
                "key_value_mismatch": 0.23,
            },
        },
        "improved_family_count": 9,
        "transition_improved_family_counts": {
            "insertion": 2, "removal": 2, "reorder": 2, "replacement": 2,
        },
        "ordering": {"changed_fraction": 0.25},
        "replay": {"exact": True, "maximum_logit_error": 0.0},
        "retention": {"maximum_drop": 0.05},
        "state_bounds": {
            "finite": True, "keys_max_abs": 1.0, "values_max_abs": 1.0,
            "usage_min": 0.0, "usage_max": 1.0,
            "acquisition_min": 0.0, "acquisition_max": 1.0,
        },
        "protocol_invariants": {
            "exact_training_schedule": True, "all_training_gradients_finite": True,
            "true_deranged_paths_differentiable": True,
            "zero_baseline_detached": True, "eight_pair_state_horizon": True,
        },
        "probe_relation_query_swap": {
            "balanced_accuracy": 0.60, "mean_probe_nll": 0.23,
            "probe_count": 384, "relation_queries_changed": True,
            "relation_inputs_changed": True, "read_strength_exact": True,
            "relation_source_exact": True, "opposite_relation_class_exact": True,
            "temporal_base_labels_exact": True,
            "state_and_stored_memory_exact": True,
            "only_read_query_swapped": True, "permutation": [3, 2, 5, 4],
        },
        "corresponding_anchor_read_mass": {
            "count": 384, "top1_accuracy": 0.75,
            "mean_matching_mass": 0.30,
            "mean_largest_registered_competitor_mass": 0.20,
            "mean_matching_minus_competitor_margin": 0.10,
            "by_probe_position": {str(index): _strata(96) for index in (2, 3, 4, 5)},
            "by_transition": {name: _strata(96) for name in (
                "insertion", "removal", "reorder", "replacement"
            )},
            "by_family": {f"family-{index}": _strata(32) for index in range(12)},
            "metadata_evaluator_only": True,
            "anchor_slot_lineage_exact": True,
            "prediction_before_sidecars_exact": True,
            "learner_state_unchanged": True,
        },
    }


class SharedMetricRunnerMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_paired_latent_contingency_credit_v15()

    def test_frozen_schedule_objective_and_fresh_identity(self) -> None:
        schedule = v15._training_schedule(self.corpus)
        self.assertEqual((len(schedule), len(schedule[0])), (96, 8))
        self.assertEqual(v17.SEED, 2026083117)
        self.assertEqual(v17.TRAIN_UPDATES, v15.TRAIN_UPDATES)
        self.assertEqual(v17.PAIRS_PER_UPDATE, v15.PAIRS_PER_UPDATE)
        self.assertEqual(v15.LOSS_WEIGHTS, (1.0, 1.0, 1.0))
        source = inspect.getsource(v17._run_train_development)
        self.assertIn("v15._fit(core, train, schedule)", source)
        self.assertNotIn("InfoNCE", source)

    def test_architecture_evidence_proves_shared_metric_and_gradients(self) -> None:
        core = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        pair = _fake_encoded(self.corpus.train[0], 0)
        report = v17._architecture_evidence(core, pair)
        self.assertTrue(report["exact"], report)
        self.assertTrue(all(report["shared_metric_invariance"].values()))
        self.assertTrue(all(report["required_gradient_prefixes_reached"].values()))
        self.assertTrue(report["erase_gradient_inactive_by_capacity"])
        architecture = report["architecture_report"]
        self.assertEqual(
            architecture["semantic_metric_shared_for"], ("read_query", "write_key")
        )
        self.assertFalse(architecture["independent_query_or_key_networks"])
        self.assertFalse(architecture["temporal_affects_semantic_code_or_decoder"])

    def test_correspondence_report_is_current_anchor_and_stratified(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, index)
            for index, pair in enumerate(self.corpus.development)
        )
        core = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        before = v17._model_digest(core)
        report = v17._corresponding_anchor_read_report(core, pairs)
        self.assertEqual(report["count"], 384)
        self.assertEqual(set(report["by_probe_position"]), {"2", "3", "4", "5"})
        self.assertEqual(set(report["by_transition"]), {
            "insertion", "removal", "reorder", "replacement",
        })
        self.assertEqual(len(report["by_family"]), 12)
        self.assertTrue(report["metadata_evaluator_only"])
        self.assertTrue(report["anchor_slot_lineage_exact"])
        self.assertTrue(report["prediction_before_sidecars_exact"])
        self.assertTrue(report["learner_state_unchanged"])
        self.assertEqual(before, v17._model_digest(core))
        source = inspect.getsource(v17._corresponding_anchor_read_report)
        prediction = source.index("output = core.predict")
        lineage_sidecars = source.index("slot_by_ref = event_slots")
        relation_sidecars = source.index("anchor_classes = tuple")
        self.assertLess(prediction, lineage_sidecars)
        self.assertLess(prediction, relation_sidecars)
        self.assertGreater(lineage_sidecars, source.index("for probe in PROBE_INDICES"))

    def test_correspondence_lineage_rejects_missing_or_duplicate_events(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, 500 + index)
            for index, pair in enumerate(self.corpus.development)
        )
        core = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        with v16._v15_control_scope():
            true = v15._arm_stream(core, pairs, arm="true")
        for mode in ("missing", "duplicate"):
            packet = copy.deepcopy(true)
            blocks = list(packet["event_blocks"])
            first = dict(blocks[0])
            events = list(first["first"])
            if mode == "missing":
                events.pop()
            else:
                events[-1] = events[0]
            first["first"] = tuple(events)
            blocks[0] = first
            packet["event_blocks"] = tuple(blocks)
            with self.subTest(mode=mode), patch.object(
                v15, "_arm_stream", return_value=packet,
            ), self.assertRaisesRegex(RuntimeError, "anchor event"):
                v17._corresponding_anchor_read_report(core, pairs)

    def test_event_slot_map_supports_noncontiguous_and_rejects_invalid_indices(self) -> None:
        events = (
            SimpleNamespace(evidence_refs=("pair:twin:anchor:0",), allocation_index=7),
            SimpleNamespace(evidence_refs=("pair:twin:anchor:1",), allocation_index=19),
        )
        self.assertEqual(
            v17._validated_event_slots(
                events,
                ("pair:twin:anchor:0", "pair:twin:anchor:1"),
                512,
            ),
            {"pair:twin:anchor:0": 7, "pair:twin:anchor:1": 19},
        )
        for invalid in (-1, 512):
            changed = (
                SimpleNamespace(
                    evidence_refs=("pair:twin:anchor:0",), allocation_index=invalid,
                ),
                events[1],
            )
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                RuntimeError, "outside memory",
            ):
                v17._validated_event_slots(
                    changed,
                    ("pair:twin:anchor:0", "pair:twin:anchor:1"),
                    512,
                )

    def test_query_swap_and_full_evaluate_reuse_v16_controls(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, 1000 + index)
            for index, pair in enumerate(self.corpus.development)
        )
        core = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        lesion = v16._relation_query_swap_control(core, pairs)
        self.assertTrue(lesion["only_read_query_swapped"], lesion)
        metrics = v17._evaluate(core, pairs, (1.0, 0.0))
        self.assertEqual(metrics["pair_count"], 48)
        self.assertEqual(metrics["corresponding_anchor_read_mass"]["count"], 384)
        self.assertTrue(all(metrics["outcome_blind_integrity"].values()))

    def test_real_fit_completes_one_optimizer_update(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, index) for index, pair in enumerate(self.corpus.train[:8])
        )
        core = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        before = {name: value.detach().clone() for name, value in core.named_parameters()}
        with patch.object(v15, "TRAIN_PAIRS", 8), patch.object(v15, "TRAIN_UPDATES", 1):
            report, _ = v15._fit(core, pairs, (tuple(range(8)),))
        self.assertEqual(report["completed_updates"], 1)
        self.assertTrue(report["updates"][0]["true_deranged_paths_differentiable"])
        self.assertTrue(any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in core.named_parameters()
        ))


class ComponentGateAndSealTests(unittest.TestCase):
    def test_component_and_full_gates_are_noninterchangeable(self) -> None:
        metrics = _passing_metrics()
        architecture = {"exact": True}
        self.assertTrue(v17._component_gate(
            metrics, architecture, identifiability_exact=True, identity_exact=True,
        ))
        self.assertTrue(v17._full_gate(
            metrics, architecture, identifiability_exact=True, identity_exact=True,
        ))
        retention = copy.deepcopy(metrics)
        retention["retention"]["maximum_drop"] = 0.0501
        self.assertTrue(v17._component_gate(
            retention, architecture, identifiability_exact=True, identity_exact=True,
        ))
        self.assertFalse(v17._full_gate(
            retention, architecture, identifiability_exact=True, identity_exact=True,
        ))

    def test_component_gate_rejects_correspondence_and_removal_failures(self) -> None:
        passing = _passing_metrics()
        architecture = {"exact": True}
        mutations = (
            ("top1_accuracy", 0.749),
            ("mean_matching_mass", 0.299),
            ("mean_matching_minus_competitor_margin", 0.099),
            ("metadata_evaluator_only", False),
            ("learner_state_unchanged", False),
        )
        for key, value in mutations:
            packet = copy.deepcopy(passing)
            packet["corresponding_anchor_read_mass"][key] = value
            with self.subTest(key=key):
                self.assertFalse(v17._component_gate(
                    packet, architecture, identifiability_exact=True, identity_exact=True,
                ))
        bad_count = copy.deepcopy(passing)
        bad_count["pair_count"] = 47
        self.assertFalse(v17._component_gate(
            bad_count, architecture, identifiability_exact=True, identity_exact=True,
        ))
        bad_probe_count = copy.deepcopy(passing)
        bad_probe_count["probe_relation_query_swap"]["probe_count"] = 383
        self.assertFalse(v17._component_gate(
            bad_probe_count, architecture,
            identifiability_exact=True, identity_exact=True,
        ))
        bad_lineage = copy.deepcopy(passing)
        bad_lineage["corresponding_anchor_read_mass"]["anchor_slot_lineage_exact"] = False
        self.assertFalse(v17._component_gate(
            bad_lineage, architecture,
            identifiability_exact=True, identity_exact=True,
        ))
        stratum = copy.deepcopy(passing)
        stratum["corresponding_anchor_read_mass"]["by_transition"]["reorder"]["top1_accuracy"] = 0.649
        self.assertFalse(v17._component_gate(
            stratum, architecture, identifiability_exact=True, identity_exact=True,
        ))
        reset = copy.deepcopy(passing)
        reset["arms"]["post_acquisition_reset"]["balanced_accuracy"] = 0.6101
        self.assertFalse(v17._component_gate(
            reset, architecture, identifiability_exact=True, identity_exact=True,
        ))
        both_query_operands = copy.deepcopy(passing)
        both_query_operands["probe_relation_query_swap"]["balanced_accuracy"] = 0.6101
        both_query_operands["probe_relation_query_swap"]["mean_probe_nll"] = 0.219
        self.assertFalse(v17._component_gate(
            both_query_operands, architecture,
            identifiability_exact=True, identity_exact=True,
        ))

    def test_final_refuses_component_only_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save({}, checkpoint)
            with patch.object(v17, "_sha256", return_value="A" * 64), patch.object(
                v17, "_source_hashes", return_value={"runner": "B" * 64}
            ):
                packet = {
                    "identity": v17.IDENTITY, "phase": "train-development",
                    "classification": "SHARED_SEMANTIC_CORRESPONDENCE_SUPPORTED",
                    "development_authorized": False,
                    "source_hashes": {"runner": "B" * 64},
                    "checkpoint_sha256": "A" * 64,
                    "identifiability_preflight": {"exact": True},
                    "identity_checks": {"exact": True},
                    "architecture_evidence": {"exact": True},
                    "development_metrics": _passing_metrics(),
                }
                with self.assertRaises((RuntimeError, ValueError)):
                    v17._validate_final_authorization(packet, checkpoint)

    def test_final_binds_development_packet_to_checkpoint_copies(self) -> None:
        source_map = {"runner": "B" * 64}
        metrics = _passing_metrics()
        architecture = {
            "exact": True,
            "semantic_metric_inputs": ("detached_relation",),
            "semantic_metric_shared_for": ("read_query", "write_key"),
            "parameter_groups": {"semantic_metric": ("semantic_metric_network.1.weight",)},
        }
        preflight = {"exact": True, "schedule_shape": (96, 8)}
        identity_checks = {
            "exact": True, "qwen_before": "Q", "qwen_after": "Q",
            "source_pair": ("BEFORE", "AFTER"),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            checkpoint = {
                "identity": v17.IDENTITY, "source_hashes": source_map,
                "core_digest": "CORE-DIGEST",
                "identifiability_preflight": preflight,
                "architecture_evidence": architecture,
                "development_metrics": metrics,
                "identity_checks": identity_checks,
            }
            torch.save(checkpoint, checkpoint_path)
            packet = json.loads(json.dumps({
                "identity": v17.IDENTITY, "phase": "train-development",
                "classification": "FULL_DURABLE_SEMANTIC_CREDIT_SUPPORTED",
                "development_authorized": True, "source_hashes": source_map,
                "checkpoint_sha256": "A" * 64, "core_digest": "CORE-DIGEST",
                "identifiability_preflight": preflight,
                "identity_checks": identity_checks,
                "architecture_evidence": architecture,
                "development_metrics": metrics,
            }))
            with patch.object(v17, "_sha256", return_value="A" * 64), patch.object(
                v17, "_source_hashes", return_value=source_map,
            ):
                v17._validate_final_authorization(packet, checkpoint_path)
                changed = copy.deepcopy(packet)
                changed["development_metrics"]["pair_count"] = 47
                with self.assertRaises((RuntimeError, ValueError)):
                    v17._validate_final_authorization(changed, checkpoint_path)
                changed = copy.deepcopy(packet)
                changed["core_digest"] = "OTHER"
                with self.assertRaises((RuntimeError, ValueError)):
                    v17._validate_final_authorization(changed, checkpoint_path)
                changed = copy.deepcopy(packet)
                changed["identity_checks"]["qwen_after"] = "CHANGED"
                with self.assertRaises((RuntimeError, ValueError)):
                    v17._validate_final_authorization(changed, checkpoint_path)

    def test_final_recomputes_state_horizon(self) -> None:
        source = inspect.getsource(v17._run_final)
        self.assertIn('metrics["maximum_state_step"]', source)
        self.assertIn('metrics["protocol_invariants"]["eight_pair_state_horizon"]', source)

    def test_source_map_and_consumed_v16_chain_fail_closed(self) -> None:
        hashes = v17._source_hashes()
        self.assertIn("runner", hashes)
        self.assertIn("shared_metric_core", hashes)
        self.assertIn("v16_runner", hashes)
        self.assertIn("v16_strict_core", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        source_map = {"runner": "C" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, checkpoint = root / "v16.json", root / "v16.pt"
            result.write_text(json.dumps({
                "identity": v16.IDENTITY, "phase": "train-development",
                "classification": "DEVELOPMENT_NOT_SUPPORTED",
                "development_authorized": False,
                "checkpoint_sha256": v17.V16_CHECKPOINT_SHA256,
                "source_hashes": source_map,
            }), encoding="utf-8")
            torch.save({"identity": v16.IDENTITY, "source_hashes": source_map}, checkpoint)
            with (
                patch.object(v17, "_sha256", side_effect=lambda path: (
                    v17.V16_RESULT_SHA256 if Path(path) == result
                    else v17.V16_CHECKPOINT_SHA256
                )),
                patch.object(v16, "_source_hashes", return_value=source_map),
            ):
                self.assertEqual(v17._v16_evidence(result, checkpoint)["source_hashes"], source_map)
            with (
                patch.object(v17, "_sha256", side_effect=lambda path: (
                    v17.V16_RESULT_SHA256 if Path(path) == result
                    else v17.V16_CHECKPOINT_SHA256
                )),
                patch.object(v16, "_source_hashes", return_value={"runner": "D" * 64}),
            ):
                with self.assertRaisesRegex(RuntimeError, "source chain"):
                    v17._v16_evidence(result, checkpoint)


if __name__ == "__main__":
    unittest.main()
