from __future__ import annotations

import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from angler.reasoning.strict_key_value_credit_memory import StrictKeyValueCreditMemoryCore
from experiments.corpora.paired_latent_contingency_credit_v15 import (
    ACQUISITION_INDICES,
    PROBE_INDICES,
    build_paired_latent_contingency_credit_v15,
)
from experiments.runners import paired_latent_contingency_credit_v15 as v15
from experiments.runners import strict_key_value_credit_v16 as v16


def _fake_encoded(pair: object, seed: int) -> v15.EncodedTwinPair:
    generator = torch.Generator().manual_seed(20_000 + seed)
    relation = torch.randn(6, 64, generator=generator)
    # Ensure every alpha/beta query swap is mechanically non-identical.
    relation += torch.arange(6, dtype=torch.float32).unsqueeze(1) * 0.05
    base = torch.randn(6, generator=generator) * 0.1
    temporal = torch.tensor(
        [v15._temporal(event, index) for index, event in enumerate(pair.first.public.events)],
        dtype=torch.float32,
    )

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
        "balanced_accuracy": ba,
        "mean_probe_nll": nll,
        "finite": True,
        "read_weight_min": 0.0,
        "read_weight_max": 1.0,
        "write_strength_min": 0.0,
        "write_strength_max": 1.0,
    }


def _passing_metrics() -> dict[str, object]:
    return {
        "pair_count": 48,
        "arms": {
            "true": _arm(0.71, 0.20),
            "deranged": _arm(0.60, 0.23),
            "zero_no_read": _arm(0.60, 0.23),
            "post_acquisition_reset": _arm(0.60, 0.30),
            "outcome_blind_writer": _arm(0.60, 0.30),
        },
        "affine_calibrator": {"balanced_accuracy": 0.60},
        "outcome_blind_integrity": {
            "write_count_exact": True,
            "allocation_exact": True,
            "write_gate_exact": True,
            "erase_gate_exact": True,
            "write_keys_exact": True,
            "usage_exact": True,
            "acquisition_exact": True,
            "stored_values_differ": True,
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
                "matched": 0.20,
                "zero": 0.23,
                "unrelated": 0.23,
                "key_value_mismatch": 0.23,
            },
        },
        "improved_family_count": 9,
        "transition_improved_family_counts": {
            "insertion": 2,
            "removal": 2,
            "reorder": 2,
            "replacement": 2,
        },
        "ordering": {"changed_fraction": 0.25},
        "replay": {"exact": True, "maximum_logit_error": 0.0},
        "retention": {"maximum_drop": 0.05},
        "state_bounds": {
            "finite": True,
            "keys_max_abs": 1.0,
            "values_max_abs": 1.0,
            "usage_min": 0.0,
            "usage_max": 1.0,
            "acquisition_min": 0.0,
            "acquisition_max": 1.0,
        },
        "protocol_invariants": {
            "exact_training_schedule": True,
            "all_training_gradients_finite": True,
            "true_deranged_paths_differentiable": True,
            "zero_baseline_detached": True,
            "eight_pair_state_horizon": True,
        },
        "probe_relation_query_swap": {
            "balanced_accuracy": 0.60,
            "mean_probe_nll": 0.23,
            "probe_count": 384,
            "relation_queries_changed": True,
            "relation_inputs_changed": True,
            "read_strength_exact": True,
            "relation_source_exact": True,
            "opposite_relation_class_exact": True,
            "temporal_base_labels_exact": True,
            "state_and_stored_memory_exact": True,
            "only_read_query_swapped": True,
            "permutation": [3, 2, 5, 4],
        },
    }


def _architecture() -> dict[str, object]:
    return {"exact": True}


class StrictRunnerMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_paired_latent_contingency_credit_v15()

    def test_v15_schedule_and_loss_identity_are_reused(self) -> None:
        schedule = v15._training_schedule(self.corpus)
        self.assertEqual(len(schedule), 96)
        self.assertTrue(all(len(row) == 8 for row in schedule))
        self.assertEqual(v16.TRAIN_UPDATES, v15.TRAIN_UPDATES)
        self.assertEqual(v16.PAIRS_PER_UPDATE, v15.PAIRS_PER_UPDATE)
        self.assertEqual(v15.LOSS_WEIGHTS, (1.0, 1.0, 1.0))

    def test_architecture_evidence_has_strict_invariance_and_gradients(self) -> None:
        core = StrictKeyValueCreditMemoryCore(temporal_width=4)
        pair = _fake_encoded(self.corpus.train[0], 0)
        report = v16._architecture_evidence(core, pair)
        self.assertTrue(report["exact"], report)
        self.assertTrue(all(report["outcome_address_invariance"].values()))
        self.assertTrue(all(report["required_gradient_prefixes_reached"].values()))
        self.assertTrue(report["erase_gradient_inactive_by_capacity"])
        architecture = report["architecture_report"]
        self.assertEqual(architecture["residual_decoder_inputs"], ("retrieved_value",))
        self.assertFalse(architecture["outcome_affects_address_or_gates"])
        self.assertFalse(architecture["query_affects_value_content_or_decoder"])

    def test_outcome_blind_value_preserves_all_address_fields(self) -> None:
        core = StrictKeyValueCreditMemoryCore(temporal_width=4)
        pair = _fake_encoded(self.corpus.train[0], 1)
        episode = pair.first
        initial = core.initial_state()
        output = core.predict(
            episode.relation_features[0:1], episode.temporal_features[0:1],
            episode.base_logits[0:1], state=initial,
        )
        normal_state, normal = core.apply_feedback(
            initial, output, episode.outcomes[0:1], evidence_refs="normal",
        )
        blind_state, blind = v16._apply_outcome_blind_feedback(
            core, initial, output, episode.outcomes[0:1], evidence_ref="blind",
        )
        self.assertEqual(normal.allocation_index, blind.allocation_index)
        self.assertEqual(normal.read_strength, blind.read_strength)
        self.assertEqual(normal.write_strength, blind.write_strength)
        self.assertEqual(normal.erase_gate, blind.erase_gate)
        self.assertTrue(torch.equal(normal.write_key, blind.write_key))
        self.assertTrue(torch.equal(normal_state.keys, blind_state.keys))
        self.assertTrue(torch.equal(normal_state.usage, blind_state.usage))
        self.assertTrue(torch.equal(normal_state.acquisition, blind_state.acquisition))
        self.assertFalse(torch.equal(normal_state.values, blind_state.values))

    def test_v15_scope_restores_bindings_on_success_and_error(self) -> None:
        names = (
            "StructureKeyedCreditState",
            "StructureKeyedCreditEvent",
            "_apply_outcome_blind_feedback",
            "_write_blind_anchors",
        )
        original = {name: getattr(v15, name) for name in names}
        with v16._v15_control_scope():
            self.assertIs(v15.StructureKeyedCreditState, v16.StrictKeyValueCreditState)
        self.assertTrue(all(getattr(v15, name) is value for name, value in original.items()))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with v16._v15_control_scope():
                raise RuntimeError("injected")
        self.assertTrue(all(getattr(v15, name) is value for name, value in original.items()))

    def test_probe_relation_swap_preserves_state_and_nonrelation_inputs(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, index)
            for index, pair in enumerate(self.corpus.development)
        )
        core = StrictKeyValueCreditMemoryCore(temporal_width=4)
        report = v16._relation_query_swap_control(core, pairs)
        self.assertEqual(report["probe_count"], 384)
        self.assertTrue(report["relation_queries_changed"])
        self.assertTrue(report["relation_inputs_changed"])
        self.assertTrue(report["read_strength_exact"])
        self.assertTrue(report["relation_source_exact"])
        self.assertTrue(report["opposite_relation_class_exact"])
        self.assertTrue(report["temporal_base_labels_exact"])
        self.assertTrue(report["state_and_stored_memory_exact"])
        self.assertTrue(report["only_read_query_swapped"])
        self.assertEqual(report["permutation"], [3, 2, 5, 4])

    def test_full_frozen_v15_control_adapter_executes_and_restores(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, 5_000 + index)
            for index, pair in enumerate(self.corpus.development)
        )
        core = StrictKeyValueCreditMemoryCore(temporal_width=4)
        original_state_type = v15.StructureKeyedCreditState
        metrics = v16._evaluate(core, pairs, (1.0, 0.0))
        self.assertEqual(metrics["pair_count"], 48)
        self.assertEqual(metrics["probe_relation_query_swap"]["probe_count"], 384)
        self.assertTrue(all(metrics["outcome_blind_integrity"].values()))
        self.assertIs(v15.StructureKeyedCreditState, original_state_type)

    def test_real_fit_completes_one_optimizer_update(self) -> None:
        pairs = tuple(
            _fake_encoded(pair, index) for index, pair in enumerate(self.corpus.train[:8])
        )
        core = StrictKeyValueCreditMemoryCore(temporal_width=4)
        before = {name: value.detach().clone() for name, value in core.named_parameters()}
        with (
            patch.object(v15, "TRAIN_PAIRS", 8),
            patch.object(v15, "TRAIN_UPDATES", 1),
        ):
            report, _ = v15._fit(core, pairs, (tuple(range(8)),))
        self.assertEqual(report["completed_updates"], 1)
        self.assertTrue(report["updates"][0]["predict_before_feedback"])
        self.assertTrue(report["updates"][0]["true_deranged_paths_differentiable"])
        self.assertTrue(report["updates"][0]["zero_baseline_detached"])
        self.assertTrue(any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in core.named_parameters()
        ))


class GateAndSealTests(unittest.TestCase):
    def test_gate_accepts_exact_boundary_and_rejects_query_swap_operands(self) -> None:
        passing = _passing_metrics()
        self.assertTrue(v16._gate(
            passing, _architecture(), identifiability_exact=True, identity_exact=True,
        ))
        # Either registered removal operand suffices; both must fail together
        # before the causal relation-query lesion is rejected.
        accuracy_route = copy.deepcopy(passing)
        accuracy_route["probe_relation_query_swap"]["mean_probe_nll"] = 0.219
        self.assertTrue(v16._gate(
            accuracy_route, _architecture(), identifiability_exact=True, identity_exact=True,
        ))
        nll_route = copy.deepcopy(passing)
        nll_route["probe_relation_query_swap"]["balanced_accuracy"] = 0.6101
        self.assertTrue(v16._gate(
            nll_route, _architecture(), identifiability_exact=True, identity_exact=True,
        ))
        failed_removal = copy.deepcopy(passing)
        failed_removal["probe_relation_query_swap"]["balanced_accuracy"] = 0.6101
        failed_removal["probe_relation_query_swap"]["mean_probe_nll"] = 0.219
        self.assertFalse(v16._gate(
            failed_removal, _architecture(), identifiability_exact=True, identity_exact=True,
        ))
        for mutation in (
            ("relation_queries_changed", False),
            ("relation_inputs_changed", False),
            ("read_strength_exact", False),
            ("relation_source_exact", False),
            ("opposite_relation_class_exact", False),
            ("temporal_base_labels_exact", False),
            ("state_and_stored_memory_exact", False),
            ("only_read_query_swapped", False),
        ):
            packet = copy.deepcopy(passing)
            packet["probe_relation_query_swap"][mutation[0]] = mutation[1]
            with self.subTest(mutation=mutation):
                self.assertFalse(v16._gate(
                    packet, _architecture(), identifiability_exact=True, identity_exact=True,
                ))
        self.assertFalse(v16._gate(
            passing, {"exact": False}, identifiability_exact=True, identity_exact=True,
        ))

    def test_source_map_binds_strict_core_and_v15_chain(self) -> None:
        hashes = v16._source_hashes()
        self.assertIn("runner", hashes)
        self.assertIn("strict_core", hashes)
        self.assertIn("v15_runner", hashes)
        self.assertIn("v15_corpus", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_consumed_v15_evidence_rejects_imported_source_drift(self) -> None:
        source_map = {"runner": "A" * 64, "corpus": "B" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "v15.json"
            checkpoint = root / "v15.pt"
            result.write_text(
                __import__("json").dumps({
                    "identity": v15.IDENTITY,
                    "phase": "train-development",
                    "classification": "DEVELOPMENT_NOT_SUPPORTED",
                    "development_authorized": False,
                    "checkpoint_sha256": v16.V15_CHECKPOINT_SHA256,
                    "source_hashes": source_map,
                }),
                encoding="utf-8",
            )
            torch.save({"identity": v15.IDENTITY, "source_hashes": source_map}, checkpoint)
            with (
                patch.object(
                    v16, "_sha256",
                    side_effect=lambda path: (
                        v16.V15_RESULT_SHA256
                        if Path(path) == result else v16.V15_CHECKPOINT_SHA256
                    ),
                ),
                patch.object(v15, "_source_hashes", return_value=source_map),
            ):
                self.assertEqual(v16._v15_evidence(result, checkpoint)["source_hashes"], source_map)
            with (
                patch.object(
                    v16, "_sha256",
                    side_effect=lambda path: (
                        v16.V15_RESULT_SHA256
                        if Path(path) == result else v16.V15_CHECKPOINT_SHA256
                    ),
                ),
                patch.object(v15, "_source_hashes", return_value={"runner": "C" * 64}),
            ):
                with self.assertRaisesRegex(RuntimeError, "source chain"):
                    v16._v15_evidence(result, checkpoint)

    def test_final_authorization_refuses_incomplete_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"not-authorized")
            with self.assertRaises((RuntimeError, ValueError)):
                v16._validate_final_authorization({}, checkpoint)

    def test_final_phase_requires_cuda1_and_enforces_resource_ceiling(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cuda:1"):
            v16._prepare_angler_device("cpu")
        with (
            patch.object(v16.torch.cuda, "set_device") as set_device,
            patch.object(v16.torch.cuda, "reset_peak_memory_stats") as reset_peak,
            patch.object(v16.time, "perf_counter", return_value=123.0),
        ):
            device, started = v16._prepare_angler_device("cuda:1")
        self.assertEqual(device, torch.device("cuda:1"))
        self.assertEqual(started, 123.0)
        set_device.assert_called_once_with(torch.device("cuda:1"))
        reset_peak.assert_called_once_with(torch.device("cuda:1"))
        final_source = inspect.getsource(v16._run_final)
        self.assertIn("_prepare_angler_device(args.device)", final_source)
        self.assertGreaterEqual(final_source.count("_enforce_resources(started, device)"), 3)


if __name__ == "__main__":
    unittest.main()
