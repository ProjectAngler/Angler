from __future__ import annotations

import copy
import inspect
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from angler.reasoning.paired_public_relation_credit_memory import (
    PairedPublicRelationCreditMemoryCore,
)
from experiments.corpora.paired_latent_contingency_credit_v15 import (
    build_paired_latent_contingency_credit_v15,
)
from experiments.runners import paired_latent_contingency_credit_v15 as v15
from experiments.runners import paired_public_relation_credit_v19 as v19
from experiments.runners import shared_semantic_metric_credit_v17 as v17


class _UnsupportedCheckpointGlobal:
    pass


def _fake_encoded(pair: object, seed: int) -> v15.EncodedTwinPair:
    generator = torch.Generator().manual_seed(190_000 + seed)
    relation = torch.randn(6, 64, generator=generator)
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
            public_payload_sha256="PUBLIC",
            input_tensor_sha256="INPUT",
        )

    return v15.EncodedTwinPair(pair.pair_ref, episode(pair.first), episode(pair.second))


def _optimizer(core: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        core.parameters(), lr=v15.LEARNING_RATE, betas=v15.ADAM_BETAS,
        eps=v15.ADAM_EPSILON, weight_decay=v15.WEIGHT_DECAY,
    )


def _read(top1: float, mass: float, margin: float) -> dict[str, float]:
    return {
        "top1_accuracy": top1,
        "mean_matching_mass": mass,
        "mean_matching_minus_competitor_margin": margin,
    }


def _supported_metrics(*, full: bool) -> dict[str, object]:
    read = _read(0.80, 0.40, 0.20) if full else _read(0.65, 0.25, 0.05)
    result: dict[str, object] = {
        "corresponding_anchor_read_mass": read,
        "arms": {"true": {"balanced_accuracy": 0.75, "mean_probe_nll": 0.30}},
    }
    if full:
        result["residual_zero_corresponding_anchor_read_mass"] = _read(0.65, 0.25, 0.05)
        result["paired_public_relation_controls"] = {
            "matched": {"balanced_accuracy": 0.80, "mean_probe_nll": 0.30},
            "residual_zero": {"balanced_accuracy": 0.68, "mean_probe_nll": 0.34},
            "sidecar_deranged": {"balanced_accuracy": 0.68, "mean_probe_nll": 0.34},
            "pair_query_swap": {"balanced_accuracy": 0.68, "mean_probe_nll": 0.34},
            "branch_state_immutable": {
                "matched": True, "residual_zero": True,
                "sidecar_deranged": True, "pair_query_swap": True,
            },
            "source_state_exact": True,
            "sidecar_rotate_by_one_exact": True,
            "sidecar_derangement_zero_fixed_points": True,
            "query_swap_zero_fixed_points": True,
            "query_rows_all_changed": True,
            "query_multiset_exact": True,
            "pair_query_swap_non_target_outputs_exact": {
                "read_queries": True, "read_strengths": True,
                "semantic_content_logits": True, "observed_temporal": True,
                "base_logits": True,
            },
            "pair_query_swap_state_exact": True,
            "pair_query_swap_raw_source_exact": True,
            "pair_query_swap_target_labels_exact": True,
            "family_exposure_exact": True,
            "transition_exposure_exact": True,
            "probe_position_exposure_exact": True,
            "matched_v17_operand_exact": True,
            "no_feedback_write_or_optimizer_step": True,
            "nonzero_pair_residual_fraction": 0.50,
            "changed_top_read_fraction": 0.10,
        }
    return result


class RunnerMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_paired_latent_contingency_credit_v15()
        cls.encoded = tuple(
            _fake_encoded(pair, index) for index, pair in enumerate(cls.corpus.train[:8])
        )

    def test_pair_mode_view_is_explicit_and_zero_init_exact(self) -> None:
        torch.manual_seed(19)
        core = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        episode = self.encoded[0].first
        args = (
            episode.relation_features[0:1], episode.temporal_features[0:1],
            episode.base_logits[0:1],
        )
        full = v19._PairModeView(core, True).predict(*args, state=core.initial_state())
        unary = v19._PairModeView(core, False).predict(*args, state=core.initial_state())
        self.assertTrue(full.pair_residual_enabled)
        self.assertFalse(unary.pair_residual_enabled)
        self.assertTrue(torch.equal(full.logits, unary.logits))
        with self.assertRaises(TypeError):
            v19._PairModeView(core, True).predict(
                *args, state=core.initial_state(), pair_residual_enabled=False,
            )

    def test_unary_default_mode_replays_unary_events_exactly(self) -> None:
        core = PairedPublicRelationCreditMemoryCore(
            temporal_width=4, pair_residual_enabled_by_default=False,
        )
        episode, twin = self.encoded[0].first, self.encoded[0].second
        state, events, _, _ = v15._write_anchors(
            v19._PairModeView(core, False), episode, twin, core.initial_state(),
            arm="true", detach_state=True,
        )
        replayed, outputs = core.replay(events)
        self.assertEqual(core.state_digest(replayed), core.state_digest(state))
        self.assertTrue(all(output.pair_residual_enabled is False for output in outputs))
        self.assertTrue(all(event.pair_residual_enabled is False for event in events))

    def test_false_view_replays_on_full_default_core_without_mutation(self) -> None:
        core = PairedPublicRelationCreditMemoryCore(
            temporal_width=4, pair_residual_enabled_by_default=True,
        )
        view = v19._PairModeView(core, False)
        episode, twin = self.encoded[0].first, self.encoded[0].second
        state, events, _, _ = v15._write_anchors(
            view, episode, twin, core.initial_state(), arm="true", detach_state=True,
        )
        model_before = v19._model_digest(core)
        replayed, outputs = view.replay(events)
        self.assertEqual(core.state_digest(replayed), core.state_digest(state))
        self.assertTrue(all(output.pair_residual_enabled is False for output in outputs))
        self.assertTrue(all(event.pair_residual_enabled is False for event in events))
        self.assertTrue(core.pair_residual_enabled_by_default)
        self.assertEqual(v19._model_digest(core), model_before)

    def test_occupied_architecture_preflight_proves_v17_zero_path_and_sidecar_replay(self) -> None:
        core = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        report = v19._architecture_evidence(core, self.encoded[0])
        paired = report["pair_public_relation_invariance"]
        self.assertTrue(report["exact"], report)
        self.assertTrue(paired["occupied_state_nonempty"])
        self.assertTrue(paired["occupied_residual_zero_direct_v17_exact"])
        self.assertTrue(paired["sidecar_capture_restore_exact"])
        self.assertTrue(paired["sidecar_replay_exact"])

    def test_sidecar_rotate_is_fixed_point_free_and_non_target_exact(self) -> None:
        core = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        episode = self.encoded[0].first
        state = core.initial_state()
        for index in (0, 1):
            output = core.predict(
                episode.relation_features[index:index + 1],
                episode.temporal_features[index:index + 1],
                episode.base_logits[index:index + 1], state=state,
            )
            state, _ = core.apply_feedback(
                state, output, episode.outcomes[index:index + 1],
                evidence_refs=f"rotate:{index}", detach_state=True,
            )
        changed = v19._rotate_public_sidecars(state)
        occupied = torch.nonzero(state.usage > 0, as_tuple=False).flatten()
        self.assertNotEqual(core.state_digest(state), core.state_digest(changed))
        self.assertEqual(
            v19._non_sidecar_state_digest(state),
            v19._non_sidecar_state_digest(changed),
        )
        self.assertTrue(torch.equal(
            changed.public_relations[occupied],
            state.public_relations[occupied.roll(-1)],
        ))
        self.assertTrue(torch.any(
            changed.public_relations[occupied] != state.public_relations[occupied],
            dim=-1,
        ).all())

    def test_blind_and_mismatch_adapters_preserve_sidecar_state(self) -> None:
        core = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        episode = self.encoded[0].first
        output = core.predict(
            episode.relation_features[0:1], episode.temporal_features[0:1],
            episode.base_logits[0:1], state=core.initial_state(),
        )
        positive, event_a = v19._apply_outcome_blind_feedback(
            core, core.initial_state(), output, torch.ones(1), evidence_ref="blind:+",
        )
        negative, event_b = v19._apply_outcome_blind_feedback(
            core, core.initial_state(), output, -torch.ones(1), evidence_ref="blind:-",
        )
        self.assertTrue(torch.equal(positive.values, negative.values))
        self.assertTrue(torch.equal(positive.public_relations, negative.public_relations))
        self.assertTrue(torch.equal(event_a.write_value, event_b.write_value))
        second = core.predict(
            episode.relation_features[1:2], episode.temporal_features[1:2],
            episode.base_logits[1:2], state=positive,
        )
        two, _ = core.apply_feedback(
            positive, second, episode.outcomes[1:2], evidence_refs="blind:2", detach_state=True,
        )
        mismatch = v19._mismatched_state(two)
        self.assertTrue(torch.equal(two.public_relations, mismatch.public_relations))
        self.assertFalse(torch.equal(two.values, mismatch.values))

    def test_pair_query_control_mechanically_preserves_every_non_target_operand(self) -> None:
        core = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        with patch.object(v19, "DEVELOPMENT_PAIRS", len(self.encoded)):
            report = v19._paired_controls(
                core, self.encoded,
                {"matched_balanced_accuracy": 0.0, "mean_nll": {"matched": 0.0}},
            )
        self.assertTrue(all(report["pair_query_swap_non_target_outputs_exact"].values()))
        self.assertTrue(report["pair_query_swap_state_exact"])
        self.assertTrue(report["pair_query_swap_raw_source_exact"])
        self.assertTrue(report["pair_query_swap_target_labels_exact"])
        self.assertTrue(report["family_exposure_exact"])
        self.assertTrue(report["transition_exposure_exact"])
        self.assertTrue(report["probe_position_exposure_exact"])
        self.assertTrue(report["no_feedback_write_or_optimizer_step"])

    def test_initial_task_parity_excludes_only_pair_scorer_gradient(self) -> None:
        torch.manual_seed(29)
        full = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        unary = copy.deepcopy(full)
        optimizers = {"full": _optimizer(full), "unary": _optimizer(unary)}
        report = v19._task_parity_preflight(
            full, unary, self.encoded, tuple(range(8)), optimizers,
        )
        self.assertTrue(report["exact"], report)
        self.assertTrue(report["full_zero_head_first_gradient_nonzero"])
        self.assertTrue(report["unary_scorer_gradients_absent"])
        self.assertTrue(report["non_scorer_task_gradients_exact"])
        self.assertTrue(report["optimizer_configuration_exact"])
        self.assertTrue(report["optimizer_parameter_order_exact"])

        optimizers["unary"].param_groups[0]["lr"] *= 2.0
        changed = v19._task_parity_preflight(
            full, unary, self.encoded, tuple(range(8)), optimizers,
        )
        self.assertFalse(changed["optimizer_configuration_exact"])
        self.assertFalse(changed["exact"])

    def test_disposable_smoke_leaves_launch_models_optimizers_rng_and_state_exact(self) -> None:
        torch.manual_seed(39)
        full = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        unary = copy.deepcopy(full)
        optimizers = {"full": _optimizer(full), "unary": _optimizer(unary)}
        before = (v19._model_digest(full), v19._model_digest(unary))
        report = v19._mechanics_smoke(
            full, unary, optimizers, self.encoded,
            (tuple(range(8)), tuple(range(7, -1, -1))),
        )
        self.assertTrue(report["exact"], report)
        self.assertEqual(before, (v19._model_digest(full), v19._model_digest(unary)))
        self.assertTrue(report["first_update_head_nonzero"])
        self.assertTrue(report["second_update_all_scorer_nonzero"])

    def test_one_paired_task_update_trains_full_but_not_unary_scorer(self) -> None:
        torch.manual_seed(49)
        full = PairedPublicRelationCreditMemoryCore(temporal_width=4)
        unary = copy.deepcopy(full)
        optimizers = {"full": _optimizer(full), "unary": _optimizer(unary)}
        unary_scorer = {
            name: value.detach().clone()
            for name, value in unary.named_parameters()
            if name.startswith(v19.PAIR_PREFIX)
        }
        with patch.object(v19, "TRAIN_PAIRS", 8), patch.object(v19, "TRAIN_UPDATES", 1):
            report, _, _ = v19._paired_fit(
                full, unary, self.encoded, (tuple(range(8)),), optimizers,
            )
        self.assertEqual(report["completed_updates"], 1)
        self.assertTrue(report["updates"][0]["chronology_exact"])
        self.assertTrue(any(
            not torch.equal(value, dict(full.named_parameters())[name])
            for name, value in unary_scorer.items()
        ))
        self.assertTrue(all(
            torch.equal(value, dict(unary.named_parameters())[name])
            for name, value in unary_scorer.items()
        ))


class GateAndProtocolTests(unittest.TestCase):
    def test_component_gate_accepts_exact_frozen_boundary(self) -> None:
        full, unary = _supported_metrics(full=True), _supported_metrics(full=False)
        with patch.object(v17, "_component_gate", return_value=True), patch.object(
            v17, "_full_gate", return_value=True,
        ):
            self.assertTrue(v19._component_gate(
                full, unary, {"exact": True}, {"exact": True}, identity_exact=True,
            ))
            self.assertTrue(v19._full_gate(
                full, unary, {"exact": True}, {"exact": True}, identity_exact=True,
            ))

    def test_component_gate_rejects_each_causal_or_activity_operand(self) -> None:
        full, unary = _supported_metrics(full=True), _supported_metrics(full=False)
        controls = full["paired_public_relation_controls"]
        mutations = (
            ("sidecar_rotate_by_one_exact", False),
            ("query_rows_all_changed", False),
            ("pair_query_swap_state_exact", False),
            ("pair_query_swap_raw_source_exact", False),
            ("pair_query_swap_target_labels_exact", False),
            ("family_exposure_exact", False),
            ("transition_exposure_exact", False),
            ("probe_position_exposure_exact", False),
            ("matched_v17_operand_exact", False),
            ("nonzero_pair_residual_fraction", 0.4999),
            ("changed_top_read_fraction", 0.0999),
        )
        with patch.object(v17, "_component_gate", return_value=True):
            for key, value in mutations:
                changed = copy.deepcopy(full)
                changed["paired_public_relation_controls"][key] = value
                with self.subTest(key=key):
                    self.assertFalse(v19._component_gate(
                        changed, unary, {"exact": True}, {"exact": True}, identity_exact=True,
                    ))
            for key in controls["pair_query_swap_non_target_outputs_exact"]:
                changed = copy.deepcopy(full)
                changed["paired_public_relation_controls"]["pair_query_swap_non_target_outputs_exact"][key] = False
                with self.subTest(pair_query_non_target=key):
                    self.assertFalse(v19._component_gate(
                        changed, unary, {"exact": True}, {"exact": True}, identity_exact=True,
                    ))
            for name in ("residual_zero", "sidecar_deranged", "pair_query_swap"):
                changed = copy.deepcopy(full)
                changed["paired_public_relation_controls"][name] = copy.deepcopy(controls["matched"])
                with self.subTest(lesion=name):
                    self.assertFalse(v19._component_gate(
                        changed, unary, {"exact": True}, {"exact": True}, identity_exact=True,
                    ))

    def test_component_gate_rejects_retrieval_delta_task_regression_and_preflight(self) -> None:
        full, unary = _supported_metrics(full=True), _supported_metrics(full=False)
        with patch.object(v17, "_component_gate", return_value=True):
            changed = copy.deepcopy(full)
            changed["corresponding_anchor_read_mass"]["top1_accuracy"] = 0.7499
            self.assertFalse(v19._component_gate(
                changed, unary, {"exact": True}, {"exact": True}, identity_exact=True,
            ))
            changed = copy.deepcopy(full)
            changed["arms"]["true"]["mean_probe_nll"] = 0.321
            self.assertFalse(v19._component_gate(
                changed, unary, {"exact": True}, {"exact": True}, identity_exact=True,
            ))
            self.assertFalse(v19._component_gate(
                full, unary, {"exact": True}, {"exact": False}, identity_exact=True,
            ))


class ProvenanceAndSealTests(unittest.TestCase):
    def test_restricted_loader_allows_torch_version_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            allowed = Path(directory) / "torch-version.pt"
            torch.save({"torch_version": torch.__version__}, allowed)
            loaded = v19._safe_torch_load(allowed)
            self.assertIsInstance(loaded["torch_version"], torch.torch_version.TorchVersion)
            self.assertEqual(str(loaded["torch_version"]), str(torch.__version__))

            rejected = Path(directory) / "unsupported.pt"
            torch.save({
                "torch_version": torch.__version__,
                "unsupported": _UnsupportedCheckpointGlobal(),
            }, rejected)
            with self.assertRaises(pickle.UnpicklingError):
                v19._safe_torch_load(rejected)
            loader_source = inspect.getsource(v19._safe_torch_load)
            self.assertIn("weights_only=True", loader_source)
            self.assertIn("torch.torch_version.TorchVersion", loader_source)
            self.assertNotIn("weights_only=False", loader_source)

    def test_source_map_includes_runner_core_tests_leaf_and_consumed_chain(self) -> None:
        source = inspect.getsource(v19._source_hashes)
        for key in (
            "runner", "runner_test", "paired_public_relation_core",
            "paired_public_relation_core_test", "leaf",
            "phase6_paired_comparator_donor", "phase6_oml_relation_donor",
            "v11_public_sidecar_donor",
        ):
            self.assertIn(f'"{key}"', source)
        self.assertIn('f"v18_{key}"', source)
        self.assertIn("**inherited", source)
        root = Path(v19.__file__).resolve().parents[2]
        self.assertEqual(
            v19._sha256(root / "experiments/runners/phase6_v12_champion_paired_graph_context.py"),
            v19.PHASE6_PAIRED_DONOR_SHA256,
        )
        self.assertEqual(
            v19._sha256(root / "experiments/runners/phase6_oml_relation_representation.py"),
            v19.PHASE6_OML_DONOR_SHA256,
        )
        self.assertEqual(
            v19._sha256(root / "src/angler/reasoning/natural_trace_graph_causal_memory.py"),
            v19.V11_SIDECAR_DONOR_SHA256,
        )

    def test_parser_has_no_final_phase_or_output(self) -> None:
        with self.assertRaises(SystemExit):
            v19._parser().parse_args(("--phase", "final"))
        parser_source = inspect.getsource(v19._parser)
        self.assertNotIn("final-result", parser_source)
        run_source = inspect.getsource(v19._run_train_development)
        self.assertIn('"development_authorized": False', run_source)
        self.assertIn('"final_partition_sealed": True', run_source)

    def test_atomic_dual_checkpoints_and_result_bindings_are_literal(self) -> None:
        source = inspect.getsource(v19._run_train_development)
        self.assertIn("v15._atomic_torch(full_path, full_checkpoint)", source)
        self.assertIn("v15._atomic_torch(unary_path, unary_checkpoint)", source)
        self.assertIn('result["full_checkpoint_sha256"]', source)
        self.assertIn('result["unary_checkpoint_sha256"]', source)
        self.assertIn('result["final_seal"] = _final_checkpoint_seal(', source)
        self.assertIn("v15._atomic_json(result_path, result)", source)

    def test_final_checkpoint_seal_reloads_both_arms_and_rejects_wrong_arm(self) -> None:
        sources = {"test_source": "A" * 64}
        common = {
            "identity": v19.IDENTITY,
            "seed": v19.SEED,
            "corpus_id": "test-corpus",
            "source_hashes": sources,
            "preflight": {"exact": True},
            "architecture_evidence": {"exact": True},
            "identity_checks": {"exact": True, "final_partition_sealed": True},
            "qwen_digest": "QWEN",
            "v13_digest": "V13",
            "initial_core_digest": "INITIAL",
            "frozen_identities": {"v18_result_sha256": "V18"},
            "json_normalization_probe": ("tuple", 1),
        }
        full = PairedPublicRelationCreditMemoryCore(temporal_width=v19.TEMPORAL_WIDTH)
        unary = copy.deepcopy(full)
        metrics = {"full": {"balanced_accuracy": 0.75}, "unary": {"balanced_accuracy": 0.50}}

        def payload(arm: str, core: torch.nn.Module, arm_metrics: object) -> dict[str, object]:
            return {
                **common,
                "json_normalization_probe": ["tuple", 1],
                "arm": arm,
                "pair_residual_enabled_by_default": arm == "full",
                "core_digest": v19._model_digest(core),
                "core_state": {name: value.detach().cpu() for name, value in core.state_dict().items()},
                "development_metrics": arm_metrics,
                "affine_calibrator": [1.0, 0.0],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path, unary_path = root / "full.pt", root / "unary.pt"
            v15._atomic_torch(full_path, payload("full", full, metrics["full"]))
            v15._atomic_torch(unary_path, payload("unary", unary, metrics["unary"]))
            with patch.object(v19, "_source_hashes", return_value=sources):
                seal = v19._final_checkpoint_seal(
                    full_path,
                    unary_path,
                    expected_common=common,
                    full_core_digest=v19._model_digest(full),
                    unary_core_digest=v19._model_digest(unary),
                    full_metrics=metrics["full"],
                    unary_metrics=metrics["unary"],
                    full_affine=(1.0, 0.0),
                    unary_affine=(1.0, 0.0),
                )
            self.assertTrue(seal["exact"])
            self.assertTrue(seal["arms"]["full"]["loaded_core_exact"])
            self.assertTrue(seal["arms"]["unary"]["metrics_exact"])

            wrong_full, wrong_unary = root / "wrong-full.pt", root / "wrong-unary.pt"
            v15._atomic_torch(wrong_full, payload("unary", full, metrics["full"]))
            v15._atomic_torch(wrong_unary, payload("unary", unary, metrics["unary"]))
            with patch.object(v19, "_source_hashes", return_value=sources):
                with self.assertRaisesRegex(RuntimeError, "dual-checkpoint seal failed"):
                    v19._final_checkpoint_seal(
                        wrong_full,
                        wrong_unary,
                        expected_common=common,
                        full_core_digest=v19._model_digest(full),
                        unary_core_digest=v19._model_digest(unary),
                        full_metrics=metrics["full"],
                        unary_metrics=metrics["unary"],
                        full_affine=(1.0, 0.0),
                        unary_affine=(1.0, 0.0),
                    )

            bad_full, bad_unary = root / "bad-common-full.pt", root / "bad-common-unary.pt"
            changed_payload = payload("full", full, metrics["full"])
            changed_payload["preflight"] = {"exact": False}
            v15._atomic_torch(bad_full, changed_payload)
            v15._atomic_torch(bad_unary, payload("unary", unary, metrics["unary"]))
            with patch.object(v19, "_source_hashes", return_value=sources):
                with self.assertRaisesRegex(RuntimeError, "dual-checkpoint seal failed"):
                    v19._final_checkpoint_seal(
                        bad_full,
                        bad_unary,
                        expected_common=common,
                        full_core_digest=v19._model_digest(full),
                        unary_core_digest=v19._model_digest(unary),
                        full_metrics=metrics["full"],
                        unary_metrics=metrics["unary"],
                        full_affine=(1.0, 0.0),
                        unary_affine=(1.0, 0.0),
                    )

    def test_frozen_lesion_mappings_and_disposable_smoke_are_literal(self) -> None:
        self.assertEqual(v19.QUERY_SWAP, (3, 2, 5, 4))
        rotate_source = inspect.getsource(v19._rotate_public_sidecars)
        self.assertIn("occupied.roll(-1)", rotate_source)
        smoke_source = inspect.getsource(v19._mechanics_smoke)
        self.assertIn("copy.deepcopy(full)", smoke_source)
        self.assertIn("global_rng_exact", smoke_source)
        controls_source = inspect.getsource(v19._paired_controls)
        self.assertIn("before[name] == core.state_digest(states[name])", controls_source)
        self.assertIn("no_feedback_write_or_optimizer_step", controls_source)

    def test_historical_reasoning_export_is_untouched(self) -> None:
        root = Path(v19.__file__).resolve().parents[2]
        self.assertEqual(
            v19._sha256(root / "src/angler/reasoning/__init__.py"),
            v19.REASONING_EXPORT_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
