from __future__ import annotations

import copy
import inspect
from pathlib import Path
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
from experiments.runners import barlow_semantic_invariance_v18 as v18
from experiments.runners import paired_latent_contingency_credit_v15 as v15
from experiments.runners import shared_semantic_metric_credit_v17 as v17


def _fake_encoded(pair: object, seed: int, *, collide: bool = False) -> v15.EncodedTwinPair:
    generator = torch.Generator().manual_seed(90_000 + seed)
    relation = (
        torch.zeros(6, 64)
        if collide else torch.randn(6, 64, generator=generator)
    )
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
            relation_features=relation.clone(), base_logits=base.clone(),
            temporal_features=temporal.clone(),
            outcomes=torch.tensor(value.supervision.outcome_values, dtype=torch.float32),
            relation_classes=tuple(value.supervision.relation_classes),
            public_payload_sha256="PUBLIC", input_tensor_sha256="INPUT",
        )

    return v15.EncodedTwinPair(pair.pair_ref, episode(pair.first), episode(pair.second))


def _read(top1: float, mass: float, margin: float) -> dict[str, float]:
    return {
        "top1_accuracy": top1,
        "mean_matching_mass": mass,
        "mean_matching_minus_competitor_margin": margin,
    }


def _metrics(top1: float, mass: float, margin: float, ba: float, nll: float) -> dict[str, object]:
    return {
        "corresponding_anchor_read_mass": _read(top1, mass, margin),
        "arms": {"true": {"balanced_accuracy": ba, "mean_probe_nll": nll}},
    }


class MaskAndCanonicalBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_paired_latent_contingency_credit_v15()
        cls.public = cls.corpus.train[:8]
        cls.encoded = tuple(
            _fake_encoded(pair, index) for index, pair in enumerate(cls.public)
        )

    def test_mask_schedule_exact_and_global_rng_isolated(self) -> None:
        torch.manual_seed(773)
        before = v18._rng_snapshot()
        masks, report = v18._precompute_mask_schedule()
        after = v18._rng_snapshot()
        self.assertTrue(v18._rng_exact(before, after))
        self.assertEqual(tuple(masks.shape), v18.MASK_SHAPE)
        self.assertEqual(int(masks.sum()), v18.MASK_RETAINED_COUNT)
        self.assertEqual(v18._mask_digest(masks), v18.MASK_SHA256)
        self.assertTrue(report["exact"])
        self.assertTrue(v18._mask_schedule_identity(masks)["exact"])
        changed = masks.clone()
        changed[0, 0, 0, 0] = 1 - changed[0, 0, 0, 0]
        self.assertFalse(v18._mask_schedule_identity(changed)["exact"])

    def test_public_payload_dedup_is_exactly_twin_only(self) -> None:
        rows, report = v18._public_relation_batch(self.public, self.encoded)
        self.assertEqual(tuple(rows.shape), (48, 64))
        self.assertEqual(report["row_count"], 48)
        self.assertEqual(len(report["ordered_event_digests"]), 48)
        self.assertEqual(len(set(report["ordered_event_digests"])), 48)
        self.assertEqual(len(report["ordered_digest_chain_sha256"]), 64)

    def test_equal_features_do_not_merge_distinct_public_events(self) -> None:
        encoded = tuple(
            _fake_encoded(pair, 100 + index, collide=True)
            for index, pair in enumerate(self.public)
        )
        rows, report = v18._public_relation_batch(self.public, encoded)
        self.assertEqual(tuple(rows.shape), (48, 64))
        self.assertGreater(report["distinct_feature_collisions_retained"], 0)

    def test_twin_payload_with_changed_relation_row_fails_closed(self) -> None:
        encoded = list(copy.deepcopy(self.encoded))
        pair = encoded[0]
        second = copy.copy(pair.second)
        object.__setattr__(second, "relation_features", second.relation_features.clone())
        second.relation_features[0, 0] += 1.0
        encoded[0] = v15.EncodedTwinPair(pair.pair_ref, pair.first, second)
        with self.assertRaisesRegex(RuntimeError, "twin relation mismatch"):
            v18._public_relation_batch(self.public, encoded)

    def test_auxiliary_batch_source_does_not_read_sidecars_or_outcomes(self) -> None:
        source = inspect.getsource(v18._public_relation_batch)
        for forbidden in (".outcomes", ".relation_classes", ".supervision", ".metadata"):
            self.assertNotIn(forbidden, source)


class BarlowMechanicsTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.core = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        self.rows = torch.randn(48, 64)
        generator = torch.Generator().manual_seed(42)
        self.masks = (torch.rand(2, 48, 64, generator=generator) < 0.8).to(torch.uint8)

    def test_population_standardization_and_zero_variance_are_finite(self) -> None:
        codes = torch.randn(48, 32)
        codes[:, 0] = 7.0
        standardized, variance = v18._standardize(codes)
        self.assertEqual(float(variance[0]), 0.0)
        self.assertTrue(torch.equal(standardized[:, 0], torch.zeros(48)))
        manual = ((codes[:, 1] - codes[:, 1].mean()).square()).mean()
        self.assertTrue(torch.equal(variance[1], manual))
        self.assertTrue(torch.isfinite(standardized).all())

    def test_barlow_is_row_permutation_and_view_swap_invariant(self) -> None:
        loss, report = v18._barlow_objective(self.core, self.rows, self.masks)
        permutation = torch.arange(47, -1, -1)
        permuted, _ = v18._barlow_objective(
            self.core, self.rows[permutation], self.masks[:, permutation],
        )
        swapped, _ = v18._barlow_objective(self.core, self.rows, self.masks.flip(0))
        self.assertLessEqual(
            abs(float(loss.detach()) - float(permuted.detach())), 1.0e-6,
        )
        self.assertLessEqual(
            abs(float(loss.detach()) - float(swapped.detach())), 1.0e-6,
        )
        self.assertTrue(report["finite"])
        self.assertIn("effective_rank", report)

    def test_auxiliary_gradient_reaches_only_every_semantic_parameter(self) -> None:
        loss, _ = v18._barlow_objective(self.core, self.rows, self.masks)
        named = tuple(self.core.named_parameters())
        gradients = torch.autograd.grad(
            loss, tuple(parameter for _, parameter in named), allow_unused=True,
        )
        semantic = []
        outside = []
        for (name, _), gradient in zip(named, gradients, strict=True):
            reached = gradient is not None and bool((gradient != 0).any())
            if name.startswith("semantic_metric_network."):
                semantic.append(reached)
            elif reached:
                outside.append(name)
        self.assertTrue(semantic and all(semantic))
        self.assertEqual(outside, [])

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "requires the frozen two-GPU workstation",
    )
    def test_cpu_canonical_rows_cross_to_cuda1_at_objective_boundary(self) -> None:
        core = SharedSemanticMetricCreditMemoryCore(temporal_width=4).to("cuda:1")
        self.assertEqual(self.rows.device.type, "cpu")
        self.assertEqual(self.masks.device.type, "cpu")
        loss, diagnostics = v18._barlow_objective(core, self.rows, self.masks)
        self.assertEqual(loss.device, torch.device("cuda:1"))
        self.assertTrue(diagnostics["finite"])


class PairedTrainingAndGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_paired_latent_contingency_credit_v15()
        cls.public = cls.corpus.train[:8]
        cls.encoded = tuple(
            _fake_encoded(pair, 500 + index) for index, pair in enumerate(cls.public)
        )

    def test_initial_task_losses_gradients_and_chronology_are_exact(self) -> None:
        full = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        off = copy.deepcopy(full)
        report = v18._task_parity_preflight(full, off, self.encoded, tuple(range(8)))
        self.assertTrue(report["exact"], report)
        self.assertTrue(report["task_loss_operands_exact"])
        self.assertTrue(report["task_gradients_exact"])

    def test_one_paired_optimizer_update_completes(self) -> None:
        full = SharedSemanticMetricCreditMemoryCore(temporal_width=4)
        off = copy.deepcopy(full)
        masks = torch.ones(1, 2, 48, 64, dtype=torch.uint8)
        before_full = v18._model_digest(full)
        before_off = v18._model_digest(off)
        self.assertEqual(before_full, before_off)
        with patch.object(v18, "TRAIN_PAIRS", 8), patch.object(v18, "TRAIN_UPDATES", 1):
            report, _, _ = v18._paired_fit(
                full, off, self.public, self.encoded, (tuple(range(8)),), masks,
            )
        self.assertEqual(report["completed_updates"], 1)
        self.assertTrue(report["updates"][0]["chronology_exact"])
        self.assertNotEqual(v18._model_digest(full), before_full)
        self.assertNotEqual(v18._model_digest(off), before_off)

    def test_objective_gate_exact_boundaries_and_durable_conjunction(self) -> None:
        full = _metrics(0.76, 0.31, 0.11, 0.71, 0.20)
        off = _metrics(0.65, 0.25, 0.05, 0.71, 0.22)
        preflight = {"exact": True}
        architecture = {"exact": True}
        with patch.object(v17, "_component_gate", return_value=True), patch.object(
            v17, "_full_gate", return_value=True,
        ):
            self.assertTrue(v18._objective_gate(
                full, off, architecture, preflight, identity_exact=True,
            ))
            self.assertTrue(v18._full_gate(
                full, off, architecture, preflight, identity_exact=True,
            ))
            for key, value in (
                ("top1_accuracy", 0.749999),
                ("mean_matching_mass", 0.299999),
                ("mean_matching_minus_competitor_margin", 0.099999),
            ):
                changed = copy.deepcopy(full)
                changed["corresponding_anchor_read_mass"][key] = value
                with self.subTest(key=key):
                    self.assertFalse(v18._objective_gate(
                        changed, off, architecture, preflight, identity_exact=True,
                    ))
        with patch.object(v17, "_component_gate", return_value=True), patch.object(
            v17, "_full_gate", return_value=False,
        ):
            self.assertTrue(v18._objective_gate(
                full, off, architecture, preflight, identity_exact=True,
            ))
            self.assertFalse(v18._full_gate(
                full, off, architecture, preflight, identity_exact=True,
            ))

    def test_task_regression_and_preflight_failure_reject(self) -> None:
        full = _metrics(0.80, 0.40, 0.20, 0.70, 0.22)
        off = _metrics(0.60, 0.20, 0.00, 0.71, 0.20)
        with patch.object(v17, "_component_gate", return_value=True):
            self.assertFalse(v18._objective_gate(
                full, off, {"exact": True}, {"exact": True}, identity_exact=True,
            ))
            full["arms"]["true"]["balanced_accuracy"] = 0.71
            self.assertTrue(v18._objective_gate(
                full, off, {"exact": True}, {"exact": True}, identity_exact=True,
            ))
            self.assertFalse(v18._objective_gate(
                full, off, {"exact": True}, {"exact": False}, identity_exact=True,
            ))


class ProvenanceAndSealTests(unittest.TestCase):
    def test_source_map_includes_runner_test_leaf_and_consumed_chain(self) -> None:
        hashes = v18._source_hashes()
        self.assertIn("runner", hashes)
        self.assertIn("runner_test", hashes)
        self.assertEqual(hashes["leaf"], v18.LEAF_SHA256)
        self.assertEqual(hashes["reasoning_export"], v18.REASONING_EXPORT_SHA256)
        self.assertEqual(hashes["v17_runner"], v18.V17_RUNNER_SHA256)
        self.assertEqual(hashes["v17_shared_metric_core"], v18.V17_CORE_SHA256)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_v18_has_no_final_phase_or_final_output(self) -> None:
        parser = v18._parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(("--phase", "final"))
        source = inspect.getsource(v18._parser)
        self.assertNotIn("final-result", source)
        run_source = inspect.getsource(v18._run_train_development)
        self.assertIn('"development_authorized": False', run_source)
        self.assertIn('"final_partition_sealed": True', run_source)

    def test_result_cross_binds_both_atomic_checkpoints_and_environment(self) -> None:
        source = inspect.getsource(v18._run_train_development)
        self.assertIn('v15._atomic_torch(full_path, full_checkpoint)', source)
        self.assertIn('v15._atomic_torch(off_path, off_checkpoint)', source)
        self.assertIn('result["full_checkpoint_sha256"]', source)
        self.assertIn('result["objective_off_checkpoint_sha256"]', source)
        self.assertIn('"environment": environment', source)
        self.assertIn('"peak_angler_gpu_bytes": peak_angler_gpu_bytes', source)
        environment_source = inspect.getsource(v18._environment_report)
        self.assertIn('device.index != 1', environment_source)
        self.assertIn('"torch_cuda": torch.version.cuda', environment_source)

    def test_preflight_uses_mask_copy_and_full_schedule_is_revalidated(self) -> None:
        source = inspect.getsource(v18._run_train_development)
        self.assertIn("masks[0].clone()", source)
        self.assertIn("mask_after_preflight = _mask_schedule_identity(masks)", source)
        self.assertIn("mask_after_training = _mask_schedule_identity(masks)", source)
        protocol = inspect.getsource(v18._protocol)
        self.assertNotIn('"mask_schedule_fixed": True', protocol)
        self.assertIn('training["mask_schedule_after_preflight"]["exact"]', protocol)
        self.assertIn('training["mask_schedule_after_training"]["exact"]', protocol)

    def test_historical_reasoning_export_is_untouched(self) -> None:
        root = Path(v18.__file__).resolve().parents[2]
        self.assertEqual(
            v18._sha256(root / "src/angler/reasoning/__init__.py"),
            v18.REASONING_EXPORT_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
