from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.runners import phase6_counterfactual_plasticity_router as router
from experiments.runners import phase6_software_pipeline_reconstruction as v13


def _training_batches(
    *,
    replicate: int = 0,
    updates: int = 1,
):
    plan = router.counterfactual_plasticity_fit_plan()
    specification = plan["replicates"][replicate]
    return v13._relation_credit_stream_batches(
        plan["commitments"],
        specification["train_seed_batches"][:updates],
    )


class CounterfactualPlasticityRouterTests(unittest.TestCase):
    def test_v14_plan_is_fresh_paired_and_full_exposure(self) -> None:
        plan = router.counterfactual_plasticity_fit_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-counterfactual-plasticity-router.paired.v14",
        )
        self.assertEqual(plan["replicate_count"], 3)
        self.assertEqual(plan["updates_per_arm_per_replicate"], 80)
        self.assertEqual(plan["streams_per_update"], 8)
        self.assertEqual(plan["streams_per_arm_per_replicate"], 640)
        self.assertEqual(plan["rows_per_arm_per_replicate"], 2_560)
        self.assertEqual(plan["heldout_folds_per_learned_update"], 8)
        self.assertIsNone(plan["minimum_cell_allocation"])
        self.assertFalse(plan["old_conflict_mixer"])
        self.assertFalse(plan["cell_or_stream_identity_input"])
        self.assertFalse(plan["fixed_cell_roles"])
        self.assertFalse(plan["hard_routing"])
        self.assertFalse(plan["deterministic_top_k"])
        self.assertFalse(plan["voting"])
        self.assertFalse(plan["stored_examples"])
        self.assertFalse(plan["deterministic_solver"])
        v13_plan = v13.capacity_matched_relation_cluster_fit_plan()
        old_pairs = {
            pair
            for replicate in v13_plan["replicates"]
            for batch in replicate["train_seed_batches"]
            for pair in batch
        } | {
            pair
            for replicate in v13_plan["replicates"]
            for key in (
                "panel_a_seed_pairs",
                "panel_a_rerender_seed_pairs",
                "panel_b_seed_pairs",
            )
            for pair in replicate[key]
        }
        all_pairs = set()
        for replicate in plan["replicates"]:
            train = {
                pair for batch in replicate["train_seed_batches"] for pair in batch
            }
            panels = (
                set(replicate["panel_a_seed_pairs"])
                | set(replicate["panel_a_rerender_seed_pairs"])
                | set(replicate["panel_b_seed_pairs"])
                | set(replicate["adaptation_seed_pairs"])
                | set(replicate["probe_seed_pairs"])
            )
            self.assertEqual(len(train), 640)
            self.assertEqual(len(panels), 40)
            self.assertFalse(train & panels)
            self.assertFalse((train | panels) & old_pairs)
            self.assertFalse((train | panels) & all_pairs)
            all_pairs |= train | panels
            self.assertEqual(
                replicate["uniform_stream_binding_digest"],
                replicate["learned_stream_binding_digest"],
            )

    def test_v14_pair_has_exact_controller_and_router_initialization(self) -> None:
        uniform, learned, uniform_router, learned_router = (
            router.build_counterfactual_plasticity_pair(0)
        )
        self.assertEqual(
            v13.software_pipeline_model_digest(uniform),
            v13.software_pipeline_model_digest(learned),
        )
        self.assertEqual(
            router.counterfactual_plasticity_router_digest(uniform_router),
            router.counterfactual_plasticity_router_digest(learned_router),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in uniform.parameters()),
            sum(parameter.numel() for parameter in learned.parameters()),
        )
        self.assertEqual(uniform.relation_composer.anchor_weight, 0.5)
        self.assertEqual(learned.relation_composer.anchor_weight, 0.5)

    def test_router_is_cell_and_stream_permutation_equivariant_without_floor(self) -> None:
        torch.manual_seed(2_026_084_201)
        model = router.CounterfactualPlasticityRouter().double()
        evidence = torch.randn(4, 8, 5, dtype=torch.float64)
        initial, logits, enriched = model(evidence)
        torch.testing.assert_close(initial, torch.full_like(initial, 0.25))
        torch.testing.assert_close(logits, torch.zeros_like(logits))
        self.assertEqual(enriched.shape, (4, 8, 20))

        with torch.no_grad():
            first = model.local_encoder[1]
            second = model.local_encoder[3]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 0] = 1.0
            second.weight.zero_()
            second.bias.zero_()
            second.weight[0, 0] = 1.0
            model.scorer.weight.zero_()
            model.scorer.weight[0, 0] = 100.0
        routed = model(evidence)[0]
        self.assertLess(float(routed.detach().min()), 0.125)
        torch.testing.assert_close(
            routed.sum(dim=0),
            torch.ones(8, dtype=torch.float64),
        )
        cell_order = torch.tensor((2, 0, 3, 1))
        stream_order = torch.tensor((5, 1, 7, 2, 0, 6, 3, 4))
        permuted = model(evidence[cell_order][:, stream_order])[0]
        torch.testing.assert_close(
            permuted,
            routed[cell_order][:, stream_order],
        )

    def test_cell_local_evidence_is_composer_independent(self) -> None:
        left, right, _, _ = router.build_counterfactual_plasticity_pair(0)
        with torch.no_grad():
            final = right.relation_composer.residual_scorer[-1].weight
            final.copy_(
                torch.linspace(-1.0, 1.0, final.numel()).reshape_as(final)
            )
        stream = _training_batches()[0][0]
        left_evidence = router.collect_cell_local_evidence(left, (stream,))
        right_evidence = router.collect_cell_local_evidence(right, (stream,))
        torch.testing.assert_close(left_evidence.features, right_evidence.features)
        torch.testing.assert_close(left_evidence.losses, right_evidence.losses)
        torch.testing.assert_close(
            left_evidence.gradient_norms,
            right_evidence.gradient_norms,
        )
        for left_cell, right_cell in zip(
            left_evidence.gradients,
            right_evidence.gradients,
            strict=True,
        ):
            for left_stream, right_stream in zip(
                left_cell,
                right_cell,
                strict=True,
            ):
                for left_value, right_value in zip(
                    left_stream,
                    right_stream,
                    strict=True,
                ):
                    torch.testing.assert_close(left_value, right_value)

    def test_functional_identity_and_router_meta_gradient_do_not_mutate_controller(self) -> None:
        _, controller, _, model = router.build_counterfactual_plasticity_pair(0)
        batch = _training_batches()[0]
        evidence = router.collect_cell_local_evidence(controller, batch)
        visible_only = router.collect_cell_local_evidence(controller, batch[1:])
        torch.testing.assert_close(
            router._features_for_stream_indices(evidence, tuple(range(1, 8))),
            visible_only.features,
        )
        zeros = tuple(
            tuple(
                torch.zeros_like(dict(controller.named_parameters())[name])
                for name in names
            )
            for names in evidence.cell_parameter_names
        )
        ordinary = router._ensemble_stream_objective(controller, batch[0])
        functional = router.functional_heldout_loss(
            controller,
            batch[0],
            evidence.cell_parameter_names,
            zeros,
        )
        torch.testing.assert_close(functional, ordinary, atol=0.0, rtol=0.0)
        before = v13.software_pipeline_model_digest(controller)
        result = router.counterfactual_router_meta_gradients(
            controller,
            model,
            batch,
            evidence,
        )
        self.assertEqual(result.heldout_indices, tuple(range(8)))
        self.assertEqual(
            result.seen_indices,
            tuple(tuple(index for index in range(8) if index != heldout) for heldout in range(8)),
        )
        self.assertTrue(
            any(float(gradient.norm()) > 0.0 for gradient in result.gradients)
        )
        self.assertEqual(before, v13.software_pipeline_model_digest(controller))
        self.assertTrue(all(parameter.grad is None for parameter in controller.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(len(result.fold_post_minus_pre), 8)
        self.assertEqual(len(result.fold_allocations), 8)
        self.assertTrue(all(len(seen) == 7 for seen in result.seen_indices))

    def test_first_uniform_update_matches_manual_update_and_owns_optimizers(self) -> None:
        actual, manual, actual_router, manual_router = (
            router.build_counterfactual_plasticity_pair(1)
        )
        batch = _training_batches(replicate=1)[0]
        evidence = router.collect_cell_local_evidence(manual, batch)
        allocations = evidence.features.new_full((4, 8), 0.25)
        directions = router._routed_cell_directions(
            evidence,
            allocations,
            tuple(range(8)),
        )[0]
        base_weights = evidence.entropic_base_weights
        expected_unclipped = tuple(
            tuple(
                sum(
                    (
                        base_weights[stream_index]
                        * evidence.gradients[cell_index][stream_index][parameter_index]
                        for stream_index in range(8)
                    ),
                    torch.zeros_like(
                        evidence.gradients[cell_index][0][parameter_index]
                    ),
                )
                for parameter_index in range(
                    len(evidence.gradients[cell_index][0])
                )
            )
            for cell_index in range(4)
        )
        global_norm = torch.stack(
            tuple(
                value.square().sum()
                for cell in expected_unclipped
                for value in cell
            )
        ).sum().sqrt()
        scale = min(1.0, router._CELL_DIRECTION_CLIP / float(global_norm))
        for actual_cell, expected_cell in zip(
            directions,
            expected_unclipped,
            strict=True,
        ):
            for actual_value, expected_value in zip(
                actual_cell,
                expected_cell,
                strict=True,
            ):
                torch.testing.assert_close(actual_value, expected_value * scale)
        composer_gradients = router._composer_gradients(manual, batch)[0]
        composer_optimizer = torch.optim.AdamW(
            manual.relation_composer.parameters(),
            lr=router._COMPOSER_LEARNING_RATE,
            weight_decay=0.0,
        )
        router._apply_controller_update(
            manual,
            evidence,
            directions,
            composer_gradients,
            composer_optimizer,
        )
        report = router.fit_counterfactual_plasticity_batches(
            actual,
            actual_router,
            (batch,),
            learned_plasticity=False,
        )
        self.assertEqual(
            v13.software_pipeline_model_digest(actual),
            v13.software_pipeline_model_digest(manual),
        )
        self.assertNotEqual(
            router.counterfactual_plasticity_router_digest(actual_router),
            router.counterfactual_plasticity_router_digest(manual_router),
        )
        self.assertTrue(report["first_allocation_exact_uniform"])
        self.assertEqual(report["cell_update"], "explicit_sgd")
        self.assertEqual(report["composer_update"], "separate_adamw")
        self.assertEqual(report["router_update"], "separate_adamw")
        self.assertFalse(report["router_scores_applied"])
        self.assertTrue(report["sham_router_compute_matched"])
        self.assertTrue(report["router_changed"])
        self.assertTrue(report["controller_before_router_step"])
        self.assertFalse(report["router_affects_current_batch"])
        self.assertTrue(report["controller_grad_fields_clear"])

    def test_checkpoint_roundtrip_and_plan_tamper_detection(self) -> None:
        systems = tuple(
            router.build_counterfactual_plasticity_pair(replicate)
            for replicate in range(3)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v14.pt"
            router.save_counterfactual_plasticity_checkpoint(path, systems)
            loaded = router.load_counterfactual_plasticity_checkpoint(path)
            self.assertEqual(len(loaded), 3)
            for replicate, (original, restored) in enumerate(
                zip(systems, loaded, strict=True)
            ):
                self.assertEqual(
                    router.counterfactual_plasticity_system_digest(
                        *original,
                        replicate,
                    ),
                    router.counterfactual_plasticity_system_digest(
                        *restored,
                        replicate,
                    ),
                )
            payload = torch.load(path, weights_only=True)
            payload["plan_digest"] = "sha256:tampered"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "identity or seed plan"):
                router.load_counterfactual_plasticity_checkpoint(path)

    def test_learning_closure_has_no_identity_solver_or_fixed_routing(self) -> None:
        source = Path(router.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        protected = {
            "CounterfactualPlasticityRouter",
            "collect_cell_local_evidence",
            "counterfactual_router_meta_gradients",
            "fit_counterfactual_plasticity_batches",
            "functional_heldout_loss",
        }
        selected = [
            node for node in tree.body if getattr(node, "name", None) in protected
        ]
        self.assertEqual({getattr(node, "name", None) for node in selected}, protected)
        forbidden = {
            "mechanism_commitment",
            "motif",
            "cell_id",
            "stream_id",
            "judge_software_pipeline_attempt",
            "make_software_pipeline_control_stream",
            "search_teacher_plan",
            "bidirectionaloperatorplanner",
            "anonymousconflictmixer",
            "topk",
            "vote",
            "replay",
            "solver",
        }
        for node in selected:
            names = {
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            attributes = {
                child.attr.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            self.assertTrue(
                forbidden.isdisjoint(names | attributes),
                msg=getattr(node, "name", ""),
            )


if __name__ == "__main__":
    unittest.main()
