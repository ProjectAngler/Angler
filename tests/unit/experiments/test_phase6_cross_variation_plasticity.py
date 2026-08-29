from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from experiments.runners import phase6_cross_variation_plasticity as v15
from experiments.runners import phase6_software_pipeline_reconstruction as v13


def _with_stateful_cell_optimizer(arm: v15.CrossVariationArm) -> None:
    groups = v15._cell_parameter_groups(arm.controller)
    arm.cell_optimizer_state = tuple(
        tuple(
            v15.AdamWSlot(
                step=3,
                exp_avg=0.01 * torch.tanh(parameter.detach()),
                exp_avg_sq=1.0e-5 + 0.001 * parameter.detach().square(),
            )
            for _, parameter in group
        )
        for group in groups
    )


def _permute_real_cells(
    arm: v15.CrossVariationArm,
    order: tuple[int, ...],
) -> None:
    old_cells = arm.controller.relation_cells
    arm.controller.relation_cells = nn.ModuleList(tuple(old_cells[index] for index in order))
    arm.cell_optimizer_state = tuple(arm.cell_optimizer_state[index] for index in order)


def _maximum_tensor_delta(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    if left.keys() != right.keys():
        raise AssertionError("state dictionaries lost key alignment")
    return max(
        float((left[name].detach() - right[name].detach()).abs().max().item())
        if left[name].is_floating_point()
        else 0.0 if torch.equal(left[name], right[name]) else float("inf")
        for name in left
    )


def _assert_optimizer_state_close(
    case: unittest.TestCase,
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    case.assertEqual(left_state["param_groups"], right_state["param_groups"])
    case.assertEqual(left_state["state"].keys(), right_state["state"].keys())
    for parameter_index in left_state["state"]:
        left_slot = left_state["state"][parameter_index]
        right_slot = right_state["state"][parameter_index]
        case.assertEqual(left_slot.keys(), right_slot.keys())
        for name in left_slot:
            left_value = left_slot[name]
            right_value = right_slot[name]
            if isinstance(left_value, torch.Tensor):
                torch.testing.assert_close(
                    left_value,
                    right_value,
                    atol=1.0e-6,
                    rtol=1.0e-6,
                )
            else:
                case.assertEqual(left_value, right_value)


class CrossVariationPlasticityTests(unittest.TestCase):
    def test_plan_schedule_balance_digests_exposure_and_denials(self) -> None:
        plan = v15.cross_variation_fit_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-anonymous-cross-variation-plasticity.paired.v15",
        )
        self.assertEqual(plan["replicate_count"], 3)
        self.assertEqual(plan["updates_per_arm_per_replicate"], 80)
        self.assertEqual(plan["streams_per_update"], 8)
        self.assertEqual(plan["streams_per_lane"], 4)
        self.assertEqual(plan["rows_per_arm_per_replicate"], 2_560)
        self.assertEqual(plan["virtual_folds_per_update"], 2)
        self.assertEqual(plan["commitment_schedule_payload_bytes"], 801)
        self.assertEqual(
            plan["commitment_schedule_sha256"],
            "8B18860D42DB4DF6979EBA3148CE94E817CF98D2A25014C58E15D34D46F8F7D1",
        )
        self.assertEqual(plan["adaptation_schedule_payload_bytes"], 41)
        self.assertEqual(
            plan["adaptation_schedule_sha256"],
            "6B449614DC824EF71022B622FF8348D444942D2F00701E6810BD119965F1D04D",
        )
        schedule = plan["commitment_schedule"]
        self.assertEqual(len(schedule), 80)
        for commitment in range(8):
            self.assertEqual(sum(commitment in row for row in schedule), 40)
            for slot in range(4):
                self.assertEqual(sum(row[slot] == commitment for row in schedule), 10)
        pairs = {
            pair: sum(pair[0] in row and pair[1] in row for row in schedule)
            for pair in (
                (left, right) for left in range(8) for right in range(left + 1, 8)
            )
        }
        self.assertEqual(set(pairs.values()), {17, 18})
        self.assertEqual(
            {pair for pair, count in pairs.items() if count == 18},
            {(0, 1), (2, 3), (4, 5), (6, 7)},
        )
        position_signatures = tuple(
            tuple(row[slot] == commitment for row in schedule)
            for commitment in range(8)
            for slot in range(4)
        )
        self.assertEqual(len(set(position_signatures)), 32)
        self.assertEqual(
            min(
                sum(left != right for left, right in zip(a, b, strict=True))
                for index, a in enumerate(position_signatures)
                for b in position_signatures[index + 1 :]
            ),
            10,
        )
        all_pairs = set()
        for replicate in plan["replicates"]:
            self.assertEqual(len(replicate["train_updates"]), 80)
            self.assertEqual(len(replicate["adaptation_updates"]), 4)
            self.assertEqual(
                replicate["uniform_stream_binding_digest"],
                replicate["learned_stream_binding_digest"],
            )
            current = {
                pair
                for update in replicate["train_updates"]
                for key in ("lane_a_seed_pairs", "lane_b_seed_pairs")
                for pair in update[key]
            } | {
                pair
                for update in replicate["adaptation_updates"]
                for key in ("lane_a_seed_pairs", "lane_b_seed_pairs")
                for pair in update[key]
            }
            for key in (
                "panel_a_seed_pairs",
                "panel_a_rerender_seed_pairs",
                "panel_b_seed_pairs",
                "probe_seed_pairs",
            ):
                current.update(replicate[key])
            self.assertEqual(len(current), 704)
            self.assertFalse(current & all_pairs)
            all_pairs.update(current)
            for update in replicate["train_updates"]:
                self.assertEqual(set(update["lane_a_order"]), {("A", i) for i in range(4)})
                self.assertEqual(set(update["lane_b_order"]), {("B", i) for i in range(4)})
                self.assertEqual(
                    set(update["real_order"]),
                    {(lane, i) for lane in ("A", "B") for i in range(4)},
                )
        self.assertIsNone(plan["meta_difference_scale"])
        self.assertIsNone(plan["minimum_cell_allocation"])
        for key in (
            "cell_lane_pair_update_seed_task_package_motif_identity_input",
            "fixed_cell_roles",
            "hard_routing",
            "deterministic_top_k",
            "voting",
            "gradient_surgery",
            "stored_examples_or_replay",
            "deterministic_solver",
        ):
            self.assertFalse(plan[key])

    def test_functional_adamw_matches_ordinary_and_distinguishes_none_from_zero(self) -> None:
        torch.manual_seed(2_026_084_501)
        reference = [
            torch.nn.Parameter(torch.randn(5, 3)),
            torch.nn.Parameter(torch.randn(7)),
        ]
        functional = tuple(value.detach().clone() for value in reference)
        rates = (3.0e-4, 1.0e-3)
        optimizer = torch.optim.AdamW(
            [
                {"params": (reference[0],), "lr": rates[0]},
                {"params": (reference[1],), "lr": rates[1]},
            ],
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        state = tuple(
            v15.AdamWSlot(0, torch.zeros_like(value), torch.zeros_like(value))
            for value in functional
        )
        for step in range(4):
            gradients = tuple(
                torch.randn_like(value) * (step + 1) for value in functional
            )
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(reference, gradients, strict=True):
                parameter.grad = gradient.clone()
            optimizer.step()
            functional, state = v15.functional_adamw_step(
                functional,
                gradients,
                state,
                rates,
            )
            for actual, expected in zip(functional, reference, strict=True):
                torch.testing.assert_close(actual, expected, atol=1.0e-7, rtol=1.0e-6)
            self.assertTrue(all(slot.step == step + 1 for slot in state))
        skipped_parameters, skipped_state = v15.functional_adamw_step(
            functional,
            (None, torch.zeros_like(functional[1])),
            state,
            rates,
        )
        self.assertIs(skipped_parameters[0], functional[0])
        self.assertIs(skipped_state[0], state[0])
        self.assertEqual(skipped_state[0].step, 4)
        self.assertEqual(skipped_state[1].step, 5)
        self.assertFalse(torch.equal(skipped_state[1].exp_avg, state[1].exp_avg))

    def test_pair_router_and_complete_optimizer_starts_are_exact(self) -> None:
        uniform, learned = v15.build_cross_variation_pair(0)
        self.assertEqual(
            v15.cross_variation_arm_digest(uniform),
            v15.cross_variation_arm_digest(learned),
        )
        self.assertEqual(
            v13.software_pipeline_model_digest(uniform.controller),
            v13.software_pipeline_model_digest(learned.controller),
        )
        self.assertTrue(
            all(
                slot.step == 0
                and torch.count_nonzero(slot.exp_avg) == 0
                and torch.count_nonzero(slot.exp_avg_sq) == 0
                for cell in uniform.cell_optimizer_state
                for slot in cell
            )
        )
        evidence = torch.randn(4, 4, 7)
        route = uniform.router(evidence)[0]
        torch.testing.assert_close(route, torch.full_like(route, 0.25), atol=0.0, rtol=0.0)
        cell_order = torch.tensor((2, 0, 3, 1))
        stream_order = torch.tensor((3, 1, 0, 2))
        torch.testing.assert_close(
            uniform.router(evidence[cell_order][:, stream_order])[0],
            route[cell_order][:, stream_order],
        )

    def test_symmetric_meta_is_nonmutating_lane_swap_invariant_and_zero_start_scoped(self) -> None:
        _, learned = v15.build_cross_variation_pair(0)
        batch = v15.build_training_batches(0, updates=1)[0]
        evidence = v15.collect_cross_variation_evidence(
            learned.controller,
            batch.streams,
            learned.cell_optimizer_state,
        )
        before = v15.cross_variation_arm_digest(learned)
        result = v15.cross_variation_meta_gradients(
            learned.controller,
            learned.router,
            batch,
            learned.cell_optimizer_state,
            evidence,
        )
        self.assertEqual(before, v15.cross_variation_arm_digest(learned))
        gradients = dict(result.parameter_gradient_norms)
        self.assertGreater(gradients["scorer.weight"], 0.0)
        self.assertEqual(
            max(value for name, value in gradients.items() if name != "scorer.weight"),
            0.0,
        )
        self.assertNotEqual(result.objective, 0.0)
        self.assertEqual(len(result.post_losses), 2)
        swapped = v15.CrossVariationBatch(
            streams=batch.streams,
            lane_a_indices=batch.lane_b_indices,
            lane_b_indices=batch.lane_a_indices,
            lane_a_slots=batch.lane_b_slots,
            lane_b_slots=batch.lane_a_slots,
            real_order=batch.real_order,
            procedure_indices=batch.procedure_indices,
            topology_surface_pairs=batch.topology_surface_pairs,
        )
        swapped_result = v15.cross_variation_meta_gradients(
            learned.controller,
            learned.router,
            swapped,
            learned.cell_optimizer_state,
            evidence,
        )
        self.assertAlmostEqual(result.objective, swapped_result.objective, places=6)
        for left, right in zip(result.gradients, swapped_result.gradients, strict=True):
            torch.testing.assert_close(left, right, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(all(parameter.grad is None for parameter in learned.controller.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in learned.router.parameters()))

    def test_real_cell_optimizer_meta_and_committed_update_permutation_twin(self) -> None:
        torch.set_num_threads(1)
        _, ordinary = v15.build_cross_variation_pair(0)
        _with_stateful_cell_optimizer(ordinary)
        permuted = v15._copy_cross_variation_arm(ordinary)
        with torch.no_grad():
            for arm in (ordinary, permuted):
                arm.router.scorer.weight.copy_(
                    torch.linspace(
                        -0.025,
                        0.025,
                        arm.router.scorer.weight.numel(),
                    ).reshape_as(arm.router.scorer.weight)
                )
        order = (2, 0, 3, 1)
        inverse = tuple(order.index(index) for index in range(4))
        _permute_real_cells(permuted, order)
        batch = v15.build_training_batches(0, updates=1)[0]
        ordinary_evidence = v15.collect_cross_variation_evidence(
            ordinary.controller,
            batch.streams,
            ordinary.cell_optimizer_state,
        )
        permuted_evidence = v15.collect_cross_variation_evidence(
            permuted.controller,
            batch.streams,
            permuted.cell_optimizer_state,
        )
        torch.testing.assert_close(
            permuted_evidence.features,
            ordinary_evidence.features[list(order)],
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        torch.testing.assert_close(
            permuted_evidence.base.ensemble_stream_losses,
            ordinary_evidence.base.ensemble_stream_losses,
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        ordinary_allocations, ordinary_a, ordinary_b = v15._combined_allocations(
            ordinary_evidence,
            batch,
            ordinary.router,
            learned_plasticity=True,
        )
        permuted_allocations, permuted_a, permuted_b = v15._combined_allocations(
            permuted_evidence,
            batch,
            permuted.router,
            learned_plasticity=True,
        )
        torch.testing.assert_close(
            permuted_allocations,
            ordinary_allocations[list(order)],
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        torch.testing.assert_close(permuted_a, ordinary_a[list(order)], atol=1.0e-6, rtol=1.0e-6)
        torch.testing.assert_close(permuted_b, ordinary_b[list(order)], atol=1.0e-6, rtol=1.0e-6)
        ordinary_meta = v15.cross_variation_meta_gradients(
            ordinary.controller,
            ordinary.router,
            batch,
            ordinary.cell_optimizer_state,
            ordinary_evidence,
        )
        permuted_meta = v15.cross_variation_meta_gradients(
            permuted.controller,
            permuted.router,
            batch,
            permuted.cell_optimizer_state,
            permuted_evidence,
        )
        self.assertAlmostEqual(ordinary_meta.objective, permuted_meta.objective, places=6)
        for left, right in zip(ordinary_meta.gradients, permuted_meta.gradients, strict=True):
            torch.testing.assert_close(left, right, atol=1.0e-6, rtol=1.0e-6)
        ordinary_directions = v15._routed_directions(
            ordinary_evidence,
            ordinary_allocations,
            tuple(range(8)),
        )[0]
        permuted_directions = v15._routed_directions(
            permuted_evidence,
            permuted_allocations,
            tuple(range(8)),
        )[0]
        for permuted_cell, ordinary_index in zip(permuted_directions, order, strict=True):
            for left, right in zip(
                permuted_cell,
                ordinary_directions[ordinary_index],
                strict=True,
            ):
                torch.testing.assert_close(left, right, atol=1.0e-6, rtol=1.0e-6)
        ordinary_composer = v15._composer_gradients(
            ordinary.controller,
            batch.streams,
        )[0]
        permuted_composer = v15._composer_gradients(
            permuted.controller,
            batch.streams,
        )[0]
        for left, right in zip(ordinary_composer, permuted_composer, strict=True):
            torch.testing.assert_close(left, right, atol=1.0e-6, rtol=1.0e-6)
        v15._apply_cell_adamw_update(ordinary, ordinary_directions)
        v15._apply_cell_adamw_update(permuted, permuted_directions)
        v15._apply_owned_optimizer_gradients(
            ordinary.composer_optimizer,
            tuple(ordinary.controller.relation_composer.parameters()),
            ordinary_composer,
        )
        v15._apply_owned_optimizer_gradients(
            permuted.composer_optimizer,
            tuple(permuted.controller.relation_composer.parameters()),
            permuted_composer,
        )
        v15._apply_owned_optimizer_gradients(
            ordinary.router_optimizer,
            tuple(ordinary.router.parameters()),
            ordinary_meta.gradients,
        )
        v15._apply_owned_optimizer_gradients(
            permuted.router_optimizer,
            tuple(permuted.router.parameters()),
            permuted_meta.gradients,
        )
        _permute_real_cells(permuted, inverse)
        controller_deltas = {
            name: float(
                (
                    ordinary.controller.state_dict()[name].detach()
                    - permuted.controller.state_dict()[name].detach()
                )
                .abs()
                .max()
                .item()
            )
            for name in ordinary.controller.state_dict()
            if ordinary.controller.state_dict()[name].is_floating_point()
        }
        maximum_controller_delta = max(
            controller_deltas.items(),
            key=lambda item: item[1],
        )
        if maximum_controller_delta[1] > 1.0e-6:
            self.fail(f"maximum controller delta: {maximum_controller_delta!r}")
        self.assertLessEqual(
            _maximum_tensor_delta(ordinary.router.state_dict(), permuted.router.state_dict()),
            1.0e-6,
        )
        _assert_optimizer_state_close(
            self,
            ordinary.composer_optimizer,
            permuted.composer_optimizer,
        )
        _assert_optimizer_state_close(
            self,
            ordinary.router_optimizer,
            permuted.router_optimizer,
        )
        for ordinary_cell, permuted_cell in zip(
            ordinary.cell_optimizer_state,
            permuted.cell_optimizer_state,
            strict=True,
        ):
            for ordinary_slot, permuted_slot in zip(ordinary_cell, permuted_cell, strict=True):
                self.assertEqual(ordinary_slot.step, permuted_slot.step)
                torch.testing.assert_close(
                    ordinary_slot.exp_avg,
                    permuted_slot.exp_avg,
                    atol=1.0e-6,
                    rtol=1.0e-6,
                )
                torch.testing.assert_close(
                    ordinary_slot.exp_avg_sq,
                    permuted_slot.exp_avg_sq,
                    atol=1.0e-6,
                    rtol=1.0e-6,
                )

    def test_first_paired_update_is_exact_twin_then_next_route_and_upstream_are_live(self) -> None:
        uniform, learned = v15.build_cross_variation_pair(1)
        first, second = v15.build_training_batches(1, updates=2)
        uniform_report = v15.fit_cross_variation_batches(
            uniform,
            (first,),
            learned_plasticity=False,
        )
        learned_report = v15.fit_cross_variation_batches(
            learned,
            (first,),
            learned_plasticity=True,
        )
        self.assertTrue(uniform_report["first_allocation_exact_uniform"])
        self.assertTrue(learned_report["first_allocation_exact_uniform"])
        self.assertEqual(
            v15.cross_variation_arm_digest(uniform),
            v15.cross_variation_arm_digest(learned),
        )
        self.assertTrue(uniform_report["sham_router_compute_matched"])
        self.assertEqual(uniform_report["cell_update"], "pure_functional_adamw")
        self.assertEqual(
            uniform_report["cell_update"],
            uniform_report["virtual_cell_update"],
        )
        self.assertTrue(
            all(slot.step == 1 for cell in learned.cell_optimizer_state for slot in cell)
        )
        evidence = v15.collect_cross_variation_evidence(
            learned.controller,
            second.streams,
            learned.cell_optimizer_state,
        )
        a_route = learned.router(v15._features_for_indices(evidence, second.lane_a_indices))[0]
        b_route = learned.router(v15._features_for_indices(evidence, second.lane_b_indices))[0]
        uniform_route = torch.full_like(a_route, 0.25)
        self.assertGreater(
            max(
                float((a_route - uniform_route).detach().abs().max()),
                float((b_route - uniform_route).detach().abs().max()),
            ),
            1.0e-6,
        )
        meta = v15.cross_variation_meta_gradients(
            learned.controller,
            learned.router,
            second,
            learned.cell_optimizer_state,
            evidence,
        )
        self.assertGreater(
            max(
                value
                for name, value in meta.parameter_gradient_norms
                if name != "scorer.weight"
            ),
            0.0,
        )

    def test_checkpoint_roundtrip_preserves_complete_lineage_and_rejects_plan_tamper(self) -> None:
        systems = tuple(v15.build_cross_variation_pair(index) for index in range(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v15.pt"
            v15.save_cross_variation_checkpoint(path, systems)
            restored = v15.load_cross_variation_checkpoint(path)
            for original_pair, restored_pair in zip(systems, restored, strict=True):
                for original, clone in zip(original_pair, restored_pair, strict=True):
                    self.assertEqual(
                        v15.cross_variation_arm_digest(original),
                        v15.cross_variation_arm_digest(clone),
                    )
            payload = torch.load(path, weights_only=True)
            payload["plan_digest"] = "sha256:tampered"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "identity or seed plan"):
                v15.load_cross_variation_checkpoint(path)

    def test_classification_precedence_requires_nonzero_complete_support(self) -> None:
        classify = v15.classify_cross_variation_result
        self.assertEqual(
            classify(
                integrity_passed=False,
                uniform_competent=True,
                uniform_materially_better=True,
                learned_competent=True,
                every_support_rule_passed=True,
            ),
            "INVALID_NO_CLAIM",
        )
        self.assertEqual(
            classify(
                integrity_passed=True,
                uniform_competent=True,
                uniform_materially_better=True,
                learned_competent=False,
                every_support_rule_passed=False,
            ),
            "PLASTICITY_ROUTER_HARMFUL",
        )
        self.assertEqual(
            classify(
                integrity_passed=True,
                uniform_competent=False,
                uniform_materially_better=False,
                learned_competent=False,
                every_support_rule_passed=False,
            ),
            "NO_COMPETENCE",
        )
        self.assertEqual(
            classify(
                integrity_passed=True,
                uniform_competent=False,
                uniform_materially_better=False,
                learned_competent=True,
                every_support_rule_passed=True,
            ),
            "PLASTICITY_ROUTER_SUPPORTED",
        )
        self.assertEqual(
            classify(
                integrity_passed=True,
                uniform_competent=False,
                uniform_materially_better=False,
                learned_competent=True,
                every_support_rule_passed=False,
            ),
            "PLASTICITY_ROUTER_NULL",
        )

    def test_learning_closure_has_no_identity_replay_solver_or_fixed_routing(self) -> None:
        source = Path(v15.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        protected_functions = {
            "collect_cross_variation_evidence",
            "cross_variation_meta_gradients",
            "fit_cross_variation_batches",
            "functional_adamw_step",
            "_apply_cell_adamw_update",
            "_apply_owned_optimizer_gradients",
            "_collect_homogeneous_cell_evidence",
            "_combined_allocations",
            "_composer_gradients",
            "_features_for_indices",
            "_functional_target_loss",
            "_global_clip_directions",
            "_lane_allocations",
            "_moment_alignment",
            "_public_rows_with_stable_separation",
            "_routed_directions",
            "_stable_cosine_similarity",
            "_stable_public_separation_losses",
            "_v15_batch_objective",
            "_v15_stream_objective",
            "_virtual_adamw_parameters",
        }
        protected_methods = {
            "SymmetricV15RelationComposer": {"forward"},
            "SymmetricV15ClusterController": {
                "_factorized_relation_embeddings",
                "begin_v15_relation_capture",
                "end_v15_relation_capture",
            },
            "SymmetricV15PlasticityRouter": {"forward"},
            "_FunctionalEnsembleObjective": {"forward"},
        }
        selected: list[tuple[str, ast.AST]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in protected_functions:
                    selected.append((node.name, node))
            elif isinstance(node, ast.ClassDef) and node.name in protected_methods:
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name in protected_methods[node.name]
                    ):
                        selected.append((f"{node.name}.{child.name}", child))
        expected = protected_functions | {
            f"{class_name}.{method_name}"
            for class_name, method_names in protected_methods.items()
            for method_name in method_names
        }
        self.assertEqual({name for name, _ in selected}, expected)
        forbidden = {
            "mechanism_commitment",
            "motif",
            "task_id",
            "package_id",
            "stream_id",
            "cell_id",
            "topology_seed",
            "surface_seed",
            "judge_software_pipeline_attempt",
            "make_software_pipeline_stream",
            "search_teacher_plan",
            "bidirectionaloperatorplanner",
            "topk",
            "vote",
            "replay",
            "solver",
        }
        for label, node in selected:
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
                msg=label,
            )


if __name__ == "__main__":
    unittest.main()
