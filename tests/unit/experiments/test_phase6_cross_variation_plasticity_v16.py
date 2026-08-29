from __future__ import annotations

import ast
import hashlib
import inspect
import itertools
import os
from pathlib import Path
import tempfile
import unittest


# This suite is deliberately CPU-only even on a CUDA-capable workstation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.runners import phase6_cross_variation_plasticity as v15
from experiments.runners import phase6_cross_variation_plasticity_v16 as v16


class Phase6CrossVariationPlasticityV16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_v15_runner_and_test_are_byte_frozen(self) -> None:
        root = Path(__file__).resolve().parents[3]
        expected = {
            "experiments/runners/phase6_cross_variation_plasticity.py": (
                "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
            ),
            "tests/unit/experiments/test_phase6_cross_variation_plasticity.py": (
                "D2560CC62D5C2031A35BE1CF951E14167CBE789AA8BACF03C86535622C40AA4E"
            ),
        }
        observed = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest().upper()
            for name in expected
        }
        self.assertEqual(observed, expected)
        self.assertEqual(v16._frozen_v15_hashes(), expected)

    def test_plan_is_exact_fresh_seed_transform_with_v15_orders(self) -> None:
        source = v15.cross_variation_fit_plan()
        plan = v16.cross_variation_fit_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-anonymous-cross-variation-plasticity.paired.v16",
        )
        self.assertEqual(
            tuple(
                (
                    record["shared_controller_seed"],
                    record["cell_seed"],
                    record["composer_seed"],
                    record["router_seed"],
                )
                for record in plan["replicates"]
            ),
            (
                (2026083601, 2026083602, 2026083603, 2026083604),
                (2026083611, 2026083612, 2026083613, 2026083614),
                (2026083621, 2026083622, 2026083623, 2026083624),
            ),
        )
        for key in (
            "commitment_schedule",
            "commitment_schedule_payload_bytes",
            "commitment_schedule_sha256",
            "adaptation_schedule",
            "adaptation_schedule_payload_bytes",
            "adaptation_schedule_sha256",
            "updates_per_arm_per_replicate",
            "streams_per_update",
            "streams_per_lane",
            "rows_per_stream",
            "arms",
            "allocation",
            "meta_objective",
            "aggregate_usage_kl_weight",
            "early_stopping",
            "adaptive_rerun",
            "deterministic_solver",
            "stored_examples_or_replay",
        ):
            self.assertEqual(plan[key], source[key], key)
        for before_rep, after_rep in zip(
            source["replicates"], plan["replicates"], strict=True
        ):
            self.assertEqual(after_rep["arm_order"], before_rep["arm_order"])
            for update_key in ("train_updates", "adaptation_updates"):
                for before, after in zip(
                    before_rep[update_key], after_rep[update_key], strict=True
                ):
                    self.assertEqual(
                        after["procedure_indices"], before["procedure_indices"]
                    )
                    for order_key in ("lane_a_order", "lane_b_order", "real_order"):
                        self.assertEqual(after[order_key], before[order_key])
                    for pair_key in ("lane_a_seed_pairs", "lane_b_seed_pairs"):
                        self.assertEqual(
                            after[pair_key],
                            tuple(
                                (left + 1_000_000_000, right + 1_000_000_000)
                                for left, right in before[pair_key]
                            ),
                        )
            for pair_key in (
                "panel_a_seed_pairs",
                "panel_a_rerender_seed_pairs",
                "panel_b_seed_pairs",
                "probe_seed_pairs",
            ):
                self.assertEqual(
                    after_rep[pair_key],
                    tuple(
                        (left + 1_000_000_000, right + 1_000_000_000)
                        for left, right in before_rep[pair_key]
                    ),
                )
            expected_binding = v16._seed_binding_digest(after_rep["train_updates"])
            self.assertEqual(
                after_rep["uniform_stream_binding_digest"], expected_binding
            )
            self.assertEqual(
                after_rep["learned_stream_binding_digest"], expected_binding
            )
        source_pairs = v16._plan_seed_pairs(source)
        transformed_pairs = v16._plan_seed_pairs(plan)
        self.assertFalse(source_pairs & transformed_pairs)
        self.assertEqual(
            transformed_pairs,
            {
                (left + 1_000_000_000, right + 1_000_000_000)
                for left, right in source_pairs
            },
        )
        for version, prior_pairs in v16._prior_v12_v14_seed_pairs().items():
            self.assertFalse(
                transformed_pairs & prior_pairs,
                f"V16 transformed identities overlap {version}",
            )
        first = v16.cross_variation_plan_digest()
        self.assertEqual(first, v16.cross_variation_plan_digest())
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")

    def test_stable_functional_adamw_is_v15_bit_exact_and_torch_close(self) -> None:
        generator = torch.Generator().manual_seed(2026083601)
        initial = torch.randn(17, generator=generator, dtype=torch.float32)
        gradients = (
            torch.randn(17, generator=generator, dtype=torch.float32),
            torch.zeros(17, dtype=torch.float32),
            None,
            torch.randn(17, generator=generator, dtype=torch.float32),
        )
        v15_parameter = (initial.clone(),)
        v16_parameter = (initial.clone(),)
        zero = torch.zeros_like(initial)
        v15_state = (v15.AdamWSlot(step=0, exp_avg=zero.clone(), exp_avg_sq=zero.clone()),)
        v16_state = (v15.AdamWSlot(step=0, exp_avg=zero.clone(), exp_avg_sq=zero.clone()),)
        torch_parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW(
            (torch_parameter,),
            lr=3.0e-4,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        for gradient in gradients:
            v15_parameter, v15_state = v15.functional_adamw_step(
                v15_parameter, (gradient,), v15_state, (3.0e-4,)
            )
            v16_parameter, v16_state = v16.functional_adamw_step(
                v16_parameter, (gradient,), v16_state, (3.0e-4,)
            )
            torch_parameter.grad = None if gradient is None else gradient.clone()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.assertTrue(torch.equal(v16_parameter[0], v15_parameter[0]))
            self.assertEqual(v16_state[0].step, v15_state[0].step)
            self.assertTrue(torch.equal(v16_state[0].exp_avg, v15_state[0].exp_avg))
            self.assertTrue(
                torch.equal(v16_state[0].exp_avg_sq, v15_state[0].exp_avg_sq)
            )
            self.assertTrue(
                torch.allclose(
                    v16_parameter[0],
                    torch_parameter.detach(),
                    atol=1.0e-7,
                    rtol=1.0e-6,
                )
            )
            if gradient is not None:
                torch_state = optimizer.state[torch_parameter]
                self.assertTrue(
                    torch.allclose(
                        v16_state[0].exp_avg,
                        torch_state["exp_avg"],
                        atol=1.0e-7,
                        rtol=1.0e-6,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        v16_state[0].exp_avg_sq,
                        torch_state["exp_avg_sq"],
                        atol=1.0e-7,
                        rtol=1.0e-6,
                    )
                )

    def test_exact_zero_direction_has_finite_functional_vjp(self) -> None:
        parameter = torch.linspace(-0.3, 0.3, 19, dtype=torch.float32)
        direction = torch.zeros_like(parameter, requires_grad=True)
        slot = v15.AdamWSlot(
            step=0,
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
        )
        updated, state = v16.functional_adamw_step(
            (parameter,), (direction,), (slot,), (3.0e-4,)
        )
        probe = torch.linspace(0.5, 1.5, 19, dtype=torch.float32)
        (vjp,) = torch.autograd.grad((updated[0] * probe).sum(), (direction,))
        self.assertTrue(torch.equal(updated[0], parameter))
        self.assertEqual(state[0].step, 1)
        self.assertEqual(torch.count_nonzero(state[0].exp_avg).item(), 0)
        self.assertEqual(torch.count_nonzero(state[0].exp_avg_sq).item(), 0)
        self.assertTrue(torch.isfinite(vjp).all().item())

    def test_pair_starts_are_exact_and_use_fresh_model_seeds(self) -> None:
        for replicate in range(3):
            uniform, learned = v16.build_cross_variation_pair(replicate)
            self.assertEqual(
                v16.cross_variation_arm_digest(uniform),
                v16.cross_variation_arm_digest(learned),
            )
            self.assertEqual(
                v16._component_digests(uniform),
                v16._component_digests(learned),
            )
        with self.assertRaises(ValueError):
            v16.build_cross_variation_pair(-1)
        with self.assertRaises(ValueError):
            v16.build_cross_variation_pair(True)

    def test_actual_and_virtual_cell_adamw_are_exact_twins(self) -> None:
        arm, _ = v16.build_cross_variation_pair(0)
        batch = v16.build_training_batches(0, updates=1)[0]
        evidence = v16.collect_cross_variation_evidence(
            arm.controller, batch.streams, arm.cell_optimizer_state
        )
        allocations, _, _ = v15._combined_allocations(
            evidence, batch, arm.router, learned_plasticity=True
        )
        directions, _, _ = v15._routed_directions(
            evidence, allocations, tuple(range(8))
        )
        virtual = v16._virtual_adamw_parameters(
            arm.controller, directions, arm.cell_optimizer_state
        )
        expected_states = []
        with torch.no_grad():
            for group, cell_direction, state in zip(
                v15._cell_parameter_groups(arm.controller),
                directions,
                arm.cell_optimizer_state,
                strict=True,
            ):
                _, next_state = v16.functional_adamw_step(
                    tuple(parameter.detach() for _, parameter in group),
                    tuple(value.detach() for value in cell_direction),
                    state,
                    tuple(v15._parameter_learning_rate(name) for name, _ in group),
                )
                expected_states.append(next_state)
        v16._apply_cell_adamw_update(arm, directions)
        for group, values in zip(
            v15._cell_parameter_groups(arm.controller), virtual, strict=True
        ):
            for (_, parameter), value in zip(group, values, strict=True):
                self.assertTrue(torch.equal(parameter.detach(), value.detach()))
        for actual_cell, expected_cell in zip(
            arm.cell_optimizer_state, expected_states, strict=True
        ):
            for actual, expected in zip(actual_cell, expected_cell, strict=True):
                self.assertEqual(actual.step, expected.step)
                self.assertTrue(torch.equal(actual.exp_avg, expected.exp_avg))
                self.assertTrue(torch.equal(actual.exp_avg_sq, expected.exp_avg_sq))

    def test_structural_preflight_covers_all_fresh_twins_without_update(self) -> None:
        report = v16.structural_preflight("cpu")
        self.assertEqual(report["protocol_id"], v16._PROTOCOL_ID)
        self.assertEqual(report["replicate_count"], 3)
        self.assertEqual(report["arms_checked"], 6)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertIs(report["evaluation_performed"], False)
        self.assertIs(report["classification_performed"], False)
        self.assertIs(report["all_routes_exact_uniform"], True)
        self.assertIs(report["all_upstream_meta_gradients_exact_zero"], True)
        for replicate in report["replicates"]:
            self.assertIs(replicate["twins_exact"], True)
            self.assertEqual(len(replicate["arms"]), 2)
            for arm in replicate["arms"]:
                self.assertEqual(arm["before_digests"], arm["after_digests"])
                self.assertIs(arm["routes"]["combined_exact_uniform"], True)
                self.assertIs(arm["routes"]["lane_a_exact_uniform"], True)
                self.assertIs(arm["routes"]["lane_b_exact_uniform"], True)
                self.assertIs(
                    arm["meta_gradient_diagnostics"]["upstream_exact_zero"],
                    True,
                )
                self.assertEqual(arm["optimizer_steps"], 0)
                for collection in (
                    arm["controller_parameters"],
                    arm["router_parameters"],
                    arm["routed_direction_parameters"],
                    arm["meta_gradient_parameters"],
                    arm["composer_gradient_parameters"],
                ):
                    self.assertTrue(collection)
                    self.assertTrue(all(value["finite"] is True for value in collection))

    def test_nonfinite_guard_rejects_before_optimizer_ownership(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            v16._require_finite_named(
                "test",
                (("broken", torch.tensor([float("nan")], dtype=torch.float32)),),
            )
        parameter = torch.zeros(1)
        slot = v15.AdamWSlot(
            step=0, exp_avg=torch.zeros(1), exp_avg_sq=torch.zeros(1)
        )
        with self.assertRaisesRegex(ValueError, "gradient is invalid"):
            v16.functional_adamw_step(
                (parameter,),
                (torch.tensor([float("inf")]),),
                (slot,),
                (1.0e-3,),
            )

    def test_checkpoint_v2_round_trip_and_plan_tamper_rejection(self) -> None:
        systems = tuple(v16.build_cross_variation_pair(index) for index in range(3))
        expected = tuple(
            tuple(v16.cross_variation_arm_digest(arm) for arm in pair)
            for pair in systems
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v16.pt"
            v16.save_cross_variation_checkpoint(path, systems)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["version"], "angler.phase6-cross-variation-plasticity.v2")
            self.assertEqual(payload["digest_version"], "v2")
            restored = v16.load_cross_variation_checkpoint(path)
            observed = tuple(
                tuple(v16.cross_variation_arm_digest(arm) for arm in pair)
                for pair in restored
            )
            self.assertEqual(observed, expected)
            payload["plan_digest"] = "sha256:" + "0" * 64
            tampered = Path(directory) / "tampered.pt"
            torch.save(payload, tampered)
            with self.assertRaisesRegex(RuntimeError, "identity or seed plan"):
                v16.load_cross_variation_checkpoint(tampered)

    def test_v16_never_monkeypatches_or_calls_broken_v15_update_paths(self) -> None:
        source_path = Path(v16.__file__).resolve()
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_attributes = {
            "functional_adamw_step",
            "collect_cross_variation_evidence",
            "_virtual_adamw_parameters",
            "cross_variation_meta_gradients",
            "_apply_cell_adamw_update",
            "fit_cross_variation_batches",
            "_adaptation_diagnostic",
            "fit_cross_variation_pilot",
            "save_cross_variation_checkpoint",
            "load_cross_variation_checkpoint",
        }
        forbidden_calls = []
        forbidden_assignments = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "v15"
                and node.func.attr in forbidden_attributes
            ):
                forbidden_calls.append(node.func.attr)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "v15"
                    ):
                        forbidden_assignments.append(target.attr)
        self.assertEqual(forbidden_calls, [])
        self.assertEqual(forbidden_assignments, [])
        self.assertNotIn("monkeypatch", source.lower())
        self.assertNotIn("setattr(v15", source.replace(" ", ""))
        for function in (
            v16.collect_cross_variation_evidence,
            v16._virtual_adamw_parameters,
            v16._apply_cell_adamw_update,
        ):
            self.assertIn("functional_adamw_step(", inspect.getsource(function))
        self.assertIn(
            "cross_variation_meta_gradients(",
            inspect.getsource(v16.fit_cross_variation_batches),
        )

    def test_progress_callback_is_arm_boundary_only_and_metric_free(self) -> None:
        events = []
        v16._emit_arm_progress(
            events.append,
            replicate=0,
            arm="uniform_adamw_plasticity",
            completed_arms=1,
            started=0.0,
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event"], "ARM_BOUNDARY_COMPLETE")
        self.assertIs(event["adaptive_metric_included"], False)
        self.assertEqual(event["optimizer_steps"], 80)
        self.assertEqual(event["streams"], 640)
        self.assertEqual(event["rows"], 2560)
        self.assertFalse(
            {
                "loss",
                "score",
                "supported_rows",
                "qualifying_streams",
                "route_movement",
            }
            & set(event)
        )
        signature = inspect.signature(v16.fit_cross_variation_pilot)
        self.assertIn("progress_callback", signature.parameters)

    def test_classification_is_exactly_v15_for_every_boolean_input(self) -> None:
        names = (
            "integrity_passed",
            "uniform_competent",
            "uniform_materially_better",
            "learned_competent",
            "every_support_rule_passed",
        )
        for values in itertools.product((False, True), repeat=len(names)):
            inputs = dict(zip(names, values, strict=True))
            self.assertEqual(
                v16.classify_cross_variation_result(**inputs),
                v15.classify_cross_variation_result(**inputs),
            )


if __name__ == "__main__":
    unittest.main()
