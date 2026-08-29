from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


# V17's pre-run suite is deliberately CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn

from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_cross_variation_plasticity as v15
from experiments.runners import phase6_cross_variation_plasticity_v16 as v16
from experiments.runners import phase6_v12_champion_context_residual as v17


_SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v12-conflict.pt"
)


def _load_system() -> v17.V12ChampionContextResidualSystem:
    return v17.load_v12_champion_context_residual_source(_SOURCE_CHECKPOINT)


def _panel_streams():
    plan = v17.v12_champion_context_residual_plan()
    return v12._relation_credit_panel_streams(
        plan["commitments"],
        plan["panel_seed_pairs"],
    )


def _first_panel_stream():
    return _panel_streams()[0]


class Phase6V12ChampionContextResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)
        if not _SOURCE_CHECKPOINT.is_file():
            raise RuntimeError("the frozen terminal V12 checkpoint is required")

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_frozen_sources_and_plan_bind_exact_v17_identity(self) -> None:
        root = Path(__file__).resolve().parents[3]
        expected_evidence = {
            "experiments/runners/phase6_software_pipeline_reconstruction.py": (
                "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
            ),
            "tests/unit/experiments/test_phase6_software_pipeline_reconstruction.py": (
                "2E6D844D24DB0A9326D84A19AEC56ED5BF6288B94C67AD5926AC05933FB6DF32"
            ),
            "experiments/runners/phase6_cross_variation_plasticity.py": (
                "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
            ),
            "tests/unit/experiments/test_phase6_cross_variation_plasticity.py": (
                "D2560CC62D5C2031A35BE1CF951E14167CBE789AA8BACF03C86535622C40AA4E"
            ),
        }
        observed = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest().upper()
            for name in expected_evidence
        }
        self.assertEqual(observed, expected_evidence)
        self.assertEqual(
            v17.frozen_dependency_hashes(),
            {
                name: expected_evidence[name]
                for name in v17.FROZEN_DEPENDENCY_HASHES
            },
        )
        plan = v17.v12_champion_context_residual_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-v12-champion-context-residual.v17",
        )
        self.assertEqual(plan["expert_seeds"], tuple(range(2026083701, 2026083705)))
        self.assertEqual(plan["composer_seed"], 2026083705)
        self.assertEqual((plan["expert_count"], plan["expert_rank"]), (4, 8))
        self.assertEqual((plan["context_updates"], plan["streams_per_update"]), (25, 8))
        self.assertEqual(len(plan["training_seed_batches"]), 25)
        self.assertTrue(all(len(batch) == 8 for batch in plan["training_seed_batches"]))
        self.assertEqual(
            plan["training_seed_batches"],
            tuple(
                tuple(
                    (
                        8_001_000_001 + 100_000 * update + 1_000 * commitment,
                        8_041_000_001 + 100_000 * update + 1_000 * commitment,
                    )
                    for commitment in range(8)
                )
                for update in range(25)
            ),
        )
        self.assertEqual(
            plan["panel_seed_pairs"],
            tuple(
                (8_081_000_001 + 1_000 * index, 8_121_000_001 + 1_000 * index)
                for index in range(8)
            ),
        )
        self.assertEqual(
            plan["rerender_seed_pairs"],
            tuple(
                (8_081_000_001 + 1_000 * index, 8_161_000_001 + 1_000 * index)
                for index in range(8)
            ),
        )
        self.assertEqual(
            (
                plan["expert_learning_rate"],
                plan["composer_learning_rate"],
                plan["weight_decay"],
                plan["gradient_clip"],
            ),
            (3.0e-4, 1.0e-3, 0.0, 5.0),
        )
        self.assertEqual(plan["context_gate"], {
            "supported_top_one": 0.80,
            "supported_valid_set_mass": 0.60,
        })
        self.assertEqual(plan["causal_support"], {
            "residual_top_one_gain": 3,
            "residual_mass_gain": 0.05,
            "composer_top_one_gain": 2,
            "composer_mass_gain": 0.02,
            "lesion_top_one_loss": 1,
            "lesion_mass_loss": 0.01,
            "required_distinct_lesions": 2,
        })
        all_pairs = tuple(
            pair for batch in plan["training_seed_batches"] for pair in batch
        ) + plan["panel_seed_pairs"] + plan["rerender_seed_pairs"]
        self.assertEqual(len(all_pairs), len(set(all_pairs)))
        self.assertTrue(all(seed >= 8_000_000_001 for pair in all_pairs for seed in pair))
        prior_pairs = v16._prior_v12_v14_seed_pairs()
        prior_pairs["v15"] = v16._plan_seed_pairs(v15.cross_variation_fit_plan())
        prior_pairs["v16"] = v16._plan_seed_pairs(v16.cross_variation_fit_plan())
        v17_pairs = set(all_pairs)
        self.assertTrue(all(v17_pairs.isdisjoint(pairs) for pairs in prior_pairs.values()))
        self.assertEqual(len(v17.MUTABLE_PARAMETER_NAMES), 17)

    def test_strict_terminal_v12_migration_is_rng_and_byte_exact(self) -> None:
        torch.manual_seed(45_901)
        before_rng = torch.get_rng_state().clone()
        system = _load_system()
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(system.source, v17._expected_source_binding())
        self.assertEqual(
            v17.inherited_v12_controller_digest(system.controller),
            v17.V12_CONTROLLER_DIGEST,
        )
        self.assertEqual(system.context_updates, 0)
        self.assertIsNone(system.optimizer_state)
        report = v17.context_residual_parameter_report(system.controller, system.mixer)
        self.assertEqual(report["new_trainable_tensors"], 17)
        self.assertEqual(report["new_trainable_parameters"], 10_007)
        self.assertEqual(report["controller_parameters"], 275_613)
        self.assertEqual(report["complete_learned_system_parameters"], 279_017)
        self.assertEqual(
            tuple(name for name, parameter in system.controller.named_parameters() if parameter.requires_grad),
            v17.MUTABLE_PARAMETER_NAMES,
        )
        self.assertFalse(any(parameter.requires_grad for parameter in system.mixer.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "not-v12.pt"
            wrong.write_bytes(b"not the terminal v12 checkpoint")
            with self.assertRaises(RuntimeError):
                v17.load_v12_champion_context_residual_source(wrong)

    def test_step_zero_is_bit_exact_for_context_relation_and_final_scores(self) -> None:
        system = _load_system()
        base, _, _ = v12.load_public_relation_conflict_checkpoint(_SOURCE_CHECKPOINT)
        torch.manual_seed(45_902)
        query_context = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
        stored_context = torch.nn.functional.normalize(torch.randn(7, 32), dim=-1)
        query_relation = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
        stored_relation = torch.nn.functional.normalize(torch.randn(7, 32), dim=-1)
        expected_logits = base._context_pair_logits(query_context, stored_context)
        actual_logits, weights, cell_logits = system.controller._context_composed_read(
            query_context,
            stored_context,
        )
        self.assertTrue(torch.equal(actual_logits, expected_logits))
        self.assertTrue(torch.equal(weights, torch.full_like(weights, 0.25)))
        for index in range(4):
            self.assertTrue(torch.equal(cell_logits[..., index], expected_logits))
        expected_read = base._relation_evidence_read(
            query_context,
            query_relation,
            stored_context,
            stored_relation,
        )
        actual_read = system.controller._relation_evidence_read(
            query_context,
            query_relation,
            stored_context,
            stored_relation,
        )
        for expected, actual in zip(expected_read, actual_read, strict=True):
            self.assertTrue(torch.equal(expected, actual))
        expected_rows = v12.public_relation_credit_rows(base, _first_panel_stream())
        actual_rows = v12.public_relation_credit_rows(
            system.controller,
            _first_panel_stream(),
        )
        for expected, actual in zip(expected_rows, actual_rows, strict=True):
            for field in (
                "positive_margin",
                "negative_margin",
                "slot_losses",
                "slot_positive_margins",
                "slot_negative_margins",
                "responsibilities",
                "context_weights",
                "context_null_weight",
            ):
                self.assertTrue(
                    torch.equal(getattr(expected, field), getattr(actual, field)),
                    field,
                )
        stream = _first_panel_stream()
        with torch.no_grad():
            expected_state = base.initial_state()
            actual_state = system.controller.initial_state()
            for pair in stream.supports:
                expected_state = v12.acquire_public_pipeline_traces(
                    base,
                    pair.learner,
                    expected_state,
                ).state
                actual_state = v12.acquire_public_pipeline_traces(
                    system.controller,
                    pair.learner,
                    actual_state,
                ).state
        expected_snapshot = v12.snapshot_software_reconstruction_state(expected_state)
        actual_snapshot = v12.snapshot_software_reconstruction_state(actual_state)
        self.assertEqual(set(expected_snapshot), set(actual_snapshot))
        for name in expected_snapshot:
            self.assertTrue(
                torch.equal(expected_snapshot[name], actual_snapshot[name]),
                name,
            )
        query = stream.queries[0].learner
        with torch.no_grad():
            expected_scores = base.score_actions(query, expected_state)
            actual_scores = system.controller.score_actions(query, actual_state)
        self.assertEqual(
            tuple(expected_scores.__dataclass_fields__),
            tuple(actual_scores.__dataclass_fields__),
        )
        for name in expected_scores.__dataclass_fields__:
            self.assertTrue(
                torch.equal(getattr(expected_scores, name), getattr(actual_scores, name)),
                name,
            )

    def test_zero_up_opens_first_then_experts_diverge_and_composer_opens(self) -> None:
        controller = _load_system().controller
        torch.manual_seed(45_903)
        query = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
        stored = torch.nn.functional.normalize(torch.randn(6, 32), dim=-1)
        coefficient = torch.linspace(-0.7, 0.9, 18).reshape(3, 6)
        optimizer = v17._context_optimizer(controller)
        loss = (controller._context_pair_logits(query, stored) * coefficient).sum()
        loss.backward()
        named = dict(controller.named_parameters())
        for index in range(4):
            up = named[f"context_residual_experts.{index}.up.weight"]
            down = named[f"context_residual_experts.{index}.down.weight"]
            self.assertIsNotNone(up.grad)
            self.assertTrue(bool(torch.isfinite(up.grad).all().item()))
            self.assertGreater(int(torch.count_nonzero(up.grad).item()), 0)
            self.assertTrue(down.grad is None or int(torch.count_nonzero(down.grad).item()) == 0)
        for name in v17._COMPOSER_PARAMETER_NAMES:
            gradient = named[name].grad
            self.assertTrue(gradient is None or int(torch.count_nonzero(gradient).item()) == 0)
        optimizer.step()
        up_states = tuple(
            named[f"context_residual_experts.{index}.up.weight"].detach()
            for index in range(4)
        )
        self.assertTrue(any(
            not torch.equal(up_states[left], up_states[right])
            for left in range(4)
            for right in range(left + 1, 4)
        ))
        optimizer.zero_grad(set_to_none=True)
        query_2 = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
        stored_2 = torch.nn.functional.normalize(torch.randn(5, 32), dim=-1)
        controller._context_pair_logits(query_2, stored_2).square().mean().backward()
        self.assertTrue(any(
            named[name].grad is not None
            and bool(torch.isfinite(named[name].grad).all().item())
            and int(torch.count_nonzero(named[name].grad).item()) > 0
            for name in v17._COMPOSER_PARAMETER_NAMES
        ))

    def test_context_read_is_cell_query_and_slot_permutation_symmetric(self) -> None:
        controller = _load_system().controller
        with torch.no_grad():
            for index, expert in enumerate(controller.context_residual_experts):
                expert.up.weight.copy_(
                    torch.linspace(
                        -0.002 * (index + 1),
                        0.002 * (index + 1),
                        expert.up.weight.numel(),
                    ).reshape_as(expert.up.weight)
                )
            final = controller.context_composer.residual_scorer[-1].weight
            final.copy_(torch.linspace(-0.02, 0.02, final.numel()).reshape_as(final))
        clone = v17.V12ChampionContextResidualController(controller.profile)
        clone.load_state_dict(controller.state_dict(), strict=True)
        order = (2, 0, 3, 1)
        clone.context_residual_experts = nn.ModuleList(
            tuple(clone.context_residual_experts[index] for index in order)
        )
        torch.manual_seed(45_904)
        query = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
        stored = torch.nn.functional.normalize(torch.randn(5, 32), dim=-1)
        fused, weights, logits = controller._context_composed_read(query, stored)
        twin_fused, twin_weights, twin_logits = clone._context_composed_read(query, stored)
        torch.testing.assert_close(twin_fused, fused, atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(twin_weights, weights[..., list(order)], atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(twin_logits, logits[..., list(order)], atol=1.0e-6, rtol=0.0)
        transpose = controller._context_pair_logits(stored, query)
        torch.testing.assert_close(transpose, fused.transpose(0, 1), atol=1.0e-6, rtol=0.0)
        slot_order = (4, 1, 3, 0, 2)
        permuted = controller._context_pair_logits(stored[list(slot_order)], query)
        torch.testing.assert_close(
            permuted,
            fused[:, list(slot_order)].transpose(0, 1),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_cell_permutation_carries_name_keyed_adamw_state_equivariantly(self) -> None:
        system_a = _load_system()
        controller_a = system_a.controller
        with torch.no_grad():
            for index, expert in enumerate(controller_a.context_residual_experts):
                expert.up.weight.copy_(
                    torch.linspace(
                        -0.001 * (index + 1),
                        0.001 * (index + 1),
                        expert.up.weight.numel(),
                    ).reshape_as(expert.up.weight)
                )
            scorer = controller_a.context_composer.residual_scorer[-1].weight
            scorer.copy_(torch.linspace(-0.01, 0.01, scorer.numel()).reshape_as(scorer))
        controller_b = v17.V12ChampionContextResidualController(controller_a.profile)
        controller_b.load_state_dict(controller_a.state_dict(), strict=True)
        order = (2, 0, 3, 1)
        inverse = tuple(order.index(index) for index in range(4))
        controller_b.context_residual_experts = nn.ModuleList(
            tuple(controller_b.context_residual_experts[index] for index in order)
        )

        optimizer_a = v17._context_optimizer(controller_a)
        named_a = dict(controller_a.named_parameters())
        for slot_index, name in enumerate(v17.MUTABLE_PARAMETER_NAMES):
            parameter = named_a[name]
            optimizer_a.state[parameter] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.full_like(parameter, (slot_index + 1) * 1.0e-5),
                "exp_avg_sq": torch.full_like(parameter, (slot_index + 1) * 1.0e-8),
            }
        state_a = v17._canonical_optimizer_state(optimizer_a, controller_a)
        state_b = copy.deepcopy(state_a)
        for new_index, old_index in enumerate(order):
            for projection in ("down", "up"):
                new_name = (
                    f"context_residual_experts.{new_index}.{projection}.weight"
                )
                old_name = (
                    f"context_residual_experts.{old_index}.{projection}.weight"
                )
                state_b["state"][new_name] = copy.deepcopy(
                    state_a["state"][old_name]
                )
        system_a.context_updates = 3
        system_a.optimizer_state = state_a
        system_b = v17.V12ChampionContextResidualSystem(
            controller=controller_b,
            mixer=system_a.mixer,
            competence_state=system_a.competence_state,
            source=system_a.source,
            context_updates=3,
            optimizer_state=state_b,
        )
        optimizer_a = v17.restore_context_residual_optimizer(system_a)
        optimizer_b = v17.restore_context_residual_optimizer(system_b)
        torch.manual_seed(45_906)
        query = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
        stored = torch.nn.functional.normalize(torch.randn(6, 32), dim=-1)
        coefficient = torch.linspace(-0.8, 0.9, 24).reshape(4, 6)
        left = controller_a._context_pair_logits(query, stored)
        right = controller_b._context_pair_logits(query, stored)
        torch.testing.assert_close(right, left, atol=1.0e-6, rtol=0.0)
        (left * coefficient).sum().backward()
        (right * coefficient).sum().backward()
        optimizer_a.step()
        optimizer_b.step()

        for index in range(4):
            left_expert = controller_a.context_residual_experts[index]
            right_expert = controller_b.context_residual_experts[inverse[index]]
            for projection in ("down", "up"):
                left_parameter = getattr(left_expert, projection).weight
                right_parameter = getattr(right_expert, projection).weight
                torch.testing.assert_close(
                    right_parameter,
                    left_parameter,
                    atol=1.0e-6,
                    rtol=0.0,
                )
                for slot_name in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        optimizer_b.state[right_parameter][slot_name],
                        optimizer_a.state[left_parameter][slot_name],
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                self.assertTrue(torch.equal(
                    optimizer_b.state[right_parameter]["step"],
                    optimizer_a.state[left_parameter]["step"],
                ))
        for name in v17._COMPOSER_PARAMETER_NAMES:
            left_parameter = dict(controller_a.named_parameters())[name]
            right_parameter = dict(controller_b.named_parameters())[name]
            torch.testing.assert_close(
                right_parameter,
                left_parameter,
                atol=1.0e-6,
                rtol=0.0,
            )
            for slot_name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer_b.state[right_parameter][slot_name],
                    optimizer_a.state[left_parameter][slot_name],
                    atol=1.0e-6,
                    rtol=0.0,
                )

    def test_three_real_context_updates_open_credit_mutate_allowlist_and_persist_moments(self) -> None:
        system = _load_system()
        plan = v17.v12_champion_context_residual_plan()
        batches = v12._relation_credit_stream_batches(
            plan["commitments"],
            plan["training_seed_batches"][:3],
        )
        inherited_before = {
            name: value.detach().clone()
            for name, value in system.controller.state_dict().items()
            if name not in v17.MUTABLE_PARAMETER_NAMES
        }

        report = v17._fit_context_residual_batches(system, batches)
        self.assertEqual(system.context_updates, 3)
        self.assertTrue(all(report["first_zero_up_gradients_nonzero"].values()))
        self.assertTrue(report["first_post_divergence_composer_credit"])
        self.assertTrue(report["experts_diverged"])
        self.assertEqual(report["trainable_parameter_names"], v17.MUTABLE_PARAMETER_NAMES)
        changed = set(report["changed_parameter_names"])
        self.assertTrue(changed)
        self.assertTrue(changed <= set(v17.MUTABLE_PARAMETER_NAMES))
        self.assertTrue(set(v17._EXPERT_PARAMETER_NAMES) <= changed)
        self.assertIn("context_composer.residual_scorer.2.weight", changed)
        self.assertEqual(
            set(report["gradient_reached_parameter_names"]),
            set(v17.MUTABLE_PARAMETER_NAMES),
        )
        self.assertEqual(
            set(report["unchanged_allowed_parameter_names"]),
            set(v17.MUTABLE_PARAMETER_NAMES) - changed,
        )
        self.assertTrue(
            set(v17._COMPOSER_PARAMETER_NAMES)
            & set(report["nonzero_gradient_parameter_names"][1])
        )
        for name, before in inherited_before.items():
            self.assertTrue(torch.equal(before, system.controller.state_dict()[name]), name)
        self.assertIsNotNone(system.optimizer_state)
        v17._validate_optimizer_state(
            system.optimizer_state,
            system.controller,
            expected_steps=3,
        )
        restored = v17.restore_context_residual_optimizer(system)
        self.assertEqual(len(restored.state), 17)
        with self.assertRaises(RuntimeError):
            v17._fit_context_residual_batches(system, batches)

    def test_classification_thresholds_are_conjunctive_and_integrity_fails_closed(self) -> None:
        learned = {"top_one_successes": 20, "valid_set_mass": 0.75}
        residual_off = {"top_one_successes": 17, "valid_set_mass": 0.69}
        uniform = {"top_one_successes": 18, "valid_set_mass": 0.72}
        lesions = (
            {"top_one_successes": 19, "valid_set_mass": 0.75},
            {"top_one_successes": 20, "valid_set_mass": 0.73},
            {"top_one_successes": 20, "valid_set_mass": 0.75},
            {"top_one_successes": 20, "valid_set_mass": 0.75},
        )
        supported = v17._classify_context_result(
            learned,
            residual_off,
            uniform,
            lesions,
            context_gate_passed=True,
            integrity_passed=True,
        )
        self.assertEqual(supported["classification"], "CONTEXT_RESIDUAL_SUPPORTED")
        below_mass = v17._classify_context_result(
            learned,
            {"top_one_successes": 17, "valid_set_mass": 0.71},
            uniform,
            lesions,
            context_gate_passed=True,
            integrity_passed=True,
        )
        self.assertEqual(below_mass["classification"], "CONTEXT_RESIDUAL_NOT_SUPPORTED")
        invalid = v17._classify_context_result(
            learned,
            residual_off,
            uniform,
            lesions,
            context_gate_passed=True,
            integrity_passed=False,
        )
        self.assertEqual(invalid["classification"], "INVALID_NO_CLAIM")

    def test_surface_lesions_need_not_share_expert_indices(self) -> None:
        system = _load_system()
        system.context_updates = 25

        def surface(indices):
            return {
                "learned": {},
                "read_diagnostics": {"learned": {}},
                "integrity_checks": {"surface": True},
                "classification": {
                    "classification": "CONTEXT_RESIDUAL_SUPPORTED",
                    "lesion_effects": tuple(
                        {
                            "expert": index,
                            "top_one_loss": 1,
                            "mass_loss": 0.0,
                        }
                        for index in indices
                    ),
                },
            }

        cross_surface = {
            "relation_masks_and_counts_exact": True,
            "context_top_one_success_exact": True,
            "context_top_one_choice_exact": True,
            "continuous_values_within_1e-6": True,
        }
        with (
            mock.patch.object(v17, "_assert_source_lineage"),
            mock.patch.object(
                v17,
                "_evaluate_surface_suite",
                side_effect=(surface((0, 1)), surface((2, 3))),
            ),
            mock.patch.object(
                v17,
                "_cross_surface_checks",
                return_value=cross_surface,
            ),
        ):
            report = v17.evaluate_v12_champion_context_residual(system)
        self.assertEqual(report["classification"], "CONTEXT_RESIDUAL_SUPPORTED")
        self.assertFalse(report["shared_causal_lesion_index_support"])
        self.assertEqual(report["cross_surface_causal_lesion_indices"], ())

    def test_cross_surface_checks_compare_choices_and_full_evidence_scores(self) -> None:
        row = {
            "stream_index": 0,
            "heldout_index": 0,
            "transition_index": 0,
            "valid_slots": (0,),
            "relation_supported": True,
            "context_valid_set_top_one": True,
            "context_weights": (0.6, 0.2),
            "context_null_mass": 0.2,
            "context_valid_set_mass": 0.6,
            "positive_margin": 0.2,
            "negative_margin": -0.2,
        }
        panel = {
            "relation_supported_rows": 1,
            "streams_with_three_supported_rows": 0,
            "supported_rows_per_stream": (1,),
            "row_reports": (row,),
        }
        read = {
            "evidence_reads": ({
                "context_weights": ((0.6, 0.2),),
                "context_null_weights": (0.2,),
                "final_evidence_scores": (0.125,),
            },),
        }
        exact = v17._cross_surface_checks(panel, panel, read, read)
        self.assertTrue(exact["context_top_one_choice_exact"])
        self.assertTrue(exact["continuous_values_within_1e-6"])
        changed_read = {
            "evidence_reads": ({
                "context_weights": ((0.6, 0.2),),
                "context_null_weights": (0.2,),
                "final_evidence_scores": (0.12501,),
            },),
        }
        changed = v17._cross_surface_checks(panel, panel, read, changed_read)
        self.assertFalse(changed["continuous_values_within_1e-6"])
        wrong_choice_row = dict(row)
        wrong_choice_row["context_weights"] = (0.2, 0.6)
        wrong_choice_row["context_valid_set_top_one"] = True
        wrong_panel = {**panel, "row_reports": (wrong_choice_row,)}
        choice = v17._cross_surface_checks(panel, wrong_panel, read, read)
        self.assertFalse(choice["context_top_one_choice_exact"])

    def test_full_diagnostic_suite_is_no_update_and_rng_exact(self) -> None:
        system = _load_system()
        before = {
            "controller": v12.software_pipeline_model_digest(system.controller),
            "mixer": v12.anonymous_conflict_mixer_digest(system.mixer),
            "competence": v12.software_reconstruction_state_digest(
                system.competence_state
            ),
            "optimizer": v17.context_residual_optimizer_digest(
                system.optimizer_state
            ),
            "system": v17.context_residual_system_digest(system),
            "updates": system.context_updates,
        }
        rng_before = torch.get_rng_state().clone()
        suite = v17._evaluate_surface_suite(system.controller, _panel_streams())
        self.assertEqual(
            set(suite),
            {
                "learned",
                "residual_off",
                "forced_uniform",
                "drop_one_lesions",
                "learned_context_metrics",
                "residual_off_context_metrics",
                "uniform_context_metrics",
                "lesion_context_metrics",
                "context_gate",
                "integrity_checks",
                "classification",
                "read_diagnostics",
            },
        )
        self.assertEqual(len(suite["drop_one_lesions"]), 4)
        after = {
            "controller": v12.software_pipeline_model_digest(system.controller),
            "mixer": v12.anonymous_conflict_mixer_digest(system.mixer),
            "competence": v12.software_reconstruction_state_digest(
                system.competence_state
            ),
            "optimizer": v17.context_residual_optimizer_digest(
                system.optimizer_state
            ),
            "system": v17.context_residual_system_digest(system),
            "updates": system.context_updates,
        }
        self.assertEqual(after, before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_successor_checkpoint_roundtrip_binds_optimizer_and_rejects_tamper(self) -> None:
        system = _load_system()
        controller = system.controller
        torch.manual_seed(45_905)
        objective = controller._context_pair_logits(
            torch.nn.functional.normalize(torch.randn(3, 32), dim=-1),
            torch.nn.functional.normalize(torch.randn(4, 32), dim=-1),
        ).square().mean()
        optimizer = v17._context_optimizer(controller)
        objective.backward()
        optimizer.step()
        system.context_updates = 1
        system.optimizer_state = v17._canonical_optimizer_state(optimizer, controller)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v17.pt"
            v17.save_v12_champion_context_residual_checkpoint(checkpoint, system)
            restored = v17.load_v12_champion_context_residual_checkpoint(checkpoint)
            self.assertEqual(
                v17.context_residual_system_digest(restored),
                v17.context_residual_system_digest(system),
            )
            v17._validate_optimizer_state(
                restored.optimizer_state,
                restored.controller,
                expected_steps=1,
            )
            payload = torch.load(checkpoint, weights_only=True)
            tamper_cases = []
            model = torch.load(checkpoint, weights_only=True)
            model["model_state"][v17.MUTABLE_PARAMETER_NAMES[0]].reshape(-1)[0].add_(1)
            tamper_cases.append(model)
            mixer = torch.load(checkpoint, weights_only=True)
            first_mixer = next(iter(mixer["mixer_state"]))
            mixer["mixer_state"][first_mixer].reshape(-1)[0].add_(1)
            tamper_cases.append(mixer)
            competence = torch.load(checkpoint, weights_only=True)
            competence["competence_state"]["context_trace_keys"].reshape(-1)[0].add_(1)
            tamper_cases.append(competence)
            optimizer_tamper = torch.load(checkpoint, weights_only=True)
            first_name = v17.MUTABLE_PARAMETER_NAMES[0]
            optimizer_tamper["optimizer_state"]["state"][first_name]["step"].add_(1)
            tamper_cases.append(optimizer_tamper)
            source = torch.load(checkpoint, weights_only=True)
            source["source"]["controller_digest"] = "sha256:" + "0" * 64
            tamper_cases.append(source)
            self.assertEqual(set(payload), set(tamper_cases[0]))
            for index, tampered in enumerate(tamper_cases):
                path = Path(directory) / f"tampered-{index}.pt"
                torch.save(tampered, path)
                with self.assertRaises(RuntimeError):
                    v17.load_v12_champion_context_residual_checkpoint(path)

    def test_only_context_logits_are_overridden_and_forbidden_mechanisms_are_absent(self) -> None:
        subclass_methods = {
            name
            for name, value in vars(v17.V12ChampionContextResidualController).items()
            if inspect.isfunction(value)
        }
        base_methods = {
            name
            for name, value in vars(v12.SoftwarePipelineController).items()
            if inspect.isfunction(value)
        }
        self.assertEqual(
            (subclass_methods & base_methods) - {"__init__"},
            {"_context_pair_logits"},
        )
        source_path = Path(v17.__file__).resolve()
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertNotIn("phase6_cross_variation_plasticity_v16", source)
        self.assertNotIn("CounterfactualPlasticityRouter", source)
        self.assertNotIn("make_software_pipeline_control_stream", imports)
        self.assertNotIn("judge_software_pipeline_attempt", imports)
        self.assertNotIn("commitment_index", {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        })


if __name__ == "__main__":
    unittest.main()
