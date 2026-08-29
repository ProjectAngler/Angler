from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import math
import os
from pathlib import Path
import tempfile
import unittest


# V18 pre-run verification is deliberately CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_context_incidence as v18


_SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v12-conflict.pt"
)


def _load_system() -> v18.V12ChampionContextIncidenceSystem:
    return v18.load_v12_champion_context_incidence_source(_SOURCE_CHECKPOINT)


def _training_batches(count: int = 3):
    plan = v18.v12_champion_context_incidence_plan()
    return v12._relation_credit_stream_batches(
        plan["commitments"],
        plan["training_seed_batches"][:count],
    )


def _first_panel_stream():
    plan = v18.v12_champion_context_incidence_plan()
    return v12._relation_credit_panel_streams(
        plan["commitments"],
        plan["panel_seed_pairs"][0],
    )[0]


def _credit_row(
    logits: torch.Tensor,
    *,
    supported: bool,
) -> v12.PublicRelationCreditRow:
    probabilities = torch.softmax(logits, dim=0)
    context_weights = probabilities[:3]
    null_weight = probabilities[3]
    if supported:
        positive = torch.tensor((0.30, -0.20, -0.25), dtype=logits.dtype)
        negative = torch.tensor((-0.30, 0.20, 0.25), dtype=logits.dtype)
    else:
        positive = torch.zeros(3, dtype=logits.dtype)
        negative = torch.zeros(3, dtype=logits.dtype)
    zero = logits.sum() * 0.0
    return v12.PublicRelationCreditRow(
        heldout_index=0,
        transition_index=0,
        positive_index=0,
        negative_index=1,
        positive_margin=zero,
        negative_margin=zero,
        instance_loss=zero,
        context_loss=zero,
        separation_loss=zero,
        joint_loss=zero,
        slot_losses=torch.ones(3, dtype=logits.dtype),
        slot_positive_margins=positive,
        slot_negative_margins=negative,
        responsibilities=torch.full((3,), 1.0 / 3.0, dtype=logits.dtype),
        context_weights=context_weights,
        context_null_weight=null_weight,
    )


class Phase6V12ChampionContextIncidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)
        if not _SOURCE_CHECKPOINT.is_file():
            raise RuntimeError("the frozen terminal V12 checkpoint is required")

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_frozen_sources_leaf_and_plan_bind_exact_v18_identity(self) -> None:
        root = Path(__file__).resolve().parents[3]
        observed = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest().upper()
            for name in v18.FROZEN_DEPENDENCY_HASHES
        }
        self.assertEqual(observed, v18.FROZEN_DEPENDENCY_HASHES)
        self.assertEqual(v18.frozen_dependency_hashes(), observed)
        plan = v18.v12_champion_context_incidence_plan()
        self.assertEqual(
            plan["protocol_id"],
            "phase6.public-v12-champion-context-incidence.v18",
        )
        self.assertEqual(plan["readout_seed"], 2_026_083_801)
        self.assertEqual((plan["context_updates"], plan["streams_per_update"]), (256, 8))
        self.assertEqual(len(plan["training_seed_batches"]), 256)
        self.assertEqual(len(plan["panel_seed_pairs"]), 4)
        self.assertEqual(
            plan["training_seed_batches"][0][0],
            (9_001_000_001, 9_101_000_001),
        )
        self.assertEqual(
            plan["training_seed_batches"][255][7],
            (
                9_001_000_001 + 25_500_000 + 7_000,
                9_101_000_001 + 25_500_000 + 7_000,
            ),
        )
        self.assertEqual(
            plan["panel_seed_pairs"][3][7],
            (
                9_201_000_001 + 300_000 + 7_000,
                9_301_000_001 + 300_000 + 7_000,
            ),
        )
        train = {pair for batch in plan["training_seed_batches"] for pair in batch}
        panels = {pair for panel in plan["panel_seed_pairs"] for pair in panel}
        self.assertEqual((len(train), len(panels), len(train & panels)), (2048, 32, 0))
        self.assertEqual(
            plan["objective"],
            {
                "supported_rank": "-log(valid_real_mass/all_real_mass)",
                "supported_presence": "-log(all_real_mass)",
                "unsupported_abstain": "-log(null_mass)",
                "rank_weight": 1.0,
                "presence_weight": 0.25,
                "abstain_weight": 0.25,
                "relation_margins_detached": True,
            },
        )
        self.assertEqual(
            plan["full_advancement"],
            {
                "relation_supported_rows": 96,
                "relation_qualifying_streams": 24,
                "supported_context_top_one_fraction": 0.80,
                "supported_valid_set_mass": 0.60,
            },
        )
        self.assertEqual(
            plan["plan_digest"],
            "sha256:aa262085903f93239a61f366334f976bd925be8df6c9c32d9b9122bacac6e09d",
        )

    def test_strict_v12_migration_preserves_rng_lineage_and_parameter_accounting(self) -> None:
        rng_before = torch.get_rng_state().clone()
        system = _load_system()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertEqual(system.source, v18._expected_source_binding())
        self.assertEqual(
            v18.inherited_v12_controller_digest(system.controller),
            v18.V12_CONTROLLER_DIGEST,
        )
        self.assertEqual(
            v12.anonymous_conflict_mixer_digest(system.mixer),
            v18.V12_MIXER_DIGEST,
        )
        self.assertEqual(
            v12.software_reconstruction_state_digest(system.competence_state),
            v18.V12_COMPETENCE_DIGEST,
        )
        self.assertEqual(
            v18.context_incidence_parameter_report(system.controller, system.mixer),
            {
                "protocol_id": v18.PROTOCOL_ID,
                "inherited_v12_controller_parameters": 265_606,
                "new_trainable_tensors": 10,
                "new_trainable_parameters": 16_992,
                "controller_parameters": 282_598,
                "mixer_parameters": 3_404,
                "complete_learned_system_parameters": 286_002,
                "trainable_parameter_names": v18.MUTABLE_PARAMETER_NAMES,
            },
        )
        self.assertEqual(
            tuple(
                name
                for name, parameter in system.controller.named_parameters()
                if parameter.requires_grad
            ),
            v18.MUTABLE_PARAMETER_NAMES,
        )
        self.assertFalse(any(parameter.requires_grad for parameter in system.mixer.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "not-v12.pt"
            wrong.write_bytes(b"not the terminal v12 checkpoint")
            with self.assertRaises(RuntimeError):
                v18.load_v12_champion_context_incidence_source(wrong)

    def test_step_zero_is_bit_exact_for_pool_rows_state_and_production_scores(self) -> None:
        system = _load_system()
        base, _, _ = v12.load_public_relation_conflict_checkpoint(_SOURCE_CHECKPOINT)
        torch.manual_seed(45_918)
        pair_states = torch.randn(4, 4, 32)
        self.assertTrue(
            torch.equal(
                system.controller._pool_context_tensor(pair_states),
                base._pool_context_tensor(pair_states),
            )
        )
        stream = _first_panel_stream()
        expected_rows = v12.public_relation_credit_rows(base, stream)
        actual_rows = v12.public_relation_credit_rows(system.controller, stream)
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
                self.assertTrue(torch.equal(getattr(expected, field), getattr(actual, field)), field)
        with torch.no_grad():
            expected_state = base.initial_state()
            actual_state = system.controller.initial_state()
            for pair in stream.supports:
                expected_state = v12.acquire_public_pipeline_traces(
                    base, pair.learner, expected_state
                ).state
                actual_state = v12.acquire_public_pipeline_traces(
                    system.controller, pair.learner, actual_state
                ).state
        for name, value in v12.snapshot_software_reconstruction_state(
            expected_state
        ).items():
            self.assertTrue(
                torch.equal(
                    value,
                    v12.snapshot_software_reconstruction_state(actual_state)[name],
                ),
                name,
            )
        query = stream.queries[0].learner
        with torch.no_grad():
            expected_scores = base.score_actions(query, expected_state)
            actual_scores = system.controller.score_actions(query, actual_state)
        for name in expected_scores.__dataclass_fields__:
            self.assertTrue(
                torch.equal(getattr(expected_scores, name), getattr(actual_scores, name)),
                name,
            )

    def test_incidence_readout_is_node_rename_invariant_and_multiset_sensitive(self) -> None:
        controller = _load_system().controller
        torch.manual_seed(45_919)
        with torch.no_grad():
            controller.context_incidence_projection.weight.normal_(std=0.05)
        pair_states = torch.randn(4, 4, 32)
        order = torch.tensor((2, 0, 3, 1))
        renamed = pair_states.index_select(0, order).index_select(1, order)
        torch.testing.assert_close(
            controller._pool_context_tensor(pair_states),
            controller._pool_context_tensor(renamed),
            atol=1.0e-6,
            rtol=0.0,
        )

        # Swap two off-diagonal cells without a consistent endpoint rename.
        # The flattened multiset is unchanged, so V12's flat pool stays equal,
        # while the incidence branch can distinguish the endpoint structure.
        rearranged = pair_states.clone()
        rearranged[0, 1], rearranged[2, 3] = (
            pair_states[2, 3].clone(),
            pair_states[0, 1].clone(),
        )
        base = v18._base_controller_from_successor(controller)
        torch.testing.assert_close(
            base._pool_context_tensor(pair_states),
            base._pool_context_tensor(rearranged),
            atol=1.0e-6,
            rtol=0.0,
        )
        self.assertGreater(
            float(
                (
                    controller._pool_context_tensor(pair_states)
                    - controller._pool_context_tensor(rearranged)
                ).norm().item()
            ),
            1.0e-6,
        )

    def test_projection_zero_lesion_is_exact_v12_and_restores_live_read(self) -> None:
        controller = _load_system().controller
        base = v18._base_controller_from_successor(controller)
        torch.manual_seed(45_920)
        with torch.no_grad():
            controller.context_incidence_projection.weight.normal_(std=0.10)
        pair_states = torch.randn(5, 5, 32)
        learned = controller._pool_context_tensor(pair_states)
        with controller.projection_zero_lesion():
            lesion = controller._pool_context_tensor(pair_states)
            self.assertTrue(torch.equal(lesion, base._pool_context_tensor(pair_states)))
        restored = controller._pool_context_tensor(pair_states)
        self.assertTrue(torch.equal(learned, restored))
        self.assertFalse(torch.equal(learned, lesion))

    def test_rank_objective_blocks_null_shortcut_and_unsupported_rows_train_abstention(self) -> None:
        first_logits = torch.tensor((0.2, -0.1, -0.4, 0.3), requires_grad=True)
        shifted_logits = torch.tensor((3.2, 2.9, 2.6, 0.3), requires_grad=True)
        first_terms, _ = v18._context_incidence_row_terms(
            _credit_row(first_logits, supported=True)
        )
        shifted_terms, _ = v18._context_incidence_row_terms(
            _credit_row(shifted_logits, supported=True)
        )
        torch.testing.assert_close(first_terms["rank"], shifted_terms["rank"], atol=1.0e-6, rtol=0.0)
        self.assertLess(float(shifted_terms["presence"].item()), float(first_terms["presence"].item()))

        all_valid_logits = torch.tensor((0.2, -0.1, -0.4, 0.3), requires_grad=True)
        all_valid_row = _credit_row(all_valid_logits, supported=True)
        all_valid_row = v12.PublicRelationCreditRow(
            **{
                field: getattr(all_valid_row, field)
                for field in all_valid_row.__dataclass_fields__
                if field not in {"slot_positive_margins", "slot_negative_margins"}
            },
            slot_positive_margins=torch.full((3,), 0.30),
            slot_negative_margins=torch.full((3,), -0.30),
        )
        all_valid_terms, all_valid_diagnostics = v18._context_incidence_row_terms(
            all_valid_row
        )
        self.assertTrue(all_valid_diagnostics["supported"])
        self.assertFalse(all_valid_diagnostics["informative"])
        self.assertIsNone(all_valid_terms["rank"])
        self.assertIsNotNone(all_valid_terms["presence"])

        unsupported_logits = torch.tensor(
            (0.2, -0.1, -0.4, 0.3), requires_grad=True
        )
        unsupported_terms, diagnostics = v18._context_incidence_row_terms(
            _credit_row(unsupported_logits, supported=False)
        )
        self.assertFalse(diagnostics["supported"])
        self.assertIsNone(unsupported_terms["rank"])
        unsupported_terms["abstain"].backward()
        self.assertLess(float(unsupported_logits.grad[3].item()), 0.0)

        supported_row = _credit_row(first_logits, supported=True)
        unsupported_row = _credit_row(shifted_logits, supported=False)
        objective, report = v18._context_incidence_objective(
            ((supported_row, unsupported_row),)
        )
        expected = (
            v18._context_incidence_row_terms(supported_row)[0]["rank"]
            + 0.25 * v18._context_incidence_row_terms(supported_row)[0]["presence"]
            + 0.25 * v18._context_incidence_row_terms(unsupported_row)[0]["abstain"]
        )
        self.assertTrue(torch.equal(objective, expected))
        self.assertEqual((report["supported_rows"], report["unsupported_rows"]), (1, 1))

        detached_logits = torch.tensor((0.4, 0.1, -0.3, 0.2), requires_grad=True)
        positive_margins = torch.tensor(
            (0.30, -0.20, -0.25), requires_grad=True
        )
        negative_margins = torch.tensor(
            (-0.30, 0.20, 0.25), requires_grad=True
        )
        detached_row = replace(
            _credit_row(detached_logits, supported=True),
            slot_positive_margins=positive_margins,
            slot_negative_margins=negative_margins,
        )
        detached_terms, _ = v18._context_incidence_row_terms(detached_row)
        (detached_terms["rank"] + detached_terms["presence"]).backward()
        self.assertIsNone(positive_margins.grad)
        self.assertIsNone(negative_margins.grad)
        self.assertIsNotNone(detached_logits.grad)

    def test_panel_diagnostics_pool_informative_mass_and_log_rank_margin(self) -> None:
        rows = (
            {
                "stream_index": 0,
                "heldout_index": 0,
                "transition_index": 0,
                "relation_supported": True,
                "valid_slot_count": 1,
                "valid_slots": (0,),
                "context_weights": (0.50, 0.30, 0.10),
                "context_valid_set_top_one": True,
                "context_valid_set_mass": 0.50,
            },
            {
                "stream_index": 0,
                "heldout_index": 1,
                "transition_index": 0,
                "relation_supported": True,
                "valid_slot_count": 2,
                "valid_slots": (0, 1),
                "context_weights": (0.20, 0.25, 0.40),
                "context_valid_set_top_one": False,
                "context_valid_set_mass": 0.45,
            },
        )
        metrics = v18._panel_context_metrics(
            {
                "row_reports": rows,
                "relation_supported_rows": 2,
                "streams_with_three_supported_rows": 0,
            }
        )
        self.assertEqual((metrics["informative_rows"], metrics["unique_valid_rows"]), (2, 1))
        self.assertEqual(metrics["unique_valid_real_top_one_successes"], 1)
        self.assertAlmostEqual(metrics["informative_valid_mass_sum"], 0.95)
        self.assertAlmostEqual(metrics["informative_real_mass_sum"], 1.75)
        self.assertAlmostEqual(
            metrics["informative_real_normalized_mass"],
            0.95 / 1.75,
        )
        self.assertAlmostEqual(
            metrics["informative_rank_margin"],
            (math.log(0.50) - math.log(0.30) + math.log(0.25) - math.log(0.40))
            / 2.0,
        )

    def test_three_updates_open_projection_then_trunk_and_mutate_only_allowlist(self) -> None:
        system = _load_system()
        inherited_before = {
            name: value.detach().clone()
            for name, value in system.controller.state_dict().items()
            if name not in v18.MUTABLE_PARAMETER_NAMES
        }
        report = v18._fit_context_incidence_batches(system, _training_batches())
        self.assertEqual(system.context_updates, 3)
        self.assertTrue(report["first_projection_gradient_nonzero"])
        self.assertTrue(report["first_trunk_gradients_exact_zero"])
        self.assertTrue(report["later_trunk_gradient_reached"])
        self.assertEqual(
            set(report["nonzero_gradient_parameter_names"][0]),
            set(v18._PROJECTION_PARAMETER_NAMES),
        )
        self.assertEqual(
            set(report["gradient_reached_parameter_names"]),
            set(v18.MUTABLE_PARAMETER_NAMES),
        )
        self.assertEqual(
            set(report["changed_parameter_names"]),
            set(v18.MUTABLE_PARAMETER_NAMES),
        )
        for name, before in inherited_before.items():
            self.assertTrue(torch.equal(before, system.controller.state_dict()[name]), name)
        v18._validate_optimizer_state(
            system.optimizer_state,
            system.controller,
            expected_steps=3,
        )
        self.assertEqual(len(v18.restore_context_incidence_optimizer(system).state), 10)

    def test_checkpoint_reload_and_split_continuation_are_byte_exact_and_tamper_fails(self) -> None:
        batches = _training_batches()
        direct = _load_system()
        split = _load_system()
        v18._fit_context_incidence_batches(direct, batches)
        v18._fit_context_incidence_batches(split, batches[:2])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v18.pt"
            v18.save_v12_champion_context_incidence_checkpoint(checkpoint, split)
            resumed = v18.load_v12_champion_context_incidence_checkpoint(checkpoint)
            v18._fit_context_incidence_batches(resumed, batches[2:])
            self.assertEqual(
                v18.context_incidence_system_digest(resumed),
                v18.context_incidence_system_digest(direct),
            )
            for name, value in resumed.controller.state_dict().items():
                self.assertTrue(torch.equal(value, direct.controller.state_dict()[name]), name)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            tampered_model = copy.deepcopy(payload)
            tampered_model["model_state"]["context_incidence_projection.weight"].reshape(-1)[0].add_(1.0)
            tampered_path = Path(directory) / "tampered-model.pt"
            torch.save(tampered_model, tampered_path)
            with self.assertRaises(RuntimeError):
                v18.load_v12_champion_context_incidence_checkpoint(tampered_path)
            tampered_optimizer = copy.deepcopy(payload)
            tampered_optimizer["optimizer_state"]["state"].pop(
                v18.MUTABLE_PARAMETER_NAMES[0]
            )
            tampered_path = Path(directory) / "tampered-optimizer.pt"
            torch.save(tampered_optimizer, tampered_path)
            with self.assertRaises(RuntimeError):
                v18.load_v12_champion_context_incidence_checkpoint(tampered_path)

    def test_component_and_full_advancement_thresholds_are_conjunctive(self) -> None:
        def panel(*, full: bool = True) -> dict[str, object]:
            return {
                "learned_context_metrics": {
                    "unique_valid_rows": 20,
                    "unique_valid_real_top_one_successes": 18,
                    "informative_rows": 24,
                    "informative_valid_mass_sum": 16.8,
                    "informative_real_mass_sum": 24.0,
                    "informative_real_normalized_mass": 0.70,
                    "informative_rank_margin_sum": 0.48,
                    "informative_rank_margin": 0.02,
                    "supported_rows": 32 if full else 20,
                    "supported_full_top_one_successes": 27 if full else 10,
                    "supported_full_valid_set_mass": 0.65 if full else 0.50,
                    "qualifying_streams": 6 if full else 3,
                },
                "projection_zero_context_metrics": {
                    "unique_valid_rows": 20,
                    "unique_valid_real_top_one_successes": 15,
                    "informative_rows": 24,
                    "informative_valid_mass_sum": 15.36,
                    "informative_real_mass_sum": 24.0,
                    "informative_real_normalized_mass": 0.64,
                    "informative_rank_margin_sum": -0.24,
                    "informative_rank_margin": -0.01,
                    "supported_rows": 32 if full else 20,
                    "supported_full_top_one_successes": 20,
                    "supported_full_valid_set_mass": 0.55,
                    "qualifying_streams": 6 if full else 3,
                },
                "panel_improved": True,
                "no_material_regression": True,
            }

        supported = v18._classify_context_incidence_panels(
            (panel(), panel(), panel(), panel()),
            integrity_passed=True,
        )
        self.assertEqual(supported["classification"], "CONTEXT_INCIDENCE_SUPPORTED")
        self.assertEqual(supported["causal_unique_valid_real_top_one_gain"], 12)
        self.assertAlmostEqual(supported["causal_real_normalized_mass_gain"], 0.06)
        component_only = v18._classify_context_incidence_panels(
            (panel(full=False),) * 4,
            integrity_passed=True,
        )
        self.assertEqual(
            component_only["classification"],
            "CONTEXT_INCIDENCE_COMPONENT_SUPPORTED",
        )
        invalid = v18._classify_context_incidence_panels(
            (panel(),) * 4,
            integrity_passed=False,
        )
        self.assertEqual(invalid["classification"], "INVALID_NO_CLAIM")
        regressed_panels = [panel() for _ in range(4)]
        regressed_panels[-1]["no_material_regression"] = False
        rejected = v18._classify_context_incidence_panels(
            tuple(regressed_panels),
            integrity_passed=True,
        )
        self.assertEqual(rejected["classification"], "CONTEXT_INCIDENCE_NOT_SUPPORTED")

    def test_override_and_source_exclude_v17_mechanisms_ids_replay_router_and_solvers(self) -> None:
        subclass_methods = {
            name
            for name, value in vars(v18.V12ChampionContextIncidenceController).items()
            if inspect.isfunction(value)
        }
        base_methods = {
            name
            for name, value in vars(v12.SoftwarePipelineController).items()
            if inspect.isfunction(value)
        }
        self.assertEqual(
            (subclass_methods & base_methods) - {"__init__"},
            {"_pool_context_tensor"},
        )
        controller = _load_system().controller
        self.assertFalse(hasattr(controller, "context_residual_experts"))
        self.assertFalse(hasattr(controller, "context_composer"))
        source = Path(v18.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        forbidden = {
            "CounterfactualPlasticityRouter",
            "SelfReferentialWeightMatrix",
            "search_teacher_plan",
            "BidirectionalOperatorPlanner",
            "commitment_index",
        }
        self.assertFalse(identifiers & forbidden)
        self.assertNotIn("context_residual_experts", source)
        self.assertNotIn("context_composer", source)
        self.assertNotIn("phase6_cross_variation_plasticity_v16", source)
        self.assertNotIn("make_software_pipeline_control_stream", source)
        self.assertNotIn("judge_software_pipeline_attempt", source)


if __name__ == "__main__":
    unittest.main()
