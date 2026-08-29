from __future__ import annotations

import copy
from dataclasses import fields, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.runners import phase6_cross_variation_plasticity_v16 as v16
from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19
from experiments.runners import phase6_oml_relation_representation as v20


_V19_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
_D2_REPORT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v11-d2-representation-overlap.json"
)


def _fresh_controller(device: str | torch.device = "cpu"):
    controller = v19.V12ChampionPairedGraphContextController(
        v12.SOFTWARE_PIPELINE_PROFILES["smoke"]
    ).to(device)
    v20._configure_oml_controller(controller, learn_rln=True)
    return controller


def _first_public_stream():
    plan = v19.v12_champion_paired_graph_context_plan()
    return v12._relation_credit_panel_streams(
        plan["commitments"], plan["panel_seed_pairs"][0]
    )[0]


def _fresh_arm(name: str = v20.ARM_SECOND_ORDER) -> v20.OMLArm:
    controller = _fresh_controller()
    inherited = v19.V12ChampionPairedGraphContextSystem(
        controller=controller,
        mixer=v12.AnonymousConflictMixer(),
        competence_state=controller.initial_state(),
        source=v19._expected_source_binding(),
        context_updates=0,
        optimizer_state=None,
    )
    report = v20._validate_parameter_partition(controller)
    named = dict(controller.named_parameters())
    optimizer = torch.optim.AdamW(
        tuple(named[item] for item in report["rln_parameter_names"]),
        lr=v20.OUTER_LEARNING_RATE,
        betas=(v20.ADAM_BETA1, v20.ADAM_BETA2),
        eps=v20.ADAM_EPSILON,
        weight_decay=v20.ADAM_WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    return v20.OMLArm(
        name=name,
        system=inherited,
        outer_optimizer=optimizer,
        fast_initial_weight=named[v20.FAST_PARAMETER_NAME].detach().clone(),
        source_frozen_digest=v20.oml_frozen_digest(controller),
        source_context_digest=v20.oml_context_digest(controller),
        source_auxiliary_digest=v20.oml_auxiliary_system_digest(inherited),
    )


def _credit_row(
    positive: float = 0.13,
    negative: float = -0.04,
    *,
    valid: tuple[bool, ...] = (True, False, True),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> v19.V19PairedGraphCreditRow:
    positive_margin = torch.tensor(positive, device=device, dtype=dtype)
    negative_margin = torch.tensor(negative, device=device, dtype=dtype)
    slot_positive = torch.tensor((0.21, -0.07, 0.04), device=device, dtype=dtype)
    slot_negative = torch.tensor((-0.03, 0.08, -0.11), device=device, dtype=dtype)
    real_logits = torch.tensor((0.2, -0.1, 0.3), device=device, dtype=dtype)
    probabilities = torch.softmax(
        torch.cat((real_logits / 0.25, real_logits.new_zeros(1))), dim=0
    )
    return v19.V19PairedGraphCreditRow(
        heldout_index=17,
        transition_index=23,
        positive_index=5,
        negative_index=7,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
        slot_positive_margins=slot_positive,
        slot_negative_margins=slot_negative,
        context_weights=probabilities[:-1],
        context_null_weight=probabilities[-1],
        context_real_logits=real_logits,
        valid_mask=torch.tensor(valid, device=device, dtype=torch.bool),
    )


def _toy_unroll(
    theta: torch.Tensor,
    initial: torch.Tensor,
    *,
    second_order: bool,
    steps: int = 3,
) -> tuple[torch.Tensor, tuple[v20.AdamWSlot, ...]]:
    weight = initial
    state = (
        v20.AdamWSlot(
            step=0,
            exp_avg=torch.zeros_like(weight),
            exp_avg_sq=torch.zeros_like(weight),
        ),
    )
    features = (
        torch.tensor((0.7, -0.4), dtype=weight.dtype, device=weight.device),
        torch.tensor((-0.2, 0.9), dtype=weight.dtype, device=weight.device),
        torch.tensor((0.5, 0.3), dtype=weight.dtype, device=weight.device),
    )
    targets = (0.15, -0.25, 0.4)
    for index in range(steps):
        feature = features[index]
        target = weight.new_tensor(targets[index])
        prediction = (weight * (feature + 0.3 * theta)).sum()
        inner_loss = 0.5 * (prediction - target).square()
        (gradient,) = torch.autograd.grad(
            inner_loss,
            (weight,),
            create_graph=second_order,
        )
        if not second_order:
            gradient = gradient.detach()
        (weight,), state = v20.functional_adamw_step(
            (weight,),
            (gradient,),
            state,
            (0.037,),
            beta1=0.8,
            beta2=0.91,
            epsilon=0.1,
            weight_decay=0.0,
        )
    return weight, state


def _d2_report(*, supported: bool = True) -> dict[str, object]:
    cell = {
        "group_overlap_summary": {
            "easy_hard_exceeds_both_within_groups": True,
        },
        "gradient_alignment": {
            "shared_comparator": {
                "observed_mean_off_diagonal_burden": 0.2,
                "p_value_one_sided": 0.04,
            }
        },
    }
    return {
        "classification": (
            "REPRESENTATION_OVERLAP_INTERFERENCE_SUPPORTED"
            if supported
            else "REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED"
        ),
        "evaluation": {
            "stages": {
                "relation_comparator_hidden": {
                    "cells": {
                        "t0_s0": copy.deepcopy(cell),
                        "t0_s1": copy.deepcopy(cell),
                    }
                }
            }
        },
    }


def _classification_families(
    *,
    second_auc: float = 0.8,
    first_auc: float = 1.0,
    second_rows: int = 100,
    first_rows: int = 94,
    second_streams: int = 25,
    first_streams: int = 22,
    retained_fraction: float | None = 0.85,
    forward_gain: float = 0.01,
    original_rows: int = 100,
    original_streams: int = 25,
    context_supported: bool = True,
    selective_harm: bool = False,
) -> dict[str, object]:
    second_panel_rows = [second_rows // 4] * 4
    for index in range(second_rows % 4):
        second_panel_rows[index] += 1
    first_panel_rows = [first_rows // 4] * 4
    for index in range(first_rows % 4):
        first_panel_rows[index] += 1
    second_panel_streams = [second_streams // 4] * 4
    for index in range(second_streams % 4):
        second_panel_streams[index] += 1
    first_panel_streams = [first_streams // 4] * 4
    for index in range(first_streams % 4):
        first_panel_streams[index] += 1
    heldout_panels = []
    for panel in range(4):
        pre = tuple(tuple(1.0 for _ in range(8)) for _ in range(8))
        post_rows = []
        for step in range(8):
            values = []
            for position in range(8):
                if position == step:
                    values.append(0.9)
                elif selective_harm:
                    values.append(1.03)
                else:
                    values.append(1.0)
            post_rows.append(tuple(values))
        heldout_panels.append(
            {
                "panel": panel,
                "arms": {
                    v20.ARM_SECOND_ORDER: {
                        "probe_pre_loss": pre,
                        "probe_post_loss": tuple(post_rows),
                        "terminal_credit": {
                            "full": {
                                "supported_rows": second_panel_rows[panel],
                                "qualifying_streams": second_panel_streams[panel],
                            }
                        },
                    },
                    v20.ARM_FIRST_ORDER: {
                        "terminal_credit": {
                            "full": {
                                "supported_rows": first_panel_rows[panel],
                                "qualifying_streams": first_panel_streams[panel],
                            }
                        }
                    },
                },
            }
        )
    heldout_arms = {
        v20.ARM_SECOND_ORDER: {
            "online_loss_auc": second_auc,
            "supported_rows": second_rows,
            "qualifying_streams": second_streams,
            "retained_fraction": retained_fraction,
            "forward_gain": forward_gain,
        },
        v20.ARM_FIRST_ORDER: {
            "online_loss_auc": first_auc,
            "supported_rows": first_rows,
            "qualifying_streams": first_streams,
        },
        v20.ARM_SOURCE_ONLINE: {"online_loss_auc": 1.1},
        v20.ARM_SECOND_NO_UPDATE: {"online_loss_auc": 1.05},
        v20.ARM_FIRST_NO_UPDATE: {"online_loss_auc": 1.08},
    }
    original_second_panel_rows = [original_rows // 4] * 4
    for index in range(original_rows % 4):
        original_second_panel_rows[index] += 1
    return {
        "heldout": {"arms": heldout_arms, "panels": tuple(heldout_panels)},
        "original": {
            "arms": {
                v20.ARM_SECOND_ORDER: {
                    "supported_rows": original_rows,
                    "qualifying_streams": original_streams,
                    "panel_supported_rows": tuple(original_second_panel_rows),
                    "context_causal": {
                        "frozen_v19_context_effect_supported": context_supported,
                    },
                },
                v20.ARM_SOURCE_ONLINE: {
                    "panel_supported_rows": tuple(original_second_panel_rows),
                },
            }
        },
    }


class Phase6OMLRelationRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_frozen_plan_has_exact_balanced_disjoint_schedule(self) -> None:
        plan = v20.oml_fit_plan()
        self.assertEqual(plan["protocol_id"], v20.PROTOCOL_ID)
        self.assertEqual(plan["source_checkpoint_sha256"], v20.SOURCE_CHECKPOINT_SHA256)
        self.assertEqual(plan["d2_result_sha256"], v20.D2_RESULT_SHA256)
        self.assertEqual(plan["retention_commitment_indices"], tuple(range(8)))
        self.assertEqual(plan["meta_training_commitment_indices"], tuple(range(8, 56)))
        self.assertEqual(plan["heldout_commitment_indices"], tuple(range(56, 64)))
        self.assertEqual(len(plan["updates"]), 240)
        self.assertEqual(plan["target_counts"], (20,) * 48)
        self.assertEqual(plan["inner_exposure_counts"], (40,) * 48)
        self.assertEqual(plan["unique_training_streams"], 240 * 16)
        self.assertEqual(plan["unique_evaluation_streams"], 960)
        self.assertEqual(plan["unique_training_seed_values"], 7_680)
        self.assertEqual(plan["unique_evaluation_seed_values"], 1_920)
        self.assertEqual(
            plan["parameter_partition"],
            {
                "rln_prefixes": v20.RLN_PARAMETER_PREFIXES,
                "rln_exact_prefix": v20.RLN_EXACT_PARAMETER_PREFIX,
                "rln_tensor_count": 67,
                "rln_parameter_count": 61_898,
                "pln_parameter_name": v20.FAST_PARAMETER_NAME,
                "pln_tensor_count": 1,
                "pln_parameter_count": 64,
                "paired_graph_frozen_tensor_count": 21,
                "paired_graph_frozen_parameter_count": 34_048,
            },
        )
        self.assertEqual(
            plan["optimization"],
            {
                "outer_updates": 240,
                "inner_steps": 8,
                "inner_learning_rate": 1.0e-3,
                "outer_learning_rate": 3.0e-4,
                "betas": (0.9, 0.999),
                "epsilon": 1.0e-8,
                "weight_decay": 0.0,
                "outer_gradient_clip": 5.0,
            },
        )
        self.assertEqual(
            plan["numerical"],
            {
                "device": "cuda",
                "dtype": "torch.float32",
                "tf32": False,
                "autocast": False,
                "exact_second_order": True,
                "allocated_memory_ceiling_bytes": 12 * 1024**3,
                "wall_time_ceiling_seconds": 150 * 60.0,
                "progress_interval": 40,
            },
        )

        training_pairs = set()
        for update_index, update in enumerate(plan["updates"]):
            self.assertEqual(update, v20._training_update_spec(update_index))
            targets = tuple((4 * update_index + slot) % 48 for slot in range(4))
            self.assertEqual(update["target_offsets"], targets)
            self.assertEqual(
                tuple(record["commitment_index"] for record in update["inner"]),
                tuple(8 + targets[step % 4] for step in range(8)),
            )
            self.assertEqual(
                tuple(record["commitment_index"] for record in update["outer_same"]),
                tuple(8 + value for value in targets),
            )
            self.assertEqual(
                tuple(record["commitment_index"] for record in update["outer_cross"]),
                tuple(8 + ((value + 24) % 48) for value in targets),
            )
            self.assertEqual(update["outer"], update["outer_same"] + update["outer_cross"])
            for role in ("inner", "outer_same", "outer_cross"):
                topology_base, surface_base = v20.TRAIN_SEED_BASES[role]
                for position, record in enumerate(update[role]):
                    self.assertEqual(record["role"], role)
                    self.assertEqual(record["position"], position)
                    self.assertEqual(
                        record["topology_seed"],
                        topology_base + 100_000 * update_index + 1_000 * position,
                    )
                    self.assertEqual(
                        record["surface_seed"],
                        surface_base + 100_000 * update_index + 1_000 * position,
                    )
                    pair = (record["topology_seed"], record["surface_seed"])
                    self.assertNotIn(pair, training_pairs)
                    training_pairs.add(pair)

        evaluation_pairs = set()
        for family_index, family in enumerate(v20.FAMILIES):
            panel_sets = []
            for panel in range(4):
                spec = v20._evaluation_panel_spec(family, panel)
                indices = tuple(spec["commitment_indices"])
                panel_sets.append(set(indices))
                if family == "original":
                    self.assertEqual(indices, tuple(range(8)))
                elif family == "heldout":
                    self.assertEqual(indices, tuple(range(56, 64)))
                else:
                    self.assertEqual(
                        indices,
                        tuple(8 + ((6 * position + panel) % 48) for position in range(8)),
                    )
                records = (
                    tuple(spec["update"])
                    + tuple(record for group in spec["probe"] for record in group)
                    + tuple(spec["terminal_credit"])
                )
                self.assertEqual((len(spec["update"]), len(spec["probe"]), len(spec["terminal_credit"])), (8, 8, 8))
                self.assertTrue(all(len(group) == 8 for group in spec["probe"]))
                for record in records:
                    pair = (record["topology_seed"], record["surface_seed"])
                    self.assertNotIn(pair, training_pairs)
                    self.assertNotIn(pair, evaluation_pairs)
                    evaluation_pairs.add(pair)
                    role_index = {"update": 0, "probe": 1, "terminal_credit": 2}[
                        record["role"]
                    ]
                    base = 20_000_000_001 + 1_000_000_000 * family_index + 100_000_000 * role_index
                    if record["role"] == "update":
                        offset = 1_000_000 * panel + 1_000 * record["step"]
                    elif record["role"] == "probe":
                        offset = 1_000_000 * panel + 100_000 * record["step"] + 1_000 * record["position"]
                    else:
                        offset = 1_000_000 * panel + 1_000 * record["position"]
                    self.assertEqual(record["topology_seed"], base + offset)
                    self.assertEqual(record["surface_seed"], base + 50_000_000 + offset)
            if family == "meta_seen":
                self.assertFalse(
                    any(
                        left & right
                        for index, left in enumerate(panel_sets)
                        for right in panel_sets[index + 1 :]
                    )
                )
        self.assertEqual((len(training_pairs), len(evaluation_pairs)), (3_840, 960))
        individual_evaluation_seeds = {
            seed for pair in evaluation_pairs for seed in pair
        }
        self.assertEqual(len(individual_evaluation_seeds), 1_920)
        first_digest = v20.oml_plan_digest()
        plan["updates"][0]["inner"][0]["topology_seed"] += 1
        self.assertEqual(first_digest, v20.oml_plan_digest())
        self.assertRegex(first_digest, r"^sha256:[0-9a-f]{64}$")

    def test_frozen_dependency_bytes_and_available_source_artifacts_are_exact(self) -> None:
        root = Path(__file__).resolve().parents[3]
        observed = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest().upper()
            for name in v20.FROZEN_DEPENDENCY_HASHES
        }
        self.assertEqual(observed, v20.FROZEN_DEPENDENCY_HASHES)
        self.assertEqual(v20.frozen_dependency_hashes(), observed)
        required = (_V19_CHECKPOINT, v20.SOURCE_RESULT_PATH, _D2_REPORT)
        if all(path.is_file() for path in required):
            report = v20.verify_oml_dependencies(
                _V19_CHECKPOINT,
                d2_path=_D2_REPORT,
                source_result_path=v20.SOURCE_RESULT_PATH,
            )
            self.assertEqual(report["source_checkpoint_sha256"], v20.SOURCE_CHECKPOINT_SHA256)
            self.assertEqual(
                report["accepted_v19_recovery_report_sha256"],
                v20.SOURCE_RESULT_SHA256,
            )
            self.assertEqual(report["d2_result_sha256"], v20.D2_RESULT_SHA256)
            self.assertEqual(
                report["d2_classification"],
                "REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED",
            )

    def test_d2_identity_uses_the_exact_frozen_protocol_literal(self) -> None:
        value = {
            "artifact_schema": v20.D2_RESULT_SCHEMA,
            "protocol_id": "phase6.public-representation-overlap.v11-d2",
            "classification": "REPRESENTATION_OVERLAP_INTERFERENCE_NOT_SUPPORTED",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d2.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(v20, "_sha256_file", return_value=v20.D2_RESULT_SHA256):
                self.assertEqual(v20._read_d2_result(path), value)
                changed = dict(value)
                changed["protocol_id"] = "phase6.public-v11-representation-overlap.v11-d2"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    v20._read_d2_result(path)

    def test_d2_overlap_gate_fails_closed_including_nonfinite_evidence(self) -> None:
        self.assertFalse(v20._d2_same_module_overlap(None))
        self.assertFalse(v20._d2_same_module_overlap({}))
        self.assertFalse(v20._d2_same_module_overlap(_d2_report(supported=False)))
        supported = _d2_report(supported=True)
        self.assertTrue(v20._d2_same_module_overlap(supported))
        for cell_id in ("t0_s0", "t0_s1"):
            changed = copy.deepcopy(supported)
            changed["evaluation"]["stages"]["relation_comparator_hidden"]["cells"][cell_id][
                "group_overlap_summary"
            ]["easy_hard_exceeds_both_within_groups"] = False
            self.assertFalse(v20._d2_same_module_overlap(changed))
        for field in ("observed_mean_off_diagonal_burden", "p_value_one_sided"):
            changed = copy.deepcopy(supported)
            changed["evaluation"]["stages"]["relation_comparator_hidden"]["cells"]["t0_s0"][
                "gradient_alignment"
            ]["shared_comparator"][field] = float("nan")
            self.assertFalse(v20._d2_same_module_overlap(changed))

    def test_frozen_classification_arithmetic_and_exclusive_priority(self) -> None:
        harmonized = v20._classify_oml(
            _classification_families(), _d2_report(supported=False)
        )
        self.assertEqual(harmonized["classification"], "OML_V19_HARMONIZED_ADVANCEMENT")
        self.assertTrue(harmonized["gates"]["SECOND_ORDER_OML_CREDIT_SUPPORTED"])
        self.assertTrue(harmonized["gates"]["OML_CROSS_MECHANISM_ADVANCEMENT"])
        self.assertTrue(harmonized["gates"]["OML_V19_HARMONIZED_ADVANCEMENT"])
        self.assertFalse(harmonized["gates"]["anml_trigger"])

        component = v20._classify_oml(
            _classification_families(context_supported=False, selective_harm=True),
            _d2_report(supported=True),
        )
        self.assertEqual(
            component["classification"], "OML_COMPONENT_SUPPORTED_NOT_INTEGRATED"
        )
        self.assertTrue(component["gates"]["anml_trigger"])
        self.assertTrue(component["gates"]["selective_plasticity_harm"])

        cross = v20._classify_oml(
            _classification_families(context_supported=False),
            _d2_report(supported=True),
        )
        self.assertEqual(cross["classification"], "OML_CROSS_MECHANISM_ADVANCEMENT")
        self.assertFalse(cross["gates"]["anml_trigger"])

        second_only = v20._classify_oml(
            _classification_families(
                second_rows=95,
                first_rows=91,
                second_streams=23,
                first_streams=21,
                context_supported=False,
            ),
            _d2_report(supported=False),
        )
        self.assertEqual(
            second_only["classification"], "SECOND_ORDER_OML_CREDIT_SUPPORTED"
        )
        self.assertFalse(second_only["gates"]["OML_CROSS_MECHANISM_ADVANCEMENT"])

        fast_only = v20._classify_oml(
            _classification_families(
                second_auc=0.97,
                first_auc=0.96,
                second_rows=94,
                first_rows=94,
                second_streams=22,
                first_streams=22,
                context_supported=False,
            ),
            _d2_report(supported=False),
        )
        self.assertEqual(
            fast_only["classification"],
            "FAST_ADAPTATION_SUPPORTED_OML_ATTRIBUTION_NOT_ESTABLISHED",
        )
        self.assertFalse(fast_only["gates"]["SECOND_ORDER_OML_CREDIT_SUPPORTED"])
        self.assertTrue(fast_only["gates"]["fast_adaptation_supported"])

        unsupported = v20._classify_oml(
            _classification_families(second_auc=1.2, first_auc=1.0),
            _d2_report(supported=False),
        )
        self.assertEqual(unsupported["classification"], "OML_NOT_SUPPORTED")

        exact_boundaries = v20._classify_oml(
            _classification_families(
                second_auc=0.95,
                first_auc=1.0,
                second_rows=99,
                first_rows=96,
                second_streams=25,
                first_streams=23,
                retained_fraction=0.80,
                forward_gain=0.0,
                context_supported=False,
            ),
            _d2_report(supported=False),
        )
        self.assertTrue(exact_boundaries["gates"]["SECOND_ORDER_OML_CREDIT_SUPPORTED"])
        self.assertTrue(exact_boundaries["gates"]["OML_CROSS_MECHANISM_ADVANCEMENT"])

    def test_immutable_d2_false_result_denies_anml_and_metrics_fail_closed(self) -> None:
        families = _classification_families(context_supported=False, selective_harm=True)
        frozen_false = v20._classify_oml(families, _d2_report(supported=False))
        self.assertFalse(frozen_false["gates"]["d2_same_module_overlap"])
        self.assertFalse(frozen_false["gates"]["anml_trigger"])
        self.assertEqual(frozen_false["classification"], "OML_CROSS_MECHANISM_ADVANCEMENT")

        regressed = _classification_families(context_supported=False)
        first_panel = regressed["heldout"]["panels"][0]["arms"][v20.ARM_FIRST_ORDER][
            "terminal_credit"
        ]["full"]
        second_panel = regressed["heldout"]["panels"][0]["arms"][v20.ARM_SECOND_ORDER][
            "terminal_credit"
        ]["full"]
        second_panel["supported_rows"] = first_panel["supported_rows"] - 2
        result = v20._classify_oml(regressed, _d2_report(supported=False))
        self.assertFalse(result["gates"]["SECOND_ORDER_OML_CREDIT_SUPPORTED"])
        self.assertFalse(result["comparisons"]["heldout_all_panels_row_nonregressed"])

        nonfinite = _classification_families()
        nonfinite["heldout"]["arms"][v20.ARM_SECOND_ORDER]["online_loss_auc"] = float(
            "nan"
        )
        with self.assertRaises(ValueError):
            v20._classify_oml(nonfinite, _d2_report(supported=False))

    def test_online_auc_formula_and_update_no_update_controls_are_separate(self) -> None:
        controller = _fresh_controller()
        initial = dict(controller.named_parameters())[v20.FAST_PARAMETER_NAME].detach().clone()
        streams = {
            "update": tuple(("update", step) for step in range(8)),
            "probe": tuple(
                tuple(("probe", step, position) for position in range(8))
                for step in range(8)
            ),
            "terminal_credit": tuple(("terminal", position) for position in range(8)),
        }

        def fake_loss(observed_controller, fast_weight, stream):
            self.assertIs(observed_controller, controller)
            if stream[0] == "update":
                value = float(stream[1] + 1)
            elif stream[0] == "probe":
                value = 0.2 + 0.01 * stream[1] + 0.001 * stream[2]
            else:
                value = 0.3
            return fast_weight.sum() * 0.0 + fast_weight.new_tensor(value)

        def fake_credit(*args, zero_residual=False, **kwargs):
            del args, kwargs
            return {
                "supported_rows": 24,
                "qualifying_streams": 6,
                "relation_signature_digest": "sha256:" + "2" * 64,
                "zero_residual": zero_residual,
            }

        with mock.patch.object(v20, "_stream_loss", side_effect=fake_loss), mock.patch.object(
            v20, "_terminal_credit_metrics", side_effect=fake_credit
        ):
            online = v20._evaluate_arm_panel(
                controller,
                initial,
                streams,
                arm_name=v20.ARM_SECOND_ORDER,
                updates_enabled=True,
            )
            no_update = v20._evaluate_arm_panel(
                controller,
                initial,
                streams,
                arm_name=v20.ARM_SECOND_NO_UPDATE,
                updates_enabled=False,
            )
        expected_auc = (0.5 * 1.0 + sum(range(2, 8)) + 0.5 * 8.0) / 7.0
        self.assertEqual(expected_auc, 4.5)
        self.assertEqual(online["online_loss_auc"], expected_auc)
        self.assertEqual(no_update["online_loss_auc"], expected_auc)
        self.assertEqual(online["terminal_fast_step"], 8)
        self.assertEqual(no_update["terminal_fast_step"], 0)
        self.assertTrue(all(item["adamw_step"] == 0 for item in no_update["step_diagnostics"]))

    def test_frozen_v19_context_causal_thresholds_and_signature_guard(self) -> None:
        signature = "sha256:" + "3" * 64

        def metric(top_one: int, mass: float, margin: float, *, digest: str = signature):
            return {
                "supported_rows": 24,
                "informative_rows": 24,
                "unique_valid_rows": 24,
                "unique_valid_top_one": top_one,
                "valid_real_mass_numerator": mass,
                "all_real_mass_denominator": 1.0,
                "real_normalized_valid_mass": mass,
                "margin_sum": margin,
                "margin_count": 1,
                "mean_informative_log_weight_margin": margin,
                "full_valid_mass_sum": mass,
                "full_top_one_count": top_one,
                "qualifying_streams": 6,
                "relation_signature_digest": digest,
            }

        full = tuple(metric(13, 0.20, 0.20) for _ in range(4))
        zero = tuple(metric(10, 0.10, 0.10) for _ in range(4))
        result = v20._terminal_context_causal_metrics(full, zero)
        self.assertEqual(result["aggregate_top_one_gain"], 12)
        self.assertGreaterEqual(result["aggregate_real_normalized_valid_mass_gain"], 0.05)
        self.assertGreaterEqual(result["aggregate_informative_margin_gain"], 0.05)
        self.assertEqual(result["positive_recurrence_panels"], 4)
        self.assertTrue(result["all_panels_nonregressed"])
        self.assertTrue(result["relation_signatures_exact"])
        self.assertTrue(result["frozen_v19_context_effect_supported"])

        changed = list(full)
        changed[0] = metric(13, 0.20, 0.20, digest="sha256:" + "4" * 64)
        rejected = v20._terminal_context_causal_metrics(tuple(changed), zero)
        self.assertFalse(rejected["relation_signatures_exact"])
        self.assertFalse(rejected["frozen_v19_context_effect_supported"])

    def test_evaluation_aggregate_uses_exact_acquisition_retention_and_forward_denominators(self) -> None:
        panels = []
        for panel in range(4):
            pre = []
            post = []
            terminal = []
            for step in range(8):
                pre_row = tuple(10.0 + panel + step + 0.01 * position for position in range(8))
                post_row = tuple(
                    value - 2.0
                    if position == step
                    else value - 0.25
                    if position > step
                    else value
                    for position, value in enumerate(pre_row)
                )
                pre.append(pre_row)
                post.append(post_row)
                terminal.append(pre_row[step] - 1.0)
            full = {
                "supported_rows": 20 + panel,
                "qualifying_streams": 5 + panel,
            }
            panels.append(
                {
                    "online_loss_auc": 0.5 + 0.1 * panel,
                    "probe_pre_loss": tuple(pre),
                    "probe_post_loss": tuple(post),
                    "probe_terminal_loss": tuple(terminal),
                    "terminal_credit": {
                        "full": full,
                        "zero_residual": dict(full),
                    },
                }
            )
        with mock.patch.object(
            v20,
            "_terminal_context_causal_metrics",
            return_value={"frozen_v19_context_effect_supported": True},
        ):
            result = v20._aggregate_arm_panels(tuple(panels))
        self.assertAlmostEqual(result["online_loss_auc"], 0.65)
        self.assertEqual(result["supported_rows"], 86)
        self.assertEqual(result["qualifying_streams"], 26)
        self.assertAlmostEqual(result["immediate_gain"], 2.0)
        self.assertAlmostEqual(result["terminal_gain"], 1.0)
        self.assertAlmostEqual(result["retained_fraction"], 0.5)
        self.assertTrue(result["retention_valid"])
        self.assertAlmostEqual(result["forward_gain"], 0.25)

        no_gain = copy.deepcopy(panels)
        for panel in no_gain:
            panel["probe_post_loss"] = panel["probe_pre_loss"]
        with mock.patch.object(
            v20,
            "_terminal_context_causal_metrics",
            return_value={"frozen_v19_context_effect_supported": True},
        ):
            invalid = v20._aggregate_arm_panels(tuple(no_gain))
        self.assertEqual(invalid["immediate_gain"], 0.0)
        self.assertIsNone(invalid["retained_fraction"])
        self.assertFalse(invalid["retention_valid"])

    def test_cuda_preflight_outer_mode_selection_uses_isolated_memory_peaks(self) -> None:
        ceiling = v20.ALLOCATED_MEMORY_CEILING_BYTES
        self.assertEqual(
            v20._select_outer_mode_from_allocations(ceiling, 0),
            ("full", ceiling),
        )
        self.assertEqual(
            v20._select_outer_mode_from_allocations(ceiling + 1, ceiling),
            ("split_4_plus_4", ceiling),
        )
        with self.assertRaises(RuntimeError):
            v20._select_outer_mode_from_allocations(ceiling + 1, ceiling + 1)
        for invalid in (True, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    v20._select_outer_mode_from_allocations(invalid, 0)
                with self.assertRaises(ValueError):
                    v20._select_outer_mode_from_allocations(0, invalid)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA synthetic preflight requires CUDA")
    def test_cuda_preflight_is_synthetic_only_and_never_enters_semantic_paths(self) -> None:
        with mock.patch.object(
            v20, "_build_fit_streams", side_effect=AssertionError("semantic streams forbidden")
        ), mock.patch.object(
            v20, "fit_oml_update", side_effect=AssertionError("semantic fit forbidden")
        ), mock.patch.object(
            v20, "evaluate_oml", side_effect=AssertionError("semantic evaluation forbidden")
        ), mock.patch.object(
            v19,
            "public_paired_graph_credit_rows",
            side_effect=AssertionError("public task rows forbidden in synthetic preflight"),
        ):
            report = v20.synthetic_cuda_preflight("cuda")
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(report["device"], "cuda")
        self.assertEqual(report["dtype"], "torch.float32")
        self.assertLessEqual(
            report["full_split_objective_abs_delta"], report["equivalence_tolerance"]
        )
        self.assertLessEqual(
            report["full_split_max_gradient_abs_delta"], report["equivalence_tolerance"]
        )
        full = report["full_mode_maximum_allocated_bytes"]
        split = report["split_mode_maximum_allocated_bytes"]
        selected = report["selected_mode_maximum_allocated_bytes"]
        ceiling = report["allocated_memory_ceiling_bytes"]
        aggregate = max(full, split)
        self.assertEqual(report["overall_maximum_allocated_bytes"], aggregate)
        self.assertEqual(report["maximum_allocated_bytes"], aggregate)
        self.assertLessEqual(selected, ceiling)
        if report["selected_outer_mode"] == "full":
            self.assertEqual(selected, full)
            self.assertLessEqual(full, ceiling)
        else:
            self.assertEqual(report["selected_outer_mode"], "split_4_plus_4")
            self.assertEqual(selected, split)
            self.assertGreater(full, ceiling)
            self.assertLessEqual(split, ceiling)

    def test_v16_functional_adamw_alias_has_exact_forward_and_zero_vjp(self) -> None:
        self.assertIs(v20.functional_adamw_step, v16.functional_adamw_step)
        generator = torch.Generator().manual_seed(2026082901)
        initial = torch.randn(11, generator=generator, dtype=torch.float32)
        gradients = (
            torch.randn(11, generator=generator, dtype=torch.float32),
            torch.zeros(11, dtype=torch.float32),
            None,
        )
        zero = torch.zeros_like(initial)
        source_parameter = (initial.clone(),)
        actual_parameter = (initial.clone(),)
        source_state = (
            v16.AdamWSlot(step=0, exp_avg=zero.clone(), exp_avg_sq=zero.clone()),
        )
        actual_state = (
            v20.AdamWSlot(step=0, exp_avg=zero.clone(), exp_avg_sq=zero.clone()),
        )
        for gradient in gradients:
            source_parameter, source_state = v16.functional_adamw_step(
                source_parameter, (gradient,), source_state, (1.0e-3,)
            )
            actual_parameter, actual_state = v20.functional_adamw_step(
                actual_parameter, (gradient,), actual_state, (1.0e-3,)
            )
            self.assertTrue(torch.equal(actual_parameter[0], source_parameter[0]))
            self.assertEqual(actual_state[0].step, source_state[0].step)
            self.assertTrue(torch.equal(actual_state[0].exp_avg, source_state[0].exp_avg))
            self.assertTrue(
                torch.equal(actual_state[0].exp_avg_sq, source_state[0].exp_avg_sq)
            )

        direction = torch.zeros(13, dtype=torch.float32, requires_grad=True)
        parameter = torch.linspace(-0.3, 0.3, 13, dtype=torch.float32)
        slot = v20.AdamWSlot(
            step=0,
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
        )
        updated, state = v20.functional_adamw_step(
            (parameter,), (direction,), (slot,), (3.0e-4,)
        )
        (vjp,) = torch.autograd.grad(
            (updated[0] * torch.linspace(0.5, 1.5, 13)).sum(), (direction,)
        )
        self.assertTrue(torch.equal(updated[0], parameter))
        self.assertEqual(state[0].step, 1)
        self.assertTrue(torch.isfinite(vjp).all().item())

    def test_missing_second_order_credit_regression_and_identity_path(self) -> None:
        theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
        initial = torch.tensor((0.35, -0.2), dtype=torch.float64, requires_grad=True)
        full_weight, _ = _toy_unroll(theta, initial, second_order=True, steps=1)
        first_weight, _ = _toy_unroll(theta, initial, second_order=False, steps=1)
        self.assertTrue(torch.equal(full_weight, first_weight))
        full_outer = 0.5 * (full_weight - full_weight.new_tensor((0.1, 0.25))).square().sum()
        first_outer = 0.5 * (first_weight - first_weight.new_tensor((0.1, 0.25))).square().sum()
        full_theta, full_initial = torch.autograd.grad(
            full_outer, (theta, initial), retain_graph=True
        )
        first_theta, first_initial = torch.autograd.grad(
            first_outer, (theta, initial), allow_unused=True
        )
        self.assertGreater(abs(float(full_theta)), 1.0e-8)
        self.assertIsNone(first_theta)
        self.assertGreater(float(full_initial.norm()), 0.0)
        self.assertGreater(float(first_initial.norm()), 0.0)

    def test_three_step_second_order_gradient_matches_fp64_finite_difference(self) -> None:
        initial_value = torch.tensor((0.35, -0.2), dtype=torch.float64)

        def objective(theta_value: torch.Tensor) -> torch.Tensor:
            initial = initial_value.clone().requires_grad_(True)
            weight, _ = _toy_unroll(theta_value, initial, second_order=True, steps=3)
            return 0.5 * (weight - weight.new_tensor((0.1, 0.25))).square().sum()

        theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
        (analytic,) = torch.autograd.grad(objective(theta), (theta,))
        step = 1.0e-5
        upper = objective(torch.tensor(0.37 + step, dtype=torch.float64)).detach()
        lower = objective(torch.tensor(0.37 - step, dtype=torch.float64)).detach()
        finite_difference = (upper - lower) / (2.0 * step)
        self.assertTrue(torch.isfinite(analytic).item())
        self.assertTrue(
            torch.allclose(analytic, finite_difference, atol=2.0e-7, rtol=2.0e-5),
            (analytic, finite_difference),
        )

    def test_all_eight_production_unroll_steps_detach_only_inner_gradient_credit(self) -> None:
        theta = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
        initial = torch.tensor(((0.3, -0.2),), dtype=torch.float64)

        def fake_loss(controller, fast_weight, stream):
            del controller
            index = int(stream)
            base = fast_weight.new_tensor((0.4 + 0.03 * index, -0.2 + 0.02 * index))
            feature = base + theta * fast_weight.new_tensor((0.5, -0.35))
            target = fast_weight.new_tensor(-0.1 + 0.025 * index)
            return 0.5 * ((fast_weight.reshape(-1) * feature).sum() - target).square()

        streams = tuple(range(8))
        with mock.patch.object(v20, "_stream_loss", side_effect=fake_loss):
            full_weight, full_state, full_diagnostic = v20._unroll_inner(
                object(), initial, streams, second_order=True
            )
            first_weight, first_state, first_diagnostic = v20._unroll_inner(
                object(), initial, streams, second_order=False
            )
        self.assertTrue(torch.equal(full_weight.detach(), first_weight.detach()))
        self.assertTrue(
            torch.equal(full_state[0].exp_avg.detach(), first_state[0].exp_avg.detach())
        )
        self.assertTrue(
            torch.equal(full_state[0].exp_avg_sq.detach(), first_state[0].exp_avg_sq.detach())
        )
        self.assertEqual(full_state[0].step, 8)
        self.assertEqual(first_state[0].step, 8)
        self.assertTrue(
            all(
                item["step"] == index + 1 and item["gradient_detached"]
                for index, item in enumerate(first_diagnostic["step_diagnostics"])
            )
        )
        self.assertTrue(
            all(not item["gradient_detached"] for item in full_diagnostic["step_diagnostics"])
        )
        (full_credit,) = torch.autograd.grad(full_weight.sum(), (theta,), retain_graph=False)
        (first_credit,) = torch.autograd.grad(
            first_weight.sum(), (theta,), allow_unused=True, retain_graph=False
        )
        self.assertGreater(abs(float(full_credit)), 1.0e-8)
        self.assertIsNone(first_credit)

    def test_exact_rln_pln_and_frozen_partition(self) -> None:
        controller = _fresh_controller()
        report = v20._validate_parameter_partition(controller)
        named = dict(controller.named_parameters())
        rln = tuple(report["rln_parameter_names"])
        pln = tuple(report["pln_parameter_names"])
        frozen = tuple(report["frozen_parameter_names"])
        self.assertEqual(len(rln), 67)
        self.assertEqual(sum(named[name].numel() for name in rln), 61_898)
        self.assertEqual(pln, ("relation_comparator.2.weight",))
        self.assertEqual(named[pln[0]].numel(), 64)
        self.assertEqual(set(rln) | set(pln) | set(frozen), set(named))
        self.assertFalse(set(rln) & set(pln))
        self.assertFalse((set(rln) | set(pln)) & set(frozen))
        self.assertEqual(
            {name for name, parameter in named.items() if parameter.requires_grad},
            set(rln),
        )
        self.assertTrue(
            all(name in frozen for name in v19.MUTABLE_PARAMETER_NAMES),
            "all V19 paired-graph context tensors must be frozen",
        )

    def test_row_and_entropic_objectives_are_exact_and_ignore_detached_credit(self) -> None:
        row = _credit_row(valid=(True, False, True))
        altered = _credit_row(valid=(False, True, False))
        expected = v12._paired_relation_margin_loss(
            row.positive_margin, row.negative_margin
        ) + 0.25 * v12._relation_instance_losses(
            row.slot_positive_margins, row.slot_negative_margins
        )[1]
        self.assertTrue(torch.equal(v20._row_loss(row), expected))
        self.assertTrue(torch.equal(v20._row_loss(row), v20._row_loss(altered)))

        for count in (4, 8):
            losses = torch.linspace(0.02, 0.31, count, dtype=torch.float64)
            expected_entropic = 0.5 * losses.mean() + 0.5 * 0.05 * (
                torch.logsumexp(losses / 0.05, dim=0) - math.log(count)
            )
            self.assertTrue(
                torch.equal(
                    v20._anonymous_entropic_objective(losses, expected_count=count),
                    expected_entropic,
                )
            )
        with self.assertRaises(ValueError):
            v20._anonymous_entropic_objective(
                torch.tensor((0.1, float("nan"))), expected_count=2
            )
        with self.assertRaises(ValueError):
            v20._anonymous_entropic_objective(
                torch.ones(3, dtype=torch.float32), expected_count=4
            )

    def test_grouped_and_independent_row_objective_evaluation_are_exact_twins(self) -> None:
        groups = tuple(
            tuple(
                _credit_row(
                    positive=0.08 + 0.01 * group + 0.002 * row,
                    negative=-0.03 - 0.004 * group + 0.001 * row,
                )
                for row in range(4)
            )
            for group in range(8)
        )
        independent = torch.stack(
            tuple(v20._stream_loss_from_rows(group) for group in groups)
        )
        row_matrix = torch.stack(
            tuple(torch.stack(tuple(v20._row_loss(row) for row in group)) for group in groups)
        )
        grouped = 0.5 * row_matrix.mean(dim=1) + 0.5 * 0.05 * (
            torch.logsumexp(row_matrix / 0.05, dim=1) - math.log(4)
        )
        self.assertTrue(torch.equal(independent, grouped))
        self.assertTrue(
            torch.equal(
                v20._anonymous_entropic_objective(independent, expected_count=8),
                0.5 * grouped.mean()
                + 0.5
                * 0.05
                * (torch.logsumexp(grouped / 0.05, dim=0) - math.log(8)),
            )
        )

    def test_functional_rows_keep_exact_v19_type_and_leave_no_wrapper_or_write(self) -> None:
        controller = _fresh_controller()
        stream = _first_public_stream()
        before_type = type(controller)
        before_state = {name: value.detach().clone() for name, value in controller.state_dict().items()}
        before_attributes = set(vars(controller))
        fast_name = v20.FAST_PARAMETER_NAME
        fast = dict(controller.named_parameters())[fast_name].detach().clone().requires_grad_(True)
        with mock.patch.object(
            v19,
            "public_paired_graph_credit_rows",
            wraps=v19.public_paired_graph_credit_rows,
        ) as row_builder:
            rows = v20._functional_credit_rows(controller, fast, stream)
        self.assertEqual(type(controller), before_type)
        self.assertIs(before_type, v19.V12ChampionPairedGraphContextController)
        self.assertGreater(row_builder.call_count, 0)
        for call in row_builder.call_args_list:
            self.assertIs(type(call.args[0]), v19.V12ChampionPairedGraphContextController)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.positive_margin.requires_grad for row in rows))
        self.assertEqual(set(vars(controller)), before_attributes)
        for name, value in controller.state_dict().items():
            self.assertTrue(torch.equal(value, before_state[name]), name)
        self.assertFalse(hasattr(controller, "_oml_fast_weight"))

    def test_actual_inner_step_is_numerically_paired_but_only_full_has_rln_credit(self) -> None:
        controller = _fresh_controller()
        stream = _first_public_stream()
        named = dict(controller.named_parameters())
        report = v20._validate_parameter_partition(controller)
        rln = tuple(named[name] for name in report["rln_parameter_names"])
        before = {name: value.detach().clone() for name, value in controller.state_dict().items()}
        initial = named[v20.FAST_PARAMETER_NAME].detach().clone()
        full_weight, full_state = v20._fresh_fast_state(initial)
        first_weight, first_state = v20._fresh_fast_state(initial)
        full_weight, full_state, full_diagnostic = v20._inner_step(
            controller,
            full_weight,
            full_state,
            stream,
            second_order=True,
        )
        first_weight, first_state, first_diagnostic = v20._inner_step(
            controller,
            first_weight,
            first_state,
            stream,
            second_order=False,
        )
        self.assertTrue(torch.equal(full_weight.detach(), first_weight.detach()))
        self.assertTrue(
            torch.equal(full_state[0].exp_avg.detach(), first_state[0].exp_avg.detach())
        )
        self.assertTrue(
            torch.equal(full_state[0].exp_avg_sq.detach(), first_state[0].exp_avg_sq.detach())
        )
        self.assertEqual((full_state[0].step, first_state[0].step), (1, 1))
        self.assertFalse(full_diagnostic["gradient_detached"])
        self.assertTrue(first_diagnostic["gradient_detached"])
        full_credit = torch.autograd.grad(
            full_weight.sum(), rln, allow_unused=True, retain_graph=False
        )
        self.assertTrue(
            any(
                gradient is not None and int(torch.count_nonzero(gradient).item()) > 0
                for gradient in full_credit
            )
        )
        first_credit = torch.autograd.grad(
            first_weight.sum(), rln, allow_unused=True, retain_graph=False
        )
        self.assertTrue(all(gradient is None for gradient in first_credit))
        for name, value in controller.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_split_outer_fallback_matches_full_objective_and_gradients(self) -> None:
        arm = _fresh_arm()
        report = v20._validate_parameter_partition(arm.controller)
        rln_names = tuple(report["rln_parameter_names"])

        def fake_unroll(controller, initial_weight, streams, *, second_order):
            del controller, streams
            fast, state = v20._fresh_fast_state(initial_weight)
            return fast, state, {"steps": 8, "second_order": second_order}

        def fake_stream_loss(controller, fast_weight, stream):
            del fast_weight
            named = dict(controller.named_parameters())
            base = sum(named[name].square().mean() for name in rln_names)
            scale = base.new_tensor(float(stream))
            return (0.03 + 0.01 * scale) * base + 0.02 * scale.square()

        inner = tuple(range(8))
        outer = tuple(range(1, 9))
        with mock.patch.object(v20, "_unroll_inner", side_effect=fake_unroll), mock.patch.object(
            v20, "_stream_loss", side_effect=fake_stream_loss
        ):
            full_objective, full_gradients, full_diagnostic = v20._outer_gradients_full(
                arm, inner, outer, second_order=True
            )
            split_objective, split_gradients, split_diagnostic = v20._outer_gradients_split(
                arm, inner, outer, second_order=True
            )
        self.assertEqual(full_diagnostic["mode"], "full")
        self.assertEqual(split_diagnostic["mode"], "split_4_plus_4")
        self.assertTrue(
            torch.allclose(full_objective, split_objective, atol=1.0e-7, rtol=1.0e-7)
        )
        self.assertEqual(len(full_gradients), 67)
        for full, split in zip(full_gradients, split_gradients, strict=True):
            self.assertTrue(torch.allclose(full, split, atol=1.0e-6, rtol=1.0e-6))

    def test_paired_fit_uses_the_same_immutable_exposures_in_the_same_order(self) -> None:
        second_arm = SimpleNamespace(controller=torch.nn.Linear(1, 1))
        first_arm = SimpleNamespace(controller=torch.nn.Linear(1, 1))
        system = SimpleNamespace(
            completed_updates=0,
            outer_mode=None,
            second_order_oml=second_arm,
            first_order_meta=first_arm,
        )
        inner = tuple(object() for _ in range(8))
        outer = tuple(object() for _ in range(8))
        calls = []

        def gradient_function(arm, observed_inner, observed_outer, *, second_order):
            calls.append((arm, observed_inner, observed_outer, second_order))
            return (
                torch.tensor(0.75),
                (torch.tensor(0.1),),
                {
                    "outer_stream_losses": tuple(float(index) for index in range(8)),
                    "arm_second_order": second_order,
                },
            )

        with mock.patch.object(v20, "_assert_system_integrity"), mock.patch.object(
            v20, "_outer_gradients_full", side_effect=gradient_function
        ), mock.patch.object(
            v20, "_apply_outer_step", return_value={"outer_update": 1}
        ) as owner_step, mock.patch.object(
            v20, "_allocated_bytes", return_value=0
        ), mock.patch.object(
            v20, "oml_system_digest", return_value="sha256:" + "1" * 64
        ):
            result = v20.fit_oml_update(
                system,
                streams={"inner": inner, "outer": outer},
            )
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], second_arm)
        self.assertIs(calls[1][0], first_arm)
        self.assertIs(calls[0][1], calls[1][1])
        self.assertIs(calls[0][2], calls[1][2])
        self.assertEqual(calls[0][1], inner)
        self.assertEqual(calls[0][2], outer)
        self.assertEqual((calls[0][3], calls[1][3]), (True, False))
        self.assertEqual(owner_step.call_count, 2)
        self.assertTrue(result["paired_forward_equal_before_owner_step"])
        self.assertEqual(result["unique_streams"], 16)
        self.assertEqual(result["public_rows"], 64)
        self.assertEqual(system.outer_mode, "full")

    def test_outer_step_writes_only_rln_and_rejects_nonfinite_before_step(self) -> None:
        arm = _fresh_arm()
        report = v20._validate_parameter_partition(arm.controller)
        named = dict(arm.controller.named_parameters())
        rln_names = tuple(report["rln_parameter_names"])
        fast_before = named[v20.FAST_PARAMETER_NAME].detach().clone()
        frozen_before = v20.oml_frozen_digest(arm.controller)
        context_before = v20.oml_context_digest(arm.controller)
        rln_before = {name: named[name].detach().clone() for name in rln_names}
        gradients = tuple(torch.full_like(named[name], 0.01) for name in rln_names)
        diagnostic = v20._apply_outer_step(arm, gradients)
        self.assertEqual(diagnostic["outer_update"], 1)
        self.assertTrue(diagnostic["fast_initial_unchanged"])
        self.assertEqual(v20.oml_frozen_digest(arm.controller), frozen_before)
        self.assertEqual(v20.oml_context_digest(arm.controller), context_before)
        self.assertTrue(torch.equal(named[v20.FAST_PARAMETER_NAME].detach(), fast_before))
        self.assertTrue(any(not torch.equal(named[name].detach(), rln_before[name]) for name in rln_names))

        before_invalid = {name: value.detach().clone() for name, value in arm.controller.state_dict().items()}
        invalid = list(gradients)
        invalid[0] = torch.full_like(invalid[0], float("nan"))
        with self.assertRaises(RuntimeError):
            v20._apply_outer_step(arm, invalid)
        self.assertEqual(arm.outer_updates, 1)
        for name, value in arm.controller.state_dict().items():
            self.assertTrue(torch.equal(value, before_invalid[name]), name)

    def test_frozen_auxiliary_system_state_is_bound_and_mutation_is_rejected(self) -> None:
        arm = _fresh_arm()
        baseline = arm.source_auxiliary_digest
        self.assertEqual(v20.oml_auxiliary_system_digest(arm.system), baseline)

        mixer_state = copy.deepcopy(arm.system.mixer.state_dict())
        mixer_parameter = next(arm.system.mixer.parameters())
        with torch.no_grad():
            mixer_parameter.reshape(-1)[0].add_(0.01)
        with self.assertRaises(RuntimeError):
            v20._assert_arm_integrity(arm)
        arm.system.mixer.load_state_dict(mixer_state, strict=True)

        state_field = next(
            field
            for field in fields(arm.system.competence_state)
            if isinstance(getattr(arm.system.competence_state, field.name), torch.Tensor)
            and getattr(arm.system.competence_state, field.name).is_floating_point()
        )
        state_tensor = getattr(arm.system.competence_state, state_field.name)
        state_before = state_tensor.clone()
        with torch.no_grad():
            state_tensor.reshape(-1)[0].add_(0.01)
        with self.assertRaises(RuntimeError):
            v20._assert_arm_integrity(arm)
        with torch.no_grad():
            state_tensor.copy_(state_before)

        source_before = arm.system.source
        arm.system.source = replace(source_before, checkpoint_sha256="0" * 64)
        with self.assertRaises(RuntimeError):
            v20._assert_arm_integrity(arm)
        arm.system.source = source_before

        arm.system.context_updates += 1
        with self.assertRaises(RuntimeError):
            v20._assert_arm_integrity(arm)
        arm.system.context_updates -= 1

        arm.system.optimizer_state = {"unexpected": torch.tensor(1.0)}
        with self.assertRaises(RuntimeError):
            v20._assert_arm_integrity(arm)
        arm.system.optimizer_state = None
        v20._assert_arm_integrity(arm)
        self.assertEqual(v20.oml_auxiliary_system_digest(arm.system), baseline)

    @unittest.skipUnless(_V19_CHECKPOINT.is_file(), "frozen V19 terminal checkpoint required")
    def test_checkpoint_round_trip_and_synthetic_continuation_are_exact(self) -> None:
        system = v20.build_oml_system(_V19_CHECKPOINT, device="cpu")
        self.assertEqual(system.source_checkpoint_sha256, v20.SOURCE_CHECKPOINT_SHA256)
        self.assertEqual(
            v20.oml_controller_digest(system.second_order_oml.controller),
            v20.oml_controller_digest(system.first_order_meta.controller),
        )
        self.assertEqual(
            system.second_order_oml.source_context_digest,
            v20.oml_context_digest(system.source_v19.controller),
        )
        self.assertEqual(
            system.second_order_oml.source_frozen_digest,
            v20.oml_frozen_digest(system.source_v19.controller),
        )
        self.assertFalse(any(parameter.requires_grad for parameter in system.source_v19.controller.parameters()))
        self.assertIsNone(system.outer_mode)
        self.assertEqual(v20.bind_oml_outer_mode(system, "full"), "full")
        self.assertEqual(system.outer_mode, "full")
        with self.assertRaises(RuntimeError):
            v20.bind_oml_outer_mode(system, "split_4_plus_4")

        def synthetic_owner_step(target: v20.OMLSystem) -> None:
            for arm in (target.second_order_oml, target.first_order_meta):
                gradients = tuple(
                    torch.full_like(parameter, 0.001 * (index + 1))
                    for index, (_, parameter) in enumerate(v20._rln_parameters(arm))
                )
                v20._apply_outer_step(arm, gradients)
            target.completed_updates += 1
            v20._assert_system_integrity(target)

        synthetic_owner_step(system)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.pt"
            expected_rng = torch.get_rng_state().clone()
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                v20.save_oml_checkpoint(path, system)
                self.assertLessEqual(path.stat().st_size, 16 * 1024**2)
                torch.manual_seed(999)
                restored = v20.load_oml_checkpoint(
                    path, _V19_CHECKPOINT, device="cpu"
                )
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertEqual(v20.oml_system_digest(restored), v20.oml_system_digest(system))
            self.assertEqual(restored.outer_mode, "full")
            with self.assertRaises(RuntimeError):
                v20.bind_oml_outer_mode(restored, "split_4_plus_4")
            self.assertEqual(
                restored.second_order_oml.source_auxiliary_digest,
                system.second_order_oml.source_auxiliary_digest,
            )
            self.assertEqual(
                v20.oml_auxiliary_system_digest(restored.second_order_oml.system),
                v20.oml_auxiliary_system_digest(system.second_order_oml.system),
            )

            synthetic_owner_step(system)
            synthetic_owner_step(restored)
            self.assertEqual(v20.oml_system_digest(restored), v20.oml_system_digest(system))

            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["plan_digest"] = "sha256:" + "0" * 64
            tampered = Path(directory) / "tampered.pt"
            torch.save(payload, tampered)
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaises(RuntimeError):
                    v20.load_oml_checkpoint(tampered, _V19_CHECKPOINT, device="cpu")

    def test_training_loss_source_contains_no_detached_credit_or_identity_input(self) -> None:
        sources = "\n".join(
            inspect.getsource(function)
            for function in (
                v20._row_loss,
                v20._stream_loss,
                v20._inner_step,
                v20._unroll_inner,
            )
        )
        for forbidden in (
            "valid_mask",
            "responsibilities",
            "heldout_index",
            "transition_index",
            "positive_index",
            "negative_index",
            "mechanism_commitment",
            "query_task",
            "evaluator",
        ):
            self.assertNotIn(forbidden, sources)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA tiny parity requires CUDA")
    def test_cpu_cuda_tiny_math_parity(self) -> None:
        cpu_losses = torch.linspace(0.02, 0.31, 8, dtype=torch.float32)
        cuda_losses = cpu_losses.cuda()
        cpu_objective = v20._anonymous_entropic_objective(
            cpu_losses, expected_count=8
        )
        cuda_objective = v20._anonymous_entropic_objective(
            cuda_losses, expected_count=8
        ).cpu()
        self.assertTrue(torch.allclose(cpu_objective, cuda_objective, atol=1.0e-6, rtol=1.0e-6))

        cpu_parameter = torch.tensor((0.3, -0.1), dtype=torch.float32)
        cpu_gradient = torch.tensor((0.2, -0.4), dtype=torch.float32)
        cpu_slot = v20.AdamWSlot(
            step=0,
            exp_avg=torch.zeros_like(cpu_parameter),
            exp_avg_sq=torch.zeros_like(cpu_parameter),
        )
        cpu_updated, _ = v20.functional_adamw_step(
            (cpu_parameter,), (cpu_gradient,), (cpu_slot,), (1.0e-3,)
        )
        cuda_slot = v20.AdamWSlot(
            step=0,
            exp_avg=torch.zeros_like(cpu_parameter).cuda(),
            exp_avg_sq=torch.zeros_like(cpu_parameter).cuda(),
        )
        cuda_updated, _ = v20.functional_adamw_step(
            (cpu_parameter.cuda(),),
            (cpu_gradient.cuda(),),
            (cuda_slot,),
            (1.0e-3,),
        )
        self.assertTrue(
            torch.allclose(cpu_updated[0], cuda_updated[0].cpu(), atol=1.0e-6, rtol=1.0e-6)
        )


if __name__ == "__main__":
    unittest.main()
