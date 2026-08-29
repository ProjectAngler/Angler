from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.evaluators import software_pipeline_reconstruction_suite as suite
from experiments.runners import phase6_cross_variation_plasticity_v16 as v16
from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19
from experiments.runners import phase6_oml_relation_representation as v20
from experiments.runners import phase6_oml_persistent_lifelong_v21 as v21


_ROOT = Path(__file__).resolve().parents[3]
_LEAF = _ROOT / (
    "docs/blueprints/branches/learning/work/"
    "ANG-WORK-LEARNING-OML-PERSISTENT-LIFELONG-V21-001.md"
)
_V19_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
_V20_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.pt"
)
_V20_REPORT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v20-oml.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fresh_controller(device: str | torch.device = "cpu"):
    controller = v19.V12ChampionPairedGraphContextController(
        v12.SOFTWARE_PIPELINE_PROFILES["smoke"]
    ).to(device)
    controller.eval()
    return controller


def _first_public_train_stream():
    plan = v19.v12_champion_paired_graph_context_plan()
    return v12._relation_credit_panel_streams(
        plan["commitments"], plan["panel_seed_pairs"][0]
    )[0]


def _public_supports(stream: suite.SoftwarePipelineStream):
    return tuple(pair.learner for pair in stream.supports)


def _assert_value_exact(
    testcase: unittest.TestCase, left: object, right: object, label: str
) -> None:
    testcase.assertIs(type(left), type(right), label)
    if isinstance(left, torch.Tensor):
        testcase.assertTrue(torch.equal(left, right), label)
    elif isinstance(left, tuple):
        testcase.assertEqual(len(left), len(right), label)
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _assert_value_exact(testcase, left_item, right_item, f"{label}[{index}]")
    else:
        testcase.assertEqual(left, right, label)


def _assert_rows_exact(
    testcase: unittest.TestCase, left: object, right: object
) -> None:
    testcase.assertIs(type(left), tuple)
    testcase.assertIs(type(right), tuple)
    testcase.assertEqual(len(left), 4)
    testcase.assertEqual(len(right), 4)
    for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        testcase.assertIs(type(left_row), v19.V19PairedGraphCreditRow)
        testcase.assertIs(type(right_row), v19.V19PairedGraphCreditRow)
        for field in fields(v19.V19PairedGraphCreditRow):
            _assert_value_exact(
                testcase,
                getattr(left_row, field.name),
                getattr(right_row, field.name),
                f"row {row_index} {field.name}",
            )


def _capture_matcher_inputs(controller, output: list[tuple[torch.Tensor, ...]]):
    original = controller._paired_graph_evidence_read

    def capture(*args, **kwargs):
        output.append(
            tuple(
                value.detach().clone() if isinstance(value, torch.Tensor) else value
                for value in args
            )
        )
        return original(*args, **kwargs)

    return mock.patch.object(controller, "_paired_graph_evidence_read", side_effect=capture)


def _all_row_gradients(controller, rows):
    partition = v20._validate_parameter_partition(controller)
    named = dict(controller.named_parameters())
    names = (v20.FAST_PARAMETER_NAME,) + tuple(partition["rln_parameter_names"])
    for name, parameter in named.items():
        parameter.requires_grad_(name in names)
        parameter.grad = None
    loss = v20._stream_loss_from_rows(rows)
    gradients = torch.autograd.grad(
        loss,
        tuple(named[name] for name in names),
        allow_unused=True,
        retain_graph=False,
    )
    return names, gradients


def _synthetic_system() -> v21.PersistentLifelongSystem:
    candidate = _fresh_controller()
    first = copy.deepcopy(candidate)
    source = copy.deepcopy(candidate)
    candidate_digest = v21._freeze_controller(candidate)
    first_digest = v21._freeze_controller(first)
    source_digest = v21._freeze_controller(source)
    initial = (
        dict(candidate.named_parameters())[v20.FAST_PARAMETER_NAME]
        .detach()
        .clone()
    )

    def arm(name: str, controller, digest: str) -> v21.PersistentArm:
        return v21.PersistentArm(
            name=name,
            controller=controller,
            initial_weight=initial.detach().clone(),
            state=v21._fresh_persistent_fast_state(initial),
            source_controller_digest=digest,
        )

    system = v21.PersistentLifelongSystem(
        second_order_oml_persistent=arm(v21.ARM_CANDIDATE, candidate, candidate_digest),
        first_order_meta_persistent=arm(v21.ARM_FIRST_ORDER, first, first_digest),
        source_v19_persistent=arm(v21.ARM_SOURCE, source, source_digest),
        second_order_boundary_reset=arm(v21.ARM_BOUNDARY, candidate, candidate_digest),
        source_bindings={
            "v20_checkpoint_sha256": v21.SOURCE_V20_CHECKPOINT_SHA256,
            "v20_report_sha256": v21.SOURCE_V20_REPORT_SHA256,
            "v20_terminal_system_digest": v21.SOURCE_V20_SYSTEM_DIGEST,
            "v19_checkpoint_sha256": v21.SOURCE_V19_CHECKPOINT_SHA256,
            "candidate_controller_digest": candidate_digest,
            "first_order_controller_digest": first_digest,
            "source_v19_controller_digest": source_digest,
        },
    )
    v21._assert_system_integrity(system)
    return system


def _cheap_persistent_step(controller, state, supports):
    del controller, supports
    slot = state.optimizer_state[0]
    next_state = v21.PersistentFastState(
        weight=(state.weight + 1.0e-6).detach().clone(),
        optimizer_state=(
            v21.AdamWSlot(
                step=slot.step + 1,
                exp_avg=(slot.exp_avg + 1.0e-7).detach().clone(),
                exp_avg_sq=(slot.exp_avg_sq + 1.0e-8).detach().clone(),
            ),
        ),
        lifetime_updates=state.lifetime_updates + 1,
        reset_count=state.reset_count,
    )
    return next_state, {
        "loss": 0.1,
        "gradient_norm": 0.01,
        "adamw_step": next_state.optimizer_state[0].step,
        "lifetime_updates": next_state.lifetime_updates,
        "state_digest": v21.persistent_fast_state_digest(next_state),
    }


def _synthetic_digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _synthetic_identity_entry(
    role: str,
    index: int,
    *,
    boundaries: tuple[str, ...] = (),
) -> dict[str, object]:
    def values(label: str, count: int) -> tuple[str, ...]:
        return tuple(
            _synthetic_digest("identity", role, index, label, offset)
            for offset in range(count)
        )

    identity = {
        "schedule_digest": _synthetic_digest("schedule", role, index),
        "stream_literal_digest": _synthetic_digest("stream", role, index),
        "support_package_literal_digests": values("support_package", 4),
        "query_package_literal_digests": values("query_package", 1),
        "public_support_literal_digests": values("public_support", 4),
        "public_query_literal_digests": values("public_query", 1),
        "public_support_canonical_digests": values("canonical_support", 4),
    }
    record: dict[str, object] = {"identity": identity}
    if role == "probe":
        record["boundaries"] = boundaries
    return record


def _populate_synthetic_identity_ledger(
    system: v21.PersistentLifelongSystem,
    *,
    diagnostic_count: int,
    update_count: int,
    probe_boundaries: tuple[str, ...],
) -> None:
    system.identity_ledger = {
        "diagnostic": {
            str(index): _synthetic_identity_entry("diagnostic", index)
            for index in range(diagnostic_count)
        },
        "update": {
            str(index): _synthetic_identity_entry("update", index)
            for index in range(update_count)
        },
        "probe": {
            str(index): _synthetic_identity_entry(
                "probe",
                index,
                boundaries=probe_boundaries,
            )
            for index in range(v21.PROBE_STREAMS if probe_boundaries else 0)
        },
    }


def _probe_record(
    group: str,
    *,
    loss: float = 1.0,
    rows: int | None = None,
    streams: int | None = None,
) -> dict[str, object]:
    size = v21.PROBE_GROUP_SIZES[group]
    return {
        "mean_loss": loss,
        "member_losses": (loss,) * size,
        "supported_rows": 3 * size if rows is None else rows,
        "qualifying_streams": size if streams is None else streams,
    }


def _passing_probe_records() -> dict[str, object]:
    cohort = tuple(
        tuple("sha256:" + format(5 * member + offset, "064x") for offset in range(4))
        for member in range(80)
    )
    probes = {
        boundary: {
            "cohort_digest_mode": "order_sensitive_literal_dataclass_fields_v1",
            "cohort_public_digests": cohort,
            "cohort_bytes_exact": True,
            "conditions": {
                condition: {
                    group: _probe_record(group)
                    for group in v21.PROBE_GROUPS
                }
                for condition in v21.MEASUREMENT_CONDITIONS
            }
        }
        for boundary in v21.PROBE_BOUNDARIES
    }

    def put(
        boundary: str,
        condition: str,
        group: str,
        *,
        loss: float,
        rows: int,
        streams: int,
    ) -> None:
        probes[boundary]["conditions"][condition][group] = _probe_record(
            group,
            loss=loss,
            rows=rows,
            streams=streams,
        )

    put("end_A", v21.ARM_CANDIDATE, "stage_a", loss=0.80, rows=180, streams=46)
    put(
        "end_A",
        v21.CONTROL_SECOND_NO_UPDATE,
        "stage_a",
        loss=1.00,
        rows=176,
        streams=44,
    )
    put("end_B", v21.ARM_CANDIDATE, "stage_a", loss=0.84, rows=171, streams=44)
    put(
        "end_B",
        v21.CONTROL_SECOND_NO_UPDATE,
        "stage_a",
        loss=1.00,
        rows=176,
        streams=44,
    )
    put("end_B", v21.ARM_BOUNDARY, "stage_a", loss=1.00, rows=170, streams=43)

    put("end_B", v21.ARM_CANDIDATE, "dev_acquired", loss=0.80, rows=28, streams=8)
    put(
        "end_B",
        v21.CONTROL_SECOND_NO_UPDATE,
        "dev_acquired",
        loss=1.00,
        rows=24,
        streams=6,
    )
    put("end_B", v21.ARM_BOUNDARY, "dev_acquired", loss=0.90, rows=27, streams=7)

    put("end_B", v21.ARM_CANDIDATE, "dev_unseen", loss=0.80, rows=28, streams=7)
    put(
        "end_B",
        v21.CONTROL_SECOND_NO_UPDATE,
        "dev_unseen",
        loss=1.00,
        rows=26,
        streams=6,
    )
    for group in ("original", "v20_heldout"):
        put("end_B", v21.ARM_CANDIDATE, group, loss=1.04, rows=30, streams=8)
        put(
            "end_B",
            v21.CONTROL_SECOND_NO_UPDATE,
            group,
            loss=1.00,
            rows=31,
            streams=8,
        )
    return probes


def _terminal_synthetic_system() -> v21.PersistentLifelongSystem:
    system = _synthetic_system()

    def terminal_state(arm: v21.PersistentArm, *, step: int, reset: int):
        zero = torch.zeros_like(arm.initial_weight)
        arm.state = v21.PersistentFastState(
            weight=arm.initial_weight.detach().clone(),
            optimizer_state=(
                v21.AdamWSlot(step=step, exp_avg=zero, exp_avg_sq=zero.clone()),
            ),
            lifetime_updates=256,
            reset_count=reset,
        )

    terminal_state(system.second_order_oml_persistent, step=256, reset=0)
    terminal_state(system.first_order_meta_persistent, step=256, reset=0)
    terminal_state(system.source_v19_persistent, step=256, reset=0)
    terminal_state(system.second_order_boundary_reset, step=64, reset=1)
    system.next_experience = 256
    system.boundary_reset_applied = True
    system.probes = _passing_probe_records()
    auc_values = {
        v21.ARM_CANDIDATE: 0.80,
        v21.ARM_FIRST_ORDER: 0.95,
        v21.ARM_SOURCE: 0.96,
        v21.ARM_BOUNDARY: 0.85,
        v21.CONTROL_SECOND_NO_UPDATE: 1.00,
        v21.CONTROL_FIRST_NO_UPDATE: 1.00,
        v21.CONTROL_SOURCE_NO_UPDATE: 1.00,
    }
    system.stage_b_online_pre_loss = {
        condition: [value] * 64 for condition, value in auc_values.items()
    }
    system.end_a_exactness = {"state_exact": True, "metrics_exact": True}
    system.gradient_geometry = {
        "descriptive_only": True,
        "learned_state_preserved": True,
        "slow_state_preserved": True,
    }
    _populate_synthetic_identity_ledger(
        system,
        diagnostic_count=v21.DIAGNOSTIC_STREAMS,
        update_count=v21.TOTAL_EXPERIENCES,
        probe_boundaries=v21.PROBE_BOUNDARIES,
    )
    v21._assert_system_integrity(system)
    return system


class Phase6OMLPersistentLifelongV21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_frozen_leaf_and_source_dependency_identities(self) -> None:
        self.assertEqual(
            _sha256(_LEAF),
            "3B6F0875A70094C8B7FF1C5912DD3133BBCF50A6312F60E6B16FE53EAA293A96",
        )
        self.assertEqual(v21.PROTOCOL_ID, "phase6.public-oml-persistent-lifelong.v21a")
        self.assertEqual(
            v21.SOURCE_V20_CHECKPOINT_SHA256,
            "D49E4CAAB64A264A11C675B295A8C453AC4475F078311EB7283A4F9A8817EF48",
        )
        self.assertEqual(v21.SOURCE_V19_CHECKPOINT_SHA256, v20.SOURCE_CHECKPOINT_SHA256)
        expected = {
            "experiments/runners/phase6_oml_relation_representation.py": (
                "6611E60BAB8D1F3C80A68BEB66AAC010F236B107B2A5E9060201BA56A50E86E3"
            ),
            "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
                "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
            ),
            "experiments/runners/phase6_software_pipeline_reconstruction.py": (
                "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
            ),
            "experiments/runners/phase6_cross_variation_plasticity_v16.py": (
                "EB1A29AC78670C6A0ECDED943E17AA62B1CFB91BF58DAB1ADC9001A3B75D63AB"
            ),
        }
        for relative, digest in expected.items():
            self.assertEqual(_sha256(_ROOT / relative), digest, relative)

    def test_plan_has_exact_schedule_balance_and_disjoint_seed_namespaces(self) -> None:
        plan = v21.persistent_lifelong_plan()
        self.assertEqual(plan["protocol_id"], v21.PROTOCOL_ID)
        self.assertRegex(v21.persistent_lifelong_plan_digest(), r"^sha256:[0-9a-f]{64}$")
        stage_a = tuple(v21._experience_spec(index) for index in range(192))
        stage_b = tuple(v21._experience_spec(index) for index in range(192, 256))
        self.assertEqual(
            tuple(record["commitment_index"] for record in stage_a),
            tuple(
                8 + (((index % 48) + 13 * (index // 48)) % 48)
                for index in range(192)
            ),
        )
        self.assertEqual(
            tuple(record["commitment_index"] for record in stage_b),
            tuple(
                2 * ((((index - 192) % 8) + 3 * ((index - 192) // 8)) % 8)
                for index in range(192, 256)
            ),
        )
        self.assertEqual(
            {value: sum(item["commitment_index"] == value for item in stage_a) for value in range(8, 56)},
            {value: 4 for value in range(8, 56)},
        )
        self.assertEqual(
            {value: sum(item["commitment_index"] == value for item in stage_b) for value in range(0, 16, 2)},
            {value: 8 for value in range(0, 16, 2)},
        )

        diagnostics = tuple(v21._diagnostic_specs())
        probes = tuple(v21._probe_specs())
        self.assertEqual((len(diagnostics), len(probes)), (56, 80))
        self.assertEqual(
            tuple(record["commitment_index"] for record in diagnostics[:48]),
            tuple(range(8, 56)),
        )
        self.assertEqual(
            tuple(record["commitment_index"] for record in diagnostics[48:]),
            tuple(range(0, 16, 2)),
        )
        groups = {name: [] for name in ("original", "v20_heldout", "stage_a", "dev_acquired", "dev_unseen")}
        for record in probes:
            groups[record["group"]].append(record["commitment_index"])
        self.assertEqual(tuple(groups["original"]), tuple(range(8)))
        self.assertEqual(tuple(groups["v20_heldout"]), tuple(range(56, 64)))
        self.assertEqual(tuple(groups["stage_a"]), tuple(range(8, 56)))
        self.assertEqual(tuple(groups["dev_acquired"]), tuple(range(0, 16, 2)))
        self.assertEqual(tuple(groups["dev_unseen"]), tuple(range(1, 16, 2)))

        all_records = stage_a + stage_b + diagnostics + probes
        pairs = tuple(
            (int(record["topology_seed"]), int(record["surface_seed"]))
            for record in all_records
        )
        self.assertEqual((len(all_records), len(set(pairs))), (392, 392))
        seed_values = tuple(seed for pair in pairs for seed in pair)
        self.assertEqual((len(seed_values), len(set(seed_values))), (784, 784))
        self.assertGreater(min(seed_values), 30_000_000_000)
        self.assertEqual(plan["distinct_generated_streams"], 392)
        self.assertEqual(plan["distinct_seed_values"], 784)
        self.assertEqual(plan["updated_arms"], v21.UPDATED_ARMS)
        self.assertEqual(plan["stateless_controls"], v21.NO_UPDATE_CONTROLS)
        self.assertEqual(
            plan["persistent_state"],
            {
                "weight_values": 64,
                "adam_first_moment_values": 64,
                "adam_second_moment_values": 64,
                "integer_step": 1,
                "replay_values": 0,
                "boundary_reset_cursor": 192,
                "boundary_reset_count": 1,
            },
        )
        self.assertEqual(
            plan["classification_priority"],
            (
                "INVALID_NO_CLAIM",
                "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED",
                "STAGE_A_NOT_ACQUIRED",
                "FAST_ACQUISITION_WITH_FORGETTING",
                "INHERITED_CAPABILITY_REGRESSION",
                "FAST_ACQUISITION_ATTRIBUTION_NOT_ESTABLISHED",
                "FAST_ACQUISITION_WITHOUT_PERSISTENT_TRANSFER",
                "STATIC_REPRESENTATION_DOMINATES",
                "PERSISTENT_OML_NOT_SUPPORTED",
            ),
        )
        self.assertEqual(
            plan["gates"]["stage_a_retained"],
            {
                "retained_fraction_min": 0.80,
                "coverage_ratio_min": 0.95,
                "paired_retained_ratio": 1.05,
                "paired_retained_min": 43,
                "members": 48,
            },
        )
        self.assertEqual(
            plan["gates"]["oml_fast_attribution"],
            {"normalized_gain_margin_min": 0.02, "candidate_auc_no_higher": True},
        )

    def test_probe_cohort_is_one_boundary_independent_80_stream_identity(self) -> None:
        first = v21._probe_specs()
        second = v21._probe_specs()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 80)
        self.assertTrue(all("boundary" not in item for item in first))
        self.assertEqual(v21.persistent_lifelong_plan()["probe_boundaries"], v21.PROBE_BOUNDARIES)
        self.assertEqual(v21.persistent_lifelong_plan()["probe_cohort"], first)

    def test_generalized_builder_accepts_only_exact_public_support_tuple(self) -> None:
        controller = _fresh_controller()
        stream = _first_public_train_stream()
        supports = _public_supports(stream)
        self.assertTrue(all(type(task) is suite.PublicSoftwarePipelineTask for task in supports))
        rows = v21.public_paired_graph_credit_rows_from_supports(controller, supports)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(type(row) is v19.V19PairedGraphCreditRow for row in rows))
        for invalid in (
            stream,
            list(supports),
            tuple(stream.supports),
            supports + (supports[0],),
            tuple(pair.hidden for pair in stream.supports),
        ):
            with self.assertRaises((TypeError, ValueError)):
                v21.public_paired_graph_credit_rows_from_supports(controller, invalid)

        development = v12.make_software_pipeline_stream(
            9_801_337,
            surface_seed=9_801_339,
            supports_per_motif=2,
            queries=1,
            maximum_steps=4,
            mechanism_commitment=v12.software_pipeline_mechanism_partition(
                "development"
            )[0],
            mechanism_partition="development",
        )
        development_public = _public_supports(development)
        development_rows = v21.public_paired_graph_credit_rows_from_supports(
            controller, development_public
        )
        self.assertEqual(len(development_rows), 4)
        with self.assertRaises((TypeError, ValueError)):
            v21.public_paired_graph_credit_rows_from_supports(
                controller, tuple(pair.hidden for pair in development.supports)
            )

    def test_generalized_builder_is_bit_and_graph_mask_exact_on_frozen_train_fixture(self) -> None:
        stream = _first_public_train_stream()
        supports = _public_supports(stream)
        original_controller = _fresh_controller()
        successor_controller = copy.deepcopy(original_controller)
        original_calls: list[tuple[torch.Tensor, ...]] = []
        successor_calls: list[tuple[torch.Tensor, ...]] = []
        with _capture_matcher_inputs(original_controller, original_calls):
            original_rows = v19.public_paired_graph_credit_rows(original_controller, stream)
        with _capture_matcher_inputs(successor_controller, successor_calls):
            successor_rows = v21.public_paired_graph_credit_rows_from_supports(
                successor_controller, supports
            )
        _assert_rows_exact(self, original_rows, successor_rows)
        self.assertEqual(len(original_calls), len(successor_calls))
        self.assertGreater(len(original_calls), 0)
        for call_index, (left, right) in enumerate(
            zip(original_calls, successor_calls, strict=True)
        ):
            _assert_value_exact(self, left, right, f"matcher call {call_index}")
            self.assertTrue(any(value.dtype is torch.bool for value in left if isinstance(value, torch.Tensor)))
            self.assertTrue(any(value.ndim >= 2 for value in left if isinstance(value, torch.Tensor)))

    def test_generalized_builder_fast_and_all_rln_gradients_are_exact(self) -> None:
        stream = _first_public_train_stream()
        supports = _public_supports(stream)
        original_controller = _fresh_controller()
        successor_controller = copy.deepcopy(original_controller)
        original_rows = v19.public_paired_graph_credit_rows(original_controller, stream)
        successor_rows = v21.public_paired_graph_credit_rows_from_supports(
            successor_controller, supports
        )
        original_names, original_gradients = _all_row_gradients(
            original_controller, original_rows
        )
        successor_names, successor_gradients = _all_row_gradients(
            successor_controller, successor_rows
        )
        self.assertEqual(original_names, successor_names)
        self.assertEqual(len(original_names), 68)
        for name, left, right in zip(
            original_names, original_gradients, successor_gradients, strict=True
        ):
            self.assertEqual(left is None, right is None, name)
            if left is not None:
                self.assertTrue(torch.equal(left, right), name)

    def test_frozen_public_train_parity_report_and_covariance_paths(self) -> None:
        controller = _fresh_controller()
        stream = _first_public_train_stream()
        report = v21.public_train_parity_report(controller, stream)
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["rln_tensor_count"], 67)
        for key in (
            "row_values_exact",
            "row_order_exact",
            "row_masks_exact",
            "loss_exact",
            "fast_gradient_exact",
            "rln_gradients_exact",
        ):
            self.assertIs(report[key], True, key)
        supports = _public_supports(stream)
        for options in (
            {"reverse_evidence_order": True},
            {"reverse_public_presentation": True},
        ):
            legacy = v19.public_paired_graph_credit_rows(controller, stream, **options)
            generalized = v21.public_paired_graph_credit_rows_from_supports(
                controller, supports, **options
            )
            _assert_rows_exact(self, legacy, generalized)

    def test_functional_public_rows_restore_exact_controller_and_fast_parameter(self) -> None:
        controller = _fresh_controller()
        stream = _first_public_train_stream()
        supports = _public_supports(stream)
        before = {
            name: value.detach().clone() for name, value in controller.state_dict().items()
        }
        before_attributes = set(vars(controller))
        fast = before[v20.FAST_PARAMETER_NAME].clone().requires_grad_(True)
        rows = v21._functional_public_rows(controller, fast, supports)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.positive_margin.requires_grad for row in rows))
        self.assertEqual(set(vars(controller)), before_attributes)
        for name, value in controller.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_public_update_objective_has_no_hidden_valid_or_query_dependency(self) -> None:
        objective_source = "\n".join(
            inspect.getsource(function)
            for function in (
                v20._row_loss,
                v20._stream_loss_from_rows,
                v21._public_stream_loss_from_rows,
            )
        )
        for forbidden in (
            "valid_mask",
            "heldout_index",
            "transition_index",
            "positive_index",
            "negative_index",
            "query",
            "partition",
            "commitment",
            "semantic",
            "judge",
        ):
            self.assertNotIn(forbidden, objective_source)

        learned_boundary = "\n".join(
            inspect.getsource(function)
            for function in (
                v21._V21PublicFunctionalAdapter.forward,
                v21._functional_public_rows,
                v21._public_stream_loss,
            )
        )
        for forbidden in (
            ".hidden",
            ".queries",
            "mechanism_commitment",
            "mechanism_partition",
            "commitment_index",
            "semantic_index",
            "required_motifs",
            "judge",
            "reference_pipeline",
            "solution_route",
        ):
            self.assertNotIn(forbidden, learned_boundary)
        signature = inspect.signature(v21.public_paired_graph_credit_rows_from_supports)
        self.assertEqual(tuple(signature.parameters), ("controller", "supports", "reverse_evidence_order", "reverse_public_presentation"))

    def test_fast_state_is_exact_constant_capacity_detached_and_digest_bound(self) -> None:
        initial = torch.linspace(-0.2, 0.2, 64, dtype=torch.float32).reshape(1, 64)
        state = v21._fresh_persistent_fast_state(initial)
        slot = state.optimizer_state[0]
        self.assertEqual(state.persistent_float_values, 192)
        self.assertEqual(v21.PERSISTENT_FLOAT_BYTES, 768)
        self.assertEqual((state.lifetime_updates, state.reset_count, slot.step), (0, 0, 0))
        for tensor in (state.weight, slot.exp_avg, slot.exp_avg_sq):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertTrue(torch.equal(state.weight, initial))
        self.assertTrue(torch.equal(slot.exp_avg, torch.zeros_like(initial)))
        self.assertTrue(torch.equal(slot.exp_avg_sq, torch.zeros_like(initial)))

        digest = v21.persistent_fast_state_digest(state)
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
        changed = v21.PersistentFastState(
            weight=state.weight + 0.001,
            optimizer_state=state.optimizer_state,
            lifetime_updates=state.lifetime_updates,
            reset_count=state.reset_count,
        )
        self.assertNotEqual(digest, v21.persistent_fast_state_digest(changed))
        with self.assertRaises(ValueError):
            v21.PersistentFastState(
                weight=state.weight.clone().requires_grad_(True),
                optimizer_state=state.optimizer_state,
            )
        with self.assertRaises(ValueError):
            v21.PersistentFastState(
                weight=torch.full_like(state.weight, float("nan")),
                optimizer_state=state.optimizer_state,
            )

    def test_fast_state_snapshot_restore_reset_and_tamper_rejection(self) -> None:
        initial = torch.linspace(-0.1, 0.1, 64, dtype=torch.float32).reshape(1, 64)
        state = v21.PersistentFastState(
            weight=(initial + 0.03).detach().clone(),
            optimizer_state=(
                v21.AdamWSlot(
                    step=17,
                    exp_avg=torch.full_like(initial, 0.2),
                    exp_avg_sq=torch.full_like(initial, 0.04),
                ),
            ),
            lifetime_updates=81,
            reset_count=0,
        )
        snapshot = v21.snapshot_fast_state(state)
        restored = v21.restore_fast_state(snapshot, device="cpu")
        self.assertEqual(
            v21.persistent_fast_state_digest(restored),
            v21.persistent_fast_state_digest(state),
        )
        self.assertIsNot(restored.weight, state.weight)
        self.assertTrue(torch.equal(restored.weight, state.weight))
        tampered = copy.deepcopy(snapshot)
        tampered["optimizer_state"]["step"] = 18
        with self.assertRaises(RuntimeError):
            v21.restore_fast_state(tampered, device="cpu")

        reset = v21.reset_fast_state(
            initial,
            lifetime_updates=192,
            reset_count=1,
        )
        reset_slot = reset.optimizer_state[0]
        self.assertEqual(
            (reset.lifetime_updates, reset.reset_count, reset_slot.step),
            (192, 1, 0),
        )
        self.assertTrue(torch.equal(reset.weight, initial))
        self.assertTrue(torch.equal(reset_slot.exp_avg, torch.zeros_like(initial)))
        self.assertTrue(torch.equal(reset_slot.exp_avg_sq, torch.zeros_like(initial)))

    def test_v16_functional_adamw_alias_is_exact_and_state_remains_detached(self) -> None:
        self.assertIs(v21.functional_adamw_step, v16.functional_adamw_step)
        initial = torch.linspace(-0.2, 0.2, 64, dtype=torch.float32).reshape(1, 64)
        state = v21._fresh_persistent_fast_state(initial)
        parameter = state.weight.detach().clone().requires_grad_(True)
        objective = (parameter.square() + 0.1 * parameter).mean()
        (gradient,) = torch.autograd.grad(objective, (parameter,))
        (updated,), optimizer = v21.functional_adamw_step(
            (parameter,),
            (gradient,),
            state.optimizer_state,
            (v21.INNER_LEARNING_RATE,),
            beta1=v21.ADAM_BETA1,
            beta2=v21.ADAM_BETA2,
            epsilon=v21.ADAM_EPSILON,
            weight_decay=v21.ADAM_WEIGHT_DECAY,
        )
        committed = v21.PersistentFastState(
            weight=updated.detach().clone(),
            optimizer_state=(
                v21.AdamWSlot(
                    step=optimizer[0].step,
                    exp_avg=optimizer[0].exp_avg.detach().clone(),
                    exp_avg_sq=optimizer[0].exp_avg_sq.detach().clone(),
                ),
            ),
            lifetime_updates=1,
            reset_count=0,
        )
        self.assertEqual(committed.optimizer_state[0].step, 1)
        for tensor in (
            committed.weight,
            committed.optimizer_state[0].exp_avg,
            committed.optimizer_state[0].exp_avg_sq,
        ):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_functional_adamw_matches_one_ordinary_fp32_step(self) -> None:
        initial = torch.linspace(-0.3, 0.3, 64, dtype=torch.float32).reshape(1, 64)
        gradient = torch.linspace(0.2, -0.2, 64, dtype=torch.float32).reshape(1, 64)
        functional_parameter = initial.clone()
        functional_state = (
            v21.AdamWSlot(
                step=0,
                exp_avg=torch.zeros_like(initial),
                exp_avg_sq=torch.zeros_like(initial),
            ),
        )
        (functional_parameter,), functional_state = v21.functional_adamw_step(
            (functional_parameter,),
            (gradient,),
            functional_state,
            (v21.INNER_LEARNING_RATE,),
            beta1=v21.ADAM_BETA1,
            beta2=v21.ADAM_BETA2,
            epsilon=v21.ADAM_EPSILON,
            weight_decay=v21.ADAM_WEIGHT_DECAY,
        )

        ordinary_parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW(
            (ordinary_parameter,),
            lr=v21.INNER_LEARNING_RATE,
            betas=(v21.ADAM_BETA1, v21.ADAM_BETA2),
            eps=v21.ADAM_EPSILON,
            weight_decay=v21.ADAM_WEIGHT_DECAY,
            foreach=False,
            fused=False,
        )
        ordinary_parameter.grad = gradient.clone()
        optimizer.step()
        ordinary_state = optimizer.state[ordinary_parameter]
        self.assertTrue(torch.equal(functional_parameter, ordinary_parameter.detach()))
        self.assertEqual(functional_state[0].step, int(ordinary_state["step"].item()))
        self.assertTrue(torch.equal(functional_state[0].exp_avg, ordinary_state["exp_avg"]))
        torch.testing.assert_close(
            functional_state[0].exp_avg_sq,
            ordinary_state["exp_avg_sq"],
            rtol=0.0,
            atol=4.0e-12,
        )

        zero_initial = initial.clone()
        zero_state = (
            v21.AdamWSlot(
                step=0,
                exp_avg=torch.zeros_like(initial),
                exp_avg_sq=torch.zeros_like(initial),
            ),
        )
        (zero_updated,), zero_state = v21.functional_adamw_step(
            (zero_initial,),
            (torch.zeros_like(initial),),
            zero_state,
            (v21.INNER_LEARNING_RATE,),
            beta1=v21.ADAM_BETA1,
            beta2=v21.ADAM_BETA2,
            epsilon=v21.ADAM_EPSILON,
            weight_decay=v21.ADAM_WEIGHT_DECAY,
        )
        self.assertTrue(torch.equal(zero_updated, initial))
        self.assertTrue(torch.equal(zero_state[0].exp_avg, torch.zeros_like(initial)))
        self.assertTrue(torch.equal(zero_state[0].exp_avg_sq, torch.zeros_like(initial)))
        self.assertEqual(zero_state[0].step, 1)

        direction = torch.zeros(64, dtype=torch.float32, requires_grad=True)
        parameter = torch.linspace(-0.3, 0.3, 64, dtype=torch.float32)
        slot = v21.AdamWSlot(
            step=0,
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
        )
        updated, _ = v21.functional_adamw_step(
            (parameter,), (direction,), (slot,), (v21.INNER_LEARNING_RATE,)
        )
        (zero_gradient_vjp,) = torch.autograd.grad(
            (updated[0] * torch.linspace(0.5, 1.5, 64)).sum(), (direction,)
        )
        self.assertTrue(torch.equal(updated[0], parameter))
        self.assertTrue(torch.isfinite(zero_gradient_vjp).all().item())

    def test_four_arm_initialization_is_separate_fixed_state_on_frozen_slow_controllers(self) -> None:
        system = _synthetic_system()
        self.assertEqual(tuple(system.stage_b_online_pre_loss), v21.MEASUREMENT_CONDITIONS)
        self.assertEqual(system.next_experience, 0)
        self.assertFalse(system.boundary_reset_applied)
        self.assertIs(
            system.second_order_oml_persistent.controller,
            system.second_order_boundary_reset.controller,
        )
        state_ids = set()
        for name in v21.UPDATED_ARMS:
            arm = system.arm(name)
            self.assertEqual(arm.state.persistent_float_values, 192)
            self.assertEqual(
                (arm.state.lifetime_updates, arm.state.reset_count, arm.state.optimizer_state[0].step),
                (0, 0, 0),
            )
            self.assertTrue(torch.equal(arm.state.weight, arm.initial_weight))
            self.assertFalse(any(parameter.requires_grad for parameter in arm.controller.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in arm.controller.parameters()))
            state_ids.add(id(arm.state.weight))
        self.assertEqual(len(state_ids), 4)

    def test_real_persistent_step_writes_only_detached_fast_state(self) -> None:
        system = _synthetic_system()
        arm = system.second_order_oml_persistent
        supports = _public_supports(_first_public_train_stream())
        controller_before = {
            name: value.detach().clone() for name, value in arm.controller.state_dict().items()
        }
        controller_digest = v20.oml_controller_digest(arm.controller)
        next_state, diagnostic = v21._persistent_step(arm.controller, arm.state, supports)
        self.assertEqual((next_state.lifetime_updates, next_state.optimizer_state[0].step), (1, 1))
        self.assertEqual(next_state.reset_count, 0)
        self.assertTrue(math.isfinite(diagnostic["loss"]))
        self.assertTrue(math.isfinite(diagnostic["gradient_norm"]))
        for tensor in (
            next_state.weight,
            next_state.optimizer_state[0].exp_avg,
            next_state.optimizer_state[0].exp_avg_sq,
        ):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
            self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertEqual(v20.oml_controller_digest(arm.controller), controller_digest)
        for name, value in arm.controller.state_dict().items():
            self.assertTrue(torch.equal(value, controller_before[name]), name)
        self.assertFalse(any(parameter.requires_grad for parameter in arm.controller.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in arm.controller.parameters()))

        with mock.patch.object(
            v21,
            "_public_stream_loss",
            return_value=torch.tensor(float("nan"), requires_grad=True),
        ):
            with self.assertRaises(RuntimeError):
                v21._persistent_step(arm.controller, arm.state, supports)
        self.assertEqual(v20.oml_controller_digest(arm.controller), controller_digest)

    def test_probe_boundary_is_no_gradient_no_write_and_reuses_conditions(self) -> None:
        system = _synthetic_system()
        before = v21.persistent_learned_state_digest(system)
        specs = []
        for group in v21.PROBE_GROUPS:
            specs.append(next(item for item in v21._probe_specs() if item["group"] == group))

        def fake_member(controller, weight, supports):
            del controller, supports
            return {
                "loss": float(weight.detach().mean().item()) + 1.0,
                "supported_rows": 3,
                "informative_rows": 4,
                "qualifying": True,
                "signed_margins": (0.1, 0.2, 0.3, 0.4),
                "row_digest": "sha256:" + "1" * 64,
                "relation_signature_digest": "sha256:" + "2" * 64,
            }, (object(),)

        def fake_aggregate(records, rows):
            del rows
            losses = tuple(float(record["loss"]) for record in records)
            return {
                "mean_loss": sum(losses) / len(losses),
                "member_losses": losses,
                "supported_rows": 3 * len(losses),
                "informative_rows": 4 * len(losses),
                "qualifying_streams": len(losses),
                "signed_margins": tuple(record["signed_margins"] for record in records),
                "row_digests": tuple(record["row_digest"] for record in records),
                "relation_signature_digest": "sha256:" + "2" * 64,
            }

        supports = _public_supports(_first_public_train_stream())
        with mock.patch.object(v21, "_probe_specs", return_value=tuple(specs)), mock.patch.object(
            v21, "_materialize_public_supports", return_value=supports
        ), mock.patch.object(v21, "_probe_member", side_effect=fake_member), mock.patch.object(
            v21, "_aggregate_probe_group", side_effect=fake_aggregate
        ):
            result = v21.evaluate_probe_boundary(system, "pre")
        self.assertEqual(set(result["conditions"]), set(v21.MEASUREMENT_CONDITIONS))
        self.assertTrue(result["learned_state_preserved"])
        self.assertTrue(result["slow_state_preserved"])
        self.assertEqual(before, v21.persistent_learned_state_digest(system))
        self.assertIn("pre", system.probes)
        with self.assertRaises(RuntimeError):
            v21.evaluate_probe_boundary(system, "pre")

    def test_terminal_causal_state_swap_and_true_w0_reset_are_exact(self) -> None:
        system = _terminal_synthetic_system()
        system.probes = {}
        before = v21.persistent_learned_state_digest(system)
        specs = tuple(
            next(item for item in v21._probe_specs() if item["group"] == group)
            for group in v21.PROBE_GROUPS
        )
        supports = _public_supports(_first_public_train_stream())

        def fake_member(controller, weight, observed):
            del controller, observed
            value = float(weight.detach().mean().item()) + 1.0
            return {
                "loss": value,
                "supported_rows": 3,
                "informative_rows": 4,
                "qualifying": True,
                "signed_margins": (0.1, 0.2, 0.3, 0.4),
                "row_digest": "sha256:" + "1" * 64,
                "relation_signature_digest": "sha256:" + "2" * 64,
            }, (object(),)

        def fake_aggregate(records, rows):
            del rows
            losses = tuple(float(record["loss"]) for record in records)
            return {
                "mean_loss": sum(losses) / len(losses),
                "member_losses": losses,
                "supported_rows": 3 * len(losses),
                "informative_rows": 4 * len(losses),
                "qualifying_streams": len(losses),
                "signed_margins": tuple(record["signed_margins"] for record in records),
                "row_digests": tuple(record["row_digest"] for record in records),
                "relation_signature_digest": "sha256:" + "2" * 64,
            }

        with mock.patch.object(v21, "_probe_specs", return_value=specs), mock.patch.object(
            v21, "_materialize_public_supports", return_value=supports
        ), mock.patch.object(v21, "_probe_member", side_effect=fake_member), mock.patch.object(
            v21, "_aggregate_probe_group", side_effect=fake_aggregate
        ):
            result = v21.evaluate_probe_boundary(system, "end_B")
        causal = result["terminal_causal"]
        for key in (
            "state_swap_exact",
            "state_swap_digest_exact",
            "clean_controller_digest_exact",
            "w0_reset_exact",
            "reset_state_valid",
        ):
            self.assertIs(causal[key], True, key)
        self.assertEqual(
            causal["reset_state_counters"],
            {"lifetime_updates": 0, "adamw_step": 0, "reset_count": 0},
        )
        self.assertEqual(before, v21.persistent_learned_state_digest(system))

    def test_gradient_geometry_is_descriptive_no_write_and_non_gating(self) -> None:
        system = _synthetic_system()
        before = v21.persistent_learned_state_digest(system)
        supports = _public_supports(_first_public_train_stream())
        diagnostic_specs = tuple(
            {"index": index, "group": "stage_a" if index < 48 else "dev_acquired"}
            for index in range(56)
        )

        def synthetic_loss(controller, weight, observed):
            del controller, observed
            return (weight * torch.linspace(0.1, 1.0, 64).reshape(1, 64)).sum()

        with mock.patch.object(
            v21, "_diagnostic_specs", return_value=diagnostic_specs
        ), mock.patch.object(
            v21, "_materialize_public_supports", return_value=supports
        ), mock.patch.object(v21, "_public_stream_loss", side_effect=synthetic_loss):
            result = v21.run_initial_gradient_geometry(system)
        self.assertEqual(result["matrix_shape"], (56, 64))
        self.assertIs(result["descriptive_only"], True)
        self.assertIs(result["learned_state_preserved"], True)
        self.assertIs(result["slow_state_preserved"], True)
        self.assertEqual(before, v21.persistent_learned_state_digest(system))
        self.assertNotIn("gradient_geometry", inspect.getsource(v21._compute_gates))
        with self.assertRaises(RuntimeError):
            v21.run_initial_gradient_geometry(system)

    def test_full_chronology_orders_end_a_probe_reset_checkpoint_and_stage_b(self) -> None:
        system = _synthetic_system()
        events: list[tuple[object, ...]] = []
        deadline_calls = 0

        def fake_geometry(target, **kwargs):
            del kwargs
            events.append(("geometry", target.next_experience))
            for record in v21._diagnostic_specs():
                v21._materialize_public_supports(record, None, system=target)
            target.gradient_geometry = {
                "descriptive_only": True,
                "matrix_shape": (56, 64),
                "learned_state_preserved": True,
                "slow_state_preserved": True,
            }
            return copy.deepcopy(target.gradient_geometry)

        def fake_probe(target, boundary, **kwargs):
            del kwargs
            events.append(("probe", boundary, target.next_experience, target.boundary_reset_applied))
            for record in v21._probe_specs():
                v21._materialize_public_supports(
                    record,
                    None,
                    system=target,
                    probe_boundary=boundary,
                )
            same = {"stage_a": {"mean_loss": 1.0}}
            target.probes[boundary] = {
                "boundary": boundary,
                "conditions": {
                    v21.ARM_CANDIDATE: copy.deepcopy(same),
                    v21.ARM_BOUNDARY: copy.deepcopy(same),
                },
            }
            return copy.deepcopy(target.probes[boundary])

        original_apply = v21.apply_persistent_experience

        def observed_apply(target, index, **kwargs):
            events.append(("experience", index, target.boundary_reset_applied))
            return original_apply(target, index)

        def progress(target, event):
            self.assertIs(target, system)
            events.append(("progress", event["cursor"], event["boundary_reset_applied"]))

        def deadline():
            nonlocal deadline_calls
            deadline_calls += 1

        with mock.patch.object(v21, "run_initial_gradient_geometry", side_effect=fake_geometry), mock.patch.object(
            v21, "evaluate_probe_boundary", side_effect=fake_probe
        ), mock.patch.object(v21, "public_train_parity_report", return_value={
            "row_values_exact": True,
            "row_order_exact": True,
            "row_masks_exact": True,
            "loss_exact": True,
            "fast_gradient_exact": True,
            "rln_gradients_exact": True,
        }), mock.patch.object(
            v21, "_persistent_step", side_effect=_cheap_persistent_step
        ), mock.patch.object(
            v21,
            "_public_stream_loss",
            side_effect=lambda controller, weight, observed: weight.sum() * 0.0 + 0.5,
        ), mock.patch.object(v21, "apply_persistent_experience", side_effect=observed_apply):
            fit = v21.fit_persistent_lifelong(
                system,
                progress_callback=progress,
                deadline_callback=deadline,
            )

        self.assertEqual(fit["terminal_cursor"], 256)
        self.assertEqual(system.next_experience, 256)
        self.assertTrue(system.boundary_reset_applied)
        self.assertIs(system.end_a_exactness["state_exact"], True)
        self.assertIs(system.end_a_exactness["metrics_exact"], True)
        candidate = system.second_order_oml_persistent.state
        boundary = system.second_order_boundary_reset.state
        self.assertEqual(
            (candidate.lifetime_updates, candidate.optimizer_state[0].step, candidate.reset_count),
            (256, 256, 0),
        )
        self.assertEqual(
            (boundary.lifetime_updates, boundary.optimizer_state[0].step, boundary.reset_count),
            (256, 64, 1),
        )
        end_a_probe = events.index(("probe", "end_A", 192, False))
        reset_progress = events.index(("progress", 192, True))
        first_stage_b = events.index(("experience", 192, True))
        self.assertLess(end_a_probe, reset_progress)
        self.assertLess(reset_progress, first_stage_b)
        progress_cursors = tuple(
            event[1] for event in events if event[0] == "progress"
        )
        self.assertEqual(progress_cursors, v21.PROGRESS_CURSORS)
        self.assertTrue(all(len(values) == 64 for values in system.stage_b_online_pre_loss.values()))
        self.assertEqual(deadline_calls, 517)

    def test_checkpoint_round_trip_binds_thin_state_rng_harness_and_resume(self) -> None:
        system = _synthetic_system()
        rebuild_template = copy.deepcopy(system)
        supports = _public_supports(_first_public_train_stream())
        for name in v21.UPDATED_ARMS:
            system.arm(name).state, _ = _cheap_persistent_step(
                system.arm(name).controller,
                system.arm(name).state,
                supports,
            )
        system.next_experience = 1
        _populate_synthetic_identity_ledger(
            system,
            diagnostic_count=0,
            update_count=1,
            probe_boundaries=(),
        )
        harness_state = {
            "claim_sha256": "A" * 64,
            "claim_created_utc": "2026-08-29T04:00:00+00:00",
            "identity_deadline_utc": "2026-08-29T04:45:00+00:00",
            "last_identity_age_seconds": 12.5,
            "publication_cursor": 1,
        }
        v21._assert_system_integrity(system)
        expected_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.pt"
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                receipt = v21.save_persistent_lifelong_checkpoint(
                    path,
                    system,
                    harness_state=harness_state,
                )
            self.assertLessEqual(receipt["bytes"], v21.CHECKPOINT_SIZE_CEILING_BYTES)
            self.assertEqual(receipt["cursor"], 1)
            with self.assertRaises(FileExistsError):
                v21.save_persistent_lifelong_checkpoint(path, system)

            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(payload["arms"]), set(v21.UPDATED_ARMS))
            serialized = repr(payload).lower()
            for forbidden in (
                "softwarepipelinestream",
                "publicsoftwarepipelinetask",
                "generatedsoftwarepipelinetask",
                "supports",
                "queries",
                "history",
                "replay",
            ):
                self.assertNotIn(forbidden, serialized)

            torch.manual_seed(12345)
            with mock.patch.object(
                v21,
                "build_persistent_lifelong_system",
                side_effect=lambda *args, **kwargs: copy.deepcopy(rebuild_template),
            ), mock.patch.object(torch.cuda, "is_available", return_value=False):
                restored = v21.load_persistent_lifelong_checkpoint(path, device="cpu")
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertEqual(restored.harness_state, harness_state)
            self.assertEqual(
                v21.persistent_lifelong_system_digest(restored),
                v21.persistent_lifelong_system_digest(system),
            )
            for name in v21.UPDATED_ARMS:
                self.assertEqual(
                    v21.persistent_fast_state_digest(restored.arm(name).state),
                    v21.persistent_fast_state_digest(system.arm(name).state),
                )

            with mock.patch.object(
                v21, "_persistent_step", side_effect=_cheap_persistent_step
            ):
                for target in (system, restored):
                    v21.apply_persistent_experience(target, 1)
            self.assertEqual(
                v21.persistent_lifelong_system_digest(restored),
                v21.persistent_lifelong_system_digest(system),
            )

            tampered_payload = torch.load(path, map_location="cpu", weights_only=True)
            tampered_payload["next_experience"] = 2
            tampered = Path(directory) / "tampered.pt"
            torch.save(tampered_payload, tampered)
            with mock.patch.object(
                v21,
                "build_persistent_lifelong_system",
                side_effect=lambda *args, **kwargs: copy.deepcopy(rebuild_template),
            ), mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaises(RuntimeError):
                    v21.load_persistent_lifelong_checkpoint(tampered, device="cpu")

            terminal = _terminal_synthetic_system()
            terminal_base = copy.deepcopy(terminal)
            for name in v21.UPDATED_ARMS:
                terminal_base.arm(name).state = v21._fresh_persistent_fast_state(
                    terminal_base.arm(name).initial_weight
                )
            terminal_base.next_experience = 0
            terminal_base.boundary_reset_applied = False
            terminal_base.probes = {}
            terminal_base.gradient_geometry = None
            terminal_base.stage_b_online_pre_loss = {
                name: [] for name in v21.MEASUREMENT_CONDITIONS
            }
            terminal_base.end_a_exactness = None
            terminal_base.identity_ledger = {
                "diagnostic": {},
                "update": {},
                "probe": {},
            }
            terminal_path = Path(directory) / "terminal.pt"
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                v21.save_persistent_lifelong_checkpoint(terminal_path, terminal)
            with mock.patch.object(
                v21,
                "build_persistent_lifelong_system",
                side_effect=lambda *args, **kwargs: copy.deepcopy(terminal_base),
            ), mock.patch.object(torch.cuda, "is_available", return_value=False):
                terminal_restored = v21.load_persistent_lifelong_checkpoint(
                    terminal_path, device="cpu"
                )
            self.assertEqual(
                v21.persistent_lifelong_system_digest(terminal_restored),
                v21.persistent_lifelong_system_digest(terminal),
            )

    def test_online_auc_and_paired_arithmetic_are_literal_and_fail_closed(self) -> None:
        values = tuple(float(index) for index in range(64))
        self.assertEqual(v21._online_auc(values), 31.5)
        self.assertEqual(v21._online_auc((0.75,) * 64), 0.75)
        self.assertEqual(v21._paired_counts((1.0, 2.0, 3.0), (1.0, 2.1, 2.0)), 2)
        self.assertEqual(
            v21._paired_counts(
                (1.0, 2.0, 3.0), (1.0, 2.1, 2.0), strict=True
            ),
            1,
        )
        self.assertEqual(
            v21._paired_counts((1.05, 1.051), (1.0, 1.0), ratio=1.05),
            1,
        )
        for invalid in ((1.0,) * 63, (1.0,) * 65, (float("nan"),) * 64):
            with self.assertRaises(ValueError):
                v21._online_auc(invalid)
        for args in (
            (((1.0,), (1.0, 2.0)), {}),
            (((float("inf"),), (1.0,)), {}),
            (((1.0,), (1.0,)), {"ratio": 0.0}),
            (((1.0,), (1.0,)), {"strict": 1}),
        ):
            with self.assertRaises(ValueError):
                v21._paired_counts(*args[0], **args[1])

    def test_terminal_comparisons_and_all_seven_frozen_gates_recompute_from_primitives(self) -> None:
        system = _terminal_synthetic_system()
        comparisons = v21._terminal_comparisons(system)
        self.assertAlmostEqual(comparisons["auc"][v21.ARM_CANDIDATE], 0.8)
        self.assertEqual(comparisons["auc"][v21.CONTROL_SECOND_NO_UPDATE], 1.0)
        self.assertAlmostEqual(
            comparisons["normalized_fast_gain"][v21.ARM_CANDIDATE], 0.2
        )
        self.assertAlmostEqual(
            comparisons["stage_a_retention"]["improvement_end_a"], 0.2
        )
        self.assertAlmostEqual(
            comparisons["stage_a_retention"]["improvement_end_b"], 0.16
        )
        self.assertAlmostEqual(
            comparisons["stage_a_retention"]["retained_fraction"], 0.8
        )
        self.assertEqual(
            comparisons["stage_a_retention"]["paired_retained_1.05"], 48
        )
        gates = v21._compute_gates(comparisons)
        self.assertTrue(all(gates[key] for key in ("S", "A", "T", "R", "U", "B", "N")))
        self.assertIs(gates["static_no_update_competent"], True)

        lesions = {
            "A": (
                "paired_better",
                f"{v21.ARM_CANDIDATE}|{v21.CONTROL_SECOND_NO_UPDATE}|dev_acquired|end_B",
                5,
            ),
            "S": (
                "paired_better",
                f"{v21.ARM_CANDIDATE}|{v21.CONTROL_SECOND_NO_UPDATE}|stage_a|end_A",
                35,
            ),
            "T": ("normalized_fast_gain", v21.ARM_CANDIDATE, 0.06),
            "R": ("stage_a_retention", "paired_retained_1.05", 42),
            "U": (
                "paired_better",
                f"{v21.ARM_CANDIDATE}|{v21.CONTROL_SECOND_NO_UPDATE}|dev_unseen|end_B",
                5,
            ),
            "B": (
                "paired_better",
                f"{v21.ARM_CANDIDATE}|{v21.ARM_BOUNDARY}|stage_a|end_B",
                35,
            ),
            "N": (
                "paired_ratio",
                f"{v21.ARM_CANDIDATE}|{v21.CONTROL_SECOND_NO_UPDATE}|original|end_B|1.05",
                6,
            ),
        }
        for gate, (section, key, value) in lesions.items():
            changed = copy.deepcopy(comparisons)
            changed[section][key] = value
            observed = v21._compute_gates(changed)
            self.assertIs(observed[gate], False, gate)

    def test_gate_operands_and_probe_denominators_fail_closed(self) -> None:
        comparisons = v21._terminal_comparisons(_terminal_synthetic_system())
        invalid = copy.deepcopy(comparisons)
        invalid["auc"][v21.ARM_CANDIDATE] = float("nan")
        with self.assertRaises(ValueError):
            v21._compute_gates(invalid)
        invalid = copy.deepcopy(comparisons)
        del invalid["stage_a_retention"]
        with self.assertRaises(ValueError):
            v21._compute_gates(invalid)
        invalid = copy.deepcopy(comparisons)
        invalid["probes"]["end_B"]["conditions"][v21.ARM_CANDIDATE][
            "dev_unseen"
        ]["member_losses"] = (0.8,) * 7
        with self.assertRaises(ValueError):
            v21._compute_gates(invalid)

        zero_denominator = _terminal_synthetic_system()
        zero_denominator.stage_b_online_pre_loss[
            v21.CONTROL_SECOND_NO_UPDATE
        ] = [0.0] * 64
        with self.assertRaises(ValueError):
            v21._terminal_comparisons(zero_denominator)

    def test_classifier_exhaustively_applies_the_exclusive_priority_tree(self) -> None:
        labels = set()
        for mask in range(1 << 7):
            gates = {
                key: bool(mask & (1 << index))
                for index, key in enumerate(("A", "S", "T", "R", "U", "B", "N"))
            }
            for static in (False, True):
                selected = {**gates, "static_no_update_competent": static}
                A, S, T, R, U, B, N = (
                    selected[key] for key in ("A", "S", "T", "R", "U", "B", "N")
                )
                expected = (
                    "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED"
                    if A and S and T and R and U and B and N
                    else "STAGE_A_NOT_ACQUIRED"
                    if not S
                    else "FAST_ACQUISITION_WITH_FORGETTING"
                    if not R
                    else "INHERITED_CAPABILITY_REGRESSION"
                    if not N
                    else "FAST_ACQUISITION_ATTRIBUTION_NOT_ESTABLISHED"
                    if A and not T
                    else "FAST_ACQUISITION_WITHOUT_PERSISTENT_TRANSFER"
                    if A and T and (not U or not B)
                    else "STATIC_REPRESENTATION_DOMINATES"
                    if not A and static
                    else "PERSISTENT_OML_NOT_SUPPORTED"
                )
                observed = v21._classify_v21a(selected, True)
                self.assertEqual(observed, expected, selected)
                labels.add(observed)
                self.assertEqual(
                    v21._classify_v21a(selected, {"valid": False}),
                    "INVALID_NO_CLAIM",
                )
        self.assertEqual(
            labels,
            {
                "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED",
                "STAGE_A_NOT_ACQUIRED",
                "FAST_ACQUISITION_WITH_FORGETTING",
                "INHERITED_CAPABILITY_REGRESSION",
                "FAST_ACQUISITION_ATTRIBUTION_NOT_ESTABLISHED",
                "FAST_ACQUISITION_WITHOUT_PERSISTENT_TRANSFER",
                "STATIC_REPRESENTATION_DOMINATES",
                "PERSISTENT_OML_NOT_SUPPORTED",
            },
        )
        with self.assertRaises(ValueError):
            v21._classify_v21a({"A": True}, True)
        with self.assertRaises(ValueError):
            v21._classify_v21a(
                {key: True for key in ("A", "S", "T", "R", "U", "B", "N", "static_no_update_competent")},
                {"valid": 1},
            )

    def test_terminal_mechanical_validity_and_report_are_causally_bound(self) -> None:
        system = _terminal_synthetic_system()
        system.public_train_parity = {
            key: True
            for key in (
                "row_values_exact",
                "row_order_exact",
                "row_masks_exact",
                "loss_exact",
                "fast_gradient_exact",
                "rln_gradients_exact",
            )
        }
        system.gradient_geometry = {
            "descriptive_only": True,
            "learned_state_preserved": True,
            "slow_state_preserved": True,
        }
        for boundary in v21.PROBE_BOUNDARIES:
            system.probes[boundary]["learned_state_preserved"] = True
            system.probes[boundary]["slow_state_preserved"] = True
        for condition in v21.NO_UPDATE_CONTROLS:
            terminal_metrics = copy.deepcopy(
                system.probes["end_B"]["conditions"][condition]
            )
            for boundary in ("pre", "end_A"):
                system.probes[boundary]["conditions"][condition] = copy.deepcopy(
                    terminal_metrics
                )
        system.probes["end_B"]["terminal_causal"] = {
            "state_swap_exact": True,
            "state_swap_digest_exact": True,
            "clean_controller_digest_exact": True,
            "w0_reset_exact": True,
            "reset_state_valid": True,
        }
        v21.configure_persistent_lifelong_numerics()
        comparisons = v21._terminal_comparisons(system)
        mechanical = v21._compute_mechanical_validity(system, comparisons)
        self.assertIs(mechanical["valid"], True, mechanical)
        self.assertIs(
            mechanical["checks"]["immutable_probe_cohort_bytes_exact"], True
        )
        self.assertEqual(
            mechanical["arm_counters"][v21.ARM_BOUNDARY], (256, 64, 1)
        )
        self.assertTrue(
            all(value == 192 for value in mechanical["persistent_float_values_per_arm"].values())
        )
        report = v21.evaluate_persistent_lifelong(system)
        self.assertEqual(
            report["classification"],
            "PERSISTENT_OML_TRANSFER_AND_RETENTION_SUPPORTED",
        )
        self.assertIs(report["passed"], True)
        self.assertLessEqual(
            report["terminal_json_bytes_without_size_field"],
            v21.TERMINAL_JSON_SIZE_CEILING_BYTES,
        )
        self.assertEqual(
            report["nonclaims"],
            {
                "no_anml_authority": True,
                "no_replay": True,
                "no_final_or_sealed_access": True,
                "no_promoted_state_mutation": True,
                "no_deployment": True,
            },
        )
        original_cohort = system.probes["end_B"]["cohort_public_digests"]
        changed_cohort = list(original_cohort)
        changed_cohort[-1] = tuple(reversed(changed_cohort[-1]))
        system.probes["end_B"]["cohort_public_digests"] = tuple(changed_cohort)
        invalid = v21._compute_mechanical_validity(system, comparisons)
        self.assertIs(
            invalid["checks"]["immutable_probe_cohort_bytes_exact"], False
        )
        self.assertIs(invalid["valid"], False)
        system.probes["end_B"]["cohort_public_digests"] = original_cohort

    def test_resource_and_static_forbidden_dependency_guards_are_literal(self) -> None:
        system = _synthetic_system()
        self.assertLessEqual(
            v21._assert_resource_ceiling(system), v21.ALLOCATED_MEMORY_CEILING_BYTES
        )
        self.assertEqual(v21.PERSISTENT_FLOAT_VALUES, 192)
        self.assertEqual(v21.PERSISTENT_FLOAT_BYTES, 768)
        self.assertEqual(v21.CHECKPOINT_SIZE_CEILING_BYTES, 16 * 1024**2)
        self.assertEqual(v21.TERMINAL_JSON_SIZE_CEILING_BYTES, 4 * 1024**2)
        self.assertEqual(v21.SEMANTIC_WALL_TIME_CEILING_SECONDS, 45 * 60.0)
        source = inspect.getsource(v21)
        for forbidden in (
            "import requests",
            "import urllib",
            "import subprocess",
            "import socket",
            "torch.utils.data",
            "DataLoader",
            "transformers",
            "final_partition",
            "sealed_partition",
        ):
            self.assertNotIn(forbidden, source)
        step_source = inspect.getsource(v21._persistent_step)
        self.assertNotIn("history", step_source)
        self.assertNotIn("replay", step_source.lower().replace("replay-free", ""))
        self.assertNotIn("query", step_source)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA synthetic preflight requires CUDA")
    def test_cuda_preflight_is_constant_capacity_and_never_enters_semantic_paths(self) -> None:
        with mock.patch.object(
            v21,
            "_make_protocol_stream",
            side_effect=AssertionError("protocol stream forbidden in synthetic preflight"),
        ), mock.patch.object(
            v21,
            "public_paired_graph_credit_rows_from_supports",
            side_effect=AssertionError("public rows forbidden in synthetic preflight"),
        ):
            report = v21.synthetic_cuda_preflight("cuda:0")
        self.assertEqual(report["status"], "PASS")
        self.assertIs(report["synthetic_only"], True)
        self.assertEqual(report["semantic_streams_generated"], 0)
        self.assertIs(report["semantic_updates_performed"], False)
        self.assertIs(report["detach_continuation_exact"], True)
        self.assertIs(report["checkpoint_resume_exact"], True)
        self.assertIs(report["functional_adamw_parity"], True)
        self.assertIs(report["selected_state_constant_capacity"], True)
        self.assertEqual(report["persistent_float_values"], 192)
        self.assertLessEqual(
            report["maximum_allocated_bytes"],
            report["allocated_memory_ceiling_bytes"],
        )

    @unittest.skipUnless(
        _V19_CHECKPOINT.is_file()
        and _V20_CHECKPOINT.is_file()
        and _V20_REPORT.is_file(),
        "frozen V19/V20 artifacts are only present in the execution environment",
    )
    def test_frozen_dependency_verification_and_real_four_arm_load(self) -> None:
        verified = v21.verify_persistent_lifelong_dependencies()
        self.assertEqual(verified["protocol_id"], v21.PROTOCOL_ID)
        self.assertEqual(verified["plan_digest"], v21.persistent_lifelong_plan_digest())
        self.assertEqual(
            verified["source_v20_system_digest"], v21.SOURCE_V20_SYSTEM_DIGEST
        )
        system = v21.build_persistent_lifelong_system(device="cpu")
        self.assertEqual(system.next_experience, 0)
        self.assertFalse(system.boundary_reset_applied)
        common = system.second_order_oml_persistent.initial_weight
        for name in v21.UPDATED_ARMS:
            arm = system.arm(name)
            self.assertTrue(torch.equal(arm.initial_weight, common), name)
            self.assertTrue(torch.equal(arm.state.weight, common), name)
            self.assertFalse(any(parameter.requires_grad for parameter in arm.controller.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in arm.controller.parameters()))
        self.assertIs(
            system.second_order_boundary_reset.controller,
            system.second_order_oml_persistent.controller,
        )


if __name__ == "__main__":
    unittest.main()
