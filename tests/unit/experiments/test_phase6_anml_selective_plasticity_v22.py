from __future__ import annotations

import copy
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

from experiments.runners import phase6_oml_relation_representation as v20
from experiments.runners import phase6_v12_champion_paired_graph_context as v19
from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


def _api(*names: str):
    for name in names:
        value = getattr(v22, name, None)
        if value is not None:
            return value
    raise AssertionError(f"V22 runner is missing required API: {' or '.join(names)}")


def _fresh_controller() -> v19.V12ChampionPairedGraphContextController:
    controller = v19.V12ChampionPairedGraphContextController(
        v12.SOFTWARE_PIPELINE_PROFILES["smoke"]
    )
    v20._configure_oml_controller(controller, learn_rln=False)
    return controller


def _gate() -> torch.nn.Module:
    constructor = _api("ANMLNeuromodulator", "SelectivePlasticityGate")
    try:
        return constructor()
    except TypeError:
        return constructor(64)


def _gate_values(module: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    helper = getattr(v22, "centered_gate", None)
    return helper(module, hidden) if helper is not None else module(hidden)


def _hook_context(
    controller: v19.V12ChampionPairedGraphContextController,
    module: torch.nn.Module,
    *,
    lesion: str = "live",
    permutation: torch.Tensor | None = None,
):
    helper = _api("scoped_anml_gate", "_scoped_anml_gate", "scoped_gate_hook")
    parameters = inspect.signature(helper).parameters
    kwargs: dict[str, object] = {}
    if "lesion" in parameters:
        kwargs["lesion"] = lesion
    elif "mode" in parameters:
        kwargs["mode"] = lesion
    if "permutation" in parameters and permutation is not None:
        kwargs["permutation"] = permutation
    return helper(controller, module, **kwargs)


def _fresh_fast_state(initial: torch.Tensor):
    helper = _api("fresh_fast_state", "_fresh_fast_state", "_fresh_anml_fast_state")
    return helper(initial)


def _state_digest(state: object) -> str:
    return _api("fast_state_digest", "anml_fast_state_digest")(state)


def _snapshot_state(state: object) -> dict[str, object]:
    return _api("snapshot_fast_state", "snapshot_anml_fast_state")(state)


def _restore_state(snapshot: dict[str, object]):
    helper = _api("restore_fast_state", "restore_anml_fast_state")
    try:
        return helper(snapshot, device="cpu")
    except TypeError:
        return helper(snapshot, "cpu")


def _classification_gates(**overrides: bool) -> dict[str, bool]:
    gates = {
        "second_beats_first_auc": True,
        "second_beats_open_auc": True,
        "second_beats_first_terminal": True,
        "second_beats_open_terminal": True,
        "second_beats_forward_only": True,
        "second_beats_mean_gate": True,
        "second_beats_permuted_gate": True,
        "panel_direction_supported": True,
        "no_catastrophic_panel": True,
        "terminal_supported": True,
        "retention_supported": True,
        "original_nonregression": True,
        "fully_heldout_improved": True,
        "reset_attribution_supported": True,
        "surface_transfer_supported": True,
        "early_improvement": True,
        "acquisition_improved": True,
    }
    gates.update(overrides)
    return gates


class Phase6ANMLSelectivePlasticityV22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def test_frozen_leaf_and_protocol_literal_are_bound(self) -> None:
        root = Path(__file__).resolve().parents[3]
        leaf = root / (
            "docs/blueprints/branches/learning/work/"
            "ANG-WORK-LEARNING-ANML-SELECTIVE-PLASTICITY-V22-001.md"
        )
        self.assertEqual(
            hashlib.sha256(leaf.read_bytes()).hexdigest().upper(),
            "9E7E4F6D3B57DF66B1FF2E696FC4370397CF2A72DFE17D5168E3B2BB12B7B8EC",
        )
        self.assertEqual(
            v22.PROTOCOL_ID,
            "phase6.public-anml-selective-plasticity.v22",
        )

    def test_neuromodulator_is_exactly_8320_parameters_and_initially_open(self) -> None:
        module = _gate()
        named = dict(module.named_parameters())
        self.assertEqual(sum(value.numel() for value in named.values()), 8_320)
        self.assertTrue(named)
        self.assertTrue(all(value.requires_grad for value in named.values()))
        hidden = torch.linspace(-1.0, 1.0, 4 * 64).reshape(4, 64)
        observed = _gate_values(module, hidden)
        self.assertEqual(observed.shape, hidden.shape)
        self.assertTrue(torch.equal(observed, torch.ones_like(hidden)))
        objective = (observed * hidden.square()).sum()
        gradients = torch.autograd.grad(objective, tuple(named.values()), allow_unused=True)
        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all().item()
                and int(torch.count_nonzero(gradient).item()) > 0
                for gradient in gradients
            )
        )
        controller = _fresh_controller()
        optimizer = v22._make_outer_optimizer(module)
        owned = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(owned, {id(parameter) for parameter in module.parameters()})
        self.assertFalse(owned & {id(parameter) for parameter in controller.parameters()})

    def test_exact_open_hook_matches_v20_and_never_mutates_controller(self) -> None:
        controller = _fresh_controller()
        module = _gate()
        layer = controller.relation_comparator[2]
        hidden = torch.randn(7, 64, generator=torch.Generator().manual_seed(2201))
        before = {name: value.detach().clone() for name, value in controller.state_dict().items()}
        before_hooks = len(layer._forward_pre_hooks)
        expected = layer(hidden)
        with _hook_context(controller, module):
            observed = layer(hidden)
            self.assertEqual(len(layer._forward_pre_hooks), before_hooks + 1)
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(len(layer._forward_pre_hooks), before_hooks)
        for name, value in controller.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
            self.assertFalse(value.requires_grad, name)

    def test_hook_is_removed_after_success_and_injected_error(self) -> None:
        controller = _fresh_controller()
        module = _gate()
        layer = controller.relation_comparator[2]
        baseline = len(layer._forward_pre_hooks)
        with _hook_context(controller, module):
            layer(torch.zeros(2, 64))
        self.assertEqual(len(layer._forward_pre_hooks), baseline)

        class InjectedError(RuntimeError):
            pass

        with self.assertRaisesRegex(InjectedError, "injected V22 failure"):
            with _hook_context(controller, module):
                layer(torch.zeros(2, 64))
                raise InjectedError("injected V22 failure")
        self.assertEqual(len(layer._forward_pre_hooks), baseline)

        fast = layer.weight.detach().clone().requires_grad_(True)
        stream = object()
        original_parameter = layer.weight
        with mock.patch.object(
            v19,
            "public_paired_graph_credit_rows",
            side_effect=InjectedError("injected public-row failure"),
        ):
            with self.assertRaisesRegex(InjectedError, "injected public-row failure"):
                v22.functional_credit_rows(controller, fast, stream, module)
        self.assertEqual(v22.active_gate_hook_count(controller), 0)
        self.assertIs(layer.weight, original_parameter)

    def test_live_mean_open_and_permutation_lesions_are_literal(self) -> None:
        helper = _api("apply_gate_lesion", "_apply_gate_lesion", "gate_lesion")
        gate = torch.linspace(0.2, 1.8, 2 * 64).reshape(2, 64)
        permutation = torch.arange(63, -1, -1)

        def apply(mode: str) -> torch.Tensor:
            parameters = inspect.signature(helper).parameters
            kwargs = {"permutation": permutation} if "permutation" in parameters else {}
            return helper(gate, mode, **kwargs)

        self.assertTrue(torch.equal(apply("live"), gate))
        self.assertTrue(torch.equal(apply("open"), torch.ones_like(gate)))
        self.assertTrue(
            torch.equal(apply("mean"), gate.mean(dim=-1, keepdim=True).expand_as(gate))
        )
        self.assertTrue(torch.equal(apply("permuted"), gate.index_select(-1, permutation)))
        with self.assertRaises((ValueError, RuntimeError)):
            apply("unknown")

    def test_second_order_credit_is_finite_and_detach_removes_only_update_consequence(self) -> None:
        step = _api("functional_adamw_step")
        theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
        initial = torch.tensor((0.35, -0.2), dtype=torch.float64, requires_grad=True)

        def unroll(second_order: bool):
            weight = initial
            state = (
                v20.AdamWSlot(
                    step=0,
                    exp_avg=torch.zeros_like(weight),
                    exp_avg_sq=torch.zeros_like(weight),
                ),
            )
            for index in range(3):
                feature = weight.new_tensor((0.7 - 0.2 * index, -0.4 + 0.3 * index))
                feature = feature + theta * weight.new_tensor((0.3, -0.25))
                loss = 0.5 * ((weight * feature).sum() - 0.1 * index).square()
                (gradient,) = torch.autograd.grad(loss, (weight,), create_graph=second_order)
                if not second_order:
                    gradient = gradient.detach()
                (weight,), state = step(
                    (weight,),
                    (gradient,),
                    state,
                    (0.01,),
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1.0e-8,
                    weight_decay=0.0,
                )
            return weight, state

        second_weight, second_state = unroll(True)
        first_weight, first_state = unroll(False)
        self.assertTrue(torch.equal(second_weight.detach(), first_weight.detach()))
        self.assertTrue(torch.equal(second_state[0].exp_avg.detach(), first_state[0].exp_avg.detach()))
        self.assertTrue(
            torch.equal(second_state[0].exp_avg_sq.detach(), first_state[0].exp_avg_sq.detach())
        )
        second_outer = second_weight.square().sum()
        first_outer = first_weight.square().sum()
        (second_credit,) = torch.autograd.grad(second_outer, (theta,), retain_graph=True)
        (first_credit,) = torch.autograd.grad(first_outer, (theta,), allow_unused=True)
        self.assertTrue(torch.isfinite(second_credit).item())
        self.assertGreater(abs(float(second_credit)), 1.0e-9)
        self.assertIsNone(first_credit)

    def test_production_meta_unroll_has_paired_forward_and_second_order_only_gate_credit(self) -> None:
        second_gate = _gate()
        first_gate = copy.deepcopy(second_gate)
        initial = torch.linspace(-0.2, 0.2, 64, dtype=torch.float32).reshape(1, 64)
        second_system = SimpleNamespace(fast_initial_weight=initial, controller=object())
        first_system = SimpleNamespace(fast_initial_weight=initial, controller=object())
        second_arm = SimpleNamespace(gate=second_gate)
        first_arm = SimpleNamespace(gate=first_gate)

        def fake_stream_loss(_controller, fast, stream, gate_module):
            position = int(stream)
            hidden = torch.linspace(-0.7, 0.9, 64, dtype=fast.dtype).reshape(1, 64)
            hidden = torch.roll(hidden, shifts=position)
            gate_values = v22.centered_gate(gate_module, hidden)
            prediction = (fast * hidden * gate_values).sum()
            return 0.5 * (prediction - fast.new_tensor(0.05 * (position - 3))).square()

        streams = tuple(range(8))
        with mock.patch.object(v22, "_stream_loss", side_effect=fake_stream_loss):
            second_weight, second_state, second_diagnostic = v22._meta_unroll(
                second_system, second_arm, streams, second_order=True
            )
            first_weight, first_state, first_diagnostic = v22._meta_unroll(
                first_system, first_arm, streams, second_order=False
            )
        self.assertTrue(torch.equal(second_weight.detach(), first_weight.detach()))
        self.assertTrue(torch.equal(second_state[0].exp_avg.detach(), first_state[0].exp_avg.detach()))
        self.assertTrue(
            torch.equal(second_state[0].exp_avg_sq.detach(), first_state[0].exp_avg_sq.detach())
        )
        self.assertTrue(
            all(not item["gradient_detached"] for item in second_diagnostic["step_diagnostics"])
        )
        self.assertTrue(
            all(item["gradient_detached"] for item in first_diagnostic["step_diagnostics"])
        )
        target = torch.full_like(second_weight, 0.03)
        second_credit = torch.autograd.grad(
            (second_weight - target).square().sum(),
            tuple(second_gate.parameters()),
            allow_unused=True,
        )
        first_credit = torch.autograd.grad(
            (first_weight - target).square().sum(),
            tuple(first_gate.parameters()),
            allow_unused=True,
        )
        self.assertTrue(
            any(
                gradient is not None
                and torch.isfinite(gradient).all().item()
                and int(torch.count_nonzero(gradient).item()) > 0
                for gradient in second_credit
            )
        )
        self.assertTrue(all(gradient is None for gradient in first_credit))

    def test_fast_state_is_192_fp32_values_and_snapshot_resume_is_exact(self) -> None:
        initial = torch.linspace(-0.2, 0.2, 64, dtype=torch.float32).reshape(1, 64)
        state = _fresh_fast_state(initial)
        tensors = (state.weight, state.optimizer_state[0].exp_avg, state.optimizer_state[0].exp_avg_sq)
        self.assertEqual(sum(value.numel() for value in tensors), 192)
        self.assertEqual(sum(value.numel() * value.element_size() for value in tensors), 768)
        self.assertTrue(all(value.dtype == torch.float32 for value in tensors))
        self.assertTrue(all(not value.requires_grad for value in tensors))
        snapshot = _snapshot_state(state)
        restored = _restore_state(snapshot)
        self.assertEqual(_state_digest(restored), _state_digest(state))
        self.assertIsNot(restored.weight, state.weight)
        tampered = copy.deepcopy(snapshot)
        optimizer = tampered.get("optimizer_state")
        if isinstance(optimizer, dict):
            optimizer["step"] = int(optimizer.get("step", 0)) + 1
        elif isinstance(optimizer, (tuple, list)):
            optimizer[0]["step"] = int(optimizer[0]["step"]) + 1
        else:
            self.fail("V22 fast-state snapshot does not expose its bound optimizer state")
        with self.assertRaises((ValueError, RuntimeError)):
            _restore_state(tampered)

    def test_meta_fit_plan_has_exact_unique_disjoint_240_update_schedule(self) -> None:
        plan = v22.anml_fit_plan()
        self.assertEqual(plan["protocol_id"], v22.PROTOCOL_ID)
        self.assertEqual(plan["optimization"]["outer_updates"], 240)
        self.assertEqual(tuple(plan["meta_fit_indices"]), tuple(range(8, 32)))
        self.assertEqual(tuple(plan["lifetime_indices"]), tuple(range(32, 64)))
        updates = tuple(plan["meta_updates"])
        self.assertEqual(len(updates), 240)
        identities: set[tuple[int, int]] = set()
        counts = {index: 0 for index in range(8, 32)}
        for update_index, update in enumerate(updates):
            target = 8 + update_index % 24
            self.assertEqual(update["target_commitment_index"], target)
            counts[target] += 1
            inner = tuple(update["inner"])
            outer = tuple(update["outer"])
            self.assertEqual((len(inner), len(outer)), (8, 8))
            self.assertEqual(tuple(item["commitment_index"] for item in inner), (target,) * 8)
            self.assertEqual(tuple(item["commitment_index"] for item in outer[:4]), (target,) * 4)
            self.assertEqual(
                tuple(item["commitment_index"] for item in outer[4:]),
                tuple(8 + ((target - 8 + delta) % 24) for delta in (5, 10, 15, 20)),
            )
            for item in inner + outer:
                identity = (item["topology_seed"], item["surface_seed"])
                self.assertNotIn(identity, identities)
                identities.add(identity)
        self.assertEqual(set(counts.values()), {10})
        self.assertEqual(len(identities), 240 * 16)
        self.assertRegex(v22.anml_plan_digest(), r"^sha256:[0-9a-f]{64}$")

    def test_lifetime_orders_are_exact_4096_balanced_and_paired(self) -> None:
        blocked0 = tuple(v22.lifetime_order(0))
        blocked1 = tuple(v22.lifetime_order(1))
        mixed2 = tuple(v22.lifetime_order(2))
        mixed3 = tuple(v22.lifetime_order(3))
        self.assertEqual(blocked0, blocked1)
        self.assertEqual(mixed2, mixed3)
        for order in (blocked0, blocked1, mixed2, mixed3):
            self.assertEqual(len(order), 4_096)
            self.assertEqual(
                {commitment: order.count(commitment) for commitment in range(32, 64)},
                {commitment: 128 for commitment in range(32, 64)},
            )
        self.assertEqual(
            blocked0,
            tuple(commitment for commitment in range(32, 64) for _ in range(128)),
        )
        expected_mixed = tuple(
            32 + ((13 * position + 7 * cycle) % 32)
            for cycle in range(128)
            for position in range(32)
        )
        self.assertEqual(mixed2, expected_mixed)

    def test_lifetime_streams_are_unique_and_disjoint_from_meta_fit(self) -> None:
        plan = v22.anml_fit_plan()
        meta = {
            (item["topology_seed"], item["surface_seed"])
            for update in plan["meta_updates"]
            for item in tuple(update["inner"]) + tuple(update["outer"])
        }
        lifetime: set[tuple[int, int]] = set()
        for panel in range(4):
            specs = tuple(v22._lifetime_record(panel, step) for step in range(4_096))
            self.assertEqual(len(specs), 4_096)
            for step, item in enumerate(specs):
                self.assertEqual(item["commitment_index"], v22.lifetime_order(panel)[step])
                identity = (item["topology_seed"], item["surface_seed"])
                self.assertNotIn(identity, meta)
                self.assertNotIn(identity, lifetime)
                lifetime.add(identity)
        self.assertEqual(len(lifetime), 4 * 4_096)

    def test_duplicate_same_contract_rows_preserve_order_and_hook_cleanup(self) -> None:
        parity = _api("exact_equivalent_feature_parity", "public_feature_parity_report")
        controller = _fresh_controller()
        stream = v12._relation_credit_panel_streams(
            v19.v12_champion_paired_graph_context_plan()["commitments"],
            v19.v12_champion_paired_graph_context_plan()["panel_seed_pairs"][0],
        )[0]
        module = _gate()
        baseline_hooks = len(controller.relation_comparator[2]._forward_pre_hooks)
        report = parity(controller, module, stream, duplicate_same_contract=True)
        for key in ("row_values_exact", "row_order_exact", "row_masks_exact", "loss_exact", "fast_gradient_exact", "adamw_transition_exact"):
            self.assertIs(report[key], True, key)
        self.assertIs(report["duplicate_same_contract_exercised"], True)
        self.assertLessEqual(float(report["maximum_abs_delta"]), 1.0e-6)
        self.assertEqual(len(controller.relation_comparator[2]._forward_pre_hooks), baseline_hooks)

    def test_classification_priority_is_exclusive_and_boundaries_are_inclusive(self) -> None:
        classifier = _api("classify_anml", "_classify_anml")
        supported = _classification_gates()
        self.assertEqual(classifier(supported, True), "ANML_SELECTIVE_PLASTICITY_SUPPORTED")
        self.assertEqual(classifier(supported, False), "INVALID_NO_CLAIM")

        cases = (
            ({"second_beats_forward_only": False}, "STATIC_GATE_ONLY"),
            ({"second_beats_mean_gate": False}, "SELECTIVITY_ATTRIBUTION_NOT_SUPPORTED"),
            ({"terminal_supported": False}, "SHORT_HORIZON_ONLY"),
            ({"retention_supported": False}, "ACQUISITION_RETENTION_TRADEOFF"),
            ({"second_beats_open_auc": False, "early_improvement": False}, "ANML_NOT_SUPPORTED"),
        )
        labels = {"ANML_SELECTIVE_PLASTICITY_SUPPORTED", "INVALID_NO_CLAIM"}
        for changes, expected in cases:
            observed = classifier(_classification_gates(**changes), True)
            self.assertEqual(observed, expected, changes)
            labels.add(observed)
        self.assertEqual(
            labels,
            {
                "INVALID_NO_CLAIM",
                "STATIC_GATE_ONLY",
                "SELECTIVITY_ATTRIBUTION_NOT_SUPPORTED",
                "SHORT_HORIZON_ONLY",
                "ACQUISITION_RETENTION_TRADEOFF",
                "ANML_SELECTIVE_PLASTICITY_SUPPORTED",
                "ANML_NOT_SUPPORTED",
            },
        )
        with self.assertRaises((ValueError, RuntimeError)):
            classifier({"second_beats_open_auc": True}, True)
        with self.assertRaises((ValueError, RuntimeError)):
            classifier(supported, 1)

    def test_metric_gate_arithmetic_uses_exact_frozen_inclusive_thresholds(self) -> None:
        self.assertTrue(v22._loss_improves(95.0, 100.0, 0.05))
        self.assertFalse(
            v22._loss_improves(math.nextafter(95.0, math.inf), 100.0, 0.05)
        )
        self.assertTrue(v22._loss_improves(97.0, 100.0, 0.03))
        self.assertFalse(
            v22._loss_improves(math.nextafter(97.0, math.inf), 100.0, 0.03)
        )

        arms = {
            v22.ARM_SECOND_ORDER: {"loss_auc": 0.95, "terminal_loss": 0.95},
            v22.ARM_FIRST_ORDER: {"loss_auc": 1.0, "terminal_loss": 1.0},
            v22.ARM_ALWAYS_OPEN: {"loss_auc": 1.0, "terminal_loss": 1.0},
            v22.ARM_FORWARD_ONLY: {"loss_auc": 1.0, "terminal_loss": 1.0},
            v22.ARM_MEAN_GATE: {"loss_auc": 1.0, "terminal_loss": 1.0},
            v22.ARM_PERMUTED_GATE: {"loss_auc": 1.0, "terminal_loss": 1.0},
        }
        metrics = {
            "aggregate": {
                "arms": arms,
                "retained_acquisition_fraction": 0.80,
                "original_pre_loss": 1.0,
                "original_terminal_loss": 1.05,
                "fully_heldout_pre_loss": 1.0,
                "fully_heldout_terminal_loss": 0.99,
                "reset_removed_fraction": 0.80,
                "surface_transfer_retained_fraction": 0.80,
                "unseen_pre_loss": 1.0,
                "unseen_terminal_loss": 0.99,
            },
            "panels": tuple(
                {
                    "arms": copy.deepcopy(arms),
                    "probe_milestones": {},
                }
                for _ in range(4)
            ),
        }
        gates = v22.compute_anml_gates(metrics)
        self.assertTrue(all(gates.values()), gates)
        self.assertEqual(
            v22.classify_anml(gates, True),
            "ANML_SELECTIVE_PLASTICITY_SUPPORTED",
        )

        for key, above in (
            ("retained_acquisition_fraction", math.nextafter(0.80, -math.inf)),
            ("reset_removed_fraction", math.nextafter(0.80, -math.inf)),
            ("surface_transfer_retained_fraction", math.nextafter(0.80, -math.inf)),
            ("original_terminal_loss", math.nextafter(1.05, math.inf)),
        ):
            changed = copy.deepcopy(metrics)
            changed["aggregate"][key] = above
            observed = v22.compute_anml_gates(changed)
            expected_gate = {
                "retained_acquisition_fraction": "retention_supported",
                "reset_removed_fraction": "reset_attribution_supported",
                "surface_transfer_retained_fraction": "surface_transfer_supported",
                "original_terminal_loss": "original_nonregression",
            }[key]
            self.assertIs(observed[expected_gate], False, key)

    def test_atomic_json_and_checkpoint_helpers_leave_no_partial_output(self) -> None:
        json_writer = _api("atomic_write_json", "_atomic_write_json")
        checkpoint_writer = _api("atomic_torch_save", "_atomic_torch_save")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            checkpoint_path = root / "checkpoint.pt"
            json_writer(json_path, {"finite": 1.25, "status": "PASS"})
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"finite": 1.25, "status": "PASS"},
            )
            checkpoint_writer(checkpoint_path, {"value": torch.arange(4)})
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(loaded["value"], torch.arange(4)))
            self.assertFalse(any(path.suffix == ".tmp" for path in root.iterdir()))

            with self.assertRaises((ValueError, RuntimeError)):
                json_writer(root / "nonfinite.json", {"value": float("nan")})
            self.assertFalse((root / "nonfinite.json").exists())
            self.assertFalse(any("nonfinite" in path.name for path in root.iterdir()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-specific V22 mechanics require CUDA")
    def test_synthetic_cuda_preflight_has_no_semantic_streams(self) -> None:
        report = v22.synthetic_cuda_preflight("cuda:0")
        self.assertEqual(report["status"], "PASS")
        self.assertIs(report["synthetic_only"], True)
        self.assertEqual(report["semantic_streams_generated"], 0)
        self.assertIs(report["semantic_updates_performed"], False)
        self.assertLessEqual(
            report["maximum_allocated_bytes"], report["allocated_memory_ceiling_bytes"]
        )


if __name__ == "__main__":
    unittest.main()
