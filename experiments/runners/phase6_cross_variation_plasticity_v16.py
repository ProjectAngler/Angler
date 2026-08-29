"""Numerically stable successor to the frozen V15 plasticity experiment.

V16 changes only fresh model/stream identities and the derivative at an exact
zero in the pure functional AdamW square-root path.  Architecture, exposure,
presentation order, optimizers, evaluation, controls, thresholds, and
classification remain those of V15.  In particular, this module never mutates
the imported V15 module and every actual or virtual cell update calls the local
stable functional implementation below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from experiments.runners import phase6_cross_variation_plasticity as v15
from experiments.runners import phase6_software_pipeline_reconstruction as v13


AdamWSlot = v15.AdamWSlot
CellAdamWState = v15.CellAdamWState
CrossVariationArm = v15.CrossVariationArm
CrossVariationBatch = v15.CrossVariationBatch
CrossVariationEvidence = v15.CrossVariationEvidence
CrossVariationMetaResult = v15.CrossVariationMetaResult
SoftwarePipelineStream = v15.SoftwarePipelineStream

_PROTOCOL_ID = "phase6.public-anonymous-cross-variation-plasticity.paired.v16"
_CHECKPOINT_VERSION = "angler.phase6-cross-variation-plasticity.v2"
_DIGEST_DOMAIN = b"project-angler.cross-variation-plasticity.v2\x00"
_BINDING_DOMAIN = b"angler.v16.transformed-stream-binding.v2\x00"
_SEED_SHIFT = 1_000_000_000
_REPLICATE_SEEDS = (
    (2_026_083_601, 2_026_083_602, 2_026_083_603, 2_026_083_604),
    (2_026_083_611, 2_026_083_612, 2_026_083_613, 2_026_083_614),
    (2_026_083_621, 2_026_083_622, 2_026_083_623, 2_026_083_624),
)
_FROZEN_V15_HASHES = {
    "experiments/runners/phase6_cross_variation_plasticity.py": (
        "C748329ED35055F80EB8859C3A22CDE9D40D59D6FA780766A162EB134711234B"
    ),
    "tests/unit/experiments/test_phase6_cross_variation_plasticity.py": (
        "D2560CC62D5C2031A35BE1CF951E14167CBE789AA8BACF03C86535622C40AA4E"
    ),
}

_CELL_COUNT = 4
_LANE_STREAMS = 4
_STREAMS_PER_UPDATE = 8
_ROWS_PER_STREAM = 4
_UPDATES_PER_ARM = 80
_ROUTER_LOCAL_FEATURES = 7
_ENCODER_LEARNING_RATE = 3.0e-4
_HEAD_LEARNING_RATE = 1.0e-3
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPSILON = 1.0e-8
_ADAM_WEIGHT_DECAY = 0.0
_USAGE_KL_WEIGHT = 0.01


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _frozen_v15_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    observed = {
        relative: _file_sha256(root / relative) for relative in _FROZEN_V15_HASHES
    }
    if observed != _FROZEN_V15_HASHES:
        raise RuntimeError("V16 frozen V15 runner or test bytes changed")
    if Path(v15.__file__).resolve() != (
        root / "experiments/runners/phase6_cross_variation_plasticity.py"
    ).resolve():
        raise RuntimeError("V16 imported an unexpected V15 module")
    return observed


def _shift_seed_pairs(values: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    pairs = tuple((int(value[0]), int(value[1])) for value in values)
    if any(len(value) != 2 for value in values):
        raise ValueError("V16 seed pair shape changed")
    return tuple((topology + _SEED_SHIFT, surface + _SEED_SHIFT) for topology, surface in pairs)


def _transform_update(record: Mapping[str, object]) -> dict[str, object]:
    """Transform only seed pairs while preserving V15 presentation orders."""

    return {
        "procedure_indices": tuple(int(value) for value in record["procedure_indices"]),
        "lane_a_seed_pairs": _shift_seed_pairs(record["lane_a_seed_pairs"]),
        "lane_b_seed_pairs": _shift_seed_pairs(record["lane_b_seed_pairs"]),
        "lane_a_order": tuple(tuple(value) for value in record["lane_a_order"]),
        "lane_b_order": tuple(tuple(value) for value in record["lane_b_order"]),
        "real_order": tuple(tuple(value) for value in record["real_order"]),
    }


def _seed_binding_digest(records: Sequence[object]) -> str:
    digest = hashlib.sha256(_BINDING_DOMAIN)
    digest.update(_canonical_json(records))
    return "sha256:" + digest.hexdigest()


def _plan_seed_pairs(plan: Mapping[str, object]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for specification in plan["replicates"]:
        for key in (
            "panel_a_seed_pairs",
            "panel_a_rerender_seed_pairs",
            "panel_b_seed_pairs",
            "probe_seed_pairs",
        ):
            pairs.update(tuple(value) for value in specification[key])
        for update_key in ("train_updates", "adaptation_updates"):
            for update in specification[update_key]:
                pairs.update(tuple(value) for value in update["lane_a_seed_pairs"])
                pairs.update(tuple(value) for value in update["lane_b_seed_pairs"])
    return {(int(left), int(right)) for left, right in pairs}


def _prior_v12_v14_seed_pairs() -> dict[str, set[tuple[int, int]]]:
    """Collect the exact public seed identities of the three prior protocols."""

    v12_plan = v13.public_relation_conflict_fit_plan()
    v12_pairs = {
        (int(left), int(right))
        for batches in v12_plan["stage_seed_batches"].values()
        for batch in batches
        for left, right in batch
    } | {
        (int(left), int(right))
        for key in ("relation_context_panel_seed_pairs", "final_panel_seed_pairs")
        for left, right in v12_plan[key]
    }
    v13_plan = v13.capacity_matched_relation_cluster_fit_plan()
    v13_pairs = {
        (int(left), int(right))
        for replicate in v13_plan["replicates"]
        for batch in replicate["train_seed_batches"]
        for left, right in batch
    } | {
        (int(left), int(right))
        for replicate in v13_plan["replicates"]
        for key in (
            "panel_a_seed_pairs",
            "panel_a_rerender_seed_pairs",
            "panel_b_seed_pairs",
        )
        for left, right in replicate[key]
    }
    v14_plan = v15.v14.counterfactual_plasticity_fit_plan()
    v14_pairs = {
        (int(left), int(right))
        for replicate in v14_plan["replicates"]
        for batch in replicate["train_seed_batches"]
        for left, right in batch
    } | {
        (int(left), int(right))
        for replicate in v14_plan["replicates"]
        for key in (
            "panel_a_seed_pairs",
            "panel_a_rerender_seed_pairs",
            "panel_b_seed_pairs",
            "adaptation_seed_pairs",
            "probe_seed_pairs",
        )
        for left, right in replicate[key]
    }
    return {"v12": v12_pairs, "v13": v13_pairs, "v14": v14_pairs}


def cross_variation_fit_plan() -> dict[str, object]:
    """Return the fixed V16 plan without constructing streams or models."""

    source = v15.cross_variation_fit_plan()
    if (
        source.get("protocol_id")
        != "phase6.public-anonymous-cross-variation-plasticity.paired.v15"
        or source.get("replicate_count") != 3
        or source.get("updates_per_arm_per_replicate") != _UPDATES_PER_ARM
    ):
        raise RuntimeError("V16 source V15 plan identity changed")
    replicates = []
    for replicate, (source_specification, seeds) in enumerate(
        zip(source["replicates"], _REPLICATE_SEEDS, strict=True)
    ):
        if int(source_specification["replicate"]) != replicate:
            raise RuntimeError("V16 source replicate order changed")
        train_updates = tuple(
            _transform_update(record) for record in source_specification["train_updates"]
        )
        adaptation_updates = tuple(
            _transform_update(record)
            for record in source_specification["adaptation_updates"]
        )
        binding = _seed_binding_digest(train_updates)
        replicates.append(
            {
                "replicate": replicate,
                "shared_controller_seed": seeds[0],
                "cell_seed": seeds[1],
                "composer_seed": seeds[2],
                "router_seed": seeds[3],
                "arm_order": tuple(source_specification["arm_order"]),
                "train_updates": train_updates,
                "adaptation_updates": adaptation_updates,
                "panel_a_seed_pairs": _shift_seed_pairs(
                    source_specification["panel_a_seed_pairs"]
                ),
                "panel_a_rerender_seed_pairs": _shift_seed_pairs(
                    source_specification["panel_a_rerender_seed_pairs"]
                ),
                "panel_b_seed_pairs": _shift_seed_pairs(
                    source_specification["panel_b_seed_pairs"]
                ),
                "probe_seed_pairs": _shift_seed_pairs(
                    source_specification["probe_seed_pairs"]
                ),
                "uniform_stream_binding_digest": binding,
                "learned_stream_binding_digest": binding,
            }
        )
    result = {
        key: value
        for key, value in source.items()
        if key not in {"protocol_id", "replicates", "cell_optimizer"}
    }
    cell_optimizer = dict(source["cell_optimizer"])
    cell_optimizer.update(
        {
            "name": "pure_functional_adamw_stable_zero_vjp_v2",
            "sqrt_input": "exp_avg_sq_clamp_min_dtype_tiny_only_for_sqrt",
        }
    )
    result.update(
        {
            "protocol_id": _PROTOCOL_ID,
            "replicates": tuple(replicates),
            "cell_optimizer": cell_optimizer,
            "successor_of": source["protocol_id"],
            "stream_seed_transform": "add_1000000000_to_each_topology_and_surface_seed",
            "presentation_orders": "copied_byte_for_byte_from_v15",
        }
    )
    source_pairs = _plan_seed_pairs(source)
    transformed_pairs = _plan_seed_pairs(result)
    expected_pairs = {
        (topology + _SEED_SHIFT, surface + _SEED_SHIFT)
        for topology, surface in source_pairs
    }
    if transformed_pairs != expected_pairs or transformed_pairs & source_pairs:
        raise RuntimeError("V16 stream identity transform or isolation changed")
    prior_overlaps = {
        version: transformed_pairs & pairs
        for version, pairs in _prior_v12_v14_seed_pairs().items()
    }
    if any(prior_overlaps.values()):
        raise RuntimeError("V16 transformed stream identities overlap V12-V14")
    for source_specification, transformed_specification in zip(
        source["replicates"], result["replicates"], strict=True
    ):
        for source_update_key in ("train_updates", "adaptation_updates"):
            for before, after in zip(
                source_specification[source_update_key],
                transformed_specification[source_update_key],
                strict=True,
            ):
                for order_key in ("lane_a_order", "lane_b_order", "real_order"):
                    if after[order_key] != before[order_key]:
                        raise RuntimeError("V16 presentation order diverged from V15")
    return result


def cross_variation_plan_digest() -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    digest.update(
        json.dumps(
            cross_variation_fit_plan(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return "sha256:" + digest.hexdigest()


def build_training_batches(
    replicate: int,
    *,
    updates: int | None = None,
) -> tuple[CrossVariationBatch, ...]:
    plan = cross_variation_fit_plan()
    if isinstance(replicate, bool) or not isinstance(replicate, int) or not 0 <= replicate < 3:
        raise ValueError("V16 replicate is outside the fixed plan")
    records = plan["replicates"][replicate]["train_updates"]
    if updates is not None:
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise ValueError("updates must be a positive integer")
        records = records[:updates]
    return tuple(
        v15._make_cross_variation_batch(plan["commitments"], record)
        for record in records
    )


def _adaptation_batches(replicate: int) -> tuple[CrossVariationBatch, ...]:
    plan = cross_variation_fit_plan()
    specification = plan["replicates"][replicate]
    return tuple(
        v15._make_cross_variation_batch(plan["commitments"], record)
        for record in specification["adaptation_updates"]
    )


def build_cross_variation_pair(
    replicate: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CrossVariationArm, CrossVariationArm]:
    if isinstance(replicate, bool) or not isinstance(replicate, int) or not 0 <= replicate < 3:
        raise ValueError("V16 replicate is outside the fixed plan")
    seeds = _REPLICATE_SEEDS[replicate]
    uniform = v15._build_arm(
        shared_seed=seeds[0],
        cell_seed=seeds[1],
        composer_seed=seeds[2],
        router_seed=seeds[3],
        device=device,
    )
    learned = v15._build_arm(
        shared_seed=seeds[0],
        cell_seed=seeds[1],
        composer_seed=seeds[2],
        router_seed=seeds[3],
        device=device,
    )
    if cross_variation_arm_digest(uniform) != cross_variation_arm_digest(learned):
        raise RuntimeError("V16 paired systems lost byte-identical starts")
    return uniform, learned


def functional_adamw_step(
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[torch.Tensor | None],
    state: Sequence[AdamWSlot],
    learning_rates: Sequence[float],
    *,
    beta1: float = _ADAM_BETA1,
    beta2: float = _ADAM_BETA2,
    epsilon: float = _ADAM_EPSILON,
    weight_decay: float = _ADAM_WEIGHT_DECAY,
) -> tuple[tuple[torch.Tensor, ...], tuple[AdamWSlot, ...]]:
    """Apply V15 AdamW exactly, stabilizing only the square-root derivative."""

    if not (len(parameters) == len(gradients) == len(state) == len(learning_rates)):
        raise ValueError("functional AdamW inputs lost parameter alignment")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("functional AdamW beta is invalid")
    if epsilon < 0.0 or weight_decay < 0.0:
        raise ValueError("functional AdamW epsilon or weight decay is invalid")
    updated_parameters = []
    updated_state = []
    for parameter, gradient, slot, learning_rate in zip(
        parameters, gradients, state, learning_rates, strict=True
    ):
        if (
            isinstance(slot.step, bool)
            or not isinstance(slot.step, int)
            or slot.step < 0
            or slot.exp_avg.shape != parameter.shape
            or slot.exp_avg_sq.shape != parameter.shape
            or slot.exp_avg.device != parameter.device
            or slot.exp_avg_sq.device != parameter.device
            or slot.exp_avg.dtype != parameter.dtype
            or slot.exp_avg_sq.dtype != parameter.dtype
            or not parameter.is_floating_point()
            or not bool(torch.isfinite(parameter).all().item())
            or not bool(torch.isfinite(slot.exp_avg).all().item())
            or not bool(torch.isfinite(slot.exp_avg_sq).all().item())
            or not math.isfinite(float(learning_rate))
            or learning_rate < 0.0
        ):
            raise ValueError("functional AdamW state is invalid")
        if gradient is None:
            updated_parameters.append(parameter)
            updated_state.append(slot)
            continue
        if (
            gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or gradient.dtype != parameter.dtype
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise ValueError("functional AdamW gradient is invalid")
        next_step = slot.step + 1
        exp_avg = beta1 * slot.exp_avg + (1.0 - beta1) * gradient
        exp_avg_sq = beta2 * slot.exp_avg_sq + (1.0 - beta2) * gradient.square()
        bias_correction1 = 1.0 - beta1**next_step
        bias_correction2 = 1.0 - beta2**next_step
        step_size = learning_rate / bias_correction1
        # Stored moments remain exact.  Only sqrt sees the clamp, removing the
        # singular backward at exp_avg_sq == 0 without changing live forwards.
        sqrt_input = exp_avg_sq.clamp_min(torch.finfo(exp_avg_sq.dtype).tiny)
        denominator = sqrt_input.sqrt() / math.sqrt(bias_correction2) + epsilon
        decayed = parameter * (1.0 - learning_rate * weight_decay)
        updated_parameters.append(decayed - step_size * exp_avg / denominator)
        updated_state.append(
            AdamWSlot(step=next_step, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        )
    return tuple(updated_parameters), tuple(updated_state)


def _tensor_record(name: str, value: torch.Tensor) -> dict[str, object]:
    detached = value.detach()
    finite = bool(torch.isfinite(detached).all().item())
    nonzero = int(torch.count_nonzero(detached).item())
    magnitude = detached.abs()
    return {
        "name": name,
        "shape": tuple(int(size) for size in detached.shape),
        "dtype": str(detached.dtype),
        "finite": finite,
        "numel": detached.numel(),
        "nonzero": nonzero,
        "zero": detached.numel() - nonzero,
        "max_abs": float(magnitude.max().item()) if detached.numel() else 0.0,
        "fp64_norm": float(detached.to(torch.float64).norm().item()),
    }


def _require_finite_named(
    scope: str,
    values: Sequence[tuple[str, torch.Tensor]],
) -> tuple[dict[str, object], ...]:
    if not values:
        raise RuntimeError(f"V16 {scope} tensor set is empty")
    records = tuple(_tensor_record(name, value) for name, value in values)
    failed = tuple(record["name"] for record in records if record["finite"] is not True)
    if failed:
        raise RuntimeError(f"V16 {scope} contains non-finite tensor(s): {failed}")
    return records


def _named_cell_directions(
    controller: v13.CapacityMatchedClusterController,
    directions: Sequence[Sequence[torch.Tensor]],
) -> tuple[tuple[str, torch.Tensor], ...]:
    groups = v15._cell_parameter_groups(controller)
    if len(directions) != len(groups):
        raise RuntimeError("V16 cell direction count changed")
    named = []
    for cell_index, (group, cell) in enumerate(zip(groups, directions, strict=True)):
        if len(group) != len(cell):
            raise RuntimeError("V16 cell direction parameter alignment changed")
        named.extend(
            (f"cell_{cell_index}.{name}", value)
            for (name, _), value in zip(group, cell, strict=True)
        )
    return tuple(named)


def collect_cross_variation_evidence(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
    cell_optimizer_state: CellAdamWState,
) -> CrossVariationEvidence:
    """Collect V15's anonymous evidence with the stable hypothetical update."""

    if len(cell_optimizer_state) != _CELL_COUNT:
        raise ValueError("V16 evidence requires four cell optimizer states")
    before = v13.software_pipeline_model_digest(controller)
    base = v15._collect_homogeneous_cell_evidence(controller, streams)
    groups = v15._cell_parameter_groups(controller)
    if any(
        len(group) != len(state)
        for group, state in zip(groups, cell_optimizer_state, strict=True)
    ):
        raise ValueError("V16 optimizer state lost cell parameter alignment")
    alignments = torch.zeros_like(base.losses)
    step_norms = torch.zeros_like(base.losses)
    for cell_index, (group, slots) in enumerate(
        zip(groups, cell_optimizer_state, strict=True)
    ):
        parameters = tuple(parameter.detach() for _, parameter in group)
        learning_rates = tuple(v15._parameter_learning_rate(name) for name, _ in group)
        for stream_index, gradients in enumerate(base.gradients[cell_index]):
            _require_finite_named(
                f"evidence cell {cell_index} stream {stream_index} direction",
                tuple(
                    (name, value)
                    for (name, _), value in zip(group, gradients, strict=True)
                ),
            )
            alignments[cell_index, stream_index] = v15._moment_alignment(
                gradients, slots
            )
            updated, _ = functional_adamw_step(
                parameters,
                tuple(value.detach() for value in gradients),
                slots,
                learning_rates,
            )
            square = parameters[0].new_zeros(())
            for parameter, value in zip(parameters, updated, strict=True):
                square = square + (value - parameter).square().sum()
            step_norms[cell_index, stream_index] = square.clamp_min(0.0).sqrt()
    features = torch.cat(
        (
            base.features,
            alignments.detach().unsqueeze(-1),
            torch.log(
                step_norms.detach().clamp_min(torch.finfo(step_norms.dtype).tiny)
            ).unsqueeze(-1),
        ),
        dim=-1,
    ).detach()
    if features.shape != (_CELL_COUNT, len(streams), _ROUTER_LOCAL_FEATURES):
        raise RuntimeError("V16 seven-feature evidence shape changed")
    _require_finite_named("local public evidence", (("features", features),))
    if before != v13.software_pipeline_model_digest(controller):
        raise RuntimeError("V16 evidence mutated controller state")
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V16 evidence populated controller gradient fields")
    return CrossVariationEvidence(base=base, features=features)


def _virtual_adamw_parameters(
    controller: v13.CapacityMatchedClusterController,
    directions: Sequence[Sequence[torch.Tensor]],
    cell_optimizer_state: CellAdamWState,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    _require_finite_named(
        "virtual cell directions", _named_cell_directions(controller, directions)
    )
    groups = v15._cell_parameter_groups(controller)
    updated = []
    for group, gradients, slots in zip(
        groups, directions, cell_optimizer_state, strict=True
    ):
        values, _ = functional_adamw_step(
            tuple(parameter.detach() for _, parameter in group),
            gradients,
            slots,
            tuple(v15._parameter_learning_rate(name) for name, _ in group),
        )
        updated.append(values)
    return tuple(updated)


def cross_variation_meta_gradients(
    controller: v13.CapacityMatchedClusterController,
    router: nn.Module,
    batch: CrossVariationBatch,
    cell_optimizer_state: CellAdamWState,
    evidence: CrossVariationEvidence | None = None,
) -> CrossVariationMetaResult:
    """Differentiate the unchanged symmetric A-to-B/B-to-A V15 objective."""

    if evidence is None:
        evidence = collect_cross_variation_evidence(
            controller, batch.streams, cell_optimizer_state
        )
    before = v13.software_pipeline_model_digest(controller)
    router_before = cross_variation_router_digest(router)
    a_route = v15._lane_allocations(router, evidence, batch.lane_a_indices)
    b_route = v15._lane_allocations(router, evidence, batch.lane_b_indices)
    _require_finite_named(
        "meta routes", (("lane_a", a_route), ("lane_b", b_route))
    )
    a_direction, a_raw, a_clipped = v15._routed_directions(
        evidence, a_route, batch.lane_a_indices
    )
    b_direction, b_raw, b_clipped = v15._routed_directions(
        evidence, b_route, batch.lane_b_indices
    )
    _require_finite_named(
        "meta lane A cell directions",
        _named_cell_directions(controller, a_direction),
    )
    _require_finite_named(
        "meta lane B cell directions",
        _named_cell_directions(controller, b_direction),
    )
    names = evidence.base.cell_parameter_names
    a_virtual = _virtual_adamw_parameters(
        controller, a_direction, cell_optimizer_state
    )
    b_virtual = _virtual_adamw_parameters(
        controller, b_direction, cell_optimizer_state
    )
    lane_a_streams = tuple(batch.streams[index] for index in batch.lane_a_indices)
    lane_b_streams = tuple(batch.streams[index] for index in batch.lane_b_indices)
    post_a_to_b = v15._functional_target_loss(
        controller, lane_b_streams, names, a_virtual
    )
    post_b_to_a = v15._functional_target_loss(
        controller, lane_a_streams, names, b_virtual
    )
    aggregate_usage = torch.cat((a_route, b_route), dim=1).mean(dim=1)
    usage_kl = (
        aggregate_usage
        * torch.log(
            (aggregate_usage * _CELL_COUNT).clamp_min(
                torch.finfo(aggregate_usage.dtype).tiny
            )
        )
    ).sum()
    objective = 0.5 * (post_a_to_b + post_b_to_a) + _USAGE_KL_WEIGHT * usage_kl
    _require_finite_named(
        "meta objective",
        (
            ("objective", objective),
            ("post_a_to_b", post_a_to_b),
            ("post_b_to_a", post_b_to_a),
            ("usage_kl", usage_kl),
        ),
    )
    router_parameters = tuple(router.parameters())
    raw_gradients = torch.autograd.grad(
        objective,
        router_parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    gradients = tuple(value.detach() for value in raw_gradients)
    gradient_records = _require_finite_named(
        "meta gradients",
        tuple(
            (name, value)
            for (name, _), value in zip(
                router.named_parameters(), gradients, strict=True
            )
        ),
    )
    with torch.no_grad():
        pre_a = v13._relation_credit_stream_objective(
            torch.stack(
                tuple(v15._v15_stream_objective(controller, stream) for stream in lane_a_streams)
            ),
            stage="relation",
        )[0]
        pre_b = v13._relation_credit_stream_objective(
            torch.stack(
                tuple(v15._v15_stream_objective(controller, stream) for stream in lane_b_streams)
            ),
            stage="relation",
        )[0]
    _require_finite_named("meta pre-update diagnostics", (("pre_a", pre_a), ("pre_b", pre_b)))
    if before != v13.software_pipeline_model_digest(controller):
        raise RuntimeError("V16 virtual folds mutated controller state")
    if router_before != cross_variation_router_digest(router):
        raise RuntimeError("V16 virtual folds mutated router state")
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V16 meta-loss populated controller gradient fields")
    if any(parameter.grad is not None for parameter in router.parameters()):
        raise RuntimeError("V16 meta-loss populated router gradient fields")
    return CrossVariationMetaResult(
        gradients=gradients,
        objective=float(objective.detach().item()),
        post_losses=(float(post_a_to_b.detach().item()), float(post_b_to_a.detach().item())),
        pre_losses=(float(pre_b.item()), float(pre_a.item())),
        aggregate_usage_kl=float(usage_kl.detach().item()),
        lane_a_allocations=tuple(
            tuple(float(value) for value in row) for row in a_route.detach().tolist()
        ),
        lane_b_allocations=tuple(
            tuple(float(value) for value in row) for row in b_route.detach().tolist()
        ),
        fold_direction_norms=(
            tuple(float(value.detach().item()) for value in a_raw),
            tuple(float(value.detach().item()) for value in b_raw),
        ),
        fold_clipped_direction_norms=(
            tuple(float(value.detach().item()) for value in a_clipped),
            tuple(float(value.detach().item()) for value in b_clipped),
        ),
        parameter_gradient_norms=tuple(
            (str(record["name"]), float(record["fp64_norm"]))
            for record in gradient_records
        ),
    )


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    v15._update_digest(
        digest,
        {
            "type": type(module).__name__,
            "state": module.state_dict(),
        },
    )
    return "sha256:" + digest.hexdigest()


def _object_digest(value: object) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    v15._update_digest(digest, value)
    return "sha256:" + digest.hexdigest()


def cross_variation_router_digest(router: nn.Module) -> str:
    if (
        not isinstance(router, v15.v14.CounterfactualPlasticityRouter)
        or router.local_features != _ROUTER_LOCAL_FEATURES
        or router.hidden_width != 48
    ):
        raise TypeError("V16 router architecture changed")
    return _module_digest(router)


def cross_variation_arm_digest(arm: CrossVariationArm) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    v15._update_digest(
        digest,
        (
            v13.software_pipeline_model_digest(arm.controller),
            cross_variation_router_digest(arm.router),
            v15._state_payload(arm.cell_optimizer_state),
            arm.composer_optimizer.state_dict(),
            arm.router_optimizer.state_dict(),
        ),
    )
    return "sha256:" + digest.hexdigest()


def _component_digests(arm: CrossVariationArm) -> dict[str, str]:
    return {
        "controller": v13.software_pipeline_model_digest(arm.controller),
        "router": cross_variation_router_digest(arm.router),
        "cell_optimizer": _object_digest(v15._state_payload(arm.cell_optimizer_state)),
        "composer_optimizer": _object_digest(arm.composer_optimizer.state_dict()),
        "router_optimizer": _object_digest(arm.router_optimizer.state_dict()),
        "arm": cross_variation_arm_digest(arm),
    }


def _apply_cell_adamw_update(
    arm: CrossVariationArm,
    directions: Sequence[Sequence[torch.Tensor]],
) -> None:
    """Commit one cell update only after every owned direction is finite."""

    _require_finite_named(
        "committed cell directions",
        _named_cell_directions(arm.controller, directions),
    )
    groups = v15._cell_parameter_groups(arm.controller)
    next_state = []
    with torch.no_grad():
        for group, gradients, slots in zip(
            groups, directions, arm.cell_optimizer_state, strict=True
        ):
            updated, updated_slots = functional_adamw_step(
                tuple(parameter.detach() for _, parameter in group),
                tuple(value.detach() for value in gradients),
                slots,
                tuple(v15._parameter_learning_rate(name) for name, _ in group),
            )
            _require_finite_named(
                "committed cell parameters",
                tuple(
                    (name, value)
                    for (name, _), value in zip(group, updated, strict=True)
                ),
            )
            for (_, parameter), value in zip(group, updated, strict=True):
                parameter.copy_(value)
            next_state.append(
                tuple(
                    AdamWSlot(
                        step=slot.step,
                        exp_avg=slot.exp_avg.detach(),
                        exp_avg_sq=slot.exp_avg_sq.detach(),
                    )
                    for slot in updated_slots
                )
            )
    arm.cell_optimizer_state = tuple(next_state)


def _apply_owned_optimizer_gradients(
    optimizer: torch.optim.Optimizer,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    gradients: Sequence[torch.Tensor],
    *,
    scope: str,
) -> None:
    parameters = tuple(parameter for _, parameter in named_parameters)
    if len(parameters) != len(gradients):
        raise RuntimeError(f"V16 {scope} optimizer gradient ownership changed")
    _require_finite_named(
        f"{scope} gradients",
        tuple(
            (name, gradient)
            for (name, _), gradient in zip(named_parameters, gradients, strict=True)
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _require_finite_named(
        f"{scope} parameters",
        tuple((name, parameter.detach()) for name, parameter in named_parameters),
    )


def _exact_uniform(value: torch.Tensor) -> bool:
    return bool(torch.equal(value, torch.full_like(value, 1.0 / _CELL_COUNT)))


def _router_gradient_diagnostics(
    router: nn.Module,
    gradients: Sequence[torch.Tensor],
) -> dict[str, object]:
    records = _require_finite_named(
        "router meta gradients",
        tuple(
            (name, value)
            for (name, _), value in zip(
                router.named_parameters(), gradients, strict=True
            )
        ),
    )
    scorer = next(
        (record for record in records if record["name"] == "scorer.weight"), None
    )
    if scorer is None:
        raise RuntimeError("V16 router scorer gradient is missing")
    upstream = tuple(
        record for record in records if record["name"] != "scorer.weight"
    )
    return {
        "parameters": records,
        "scorer": scorer,
        "upstream_exact_zero": all(record["nonzero"] == 0 for record in upstream),
        "upstream_nonzero": sum(int(record["nonzero"]) for record in upstream),
        "upstream_max_abs": max(
            (float(record["max_abs"]) for record in upstream), default=0.0
        ),
    }


def fit_cross_variation_batches(
    arm: CrossVariationArm,
    batches: Sequence[CrossVariationBatch],
    *,
    learned_plasticity: bool,
) -> dict[str, object]:
    """Fit one arm with stable actual/virtual AdamW and finite-before-step guards."""

    if type(learned_plasticity) is not bool:
        raise TypeError("learned_plasticity must be bool")
    if not batches or any(len(batch.streams) != _STREAMS_PER_UPDATE for batch in batches):
        raise ValueError("V16 fit requires nonempty eight-stream batches")
    mutable_prefixes = ("relation_cells.", "relation_composer.")
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in arm.controller.named_parameters()
        if not name.startswith(mutable_prefixes)
    }
    initial_router = cross_variation_router_digest(arm.router)
    reports = []
    for update_index, batch in enumerate(batches):
        evidence = collect_cross_variation_evidence(
            arm.controller, batch.streams, arm.cell_optimizer_state
        )
        allocations, raw_a_route, raw_b_route = v15._combined_allocations(
            evidence,
            batch,
            arm.router,
            learned_plasticity=learned_plasticity,
        )
        _require_finite_named(
            "fit allocations",
            (
                ("combined", allocations),
                ("lane_a", raw_a_route),
                ("lane_b", raw_b_route),
            ),
        )
        directions, raw_norms, clipped_norms = v15._routed_directions(
            evidence, allocations, tuple(range(_STREAMS_PER_UPDATE))
        )
        direction_records = _require_finite_named(
            "fit cell directions",
            _named_cell_directions(arm.controller, directions),
        )
        meta = cross_variation_meta_gradients(
            arm.controller,
            arm.router,
            batch,
            arm.cell_optimizer_state,
            evidence,
        )
        gradient_diagnostics = _router_gradient_diagnostics(arm.router, meta.gradients)
        composer_gradients, objective, stream_losses, composer_gradient_norm = (
            v15._composer_gradients(arm.controller, batch.streams)
        )
        composer_named = tuple(arm.controller.relation_composer.named_parameters())
        composer_records = _require_finite_named(
            "fit composer gradients",
            tuple(
                (name, value)
                for (name, _), value in zip(
                    composer_named, composer_gradients, strict=True
                )
            ),
        )
        if update_index == 0:
            if not (
                _exact_uniform(allocations)
                and _exact_uniform(raw_a_route)
                and _exact_uniform(raw_b_route)
            ):
                raise RuntimeError("V16 update-zero route is not exactly uniform")
            if gradient_diagnostics["upstream_exact_zero"] is not True:
                raise RuntimeError("V16 update-zero upstream VJP is not exactly zero")

        # No owner is stepped until directions, meta gradients, and composer
        # gradients above have all passed their precise finite checks.
        _apply_cell_adamw_update(arm, directions)
        _apply_owned_optimizer_gradients(
            arm.composer_optimizer,
            composer_named,
            composer_gradients,
            scope="composer",
        )
        _apply_owned_optimizer_gradients(
            arm.router_optimizer,
            tuple(arm.router.named_parameters()),
            meta.gradients,
            scope="router",
        )
        if any(parameter.grad is not None for parameter in arm.controller.parameters()):
            raise RuntimeError("V16 fit leaked controller gradient fields")
        a_tv_mean, a_tv_median = v15._route_statistics(raw_a_route)
        b_tv_mean, b_tv_median = v15._route_statistics(raw_b_route)
        paired_distance, unpaired_distance = v15._route_pair_distances(
            raw_a_route,
            batch.lane_a_slots,
            raw_b_route,
            batch.lane_b_slots,
        )
        reports.append(
            {
                "update": update_index,
                "objective": objective,
                "stream_losses": stream_losses,
                "applied_allocations": tuple(
                    tuple(float(value) for value in row)
                    for row in allocations.detach().tolist()
                ),
                "lane_a_route": meta.lane_a_allocations,
                "lane_b_route": meta.lane_b_allocations,
                "lane_a_slots": batch.lane_a_slots,
                "lane_b_slots": batch.lane_b_slots,
                "route_tv_mean": 0.5 * (a_tv_mean + b_tv_mean),
                "route_tv_median": 0.5 * (a_tv_median + b_tv_median),
                "matched_route_distance": paired_distance,
                "unmatched_route_distance": unpaired_distance,
                "cell_direction_norms": tuple(float(value.item()) for value in raw_norms),
                "cell_clipped_direction_norms": tuple(
                    float(value.item()) for value in clipped_norms
                ),
                "cell_direction_parameters": direction_records,
                "composer_gradient_norm": composer_gradient_norm,
                "composer_gradient_parameters": composer_records,
                "meta": {
                    "objective": meta.objective,
                    "direct_post_update_losses": meta.post_losses,
                    "pre_update_losses_record_only": meta.pre_losses,
                    "aggregate_usage_kl": meta.aggregate_usage_kl,
                    "fold_direction_norms": meta.fold_direction_norms,
                    "fold_clipped_direction_norms": meta.fold_clipped_direction_norms,
                    "parameter_gradient_norms": meta.parameter_gradient_norms,
                    "gradient_diagnostics": gradient_diagnostics,
                },
                "diagnostic_update1_route_movement": (
                    {
                        "lane_a_tv_mean": a_tv_mean,
                        "lane_b_tv_mean": b_tv_mean,
                        "maximum": max(a_tv_mean, b_tv_mean),
                    }
                    if update_index == 1
                    else None
                ),
            }
        )
    named = dict(arm.controller.named_parameters())
    for name, before in frozen.items():
        if not torch.equal(before, named[name].detach()):
            raise RuntimeError(f"V16 fit changed frozen parameter: {name}")
    terminal_router = cross_variation_router_digest(arm.router)
    return {
        "arm": (
            "learned_episodic_plasticity"
            if learned_plasticity
            else "uniform_adamw_plasticity"
        ),
        "optimizer_steps": len(batches),
        "streams": len(batches) * _STREAMS_PER_UPDATE,
        "rows": len(batches) * _STREAMS_PER_UPDATE * _ROWS_PER_STREAM,
        "virtual_folds_per_update": 2,
        "updates": tuple(reports),
        "first_allocation_exact_uniform": all(
            value == 0.25 for row in reports[0]["applied_allocations"] for value in row
        ),
        "update0_upstream_exact_zero": reports[0]["meta"]["gradient_diagnostics"][
            "upstream_exact_zero"
        ],
        "update0_scorer_gradient_diagnostic": reports[0]["meta"][
            "gradient_diagnostics"
        ]["scorer"],
        "update1_route_movement_diagnostic": (
            reports[1]["diagnostic_update1_route_movement"] if len(reports) > 1 else None
        ),
        "cell_update": "pure_functional_adamw_stable_zero_vjp_v2",
        "virtual_cell_update": "pure_functional_adamw_stable_zero_vjp_v2",
        "actual_and_virtual_cell_update_identical": True,
        "composer_update": "separately_owned_adamw",
        "router_update": "separately_owned_adamw",
        "router_scores_applied": learned_plasticity,
        "sham_router_compute_matched": not learned_plasticity,
        "router_affects_current_batch": False,
        "router_affects_next_batch": learned_plasticity,
        "router_initial_digest": initial_router,
        "router_terminal_digest": terminal_router,
        "router_changed": initial_router != terminal_router,
        "frozen_controller_parameters_unchanged": True,
        "controller_grad_fields_clear": True,
    }


def structural_preflight(
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Exercise every fresh update-zero closure without performing an update.

    The preflight builds only the first public training batch for each
    replicate.  It executes evidence, routed directions, symmetric meta VJP,
    and composer VJP for both twins, then proves every owned state digest is
    unchanged.  It does not step an optimizer, evaluate a panel, classify a
    result, or write a checkpoint.
    """

    resolved_device = torch.device(device)
    plan = cross_variation_fit_plan()
    reports = []
    for replicate in range(3):
        uniform, learned = build_cross_variation_pair(
            replicate, device=resolved_device
        )
        batch = build_training_batches(replicate, updates=1)[0]
        arm_reports = []
        closure_digests = []
        for label, arm, learned_plasticity in (
            ("uniform_adamw_plasticity", uniform, False),
            ("learned_episodic_plasticity", learned, True),
        ):
            before = _component_digests(arm)
            controller_parameters = _require_finite_named(
                "preflight controller parameters",
                tuple(arm.controller.named_parameters()),
            )
            router_parameters = _require_finite_named(
                "preflight router parameters",
                tuple(arm.router.named_parameters()),
            )
            evidence = collect_cross_variation_evidence(
                arm.controller, batch.streams, arm.cell_optimizer_state
            )
            allocations, raw_a_route, raw_b_route = v15._combined_allocations(
                evidence,
                batch,
                arm.router,
                learned_plasticity=learned_plasticity,
            )
            _require_finite_named(
                "preflight routes",
                (
                    ("combined", allocations),
                    ("lane_a", raw_a_route),
                    ("lane_b", raw_b_route),
                ),
            )
            directions, raw_norms, clipped_norms = v15._routed_directions(
                evidence, allocations, tuple(range(_STREAMS_PER_UPDATE))
            )
            direction_records = _require_finite_named(
                "preflight routed cell directions",
                _named_cell_directions(arm.controller, directions),
            )
            meta = cross_variation_meta_gradients(
                arm.controller,
                arm.router,
                batch,
                arm.cell_optimizer_state,
                evidence,
            )
            meta_records = _require_finite_named(
                "preflight meta gradients",
                tuple(
                    (name, value)
                    for (name, _), value in zip(
                        arm.router.named_parameters(), meta.gradients, strict=True
                    )
                ),
            )
            gradient_diagnostics = _router_gradient_diagnostics(
                arm.router, meta.gradients
            )
            composer_gradients, objective, stream_losses, composer_norm = (
                v15._composer_gradients(arm.controller, batch.streams)
            )
            composer_records = _require_finite_named(
                "preflight composer gradients",
                tuple(
                    (name, value)
                    for (name, _), value in zip(
                        arm.controller.relation_composer.named_parameters(),
                        composer_gradients,
                        strict=True,
                    )
                ),
            )
            if not (
                _exact_uniform(allocations)
                and _exact_uniform(raw_a_route)
                and _exact_uniform(raw_b_route)
            ):
                raise RuntimeError("V16 preflight update-zero route is not exact uniform")
            if gradient_diagnostics["upstream_exact_zero"] is not True:
                raise RuntimeError("V16 preflight upstream VJP is not exactly zero")
            after = _component_digests(arm)
            if after != before:
                raise RuntimeError(f"V16 preflight mutated {label} replicate {replicate}")
            closure_digest = _object_digest(
                (
                    evidence.features,
                    allocations,
                    directions,
                    meta.gradients,
                    composer_gradients,
                )
            )
            closure_digests.append(closure_digest)
            arm_reports.append(
                {
                    "arm": label,
                    "controller_parameters": controller_parameters,
                    "router_parameters": router_parameters,
                    "routes": {
                        "combined_exact_uniform": _exact_uniform(allocations),
                        "lane_a_exact_uniform": _exact_uniform(raw_a_route),
                        "lane_b_exact_uniform": _exact_uniform(raw_b_route),
                        "combined": tuple(
                            tuple(float(value) for value in row)
                            for row in allocations.detach().tolist()
                        ),
                    },
                    "routed_direction_parameters": direction_records,
                    "routed_direction_zero_count": sum(
                        int(record["zero"]) for record in direction_records
                    ),
                    "routed_direction_nonzero_count": sum(
                        int(record["nonzero"]) for record in direction_records
                    ),
                    "routed_direction_norms": tuple(
                        float(value.item()) for value in raw_norms
                    ),
                    "routed_direction_clipped_norms": tuple(
                        float(value.item()) for value in clipped_norms
                    ),
                    "meta_gradient_parameters": meta_records,
                    "meta_gradient_diagnostics": gradient_diagnostics,
                    "composer_gradient_parameters": composer_records,
                    "composer_objective": objective,
                    "composer_stream_losses": stream_losses,
                    "composer_gradient_norm": composer_norm,
                    "before_digests": before,
                    "after_digests": after,
                    "closure_digest": closure_digest,
                    "optimizer_steps": 0,
                    "evaluation_performed": False,
                }
            )
        if closure_digests[0] != closure_digests[1]:
            raise RuntimeError(f"V16 preflight twins diverged in replicate {replicate}")
        reports.append(
            {
                "replicate": replicate,
                "twins_exact": closure_digests[0] == closure_digests[1],
                "arms": tuple(arm_reports),
            }
        )
    return {
        "protocol_id": _PROTOCOL_ID,
        "kind": "NO_UPDATE_STRUCTURAL_PREFLIGHT",
        "device": str(resolved_device),
        "plan_digest": cross_variation_plan_digest(),
        "replicates": tuple(reports),
        "replicate_count": 3,
        "arms_checked": 6,
        "update_zero_batches_only": True,
        "optimizer_steps": 0,
        "evaluation_performed": False,
        "classification_performed": False,
        "checkpoint_written": False,
        "all_routes_exact_uniform": True,
        "all_upstream_meta_gradients_exact_zero": True,
        "all_tensors_finite": True,
        "all_before_after_digests_exact": True,
    }


def _copy_cross_variation_arm(arm: CrossVariationArm) -> CrossVariationArm:
    clone = v15._copy_cross_variation_arm(arm)
    if cross_variation_arm_digest(clone) != cross_variation_arm_digest(arm):
        raise RuntimeError("V16 complete arm clone changed lineage")
    return clone


def _adaptation_diagnostic(
    trained: CrossVariationArm,
    batches: Sequence[CrossVariationBatch],
    probe_streams: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    """Run V15's fixed four-update adaptation controls with stable AdamW."""

    if len(batches) != 4 or len(probe_streams) != 8:
        raise ValueError("V16 adaptation diagnostic shape changed")
    before = cross_variation_arm_digest(trained)
    arms = {
        label: _copy_cross_variation_arm(trained)
        for label in ("correct", "uniform", "cell_permuted", "no_update")
    }
    reports: dict[str, object] = {}
    for label, arm in arms.items():
        initial_composer = _module_digest(arm.controller.relation_composer)
        initial_router = cross_variation_router_digest(arm.router)
        initial_steps = v15._cell_state_steps(arm.cell_optimizer_state)
        updates = []
        if label != "no_update":
            for update_index, batch in enumerate(batches):
                evidence = collect_cross_variation_evidence(
                    arm.controller, batch.streams, arm.cell_optimizer_state
                )
                combined, _, _ = v15._combined_allocations(
                    evidence, batch, arm.router, learned_plasticity=True
                )
                if label == "uniform":
                    allocation = torch.full_like(combined, 1.0 / _CELL_COUNT)
                elif label == "cell_permuted":
                    allocation = torch.roll(combined, shifts=1, dims=0)
                else:
                    allocation = combined
                directions, raw_norms, clipped_norms = v15._routed_directions(
                    evidence, allocation, tuple(range(_STREAMS_PER_UPDATE))
                )
                direction_records = _require_finite_named(
                    "adaptation cell directions",
                    _named_cell_directions(arm.controller, directions),
                )
                _apply_cell_adamw_update(arm, directions)
                updates.append(
                    {
                        "update": update_index,
                        "allocations": tuple(
                            tuple(float(value) for value in row)
                            for row in allocation.detach().tolist()
                        ),
                        "direction_norms": tuple(
                            float(value.item()) for value in raw_norms
                        ),
                        "clipped_direction_norms": tuple(
                            float(value.item()) for value in clipped_norms
                        ),
                        "direction_parameters": direction_records,
                    }
                )
        panel = v13.evaluate_public_relation_credit_panel(arm.controller, probe_streams)
        if (
            _module_digest(arm.controller.relation_composer) != initial_composer
            or cross_variation_router_digest(arm.router) != initial_router
        ):
            raise RuntimeError("V16 adaptation stepped composer or router")
        terminal_steps = v15._cell_state_steps(arm.cell_optimizer_state)
        expected_increment = 0 if label == "no_update" else len(batches)
        if any(
            after != initial + expected_increment
            for before_cell, after_cell in zip(
                initial_steps, terminal_steps, strict=True
            )
            for initial, after in zip(before_cell, after_cell, strict=True)
        ):
            raise RuntimeError("V16 adaptation did not continue exact AdamW state")
        reports[label] = {
            "updates": tuple(updates),
            "probe": v15._single_panel_metrics(panel),
            "initial_cell_steps": initial_steps,
            "terminal_cell_steps": terminal_steps,
            "composer_stepped": False,
            "router_stepped": False,
        }
    if before != cross_variation_arm_digest(trained):
        raise RuntimeError("V16 adaptation mutated the trained lineage")
    return {
        "arms": reports,
        "updates": 4,
        "same_terminal_clone": True,
        "same_experience": True,
        "same_probe_streams": True,
        "cell_permutation": "global_one_cell_cycle",
        "composer_stepped": False,
        "router_stepped": False,
    }


def _arm_checkpoint_record(arm: CrossVariationArm) -> dict[str, object]:
    return {
        "controller_state": {
            name: value.detach().cpu().clone()
            for name, value in arm.controller.state_dict().items()
        },
        "router_state": {
            name: value.detach().cpu().clone()
            for name, value in arm.router.state_dict().items()
        },
        "cell_optimizer_state": v15._state_payload(arm.cell_optimizer_state),
        "composer_optimizer_state": v15._to_cpu(arm.composer_optimizer.state_dict()),
        "router_optimizer_state": v15._to_cpu(arm.router_optimizer.state_dict()),
        "digest": cross_variation_arm_digest(arm),
    }


def save_cross_variation_checkpoint(
    path: str | Path,
    systems: Sequence[tuple[CrossVariationArm, CrossVariationArm]],
) -> None:
    if len(systems) != len(_REPLICATE_SEEDS):
        raise ValueError("V16 checkpoint requires all three replicates")
    records = tuple(
        {
            "replicate": replicate,
            "uniform": _arm_checkpoint_record(uniform),
            "learned": _arm_checkpoint_record(learned),
        }
        for replicate, (uniform, learned) in enumerate(systems)
    )
    torch.save(
        {
            "version": _CHECKPOINT_VERSION,
            "protocol_id": _PROTOCOL_ID,
            "digest_version": "v2",
            "plan": cross_variation_fit_plan(),
            "plan_digest": cross_variation_plan_digest(),
            "replicates": records,
        },
        Path(path),
    )


def _restore_arm_record(arm: CrossVariationArm, record: object) -> None:
    if not isinstance(record, dict):
        raise RuntimeError("V16 checkpoint arm record is invalid")
    arm.controller.load_state_dict(record["controller_state"], strict=True)
    arm.router.load_state_dict(record["router_state"], strict=True)
    arm.cell_optimizer_state = v15._state_from_payload(
        record["cell_optimizer_state"], arm.controller
    )
    arm.composer_optimizer.load_state_dict(record["composer_optimizer_state"])
    arm.router_optimizer.load_state_dict(record["router_optimizer_state"])
    if record.get("digest") != cross_variation_arm_digest(arm):
        raise RuntimeError("V16 checkpoint lineage changed")


def load_cross_variation_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[tuple[CrossVariationArm, CrossVariationArm], ...]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _CHECKPOINT_VERSION
        or payload.get("protocol_id") != _PROTOCOL_ID
        or payload.get("digest_version") != "v2"
        or payload.get("plan") != cross_variation_fit_plan()
        or payload.get("plan_digest") != cross_variation_plan_digest()
    ):
        raise RuntimeError("V16 checkpoint identity or seed plan is invalid")
    records = payload.get("replicates")
    if not isinstance(records, (tuple, list)) or len(records) != 3:
        raise RuntimeError("V16 checkpoint replicate set changed")
    systems = []
    for replicate, record in enumerate(records):
        if not isinstance(record, dict) or record.get("replicate") != replicate:
            raise RuntimeError("V16 checkpoint replicate identity changed")
        uniform, learned = build_cross_variation_pair(replicate, device=device)
        _restore_arm_record(uniform, record["uniform"])
        _restore_arm_record(learned, record["learned"])
        uniform.controller.eval()
        learned.controller.eval()
        uniform.router.eval()
        learned.router.eval()
        systems.append((uniform, learned))
    return tuple(systems)


def classify_cross_variation_result(
    *,
    integrity_passed: bool,
    uniform_competent: bool,
    uniform_materially_better: bool,
    learned_competent: bool,
    every_support_rule_passed: bool,
) -> str:
    """Apply V15's frozen classification precedence without alteration."""

    return v15.classify_cross_variation_result(
        integrity_passed=integrity_passed,
        uniform_competent=uniform_competent,
        uniform_materially_better=uniform_materially_better,
        learned_competent=learned_competent,
        every_support_rule_passed=every_support_rule_passed,
    )


def _emit_arm_progress(
    callback: Callable[[Mapping[str, object]], None] | None,
    *,
    replicate: int,
    arm: str,
    completed_arms: int,
    started: float,
) -> None:
    if callback is None:
        return
    callback(
        MappingProxyType(
            {
                "protocol_id": _PROTOCOL_ID,
                "event": "ARM_BOUNDARY_COMPLETE",
                "replicate": replicate,
                "arm": arm,
                "completed_arms": completed_arms,
                "total_arms": 6,
                "optimizer_steps": _UPDATES_PER_ARM,
                "streams": _UPDATES_PER_ARM * _STREAMS_PER_UPDATE,
                "rows": _UPDATES_PER_ARM
                * _STREAMS_PER_UPDATE
                * _ROWS_PER_STREAM,
                "elapsed_seconds": time.perf_counter() - started,
                "adaptive_metric_included": False,
            }
        )
    )


def fit_cross_variation_pilot(
    *,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run the one fixed V16 experiment on deterministic single-thread CPU."""

    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise ValueError("V16 semantic fit is fixed to single-thread CPU")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        preflight = structural_preflight(resolved_device)
        frozen_hashes = {
            **v15._frozen_dependency_hashes(),
            **_frozen_v15_hashes(),
        }
        plan = cross_variation_fit_plan()
        commitments = plan["commitments"]
        systems: list[tuple[CrossVariationArm, CrossVariationArm]] = []
        replicate_reports: list[dict[str, object]] = []
        completed_arms = 0
        for specification in plan["replicates"]:
            replicate = int(specification["replicate"])
            uniform, learned = build_cross_variation_pair(
                replicate, device=resolved_device
            )
            systems.append((uniform, learned))
            initial_uniform = cross_variation_arm_digest(uniform)
            initial_learned = cross_variation_arm_digest(learned)
            batches = tuple(
                v15._make_cross_variation_batch(commitments, record)
                for record in specification["train_updates"]
            )
            fits: dict[str, object] = {}
            for label in specification["arm_order"]:
                if label == "uniform_adamw_plasticity":
                    fits[label] = fit_cross_variation_batches(
                        uniform, batches, learned_plasticity=False
                    )
                elif label == "learned_episodic_plasticity":
                    fits[label] = fit_cross_variation_batches(
                        learned, batches, learned_plasticity=True
                    )
                else:
                    raise RuntimeError("V16 arm order changed")
                completed_arms += 1
                _emit_arm_progress(
                    progress_callback,
                    replicate=replicate,
                    arm=label,
                    completed_arms=completed_arms,
                    started=started,
                )
            panel_a = v13._relation_credit_panel_streams(
                commitments, specification["panel_a_seed_pairs"]
            )
            panel_a_rerender = v13._relation_credit_panel_streams(
                commitments, specification["panel_a_rerender_seed_pairs"]
            )
            panel_b = v13._relation_credit_panel_streams(
                commitments, specification["panel_b_seed_pairs"]
            )
            probe = v13._relation_credit_panel_streams(
                commitments, specification["probe_seed_pairs"]
            )
            evaluations = {
                "uniform_adamw_plasticity": v15._paired_evaluation(
                    uniform.controller, panel_a, panel_a_rerender, panel_b
                ),
                "learned_episodic_plasticity": v15._paired_evaluation(
                    learned.controller, panel_a, panel_a_rerender, panel_b
                ),
            }
            adaptation = _adaptation_diagnostic(
                learned, _adaptation_batches(replicate), probe
            )
            specialization = v15._specialization_metrics(
                learned.controller, panel_a + panel_b
            )
            replicate_reports.append(
                {
                    "replicate": replicate,
                    "arm_order": specification["arm_order"],
                    "paired_start_exact": initial_uniform == initial_learned,
                    "paired_exposure_exact": specification[
                        "uniform_stream_binding_digest"
                    ]
                    == specification["learned_stream_binding_digest"],
                    "initial_digests": {
                        "uniform": initial_uniform,
                        "learned": initial_learned,
                    },
                    "terminal_digests": {
                        "uniform": cross_variation_arm_digest(uniform),
                        "learned": cross_variation_arm_digest(learned),
                    },
                    "fits": fits,
                    "evaluations": evaluations,
                    "adaptation": adaptation,
                    "specialization": specialization,
                    "_systems": (uniform, learned),
                }
            )
        if checkpoint_path is not None:
            save_cross_variation_checkpoint(checkpoint_path, systems)
        support_checks, diagnostics = v15._build_support_checks(replicate_reports)
        status = classify_cross_variation_result(
            integrity_passed=bool(support_checks["integrity"]),
            uniform_competent=bool(diagnostics["uniform_competent"]),
            uniform_materially_better=bool(
                diagnostics["uniform_materially_better"]
            ),
            learned_competent=bool(
                support_checks["absolute_learned_competence"]
            ),
            every_support_rule_passed=all(
                bool(value) for value in support_checks.values()
            ),
        )
        for report in replicate_reports:
            del report["_systems"]
        return {
            "protocol_id": _PROTOCOL_ID,
            "status": status,
            "plasticity_router_supported": status
            == "PLASTICITY_ROUTER_SUPPORTED",
            "plan": plan,
            "plan_digest": cross_variation_plan_digest(),
            "frozen_dependency_hashes": frozen_hashes,
            "structural_preflight": preflight,
            "replicates": tuple(replicate_reports),
            "aggregate": diagnostics,
            "support_checks": support_checks,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_written": checkpoint_path is not None,
            "semantic_fit_device": "cpu",
            "semantic_fit_threads": 1,
            "context_or_joint_training_performed": False,
            "development_or_final_access": False,
            "wrong_evidence_training_streams": 0,
            "stored_examples_or_replay": 0,
            "scalar_judge_calls": 0,
            "deterministic_solver_used": False,
            "result_conditioned_continuation": False,
        }
    finally:
        torch.set_num_threads(previous_threads)


__all__ = [
    "AdamWSlot",
    "CrossVariationArm",
    "CrossVariationBatch",
    "CrossVariationEvidence",
    "CrossVariationMetaResult",
    "build_cross_variation_pair",
    "build_training_batches",
    "classify_cross_variation_result",
    "collect_cross_variation_evidence",
    "cross_variation_arm_digest",
    "cross_variation_fit_plan",
    "cross_variation_meta_gradients",
    "cross_variation_plan_digest",
    "cross_variation_router_digest",
    "fit_cross_variation_batches",
    "fit_cross_variation_pilot",
    "functional_adamw_step",
    "load_cross_variation_checkpoint",
    "save_cross_variation_checkpoint",
    "structural_preflight",
]
