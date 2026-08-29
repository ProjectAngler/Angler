"""Small development-only ANML implementation-fidelity diagnostic.

This module never mutates the sealed V20/V22 checkpoints.  It compares the
fast-optimizer and meta-unroll choices on fresh public development identities
and publishes scalar evidence only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from experiments.runners import phase6_anml_selective_plasticity_v22 as v22


PROTOCOL_ID = "phase6.public-anml-fidelity.v23-d1"
V22_CHECKPOINT = Path("/opt/angler/results/phase6-anml-selective-plasticity-v22.pt")
V22_CHECKPOINT_SHA256 = "37DACF2F27092EED51473757C3C6EC6631D7DA407A480F4C3E58EEA6B58F54B4"

OUTER_UPDATES = 48
OUTER_STREAMS = 8
MAX_INNER_STEPS = 20
LIFETIME_UPDATES = 512
PROBE_MILESTONES = (0, 128, 512)
PILOT_COMMITMENTS = tuple(range(32, 48))
PROBE_COMMITMENTS = tuple(range(64))
FAST_LR = 1.0e-3
MEMORY_CEILING_BYTES = 2 * 1024**3
WALL_CEILING_SECONDS = 90 * 60
META_SEED_BASE = 81_000_000_001
LIFETIME_SEED_BASE = 85_000_000_001
PROBE_SEED_BASE = 89_000_000_001
D0_SEED_BASE = 93_000_000_001


@dataclass(frozen=True, slots=True)
class PilotConfig:
    name: str
    fast_optimizer: str
    inner_steps: int


CONFIGS = (
    PilotConfig("adamw_8", "adamw", 8),
    PilotConfig("sgd_8", "sgd", 8),
    PilotConfig("adamw_20", "adamw", 20),
    PilotConfig("sgd_20", "sgd", 20),
)


@dataclass(slots=True)
class LearnedPair:
    config: PilotConfig
    second_gate: v22.ANMLNeuromodulator
    first_gate: v22.ANMLNeuromodulator
    second_outer: torch.optim.Optimizer
    first_outer: torch.optim.Optimizer


@dataclass(frozen=True, slots=True)
class FastState:
    weight: torch.Tensor
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"V23-D1 {label} is non-finite")
    return result


def _flat(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("V23-D1 cannot flatten an empty tensor sequence")
    return torch.cat(tuple(value.detach().reshape(-1).to(torch.float64) for value in tensors))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm().item() * right.norm().item())
    return float(torch.dot(left, right).item() / denominator) if denominator > 0.0 else 0.0


def _relative_improvement(candidate: float, baseline: float) -> float:
    candidate = _finite(candidate, "candidate metric")
    baseline = _finite(baseline, "baseline metric")
    if baseline <= 0.0:
        raise RuntimeError("V23-D1 baseline metric must be positive")
    return (baseline - candidate) / baseline


def _stream_record(
    *, role: int, update: int, position: int, commitment_index: int, base: int
) -> dict[str, int]:
    if commitment_index not in range(64):
        raise ValueError("V23-D1 commitment is outside the public partition")
    seed = base + 10_000_000 * role + 100_000 * update + 1_000 * position
    return {
        "commitment_index": commitment_index,
        "topology_seed": seed,
        "surface_seed": seed + 500_000_000,
    }


def meta_records(update: int) -> dict[str, tuple[dict[str, int], ...]]:
    if type(update) is not int or not 0 <= update < OUTER_UPDATES:
        raise ValueError("V23-D1 meta update is outside the frozen pilot")
    target = 8 + update % 24
    inner = tuple(
        _stream_record(
            role=0,
            update=update,
            position=position,
            commitment_index=target,
            base=META_SEED_BASE,
        )
        for position in range(MAX_INNER_STEPS)
    )
    current = tuple(
        _stream_record(
            role=1,
            update=update,
            position=position,
            commitment_index=target,
            base=META_SEED_BASE,
        )
        for position in range(4)
    )
    remember = tuple(
        _stream_record(
            role=2,
            update=update,
            position=position,
            commitment_index=8 + ((target - 8 + offset) % 24),
            base=META_SEED_BASE,
        )
        for position, offset in enumerate((5, 10, 15, 20))
    )
    records = inner + current + remember
    identities = {(item["topology_seed"], item["surface_seed"]) for item in records}
    if len(identities) != len(records):
        raise RuntimeError("V23-D1 meta identities collided")
    return {"inner": inner, "outer": current + remember}


def lifetime_order(panel: int) -> tuple[int, ...]:
    if panel == 0:
        result = tuple(index for index in PILOT_COMMITMENTS for _ in range(32))
    elif panel == 1:
        result = tuple(
            32 + ((5 * position + 3 * cycle) % 16)
            for cycle in range(32)
            for position in range(16)
        )
    else:
        raise ValueError("V23-D1 lifetime panel must be zero or one")
    if len(result) != LIFETIME_UPDATES or any(result.count(index) != 32 for index in PILOT_COMMITMENTS):
        raise RuntimeError("V23-D1 lifetime order is unbalanced")
    return result


def lifetime_record(panel: int, step: int) -> dict[str, int]:
    if type(step) is not int or not 0 <= step < LIFETIME_UPDATES:
        raise ValueError("V23-D1 lifetime step is invalid")
    return _stream_record(
        role=panel,
        update=step,
        position=0,
        commitment_index=lifetime_order(panel)[step],
        base=LIFETIME_SEED_BASE + 2_000_000_000 * panel,
    )


def probe_records(panel: int) -> tuple[dict[str, int], ...]:
    if panel not in (0, 1):
        raise ValueError("V23-D1 probe panel must be zero or one")
    return tuple(
        _stream_record(
            role=panel,
            update=index,
            position=0,
            commitment_index=index,
            base=PROBE_SEED_BASE + 2_000_000_000 * panel,
        )
        for index in PROBE_COMMITMENTS
    )


def _make_stream(record: Mapping[str, int]):
    commitments = v22.v12.software_pipeline_mechanism_partition("train")[:64]
    return v22.v12.make_software_pipeline_stream(
        int(record["topology_seed"]),
        surface_seed=int(record["surface_seed"]),
        supports_per_motif=2,
        queries=1,
        maximum_steps=4,
        mechanism_commitment=commitments[int(record["commitment_index"])],
        mechanism_partition="train",
    )


def _bundles(
    system: v22.ANMLSystem, records: Iterable[Mapping[str, int]]
) -> tuple[v22.ANMLFeatureBundle, ...]:
    return tuple(
        v22.capture_feature_bundle(
            system.controller, system.fast_initial_weight, _make_stream(record)
        )
        for record in records
    )


def _new_adam_slot(weight: torch.Tensor) -> tuple[v22.AdamWSlot, ...]:
    zero = torch.zeros_like(weight)
    return (v22.AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)


def _functional_unroll(
    system: v22.ANMLSystem,
    gate: v22.ANMLNeuromodulator,
    streams: Sequence[v22.ANMLFeatureBundle],
    *,
    optimizer_name: str,
    second_order: bool,
) -> torch.Tensor:
    if optimizer_name not in ("adamw", "sgd"):
        raise ValueError("V23-D1 fast optimizer is invalid")
    fast = system.fast_initial_weight.detach().clone().requires_grad_(True)
    slots = _new_adam_slot(fast)
    for stream in streams:
        loss = v22._stream_loss(system.controller, fast, stream, gate)
        gradient = torch.autograd.grad(
            loss,
            fast,
            create_graph=second_order,
            retain_graph=second_order,
            allow_unused=False,
        )[0]
        used = gradient if second_order else gradient.detach()
        if optimizer_name == "sgd":
            fast = fast - FAST_LR * used
        else:
            (fast,), slots = v22.functional_adamw_step(
                (fast,),
                (used,),
                slots,
                (FAST_LR,),
                beta1=v22.ADAM_BETA1,
                beta2=v22.ADAM_BETA2,
                epsilon=v22.ADAM_EPSILON,
                weight_decay=v22.ADAM_WEIGHT_DECAY,
            )
    return fast


def _outer_gradients(
    system: v22.ANMLSystem,
    gate: v22.ANMLNeuromodulator,
    inner: Sequence[v22.ANMLFeatureBundle],
    outer: Sequence[v22.ANMLFeatureBundle],
    *,
    optimizer_name: str,
    second_order: bool,
) -> tuple[float, tuple[torch.Tensor, ...]]:
    fast = _functional_unroll(
        system,
        gate,
        inner,
        optimizer_name=optimizer_name,
        second_order=second_order,
    )
    losses = torch.stack(tuple(v22._stream_loss(system.controller, fast, stream, gate) for stream in outer))
    objective = v22.v20._anonymous_entropic_objective(losses, len(outer))
    gradients = torch.autograd.grad(
        objective,
        tuple(gate.parameters()),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    return float(objective.detach().item()), tuple(value.detach() for value in gradients)


def _apply_outer(
    gate: v22.ANMLNeuromodulator,
    optimizer: torch.optim.Optimizer,
    gradients: Sequence[torch.Tensor],
) -> float:
    parameters = tuple(gate.parameters())
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(parameters, v22.OUTER_GRADIENT_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
        raise RuntimeError("V23-D1 gate update became non-finite")
    return float(norm.detach().item())


def _fresh_state(weight: torch.Tensor) -> FastState:
    zero = torch.zeros_like(weight)
    return FastState(weight.detach().clone(), 0, zero, zero.clone())


def _fast_step(
    system: v22.ANMLSystem,
    state: FastState,
    bundle: v22.ANMLFeatureBundle,
    gate: v22.ANMLNeuromodulator | None,
    optimizer_name: str,
) -> tuple[FastState, float]:
    weight = state.weight.detach().clone().requires_grad_(True)
    loss = v22._stream_loss(
        system.controller,
        weight,
        bundle,
        gate,
        lesion="open" if gate is None else "live",
    )
    gradient = torch.autograd.grad(loss, weight, create_graph=False, allow_unused=False)[0]
    if optimizer_name == "sgd":
        updated = weight - FAST_LR * gradient.detach()
        next_state = FastState(
            updated.detach(), state.step + 1, state.exp_avg, state.exp_avg_sq
        )
    elif optimizer_name == "adamw":
        (updated,), slots = v22.functional_adamw_step(
            (weight,),
            (gradient.detach(),),
            (v22.AdamWSlot(state.step, state.exp_avg, state.exp_avg_sq),),
            (FAST_LR,),
            beta1=v22.ADAM_BETA1,
            beta2=v22.ADAM_BETA2,
            epsilon=v22.ADAM_EPSILON,
            weight_decay=v22.ADAM_WEIGHT_DECAY,
        )
        slot = slots[0]
        next_state = FastState(
            updated.detach(), slot.step, slot.exp_avg.detach(), slot.exp_avg_sq.detach()
        )
    else:
        raise ValueError("V23-D1 fast optimizer is invalid")
    return next_state, float(loss.detach().item())


def _score(
    system: v22.ANMLSystem,
    pair: LearnedPair,
    states: Mapping[str, FastState],
    bundles: Sequence[v22.ANMLFeatureBundle],
) -> dict[str, float]:
    gates = {
        "second_order": pair.second_gate,
        "first_order": pair.first_gate,
        "always_open": None,
    }
    result: dict[str, float] = {}
    with torch.no_grad():
        for name, gate in gates.items():
            losses = tuple(
                float(
                    v22._stream_loss(
                        system.controller,
                        states[name].weight,
                        bundle,
                        gate,
                        lesion="open" if gate is None else "live",
                    ).item()
                )
                for bundle in bundles
            )
            result[name] = sum(losses) / len(losses)
    return result


def _auc(values: Mapping[int, float]) -> float:
    if tuple(sorted(values)) != PROBE_MILESTONES:
        raise ValueError("V23-D1 probe milestones are incomplete")
    total = 0.0
    for left, right in zip(PROBE_MILESTONES[:-1], PROBE_MILESTONES[1:], strict=True):
        total += 0.5 * (float(values[left]) + float(values[right])) * (right - left)
    return total / LIFETIME_UPDATES


def _gradient_comparison(
    second: Sequence[torch.Tensor], first: Sequence[torch.Tensor]
) -> dict[str, float]:
    full = _flat(second)
    direct = _flat(first)
    indirect = full - direct
    full_norm = float(full.norm().item())
    direct_norm = float(direct.norm().item())
    indirect_norm = float(indirect.norm().item())
    return {
        "full_norm": full_norm,
        "direct_norm": direct_norm,
        "indirect_difference_norm": indirect_norm,
        "indirect_over_full": indirect_norm / full_norm if full_norm > 0.0 else 0.0,
        "full_direct_cosine": _cosine(full, direct),
    }


def optimizer_path_diagnostic(device: torch.device) -> dict[str, object]:
    if _sha256_file(V22_CHECKPOINT) != V22_CHECKPOINT_SHA256:
        raise RuntimeError("V23-D1 sealed V22 checkpoint hash changed")
    system = v22.load_anml_checkpoint(V22_CHECKPOINT, device=device)
    record = _stream_record(
        role=0,
        update=0,
        position=0,
        commitment_index=37,
        base=D0_SEED_BASE,
    )
    bundle = _bundles(system, (record,))[0]
    fast = system.fast_initial_weight.detach().clone().requires_grad_(True)
    live_loss = v22._stream_loss(system.controller, fast, bundle, system.second_order_anml.gate)
    live_gradient = torch.autograd.grad(live_loss, fast)[0].detach()
    fast_open = system.fast_initial_weight.detach().clone().requires_grad_(True)
    open_loss = v22._stream_loss(system.controller, fast_open, bundle, None, lesion="open")
    open_gradient = torch.autograd.grad(open_loss, fast_open)[0].detach()

    def update_separation(name: str) -> dict[str, float]:
        initial = system.fast_initial_weight.detach().clone()
        if name == "sgd":
            live = initial - FAST_LR * live_gradient
            opened = initial - FAST_LR * open_gradient
        else:
            zeros = _new_adam_slot(initial)
            (live,), _ = v22.functional_adamw_step(
                (initial,), (live_gradient,), zeros, (FAST_LR,),
                beta1=v22.ADAM_BETA1, beta2=v22.ADAM_BETA2,
                epsilon=v22.ADAM_EPSILON, weight_decay=v22.ADAM_WEIGHT_DECAY,
            )
            zeros = _new_adam_slot(initial)
            (opened,), _ = v22.functional_adamw_step(
                (initial,), (open_gradient,), zeros, (FAST_LR,),
                beta1=v22.ADAM_BETA1, beta2=v22.ADAM_BETA2,
                epsilon=v22.ADAM_EPSILON, weight_decay=v22.ADAM_WEIGHT_DECAY,
            )
        live_delta = live.detach() - initial
        open_delta = opened.detach() - initial
        denominator = float(open_delta.to(torch.float64).norm().item())
        separation = float((live_delta - open_delta).to(torch.float64).norm().item())
        return {
            "live_update_norm": float(live_delta.to(torch.float64).norm().item()),
            "open_update_norm": denominator,
            "live_open_update_difference_norm": separation,
            "live_open_update_separation_fraction": separation / denominator if denominator > 0.0 else 0.0,
        }

    meta = meta_records(0)
    all_bundles = _bundles(system, meta["inner"] + meta["outer"])
    inner20 = all_bundles[:MAX_INNER_STEPS]
    outer = all_bundles[MAX_INNER_STEPS:]
    path = {}
    for config in CONFIGS:
        inner = inner20[: config.inner_steps]
        _, second = _outer_gradients(
            system, system.second_order_anml.gate, inner, outer,
            optimizer_name=config.fast_optimizer, second_order=True,
        )
        _, first = _outer_gradients(
            system, system.second_order_anml.gate, inner, outer,
            optimizer_name=config.fast_optimizer, second_order=False,
        )
        path[config.name] = _gradient_comparison(second, first)
    return {
        "sealed_v22_checkpoint_sha256": V22_CHECKPOINT_SHA256,
        "live_loss": float(live_loss.detach().item()),
        "open_loss": float(open_loss.detach().item()),
        "live_gradient_norm": float(live_gradient.to(torch.float64).norm().item()),
        "open_gradient_norm": float(open_gradient.to(torch.float64).norm().item()),
        "live_open_gradient_cosine": _cosine(
            live_gradient.reshape(-1).to(torch.float64),
            open_gradient.reshape(-1).to(torch.float64),
        ),
        "updates": {
            "adamw": update_separation("adamw"),
            "sgd": update_separation("sgd"),
        },
        "meta_gradient_paths": path,
    }


def _build_pairs(system: v22.ANMLSystem) -> dict[str, LearnedPair]:
    initial = copy.deepcopy(system.second_order_anml.gate)
    result = {}
    for config in CONFIGS:
        second = copy.deepcopy(initial)
        first = copy.deepcopy(initial)
        result[config.name] = LearnedPair(
            config=config,
            second_gate=second,
            first_gate=first,
            second_outer=v22._make_outer_optimizer(second),
            first_outer=v22._make_outer_optimizer(first),
        )
    return result


def fit_pilot(
    system: v22.ANMLSystem, pairs: Mapping[str, LearnedPair]
) -> dict[str, object]:
    diagnostics = {name: [] for name in pairs}
    for update in range(OUTER_UPDATES):
        records = meta_records(update)
        bundles = _bundles(system, records["inner"] + records["outer"])
        inner20 = bundles[:MAX_INNER_STEPS]
        outer = bundles[MAX_INNER_STEPS:]
        for name, pair in pairs.items():
            inner = inner20[: pair.config.inner_steps]
            second_objective, second_gradients = _outer_gradients(
                system, pair.second_gate, inner, outer,
                optimizer_name=pair.config.fast_optimizer, second_order=True,
            )
            first_objective, first_gradients = _outer_gradients(
                system, pair.first_gate, inner, outer,
                optimizer_name=pair.config.fast_optimizer, second_order=False,
            )
            comparison = _gradient_comparison(second_gradients, first_gradients)
            second_norm = _apply_outer(pair.second_gate, pair.second_outer, second_gradients)
            first_norm = _apply_outer(pair.first_gate, pair.first_outer, first_gradients)
            diagnostics[name].append(
                {
                    "update": update,
                    "second_objective": second_objective,
                    "first_objective": first_objective,
                    "second_gradient_norm": second_norm,
                    "first_gradient_norm": first_norm,
                    **comparison,
                }
            )
        del bundles
    return {"outer_updates": OUTER_UPDATES, "configurations": diagnostics}


def evaluate_pilot(
    system: v22.ANMLSystem, pairs: Mapping[str, LearnedPair]
) -> dict[str, object]:
    reports = {}
    for panel in (0, 1):
        probe_bundles = _bundles(system, probe_records(panel))
        states = {
            name: {
                arm: _fresh_state(system.fast_initial_weight)
                for arm in ("second_order", "first_order", "always_open")
            }
            for name in pairs
        }
        probes = {
            name: {0: _score(system, pair, states[name], probe_bundles)}
            for name, pair in pairs.items()
        }
        online = {
            name: {arm: {"loss_sum": 0.0, "count": 0} for arm in states[name]}
            for name in pairs
        }
        for step in range(LIFETIME_UPDATES):
            bundle = _bundles(system, (lifetime_record(panel, step),))[0]
            for name, pair in pairs.items():
                gates = {
                    "second_order": pair.second_gate,
                    "first_order": pair.first_gate,
                    "always_open": None,
                }
                for arm, gate in gates.items():
                    next_state, loss = _fast_step(
                        system,
                        states[name][arm],
                        bundle,
                        gate,
                        pair.config.fast_optimizer,
                    )
                    states[name][arm] = next_state
                    online[name][arm]["loss_sum"] += loss
                    online[name][arm]["count"] += 1
            if step + 1 in PROBE_MILESTONES:
                for name, pair in pairs.items():
                    probes[name][step + 1] = _score(
                        system, pair, states[name], probe_bundles
                    )
            del bundle
        for name, pair in pairs.items():
            panel_report = reports.setdefault(name, {"panels": []})
            arms = {}
            for arm in ("second_order", "first_order", "always_open"):
                losses = {milestone: probes[name][milestone][arm] for milestone in PROBE_MILESTONES}
                arms[arm] = {
                    "loss_auc": _auc(losses),
                    "terminal_loss": losses[LIFETIME_UPDATES],
                    "milestone_losses": losses,
                    "online_loss_mean": online[name][arm]["loss_sum"] / online[name][arm]["count"],
                }
            second = arms["second_order"]
            first = arms["first_order"]
            opened = arms["always_open"]
            panel_report["panels"].append(
                {
                    "panel": panel,
                    "order": "blocked" if panel == 0 else "interleaved",
                    "arms": arms,
                    "second_vs_first_auc": _relative_improvement(second["loss_auc"], first["loss_auc"]),
                    "second_vs_first_terminal": _relative_improvement(second["terminal_loss"], first["terminal_loss"]),
                    "second_vs_open_auc": _relative_improvement(second["loss_auc"], opened["loss_auc"]),
                    "second_vs_open_terminal": _relative_improvement(second["terminal_loss"], opened["terminal_loss"]),
                }
            )
        del probe_bundles
    for name, report in reports.items():
        panels = report["panels"]
        report["full_v23_eligible"] = all(
            panel["second_vs_first_auc"] > 0.0
            and panel["second_vs_first_terminal"] > 0.0
            and panel["second_vs_open_auc"] >= 0.0
            and panel["second_vs_open_terminal"] >= 0.0
            for panel in panels
        )
        report["minimum_second_vs_first_auc"] = min(
            panel["second_vs_first_auc"] for panel in panels
        )
    return reports


def classify_result(
    d0: Mapping[str, object], evaluation: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    eligible = [name for name, report in evaluation.items() if report["full_v23_eligible"]]
    selected = None
    if eligible:
        config_order = {config.name: index for index, config in enumerate(CONFIGS)}
        selected = max(
            eligible,
            key=lambda name: (
                float(evaluation[name]["minimum_second_vs_first_auc"]),
                -next(config.inner_steps for config in CONFIGS if config.name == name),
                1 if next(config.fast_optimizer for config in CONFIGS if config.name == name) == "sgd" else 0,
                -config_order[name],
            ),
        )
        classification = "FULL_V23_ELIGIBLE"
    else:
        updates = d0["updates"]
        adamw = float(updates["adamw"]["live_open_update_separation_fraction"])
        sgd = float(updates["sgd"]["live_open_update_separation_fraction"])
        if sgd >= 2.0 * adamw and sgd - adamw >= 0.05:
            classification = "OPTIMIZER_MISMATCH_SUPPORTED"
        else:
            classification = "FIDELITY_HYPOTHESES_NOT_SUPPORTED"
    return {
        "classification": classification,
        "eligible_configurations": tuple(eligible),
        "selected_configuration": selected,
    }


def synthetic_preflight(device: torch.device | str = "cuda") -> dict[str, object]:
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V23-D1 CUDA preflight requires CUDA")
    weight = torch.tensor([[1.0, -2.0]], device=selected, dtype=torch.float32)
    gradient = torch.tensor([[0.25, -0.5]], device=selected, dtype=torch.float32)
    sgd = weight - FAST_LR * gradient
    zero = torch.zeros_like(weight)
    (adamw,), _ = v22.functional_adamw_step(
        (weight,),
        (gradient,),
        (v22.AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),),
        (FAST_LR,),
        beta1=v22.ADAM_BETA1,
        beta2=v22.ADAM_BETA2,
        epsilon=v22.ADAM_EPSILON,
        weight_decay=v22.ADAM_WEIGHT_DECAY,
    )
    return {
        "passed": bool(torch.isfinite(sgd).all() and torch.isfinite(adamw).all()),
        "device": str(selected),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "configurations": tuple(config.name for config in CONFIGS),
        "outer_updates": OUTER_UPDATES,
        "lifetime_updates": LIFETIME_UPDATES,
        "orders_balanced": all(len(lifetime_order(panel)) == LIFETIME_UPDATES for panel in (0, 1)),
    }


def run(device: torch.device | str = "cuda") -> dict[str, object]:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V23-D1 semantic pilot requires CUDA")
    v22.configure_anml_numerics(selected)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(0 if selected.index is None else selected.index)
    d0 = optimizer_path_diagnostic(selected)
    system = v22.build_anml_system(device=selected)
    controller_before = v22._controller_digest(system.controller)
    pairs = _build_pairs(system)
    fit = fit_pilot(system, pairs)
    evaluation = evaluate_pilot(system, pairs)
    decision = classify_result(d0, evaluation)
    elapsed = time.monotonic() - started
    maximum = int(torch.cuda.max_memory_allocated(0 if selected.index is None else selected.index))
    if elapsed > WALL_CEILING_SECONDS:
        raise RuntimeError("V23-D1 exceeded its wall-time ceiling")
    if maximum > MEMORY_CEILING_BYTES:
        raise RuntimeError("V23-D1 exceeded its allocated-memory ceiling")
    if v22._controller_digest(system.controller) != controller_before:
        raise RuntimeError("V23-D1 changed the frozen V20 controller")
    result = {
        "artifact_schema": "angler.anml-fidelity-v23-d1.result.v1",
        "protocol_id": PROTOCOL_ID,
        "classification": decision["classification"],
        "decision": decision,
        "d0": d0,
        "fit": fit,
        "evaluation": evaluation,
        "mechanical_validity": {
            "passed": True,
            "controller_digest_before": controller_before,
            "controller_digest_after": v22._controller_digest(system.controller),
            "maximum_allocated_bytes": maximum,
            "allocated_memory_ceiling_bytes": MEMORY_CEILING_BYTES,
            "elapsed_seconds": elapsed,
            "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        },
        "interpretation_limit": "development-only optimizer/horizon direction; not ANML support, promotion, Qwen evidence, or AGI",
    }
    v22._validate_json_value(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = synthetic_preflight(args.device) if args.mode == "preflight" else run(args.device)
    if args.output:
        v22.atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

