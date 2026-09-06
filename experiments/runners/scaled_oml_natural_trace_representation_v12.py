"""Frozen scaled OML natural-trace representation evaluation V12."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from angler.reasoning.natural_trace_graph_causal_memory import parse_step_trace
from angler.reasoning.oml_natural_trace_representation import (
    OMLNaturalTraceRepresentation,
    representation_metrics,
)
from angler.runtime import foundation_tensor_digest
from experiments.corpora.scaled_oml_natural_trace_v12 import (
    CORPUS_ID,
    DEVELOPMENT_MECHANISMS,
    FINAL_MECHANISMS,
    TRAIN_INNER_MECHANISMS,
    TRAIN_OUTER_MECHANISMS,
    TRAIN_OUTER_SLOTS,
    TRAIN_OUTER_UPDATES,
    FinalPartitionSealedError,
    build_scaled_oml_natural_trace_v12,
    build_trace_representation_contrasts,
)
from experiments.runners import outcome_aware_apprenticeship_v5 as v5
from experiments.runners.phase6_cross_variation_plasticity import AdamWSlot
from experiments.runners.phase6_cross_variation_plasticity_v16 import (
    functional_adamw_step,
)


IDENTITY = "angler.scaled-oml-natural-trace-representation.v12-first-result"
SEED = 2026083112
MAXIMUM_TRACE_STEPS = 8
INNER_STEPS = 8
OUTER_MECHANISMS = 8
OUTER_UPDATES = 192
INNER_LEARNING_RATE = 1.0e-3
OUTER_LEARNING_RATE = 3.0e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1.0e-8
ADAM_WEIGHT_DECAY = 0.0
OUTER_GRADIENT_CLIP = 5.0
ROBUST_TEMPERATURE = 0.05
ROBUST_MEAN_WEIGHT = 0.5
ROBUST_TAIL_WEIGHT = 0.5
LESION_ACCURACY_MARGIN = 0.05
LESION_LOSS_MARGIN = 0.05

ARM_SECOND_ONLINE = "second_order_online"
ARM_FIRST_ONLINE = "first_order_online"
ARM_SOURCE_ONLINE = "source_online"
ARM_SECOND_NO_UPDATE = "second_order_no_update"
ARM_FIRST_NO_UPDATE = "first_order_no_update"
ARM_SOURCE_NO_UPDATE = "source_no_update"
ARM_SHUFFLED = "shuffled_outcome_adaptation"
ARM_DIRECTION_REMOVED = "direction_removed"
ARM_SEMANTICS_REMOVED = "step_semantics_removed"
SHUFFLE_PERMUTATION = (1, 0, 3, 2, 5, 4)


@dataclass(frozen=True, slots=True)
class EncodedMechanism:
    mechanism_ref: str
    generator_family: str
    transition_type: str
    step_features: torch.Tensor
    step_mask: torch.Tensor
    outcomes: torch.Tensor

    def to(self, device: torch.device) -> "EncodedMechanism":
        return EncodedMechanism(
            mechanism_ref=self.mechanism_ref,
            generator_family=self.generator_family,
            transition_type=self.transition_type,
            step_features=self.step_features.to(device=device, dtype=torch.float32),
            step_mask=self.step_mask.to(device=device),
            outcomes=self.outcomes.to(device=device, dtype=torch.float32),
        )


@dataclass(frozen=True, slots=True)
class EncodedContrasts:
    step_features: torch.Tensor
    step_mask: torch.Tensor

    def to(self, device: torch.device) -> "EncodedContrasts":
        return EncodedContrasts(
            step_features=self.step_features.to(device=device, dtype=torch.float32),
            step_mask=self.step_mask.to(device=device),
        )


@dataclass(slots=True)
class OmlArm:
    name: str
    model: OMLNaturalTraceRepresentation
    optimizer: torch.optim.Optimizer | None
    outer_updates: int = 0


@dataclass(slots=True)
class OmlSystem:
    second_order: OmlArm
    first_order: OmlArm
    source: OmlArm
    initial_model_digest: str


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = {
        "runner": Path(__file__).resolve(),
        "core": root / "src/angler/reasoning/oml_natural_trace_representation.py",
        "trace_graph_core": root / "src/angler/reasoning/natural_trace_graph_causal_memory.py",
        "corpus": root / "experiments/corpora/scaled_oml_natural_trace_v12.py",
        "functional_adamw": root / "experiments/runners/phase6_cross_variation_plasticity_v16.py",
        "adamw_slot": root / "experiments/runners/phase6_cross_variation_plasticity.py",
        "qwen_runtime": root / "src/angler/runtime/qwen_knowledge.py",
        "qwen_loader": root / "experiments/runners/outcome_aware_apprenticeship_v5.py",
    }
    return {name: _sha256(path) for name, path in files.items()}


def _model_digest(model: torch.nn.Module) -> str:
    return v5._module_digest(model)


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(tuple(value.shape)).encode("ascii") + b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V12 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale V12 temporary output exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V12 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale V12 temporary output exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parsed_traces(mechanisms: Sequence[Any]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    values = []
    for mechanism in mechanisms:
        traces = []
        for episode in mechanism.public.episodes:
            parsed = parse_step_trace(episode.action_trace_text)
            if not 1 <= len(parsed) <= MAXIMUM_TRACE_STEPS:
                raise ValueError("V12 public trace exceeds its frozen step ceiling")
            traces.append(parsed)
        if len(traces) != 6:
            raise ValueError("V12 mechanism must expose exactly six public episodes")
        values.append(tuple(traces))
    return tuple(values)


def _embed_trace_groups(qwen: Any, groups: Sequence[Sequence[Sequence[str]]]) -> tuple[torch.Tensor, torch.Tensor]:
    sentences: list[str] = []
    spans = []
    for group in groups:
        row = []
        for trace in group:
            if not 1 <= len(trace) <= MAXIMUM_TRACE_STEPS:
                raise ValueError("V12 trace group violates the frozen step ceiling")
            start = len(sentences)
            sentences.extend(trace)
            row.append((start, len(trace)))
        if len(row) != 6:
            raise ValueError("V12 encoded groups must contain six traces")
        spans.append(tuple(row))
    encoded = qwen.embed(sentences).to(device="cpu", dtype=torch.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(sentences):
        raise RuntimeError("Qwen embeddings do not align with V12 raw step sentences")
    width = int(encoded.shape[-1])
    features = encoded.new_zeros((len(groups), 6, MAXIMUM_TRACE_STEPS, width))
    mask = torch.zeros((len(groups), 6, MAXIMUM_TRACE_STEPS), dtype=torch.bool)
    for group_index, row in enumerate(spans):
        for trace_index, (start, length) in enumerate(row):
            features[group_index, trace_index, :length] = encoded[start : start + length]
            mask[group_index, trace_index, :length] = True
    return features, mask


def _encode_mechanisms(qwen: Any, mechanisms: Sequence[Any]) -> tuple[EncodedMechanism, ...]:
    traces = _parsed_traces(mechanisms)
    features, mask = _embed_trace_groups(qwen, traces)
    rows = []
    for index, mechanism in enumerate(mechanisms):
        # Metadata remains evaluator/orchestrator-only and is never tensorized.
        rows.append(
            EncodedMechanism(
                mechanism_ref=mechanism.metadata.mechanism_ref,
                generator_family=mechanism.metadata.generator_family,
                transition_type=mechanism.metadata.heldout_variant,
                step_features=features[index].clone(),
                step_mask=mask[index].clone(),
                outcomes=torch.tensor(
                    tuple(int(episode.outcome_value) for episode in mechanism.public.episodes),
                    dtype=torch.float32,
                ),
            )
        )
    return tuple(rows)


def _encode_contrasts(qwen: Any, mechanisms: Sequence[Any]) -> EncodedContrasts:
    groups = []
    for mechanism in mechanisms:
        contrast = build_trace_representation_contrasts(mechanism)
        texts = (
            contrast.reference_trace_text,
            contrast.paraphrase_trace_text,
            contrast.reordered_trace_text,
            contrast.omitted_trace_text,
            contrast.inserted_trace_text,
            contrast.replacement_trace_text,
        )
        groups.append(tuple(parse_step_trace(text) for text in texts))
    features, mask = _embed_trace_groups(qwen, groups)
    return EncodedContrasts(features, mask)


def _public_overlap_audit(
    inner: Sequence[Any], outer: Sequence[Any], development: Sequence[Any]
) -> dict[str, Any]:
    def surfaces(mechanisms: Sequence[Any]) -> tuple[set[str], list[str]]:
        values: list[str] = []
        traces: list[str] = []
        for mechanism in mechanisms:
            for episode in mechanism.public.episodes:
                values.extend(
                    (episode.task_text, episode.request_text, episode.action_trace_text)
                )
                traces.append(episode.action_trace_text)
            challenge = mechanism.public.challenge
            values.extend((challenge.task_text, challenge.request_text))
            values.extend(challenge.candidate_action_trace_texts)
        return set(values), traces

    inner_public, inner_traces = surfaces(inner)
    outer_public, outer_traces = surfaces(outer)
    development_public, _ = surfaces(development)
    unique_fraction = len(set(inner_traces)) / len(inner_traces)
    report = {
        "inner_unique_trace_fraction": unique_fraction,
        "inner_outer_public_overlap": len(inner_public & outer_public),
        "inner_development_public_overlap": len(inner_public & development_public),
        "outer_development_public_overlap": len(outer_public & development_public),
        "outer_trace_count": len(outer_traces),
        "outer_unique_trace_count": len(set(outer_traces)),
        "outer_mechanism_count": len(outer),
        "outer_unique_mechanism_count": len(
            {mechanism.metadata.mechanism_ref for mechanism in outer}
        ),
    }
    report["passed"] = bool(
        unique_fraction >= 0.95
        and report["inner_outer_public_overlap"] == 0
        and report["inner_development_public_overlap"] == 0
        and report["outer_development_public_overlap"] == 0
        and report["outer_mechanism_count"] == TRAIN_OUTER_MECHANISMS
        and report["outer_unique_mechanism_count"] == TRAIN_OUTER_MECHANISMS
    )
    if not report["passed"]:
        raise RuntimeError("V12 public surface overlap audit failed")
    return report


def _batch(rows: Sequence[EncodedMechanism], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("V12 tensor batch cannot be empty")
    moved = tuple(row.to(device) for row in rows)
    return (
        torch.stack(tuple(row.step_features for row in moved)),
        torch.stack(tuple(row.step_mask for row in moved)),
        torch.stack(tuple(row.outcomes for row in moved)),
    )


def _robust_objective(losses: torch.Tensor, expected_count: int | None = None) -> torch.Tensor:
    if (
        losses.ndim != 1
        or losses.numel() == 0
        or (expected_count is not None and losses.numel() != expected_count)
        or not bool(torch.isfinite(losses).all().item())
    ):
        raise ValueError("V12 robust objective requires a finite declared loss vector")
    temperature = losses.new_tensor(ROBUST_TEMPERATURE)
    return ROBUST_MEAN_WEIGHT * losses.mean() + ROBUST_TAIL_WEIGHT * temperature * (
        torch.logsumexp(losses / temperature, dim=0) - math.log(losses.numel())
    )


def _robust_weights(losses: torch.Tensor) -> torch.Tensor:
    if losses.ndim != 1 or losses.numel() == 0 or not bool(torch.isfinite(losses).all().item()):
        raise ValueError("V12 robust weights require finite losses")
    return ROBUST_MEAN_WEIGHT / losses.numel() + ROBUST_TAIL_WEIGHT * torch.softmax(
        losses.detach() / ROBUST_TEMPERATURE, dim=0
    )


def _fresh_fast(model: OMLNaturalTraceRepresentation) -> tuple[torch.Tensor, tuple[AdamWSlot, ...]]:
    fast = model.fresh_fast_weight()
    zero = torch.zeros_like(fast)
    return fast, (AdamWSlot(step=0, exp_avg=zero, exp_avg_sq=zero.clone()),)


def _inner_step(
    model: OMLNaturalTraceRepresentation,
    fast: torch.Tensor,
    state: Sequence[AdamWSlot],
    row: EncodedMechanism,
    *,
    second_order: bool,
    outcomes: torch.Tensor | None = None,
    include_direction: bool = True,
    include_step_semantics: bool = True,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], float]:
    if type(second_order) is not bool:
        raise TypeError("V12 second_order must be bool")
    device = fast.device
    features, mask, observed = _batch((row,), device)
    if outcomes is not None:
        observed = outcomes.reshape(1, 6).to(device=device, dtype=torch.float32)
    loss = model.functional_loss(
        features,
        mask,
        observed,
        fast,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    gradient = torch.autograd.grad(
        loss,
        (fast,),
        create_graph=second_order,
        retain_graph=second_order,
        allow_unused=False,
    )[0]
    if not bool(torch.isfinite(gradient).all().item()):
        raise RuntimeError("V12 inner gradient is non-finite")
    used = gradient if second_order else gradient.detach()
    updated, next_state = functional_adamw_step(
        (fast,),
        (used,),
        tuple(state),
        (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    return updated[0], tuple(next_state), float(loss.detach().item())


def _unroll_inner(
    model: OMLNaturalTraceRepresentation,
    rows: Sequence[EncodedMechanism],
    *,
    second_order: bool,
) -> tuple[torch.Tensor, tuple[AdamWSlot, ...], tuple[float, ...]]:
    if len(rows) != INNER_STEPS:
        raise ValueError("V12 inner trajectory requires exactly eight mechanisms")
    before = _model_digest(model)
    fast, state = _fresh_fast(model)
    losses = []
    for row in rows:
        fast, state, loss = _inner_step(
            model, fast, state, row, second_order=second_order
        )
        losses.append(loss)
    if state[0].step != INNER_STEPS or _model_digest(model) != before:
        raise RuntimeError("V12 inner trajectory crossed the fast-head ownership boundary")
    return fast, state, tuple(losses)


def _outer_losses(
    model: OMLNaturalTraceRepresentation,
    fast: torch.Tensor,
    rows: Sequence[EncodedMechanism],
) -> torch.Tensor:
    if not rows or len(rows) > OUTER_MECHANISMS:
        raise ValueError("V12 outer loss group must contain one to eight mechanisms")
    device = fast.device
    return torch.stack(
        tuple(
            model.functional_loss(*_batch((row,), device), fast)
            for row in rows
        )
    )


def _outer_gradients(
    model: OMLNaturalTraceRepresentation,
    inner_rows: Sequence[EncodedMechanism],
    outer_rows: Sequence[EncodedMechanism],
    *,
    second_order: bool,
    split: bool,
) -> tuple[float, tuple[torch.Tensor, ...], dict[str, Any]]:
    fast, state, inner_losses = _unroll_inner(
        model, inner_rows, second_order=second_order
    )
    parameters = tuple(value for _, value in model.rln_named_parameters())
    if not split:
        losses = _outer_losses(model, fast, outer_rows)
        objective = _robust_objective(losses, OUTER_MECHANISMS)
        gradients = torch.autograd.grad(objective, parameters, allow_unused=False)
    else:
        with torch.no_grad():
            detached_losses = _outer_losses(model, fast, outer_rows)
            objective = _robust_objective(detached_losses, OUTER_MECHANISMS)
            coefficients = _robust_weights(detached_losses)
        accumulated = [torch.zeros_like(value) for value in parameters]
        for group, start in enumerate((0, 4)):
            losses_group = _outer_losses(model, fast, outer_rows[start : start + 4])
            weighted = (
                coefficients[start : start + 4].to(losses_group) * losses_group
            ).sum()
            gradients_group = torch.autograd.grad(
                weighted,
                parameters,
                retain_graph=group == 0,
                allow_unused=False,
            )
            for index, gradient in enumerate(gradients_group):
                accumulated[index] = accumulated[index] + gradient.detach()
        gradients = tuple(accumulated)
        losses = detached_losses
    if any(not bool(torch.isfinite(value).all().item()) for value in gradients):
        raise RuntimeError("V12 outer meta-gradient is non-finite")
    return float(objective.detach().item()), tuple(value.detach() for value in gradients), {
        "inner_losses": inner_losses,
        "outer_losses": tuple(float(value) for value in losses.detach().tolist()),
        "terminal_fast_step": state[0].step,
        "mode": "split_4_plus_4" if split else "full",
    }


def _apply_outer_step(arm: OmlArm, gradients: Sequence[torch.Tensor]) -> dict[str, float]:
    if arm.optimizer is None:
        raise RuntimeError("source arm cannot receive an outer update")
    named = arm.model.rln_named_parameters()
    if len(named) != len(gradients):
        raise RuntimeError("V12 RLN gradient partition changed")
    fast_before = arm.model.fast_initial_weight.detach().clone()
    arm.optimizer.zero_grad(set_to_none=True)
    for (name, parameter), gradient in zip(named, gradients, strict=True):
        if gradient.shape != parameter.shape:
            raise RuntimeError(f"V12 outer gradient shape changed: {name}")
        parameter.grad = gradient.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(
        tuple(parameter for _, parameter in named), OUTER_GRADIENT_CLIP
    )
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError("V12 clipped outer gradient is non-finite")
    arm.optimizer.step()
    arm.optimizer.zero_grad(set_to_none=True)
    arm.outer_updates += 1
    if not torch.equal(arm.model.fast_initial_weight, fast_before):
        raise RuntimeError("V12 outer update mutated the fixed PLN initialization")
    return {"gradient_norm_before_clip": float(norm.item()), "outer_update": arm.outer_updates}


def _build_system(step_width: int, device: torch.device) -> OmlSystem:
    torch.manual_seed(SEED)
    initial = OMLNaturalTraceRepresentation(step_width=step_width).to(
        device=device, dtype=torch.float32
    )
    second_model = copy.deepcopy(initial)
    first_model = copy.deepcopy(initial)
    source_model = copy.deepcopy(initial)
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    digest = _model_digest(initial)
    if len({_model_digest(second_model), _model_digest(first_model), _model_digest(source_model), digest}) != 1:
        raise RuntimeError("V12 paired/source arms did not start byte-exact")
    del initial
    def learned(name: str, model: OMLNaturalTraceRepresentation) -> OmlArm:
        optimizer = torch.optim.AdamW(
            tuple(model.parameters()),
            lr=OUTER_LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPSILON,
            weight_decay=ADAM_WEIGHT_DECAY,
            foreach=False,
            fused=False,
        )
        return OmlArm(name, model, optimizer)
    return OmlSystem(
        second_order=learned(ARM_SECOND_ONLINE, second_model),
        first_order=learned(ARM_FIRST_ONLINE, first_model),
        source=OmlArm(ARM_SOURCE_ONLINE, source_model, None),
        initial_model_digest=digest,
    )


def _full_split_equivalence(
    model: OMLNaturalTraceRepresentation,
    inner: Sequence[EncodedMechanism],
    outer: Sequence[EncodedMechanism],
    *,
    second_order: bool,
) -> dict[str, float]:
    full_objective, full_gradients, _ = _outer_gradients(
        model, inner, outer, second_order=second_order, split=False
    )
    split_objective, split_gradients, _ = _outer_gradients(
        model, inner, outer, second_order=second_order, split=True
    )
    maximum = max(
        float((left - right).abs().max().item())
        for left, right in zip(full_gradients, split_gradients, strict=True)
    )
    objective_delta = abs(full_objective - split_objective)
    if objective_delta > 1.0e-6 or maximum > 1.0e-6:
        raise RuntimeError("V12 full/split outer gradients are not equivalent")
    return {
        "objective_absolute_delta": objective_delta,
        "maximum_gradient_absolute_delta": maximum,
        "tolerance": 1.0e-6,
    }


def _validate_inner_schedule(
    inner_schedule: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    if (
        len(inner_schedule) != OUTER_UPDATES
        or any(len(update) != INNER_STEPS for update in inner_schedule)
    ):
        raise ValueError("V12 inner schedule shape changed")
    exposures = [0] * TRAIN_INNER_MECHANISMS
    for update in inner_schedule:
        if len(set(update)) != INNER_STEPS:
            raise RuntimeError("V12 update reused an inner mechanism")
        for index in update:
            if type(index) is not int or not 0 <= index < TRAIN_INNER_MECHANISMS:
                raise ValueError("V12 inner schedule index is invalid")
            exposures[index] += 1
    if any(count != 4 for count in exposures):
        raise RuntimeError("V12 inner schedule lost exact four-use balance")
    return tuple(exposures)


def _inner_connectivity_preflight(
    second_model: OMLNaturalTraceRepresentation,
    first_model: OMLNaturalTraceRepresentation,
    row: EncodedMechanism,
) -> dict[str, Any]:
    """Prove that paired inner forwards differ only in meta-gradient connectivity."""

    second_fast, second_state = _fresh_fast(second_model)
    first_fast, first_state = _fresh_fast(first_model)
    second_features, second_mask, second_outcomes = _batch(
        (row,), second_fast.device
    )
    first_features, first_mask, first_outcomes = _batch((row,), first_fast.device)
    second_loss = second_model.functional_loss(
        second_features, second_mask, second_outcomes, second_fast
    )
    first_loss = first_model.functional_loss(
        first_features, first_mask, first_outcomes, first_fast
    )
    second_gradient = torch.autograd.grad(
        second_loss,
        (second_fast,),
        create_graph=True,
        retain_graph=True,
    )[0]
    first_gradient = torch.autograd.grad(first_loss, (first_fast,))[0]
    (second_updated,), second_moments = functional_adamw_step(
        (second_fast,),
        (second_gradient,),
        second_state,
        (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    (first_updated,), first_moments = functional_adamw_step(
        (first_fast,),
        (first_gradient.detach(),),
        first_state,
        (INNER_LEARNING_RATE,),
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    second_hessian = torch.autograd.grad(
        second_updated.sum(),
        tuple(second_model.parameters()),
        allow_unused=True,
    )
    first_hessian = torch.autograd.grad(
        first_updated.sum(),
        tuple(first_model.parameters()),
        allow_unused=True,
    )
    raw_equal = bool(torch.equal(second_gradient.detach(), first_gradient.detach()))
    updated_equal = bool(torch.equal(second_updated.detach(), first_updated.detach()))
    moments_equal = bool(
        torch.equal(second_moments[0].exp_avg.detach(), first_moments[0].exp_avg.detach())
        and torch.equal(
            second_moments[0].exp_avg_sq.detach(),
            first_moments[0].exp_avg_sq.detach(),
        )
    )
    second_connected = any(
        value is not None and float(value.abs().sum().item()) > 0.0
        for value in second_hessian
    )
    first_detached = all(value is None for value in first_hessian)
    report = {
        "raw_loss_equal": bool(torch.equal(second_loss.detach(), first_loss.detach())),
        "raw_fast_gradient_equal": raw_equal,
        "updated_fast_weight_equal": updated_equal,
        "adamw_moments_equal": moments_equal,
        "second_order_hessian_path_nonzero": second_connected,
        "first_order_hessian_path_absent": first_detached,
        "paired_inputs_equal": bool(
            torch.equal(second_features, first_features)
            and torch.equal(second_mask, first_mask)
            and torch.equal(second_outcomes, first_outcomes)
        ),
    }
    if not all(report.values()):
        raise RuntimeError("V12 paired inner connectivity preflight failed")
    return report


def _fit(
    system: OmlSystem,
    inner_rows: Sequence[EncodedMechanism],
    outer_rows: Sequence[EncodedMechanism],
    inner_schedule: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(inner_rows) != TRAIN_INNER_MECHANISMS or len(outer_rows) != TRAIN_OUTER_MECHANISMS:
        raise ValueError("V12 encoded training corpus counts changed")
    exposures = _validate_inner_schedule(inner_schedule)
    source_before = _model_digest(system.source.model)
    diagnostics = []
    preflight: dict[str, Any] = {
        "inner_connectivity": _inner_connectivity_preflight(
            system.second_order.model, system.first_order.model, inner_rows[0]
        )
    }
    for update in range(OUTER_UPDATES):
        inner = tuple(
            inner_rows[index] for index in inner_schedule[update]
        )
        outer = tuple(
            outer_rows[update * OUTER_MECHANISMS + slot]
            for slot in range(OUTER_MECHANISMS)
        )
        if update == 0:
            preflight.update({
                "second_order": _full_split_equivalence(
                    system.second_order.model, inner, outer, second_order=True
                ),
                "first_order": _full_split_equivalence(
                    system.first_order.model, inner, outer, second_order=False
                ),
            })
        second_objective, second_gradients, second_details = _outer_gradients(
            system.second_order.model,
            inner,
            outer,
            second_order=True,
            split=False,
        )
        first_objective, first_gradients, first_details = _outer_gradients(
            system.first_order.model,
            inner,
            outer,
            second_order=False,
            split=False,
        )
        paired_equal = second_details["outer_losses"] == first_details["outer_losses"]
        if update == 0 and (not paired_equal or second_objective != first_objective):
            raise RuntimeError("V12 paired arms lost numeric equality before update zero")
        second_step = _apply_outer_step(system.second_order, second_gradients)
        first_step = _apply_outer_step(system.first_order, first_gradients)
        if update == 0:
            preflight["direct_outer_gradients_nonzero"] = bool(
                any(float(value.abs().sum().item()) > 0.0 for value in second_gradients)
                and any(float(value.abs().sum().item()) > 0.0 for value in first_gradients)
            )
            if not preflight["direct_outer_gradients_nonzero"]:
                raise RuntimeError("V12 direct outer gradient path is missing")
        diagnostics.append(
            {
                "update": update,
                "paired_forward_equal_before_owner_step": paired_equal,
                "second_order_objective": second_objective,
                "first_order_objective": first_objective,
                "second_order": second_step,
                "first_order": first_step,
            }
        )
        if (update + 1) % 32 == 0:
            print(json.dumps({"v12_update": update + 1, "total": OUTER_UPDATES}), flush=True)
    if (
        system.second_order.outer_updates != OUTER_UPDATES
        or system.first_order.outer_updates != OUTER_UPDATES
        or _model_digest(system.source.model) != source_before
    ):
        raise RuntimeError("V12 paired owner counts or source immutability changed")
    return {
        "updates": OUTER_UPDATES,
        "inner_mechanism_uses": OUTER_UPDATES * INNER_STEPS,
        "outer_mechanism_uses": OUTER_UPDATES * OUTER_MECHANISMS,
        "full_split_equivalence": preflight,
        "update_diagnostics": diagnostics,
        "source_unchanged": True,
        "paired_data_and_exposure_exact": True,
        "minimum_inner_exposure": min(exposures),
        "maximum_inner_exposure": max(exposures),
        "fast_head_only_inner_mutation": True,
        "rln_only_outer_mutation": True,
    }


def _panel_pairs(rows: Sequence[EncodedMechanism]) -> tuple[tuple[tuple[EncodedMechanism, EncodedMechanism], ...], ...]:
    by_family: dict[str, list[EncodedMechanism]] = {}
    for row in rows:
        by_family.setdefault(row.generator_family, []).append(row)
    families = tuple(by_family)
    if len(families) != 12 or any(len(by_family[name]) != 4 for name in families):
        raise ValueError("V12 evaluation partition lost its 12x4 family layout")
    panels = []
    panels.append(tuple((by_family[name][0], by_family[name][1]) for name in families))
    panels.append(tuple((by_family[name][2], by_family[name][3]) for name in families))
    panels.append(
        tuple((by_family[name][0], by_family[families[(index + 3) % 12]][2]) for index, name in enumerate(families))
    )
    panels.append(
        tuple((by_family[name][1], by_family[families[(index + 6) % 12]][3]) for index, name in enumerate(families))
    )
    return tuple(panels)


def _balanced_accuracy(logits: torch.Tensor, outcomes: torch.Tensor) -> float:
    predicted = torch.where(logits >= 0, 1.0, -1.0)
    positive = outcomes == 1
    negative = outcomes == -1
    sensitivity = (predicted[positive] == 1).float().mean()
    specificity = (predicted[negative] == -1).float().mean()
    return float((0.5 * (sensitivity + specificity)).item())


def _shuffled_outcomes(outcomes: torch.Tensor) -> torch.Tensor:
    if outcomes.shape != (6,) or int((outcomes == 1).sum().item()) != 3 or int(
        (outcomes == -1).sum().item()
    ) != 3:
        raise ValueError("V12 shuffled control requires one balanced six-label stream")
    shuffled = outcomes[list(SHUFFLE_PERMUTATION)]
    if torch.equal(shuffled, outcomes) or sorted(shuffled.tolist()) != sorted(outcomes.tolist()):
        raise RuntimeError("V12 frozen shuffled-outcome permutation is invalid")
    return shuffled


def _evaluate_pair(
    model: OMLNaturalTraceRepresentation,
    support: EncodedMechanism,
    probe: EncodedMechanism,
    *,
    updates_enabled: bool,
    shuffle_outcomes: bool = False,
    include_direction: bool = True,
    include_step_semantics: bool = True,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    fast, state = _fresh_fast(model)
    probe_features, probe_mask, probe_outcomes = _batch((probe,), device)
    pre = model.functional_loss(
        probe_features,
        probe_mask,
        probe_outcomes,
        fast,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    support_outcomes = support.outcomes
    if updates_enabled:
        support_outcomes = (
            _shuffled_outcomes(support.outcomes)
            if shuffle_outcomes
            else support.outcomes
        )
        fast, state, adaptation_loss = _inner_step(
            model,
            fast,
            state,
            support,
            second_order=False,
            outcomes=support_outcomes,
            include_direction=include_direction,
            include_step_semantics=include_step_semantics,
        )
    else:
        adaptation_loss = None
    post = model.functional_loss(
        probe_features,
        probe_mask,
        probe_outcomes,
        fast,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    logits = model.functional_logits(
        probe_features,
        probe_mask,
        fast,
        include_direction=include_direction,
        include_step_semantics=include_step_semantics,
    )
    return {
        "support_family": support.generator_family,
        "probe_family": probe.generator_family,
        "support_transition": support.transition_type,
        "probe_transition": probe.transition_type,
        "pre_loss": float(pre.item()),
        "post_loss": float(post.item()),
        "online_loss_auc": float((0.5 * (pre + post)).item()),
        "balanced_accuracy": _balanced_accuracy(logits[0], probe_outcomes[0]),
        "adaptation_loss": adaptation_loss,
        "fast_step": state[0].step,
        "support_trace_sha256": _tensor_digest(support.step_features),
        "support_mask_sha256": _tensor_digest(support.step_mask),
        "probe_outcome_sha256": _tensor_digest(probe.outcomes),
        "adaptation_outcome_sha256": _tensor_digest(support_outcomes),
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if len(records) != 48:
        raise ValueError("V12 arm aggregation requires 48 panel-family records")
    return {
        "online_loss_auc": sum(float(row["online_loss_auc"]) for row in records) / 48.0,
        "balanced_accuracy": sum(float(row["balanced_accuracy"]) for row in records) / 48.0,
        "mean_pre_loss": sum(float(row["pre_loss"]) for row in records) / 48.0,
        "mean_post_loss": sum(float(row["post_loss"]) for row in records) / 48.0,
    }


def _representation_report(
    model: OMLNaturalTraceRepresentation,
    rows: Sequence[EncodedMechanism],
    contrasts: EncodedContrasts,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    features, mask, _ = _batch(rows, device)
    with torch.no_grad():
        codes = model.encode(features, mask)
        contrast = contrasts.to(device)
        contrast_codes = model.encode(contrast.step_features, contrast.step_mask)
        padded_features = torch.cat(
            (features[:1], torch.zeros((*features[:1].shape[:2], 2, features.shape[-1]), device=device)),
            dim=2,
        )
        padded_mask = torch.cat(
            (mask[:1], torch.zeros((*mask[:1].shape[:2], 2), device=device, dtype=torch.bool)),
            dim=2,
        )
        original = model.encode(features[:1], mask[:1])
        repeated = model.encode(features[:1].clone(), mask[:1].clone())
        padded = model.encode(padded_features, padded_mask)
    paraphrase = torch.linalg.vector_norm(contrast_codes[:, 0] - contrast_codes[:, 1], dim=-1)
    reordered = torch.linalg.vector_norm(contrast_codes[:, 0] - contrast_codes[:, 2], dim=-1)
    report = representation_metrics(codes)
    raw_codes = codes.detach().cpu().contiguous()
    report.update(
        {
            "mean_same_procedure_paraphrase_distance": float(paraphrase.mean().item()),
            "mean_same_multiset_reorder_distance": float(reordered.mean().item()),
            "reorder_minus_paraphrase_distance": float((reordered.mean() - paraphrase.mean()).item()),
            "padding_exact": bool(torch.equal(original, padded)),
            "repeat_exact": bool(torch.equal(original, repeated)),
            "raw_code_shape": list(raw_codes.shape),
            "raw_code_sha256": _tensor_digest(raw_codes),
            "raw_code_tensor": raw_codes.tolist(),
        }
    )
    return report


def _evaluate(
    system: OmlSystem,
    rows: Sequence[EncodedMechanism],
    contrasts: EncodedContrasts,
) -> dict[str, Any]:
    panels = _panel_pairs(rows)
    configurations = {
        ARM_SECOND_ONLINE: (system.second_order.model, True, False, True, True),
        ARM_FIRST_ONLINE: (system.first_order.model, True, False, True, True),
        ARM_SOURCE_ONLINE: (system.source.model, True, False, True, True),
        ARM_SECOND_NO_UPDATE: (system.second_order.model, False, False, True, True),
        ARM_FIRST_NO_UPDATE: (system.first_order.model, False, False, True, True),
        ARM_SOURCE_NO_UPDATE: (system.source.model, False, False, True, True),
        ARM_SHUFFLED: (system.second_order.model, True, True, True, True),
        ARM_DIRECTION_REMOVED: (system.second_order.model, True, False, False, True),
        ARM_SEMANTICS_REMOVED: (system.second_order.model, True, False, True, False),
    }
    by_arm: dict[str, list[dict[str, Any]]] = {name: [] for name in configurations}
    panel_reports = []
    with torch.enable_grad():
        for panel_index, pairs in enumerate(panels):
            arm_reports = {}
            for name, (model, enabled, shuffled, direction, semantics) in configurations.items():
                model.eval()
                records = [
                    _evaluate_pair(
                        model,
                        support,
                        probe,
                        updates_enabled=enabled,
                        shuffle_outcomes=shuffled,
                        include_direction=direction,
                        include_step_semantics=semantics,
                    )
                    for support, probe in pairs
                ]
                by_arm[name].extend(records)
                arm_reports[name] = {
                    "online_loss_auc": sum(row["online_loss_auc"] for row in records) / 12.0,
                    "balanced_accuracy": sum(row["balanced_accuracy"] for row in records) / 12.0,
                    "records": records,
                }
            panel_reports.append(
                {
                    "panel": panel_index,
                    "pairing": "same_transition" if panel_index < 2 else "cross_transition",
                    "arms": arm_reports,
                }
            )
    aggregate = {name: _aggregate_records(records) for name, records in by_arm.items()}
    family_improvements = {}
    transition_improvements: dict[str, int] = {}
    for family in sorted({row.generator_family for row in rows}):
        second = [row for row in by_arm[ARM_SECOND_ONLINE] if row["support_family"] == family]
        source = [row for row in by_arm[ARM_SOURCE_ONLINE] if row["support_family"] == family]
        improved = (
            sum(row["online_loss_auc"] for row in second) / len(second)
            < sum(row["online_loss_auc"] for row in source) / len(source)
        )
        transition = second[0]["support_transition"]
        family_improvements[family] = {"improved": improved, "transition": transition}
        transition_improvements[transition] = transition_improvements.get(transition, 0) + int(improved)
    panel_comparison = []
    for panel in panel_reports:
        second_auc = panel["arms"][ARM_SECOND_ONLINE]["online_loss_auc"]
        first_auc = panel["arms"][ARM_FIRST_ONLINE]["online_loss_auc"]
        panel_comparison.append(
            {
                "panel": panel["panel"],
                "improved": second_auc < first_auc,
                "nonregressed": second_auc <= first_auc,
                "second_order_auc": second_auc,
                "first_order_auc": first_auc,
            }
        )
    report = {
        "mechanism_count": len(rows),
        "panels": panel_reports,
        "arms": aggregate,
        "panel_comparison": panel_comparison,
        "improved_panel_count": sum(int(row["improved"]) for row in panel_comparison),
        "all_panels_nonregressed": all(row["nonregressed"] for row in panel_comparison),
        "family_improvements": family_improvements,
        "improved_family_count": sum(int(row["improved"]) for row in family_improvements.values()),
        "transition_improved_family_counts": transition_improvements,
        "representation": _representation_report(system.second_order.model, rows, contrasts),
    }
    full_records = by_arm[ARM_SECOND_ONLINE]
    shuffled_records = by_arm[ARM_SHUFFLED]
    report["shuffled_control_integrity"] = {
        "permutation": list(SHUFFLE_PERMUTATION),
        "same_trace_order": all(
            left["support_trace_sha256"] == right["support_trace_sha256"]
            and left["support_mask_sha256"] == right["support_mask_sha256"]
            for left, right in zip(full_records, shuffled_records, strict=True)
        ),
        "same_probe_labels": all(
            left["probe_outcome_sha256"] == right["probe_outcome_sha256"]
            for left, right in zip(full_records, shuffled_records, strict=True)
        ),
        "all_support_associations_changed": all(
            left["adaptation_outcome_sha256"]
            != right["adaptation_outcome_sha256"]
            for left, right in zip(full_records, shuffled_records, strict=True)
        ),
        "balanced_multiset_preserved": True,
    }
    return report


def _development_gate(metrics: Mapping[str, Any], identities_exact: bool) -> bool:
    try:
        arms = metrics["arms"]
        second = arms[ARM_SECOND_ONLINE]
        first = arms[ARM_FIRST_ONLINE]
        source = arms[ARM_SOURCE_ONLINE]
        second_no = arms[ARM_SECOND_NO_UPDATE]
        source_no = arms[ARM_SOURCE_NO_UPDATE]
        shuffled = arms[ARM_SHUFFLED]
        direction = arms[ARM_DIRECTION_REMOVED]
        semantics = arms[ARM_SEMANTICS_REMOVED]
        representation = metrics["representation"]
        protocol = metrics["protocol_invariants"]
        shuffled_integrity = metrics["shuffled_control_integrity"]
        second_auc = float(second["online_loss_auc"])
        second_accuracy = float(second["balanced_accuracy"])
        return bool(
            identities_exact is True
            and metrics["mechanism_count"] == 48
            and second_auc < float(first["online_loss_auc"])
            and second_auc <= 0.95 * float(first["online_loss_auc"])
            and second_auc < float(source["online_loss_auc"])
            and second_auc < float(second_no["online_loss_auc"])
            and metrics["improved_panel_count"] >= 3
            and metrics["all_panels_nonregressed"] is True
            and float(second_no["balanced_accuracy"]) >= 0.75
            and float(second_no["balanced_accuracy"]) >= float(source_no["balanced_accuracy"]) + 0.10
            and second_auc < float(shuffled["online_loss_auc"])
            and (
                second_accuracy - float(direction["balanced_accuracy"]) >= LESION_ACCURACY_MARGIN
                or float(direction["online_loss_auc"]) - second_auc >= LESION_LOSS_MARGIN
            )
            and (
                second_accuracy - float(semantics["balanced_accuracy"]) >= LESION_ACCURACY_MARGIN
                or float(semantics["online_loss_auc"]) - second_auc >= LESION_LOSS_MARGIN
            )
            and _representation_gate(representation)
            and metrics["improved_family_count"] >= 9
            and set(metrics["transition_improved_family_counts"]) == {
                "insertion", "removal", "reorder", "replacement"
            }
            and all(
                int(value) >= 2
                for value in metrics["transition_improved_family_counts"].values()
            )
            and representation["padding_exact"] is True
            and representation["repeat_exact"] is True
            and protocol["paired_starts_exact"] is True
            and protocol["paired_data_and_exposure_exact"] is True
            and protocol["fast_head_only_inner_mutation"] is True
            and protocol["rln_only_outer_mutation"] is True
            and protocol["source_unchanged"] is True
            and protocol["final_sealed_during_development"] is True
            and protocol["learner_boundary_exact"] is True
            and protocol["public_overlap_audit_passed"] is True
            and protocol["inner_connectivity"]["raw_loss_equal"] is True
            and protocol["inner_connectivity"]["raw_fast_gradient_equal"] is True
            and protocol["inner_connectivity"]["updated_fast_weight_equal"] is True
            and protocol["inner_connectivity"]["adamw_moments_equal"] is True
            and protocol["inner_connectivity"]["second_order_hessian_path_nonzero"] is True
            and protocol["inner_connectivity"]["first_order_hessian_path_absent"] is True
            and protocol["direct_outer_gradients_nonzero"] is True
            and float(protocol["full_split_maximum_delta"]) <= 1.0e-6
            and shuffled_integrity["same_trace_order"] is True
            and shuffled_integrity["same_probe_labels"] is True
            and shuffled_integrity["all_support_associations_changed"] is True
            and shuffled_integrity["balanced_multiset_preserved"] is True
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _representation_gate(representation: Mapping[str, Any]) -> bool:
    try:
        return bool(
            float(representation["effective_rank"]) >= 8.0
            and float(representation["mean_off_diagonal_cosine"]) <= 0.95
            and float(representation["fraction_distinct_pairs_above_0_999"]) <= 0.10
            and representation["finite_nonzero_variance_dimensions"] == 32
            and float(representation["minimum_dimension_variance"]) > 0.0
            and float(representation["reorder_minus_paraphrase_distance"]) >= 0.05
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _development_classification(
    metrics: Mapping[str, Any], *, authorized: bool
) -> str:
    if authorized:
        return "DEVELOPMENT_GATE_PASSED"
    try:
        arms = metrics["arms"]
        second_auc = float(arms[ARM_SECOND_ONLINE]["online_loss_auc"])
        first_auc = float(arms[ARM_FIRST_ONLINE]["online_loss_auc"])
        loss_signal = bool(
            second_auc < first_auc
            and second_auc <= 0.95 * first_auc
            and second_auc < float(arms[ARM_SOURCE_ONLINE]["online_loss_auc"])
            and second_auc < float(arms[ARM_SECOND_NO_UPDATE]["online_loss_auc"])
        )
        if loss_signal and not _representation_gate(metrics["representation"]):
            return "DEVELOPMENT_SHORTCUT_CODE_COLLAPSE"
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return "DEVELOPMENT_NOT_SUPPORTED"


def _identity_report(
    system: OmlSystem,
    *,
    foundation_before: str,
    foundation_after: str,
    sources_before: Mapping[str, str],
    sources_after: Mapping[str, str],
    second_before_evaluation: str,
    first_before_evaluation: str,
    source_before_evaluation: str,
) -> dict[str, Any]:
    report = {
        "foundation_before": foundation_before,
        "foundation_after": foundation_after,
        "sources_before": dict(sources_before),
        "sources_after": dict(sources_after),
        "second_before_evaluation": second_before_evaluation,
        "second_after_evaluation": _model_digest(system.second_order.model),
        "first_before_evaluation": first_before_evaluation,
        "first_after_evaluation": _model_digest(system.first_order.model),
        "source_before_evaluation": source_before_evaluation,
        "source_after_evaluation": _model_digest(system.source.model),
    }
    report["exact"] = bool(
        foundation_before == foundation_after
        and sources_before == sources_after
        and report["second_before_evaluation"] == report["second_after_evaluation"]
        and report["first_before_evaluation"] == report["first_after_evaluation"]
        and report["source_before_evaluation"] == report["source_after_evaluation"]
    )
    return report


def _checkpoint_payload(
    system: OmlSystem,
    *,
    step_width: int,
    qwen_digest: str,
    sources: Mapping[str, str],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "identity": IDENTITY,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "completed_updates": OUTER_UPDATES,
        "step_width": step_width,
        "qwen_digest": qwen_digest,
        "source_hashes": dict(sources),
        "development_metrics": dict(metrics),
        "initial_model_digest": system.initial_model_digest,
        "second_order_state": {name: value.detach().cpu() for name, value in system.second_order.model.state_dict().items()},
        "first_order_state": {name: value.detach().cpu() for name, value in system.first_order.model.state_dict().items()},
        "source_state": {name: value.detach().cpu() for name, value in system.source.model.state_dict().items()},
        "second_order_digest": _model_digest(system.second_order.model),
        "first_order_digest": _model_digest(system.first_order.model),
        "source_digest": _model_digest(system.source.model),
    }


def _validate_final_authorization(
    development: Mapping[str, Any], checkpoint_path: Path
) -> None:
    identity = development.get("identity_checks", {})
    valid = bool(
        development.get("identity") == IDENTITY
        and development.get("phase") == "train-development"
        and development.get("classification") == "DEVELOPMENT_GATE_PASSED"
        and development.get("development_authorized") is True
        and development.get("source_hashes") == _source_hashes()
        and development.get("checkpoint_sha256") == _sha256(checkpoint_path)
        and identity.get("exact") is True
        and isinstance(development.get("development_metrics"), dict)
        and _development_gate(development["development_metrics"], True)
    )
    v5._validate_final_authorization(valid)


def _load_checkpoint_system(checkpoint: Mapping[str, Any], device: torch.device) -> OmlSystem:
    system = _build_system(int(checkpoint["step_width"]), device)
    system.second_order.model.load_state_dict(checkpoint["second_order_state"], strict=True)
    system.first_order.model.load_state_dict(checkpoint["first_order_state"], strict=True)
    system.source.model.load_state_dict(checkpoint["source_state"], strict=True)
    system.second_order.outer_updates = OUTER_UPDATES
    system.first_order.outer_updates = OUTER_UPDATES
    if (
        _model_digest(system.second_order.model) != checkpoint["second_order_digest"]
        or _model_digest(system.first_order.model) != checkpoint["first_order_digest"]
        or _model_digest(system.source.model) != checkpoint["source_digest"]
    ):
        raise RuntimeError("V12 checkpoint model digest mismatch")
    return system


def _run_train_development(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.development_result)
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError("V12 train-development identity is already consumed")
    sources_before = _source_hashes()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    corpus = build_scaled_oml_natural_trace_v12()
    if corpus.final_is_open:
        raise RuntimeError("V12 final partition opened before development")
    try:
        _ = corpus.final
    except FinalPartitionSealedError:
        pass
    else:
        raise RuntimeError("V12 final seal did not fail closed")
    outer_mechanisms = tuple(
        corpus.train_outer(update, slot)
        for update in range(TRAIN_OUTER_UPDATES)
        for slot in range(TRAIN_OUTER_SLOTS)
    )
    overlap_audit = _public_overlap_audit(
        corpus.train_inner, outer_mechanisms, corpus.development
    )
    inner_indices = {
        mechanism.metadata.mechanism_ref: index
        for index, mechanism in enumerate(corpus.train_inner)
    }
    inner_schedule = tuple(
        tuple(
            inner_indices[mechanism.metadata.mechanism_ref]
            for mechanism in corpus.train_inner_for_update(update)
        )
        for update in range(TRAIN_OUTER_UPDATES)
    )
    for update, indices in enumerate(inner_schedule):
        families = {
            corpus.train_inner[index].metadata.generator_family for index in indices
        }
        if len(families) != INNER_STEPS:
            raise RuntimeError(f"V12 update {update} lost eight-family inner diversity")
    inner_rows = _encode_mechanisms(qwen, corpus.train_inner)
    outer_rows = _encode_mechanisms(qwen, outer_mechanisms)
    development_rows = _encode_mechanisms(qwen, corpus.development)
    development_contrasts = _encode_contrasts(qwen, corpus.development)
    step_width = int(inner_rows[0].step_features.shape[-1])
    system = _build_system(step_width, device)
    partition = system.second_order.model.parameter_partition_report()
    learner_boundary_exact = bool(
        partition["fast_initial_shape"] == (1, 64)
        and partition["fast_initial_outer_owned"] is False
        and partition["outcome_encoder_inputs"] is False
        and all(
            name.startswith(("trace_graph_encoder.", "outcome_trunk."))
            for name in partition["rln_parameter_names"]
        )
    )
    training = _fit(system, inner_rows, outer_rows, inner_schedule)
    second_before = _model_digest(system.second_order.model)
    first_before = _model_digest(system.first_order.model)
    source_before = _model_digest(system.source.model)
    metrics = _evaluate(system, development_rows, development_contrasts)
    full_split_maximum = max(
        float(training["full_split_equivalence"][arm][field])
        for arm in ("second_order", "first_order")
        for field in (
            "objective_absolute_delta",
            "maximum_gradient_absolute_delta",
        )
    )
    metrics["protocol_invariants"] = {
        "paired_starts_exact": True,
        "paired_data_and_exposure_exact": training["paired_data_and_exposure_exact"],
        "fast_head_only_inner_mutation": training["fast_head_only_inner_mutation"],
        "rln_only_outer_mutation": training["rln_only_outer_mutation"],
        "source_unchanged": training["source_unchanged"],
        "final_sealed_during_development": True,
        "learner_boundary_exact": learner_boundary_exact,
        "public_overlap_audit_passed": overlap_audit["passed"],
        "inner_connectivity": training["full_split_equivalence"]["inner_connectivity"],
        "direct_outer_gradients_nonzero": training["full_split_equivalence"][
            "direct_outer_gradients_nonzero"
        ],
        "full_split_maximum_delta": full_split_maximum,
        "parameter_partition": partition,
        "public_overlap_audit": overlap_audit,
    }
    foundation_after = foundation_tensor_digest(qwen.model)
    sources_after = _source_hashes()
    identity = _identity_report(
        system,
        foundation_before=foundation_before,
        foundation_after=foundation_after,
        sources_before=sources_before,
        sources_after=sources_after,
        second_before_evaluation=second_before,
        first_before_evaluation=first_before,
        source_before_evaluation=source_before,
    )
    authorized = _development_gate(metrics, bool(identity["exact"]))
    checkpoint = _checkpoint_payload(
        system,
        step_width=step_width,
        qwen_digest=foundation_before,
        sources=sources_before,
        metrics=metrics,
    )
    result = {
        "identity": IDENTITY,
        "phase": "train-development",
        "classification": _development_classification(metrics, authorized=authorized),
        "development_authorized": authorized,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "frozen_compute": {
            "train_inner_mechanisms": TRAIN_INNER_MECHANISMS,
            "train_outer_mechanisms": TRAIN_OUTER_MECHANISMS,
            "development_mechanisms": DEVELOPMENT_MECHANISMS,
            "final_mechanisms_opened": 0,
            "owner_updates": OUTER_UPDATES,
            "inner_steps": INNER_STEPS,
            "outer_mechanisms_per_update": OUTER_MECHANISMS,
        },
        "training": training,
        "development_metrics": metrics,
        "identity_checks": identity,
        "source_hashes": sources_before,
        "checkpoint": str(checkpoint_path),
        "environment": v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "interpretation": (
            "This tests scaled OML natural-trace representations on generated software-workflow "
            "families only; it is not arbitrary reasoning, AGI, or deployment authority."
        ),
    }
    if sources_before != sources_after:
        raise RuntimeError("V12 source changed during execution; outputs were not written")
    _atomic_torch_save(checkpoint_path, checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint_path)
    _atomic_json(result_path, result)
    print(json.dumps({
        "classification": result["classification"],
        "development_result": str(result_path),
        "checkpoint": str(checkpoint_path),
        "second_order_auc": metrics["arms"][ARM_SECOND_ONLINE]["online_loss_auc"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True), flush=True)


def _run_final(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    development_path = Path(args.development_result)
    result_path = Path(args.final_result)
    if result_path.exists():
        raise FileExistsError("V12 final identity is already consumed")
    sources_before = _source_hashes()
    development = json.loads(development_path.read_text(encoding="utf-8"))
    _validate_final_authorization(development, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("identity") != IDENTITY
        or checkpoint.get("source_hashes") != sources_before
        or checkpoint.get("development_metrics") != development.get("development_metrics")
        or checkpoint.get("completed_updates") != OUTER_UPDATES
    ):
        raise RuntimeError("V12 checkpoint/development binding mismatch")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    qwen = v5._load_qwen(args.model_path)
    foundation_before = foundation_tensor_digest(qwen.model)
    if foundation_before != checkpoint["qwen_digest"]:
        raise RuntimeError("V12 frozen Qwen digest changed")
    corpus = build_scaled_oml_natural_trace_v12(include_final=True)
    if not corpus.final_is_open or len(corpus.final) != FINAL_MECHANISMS:
        raise RuntimeError("V12 authorized final partition is incomplete")
    rows = _encode_mechanisms(qwen, corpus.final)
    contrasts = _encode_contrasts(qwen, corpus.final)
    system = _load_checkpoint_system(checkpoint, device)
    second_before = _model_digest(system.second_order.model)
    first_before = _model_digest(system.first_order.model)
    source_before = _model_digest(system.source.model)
    metrics = _evaluate(system, rows, contrasts)
    metrics["protocol_invariants"] = copy.deepcopy(
        checkpoint["development_metrics"]["protocol_invariants"]
    )
    foundation_after = foundation_tensor_digest(qwen.model)
    sources_after = _source_hashes()
    identity = _identity_report(
        system,
        foundation_before=foundation_before,
        foundation_after=foundation_after,
        sources_before=sources_before,
        sources_after=sources_after,
        second_before_evaluation=second_before,
        first_before_evaluation=first_before,
        source_before_evaluation=source_before,
    )
    supported = _development_gate(metrics, bool(identity["exact"]))
    result = {
        "identity": IDENTITY,
        "phase": "final",
        "classification": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "supported": supported,
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "development_result_sha256": _sha256(development_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_metrics": metrics,
        "identity_checks": identity,
        "source_hashes": sources_before,
        "environment": v5._environment_record(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_angler_gpu_bytes": torch.cuda.max_memory_allocated(device),
    }
    if sources_before != sources_after:
        raise RuntimeError("V12 source changed during final evaluation")
    _atomic_json(result_path, result)
    print(json.dumps({
        "classification": result["classification"],
        "final_result": str(result_path),
        "second_order_auc": metrics["arms"][ARM_SECOND_ONLINE]["online_loss_auc"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train-dev", "final"), required=True)
    parser.add_argument("--model-path", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint", default="/opt/angler/results/scaled-oml-natural-trace-representation-v12.pt")
    parser.add_argument("--development-result", default="/opt/angler/results/scaled-oml-natural-trace-representation-v12-development.json")
    parser.add_argument("--final-result", default="/opt/angler/results/scaled-oml-natural-trace-representation-v12-final.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "train-dev":
        _run_train_development(args)
    else:
        _run_final(args)


if __name__ == "__main__":
    main()
