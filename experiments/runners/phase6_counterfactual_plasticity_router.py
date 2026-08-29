"""Anonymous counterfactual plasticity routing for the phase-6 relation cluster.

V14 isolates one causal question left by the all-active V13 pilot: can a
learned, identity-free update rule place incompatible public updates into
different plastic cells?  Both paired arms retain the exact V13 four-cell
controller and its anchored read composer.  They differ only in write credit:
one allocates every stream uniformly, while the other learns soft allocation
from detached public loss and gradient geometry by differentiating through
withheld-stream virtual updates.

The router never receives cell, stream, package, task, component, motif, or
partition identities.  It cannot mutate the controller through its meta-loss;
all virtual controller state is supplied through ``torch.func.functional_call``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.evaluators.software_pipeline_reconstruction_suite import (
    SoftwarePipelineStream,
    software_pipeline_mechanism_partition,
)
from experiments.runners import phase6_software_pipeline_reconstruction as v13


_PROTOCOL_ID = "phase6.public-counterfactual-plasticity-router.paired.v14"
_CHECKPOINT_VERSION = "angler.phase6-counterfactual-plasticity-router.v1"
_DIGEST_DOMAIN = b"project-angler.counterfactual-plasticity-router.v1\x00"
_CELL_COUNT = 4
_STREAMS_PER_UPDATE = 8
_ROWS_PER_STREAM = 4
_UPDATES_PER_ARM = 80
_ROUTER_LOCAL_FEATURES = 5
_ROUTER_ENRICHED_FEATURES = 4 * _ROUTER_LOCAL_FEATURES
_ROUTER_HIDDEN_WIDTH = 48
_ENCODER_LEARNING_RATE = 3.0e-4
_HEAD_LEARNING_RATE = 1.0e-3
_COMPOSER_LEARNING_RATE = 1.0e-3
_ROUTER_LEARNING_RATE = 1.0e-3
_CELL_DIRECTION_CLIP = 5.0
_COMPOSER_GRADIENT_CLIP = 5.0
_META_DIFFERENCE_SCALE = 1.0e4
_USAGE_KL_WEIGHT = 0.01
_REPLICATE_SEEDS = (
    (2_026_083_201, 2_026_083_202, 2_026_083_203, 2_026_083_204),
    (2_026_083_211, 2_026_083_212, 2_026_083_213, 2_026_083_214),
    (2_026_083_221, 2_026_083_222, 2_026_083_223, 2_026_083_224),
)
_TRAIN_TOPOLOGY_BASE = 5_001_000_001
_TRAIN_SURFACE_BASE = 5_041_000_001
_PANEL_A_TOPOLOGY_BASE = 5_081_000_001
_PANEL_A_SURFACE_BASE = 5_121_000_001
_PANEL_A_RERENDER_SURFACE_BASE = 5_161_000_001
_PANEL_B_TOPOLOGY_BASE = 5_201_000_001
_PANEL_B_SURFACE_BASE = 5_241_000_001
_ADAPTATION_TOPOLOGY_BASE = 5_281_000_001
_ADAPTATION_SURFACE_BASE = 5_321_000_001
_PROBE_TOPOLOGY_BASE = 5_361_000_001
_PROBE_SURFACE_BASE = 5_401_000_001
_REPLICATE_SEED_STRIDE = 10_000_000


class CounterfactualPlasticityRouter(nn.Module):
    """Score an anonymous cell-by-stream set without identity features.

    The final scorer is exactly zero initialized.  Consequently the first
    learned-arm allocation is exactly uniform, while later allocations are an
    unconstrained softmax across cells and have no positive floor.
    """

    def __init__(
        self,
        *,
        local_features: int = _ROUTER_LOCAL_FEATURES,
        hidden_width: int = _ROUTER_HIDDEN_WIDTH,
    ) -> None:
        super().__init__()
        if (
            isinstance(local_features, bool)
            or not isinstance(local_features, int)
            or local_features <= 0
            or isinstance(hidden_width, bool)
            or not isinstance(hidden_width, int)
            or hidden_width <= 0
        ):
            raise ValueError("router dimensions must be positive integers")
        self.local_features = local_features
        self.hidden_width = hidden_width
        enriched = 4 * local_features
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(enriched),
            nn.Linear(enriched, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.scorer = nn.Linear(hidden_width, 1, bias=False)
        nn.init.zeros_(self.scorer.weight)

    def forward(
        self,
        local_evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            local_evidence.ndim != 3
            or local_evidence.shape[0] != _CELL_COUNT
            or local_evidence.shape[1] <= 0
            or local_evidence.shape[2] != self.local_features
            or not local_evidence.is_floating_point()
            or not bool(torch.isfinite(local_evidence).all().item())
        ):
            raise ValueError(
                "router evidence must be finite [four cells, streams, features]"
            )
        # Geometry is evidence, not a path back into controller parameters.
        local = local_evidence.detach()
        cell_mean = local.mean(dim=1, keepdim=True).expand_as(local)
        stream_mean = local.mean(dim=0, keepdim=True).expand_as(local)
        global_mean = local.mean(dim=(0, 1), keepdim=True).expand_as(local)
        enriched = torch.cat((local, cell_mean, stream_mean, global_mean), dim=-1)
        logits = self.scorer(self.local_encoder(enriched)).squeeze(-1)
        allocations = torch.softmax(logits, dim=0)
        return allocations, logits, enriched


@dataclass(frozen=True, slots=True)
class CellLocalEvidence:
    """Detached single-cell public consequences for one update batch."""

    features: torch.Tensor
    losses: torch.Tensor
    gradient_norms: torch.Tensor
    prediction_strengths: torch.Tensor
    ensemble_stream_losses: torch.Tensor
    entropic_base_weights: torch.Tensor
    cell_parameter_names: tuple[tuple[str, ...], ...]
    gradients: tuple[tuple[tuple[torch.Tensor, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class CounterfactualMetaResult:
    """Router-only gradients and complete withheld-stream accounting."""

    gradients: tuple[torch.Tensor, ...]
    objective: float
    unscaled_mean_post_minus_pre: float
    scaled_mean_post_minus_pre: float
    aggregate_usage_kl: float
    fold_post_minus_pre: tuple[float, ...]
    fold_pre_losses: tuple[float, ...]
    fold_post_losses: tuple[float, ...]
    heldout_indices: tuple[int, ...]
    seen_indices: tuple[tuple[int, ...], ...]
    fold_allocations: tuple[tuple[tuple[float, ...], ...], ...]
    fold_direction_norms: tuple[tuple[float, ...], ...]
    fold_clipped_direction_norms: tuple[tuple[float, ...], ...]


class _HeldoutEnsembleObjective(nn.Module):
    """Functional-call seam around the unchanged V13 ensemble reader."""

    def __init__(self, controller: v13.CapacityMatchedClusterController) -> None:
        super().__init__()
        self.controller = controller

    def forward(self, stream: SoftwarePipelineStream) -> torch.Tensor:
        return _ensemble_stream_objective(self.controller, stream)


def _relation_row_loss(row: v13.PublicRelationCreditRow) -> torch.Tensor:
    return (
        row.instance_loss
        + v13._RELATION_CREDIT_SEPARATION_WEIGHT * row.separation_loss
    )


def _row_risk_objective(rows: Sequence[v13.PublicRelationCreditRow]) -> torch.Tensor:
    if len(rows) != _ROWS_PER_STREAM:
        raise ValueError("a relation stream must expose exactly four public rows")
    row_losses = torch.stack(tuple(_relation_row_loss(row) for row in rows))
    return v13._anonymous_entropic_row_objective(row_losses)[0]


def _ensemble_stream_objective(
    controller: v13.CapacityMatchedClusterController,
    stream: SoftwarePipelineStream,
) -> torch.Tensor:
    return _row_risk_objective(v13.public_relation_credit_rows(controller, stream))


def _ensemble_batch_objective(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(streams) != _STREAMS_PER_UPDATE:
        raise ValueError("the plasticity update requires exactly eight streams")
    stream_losses = torch.stack(
        tuple(_ensemble_stream_objective(controller, stream) for stream in streams)
    )
    aggregate = v13._relation_credit_stream_objective(
        stream_losses,
        stage="relation",
    )
    return aggregate[0], stream_losses, aggregate[3]


def _cell_parameter_groups(
    controller: v13.CapacityMatchedClusterController,
) -> tuple[tuple[tuple[str, nn.Parameter], ...], ...]:
    named = tuple(controller.named_parameters())
    groups = tuple(
        tuple(
            (name, parameter)
            for name, parameter in named
            if name.startswith(f"relation_cells.{cell_index}.")
        )
        for cell_index in range(_CELL_COUNT)
    )
    if any(not group for group in groups):
        raise RuntimeError("one or more plastic relation cells have no parameters")
    if len({name for group in groups for name, _ in group}) != sum(
        len(group) for group in groups
    ):
        raise RuntimeError("plastic relation cell ownership overlaps")
    return groups


def _gradient_norm(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    if not gradients:
        raise ValueError("gradient norm requires a nonempty direction")
    total = gradients[0].new_zeros(())
    for gradient in gradients:
        total = total + gradient.square().sum()
    return total.clamp_min(0.0).sqrt()


def _gradient_cosine(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(left) != len(right) or not left:
        raise ValueError("gradient cosine requires aligned nonempty directions")
    dot = left[0].new_zeros(())
    left_square = left[0].new_zeros(())
    right_square = left[0].new_zeros(())
    for left_value, right_value in zip(left, right, strict=True):
        dot = dot + (left_value * right_value).sum()
        left_square = left_square + left_value.square().sum()
        right_square = right_square + right_value.square().sum()
    denominator = left_square.sqrt() * right_square.sqrt()
    return torch.where(
        denominator > 0.0,
        dot / denominator.clamp_min(torch.finfo(dot.dtype).tiny),
        dot.new_zeros(()),
    )


def collect_cell_local_evidence(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
) -> CellLocalEvidence:
    """Measure each cell independently of the read composer, without mutation."""

    if not isinstance(controller, v13.CapacityMatchedClusterController):
        raise TypeError("cell-local evidence requires a V13 cluster controller")
    if not streams:
        raise ValueError("cell-local evidence requires public streams")
    if controller._relation_diagnostic_lesion is not None:
        raise RuntimeError("cell-local evidence requires an unlesioned controller")
    before = v13.software_pipeline_model_digest(controller)
    groups = _cell_parameter_groups(controller)
    all_losses: list[tuple[torch.Tensor, ...]] = []
    all_strengths: list[tuple[torch.Tensor, ...]] = []
    all_gradients: list[tuple[tuple[torch.Tensor, ...], ...]] = []
    try:
        for cell_index, group in enumerate(groups):
            controller.set_relation_diagnostic_lesion("single", cell_index)
            cell_losses = []
            cell_strengths = []
            cell_gradients = []
            parameters = tuple(parameter for _, parameter in group)
            for stream in streams:
                rows = v13.public_relation_credit_rows(controller, stream)
                loss = _row_risk_objective(rows)
                raw = torch.autograd.grad(
                    loss,
                    parameters,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
                gradients = tuple(
                    torch.zeros_like(parameter)
                    if gradient is None
                    else gradient.detach()
                    for parameter, gradient in zip(parameters, raw, strict=True)
                )
                strength = torch.stack(
                    tuple(
                        (row.positive_margin - row.negative_margin).detach().abs()
                        for row in rows
                    )
                ).mean()
                cell_losses.append(loss.detach())
                cell_strengths.append(strength)
                cell_gradients.append(gradients)
            all_losses.append(tuple(cell_losses))
            all_strengths.append(tuple(cell_strengths))
            all_gradients.append(tuple(cell_gradients))
    finally:
        controller.set_relation_diagnostic_lesion(None)
    if v13.software_pipeline_model_digest(controller) != before:
        raise RuntimeError("cell-local evidence mutated controller state")
    losses = torch.stack(tuple(torch.stack(values) for values in all_losses))
    strengths = torch.stack(tuple(torch.stack(values) for values in all_strengths))
    norms = torch.stack(
        tuple(
            torch.stack(tuple(_gradient_norm(direction) for direction in cell))
            for cell in all_gradients
        )
    )
    cosine_means = torch.zeros_like(norms)
    cosine_minima = torch.zeros_like(norms)
    if len(streams) > 1:
        for cell_index, cell in enumerate(all_gradients):
            for stream_index, direction in enumerate(cell):
                cosines = torch.stack(
                    tuple(
                        _gradient_cosine(direction, other)
                        for other_index, other in enumerate(cell)
                        if other_index != stream_index
                    )
                )
                cosine_means[cell_index, stream_index] = cosines.mean()
                cosine_minima[cell_index, stream_index] = cosines.min()
    features = torch.stack(
        (
            losses,
            torch.log(norms.clamp_min(torch.finfo(norms.dtype).tiny)),
            cosine_means,
            cosine_minima,
            strengths,
        ),
        dim=-1,
    ).detach()
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("cell-local public evidence is non-finite")
    with torch.no_grad():
        ensemble_stream_losses = torch.stack(
            tuple(
                _ensemble_stream_objective(controller, stream)
                for stream in streams
            )
        )
        entropic_base_weights = v13._relation_credit_stream_objective(
            ensemble_stream_losses,
            stage="relation",
        )[3]
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("cell-local evidence populated controller .grad fields")
    return CellLocalEvidence(
        features=features,
        losses=losses.detach(),
        gradient_norms=norms.detach(),
        prediction_strengths=strengths.detach(),
        ensemble_stream_losses=ensemble_stream_losses.detach(),
        entropic_base_weights=entropic_base_weights.detach(),
        cell_parameter_names=tuple(
            tuple(name for name, _ in group) for group in groups
        ),
        gradients=tuple(all_gradients),
    )


def _features_for_stream_indices(
    evidence: CellLocalEvidence,
    stream_indices: Sequence[int],
) -> torch.Tensor:
    """Recompute cosine summaries using only the explicitly visible streams."""

    indices = tuple(int(index) for index in stream_indices)
    stream_count = evidence.features.shape[1]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= stream_count
    ):
        raise ValueError("router feature subset is invalid")
    index_tensor = torch.tensor(
        indices,
        device=evidence.features.device,
        dtype=torch.long,
    )
    losses = evidence.losses.index_select(1, index_tensor)
    norms = evidence.gradient_norms.index_select(1, index_tensor)
    strengths = evidence.prediction_strengths.index_select(1, index_tensor)
    cosine_means = torch.zeros_like(norms)
    cosine_minima = torch.zeros_like(norms)
    if len(indices) > 1:
        for cell_index in range(_CELL_COUNT):
            for local_index, stream_index in enumerate(indices):
                direction = evidence.gradients[cell_index][stream_index]
                cosines = torch.stack(
                    tuple(
                        _gradient_cosine(
                            direction,
                            evidence.gradients[cell_index][other_stream_index],
                        )
                        for other_stream_index in indices
                        if other_stream_index != stream_index
                    )
                )
                cosine_means[cell_index, local_index] = cosines.mean()
                cosine_minima[cell_index, local_index] = cosines.min()
    features = torch.stack(
        (
            losses,
            torch.log(norms.clamp_min(torch.finfo(norms.dtype).tiny)),
            cosine_means,
            cosine_minima,
            strengths,
        ),
        dim=-1,
    ).detach()
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("visible-only router features are non-finite")
    return features


def _routed_cell_directions(
    evidence: CellLocalEvidence,
    allocations: torch.Tensor,
    stream_indices: Sequence[int],
) -> tuple[
    tuple[tuple[torch.Tensor, ...], ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    indices = tuple(int(index) for index in stream_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= evidence.features.shape[1]
        or allocations.shape != (_CELL_COUNT, len(indices))
        or allocations.device != evidence.features.device
        or allocations.dtype != evidence.features.dtype
        or not bool(torch.isfinite(allocations).all().item())
        or not torch.allclose(
            allocations.sum(dim=0),
            torch.ones_like(allocations.sum(dim=0)),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    ):
        raise ValueError("routed cell allocations are invalid")
    selected_losses = evidence.ensemble_stream_losses[
        torch.tensor(
            indices,
            device=evidence.ensemble_stream_losses.device,
            dtype=torch.long,
        )
    ]
    # Recompute the anonymous entropic weights on exactly the visible streams.
    # The heldout stream therefore cannot affect a virtual proposal even through
    # the normalization denominator.  Multiplication by four makes uniform
    # allocation exactly reproduce the base-weighted per-cell direction.
    base_weights = v13._relation_credit_stream_objective(
        selected_losses,
        stage="relation",
    )[3].detach()
    directions = []
    raw_norms = []
    for cell_index in range(_CELL_COUNT):
        parameter_count = len(evidence.gradients[cell_index][indices[0]])
        cell_direction = tuple(
            sum(
                (
                    _CELL_COUNT
                    * base_weights[local_index]
                    * allocations[cell_index, local_index]
                    * evidence.gradients[cell_index][stream_index][parameter_index]
                    for local_index, stream_index in enumerate(indices)
                ),
                torch.zeros_like(
                    evidence.gradients[cell_index][indices[0]][parameter_index]
                ),
            )
            for parameter_index in range(parameter_count)
        )
        raw_norm = _gradient_norm(cell_direction)
        directions.append(cell_direction)
        raw_norms.append(raw_norm)
    global_square = raw_norms[0].new_zeros(())
    for direction in directions:
        for value in direction:
            global_square = global_square + value.square().sum()
    global_norm = global_square.clamp_min(0.0).sqrt()
    global_scale = torch.minimum(
        global_norm.new_ones(()),
        global_norm.new_tensor(_CELL_DIRECTION_CLIP)
        / global_norm.clamp_min(torch.finfo(global_norm.dtype).tiny),
    )
    clipped_directions = tuple(
        tuple(value * global_scale for value in direction)
        for direction in directions
    )
    clipped_norms = tuple(
        _gradient_norm(direction) for direction in clipped_directions
    )
    return clipped_directions, tuple(raw_norms), clipped_norms


def functional_heldout_loss(
    controller: v13.CapacityMatchedClusterController,
    stream: SoftwarePipelineStream,
    cell_parameter_names: Sequence[Sequence[str]],
    directions: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Evaluate one ensemble stream after a virtual explicit-SGD cell update."""

    if len(cell_parameter_names) != _CELL_COUNT or len(directions) != _CELL_COUNT:
        raise ValueError("functional update requires exactly four cell directions")
    wrapper = _HeldoutEnsembleObjective(controller)
    state = {name: parameter.detach() for name, parameter in wrapper.named_parameters()}
    buffers = {name: buffer.detach() for name, buffer in wrapper.named_buffers()}
    controller_parameters = dict(controller.named_parameters())
    for names, values in zip(cell_parameter_names, directions, strict=True):
        if len(names) != len(values):
            raise ValueError("functional direction lost parameter alignment")
        for name, direction in zip(names, values, strict=True):
            parameter = controller_parameters.get(name)
            if parameter is None or parameter.shape != direction.shape:
                raise ValueError("functional direction changed controller shape")
            learning_rate = (
                _ENCODER_LEARNING_RATE
                if v13._relation_encoder_parameter_name(name)
                else _HEAD_LEARNING_RATE
            )
            state["controller." + name] = (
                parameter.detach() - learning_rate * direction
            )
    return torch.func.functional_call(
        wrapper,
        (state, buffers),
        (stream,),
        strict=True,
    )


def counterfactual_router_meta_gradients(
    controller: v13.CapacityMatchedClusterController,
    router: CounterfactualPlasticityRouter,
    streams: Sequence[SoftwarePipelineStream],
    evidence: CellLocalEvidence | None = None,
) -> CounterfactualMetaResult:
    """Differentiate eight true withheld-stream virtual-update consequences."""

    if len(streams) != _STREAMS_PER_UPDATE:
        raise ValueError("counterfactual routing requires eight heldout folds")
    if evidence is None:
        evidence = collect_cell_local_evidence(controller, streams)
    if evidence.features.shape != (
        _CELL_COUNT,
        _STREAMS_PER_UPDATE,
        _ROUTER_LOCAL_FEATURES,
    ):
        raise ValueError("counterfactual evidence shape changed")
    before = v13.software_pipeline_model_digest(controller)
    router_parameters = tuple(router.parameters())
    accumulated = tuple(torch.zeros_like(parameter) for parameter in router_parameters)
    fold_deltas = []
    fold_pre = []
    fold_post = []
    fold_allocations = []
    fold_raw_norms = []
    fold_clipped_norms = []
    seen_records = []
    heldout_records = []
    for heldout_index, heldout_stream in enumerate(streams):
        seen = tuple(
            index for index in range(_STREAMS_PER_UPDATE) if index != heldout_index
        )
        visible_features = _features_for_stream_indices(evidence, seen)
        allocations, _, _ = router(visible_features)
        directions, raw_norms, clipped_norms = _routed_cell_directions(
            evidence,
            allocations,
            seen,
        )
        with torch.no_grad():
            pre = _ensemble_stream_objective(controller, heldout_stream).detach()
        post = functional_heldout_loss(
            controller,
            heldout_stream,
            evidence.cell_parameter_names,
            directions,
        )
        delta = post - pre
        fold_gradient = torch.autograd.grad(
            _META_DIFFERENCE_SCALE * delta / _STREAMS_PER_UPDATE,
            router_parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        accumulated = tuple(
            total + value.detach()
            for total, value in zip(accumulated, fold_gradient, strict=True)
        )
        fold_deltas.append(float(delta.detach().item()))
        fold_pre.append(float(pre.item()))
        fold_post.append(float(post.detach().item()))
        fold_allocations.append(
            tuple(
                tuple(float(value) for value in row)
                for row in allocations.detach().tolist()
            )
        )
        fold_raw_norms.append(tuple(float(value.detach().item()) for value in raw_norms))
        fold_clipped_norms.append(
            tuple(float(value.detach().item()) for value in clipped_norms)
        )
        heldout_records.append(heldout_index)
        seen_records.append(seen)
    # Across-experience balance, not compulsory participation on each stream.
    usage_routes = tuple(
        router(
            _features_for_stream_indices(
                evidence,
                tuple(
                    index
                    for index in range(_STREAMS_PER_UPDATE)
                    if index != heldout
                ),
            )
        )[0]
        for heldout in range(_STREAMS_PER_UPDATE)
    )
    aggregate_usage = torch.cat(usage_routes, dim=1).mean(dim=1)
    usage_kl = (
        aggregate_usage
        * torch.log(
            (aggregate_usage * _CELL_COUNT).clamp_min(
                torch.finfo(aggregate_usage.dtype).tiny
            )
        )
    ).sum()
    usage_gradients = torch.autograd.grad(
        _USAGE_KL_WEIGHT * usage_kl,
        router_parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    accumulated = tuple(
        total + value.detach()
        for total, value in zip(accumulated, usage_gradients, strict=True)
    )
    if v13.software_pipeline_model_digest(controller) != before:
        raise RuntimeError("counterfactual functional evaluation mutated controller")
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("counterfactual meta-loss populated controller gradients")
    mean_delta = sum(fold_deltas) / len(fold_deltas)
    return CounterfactualMetaResult(
        gradients=accumulated,
        objective=_META_DIFFERENCE_SCALE * mean_delta
        + _USAGE_KL_WEIGHT * float(usage_kl.detach().item()),
        unscaled_mean_post_minus_pre=mean_delta,
        scaled_mean_post_minus_pre=_META_DIFFERENCE_SCALE * mean_delta,
        aggregate_usage_kl=float(usage_kl.detach().item()),
        fold_post_minus_pre=tuple(fold_deltas),
        fold_pre_losses=tuple(fold_pre),
        fold_post_losses=tuple(fold_post),
        heldout_indices=tuple(heldout_records),
        seen_indices=tuple(seen_records),
        fold_allocations=tuple(fold_allocations),
        fold_direction_norms=tuple(fold_raw_norms),
        fold_clipped_direction_norms=tuple(fold_clipped_norms),
    )


def _composer_gradients(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
) -> tuple[tuple[torch.Tensor, ...], float, tuple[float, ...], float]:
    parameters = tuple(controller.relation_composer.parameters())
    objective, stream_losses, _ = _ensemble_batch_objective(controller, streams)
    raw = torch.autograd.grad(
        objective,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    gradients = tuple(value.detach() for value in raw)
    norm = _gradient_norm(gradients)
    scale = torch.minimum(
        norm.new_ones(()),
        norm.new_tensor(_COMPOSER_GRADIENT_CLIP)
        / norm.clamp_min(torch.finfo(norm.dtype).tiny),
    )
    return (
        tuple(value * scale for value in gradients),
        float(objective.detach().item()),
        tuple(float(value) for value in stream_losses.detach().tolist()),
        float(norm.detach().item()),
    )


def _apply_controller_update(
    controller: v13.CapacityMatchedClusterController,
    evidence: CellLocalEvidence,
    directions: Sequence[Sequence[torch.Tensor]],
    composer_gradients: Sequence[torch.Tensor],
    composer_optimizer: torch.optim.Optimizer,
) -> None:
    named = dict(controller.named_parameters())
    with torch.no_grad():
        for names, values in zip(
            evidence.cell_parameter_names,
            directions,
            strict=True,
        ):
            for name, direction in zip(names, values, strict=True):
                parameter = named[name]
                learning_rate = (
                    _ENCODER_LEARNING_RATE
                    if v13._relation_encoder_parameter_name(name)
                    else _HEAD_LEARNING_RATE
                )
                parameter.add_(direction, alpha=-learning_rate)
    composer_optimizer.zero_grad(set_to_none=True)
    composer_parameters = tuple(controller.relation_composer.parameters())
    if len(composer_parameters) != len(composer_gradients):
        raise RuntimeError("composer gradient ownership changed")
    for parameter, gradient in zip(
        composer_parameters,
        composer_gradients,
        strict=True,
    ):
        parameter.grad = gradient.detach().clone()
    composer_optimizer.step()
    composer_optimizer.zero_grad(set_to_none=True)


def fit_counterfactual_plasticity_batches(
    controller: v13.CapacityMatchedClusterController,
    router: CounterfactualPlasticityRouter,
    stream_batches: Sequence[Sequence[SoftwarePipelineStream]],
    *,
    learned_plasticity: bool,
) -> dict[str, object]:
    """Fit one V14 arm with explicit cell SGD and a separate composer optimizer."""

    if type(learned_plasticity) is not bool:
        raise TypeError("learned_plasticity must be bool")
    if not stream_batches or any(
        len(batch) != _STREAMS_PER_UPDATE for batch in stream_batches
    ):
        raise ValueError("V14 fit requires nonempty eight-stream batches")
    named = tuple(controller.named_parameters())
    mutable_prefixes = ("relation_cells.", "relation_composer.")
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in named
        if not name.startswith(mutable_prefixes)
    }
    composer_optimizer = torch.optim.AdamW(
        controller.relation_composer.parameters(),
        lr=_COMPOSER_LEARNING_RATE,
        weight_decay=0.0,
    )
    # The uniform arm trains a sham router through the same eight consequences;
    # only applying its scores is withheld, matching meta compute and learning.
    router_optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=_ROUTER_LEARNING_RATE,
        weight_decay=0.0,
    )
    initial_router_digest = counterfactual_plasticity_router_digest(router)
    reports = []
    for update_index, batch in enumerate(stream_batches):
        evidence = collect_cell_local_evidence(controller, batch)
        if learned_plasticity:
            allocations = router(evidence.features)[0].detach()
        else:
            allocations = evidence.features.new_full(
                (_CELL_COUNT, _STREAMS_PER_UPDATE),
                1.0 / _CELL_COUNT,
            )
        # This detached allocation is fixed before the current meta consequence;
        # the router update can affect only the next fresh batch.
        directions, raw_norms, clipped_norms = _routed_cell_directions(
            evidence,
            allocations,
            tuple(range(_STREAMS_PER_UPDATE)),
        )
        meta = counterfactual_router_meta_gradients(
            controller,
            router,
            batch,
            evidence,
        )
        composer_gradient, objective, stream_losses, composer_gradient_norm = (
            _composer_gradients(controller, batch)
        )
        # Controller and composer advance before the separately owned router.
        _apply_controller_update(
            controller,
            evidence,
            directions,
            composer_gradient,
            composer_optimizer,
        )
        router_optimizer.zero_grad(set_to_none=True)
        router_parameters = tuple(router.parameters())
        if len(router_parameters) != len(meta.gradients):
            raise RuntimeError("router meta-gradient ownership changed")
        for parameter, gradient in zip(
            router_parameters,
            meta.gradients,
            strict=True,
        ):
            parameter.grad = gradient.detach().clone()
        router_optimizer.step()
        router_optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for parameter in controller.parameters()):
            raise RuntimeError("V14 update leaked persistent controller gradients")
        reports.append(
            {
                "update": update_index,
                "objective": objective,
                "stream_losses": stream_losses,
                "allocations": tuple(
                    tuple(float(value) for value in row)
                    for row in allocations.tolist()
                ),
                "cell_direction_norms": tuple(
                    float(value.detach().item()) for value in raw_norms
                ),
                "cell_clipped_direction_norms": tuple(
                    float(value.detach().item()) for value in clipped_norms
                ),
                "composer_gradient_norm": composer_gradient_norm,
                "meta": {
                    "objective": meta.objective,
                    "unscaled_mean_post_minus_pre": (
                        meta.unscaled_mean_post_minus_pre
                    ),
                    "scaled_mean_post_minus_pre": (
                        meta.scaled_mean_post_minus_pre
                    ),
                    "aggregate_usage_kl": meta.aggregate_usage_kl,
                    "fold_post_minus_pre": meta.fold_post_minus_pre,
                    "heldout_indices": meta.heldout_indices,
                    "seen_indices": meta.seen_indices,
                    "fold_allocations": meta.fold_allocations,
                    "fold_direction_norms": meta.fold_direction_norms,
                    "fold_clipped_direction_norms": (
                        meta.fold_clipped_direction_norms
                    ),
                },
            }
        )
    for name, before in frozen.items():
        if not torch.equal(before, dict(controller.named_parameters())[name].detach()):
            raise RuntimeError(f"V14 fit changed frozen controller parameter: {name}")
    terminal_router_digest = counterfactual_plasticity_router_digest(router)
    return {
        "arm": "learned_plasticity" if learned_plasticity else "uniform_plasticity",
        "optimizer_steps": len(stream_batches),
        "streams": len(stream_batches) * _STREAMS_PER_UPDATE,
        "rows": len(stream_batches) * _STREAMS_PER_UPDATE * _ROWS_PER_STREAM,
        "public_leave_one_stream_out_folds_per_update": _STREAMS_PER_UPDATE,
        "updates": tuple(reports),
        "first_allocation_exact_uniform": all(
            value == 1.0 / _CELL_COUNT for row in reports[0]["allocations"] for value in row
        ),
        "router_initial_digest": initial_router_digest,
        "router_terminal_digest": terminal_router_digest,
        "router_changed": terminal_router_digest != initial_router_digest,
        "cell_update": "explicit_sgd",
        "composer_update": "separate_adamw",
        "router_update": "separate_adamw",
        "router_scores_applied": learned_plasticity,
        "sham_router_compute_matched": not learned_plasticity,
        "controller_before_router_step": True,
        "router_affects_current_batch": False,
        "router_affects_next_batch": learned_plasticity,
        "old_conflict_mixer_present": False,
        "frozen_controller_parameters_unchanged": True,
        "controller_grad_fields_clear": True,
    }


def _seed_binding_digest(
    commitments: Sequence[str],
    seed_batches: Sequence[Sequence[tuple[int, int]]],
) -> str:
    payload = tuple(
        tuple(
            (commitment, int(pair[0]), int(pair[1]))
            for commitment, pair in zip(commitments, batch, strict=True)
        )
        for batch in seed_batches
    )
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    digest.update(json.dumps(payload, separators=(",", ":")).encode("ascii"))
    return "sha256:" + digest.hexdigest()


def counterfactual_plasticity_fit_plan() -> dict[str, object]:
    """Return the fixed three-replicate V14 paired causal plan."""

    commitments = software_pipeline_mechanism_partition("train")[:_STREAMS_PER_UPDATE]
    if len(commitments) != _STREAMS_PER_UPDATE:
        raise RuntimeError("train partition cannot satisfy V14")
    replicates = []
    all_pairs: set[tuple[int, int]] = set()
    for replicate, seeds in enumerate(_REPLICATE_SEEDS):
        offset = _REPLICATE_SEED_STRIDE * replicate
        train_batches = tuple(
            tuple(
                (
                    _TRAIN_TOPOLOGY_BASE + offset + 100_000 * update + 1_000 * stream,
                    _TRAIN_SURFACE_BASE + offset + 100_000 * update + 1_000 * stream,
                )
                for stream in range(_STREAMS_PER_UPDATE)
            )
            for update in range(_UPDATES_PER_ARM)
        )
        panel_a = v13._relation_credit_panel_seed_pairs(
            _PANEL_A_TOPOLOGY_BASE + offset,
            _PANEL_A_SURFACE_BASE + offset,
        )
        panel_a_rerender = v13._relation_credit_panel_seed_pairs(
            _PANEL_A_TOPOLOGY_BASE + offset,
            _PANEL_A_RERENDER_SURFACE_BASE + offset,
        )
        panel_b = v13._relation_credit_panel_seed_pairs(
            _PANEL_B_TOPOLOGY_BASE + offset,
            _PANEL_B_SURFACE_BASE + offset,
        )
        adaptation = v13._relation_credit_panel_seed_pairs(
            _ADAPTATION_TOPOLOGY_BASE + offset,
            _ADAPTATION_SURFACE_BASE + offset,
        )
        probe = v13._relation_credit_panel_seed_pairs(
            _PROBE_TOPOLOGY_BASE + offset,
            _PROBE_SURFACE_BASE + offset,
        )
        current_pairs = {
            pair for batch in train_batches for pair in batch
        } | set(panel_a) | set(panel_a_rerender) | set(panel_b) | set(
            adaptation
        ) | set(probe)
        if (
            len({pair for batch in train_batches for pair in batch})
            != _UPDATES_PER_ARM * _STREAMS_PER_UPDATE
            or len(
                set(panel_a)
                | set(panel_a_rerender)
                | set(panel_b)
                | set(adaptation)
                | set(probe)
            )
            != 5 * _STREAMS_PER_UPDATE
            or all_pairs & current_pairs
        ):
            raise RuntimeError("V14 replicate identities overlap")
        all_pairs.update(current_pairs)
        binding = _seed_binding_digest(commitments, train_batches)
        replicates.append(
            {
                "replicate": replicate,
                "shared_controller_seed": seeds[0],
                "cell_seed": seeds[1],
                "composer_seed": seeds[2],
                "router_seed": seeds[3],
                "arm_order": (
                    ("uniform_plasticity", "learned_plasticity")
                    if replicate % 2 == 0
                    else ("learned_plasticity", "uniform_plasticity")
                ),
                "train_seed_batches": train_batches,
                "panel_a_seed_pairs": panel_a,
                "panel_a_rerender_seed_pairs": panel_a_rerender,
                "panel_b_seed_pairs": panel_b,
                "adaptation_seed_pairs": adaptation,
                "probe_seed_pairs": probe,
                "uniform_stream_binding_digest": binding,
                "learned_stream_binding_digest": binding,
            }
        )
    v13_plan = v13.capacity_matched_relation_cluster_fit_plan()
    v13_pairs = {
        pair
        for replicate in v13_plan["replicates"]
        for batch in replicate["train_seed_batches"]
        for pair in batch
    } | {
        pair
        for replicate in v13_plan["replicates"]
        for key in (
            "panel_a_seed_pairs",
            "panel_a_rerender_seed_pairs",
            "panel_b_seed_pairs",
        )
        for pair in replicate[key]
    }
    if all_pairs & v13_pairs:
        raise RuntimeError("V14 identities overlap V13")
    return {
        "protocol_id": _PROTOCOL_ID,
        "partition": "train",
        "replicate_count": len(replicates),
        "replicates": tuple(replicates),
        "commitments": commitments,
        "stage": "relation",
        "updates_per_arm_per_replicate": _UPDATES_PER_ARM,
        "streams_per_update": _STREAMS_PER_UPDATE,
        "rows_per_stream": _ROWS_PER_STREAM,
        "streams_per_arm_per_replicate": _UPDATES_PER_ARM * _STREAMS_PER_UPDATE,
        "rows_per_arm_per_replicate": (
            _UPDATES_PER_ARM * _STREAMS_PER_UPDATE * _ROWS_PER_STREAM
        ),
        "arms": ("uniform_plasticity", "learned_plasticity"),
        "controller": "unchanged_v13_four_cell_cluster_and_read_composer",
        "router": "zero_start_permutation_equivariant_counterfactual_plasticity",
        "router_local_features": (
            "detached_cell_local_public_loss",
            "detached_log_gradient_norm",
            "detached_within_cell_cosine_mean",
            "detached_within_cell_cosine_minimum",
            "detached_prediction_strength",
        ),
        "router_context": ("cell_mean", "stream_mean", "global_mean"),
        "allocation_axis": "softmax_across_cells",
        "minimum_cell_allocation": None,
        "heldout_folds_per_learned_update": _STREAMS_PER_UPDATE,
        "meta_difference_scale": _META_DIFFERENCE_SCALE,
        "aggregate_usage_kl_weight": _USAGE_KL_WEIGHT,
        "cell_optimizer": "explicit_sgd",
        "encoder_learning_rate": _ENCODER_LEARNING_RATE,
        "head_learning_rate": _HEAD_LEARNING_RATE,
        "cell_direction_global_clip": _CELL_DIRECTION_CLIP,
        "composer_optimizer": "adamw",
        "composer_learning_rate": _COMPOSER_LEARNING_RATE,
        "router_optimizer": "adamw",
        "router_learning_rate": _ROUTER_LEARNING_RATE,
        "one_step_diagnostic": {
            "arms": ("correct", "uniform", "cell_permuted"),
            "update_scope": "cells_only",
            "permutation": "global_one_cell_cycle",
            "composer_step": False,
            "router_step": False,
            "correct_loss": "strictly_below_both_controls_aggregate",
            "per_control_loss_wins": 2,
            "rows_and_streams": "aggregate_nonregressing_against_both",
        },
        "read_diagnostic": {
            "arms": ("normal_anchored_learned", "forced_uniform"),
            "normal_loss": "strictly_lower_aggregate",
            "normal_loss_wins": 2,
            "rows_and_streams": "aggregate_nonregressing",
        },
        "context_or_joint_training": False,
        "early_stopping": False,
        "adaptive_rerun": False,
        "historical_score_control": False,
        "v13_checkpoint_reuse": False,
        "cell_or_stream_identity_input": False,
        "fixed_cell_roles": False,
        "hard_routing": False,
        "stream_sharding": False,
        "deterministic_top_k": False,
        "voting": False,
        "gradient_surgery": False,
        "stored_examples": False,
        "old_conflict_mixer": False,
        "deterministic_solver": False,
    }


def build_counterfactual_plasticity_pair(
    replicate: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[
    v13.CapacityMatchedClusterController,
    v13.CapacityMatchedClusterController,
    CounterfactualPlasticityRouter,
    CounterfactualPlasticityRouter,
]:
    """Build byte-identical controller/router starts for one paired replicate."""

    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or not 0 <= replicate < len(_REPLICATE_SEEDS)
    ):
        raise ValueError("V14 replicate is outside the fixed plan")
    shared_seed, cell_seed, composer_seed, router_seed = _REPLICATE_SEEDS[replicate]
    profile = v13.SOFTWARE_PIPELINE_PROFILES["smoke"]
    cpu_rng_state = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(shared_seed)
        uniform = v13.CapacityMatchedClusterController(
            profile,
            cell_seed=cell_seed,
            composer_seed=composer_seed,
        )
        torch.default_generator.manual_seed(shared_seed)
        learned = v13.CapacityMatchedClusterController(
            profile,
            cell_seed=cell_seed,
            composer_seed=composer_seed,
        )
        torch.default_generator.manual_seed(router_seed)
        uniform_router = CounterfactualPlasticityRouter()
        torch.default_generator.manual_seed(router_seed)
        learned_router = CounterfactualPlasticityRouter()
    finally:
        torch.set_rng_state(cpu_rng_state)
    if v13.software_pipeline_model_digest(uniform) != v13.software_pipeline_model_digest(
        learned
    ):
        raise RuntimeError("paired V14 controllers lost exact initialization")
    if counterfactual_plasticity_router_digest(
        uniform_router
    ) != counterfactual_plasticity_router_digest(learned_router):
        raise RuntimeError("paired V14 routers lost exact initialization")
    return (
        uniform.to(device),
        learned.to(device),
        uniform_router.to(device),
        learned_router.to(device),
    )


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    encoded_type = type(module).__name__.encode("ascii")
    digest.update(len(encoded_type).to_bytes(4, "big"))
    digest.update(encoded_type)
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(4, "big"))
        digest.update(encoded_dtype)
        digest.update(tensor.ndim.to_bytes(4, "big"))
        for size in tensor.shape:
            digest.update(int(size).to_bytes(8, "big"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def counterfactual_plasticity_router_digest(
    router: CounterfactualPlasticityRouter,
) -> str:
    if not isinstance(router, CounterfactualPlasticityRouter):
        raise TypeError("router digest requires CounterfactualPlasticityRouter")
    return _module_digest(router)


def counterfactual_plasticity_plan_digest() -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    digest.update(
        json.dumps(
            counterfactual_plasticity_fit_plan(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return "sha256:" + digest.hexdigest()


def counterfactual_plasticity_system_digest(
    uniform: v13.CapacityMatchedClusterController,
    learned: v13.CapacityMatchedClusterController,
    uniform_router: CounterfactualPlasticityRouter,
    learned_router: CounterfactualPlasticityRouter,
    replicate: int,
) -> str:
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or not 0 <= replicate < len(_REPLICATE_SEEDS)
    ):
        raise ValueError("V14 system replicate is invalid")
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    values = (
        _PROTOCOL_ID,
        str(replicate),
        v13.software_pipeline_model_digest(uniform),
        v13.software_pipeline_model_digest(learned),
        counterfactual_plasticity_router_digest(uniform_router),
        counterfactual_plasticity_router_digest(learned_router),
    )
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def save_counterfactual_plasticity_checkpoint(
    path: str | Path,
    systems: Sequence[
        tuple[
            v13.CapacityMatchedClusterController,
            v13.CapacityMatchedClusterController,
            CounterfactualPlasticityRouter,
            CounterfactualPlasticityRouter,
        ]
    ],
) -> None:
    if len(systems) != len(_REPLICATE_SEEDS):
        raise ValueError("V14 checkpoint requires all three fixed replicates")
    records = []
    for replicate, system in enumerate(systems):
        uniform, learned, uniform_router, learned_router = system
        records.append(
            {
                "replicate": replicate,
                "uniform_state": {
                    name: value.detach().cpu().clone()
                    for name, value in uniform.state_dict().items()
                },
                "learned_state": {
                    name: value.detach().cpu().clone()
                    for name, value in learned.state_dict().items()
                },
                "uniform_router_state": {
                    name: value.detach().cpu().clone()
                    for name, value in uniform_router.state_dict().items()
                },
                "learned_router_state": {
                    name: value.detach().cpu().clone()
                    for name, value in learned_router.state_dict().items()
                },
                "uniform_digest": v13.software_pipeline_model_digest(uniform),
                "learned_digest": v13.software_pipeline_model_digest(learned),
                "uniform_router_digest": counterfactual_plasticity_router_digest(
                    uniform_router
                ),
                "learned_router_digest": counterfactual_plasticity_router_digest(
                    learned_router
                ),
                "system_digest": counterfactual_plasticity_system_digest(
                    *system,
                    replicate,
                ),
            }
        )
    torch.save(
        {
            "version": _CHECKPOINT_VERSION,
            "protocol_id": _PROTOCOL_ID,
            "plan": counterfactual_plasticity_fit_plan(),
            "plan_digest": counterfactual_plasticity_plan_digest(),
            "replicates": tuple(records),
        },
        Path(path),
    )


def load_counterfactual_plasticity_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[
    tuple[
        v13.CapacityMatchedClusterController,
        v13.CapacityMatchedClusterController,
        CounterfactualPlasticityRouter,
        CounterfactualPlasticityRouter,
    ],
    ...,
]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    expected_plan = counterfactual_plasticity_fit_plan()
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _CHECKPOINT_VERSION
        or payload.get("protocol_id") != _PROTOCOL_ID
        or payload.get("plan") != expected_plan
        or payload.get("plan_digest") != counterfactual_plasticity_plan_digest()
    ):
        raise RuntimeError("V14 checkpoint identity or seed plan is invalid")
    records = payload.get("replicates")
    if not isinstance(records, (tuple, list)) or len(records) != len(_REPLICATE_SEEDS):
        raise RuntimeError("V14 checkpoint replicate set is invalid")
    restored = []
    for replicate, record in enumerate(records):
        if not isinstance(record, dict) or record.get("replicate") != replicate:
            raise RuntimeError("V14 checkpoint replicate identity changed")
        system = build_counterfactual_plasticity_pair(replicate, device=device)
        uniform, learned, uniform_router, learned_router = system
        uniform.load_state_dict(record["uniform_state"], strict=True)
        learned.load_state_dict(record["learned_state"], strict=True)
        uniform_router.load_state_dict(record["uniform_router_state"], strict=True)
        learned_router.load_state_dict(record["learned_router_state"], strict=True)
        observed = {
            "uniform_digest": v13.software_pipeline_model_digest(uniform),
            "learned_digest": v13.software_pipeline_model_digest(learned),
            "uniform_router_digest": counterfactual_plasticity_router_digest(
                uniform_router
            ),
            "learned_router_digest": counterfactual_plasticity_router_digest(
                learned_router
            ),
            "system_digest": counterfactual_plasticity_system_digest(
                *system,
                replicate,
            ),
        }
        if any(record.get(key) != value for key, value in observed.items()):
            raise RuntimeError("V14 checkpoint learned lineage changed")
        uniform.eval()
        learned.eval()
        uniform_router.eval()
        learned_router.eval()
        restored.append(system)
    return tuple(restored)


def _copy_cluster_controller(
    controller: v13.CapacityMatchedClusterController,
) -> v13.CapacityMatchedClusterController:
    clone = v13.CapacityMatchedClusterController(
        controller.profile,
        cell_seed=0,
        composer_seed=0,
    ).to(controller.procedure_start.device)
    clone.load_state_dict(controller.state_dict(), strict=True)
    return clone


def _apply_cells_only(
    controller: v13.CapacityMatchedClusterController,
    evidence: CellLocalEvidence,
    allocations: torch.Tensor,
) -> dict[str, object]:
    directions, raw_norms, clipped_norms = _routed_cell_directions(
        evidence,
        allocations.detach(),
        tuple(range(_STREAMS_PER_UPDATE)),
    )
    named = dict(controller.named_parameters())
    with torch.no_grad():
        for names, values in zip(
            evidence.cell_parameter_names,
            directions,
            strict=True,
        ):
            for name, direction in zip(names, values, strict=True):
                learning_rate = (
                    _ENCODER_LEARNING_RATE
                    if v13._relation_encoder_parameter_name(name)
                    else _HEAD_LEARNING_RATE
                )
                named[name].add_(direction, alpha=-learning_rate)
    return {
        "allocations": tuple(
            tuple(float(value) for value in row) for row in allocations.tolist()
        ),
        "cell_direction_norms": tuple(
            float(value.detach().item()) for value in raw_norms
        ),
        "cell_clipped_direction_norms": tuple(
            float(value.detach().item()) for value in clipped_norms
        ),
    }


def _single_panel_metrics(panel: Mapping[str, object]) -> dict[str, float | int]:
    if panel.get("rows") != 32 or panel.get("streams") != 8:
        raise RuntimeError("V14 diagnostic panel shape changed")
    return {
        "supported_rows": int(panel["relation_supported_rows"]),
        "qualifying_streams": int(panel["streams_with_three_supported_rows"]),
        "target_loss_mean": float(panel["target_loss_mean"]),
    }


def _one_step_allocation_diagnostic(
    controller: v13.CapacityMatchedClusterController,
    plasticity_router: CounterfactualPlasticityRouter,
    adaptation_streams: Sequence[SoftwarePipelineStream],
    probe_streams: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    if len(adaptation_streams) != _STREAMS_PER_UPDATE or len(
        probe_streams
    ) != _STREAMS_PER_UPDATE:
        raise ValueError("V14 one-step diagnostic requires two fresh eight-stream sets")
    before = v13.software_pipeline_model_digest(controller)
    evidence = collect_cell_local_evidence(controller, adaptation_streams)
    correct = plasticity_router(evidence.features)[0].detach()
    allocations = {
        "correct": correct,
        "uniform": torch.full_like(correct, 1.0 / _CELL_COUNT),
        "cell_permuted": torch.roll(correct, shifts=1, dims=0),
    }
    reports = {}
    for label, allocation in allocations.items():
        clone = _copy_cluster_controller(controller)
        update = _apply_cells_only(clone, evidence, allocation)
        panel = v13.evaluate_public_relation_credit_panel(clone, probe_streams)
        reports[label] = {
            "update": update,
            "probe": _single_panel_metrics(panel),
            "composer_stepped": False,
            "router_stepped": False,
        }
    if v13.software_pipeline_model_digest(controller) != before:
        raise RuntimeError("V14 one-step diagnostic mutated the trained lineage")
    return {
        "arms": reports,
        "same_adaptation_evidence": True,
        "same_probe_streams": True,
        "cell_permutation": "global_one_cell_cycle",
        "cell_marginal_multiset_preserved": sorted(
            float(value) for value in correct.mean(dim=1).tolist()
        )
        == sorted(
            float(value)
            for value in allocations["cell_permuted"].mean(dim=1).tolist()
        ),
        "cells_only": True,
    }


def _paired_panels(
    controller: v13.CapacityMatchedClusterController,
    panel_a: Sequence[SoftwarePipelineStream],
    panel_a_rerender: Sequence[SoftwarePipelineStream],
    panel_b: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    panels = {
        "panel_a": v13.evaluate_public_relation_credit_panel(controller, panel_a),
        "panel_a_rerender": v13.evaluate_public_relation_credit_panel(
            controller,
            panel_a_rerender,
        ),
        "panel_b": v13.evaluate_public_relation_credit_panel(controller, panel_b),
    }
    return {
        "panels": panels,
        "summary": v13._relation_panel_pair_summary(
            panels["panel_a"],
            panels["panel_b"],
        ),
        "surface_stability": v13._relation_surface_stability(
            panels["panel_a"],
            panels["panel_a_rerender"],
        ),
        "read_lesions": v13._evaluate_cluster_lesions(
            controller,
            panel_a,
            panel_b,
        ),
    }


def fit_counterfactual_plasticity_pilot(
    *,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """Run all three fresh paired V14 replicates without score-selected stopping."""

    plan = counterfactual_plasticity_fit_plan()
    commitments = plan["commitments"]
    assert isinstance(commitments, tuple)
    systems = []
    replicate_reports = []
    started = time.perf_counter()
    for replicate_spec in plan["replicates"]:
        replicate = int(replicate_spec["replicate"])
        system = build_counterfactual_plasticity_pair(replicate, device=device)
        systems.append(system)
        uniform, learned, uniform_router, learned_router = system
        initial = {
            "uniform": v13.software_pipeline_model_digest(uniform),
            "learned": v13.software_pipeline_model_digest(learned),
            "uniform_router": counterfactual_plasticity_router_digest(
                uniform_router
            ),
            "learned_router": counterfactual_plasticity_router_digest(
                learned_router
            ),
            "uniform_cells": tuple(
                _module_digest(cell) for cell in uniform.relation_cells
            ),
            "learned_cells": tuple(
                _module_digest(cell) for cell in learned.relation_cells
            ),
        }
        if initial["uniform"] != initial["learned"] or initial[
            "uniform_router"
        ] != initial["learned_router"]:
            raise RuntimeError("V14 paired start changed before fitting")
        batches = v13._relation_credit_stream_batches(
            commitments,
            replicate_spec["train_seed_batches"],
        )
        fits: dict[str, object] = {}
        for arm in replicate_spec["arm_order"]:
            if arm == "uniform_plasticity":
                fits[arm] = fit_counterfactual_plasticity_batches(
                    uniform,
                    uniform_router,
                    batches,
                    learned_plasticity=False,
                )
            elif arm == "learned_plasticity":
                fits[arm] = fit_counterfactual_plasticity_batches(
                    learned,
                    learned_router,
                    batches,
                    learned_plasticity=True,
                )
            else:
                raise RuntimeError("V14 arm identity changed")
        for fit in fits.values():
            if fit["optimizer_steps"] != _UPDATES_PER_ARM or fit["streams"] != 640 or fit[
                "rows"
            ] != 2_560:
                raise RuntimeError("V14 paired exposure changed")
        panel_a = v13._relation_credit_panel_streams(
            commitments,
            replicate_spec["panel_a_seed_pairs"],
        )
        panel_a_rerender = v13._relation_credit_panel_streams(
            commitments,
            replicate_spec["panel_a_rerender_seed_pairs"],
        )
        panel_b = v13._relation_credit_panel_streams(
            commitments,
            replicate_spec["panel_b_seed_pairs"],
        )
        adaptation_streams = v13._relation_credit_panel_streams(
            commitments,
            replicate_spec["adaptation_seed_pairs"],
        )
        probe_streams = v13._relation_credit_panel_streams(
            commitments,
            replicate_spec["probe_seed_pairs"],
        )
        uniform_evaluation = _paired_panels(
            uniform,
            panel_a,
            panel_a_rerender,
            panel_b,
        )
        learned_evaluation = _paired_panels(
            learned,
            panel_a,
            panel_a_rerender,
            panel_b,
        )
        one_step_diagnostic = _one_step_allocation_diagnostic(
            learned,
            learned_router,
            adaptation_streams,
            probe_streams,
        )
        terminal = {
            "uniform": v13.software_pipeline_model_digest(uniform),
            "learned": v13.software_pipeline_model_digest(learned),
            "uniform_router": counterfactual_plasticity_router_digest(
                uniform_router
            ),
            "learned_router": counterfactual_plasticity_router_digest(
                learned_router
            ),
            "uniform_cells": tuple(
                _module_digest(cell) for cell in uniform.relation_cells
            ),
            "learned_cells": tuple(
                _module_digest(cell) for cell in learned.relation_cells
            ),
            "system": counterfactual_plasticity_system_digest(*system, replicate),
        }
        replicate_reports.append(
            {
                "replicate": replicate,
                "arm_order": replicate_spec["arm_order"],
                "paired_stream_binding_exact": replicate_spec[
                    "uniform_stream_binding_digest"
                ]
                == replicate_spec["learned_stream_binding_digest"],
                "initial_digests": initial,
                "terminal_digests": terminal,
                "fits": fits,
                "evaluations": {
                    "uniform_plasticity": uniform_evaluation,
                    "learned_plasticity": learned_evaluation,
                },
                "one_step_allocation_diagnostic": one_step_diagnostic,
                "all_uniform_cells_changed": all(
                    before != after
                    for before, after in zip(
                        initial["uniform_cells"],
                        terminal["uniform_cells"],
                        strict=True,
                    )
                ),
                "all_learned_cells_changed": all(
                    before != after
                    for before, after in zip(
                        initial["learned_cells"],
                        terminal["learned_cells"],
                        strict=True,
                    )
                ),
                "uniform_router_changed": initial["uniform_router"]
                != terminal["uniform_router"],
                "learned_router_changed": initial["learned_router"]
                != terminal["learned_router"],
            }
        )
    if checkpoint_path is not None:
        save_counterfactual_plasticity_checkpoint(checkpoint_path, systems)
    uniform_rows = sum(
        int(report["evaluations"]["uniform_plasticity"]["summary"]["supported_rows"])
        for report in replicate_reports
    )
    learned_rows = sum(
        int(report["evaluations"]["learned_plasticity"]["summary"]["supported_rows"])
        for report in replicate_reports
    )
    uniform_streams = sum(
        int(
            report["evaluations"]["uniform_plasticity"]["summary"][
                "qualifying_streams"
            ]
        )
        for report in replicate_reports
    )
    learned_streams = sum(
        int(
            report["evaluations"]["learned_plasticity"]["summary"][
                "qualifying_streams"
            ]
        )
        for report in replicate_reports
    )
    uniform_loss = sum(
        float(
            report["evaluations"]["uniform_plasticity"]["summary"][
                "target_loss_mean"
            ]
        )
        for report in replicate_reports
    ) / len(replicate_reports)
    learned_loss = sum(
        float(
            report["evaluations"]["learned_plasticity"]["summary"][
                "target_loss_mean"
            ]
        )
        for report in replicate_reports
    ) / len(replicate_reports)
    row_nonregressing = sum(
        int(
            report["evaluations"]["learned_plasticity"]["summary"][
                "supported_rows"
            ]
        )
        >= int(
            report["evaluations"]["uniform_plasticity"]["summary"][
                "supported_rows"
            ]
        )
        for report in replicate_reports
    )
    loss_wins = sum(
        float(
            report["evaluations"]["learned_plasticity"]["summary"][
                "target_loss_mean"
            ]
        )
        < float(
            report["evaluations"]["uniform_plasticity"]["summary"][
                "target_loss_mean"
            ]
        )
        for report in replicate_reports
    )
    allocation_records = tuple(
        update["allocations"]
        for report in replicate_reports
        for update in report["fits"]["learned_plasticity"]["updates"]
    )
    learned_cell_usage = tuple(
        sum(
            float(allocation[cell_index][stream_index])
            for allocation in allocation_records
            for stream_index in range(_STREAMS_PER_UPDATE)
        )
        / (len(allocation_records) * _STREAMS_PER_UPDATE)
        for cell_index in range(_CELL_COUNT)
    )
    one_step_aggregate: dict[str, dict[str, float | int]] = {}
    for arm in ("correct", "uniform", "cell_permuted"):
        metrics = tuple(
            report["one_step_allocation_diagnostic"]["arms"][arm]["probe"]
            for report in replicate_reports
        )
        one_step_aggregate[arm] = {
            "supported_rows": sum(int(value["supported_rows"]) for value in metrics),
            "qualifying_streams": sum(
                int(value["qualifying_streams"]) for value in metrics
            ),
            "target_loss_mean": sum(
                float(value["target_loss_mean"]) for value in metrics
            )
            / len(metrics),
        }
    one_step_loss_wins = {
        control: sum(
            float(
                report["one_step_allocation_diagnostic"]["arms"]["correct"][
                    "probe"
                ]["target_loss_mean"]
            )
            < float(
                report["one_step_allocation_diagnostic"]["arms"][control][
                    "probe"
                ]["target_loss_mean"]
            )
            for report in replicate_reports
        )
        for control in ("uniform", "cell_permuted")
    }
    one_step_passed = all(
        float(one_step_aggregate["correct"]["target_loss_mean"])
        < float(one_step_aggregate[control]["target_loss_mean"])
        and one_step_loss_wins[control] >= 2
        and int(one_step_aggregate["correct"]["supported_rows"])
        >= int(one_step_aggregate[control]["supported_rows"])
        and int(one_step_aggregate["correct"]["qualifying_streams"])
        >= int(one_step_aggregate[control]["qualifying_streams"])
        for control in ("uniform", "cell_permuted")
    )
    normal_read = {
        "supported_rows": learned_rows,
        "qualifying_streams": learned_streams,
        "target_loss_mean": learned_loss,
    }
    uniform_read = {
        "supported_rows": sum(
            int(
                report["evaluations"]["learned_plasticity"]["read_lesions"][
                    "uniform"
                ]["supported_rows"]
            )
            for report in replicate_reports
        ),
        "qualifying_streams": sum(
            int(
                report["evaluations"]["learned_plasticity"]["read_lesions"][
                    "uniform"
                ]["qualifying_streams"]
            )
            for report in replicate_reports
        ),
        "target_loss_mean": sum(
            float(
                report["evaluations"]["learned_plasticity"]["read_lesions"][
                    "uniform"
                ]["target_loss_mean"]
            )
            for report in replicate_reports
        )
        / len(replicate_reports),
    }
    normal_read_loss_wins = sum(
        float(
            report["evaluations"]["learned_plasticity"]["summary"][
                "target_loss_mean"
            ]
        )
        < float(
            report["evaluations"]["learned_plasticity"]["read_lesions"][
                "uniform"
            ]["target_loss_mean"]
        )
        for report in replicate_reports
    )
    read_diagnostic_passed = (
        float(normal_read["target_loss_mean"])
        < float(uniform_read["target_loss_mean"])
        and int(normal_read["supported_rows"]) >= int(uniform_read["supported_rows"])
        and int(normal_read["qualifying_streams"])
        >= int(uniform_read["qualifying_streams"])
        and normal_read_loss_wins >= 2
    )
    checks = {
        "aggregate_supported_rows_at_least_uniform": learned_rows >= uniform_rows,
        "aggregate_qualifying_streams_at_least_uniform": (
            learned_streams >= uniform_streams
        ),
        "mean_target_loss_strictly_lower": learned_loss < uniform_loss,
        "target_loss_lower_in_two_replicates": loss_wins >= 2,
        "every_cell_aggregate_usage_at_least_ten_percent": min(
            learned_cell_usage
        )
        >= 0.10,
        "fresh_one_step_allocation_diagnostic": one_step_passed,
        "normal_read_beats_forced_uniform_read": read_diagnostic_passed,
        "learned_router_changed_every_replicate": all(
            report["learned_router_changed"] is True for report in replicate_reports
        ),
        "sham_router_changed_every_replicate": all(
            report["uniform_router_changed"] is True for report in replicate_reports
        ),
        "all_cells_changed_every_replicate": all(
            report["all_uniform_cells_changed"] is True
            and report["all_learned_cells_changed"] is True
            for report in replicate_reports
        ),
        "paired_exposure_exact": all(
            report["paired_stream_binding_exact"] is True
            for report in replicate_reports
        ),
    }
    supported = all(checks.values())
    harmful = (
        learned_rows <= uniform_rows
        and learned_streams <= uniform_streams
        and learned_loss >= uniform_loss
        and (
            learned_rows < uniform_rows
            or learned_streams < uniform_streams
            or learned_loss > uniform_loss
        )
    )
    return {
        "protocol_id": _PROTOCOL_ID,
        "status": (
            "PLASTICITY_ROUTER_SUPPORTED"
            if supported
            else "PLASTICITY_ROUTER_HARMFUL"
            if harmful
            else "PLASTICITY_ROUTER_INCONCLUSIVE"
        ),
        "plasticity_router_supported": supported,
        "plan": plan,
        "replicates": tuple(replicate_reports),
        "aggregate": {
            "uniform_supported_rows": uniform_rows,
            "learned_supported_rows": learned_rows,
            "uniform_qualifying_streams": uniform_streams,
            "learned_qualifying_streams": learned_streams,
            "uniform_target_loss_mean": uniform_loss,
            "learned_target_loss_mean": learned_loss,
            "supported_row_nonregressing_replicates": row_nonregressing,
            "target_loss_winning_replicates": loss_wins,
            "learned_cell_usage": learned_cell_usage,
            "one_step_allocation": one_step_aggregate,
            "one_step_loss_wins": one_step_loss_wins,
            "one_step_allocation_passed": one_step_passed,
            "normal_read": normal_read,
            "forced_uniform_read": uniform_read,
            "normal_read_loss_wins": normal_read_loss_wins,
            "read_diagnostic_passed": read_diagnostic_passed,
        },
        "support_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_written": checkpoint_path is not None,
        "context_or_joint_training_performed": False,
        "historical_v13_performance_used_as_control": False,
        "development_or_final_access": False,
        "wrong_evidence_training_streams": 0,
        "scalar_judge_calls": 0,
        "deterministic_solver_used": False,
    }


__all__ = [
    "CellLocalEvidence",
    "CounterfactualMetaResult",
    "CounterfactualPlasticityRouter",
    "build_counterfactual_plasticity_pair",
    "collect_cell_local_evidence",
    "counterfactual_plasticity_fit_plan",
    "counterfactual_plasticity_plan_digest",
    "counterfactual_plasticity_router_digest",
    "counterfactual_plasticity_system_digest",
    "counterfactual_router_meta_gradients",
    "fit_counterfactual_plasticity_batches",
    "fit_counterfactual_plasticity_pilot",
    "functional_heldout_loss",
    "load_counterfactual_plasticity_checkpoint",
    "save_counterfactual_plasticity_checkpoint",
]
