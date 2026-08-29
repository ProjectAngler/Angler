"""Anonymous cross-variation procedural plasticity for phase-6 relation cells.

V15 tests a narrow learned mechanism: whether public outcome consequences on
two fresh renderings of the same procedures can teach an identity-free router
to place updates into different persistent cells.  The foundation controller,
the four V13 relation cells, and the anchored read composer are unchanged.

The learning closure never receives procedure, task, package, motif, seed,
lane, stream, or cell identities.  Commitments and seeds exist only in the
deterministic synthetic curriculum builder.  Both real and counterfactual cell
updates call the same pure functional AdamW implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.evaluators import software_pipeline_reconstruction_suite as evaluator
from experiments.evaluators.software_pipeline_reconstruction_suite import (
    SoftwarePipelineStream,
    make_software_pipeline_stream,
    software_pipeline_mechanism_partition,
)
from experiments.runners import phase6_counterfactual_plasticity_router as v14
from experiments.runners import phase6_software_pipeline_reconstruction as v13


_PROTOCOL_ID = "phase6.public-anonymous-cross-variation-plasticity.paired.v15"
_CHECKPOINT_VERSION = "angler.phase6-cross-variation-plasticity.v1"
_DIGEST_DOMAIN = b"project-angler.cross-variation-plasticity.v1\x00"
_SCHEDULE_DOMAIN = b"angler.v15.commitment-schedule.v1\x00"
_ADAPTATION_SCHEDULE_DOMAIN = b"angler.v15.adaptation-schedule.v1\x00"
_SCHEDULE_SHA256 = (
    "8B18860D42DB4DF6979EBA3148CE94E817CF98D2A25014C58E15D34D46F8F7D1"
)
_ADAPTATION_SCHEDULE_SHA256 = (
    "6B449614DC824EF71022B622FF8348D444942D2F00701E6810BD119965F1D04D"
)

_CELL_COUNT = 4
_LANE_STREAMS = 4
_STREAMS_PER_UPDATE = 8
_ROWS_PER_STREAM = 4
_UPDATES_PER_ARM = 80
_ROUTER_LOCAL_FEATURES = 7
_ENCODER_LEARNING_RATE = 3.0e-4
_HEAD_LEARNING_RATE = 1.0e-3
_COMPOSER_LEARNING_RATE = 1.0e-3
_ROUTER_LEARNING_RATE = 1.0e-3
_CELL_DIRECTION_CLIP = 5.0
_USAGE_KL_WEIGHT = 0.01
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPSILON = 1.0e-8
_ADAM_WEIGHT_DECAY = 0.0
_NUMERICAL_TOLERANCE = 1.0e-6
_FROZEN_FILE_HASHES = {
    "experiments/evaluators/software_pipeline_reconstruction_suite.py": (
        "45D2282D5CC7FC504B817BA6ECB656B31DD568F85916B65E9145D1E1B0DFCE44"
    ),
    "tests/unit/experiments/test_software_pipeline_reconstruction_suite.py": (
        "CFB02F8D66CFFC9E326705969E6B3309025FB2BEE4A6D066A0D4780EB86586D3"
    ),
    "experiments/runners/phase6_software_pipeline_reconstruction.py": (
        "F1045756E77D60A7968265867035CEA55BFFE8BF6E1A73AB50C12A719EC8B529"
    ),
    "experiments/runners/phase6_counterfactual_plasticity_router.py": (
        "1AA64AAC3716F5C2C8333EE46852F839D19FC80AD39B1F5ED041E1738210C068"
    ),
}

_REPLICATE_SEEDS = (
    (2_026_083_401, 2_026_083_402, 2_026_083_403, 2_026_083_404),
    (2_026_083_411, 2_026_083_412, 2_026_083_413, 2_026_083_414),
    (2_026_083_421, 2_026_083_422, 2_026_083_423, 2_026_083_424),
)
_TRAIN_A_TOPOLOGY_BASE = 6_001_000_001
_TRAIN_A_SURFACE_BASE = 6_041_000_001
_TRAIN_B_TOPOLOGY_BASE = 6_081_000_001
_TRAIN_B_SURFACE_BASE = 6_121_000_001
_PANEL_A_TOPOLOGY_BASE = 6_201_000_001
_PANEL_A_SURFACE_BASE = 6_241_000_001
_PANEL_A_RERENDER_SURFACE_BASE = 6_281_000_001
_PANEL_B_TOPOLOGY_BASE = 6_321_000_001
_PANEL_B_SURFACE_BASE = 6_361_000_001
_ADAPT_A_TOPOLOGY_BASE = 6_401_000_001
_ADAPT_A_SURFACE_BASE = 6_441_000_001
_ADAPT_B_TOPOLOGY_BASE = 6_481_000_001
_ADAPT_B_SURFACE_BASE = 6_521_000_001
_PROBE_TOPOLOGY_BASE = 6_601_000_001
_PROBE_SURFACE_BASE = 6_641_000_001
_REPLICATE_SEED_STRIDE = 10_000_000

_SCHEDULE_TEXT = (
    "2567 0126 0457 1467 2467 1357 0167 1247 0236 0567 "
    "0134 1237 1347 0456 2374 2456 0173 0273 0136 3476 "
    "1035 1276 3456 3475 1075 2034 1056 1046 2065 2063 "
    "2014 2145 2173 4165 4012 4275 3605 5712 4705 3670 "
    "5612 3015 3604 5102 3152 3524 4602 5723 6713 4760 "
    "4613 7201 6701 6723 5762 6702 4612 6513 4531 7240 "
    "6753 7530 6324 7634 6541 7340 5324 7520 3201 5401 "
    "4321 5240 7651 5320 5340 7410 6321 7654 7541 6532"
)
_COMMITMENT_SCHEDULE = tuple(
    tuple(int(value) for value in token) for token in _SCHEDULE_TEXT.split()
)
_ADAPTATION_SCHEDULE = (
    (0, 1, 2, 3),
    (2, 3, 4, 5),
    (4, 5, 6, 7),
    (6, 7, 0, 1),
)


@dataclass(frozen=True, slots=True)
class AdamWSlot:
    """One exact functional AdamW parameter state."""

    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


CellAdamWState = tuple[tuple[AdamWSlot, ...], ...]


@dataclass(frozen=True, slots=True)
class CrossVariationEvidence:
    """V14 public local evidence enriched only by anonymous optimizer geometry."""

    base: v14.CellLocalEvidence
    features: torch.Tensor


@dataclass(frozen=True, slots=True)
class CrossVariationBatch:
    """One paired A/B experience with order metadata kept outside learning."""

    streams: tuple[SoftwarePipelineStream, ...]
    lane_a_indices: tuple[int, ...]
    lane_b_indices: tuple[int, ...]
    lane_a_slots: tuple[int, ...]
    lane_b_slots: tuple[int, ...]
    real_order: tuple[tuple[str, int], ...]
    procedure_indices: tuple[int, ...]
    topology_surface_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CrossVariationMetaResult:
    """Router-only gradients from symmetric cross-variation consequences."""

    gradients: tuple[torch.Tensor, ...]
    objective: float
    post_losses: tuple[float, float]
    pre_losses: tuple[float, float]
    aggregate_usage_kl: float
    lane_a_allocations: tuple[tuple[float, ...], ...]
    lane_b_allocations: tuple[tuple[float, ...], ...]
    fold_direction_norms: tuple[tuple[float, ...], tuple[float, ...]]
    fold_clipped_direction_norms: tuple[tuple[float, ...], tuple[float, ...]]
    parameter_gradient_norms: tuple[tuple[str, float], ...]


@dataclass(slots=True)
class CrossVariationArm:
    """One persistent V15 controller/router/optimizer lineage."""

    controller: v13.CapacityMatchedClusterController
    router: v14.CounterfactualPlasticityRouter
    cell_optimizer_state: CellAdamWState
    composer_optimizer: torch.optim.AdamW
    router_optimizer: torch.optim.AdamW


class SymmetricV15RelationComposer(v13.AnonymousAllActiveRelationComposer):
    """V13 composer with permutation-stable FP64 reduction accumulators."""

    def forward(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
        cell_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            query_codes.ndim != 3
            or stored_codes.ndim != 3
            or query_codes.shape[1:] != (self.cell_count, self.cell_width)
            or stored_codes.shape[1:] != (self.cell_count, self.cell_width)
            or cell_logits.shape
            != (query_codes.shape[0], stored_codes.shape[0], self.cell_count)
            or query_codes.device != stored_codes.device
            or query_codes.device != cell_logits.device
            or query_codes.dtype != stored_codes.dtype
            or query_codes.dtype != cell_logits.dtype
            or not query_codes.is_floating_point()
            or not bool(torch.isfinite(query_codes).all().item())
            or not bool(torch.isfinite(stored_codes).all().item())
            or not bool(torch.isfinite(cell_logits).all().item())
        ):
            raise ValueError("V15 relation composer inputs must be finite and aligned")
        query = query_codes[:, None, :, :]
        stored = stored_codes[None, :, :, :]
        features = torch.cat((query * stored, (query - stored).abs()), dim=-1)
        local = self.local_encoder(features)
        # Accumulate only the small anonymous set reductions in higher
        # precision.  Live parameters, codes, logits, and returned state stay
        # FP32; no cell is sorted or assigned a canonical position.
        shared = local.to(torch.float64).mean(dim=2, keepdim=True).to(local.dtype)
        shared = shared.expand_as(local)
        residual_logits = self.residual_scorer(
            torch.cat((local, shared), dim=-1)
        ).squeeze(-1)
        learned = torch.softmax(residual_logits.to(torch.float64), dim=-1).to(
            residual_logits.dtype
        )
        uniform = torch.full_like(learned, 1.0 / self.cell_count)
        weights = self.anchor_weight * uniform + (1.0 - self.anchor_weight) * learned
        fused = (
            weights.to(torch.float64) * cell_logits.to(torch.float64)
        ).sum(dim=-1).to(cell_logits.dtype)
        return fused, weights, residual_logits, features


class SymmetricV15ClusterController(v13.CapacityMatchedClusterController):
    """V13 cluster with capture-only support for stable public reductions."""

    def __init__(
        self,
        profile: v13.SoftwarePipelineRunProfile,
        *,
        cell_seed: int,
        composer_seed: int,
    ) -> None:
        super().__init__(
            profile,
            cell_seed=cell_seed,
            composer_seed=composer_seed,
        )
        original = self.relation_composer
        replacement = SymmetricV15RelationComposer(
            cell_count=original.cell_count,
            cell_width=original.cell_width,
            hidden_width=original.hidden_width,
            anchor_weight=original.anchor_weight,
        )
        replacement.load_state_dict(original.state_dict(), strict=True)
        self.relation_composer = replacement
        self._v15_relation_code_capture: list[torch.Tensor] | None = None

    def _factorized_relation_embeddings(
        self,
        components: Sequence[object],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_codes, relation_codes = super()._factorized_relation_embeddings(
            components,
            reference,
        )
        if self._v15_relation_code_capture is not None:
            self._v15_relation_code_capture.append(relation_codes)
        return context_codes, relation_codes

    def begin_v15_relation_capture(self) -> None:
        if self._v15_relation_code_capture is not None:
            raise RuntimeError("V15 relation capture is already active")
        self._v15_relation_code_capture = []

    def end_v15_relation_capture(self) -> tuple[torch.Tensor, ...]:
        captured = self._v15_relation_code_capture
        self._v15_relation_code_capture = None
        if captured is None or len(captured) != 4:
            raise RuntimeError("V15 relation capture lost public support alignment")
        return tuple(captured)


class SymmetricV15PlasticityRouter(v14.CounterfactualPlasticityRouter):
    """V14 set scorer with stable anonymous-set reduction accumulators."""

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
            raise ValueError("V15 router evidence must be finite and aligned")
        local = local_evidence.detach()
        cell_mean = local.to(torch.float64).mean(dim=1, keepdim=True).to(local.dtype)
        cell_mean = cell_mean.expand_as(local)
        stream_mean = local.to(torch.float64).mean(dim=0, keepdim=True).to(local.dtype)
        stream_mean = stream_mean.expand_as(local)
        global_mean = (
            local.to(torch.float64)
            .mean(dim=(0, 1), keepdim=True)
            .to(local.dtype)
            .expand_as(local)
        )
        enriched = torch.cat((local, cell_mean, stream_mean, global_mean), dim=-1)
        logits = self.scorer(self.local_encoder(enriched)).squeeze(-1)
        allocations = torch.softmax(logits.to(torch.float64), dim=0).to(logits.dtype)
        return allocations, logits, enriched


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("ascii")


def _validate_schedules() -> None:
    payload = _canonical_json(_COMMITMENT_SCHEDULE)
    adaptation_payload = _canonical_json(_ADAPTATION_SCHEDULE)
    if len(_COMMITMENT_SCHEDULE) != _UPDATES_PER_ARM or len(payload) != 801:
        raise RuntimeError("V15 commitment schedule shape or payload changed")
    if hashlib.sha256(_SCHEDULE_DOMAIN + payload).hexdigest().upper() != _SCHEDULE_SHA256:
        raise RuntimeError("V15 commitment schedule identity changed")
    if (
        len(adaptation_payload) != 41
        or hashlib.sha256(
            _ADAPTATION_SCHEDULE_DOMAIN + adaptation_payload
        ).hexdigest().upper()
        != _ADAPTATION_SCHEDULE_SHA256
    ):
        raise RuntimeError("V15 adaptation schedule identity changed")
    incidence = tuple(
        tuple(int(commitment in row) for row in _COMMITMENT_SCHEDULE)
        for commitment in range(8)
    )
    if any(sum(row) != 40 for row in incidence):
        raise RuntimeError("V15 commitment incidence is not balanced")
    if any(
        sum(row[slot] == commitment for row in _COMMITMENT_SCHEDULE) != 10
        for commitment in range(8)
        for slot in range(4)
    ):
        raise RuntimeError("V15 seed-slot incidence is not balanced")
    distances = {
        sum(left != right for left, right in zip(incidence[a], incidence[b], strict=True))
        for a in range(8)
        for b in range(a + 1, 8)
    }
    if distances != {44, 46}:
        raise RuntimeError("V15 commitment signatures lost separation")
    pair_counts = {
        (left, right): sum(left in row and right in row for row in _COMMITMENT_SCHEDULE)
        for left in range(8)
        for right in range(left + 1, 8)
    }
    if (
        tuple(sorted(pair_counts.values())).count(17) != 24
        or tuple(sorted(pair_counts.values())).count(18) != 4
        or {pair for pair, count in pair_counts.items() if count == 18}
        != {(0, 1), (2, 3), (4, 5), (6, 7)}
    ):
        raise RuntimeError("V15 pair co-occurrence balance changed")
    position_signatures = tuple(
        tuple(int(row[slot] == commitment) for row in _COMMITMENT_SCHEDULE)
        for commitment in range(8)
        for slot in range(4)
    )
    if len(set(position_signatures)) != 32 or min(
        sum(left != right for left, right in zip(a, b, strict=True))
        for index, a in enumerate(position_signatures)
        for b in position_signatures[index + 1 :]
    ) != 10:
        raise RuntimeError("V15 commitment-position signatures lost separation")


_validate_schedules()


def _hash_order(
    items: Sequence[tuple[str, int]],
    *,
    replicate: int,
    update: int,
    scope: str,
) -> tuple[tuple[str, int], ...]:
    """Return a deterministic presentation order; the key never enters a tensor."""

    domain = f"{_PROTOCOL_ID}|{replicate}|{update}|{scope}|".encode("ascii")
    return tuple(
        sorted(
            items,
            key=lambda item: hashlib.sha256(
                domain + f"{item[0]}|{item[1]}".encode("ascii")
            ).digest(),
        )
    )


def _panel_seed_pairs(topology_base: int, surface_base: int) -> tuple[tuple[int, int], ...]:
    return v13._relation_credit_panel_seed_pairs(topology_base, surface_base)


def _seed_binding_digest(records: Sequence[object]) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    digest.update(_canonical_json(records))
    return "sha256:" + digest.hexdigest()


def cross_variation_fit_plan() -> dict[str, object]:
    """Return the complete frozen V15 plan without constructing any stream."""

    commitments = software_pipeline_mechanism_partition("train")[:8]
    if len(commitments) != 8 or len(set(commitments)) != 8:
        raise RuntimeError("train partition cannot satisfy V15")
    replicates = []
    all_pairs: set[tuple[int, int]] = set()
    for replicate, seeds in enumerate(_REPLICATE_SEEDS):
        offset = replicate * _REPLICATE_SEED_STRIDE
        train_updates = []
        for update, schedule_row in enumerate(_COMMITMENT_SCHEDULE):
            a_pairs = tuple(
                (
                    _TRAIN_A_TOPOLOGY_BASE + offset + update * 100_000 + slot * 1_000,
                    _TRAIN_A_SURFACE_BASE + offset + update * 100_000 + slot * 1_000,
                )
                for slot in range(_LANE_STREAMS)
            )
            b_pairs = tuple(
                (
                    _TRAIN_B_TOPOLOGY_BASE + offset + update * 100_000 + slot * 1_000,
                    _TRAIN_B_SURFACE_BASE + offset + update * 100_000 + slot * 1_000,
                )
                for slot in range(_LANE_STREAMS)
            )
            train_updates.append(
                {
                    "procedure_indices": schedule_row,
                    "lane_a_seed_pairs": a_pairs,
                    "lane_b_seed_pairs": b_pairs,
                    "lane_a_order": _hash_order(
                        tuple(("A", slot) for slot in range(4)),
                        replicate=replicate,
                        update=update,
                        scope="lane-a",
                    ),
                    "lane_b_order": _hash_order(
                        tuple(("B", slot) for slot in range(4)),
                        replicate=replicate,
                        update=update,
                        scope="lane-b",
                    ),
                    "real_order": _hash_order(
                        tuple((lane, slot) for lane in ("A", "B") for slot in range(4)),
                        replicate=replicate,
                        update=update,
                        scope="real-eight",
                    ),
                }
            )
        panel_a = _panel_seed_pairs(
            _PANEL_A_TOPOLOGY_BASE + offset,
            _PANEL_A_SURFACE_BASE + offset,
        )
        panel_a_rerender = _panel_seed_pairs(
            _PANEL_A_TOPOLOGY_BASE + offset,
            _PANEL_A_RERENDER_SURFACE_BASE + offset,
        )
        panel_b = _panel_seed_pairs(
            _PANEL_B_TOPOLOGY_BASE + offset,
            _PANEL_B_SURFACE_BASE + offset,
        )
        probe = _panel_seed_pairs(
            _PROBE_TOPOLOGY_BASE + offset,
            _PROBE_SURFACE_BASE + offset,
        )
        adaptation_updates = []
        for update, schedule_row in enumerate(_ADAPTATION_SCHEDULE):
            adaptation_updates.append(
                {
                    "procedure_indices": schedule_row,
                    "lane_a_seed_pairs": tuple(
                        (
                            _ADAPT_A_TOPOLOGY_BASE
                            + offset
                            + update * 100_000
                            + slot * 1_000,
                            _ADAPT_A_SURFACE_BASE
                            + offset
                            + update * 100_000
                            + slot * 1_000,
                        )
                        for slot in range(4)
                    ),
                    "lane_b_seed_pairs": tuple(
                        (
                            _ADAPT_B_TOPOLOGY_BASE
                            + offset
                            + update * 100_000
                            + slot * 1_000,
                            _ADAPT_B_SURFACE_BASE
                            + offset
                            + update * 100_000
                            + slot * 1_000,
                        )
                        for slot in range(4)
                    ),
                    "lane_a_order": _hash_order(
                        tuple(("A", slot) for slot in range(4)),
                        replicate=replicate,
                        update=update,
                        scope="adapt-lane-a",
                    ),
                    "lane_b_order": _hash_order(
                        tuple(("B", slot) for slot in range(4)),
                        replicate=replicate,
                        update=update,
                        scope="adapt-lane-b",
                    ),
                    "real_order": _hash_order(
                        tuple((lane, slot) for lane in ("A", "B") for slot in range(4)),
                        replicate=replicate,
                        update=update,
                        scope="adapt-real-eight",
                    ),
                }
            )
        current_pairs = {
            pair
            for update in train_updates
            for key in ("lane_a_seed_pairs", "lane_b_seed_pairs")
            for pair in update[key]
        } | {
            pair
            for update in adaptation_updates
            for key in ("lane_a_seed_pairs", "lane_b_seed_pairs")
            for pair in update[key]
        } | set(panel_a) | set(panel_a_rerender) | set(panel_b) | set(probe)
        expected = 80 * 8 + 4 * 8 + 8 * 4
        if len(current_pairs) != expected or current_pairs & all_pairs:
            raise RuntimeError("V15 seed identities overlap")
        all_pairs.update(current_pairs)
        binding = _seed_binding_digest(train_updates)
        replicates.append(
            {
                "replicate": replicate,
                "shared_controller_seed": seeds[0],
                "cell_seed": seeds[1],
                "composer_seed": seeds[2],
                "router_seed": seeds[3],
                "arm_order": (
                    ("uniform_adamw_plasticity", "learned_episodic_plasticity")
                    if replicate != 1
                    else ("learned_episodic_plasticity", "uniform_adamw_plasticity")
                ),
                "train_updates": tuple(train_updates),
                "adaptation_updates": tuple(adaptation_updates),
                "panel_a_seed_pairs": panel_a,
                "panel_a_rerender_seed_pairs": panel_a_rerender,
                "panel_b_seed_pairs": panel_b,
                "probe_seed_pairs": probe,
                "uniform_stream_binding_digest": binding,
                "learned_stream_binding_digest": binding,
            }
        )
    prior_pairs = set()
    for prior in (
        v13.capacity_matched_relation_cluster_fit_plan(),
        v14.counterfactual_plasticity_fit_plan(),
    ):
        for specification in prior["replicates"]:
            for key, value in specification.items():
                if key.endswith("seed_pairs"):
                    prior_pairs.update(value)
                elif key == "train_seed_batches":
                    prior_pairs.update(pair for batch in value for pair in batch)
    if all_pairs & prior_pairs:
        raise RuntimeError("V15 identities overlap V13 or V14")
    return {
        "protocol_id": _PROTOCOL_ID,
        "partition": "train",
        "replicate_count": 3,
        "replicates": tuple(replicates),
        "commitments": commitments,
        "commitment_schedule": _COMMITMENT_SCHEDULE,
        "commitment_schedule_payload_bytes": 801,
        "commitment_schedule_sha256": _SCHEDULE_SHA256,
        "adaptation_schedule": _ADAPTATION_SCHEDULE,
        "adaptation_schedule_payload_bytes": 41,
        "adaptation_schedule_sha256": _ADAPTATION_SCHEDULE_SHA256,
        "updates_per_arm_per_replicate": 80,
        "streams_per_update": 8,
        "streams_per_lane": 4,
        "rows_per_stream": 4,
        "streams_per_arm_per_replicate": 640,
        "rows_per_arm_per_replicate": 2_560,
        "virtual_folds_per_update": 2,
        "arms": ("uniform_adamw_plasticity", "learned_episodic_plasticity"),
        "cell_optimizer": {
            "name": "pure_functional_adamw",
            "betas": (_ADAM_BETA1, _ADAM_BETA2),
            "epsilon": _ADAM_EPSILON,
            "weight_decay": 0.0,
            "encoder_learning_rate": _ENCODER_LEARNING_RATE,
            "head_learning_rate": _HEAD_LEARNING_RATE,
            "direction_global_clip": _CELL_DIRECTION_CLIP,
            "none_gradient": "skip_parameter_and_state",
            "explicit_zero_gradient": "advance_state",
        },
        "composer_optimizer": "separately_owned_adamw_1e-3",
        "router_optimizer": "separately_owned_adamw_1e-3",
        "router_local_features": (
            "detached_single_cell_public_loss",
            "detached_log_gradient_norm",
            "detached_within_cell_cosine_mean",
            "detached_within_cell_cosine_minimum",
            "detached_prediction_strength",
            "detached_raw_adam_moment_alignment",
            "detached_log_hypothetical_preconditioned_step_norm",
        ),
        "router_context": ("cell_mean", "stream_mean", "global_mean"),
        "router_hidden_width": 48,
        "router_final_scorer_zero_start": True,
        "allocation": "lane_local_softmax_across_four_cells",
        "meta_objective": "mean_direct_post_update_cross_lane_loss_plus_usage_kl",
        "meta_difference_scale": None,
        "aggregate_usage_kl_weight": _USAGE_KL_WEIGHT,
        "minimum_cell_allocation": None,
        "early_stopping": False,
        "adaptive_rerun": False,
        "historical_checkpoint_reuse": False,
        "cell_lane_pair_update_seed_task_package_motif_identity_input": False,
        "fixed_cell_roles": False,
        "hard_routing": False,
        "deterministic_top_k": False,
        "voting": False,
        "gradient_surgery": False,
        "stored_examples_or_replay": False,
        "deterministic_solver": False,
    }


def _make_cross_variation_batch(
    commitments: Sequence[str],
    specification: Mapping[str, object],
) -> CrossVariationBatch:
    procedure_indices = tuple(int(value) for value in specification["procedure_indices"])
    a_pairs = tuple(specification["lane_a_seed_pairs"])
    b_pairs = tuple(specification["lane_b_seed_pairs"])
    if len(procedure_indices) != 4 or len(set(procedure_indices)) != 4:
        raise ValueError("V15 batch requires four distinct procedures")
    by_key: dict[tuple[str, int], SoftwarePipelineStream] = {}
    for lane, pairs in (("A", a_pairs), ("B", b_pairs)):
        for slot, (procedure_index, pair) in enumerate(
            zip(procedure_indices, pairs, strict=True)
        ):
            by_key[(lane, slot)] = make_software_pipeline_stream(
                int(pair[0]),
                surface_seed=int(pair[1]),
                supports_per_motif=2,
                queries=1,
                maximum_steps=4,
                mechanism_commitment=commitments[procedure_index],
                mechanism_partition="train",
            )
    real_order = tuple(specification["real_order"])
    streams = tuple(by_key[key] for key in real_order)
    location = {key: index for index, key in enumerate(real_order)}
    lane_a_order = tuple(specification["lane_a_order"])
    lane_b_order = tuple(specification["lane_b_order"])
    return CrossVariationBatch(
        streams=streams,
        lane_a_indices=tuple(location[key] for key in lane_a_order),
        lane_b_indices=tuple(location[key] for key in lane_b_order),
        lane_a_slots=tuple(int(key[1]) for key in lane_a_order),
        lane_b_slots=tuple(int(key[1]) for key in lane_b_order),
        real_order=real_order,
        procedure_indices=procedure_indices,
        topology_surface_pairs=tuple(
            (int(pair[0]), int(pair[1])) for pair in a_pairs + b_pairs
        ),
    )


def build_training_batches(
    replicate: int,
    *,
    updates: int | None = None,
) -> tuple[CrossVariationBatch, ...]:
    plan = cross_variation_fit_plan()
    specification = plan["replicates"][replicate]
    records = specification["train_updates"]
    if updates is not None:
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise ValueError("updates must be a positive integer")
        records = records[:updates]
    return tuple(
        _make_cross_variation_batch(plan["commitments"], record) for record in records
    )


def _cell_parameter_groups(
    controller: v13.CapacityMatchedClusterController,
) -> tuple[tuple[tuple[str, nn.Parameter], ...], ...]:
    return v14._cell_parameter_groups(controller)


def _parameter_learning_rate(name: str) -> float:
    return (
        _ENCODER_LEARNING_RATE
        if v13._relation_encoder_parameter_name(name)
        else _HEAD_LEARNING_RATE
    )


def initial_cell_adamw_state(
    controller: v13.CapacityMatchedClusterController,
) -> CellAdamWState:
    """Create exact zero state aligned to the four owned cell parameter groups."""

    return tuple(
        tuple(
            AdamWSlot(
                step=0,
                exp_avg=torch.zeros_like(parameter, memory_format=torch.preserve_format),
                exp_avg_sq=torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                ),
            )
            for _, parameter in group
        )
        for group in _cell_parameter_groups(controller)
    )


def clone_cell_adamw_state(state: CellAdamWState) -> CellAdamWState:
    return tuple(
        tuple(
            AdamWSlot(
                step=slot.step,
                exp_avg=slot.exp_avg.detach().clone(),
                exp_avg_sq=slot.exp_avg_sq.detach().clone(),
            )
            for slot in cell
        )
        for cell in state
    )


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
    """Apply out-of-place AdamW with PyTorch's ordinary scalar-step semantics.

    A ``None`` gradient skips both parameter and state.  An explicit zero
    gradient advances the step and decays moments, exactly as an optimizer
    receives a materialized zero ``.grad`` tensor.
    """

    if not (
        len(parameters) == len(gradients) == len(state) == len(learning_rates)
    ):
        raise ValueError("functional AdamW inputs lost parameter alignment")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("functional AdamW beta is invalid")
    if epsilon < 0.0 or weight_decay < 0.0:
        raise ValueError("functional AdamW epsilon or weight decay is invalid")
    updated_parameters = []
    updated_state = []
    for parameter, gradient, slot, learning_rate in zip(
        parameters,
        gradients,
        state,
        learning_rates,
        strict=True,
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
        denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + epsilon
        decayed = parameter * (1.0 - learning_rate * weight_decay)
        updated_parameters.append(decayed - step_size * exp_avg / denominator)
        updated_state.append(
            AdamWSlot(
                step=next_step,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
            )
        )
    return tuple(updated_parameters), tuple(updated_state)


def _global_clip_directions(
    directions: Sequence[Sequence[torch.Tensor]],
) -> tuple[tuple[tuple[torch.Tensor, ...], ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    if len(directions) != _CELL_COUNT or any(not cell for cell in directions):
        raise ValueError("V15 direction must contain four nonempty cells")
    raw_norms = tuple(v14._gradient_norm(cell) for cell in directions)
    total = raw_norms[0].new_zeros(())
    for cell in directions:
        for value in cell:
            total = total + value.square().sum()
    global_norm = total.clamp_min(0.0).sqrt()
    scale = torch.minimum(
        global_norm.new_ones(()),
        global_norm.new_tensor(_CELL_DIRECTION_CLIP)
        / global_norm.clamp_min(torch.finfo(global_norm.dtype).tiny),
    )
    clipped = tuple(tuple(value * scale for value in cell) for cell in directions)
    return clipped, raw_norms, tuple(v14._gradient_norm(cell) for cell in clipped)


def _routed_directions(
    evidence: CrossVariationEvidence,
    allocations: torch.Tensor,
    stream_indices: Sequence[int],
) -> tuple[tuple[tuple[torch.Tensor, ...], ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Build one anonymous entropic routed direction and globally clip it."""

    indices = tuple(int(index) for index in stream_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or allocations.shape != (_CELL_COUNT, len(indices))
        or min(indices) < 0
        or max(indices) >= evidence.features.shape[1]
        or not torch.allclose(
            allocations.sum(dim=0),
            torch.ones_like(allocations.sum(dim=0)),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    ):
        raise ValueError("V15 routed allocation is invalid")
    index_tensor = torch.tensor(
        indices,
        device=evidence.base.ensemble_stream_losses.device,
        dtype=torch.long,
    )
    selected_losses = evidence.base.ensemble_stream_losses.index_select(0, index_tensor)
    base_weights = v13._relation_credit_stream_objective(
        selected_losses,
        stage="relation",
    )[3].detach()
    directions = []
    for cell_index in range(_CELL_COUNT):
        parameter_count = len(evidence.base.gradients[cell_index][indices[0]])
        directions.append(
            tuple(
                sum(
                    (
                        _CELL_COUNT
                        * base_weights[local_index]
                        * allocations[cell_index, local_index]
                        * evidence.base.gradients[cell_index][stream_index][parameter_index]
                        for local_index, stream_index in enumerate(indices)
                    ),
                    torch.zeros_like(
                        evidence.base.gradients[cell_index][indices[0]][parameter_index]
                    ),
                )
                for parameter_index in range(parameter_count)
            )
        )
    return _global_clip_directions(tuple(directions))


def _moment_alignment(
    gradients: Sequence[torch.Tensor],
    slots: Sequence[AdamWSlot],
) -> torch.Tensor:
    if len(gradients) != len(slots) or not gradients:
        raise ValueError("moment alignment lost parameter alignment")
    dot = gradients[0].new_zeros(())
    grad_square = gradients[0].new_zeros(())
    moment_square = gradients[0].new_zeros(())
    for gradient, slot in zip(gradients, slots, strict=True):
        dot = dot + (gradient * slot.exp_avg).sum()
        grad_square = grad_square + gradient.square().sum()
        moment_square = moment_square + slot.exp_avg.square().sum()
    denominator = grad_square.sqrt() * moment_square.sqrt()
    return torch.where(
        denominator > 0.0,
        dot / denominator.clamp_min(torch.finfo(dot.dtype).tiny),
        dot.new_zeros(()),
    )


def _stable_cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("V15 stable cosine requires aligned vectors")
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    dot = (left64 * right64).sum()
    left_norm = left64.square().sum().sqrt().clamp_min(1.0e-8)
    right_norm = right64.square().sum().sqrt().clamp_min(1.0e-8)
    return (dot / (left_norm * right_norm)).to(left.dtype)


def _stable_public_separation_losses(
    stream: SoftwarePipelineStream,
    captured_relation_codes: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Recompute only anonymous relation-vector reductions stably."""

    if len(captured_relation_codes) != len(stream.supports) or len(stream.supports) != 4:
        raise ValueError("V15 stable separation requires four captured supports")
    task_records = []
    for pair, relation_codes in zip(
        stream.supports,
        captured_relation_codes,
        strict=True,
    ):
        task = pair.learner
        components = v13._components_in_candidate_order(task)
        transitions = v13._public_transitions(task)
        observed = tuple(
            v13._action_index(task.grounded_candidates, transition.action)
            for transition in transitions
        )
        alternatives = tuple(
            v13._same_contract_alternative_index(components, index)
            for index in observed
        )
        task_records.append((observed, alternatives, relation_codes))
    losses = []
    for heldout_index, (observed, alternatives, query_relations) in enumerate(
        task_records
    ):
        discriminating = tuple(
            index for index, alternative in enumerate(alternatives) if alternative is not None
        )
        if len(discriminating) != 1:
            raise RuntimeError("V15 stable separation lost declared contrast")
        transition_index = discriminating[0]
        positive_index = observed[transition_index]
        negative_index = alternatives[transition_index]
        if negative_index is None:
            raise AssertionError("V15 stable separation contrast disappeared")
        terms = [
            F.relu(
                query_relations.new_tensor(v13._RELATION_FIT_MARGIN)
                - (
                    1.0
                    - _stable_cosine_similarity(
                        query_relations[positive_index],
                        query_relations[negative_index],
                    )
                )
            )
        ]
        for evidence_index, (
            evidence_observed,
            evidence_alternatives,
            evidence_relations,
        ) in enumerate(task_records):
            if evidence_index == heldout_index:
                continue
            for observed_index, alternative_index in zip(
                evidence_observed,
                evidence_alternatives,
                strict=True,
            ):
                if alternative_index is not None:
                    terms.append(
                        F.relu(
                            evidence_relations.new_tensor(v13._RELATION_FIT_MARGIN)
                            - (
                                1.0
                                - _stable_cosine_similarity(
                                    evidence_relations[observed_index],
                                    evidence_relations[alternative_index],
                                )
                            )
                        )
                    )
        losses.append(torch.stack(terms).mean())
    return tuple(losses)


def _public_rows_with_stable_separation(
    controller: v13.CapacityMatchedClusterController,
    stream: SoftwarePipelineStream,
) -> tuple[tuple[v13.PublicRelationCreditRow, ...], tuple[torch.Tensor, ...]]:
    if not isinstance(controller, SymmetricV15ClusterController):
        raise TypeError("V15 stable public loss requires its symmetric controller")
    controller.begin_v15_relation_capture()
    try:
        rows = v13.public_relation_credit_rows(controller, stream)
    except BaseException:
        controller._v15_relation_code_capture = None
        raise
    captured = controller.end_v15_relation_capture()
    return rows, _stable_public_separation_losses(stream, captured)


def _v15_stream_objective(
    controller: v13.CapacityMatchedClusterController,
    stream: SoftwarePipelineStream,
) -> torch.Tensor:
    rows, separation = _public_rows_with_stable_separation(controller, stream)
    row_losses = torch.stack(
        tuple(
            row.instance_loss
            + v13._RELATION_CREDIT_SEPARATION_WEIGHT * stable_separation
            for row, stable_separation in zip(rows, separation, strict=True)
        )
    )
    return v13._anonymous_entropic_row_objective(row_losses)[0]


def _v15_batch_objective(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(streams) != _STREAMS_PER_UPDATE:
        raise ValueError("V15 batch objective requires exactly eight streams")
    losses = torch.stack(tuple(_v15_stream_objective(controller, stream) for stream in streams))
    aggregate = v13._relation_credit_stream_objective(losses, stage="relation")
    return aggregate[0], losses, aggregate[3]


def _composer_gradients(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
) -> tuple[tuple[torch.Tensor, ...], float, tuple[float, ...], float]:
    parameters = tuple(controller.relation_composer.parameters())
    objective, stream_losses, _ = _v15_batch_objective(controller, streams)
    raw = torch.autograd.grad(
        objective,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    gradients = tuple(value.detach() for value in raw)
    norm = v14._gradient_norm(gradients)
    scale = torch.minimum(
        norm.new_ones(()),
        norm.new_tensor(v14._COMPOSER_GRADIENT_CLIP)
        / norm.clamp_min(torch.finfo(norm.dtype).tiny),
    )
    return (
        tuple(value * scale for value in gradients),
        float(objective.detach().item()),
        tuple(float(value) for value in stream_losses.detach().tolist()),
        float(norm.detach().item()),
    )


def _collect_homogeneous_cell_evidence(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
) -> v14.CellLocalEvidence:
    """Measure a cell through a homogeneous anonymous four-slot lift.

    V13's diagnostic ``single`` lesion still builds a heterogeneous four-cell
    relation vector and then reduces a one-hot weighted vector.  Its public
    separation term also reduces across that heterogeneous concatenation.
    Those mathematically symmetric FP32 reductions depend on physical cell
    order and AdamW can amplify their ULP differences.  For a genuinely local
    anonymous consequence, V15 lifts the target module into all four anonymous
    slots.  The ordinary V13 public objective is otherwise unchanged.  Every
    slot then has the same code/logit, so the result depends on the target
    module but not its original position or the identities of other cells.
    """

    if not isinstance(controller, v13.CapacityMatchedClusterController):
        raise TypeError("V15 evidence requires a V13 cluster controller")
    if not streams:
        raise ValueError("V15 evidence requires public streams")
    if controller._relation_diagnostic_lesion is not None:
        raise RuntimeError("V15 evidence requires an unlesioned controller")
    before = v13.software_pipeline_model_digest(controller)
    original_cells = controller.relation_cells
    groups = _cell_parameter_groups(controller)
    all_losses: list[tuple[torch.Tensor, ...]] = []
    all_strengths: list[tuple[torch.Tensor, ...]] = []
    all_gradients: list[tuple[tuple[torch.Tensor, ...], ...]] = []
    try:
        for cell_index, group in enumerate(groups):
            target = original_cells[cell_index]
            controller.relation_cells = nn.ModuleList(
                tuple(target for _ in range(_CELL_COUNT))
            )
            parameters = tuple(parameter for _, parameter in group)
            cell_losses = []
            cell_strengths = []
            cell_gradients = []
            for stream in streams:
                rows, separation = _public_rows_with_stable_separation(
                    controller,
                    stream,
                )
                row_losses = torch.stack(
                    tuple(
                        row.instance_loss
                        + v13._RELATION_CREDIT_SEPARATION_WEIGHT * stable_separation
                        for row, stable_separation in zip(rows, separation, strict=True)
                    )
                )
                loss = v13._anonymous_entropic_row_objective(row_losses)[0]
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
        controller.relation_cells = original_cells
    if v13.software_pipeline_model_digest(controller) != before:
        raise RuntimeError("V15 homogeneous evidence mutated controller state")
    losses = torch.stack(tuple(torch.stack(values) for values in all_losses))
    strengths = torch.stack(tuple(torch.stack(values) for values in all_strengths))
    norms = torch.stack(
        tuple(
            torch.stack(tuple(v14._gradient_norm(direction) for direction in cell))
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
                        v14._gradient_cosine(direction, other)
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
        raise RuntimeError("V15 homogeneous evidence is non-finite")
    with torch.no_grad():
        ensemble_stream_losses = torch.stack(
            tuple(_v15_stream_objective(controller, stream) for stream in streams)
        )
        entropic_base_weights = v13._relation_credit_stream_objective(
            ensemble_stream_losses,
            stage="relation",
        )[3]
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V15 homogeneous evidence populated controller gradients")
    return v14.CellLocalEvidence(
        features=features,
        losses=losses.detach(),
        gradient_norms=norms.detach(),
        prediction_strengths=strengths.detach(),
        ensemble_stream_losses=ensemble_stream_losses.detach(),
        entropic_base_weights=entropic_base_weights.detach(),
        cell_parameter_names=tuple(tuple(name for name, _ in group) for group in groups),
        gradients=tuple(all_gradients),
    )


def collect_cross_variation_evidence(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
    cell_optimizer_state: CellAdamWState,
) -> CrossVariationEvidence:
    """Collect seven detached, identity-free local features without mutation."""

    if len(cell_optimizer_state) != _CELL_COUNT:
        raise ValueError("V15 evidence requires four cell optimizer states")
    before = v13.software_pipeline_model_digest(controller)
    base = _collect_homogeneous_cell_evidence(controller, streams)
    groups = _cell_parameter_groups(controller)
    if any(len(group) != len(state) for group, state in zip(groups, cell_optimizer_state, strict=True)):
        raise ValueError("V15 optimizer state lost cell parameter alignment")
    alignments = torch.zeros_like(base.losses)
    step_norms = torch.zeros_like(base.losses)
    for cell_index, (group, slots) in enumerate(
        zip(groups, cell_optimizer_state, strict=True)
    ):
        parameters = tuple(parameter.detach() for _, parameter in group)
        learning_rates = tuple(_parameter_learning_rate(name) for name, _ in group)
        for stream_index, gradients in enumerate(base.gradients[cell_index]):
            alignments[cell_index, stream_index] = _moment_alignment(gradients, slots)
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
        raise RuntimeError("V15 seven-feature evidence shape changed")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("V15 local public evidence is non-finite")
    if before != v13.software_pipeline_model_digest(controller):
        raise RuntimeError("V15 evidence mutated controller state")
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V15 evidence populated controller gradient fields")
    return CrossVariationEvidence(base=base, features=features)


def _features_for_indices(
    evidence: CrossVariationEvidence,
    indices: Sequence[int],
) -> torch.Tensor:
    selected = tuple(int(index) for index in indices)
    first_five = v14._features_for_stream_indices(evidence.base, selected)
    index_tensor = torch.tensor(
        selected,
        device=evidence.features.device,
        dtype=torch.long,
    )
    optimizer_geometry = evidence.features[..., 5:].index_select(1, index_tensor)
    return torch.cat((first_five, optimizer_geometry), dim=-1).detach()


class _FunctionalEnsembleObjective(nn.Module):
    def __init__(self, controller: v13.CapacityMatchedClusterController) -> None:
        super().__init__()
        self.controller = controller

    def forward(self, stream: SoftwarePipelineStream) -> torch.Tensor:
        return _v15_stream_objective(self.controller, stream)


def _functional_target_loss(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
    cell_parameter_names: Sequence[Sequence[str]],
    updated_parameters: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Evaluate a four-stream target lane under virtual cell parameters."""

    if len(streams) != _LANE_STREAMS:
        raise ValueError("V15 functional target requires exactly four streams")
    wrapper = _FunctionalEnsembleObjective(controller)
    state = {name: parameter.detach() for name, parameter in wrapper.named_parameters()}
    buffers = {name: buffer.detach() for name, buffer in wrapper.named_buffers()}
    for names, values in zip(cell_parameter_names, updated_parameters, strict=True):
        if len(names) != len(values):
            raise ValueError("V15 virtual update lost parameter alignment")
        for name, value in zip(names, values, strict=True):
            expected = state.get("controller." + name)
            if expected is None or expected.shape != value.shape:
                raise ValueError("V15 virtual update changed controller shape")
            state["controller." + name] = value
    losses = torch.stack(
        tuple(
            torch.func.functional_call(
                wrapper,
                (state, buffers),
                (stream,),
                strict=True,
            )
            for stream in streams
        )
    )
    return v13._relation_credit_stream_objective(losses, stage="relation")[0]


def _virtual_adamw_parameters(
    controller: v13.CapacityMatchedClusterController,
    directions: Sequence[Sequence[torch.Tensor]],
    cell_optimizer_state: CellAdamWState,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    groups = _cell_parameter_groups(controller)
    updated = []
    for group, gradients, slots in zip(
        groups,
        directions,
        cell_optimizer_state,
        strict=True,
    ):
        values, _ = functional_adamw_step(
            tuple(parameter.detach() for _, parameter in group),
            gradients,
            slots,
            tuple(_parameter_learning_rate(name) for name, _ in group),
        )
        updated.append(values)
    return tuple(updated)


def _lane_allocations(
    router: v14.CounterfactualPlasticityRouter,
    evidence: CrossVariationEvidence,
    indices: Sequence[int],
) -> torch.Tensor:
    return router(_features_for_indices(evidence, indices))[0]


def cross_variation_meta_gradients(
    controller: v13.CapacityMatchedClusterController,
    router: v14.CounterfactualPlasticityRouter,
    batch: CrossVariationBatch,
    cell_optimizer_state: CellAdamWState,
    evidence: CrossVariationEvidence | None = None,
) -> CrossVariationMetaResult:
    """Differentiate symmetric A-to-B and B-to-A direct post-update losses."""

    if evidence is None:
        evidence = collect_cross_variation_evidence(
            controller,
            batch.streams,
            cell_optimizer_state,
        )
    before = v13.software_pipeline_model_digest(controller)
    router_before = cross_variation_router_digest(router)
    a_route = _lane_allocations(router, evidence, batch.lane_a_indices)
    b_route = _lane_allocations(router, evidence, batch.lane_b_indices)
    a_direction, a_raw, a_clipped = _routed_directions(
        evidence,
        a_route,
        batch.lane_a_indices,
    )
    b_direction, b_raw, b_clipped = _routed_directions(
        evidence,
        b_route,
        batch.lane_b_indices,
    )
    names = evidence.base.cell_parameter_names
    a_virtual = _virtual_adamw_parameters(
        controller,
        a_direction,
        cell_optimizer_state,
    )
    b_virtual = _virtual_adamw_parameters(
        controller,
        b_direction,
        cell_optimizer_state,
    )
    lane_a_streams = tuple(batch.streams[index] for index in batch.lane_a_indices)
    lane_b_streams = tuple(batch.streams[index] for index in batch.lane_b_indices)
    post_a_to_b = _functional_target_loss(
        controller,
        lane_b_streams,
        names,
        a_virtual,
    )
    post_b_to_a = _functional_target_loss(
        controller,
        lane_a_streams,
        names,
        b_virtual,
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
    router_parameters = tuple(router.parameters())
    raw_gradients = torch.autograd.grad(
        objective,
        router_parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    gradients = tuple(value.detach() for value in raw_gradients)
    with torch.no_grad():
        pre_a = v13._relation_credit_stream_objective(
            torch.stack(
                tuple(_v15_stream_objective(controller, stream) for stream in lane_a_streams)
            ),
            stage="relation",
        )[0]
        pre_b = v13._relation_credit_stream_objective(
            torch.stack(
                tuple(_v15_stream_objective(controller, stream) for stream in lane_b_streams)
            ),
            stage="relation",
        )[0]
    if before != v13.software_pipeline_model_digest(controller):
        raise RuntimeError("V15 virtual folds mutated controller state")
    if router_before != cross_variation_router_digest(router):
        raise RuntimeError("V15 virtual folds mutated router state")
    if any(parameter.grad is not None for parameter in controller.parameters()):
        raise RuntimeError("V15 meta-loss populated controller gradient fields")
    if any(parameter.grad is not None for parameter in router.parameters()):
        raise RuntimeError("V15 meta-loss populated router gradient fields")
    return CrossVariationMetaResult(
        gradients=gradients,
        objective=float(objective.detach().item()),
        post_losses=(
            float(post_a_to_b.detach().item()),
            float(post_b_to_a.detach().item()),
        ),
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
            (name, float(value.norm().item()))
            for (name, _), value in zip(
                router.named_parameters(),
                gradients,
                strict=True,
            )
        ),
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


def _update_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\x00")
        digest.update(str(tensor.dtype).encode("ascii") + b"\x00")
        digest.update(_canonical_json(tuple(int(size) for size in tensor.shape)))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\x00")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence\x00" + len(value).to_bytes(8, "big"))
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        encoded = repr(value).encode("utf-8")
        digest.update(b"scalar\x00" + len(encoded).to_bytes(8, "big") + encoded)
    else:
        raise TypeError(f"unsupported digest value: {type(value).__name__}")


def _state_payload(state: CellAdamWState) -> tuple[tuple[dict[str, object], ...], ...]:
    return tuple(
        tuple(
            {
                "step": slot.step,
                "exp_avg": slot.exp_avg.detach().cpu().clone(),
                "exp_avg_sq": slot.exp_avg_sq.detach().cpu().clone(),
            }
            for slot in cell
        )
        for cell in state
    )


def _state_from_payload(
    payload: object,
    controller: v13.CapacityMatchedClusterController,
) -> CellAdamWState:
    groups = _cell_parameter_groups(controller)
    if not isinstance(payload, (tuple, list)) or len(payload) != _CELL_COUNT:
        raise RuntimeError("V15 checkpoint cell optimizer shape changed")
    restored = []
    for records, group in zip(payload, groups, strict=True):
        if not isinstance(records, (tuple, list)) or len(records) != len(group):
            raise RuntimeError("V15 checkpoint cell parameter state changed")
        cell = []
        for record, (_, parameter) in zip(records, group, strict=True):
            if not isinstance(record, dict):
                raise RuntimeError("V15 checkpoint cell state is invalid")
            exp_avg = record.get("exp_avg")
            exp_avg_sq = record.get("exp_avg_sq")
            step = record.get("step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or not isinstance(exp_avg, torch.Tensor)
                or not isinstance(exp_avg_sq, torch.Tensor)
                or exp_avg.shape != parameter.shape
                or exp_avg_sq.shape != parameter.shape
                or exp_avg.dtype != parameter.dtype
                or exp_avg_sq.dtype != parameter.dtype
            ):
                raise RuntimeError("V15 checkpoint cell state identity changed")
            cell.append(
                AdamWSlot(
                    step=step,
                    exp_avg=exp_avg.to(parameter.device),
                    exp_avg_sq=exp_avg_sq.to(parameter.device),
                )
            )
        restored.append(tuple(cell))
    return tuple(restored)


def cross_variation_router_digest(router: v14.CounterfactualPlasticityRouter) -> str:
    if (
        not isinstance(router, v14.CounterfactualPlasticityRouter)
        or router.local_features != _ROUTER_LOCAL_FEATURES
        or router.hidden_width != 48
    ):
        raise TypeError("V15 router architecture changed")
    return _module_digest(router)


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


def cross_variation_arm_digest(arm: CrossVariationArm) -> str:
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    _update_digest(
        digest,
        (
            v13.software_pipeline_model_digest(arm.controller),
            cross_variation_router_digest(arm.router),
            _state_payload(arm.cell_optimizer_state),
            arm.composer_optimizer.state_dict(),
            arm.router_optimizer.state_dict(),
        ),
    )
    return "sha256:" + digest.hexdigest()


def _build_arm(
    *,
    shared_seed: int,
    cell_seed: int,
    composer_seed: int,
    router_seed: int,
    device: torch.device | str,
) -> CrossVariationArm:
    cpu_rng_state = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(shared_seed)
        controller = SymmetricV15ClusterController(
            v13.SOFTWARE_PIPELINE_PROFILES["smoke"],
            cell_seed=cell_seed,
            composer_seed=composer_seed,
        ).to(device)
        torch.default_generator.manual_seed(router_seed)
        router = SymmetricV15PlasticityRouter(
            local_features=_ROUTER_LOCAL_FEATURES,
            hidden_width=48,
        ).to(device)
    finally:
        torch.set_rng_state(cpu_rng_state)
    return CrossVariationArm(
        controller=controller,
        router=router,
        cell_optimizer_state=initial_cell_adamw_state(controller),
        composer_optimizer=torch.optim.AdamW(
            controller.relation_composer.parameters(),
            lr=_COMPOSER_LEARNING_RATE,
            betas=(_ADAM_BETA1, _ADAM_BETA2),
            eps=_ADAM_EPSILON,
            weight_decay=0.0,
        ),
        router_optimizer=torch.optim.AdamW(
            router.parameters(),
            lr=_ROUTER_LEARNING_RATE,
            betas=(_ADAM_BETA1, _ADAM_BETA2),
            eps=_ADAM_EPSILON,
            weight_decay=0.0,
        ),
    )


def build_cross_variation_pair(
    replicate: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CrossVariationArm, CrossVariationArm]:
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or not 0 <= replicate < len(_REPLICATE_SEEDS)
    ):
        raise ValueError("V15 replicate is outside the frozen plan")
    seeds = _REPLICATE_SEEDS[replicate]
    uniform = _build_arm(
        shared_seed=seeds[0],
        cell_seed=seeds[1],
        composer_seed=seeds[2],
        router_seed=seeds[3],
        device=device,
    )
    learned = _build_arm(
        shared_seed=seeds[0],
        cell_seed=seeds[1],
        composer_seed=seeds[2],
        router_seed=seeds[3],
        device=device,
    )
    if (
        v13.software_pipeline_model_digest(uniform.controller)
        != v13.software_pipeline_model_digest(learned.controller)
        or cross_variation_router_digest(uniform.router)
        != cross_variation_router_digest(learned.router)
        or cross_variation_arm_digest(uniform) != cross_variation_arm_digest(learned)
    ):
        raise RuntimeError("V15 paired systems lost byte-identical starts")
    return uniform, learned


def _canonical_route(
    route: torch.Tensor,
    slots: Sequence[int],
) -> torch.Tensor:
    if route.shape != (_CELL_COUNT, _LANE_STREAMS) or set(slots) != set(range(4)):
        raise ValueError("V15 lane route cannot be aligned")
    aligned = torch.empty_like(route)
    for column, slot in enumerate(slots):
        aligned[:, int(slot)] = route[:, column]
    return aligned


def _combined_allocations(
    evidence: CrossVariationEvidence,
    batch: CrossVariationBatch,
    router: v14.CounterfactualPlasticityRouter,
    *,
    learned_plasticity: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_route = _lane_allocations(router, evidence, batch.lane_a_indices)
    b_route = _lane_allocations(router, evidence, batch.lane_b_indices)
    if not learned_plasticity:
        a_applied = torch.full_like(a_route, 1.0 / _CELL_COUNT)
        b_applied = torch.full_like(b_route, 1.0 / _CELL_COUNT)
    else:
        a_applied = a_route.detach()
        b_applied = b_route.detach()
    combined = evidence.features.new_empty((_CELL_COUNT, _STREAMS_PER_UPDATE))
    combined[:, torch.tensor(batch.lane_a_indices, device=combined.device)] = a_applied
    combined[:, torch.tensor(batch.lane_b_indices, device=combined.device)] = b_applied
    return combined, a_route.detach(), b_route.detach()


def _apply_cell_adamw_update(
    arm: CrossVariationArm,
    directions: Sequence[Sequence[torch.Tensor]],
) -> None:
    groups = _cell_parameter_groups(arm.controller)
    next_state = []
    with torch.no_grad():
        for group, gradients, slots in zip(
            groups,
            directions,
            arm.cell_optimizer_state,
            strict=True,
        ):
            updated, updated_slots = functional_adamw_step(
                tuple(parameter.detach() for _, parameter in group),
                tuple(value.detach() for value in gradients),
                slots,
                tuple(_parameter_learning_rate(name) for name, _ in group),
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
    parameters: Sequence[nn.Parameter],
    gradients: Sequence[torch.Tensor],
) -> None:
    if len(parameters) != len(gradients):
        raise RuntimeError("V15 optimizer gradient ownership changed")
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _route_statistics(route: torch.Tensor) -> tuple[float, float]:
    uniform = torch.full_like(route, 1.0 / _CELL_COUNT)
    tv = 0.5 * (route - uniform).abs().sum(dim=0)
    return float(tv.mean().item()), _median(tuple(float(value) for value in tv.tolist()))


def _route_pair_distances(
    a_route: torch.Tensor,
    a_slots: Sequence[int],
    b_route: torch.Tensor,
    b_slots: Sequence[int],
) -> tuple[float, float]:
    a = _canonical_route(a_route, a_slots)
    b = _canonical_route(b_route, b_slots)
    matched = torch.stack(
        tuple(0.5 * (a[:, index] - b[:, index]).abs().sum() for index in range(4))
    ).mean()
    unmatched = torch.stack(
        tuple(
            0.5 * (a[:, left] - b[:, right]).abs().sum()
            for left in range(4)
            for right in range(4)
            if left != right
        )
    ).mean()
    return float(matched.item()), float(unmatched.item())


def fit_cross_variation_batches(
    arm: CrossVariationArm,
    batches: Sequence[CrossVariationBatch],
    *,
    learned_plasticity: bool,
) -> dict[str, object]:
    """Fit one arm with exact cell AdamW and equal-compute sham routing."""

    if type(learned_plasticity) is not bool:
        raise TypeError("learned_plasticity must be bool")
    if not batches or any(len(batch.streams) != 8 for batch in batches):
        raise ValueError("V15 fit requires nonempty eight-stream batches")
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
            arm.controller,
            batch.streams,
            arm.cell_optimizer_state,
        )
        allocations, raw_a_route, raw_b_route = _combined_allocations(
            evidence,
            batch,
            arm.router,
            learned_plasticity=learned_plasticity,
        )
        directions, raw_norms, clipped_norms = _routed_directions(
            evidence,
            allocations,
            tuple(range(_STREAMS_PER_UPDATE)),
        )
        meta = cross_variation_meta_gradients(
            arm.controller,
            arm.router,
            batch,
            arm.cell_optimizer_state,
            evidence,
        )
        gradient_by_name = dict(meta.parameter_gradient_norms)
        scorer_gradient = gradient_by_name.get("scorer.weight", 0.0)
        upstream_gradient = max(
            (value for name, value in meta.parameter_gradient_norms if name != "scorer.weight"),
            default=0.0,
        )
        if update_index == 0 and (
            not math.isfinite(scorer_gradient)
            or scorer_gradient <= 0.0
            or upstream_gradient != 0.0
        ):
            raise RuntimeError("V15 zero-start structural warm-up failed")
        if update_index == 1:
            a_tv = _route_statistics(raw_a_route)[0]
            b_tv = _route_statistics(raw_b_route)[0]
            if max(a_tv, b_tv) <= 1.0e-6 or upstream_gradient <= 0.0:
                raise RuntimeError("V15 post-warm-up routing path is unreachable")
        composer_gradients, objective, stream_losses, composer_gradient_norm = (
            _composer_gradients(arm.controller, batch.streams)
        )
        _apply_cell_adamw_update(arm, directions)
        _apply_owned_optimizer_gradients(
            arm.composer_optimizer,
            tuple(arm.controller.relation_composer.parameters()),
            composer_gradients,
        )
        _apply_owned_optimizer_gradients(
            arm.router_optimizer,
            tuple(arm.router.parameters()),
            meta.gradients,
        )
        if any(parameter.grad is not None for parameter in arm.controller.parameters()):
            raise RuntimeError("V15 fit leaked controller gradient fields")
        a_tv_mean, a_tv_median = _route_statistics(raw_a_route)
        b_tv_mean, b_tv_median = _route_statistics(raw_b_route)
        paired_distance, unpaired_distance = _route_pair_distances(
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
                "composer_gradient_norm": composer_gradient_norm,
                "meta": {
                    "objective": meta.objective,
                    "direct_post_update_losses": meta.post_losses,
                    "pre_update_losses_record_only": meta.pre_losses,
                    "aggregate_usage_kl": meta.aggregate_usage_kl,
                    "fold_direction_norms": meta.fold_direction_norms,
                    "fold_clipped_direction_norms": meta.fold_clipped_direction_norms,
                    "parameter_gradient_norms": meta.parameter_gradient_norms,
                },
            }
        )
    named = dict(arm.controller.named_parameters())
    for name, before in frozen.items():
        if not torch.equal(before, named[name].detach()):
            raise RuntimeError(f"V15 fit changed frozen parameter: {name}")
    return {
        "arm": (
            "learned_episodic_plasticity"
            if learned_plasticity
            else "uniform_adamw_plasticity"
        ),
        "optimizer_steps": len(batches),
        "streams": len(batches) * 8,
        "rows": len(batches) * 32,
        "virtual_folds_per_update": 2,
        "updates": tuple(reports),
        "first_allocation_exact_uniform": all(
            value == 0.25 for row in reports[0]["applied_allocations"] for value in row
        ),
        "cell_update": "pure_functional_adamw",
        "virtual_cell_update": "pure_functional_adamw",
        "actual_and_virtual_cell_update_identical": True,
        "composer_update": "separately_owned_adamw",
        "router_update": "separately_owned_adamw",
        "router_scores_applied": learned_plasticity,
        "sham_router_compute_matched": not learned_plasticity,
        "router_affects_current_batch": False,
        "router_affects_next_batch": learned_plasticity,
        "router_initial_digest": initial_router,
        "router_terminal_digest": cross_variation_router_digest(arm.router),
        "router_changed": initial_router != cross_variation_router_digest(arm.router),
        "frozen_controller_parameters_unchanged": True,
        "controller_grad_fields_clear": True,
    }


def _copy_cross_variation_arm(arm: CrossVariationArm) -> CrossVariationArm:
    controller = SymmetricV15ClusterController(
        arm.controller.profile,
        cell_seed=0,
        composer_seed=0,
    ).to(arm.controller.procedure_start.device)
    controller.load_state_dict(arm.controller.state_dict(), strict=True)
    router = SymmetricV15PlasticityRouter(
        local_features=_ROUTER_LOCAL_FEATURES,
        hidden_width=48,
    ).to(arm.controller.procedure_start.device)
    router.load_state_dict(arm.router.state_dict(), strict=True)
    clone = CrossVariationArm(
        controller=controller,
        router=router,
        cell_optimizer_state=clone_cell_adamw_state(arm.cell_optimizer_state),
        composer_optimizer=torch.optim.AdamW(
            controller.relation_composer.parameters(),
            lr=_COMPOSER_LEARNING_RATE,
            betas=(_ADAM_BETA1, _ADAM_BETA2),
            eps=_ADAM_EPSILON,
            weight_decay=0.0,
        ),
        router_optimizer=torch.optim.AdamW(
            router.parameters(),
            lr=_ROUTER_LEARNING_RATE,
            betas=(_ADAM_BETA1, _ADAM_BETA2),
            eps=_ADAM_EPSILON,
            weight_decay=0.0,
        ),
    )
    clone.composer_optimizer.load_state_dict(arm.composer_optimizer.state_dict())
    clone.router_optimizer.load_state_dict(arm.router_optimizer.state_dict())
    if cross_variation_arm_digest(clone) != cross_variation_arm_digest(arm):
        raise RuntimeError("V15 complete arm clone changed lineage")
    return clone


def _paired_evaluation(
    controller: v13.CapacityMatchedClusterController,
    panel_a: Sequence[SoftwarePipelineStream],
    panel_a_rerender: Sequence[SoftwarePipelineStream],
    panel_b: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    return v14._paired_panels(controller, panel_a, panel_a_rerender, panel_b)


def _single_panel_metrics(panel: Mapping[str, object]) -> dict[str, float | int]:
    return v14._single_panel_metrics(panel)


def _specialization_metrics(
    controller: v13.CapacityMatchedClusterController,
    streams: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    if len(streams) != 16:
        raise ValueError("V15 specialization requires sixteen panel streams")
    before = v13.software_pipeline_model_digest(controller)
    evidence = _collect_homogeneous_cell_evidence(controller, streams)
    winner_counts = [0] * _CELL_COUNT
    margins = []
    winners = []
    for stream_index in range(len(streams)):
        losses = evidence.losses[:, stream_index]
        ordered = torch.argsort(losses)
        winner = int(ordered[0].item())
        margin = float((losses[ordered[1]] - losses[ordered[0]]).item())
        winners.append(winner)
        margins.append(margin)
        if margin >= 1.0e-4:
            winner_counts[winner] += 1
    if before != v13.software_pipeline_model_digest(controller):
        raise RuntimeError("V15 specialization read mutated controller")
    return {
        "unique_margin": 1.0e-4,
        "winner_counts": tuple(winner_counts),
        "winning_cells_with_two_streams": sum(value >= 2 for value in winner_counts),
        "stream_winners": tuple(winners),
        "winner_margins": tuple(margins),
        "passed": sum(value >= 2 for value in winner_counts) >= 2,
    }


def _adaptation_batches(replicate: int) -> tuple[CrossVariationBatch, ...]:
    plan = cross_variation_fit_plan()
    specification = plan["replicates"][replicate]
    return tuple(
        _make_cross_variation_batch(plan["commitments"], record)
        for record in specification["adaptation_updates"]
    )


def _cell_state_steps(state: CellAdamWState) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(slot.step for slot in cell) for cell in state)


def _adaptation_diagnostic(
    trained: CrossVariationArm,
    batches: Sequence[CrossVariationBatch],
    probe_streams: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    """Run the frozen four-update correct/uniform/permuted/no-update ablation."""

    if len(batches) != 4 or len(probe_streams) != 8:
        raise ValueError("V15 adaptation diagnostic shape changed")
    before = cross_variation_arm_digest(trained)
    arms = {
        label: _copy_cross_variation_arm(trained)
        for label in ("correct", "uniform", "cell_permuted", "no_update")
    }
    reports: dict[str, object] = {}
    for label, arm in arms.items():
        initial_composer = _module_digest(arm.controller.relation_composer)
        initial_router = cross_variation_router_digest(arm.router)
        initial_steps = _cell_state_steps(arm.cell_optimizer_state)
        updates = []
        if label != "no_update":
            for update_index, batch in enumerate(batches):
                evidence = collect_cross_variation_evidence(
                    arm.controller,
                    batch.streams,
                    arm.cell_optimizer_state,
                )
                combined, _, _ = _combined_allocations(
                    evidence,
                    batch,
                    arm.router,
                    learned_plasticity=True,
                )
                if label == "uniform":
                    allocation = torch.full_like(combined, 1.0 / _CELL_COUNT)
                elif label == "cell_permuted":
                    allocation = torch.roll(combined, shifts=1, dims=0)
                else:
                    allocation = combined
                directions, raw_norms, clipped_norms = _routed_directions(
                    evidence,
                    allocation,
                    tuple(range(8)),
                )
                _apply_cell_adamw_update(arm, directions)
                updates.append(
                    {
                        "update": update_index,
                        "allocations": tuple(
                            tuple(float(value) for value in row)
                            for row in allocation.detach().tolist()
                        ),
                        "direction_norms": tuple(float(value.item()) for value in raw_norms),
                        "clipped_direction_norms": tuple(
                            float(value.item()) for value in clipped_norms
                        ),
                    }
                )
        panel = v13.evaluate_public_relation_credit_panel(arm.controller, probe_streams)
        if (
            _module_digest(arm.controller.relation_composer) != initial_composer
            or cross_variation_router_digest(arm.router) != initial_router
        ):
            raise RuntimeError("V15 adaptation stepped composer or router")
        terminal_steps = _cell_state_steps(arm.cell_optimizer_state)
        expected_increment = 0 if label == "no_update" else len(batches)
        if any(
            after != initial + expected_increment
            for before_cell, after_cell in zip(initial_steps, terminal_steps, strict=True)
            for initial, after in zip(before_cell, after_cell, strict=True)
        ):
            raise RuntimeError("V15 adaptation did not continue exact AdamW state")
        reports[label] = {
            "updates": tuple(updates),
            "probe": _single_panel_metrics(panel),
            "initial_cell_steps": initial_steps,
            "terminal_cell_steps": terminal_steps,
            "composer_stepped": False,
            "router_stepped": False,
        }
    if before != cross_variation_arm_digest(trained):
        raise RuntimeError("V15 adaptation mutated the trained lineage")
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


def _to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


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
        "cell_optimizer_state": _state_payload(arm.cell_optimizer_state),
        "composer_optimizer_state": _to_cpu(arm.composer_optimizer.state_dict()),
        "router_optimizer_state": _to_cpu(arm.router_optimizer.state_dict()),
        "digest": cross_variation_arm_digest(arm),
    }


def save_cross_variation_checkpoint(
    path: str | Path,
    systems: Sequence[tuple[CrossVariationArm, CrossVariationArm]],
) -> None:
    if len(systems) != len(_REPLICATE_SEEDS):
        raise ValueError("V15 checkpoint requires all three replicates")
    records = []
    for replicate, (uniform, learned) in enumerate(systems):
        records.append(
            {
                "replicate": replicate,
                "uniform": _arm_checkpoint_record(uniform),
                "learned": _arm_checkpoint_record(learned),
            }
        )
    torch.save(
        {
            "version": _CHECKPOINT_VERSION,
            "protocol_id": _PROTOCOL_ID,
            "plan": cross_variation_fit_plan(),
            "plan_digest": cross_variation_plan_digest(),
            "replicates": tuple(records),
        },
        Path(path),
    )


def _restore_arm_record(arm: CrossVariationArm, record: object) -> None:
    if not isinstance(record, dict):
        raise RuntimeError("V15 checkpoint arm record is invalid")
    arm.controller.load_state_dict(record["controller_state"], strict=True)
    arm.router.load_state_dict(record["router_state"], strict=True)
    arm.cell_optimizer_state = _state_from_payload(
        record["cell_optimizer_state"],
        arm.controller,
    )
    arm.composer_optimizer.load_state_dict(record["composer_optimizer_state"])
    arm.router_optimizer.load_state_dict(record["router_optimizer_state"])
    if record.get("digest") != cross_variation_arm_digest(arm):
        raise RuntimeError("V15 checkpoint learned lineage changed")


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
        or payload.get("plan") != cross_variation_fit_plan()
        or payload.get("plan_digest") != cross_variation_plan_digest()
    ):
        raise RuntimeError("V15 checkpoint identity or seed plan is invalid")
    records = payload.get("replicates")
    if not isinstance(records, (tuple, list)) or len(records) != 3:
        raise RuntimeError("V15 checkpoint replicate set changed")
    systems = []
    for replicate, record in enumerate(records):
        if not isinstance(record, dict) or record.get("replicate") != replicate:
            raise RuntimeError("V15 checkpoint replicate identity changed")
        uniform, learned = build_cross_variation_pair(replicate, device=device)
        _restore_arm_record(uniform, record["uniform"])
        _restore_arm_record(learned, record["learned"])
        uniform.controller.eval()
        learned.controller.eval()
        uniform.router.eval()
        learned.router.eval()
        systems.append((uniform, learned))
    return tuple(systems)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _frozen_dependency_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    observed = {
        relative: _file_sha256(root / relative) for relative in _FROZEN_FILE_HASHES
    }
    if observed != _FROZEN_FILE_HASHES:
        raise RuntimeError("V15 frozen evaluator or inherited runner bytes changed")
    if Path(evaluator.__file__).resolve() != (
        root / "experiments/evaluators/software_pipeline_reconstruction_suite.py"
    ).resolve():
        raise RuntimeError("V15 imported an unexpected evaluator module")
    return observed


def _summary_values(summary: Mapping[str, object]) -> tuple[int, int, float]:
    return (
        int(summary["supported_rows"]),
        int(summary["qualifying_streams"]),
        float(summary["target_loss_mean"]),
    )


def _aggregate_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    if not summaries:
        raise ValueError("V15 aggregate requires summaries")
    return {
        "supported_rows": sum(int(value["supported_rows"]) for value in summaries),
        "qualifying_streams": sum(
            int(value["qualifying_streams"]) for value in summaries
        ),
        "target_loss_mean": sum(
            float(value["target_loss_mean"]) for value in summaries
        )
        / len(summaries),
    }


def _material_loss_gain(candidate: float, control: float, fraction: float) -> bool:
    return control - candidate >= max(1.0e-4, fraction * control)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires values")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def classify_cross_variation_result(
    *,
    integrity_passed: bool,
    uniform_competent: bool,
    uniform_materially_better: bool,
    learned_competent: bool,
    every_support_rule_passed: bool,
) -> str:
    """Apply the frozen V15 classification precedence without score repair."""

    values = (
        integrity_passed,
        uniform_competent,
        uniform_materially_better,
        learned_competent,
        every_support_rule_passed,
    )
    if any(type(value) is not bool for value in values):
        raise TypeError("V15 classification inputs must be bool")
    if not integrity_passed:
        return "INVALID_NO_CLAIM"
    if uniform_competent and uniform_materially_better:
        return "PLASTICITY_ROUTER_HARMFUL"
    if not learned_competent:
        return "NO_COMPETENCE"
    if every_support_rule_passed:
        return "PLASTICITY_ROUTER_SUPPORTED"
    return "PLASTICITY_ROUTER_NULL"


def _routing_aggregate(replicate_reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    replicate_values = []
    all_tv = []
    all_routes = []
    for report in replicate_reports:
        updates = report["fits"]["learned_episodic_plasticity"]["updates"][1:]
        tv_values = []
        routes = []
        for update in updates:
            for key in ("lane_a_route", "lane_b_route"):
                route = torch.tensor(update[key], dtype=torch.float64)
                uniform = torch.full_like(route, 0.25)
                tv_values.extend(
                    float(value)
                    for value in (0.5 * (route - uniform).abs().sum(dim=0)).tolist()
                )
                routes.append(route)
        pair_distance = sum(float(value["matched_route_distance"]) for value in updates) / len(updates)
        unpaired_distance = sum(
            float(value["unmatched_route_distance"]) for value in updates
        ) / len(updates)
        gap = unpaired_distance - pair_distance
        replicate_values.append(
            {
                "replicate": int(report["replicate"]),
                "tv_mean": sum(tv_values) / len(tv_values),
                "tv_median": _median(tv_values),
                "matched_distance": pair_distance,
                "unmatched_distance": unpaired_distance,
                "distance_gap": gap,
                "ratio_passed": pair_distance <= 0.75 * unpaired_distance,
                "gap_passed": gap >= 0.015,
            }
        )
        all_tv.extend(tv_values)
        all_routes.extend(routes)
    usage = torch.cat(all_routes, dim=1).mean(dim=1)
    macro_pair = sum(value["matched_distance"] for value in replicate_values) / 3.0
    macro_unpaired = sum(value["unmatched_distance"] for value in replicate_values) / 3.0
    return {
        "post_first_route_tv_mean": sum(all_tv) / len(all_tv),
        "post_first_route_tv_median": _median(all_tv),
        "aggregate_cell_usage": tuple(float(value) for value in usage.tolist()),
        "macro_matched_distance": macro_pair,
        "macro_unmatched_distance": macro_unpaired,
        "macro_distance_gap": macro_unpaired - macro_pair,
        "replicates": tuple(replicate_values),
    }


def _build_support_checks(
    replicate_reports: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    uniform_summaries = tuple(
        report["evaluations"]["uniform_adamw_plasticity"]["summary"]
        for report in replicate_reports
    )
    learned_summaries = tuple(
        report["evaluations"]["learned_episodic_plasticity"]["summary"]
        for report in replicate_reports
    )
    uniform = _aggregate_summaries(uniform_summaries)
    learned = _aggregate_summaries(learned_summaries)
    uniform_rows, uniform_streams, uniform_loss = _summary_values(uniform)
    learned_rows, learned_streams, learned_loss = _summary_values(learned)
    learned_per_rep = all(
        int(value["supported_rows"]) >= 32
        and int(value["qualifying_streams"]) >= 4
        for value in learned_summaries
    )
    learned_competent = (
        learned_rows >= 120
        and learned_streams >= 18
        and learned_per_rep
    )
    uniform_competent = (
        uniform_rows >= 120
        and uniform_streams >= 18
        and all(
            int(value["supported_rows"]) >= 32
            and int(value["qualifying_streams"]) >= 4
            for value in uniform_summaries
        )
    )
    paired_winning_replicates = sum(
        float(learned_value["target_loss_mean"])
        <= 0.99 * float(uniform_value["target_loss_mean"])
        and int(learned_value["supported_rows"])
        >= int(uniform_value["supported_rows"])
        and int(learned_value["qualifying_streams"])
        >= int(uniform_value["qualifying_streams"])
        for learned_value, uniform_value in zip(
            learned_summaries,
            uniform_summaries,
            strict=True,
        )
    )
    paired_passed = (
        learned_rows >= uniform_rows + 6
        and learned_streams >= uniform_streams + 3
        and _material_loss_gain(learned_loss, uniform_loss, 0.02)
        and paired_winning_replicates >= 2
    )
    uniform_better_replicates = sum(
        float(uniform_value["target_loss_mean"])
        <= 0.99 * float(learned_value["target_loss_mean"])
        and int(uniform_value["supported_rows"])
        >= int(learned_value["supported_rows"])
        and int(uniform_value["qualifying_streams"])
        >= int(learned_value["qualifying_streams"])
        for learned_value, uniform_value in zip(
            learned_summaries,
            uniform_summaries,
            strict=True,
        )
    )
    uniform_materially_better = (
        uniform_rows >= learned_rows + 6
        and uniform_streams >= learned_streams + 3
        and _material_loss_gain(uniform_loss, learned_loss, 0.02)
        and uniform_better_replicates >= 2
    )
    routing = _routing_aggregate(replicate_reports)
    routing_passed = (
        float(routing["post_first_route_tv_mean"]) >= 0.10
        and float(routing["post_first_route_tv_median"]) >= 0.075
        and all(0.10 <= value <= 0.40 for value in routing["aggregate_cell_usage"])
        and float(routing["macro_matched_distance"])
        <= 0.75 * float(routing["macro_unmatched_distance"])
        and float(routing["macro_distance_gap"]) >= 0.025
        and sum(value["distance_gap"] >= 0.015 for value in routing["replicates"])
        >= 2
    )

    adaptation_by_arm = {
        label: tuple(
            report["adaptation"]["arms"][label]["probe"]
            for report in replicate_reports
        )
        for label in ("correct", "uniform", "cell_permuted", "no_update")
    }
    adaptation_aggregate = {
        label: _aggregate_summaries(values)
        for label, values in adaptation_by_arm.items()
    }
    correct = adaptation_aggregate["correct"]
    adaptation_absolute = (
        int(correct["supported_rows"]) >= 48
        and int(correct["qualifying_streams"]) >= 6
        and all(
            int(value["supported_rows"]) >= 12
            and int(value["qualifying_streams"]) >= 1
            for value in adaptation_by_arm["correct"]
        )
    )
    adaptation_controls = {}
    for control in ("uniform", "cell_permuted", "no_update"):
        reference = adaptation_aggregate[control]
        material_replicates = sum(
            _material_loss_gain(
                float(candidate["target_loss_mean"]),
                float(baseline["target_loss_mean"]),
                0.01,
            )
            for candidate, baseline in zip(
                adaptation_by_arm["correct"],
                adaptation_by_arm[control],
                strict=True,
            )
        )
        adaptation_controls[control] = {
            "material_loss_winning_replicates": material_replicates,
            "passed": (
                int(correct["supported_rows"]) >= int(reference["supported_rows"]) + 3
                and int(correct["qualifying_streams"])
                >= int(reference["qualifying_streams"]) + 1
                and _material_loss_gain(
                    float(correct["target_loss_mean"]),
                    float(reference["target_loss_mean"]),
                    0.01,
                )
                and material_replicates >= 2
            ),
        }
    adaptation_passed = adaptation_absolute and all(
        value["passed"] for value in adaptation_controls.values()
    )

    comparator_labels = ("uniform",) + tuple(
        f"single_{index}" for index in range(4)
    ) + tuple(f"drop_{index}" for index in range(4))
    harmonization = {}
    for label in comparator_labels:
        comparator_summaries = tuple(
            report["evaluations"]["learned_episodic_plasticity"]["read_lesions"][label]
            for report in replicate_reports
        )
        comparator = _aggregate_summaries(comparator_summaries)
        per_rep = sum(
            _material_loss_gain(
                float(normal["target_loss_mean"]),
                float(lesion["target_loss_mean"]),
                0.01,
            )
            and int(normal["supported_rows"]) >= int(lesion["supported_rows"])
            and int(normal["qualifying_streams"])
            >= int(lesion["qualifying_streams"])
            for normal, lesion in zip(
                learned_summaries,
                comparator_summaries,
                strict=True,
            )
        )
        harmonization[label] = {
            "aggregate": comparator,
            "material_nonregressing_replicates": per_rep,
            "passed": (
                learned_rows >= int(comparator["supported_rows"]) + 2
                and learned_streams >= int(comparator["qualifying_streams"]) + 1
                and _material_loss_gain(
                    learned_loss,
                    float(comparator["target_loss_mean"]),
                    0.01,
                )
                and per_rep >= 2
            ),
        }
    specialization_passed = sum(
        report["specialization"]["passed"] is True for report in replicate_reports
    ) >= 2
    harmonization_passed = all(
        value["passed"] for value in harmonization.values()
    ) and specialization_passed
    integrity_checks = {
        "paired_starts_exact": all(report["paired_start_exact"] for report in replicate_reports),
        "paired_exposure_exact": all(report["paired_exposure_exact"] for report in replicate_reports),
        "fit_exposure_exact": all(
            fit["optimizer_steps"] == 80
            and fit["streams"] == 640
            and fit["rows"] == 2_560
            and fit["virtual_folds_per_update"] == 2
            for report in replicate_reports
            for fit in report["fits"].values()
        ),
        "surface_stability": all(
            evaluation["surface_stability"]["passed"] is True
            for report in replicate_reports
            for evaluation in report["evaluations"].values()
        ),
        "all_routers_changed": all(
            fit["router_changed"] is True
            for report in replicate_reports
            for fit in report["fits"].values()
        ),
        "all_cells_advanced_eighty_steps": all(
            all(
                slot.step == 80
                for cell in arm.cell_optimizer_state
                for slot in cell
            )
            for report in replicate_reports
            for arm in report["_systems"]
        ),
        "frozen_dependencies_exact": True,
    }
    integrity_passed = all(integrity_checks.values())
    support_checks = {
        "integrity": integrity_passed,
        "absolute_learned_competence": learned_competent,
        "paired_learned_advantage": paired_passed,
        "routing_specialization": routing_passed,
        "fresh_adaptation": adaptation_passed,
        "read_harmonization": harmonization_passed,
    }
    diagnostics = {
        "uniform": uniform,
        "learned": learned,
        "paired_winning_replicates": paired_winning_replicates,
        "uniform_better_replicates": uniform_better_replicates,
        "uniform_competent": uniform_competent,
        "uniform_materially_better": uniform_materially_better,
        "routing": routing,
        "adaptation": adaptation_aggregate,
        "adaptation_controls": adaptation_controls,
        "harmonization": harmonization,
        "specialization_passing_replicates": sum(
            report["specialization"]["passed"] is True for report in replicate_reports
        ),
        "integrity_checks": integrity_checks,
    }
    return support_checks, diagnostics


def fit_cross_variation_pilot(
    *,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """Run the one fixed three-replicate V15 experiment without adaptation."""

    frozen_hashes = _frozen_dependency_hashes()
    plan = cross_variation_fit_plan()
    commitments = plan["commitments"]
    systems: list[tuple[CrossVariationArm, CrossVariationArm]] = []
    replicate_reports: list[dict[str, object]] = []
    started = time.perf_counter()
    for specification in plan["replicates"]:
        replicate = int(specification["replicate"])
        uniform, learned = build_cross_variation_pair(replicate, device=device)
        systems.append((uniform, learned))
        initial_uniform = cross_variation_arm_digest(uniform)
        initial_learned = cross_variation_arm_digest(learned)
        batches = tuple(
            _make_cross_variation_batch(commitments, record)
            for record in specification["train_updates"]
        )
        fits: dict[str, object] = {}
        for label in specification["arm_order"]:
            if label == "uniform_adamw_plasticity":
                fits[label] = fit_cross_variation_batches(
                    uniform,
                    batches,
                    learned_plasticity=False,
                )
            elif label == "learned_episodic_plasticity":
                fits[label] = fit_cross_variation_batches(
                    learned,
                    batches,
                    learned_plasticity=True,
                )
            else:
                raise RuntimeError("V15 arm order changed")
        panel_a = v13._relation_credit_panel_streams(
            commitments,
            specification["panel_a_seed_pairs"],
        )
        panel_a_rerender = v13._relation_credit_panel_streams(
            commitments,
            specification["panel_a_rerender_seed_pairs"],
        )
        panel_b = v13._relation_credit_panel_streams(
            commitments,
            specification["panel_b_seed_pairs"],
        )
        probe = v13._relation_credit_panel_streams(
            commitments,
            specification["probe_seed_pairs"],
        )
        evaluations = {
            "uniform_adamw_plasticity": _paired_evaluation(
                uniform.controller,
                panel_a,
                panel_a_rerender,
                panel_b,
            ),
            "learned_episodic_plasticity": _paired_evaluation(
                learned.controller,
                panel_a,
                panel_a_rerender,
                panel_b,
            ),
        }
        adaptation = _adaptation_diagnostic(
            learned,
            _adaptation_batches(replicate),
            probe,
        )
        specialization = _specialization_metrics(
            learned.controller,
            panel_a + panel_b,
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
    support_checks, diagnostics = _build_support_checks(replicate_reports)
    status = classify_cross_variation_result(
        integrity_passed=bool(support_checks["integrity"]),
        uniform_competent=bool(diagnostics["uniform_competent"]),
        uniform_materially_better=bool(diagnostics["uniform_materially_better"]),
        learned_competent=bool(support_checks["absolute_learned_competence"]),
        every_support_rule_passed=all(bool(value) for value in support_checks.values()),
    )
    for report in replicate_reports:
        del report["_systems"]
    return {
        "protocol_id": _PROTOCOL_ID,
        "status": status,
        "plasticity_router_supported": status == "PLASTICITY_ROUTER_SUPPORTED",
        "plan": plan,
        "plan_digest": cross_variation_plan_digest(),
        "frozen_dependency_hashes": frozen_hashes,
        "replicates": tuple(replicate_reports),
        "aggregate": diagnostics,
        "support_checks": support_checks,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_written": checkpoint_path is not None,
        "context_or_joint_training_performed": False,
        "development_or_final_access": False,
        "wrong_evidence_training_streams": 0,
        "stored_examples_or_replay": 0,
        "scalar_judge_calls": 0,
        "deterministic_solver_used": False,
        "result_conditioned_continuation": False,
    }


__all__ = [
    "AdamWSlot",
    "CrossVariationArm",
    "CrossVariationBatch",
    "CrossVariationEvidence",
    "CrossVariationMetaResult",
    "build_cross_variation_pair",
    "build_training_batches",
    "classify_cross_variation_result",
    "clone_cell_adamw_state",
    "collect_cross_variation_evidence",
    "cross_variation_arm_digest",
    "cross_variation_fit_plan",
    "cross_variation_meta_gradients",
    "cross_variation_plan_digest",
    "cross_variation_router_digest",
    "fit_cross_variation_batches",
    "fit_cross_variation_pilot",
    "functional_adamw_step",
    "initial_cell_adamw_state",
    "load_cross_variation_checkpoint",
    "save_cross_variation_checkpoint",
]
