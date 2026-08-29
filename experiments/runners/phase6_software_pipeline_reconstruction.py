"""Learned cross-package reconstruction for synthetic software pipelines.

The controller deliberately separates two kinds of competence.  A public
pointer lane retains exact state/component observations inside one package,
while a learned structural-role lane is invariant to package-local renaming
and is the only lane allowed to transfer between packages.  Both lanes are
bounded transactional memories.  A reused recurrent backward reasoner emits
one declared component or ``STOP`` at a time; this module never constructs or
scores complete pipeline candidates and never imports evaluator internals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from angler.procedures.records import Goal, GroundAction, Record, State, Trace, Transition
from angler.procedures.trunk import FrozenHashTextEncoder
from experiments.evaluators.software_pipeline_reconstruction_suite import (
    CommittedSoftwarePipeline,
    GeneratedSoftwarePipelineTask,
    PublicComponentContract,
    PublicSoftwarePipelineTask,
    SoftwarePipelineStream,
    commit_software_pipeline,
    judge_software_pipeline_attempt,
    make_software_pipeline_control_stream,
    make_software_pipeline_stream,
    software_pipeline_mechanism_partition,
)
from experiments.runners.phase5_glyph_machine_trace import (
    GlyphAssociativeMemory,
    GlyphAssociativeState,
    GlyphBackwardProcedureReasoner,
    GlyphMachineRunProfile,
    GlyphTransitionLattice,
    restore_glyph_state,
    snapshot_glyph_state,
)
from experiments.runners.phase5_skill_memory_stream import ConditionalReversibleTransition


_CHECKPOINT_VERSION = "angler.phase6-software-pipeline.v6"
_CONFLICT_CHECKPOINT_VERSION = "angler.phase6-public-conflict-reconcile.v1"
_STATE_DIGEST_DOMAIN = b"project-angler.software-pipeline.state.v6\x00"
_MODEL_DIGEST_DOMAIN = b"project-angler.software-pipeline.model.v6\x00"
_CONFLICT_MIXER_DIGEST_DOMAIN = b"project-angler.conflict-mixer.v1\x00"
_CONFLICT_DIRECTION_DIGEST_DOMAIN = b"project-angler.conflict-direction.v1\x00"
_CONFLICT_SYSTEM_DIGEST_DOMAIN = b"project-angler.conflict-system.v1\x00"
_CONFLICT_WEIGHT_TRACE_DIGEST_DOMAIN = b"project-angler.conflict-weight-trace.v1\x00"
_TASK_BINDING_DOMAIN = b"project-angler.software-pipeline.task.v1\x00"
_POINTER_ID_DOMAIN = b"project-angler.software-pipeline.pointer.v1\x00"
_POINTER_WORDS = 4
_MAX_STEPS = 4
_ROLE_RESIDUAL_LIMIT = 0.25
_TRAINING_ATTEMPTS = 2
_INITIAL_EVIDENCE_ACTION_GATE = 0.10
_PUBLIC_RETRIEVAL_MARGIN = 0.10
_PUBLIC_RETRIEVAL_MARGIN_WEIGHT = 0.50
_PUBLIC_ROLE_CAUSAL_MARGIN_NATS = 0.05
_PUBLIC_ROLE_CAUSAL_MARGIN_WEIGHT = 1.0
_RELATION_FIT_MARGIN = 0.10
_RELATION_FIT_BALANCE_WEIGHT = 0.10
_RELATION_CREDIT_TEMPERATURE = 0.025
_RELATION_CONTEXT_TARGET_TEMPERATURE = 0.10
_RELATION_CONTEXT_AUX_TEMPERATURE = 0.25
_RELATION_CREDIT_CONTEXT_WEIGHT = 0.50
_RELATION_CREDIT_INSTANCE_WEIGHT = 0.25
_RELATION_CREDIT_SEPARATION_WEIGHT = 0.25
_RELATION_FIT_COMMITMENTS = 32
_RELATION_GATE_COMMITMENT_OFFSET = 32
_RELATION_GATE_COMMITMENTS = 8
_RELATION_FIT_SEED_PAIRS = (
    (26_082_801, 36_082_801),
    (26_082_802, 36_082_802),
)
_RELATION_GATE_SEED_PAIRS = (
    (26_082_811, 36_082_811),
    (26_082_812, 36_082_812),
    (26_082_813, 36_082_813),
)
_RELATION_GATE_ROWS = 96
_RELATION_GATE_SIGN_ROWS = 77
_RELATION_GATE_PERMUTATION_TOLERANCE = 1.0e-6
_RELATION_GATE_RERENDER_RETENTION = 0.80
_RELATION_GATE_RERENDER_SIGN_FRACTION = 0.90
_LEGACY_RELATION_PROTOCOL_ID = "phase6.public-paired-wrong-evidence.audit.v2"
_RELATION_PROTOCOL_ID = "phase6.public-relation-credit.v11"
_RELATION_CREDIT_INITIALIZATION_SEED = 2_026_082_891
_RELATION_CREDIT_COMMITMENTS = 8
_RELATION_CREDIT_STREAMS_PER_UPDATE = 8
_RELATION_CREDIT_STAGE_UPDATES = {
    "relation": 80,
    "context": 25,
    "joint": 35,
}
_RELATION_CREDIT_TRAIN_TOPOLOGY_BASE = 1_901_000_001
_RELATION_CREDIT_TRAIN_SURFACE_BASE = 2_001_000_001
_RELATION_CREDIT_PANEL_TOPOLOGY_BASE = 2_021_000_001
_RELATION_CREDIT_PANEL_SURFACE_BASE = 2_031_000_001
_RELATION_CREDIT_FINAL_TOPOLOGY_BASE = 2_041_000_001
_RELATION_CREDIT_FINAL_SURFACE_BASE = 2_051_000_001
_RELATION_CREDIT_STREAM_TEMPERATURE = 0.05
_RELATION_CREDIT_STREAM_MEAN_WEIGHT = 0.50
_RELATION_CREDIT_STREAM_ROBUST_WEIGHT = 0.50
_RELATION_CREDIT_ROW_TEMPERATURE = 0.05
_RELATION_CREDIT_ROW_MEAN_WEIGHT = 0.50
_RELATION_CREDIT_ROW_ROBUST_WEIGHT = 0.50
_RELATION_CREDIT_RELATION_CONFIDENT_ROWS = 24
_RELATION_CREDIT_RELATION_CONFIDENT_STREAMS = 6
_RELATION_CREDIT_FINAL_SIGNED_ROWS = 26
_RELATION_CREDIT_FINAL_SIGNED_STREAMS = 7
_RELATION_CREDIT_CONTEXT_TOP_ONE = 0.80
_RELATION_CREDIT_CONTEXT_MASS = 0.60
_CONFLICT_PROTOCOL_ID = "phase6.public-conflict-reconcile.single.v12"
_CONFLICT_INITIALIZATION_SEED = 2_026_082_931
_CONFLICT_MIXER_INITIALIZATION_SEED = 2_026_082_932
_CONFLICT_TRAIN_TOPOLOGY_BASE = 3_301_000_001
_CONFLICT_TRAIN_SURFACE_BASE = 3_401_000_001
_CONFLICT_PANEL_TOPOLOGY_BASE = 3_501_000_001
_CONFLICT_PANEL_SURFACE_BASE = 3_511_000_001
_CONFLICT_FINAL_TOPOLOGY_BASE = 3_601_000_001
_CONFLICT_FINAL_SURFACE_BASE = 3_611_000_001
_CONFLICT_MIXER_FEATURES = 6
_CONFLICT_MIXER_HIDDEN_WIDTH = 32
_CONFLICT_MIXER_ANCHOR_WEIGHT = 0.50
_CONFLICT_MIXER_LEARNING_RATE = 1.0e-3
_CONFLICT_ALIGNMENT_MARGIN = 0.05
_CONFLICT_ALIGNMENT_TEMPERATURE = 0.05
_CONFLICT_META_MEAN_WEIGHT = 0.50
_CONFLICT_META_ROBUST_WEIGHT = 0.50
_CONFLICT_META_KL_WEIGHT = 0.01
_CLUSTER_PROTOCOL_ID = "phase6.public-anonymous-cluster.paired.v13"
_CLUSTER_CHECKPOINT_VERSION = "angler.phase6-public-anonymous-cluster.v1"
_CLUSTER_DIGEST_DOMAIN = b"project-angler.anonymous-relation-cluster.v1\x00"
_CLUSTER_CELL_COUNT = 4
_CLUSTER_CELL_WIDTH = 16
_CLUSTER_CELL_HIDDEN_WIDTH = 32
_CLUSTER_COMPOSER_HIDDEN_WIDTH = 41
_CLUSTER_COMPOSER_ANCHOR_WEIGHT = 0.50
_CLUSTER_REPLICATE_SEEDS = (
    (2_026_083_101, 2_026_083_102, 2_026_083_103, 2_026_083_104),
    (2_026_083_111, 2_026_083_112, 2_026_083_113, 2_026_083_114),
    (2_026_083_121, 2_026_083_122, 2_026_083_123, 2_026_083_124),
)
_CLUSTER_TRAIN_TOPOLOGY_BASE = 4_001_000_001
_CLUSTER_TRAIN_SURFACE_BASE = 4_041_000_001
_CLUSTER_PANEL_A_TOPOLOGY_BASE = 4_081_000_001
_CLUSTER_PANEL_A_SURFACE_BASE = 4_121_000_001
_CLUSTER_PANEL_A_RERENDER_SURFACE_BASE = 4_161_000_001
_CLUSTER_PANEL_B_TOPOLOGY_BASE = 4_201_000_001
_CLUSTER_PANEL_B_SURFACE_BASE = 4_241_000_001
_CLUSTER_REPLICATE_SEED_STRIDE = 10_000_000
_CLUSTER_PARAMETER_TOLERANCE_FRACTION = 0.001


class AnonymousConflictMixer(nn.Module):
    """Learn blockwise stream credit without stream or task identities."""

    def __init__(
        self,
        *,
        feature_count: int = _CONFLICT_MIXER_FEATURES,
        hidden_width: int = _CONFLICT_MIXER_HIDDEN_WIDTH,
        anchor_weight: float = _CONFLICT_MIXER_ANCHOR_WEIGHT,
    ) -> None:
        super().__init__()
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count <= 0
            or isinstance(hidden_width, bool)
            or not isinstance(hidden_width, int)
            or hidden_width <= 0
        ):
            raise ValueError("conflict mixer dimensions must be positive integers")
        if not math.isfinite(anchor_weight) or not 0.0 < anchor_weight <= 1.0:
            raise ValueError("conflict mixer anchor weight must be in (0, 1]")
        self.feature_count = feature_count
        self.hidden_width = hidden_width
        self.anchor_weight = float(anchor_weight)
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.residual_scorer = nn.Sequential(
            nn.Linear(2 * hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1, bias=False),
        )
        nn.init.zeros_(self.residual_scorer[-1].weight)

    def forward(
        self,
        stream_losses: torch.Tensor,
        base_weights: torch.Tensor,
        gradient_norms: torch.Tensor,
        cosine_grams: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return anchored learned weights for each anonymous parameter block."""

        if (
            stream_losses.ndim != 1
            or stream_losses.numel() < 2
            or base_weights.shape != stream_losses.shape
            or gradient_norms.ndim != 2
            or gradient_norms.shape[1] != stream_losses.numel()
            or cosine_grams.shape
            != (
                gradient_norms.shape[0],
                stream_losses.numel(),
                stream_losses.numel(),
            )
            or stream_losses.device != base_weights.device
            or stream_losses.device != gradient_norms.device
            or stream_losses.device != cosine_grams.device
            or stream_losses.dtype != base_weights.dtype
            or stream_losses.dtype != gradient_norms.dtype
            or stream_losses.dtype != cosine_grams.dtype
            or not stream_losses.is_floating_point()
            or not bool(torch.isfinite(stream_losses).all().item())
            or not bool(torch.isfinite(base_weights).all().item())
            or not bool(torch.isfinite(gradient_norms).all().item())
            or not bool(torch.isfinite(cosine_grams).all().item())
            or bool((base_weights <= 0.0).any().item())
            or not torch.allclose(
                base_weights.sum(),
                base_weights.new_tensor(1.0),
                atol=1.0e-6,
                rtol=0.0,
            )
            or bool((gradient_norms < 0.0).any().item())
        ):
            raise ValueError("conflict mixer inputs must be finite and aligned")
        stream_count = stream_losses.numel()
        block_count = gradient_norms.shape[0]
        detached_losses = stream_losses.detach()
        detached_base = base_weights.detach()
        detached_norms = gradient_norms.detach()
        detached_grams = cosine_grams.detach()
        loss_z = _anonymous_standardize(detached_losses, dim=0)
        log_norm_z = _anonymous_standardize(
            torch.log(detached_norms.clamp_min(torch.finfo(detached_norms.dtype).tiny)),
            dim=1,
        )
        diagonal = torch.eye(
            stream_count,
            device=detached_grams.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        off_diagonal = detached_grams.masked_fill(diagonal, 0.0)
        mean_cosine = off_diagonal.sum(dim=-1) / (stream_count - 1)
        minimum_cosine = detached_grams.masked_fill(diagonal, 1.0).amin(dim=-1)
        negative_fraction = (
            (detached_grams < 0.0).masked_fill(diagonal, False).sum(dim=-1)
            / float(stream_count - 1)
        )
        features = torch.stack(
            (
                loss_z.unsqueeze(0).expand(block_count, -1),
                log_norm_z,
                detached_base.unsqueeze(0).expand(block_count, -1),
                mean_cosine,
                minimum_cosine,
                negative_fraction,
            ),
            dim=-1,
        )
        if features.shape[-1] != self.feature_count:
            raise RuntimeError("conflict mixer feature contract changed")
        local = self.local_encoder(features)
        shared = local.mean(dim=1, keepdim=True).expand(-1, stream_count, -1)
        residual_logits = self.residual_scorer(
            torch.cat((local, shared), dim=-1)
        ).squeeze(-1)
        learned_weights = torch.softmax(
            torch.log(detached_base).unsqueeze(0) + residual_logits,
            dim=-1,
        )
        anchored = (
            self.anchor_weight * detached_base.unsqueeze(0)
            + (1.0 - self.anchor_weight) * learned_weights
        )
        return anchored, residual_logits, features


def _anonymous_standardize(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Standardize one anonymous set without batch- or identity-specific state."""

    if (
        not isinstance(values, torch.Tensor)
        or not values.is_floating_point()
        or values.numel() == 0
        or not bool(torch.isfinite(values).all().item())
    ):
        raise ValueError("anonymous standardization requires finite floats")
    centered = values - values.mean(dim=dim, keepdim=True)
    scale = centered.square().mean(dim=dim, keepdim=True).sqrt().clamp_min(1.0e-8)
    return centered / scale


@dataclass(frozen=True, slots=True)
class SoftwarePipelineRunProfile:
    """One architecture at smoke or resource scale."""

    name: str
    width: int
    hidden_width: int
    graph_layers: int
    graph_heads: int
    transition_rank: int
    pointer_slots: int
    role_slots: int
    memory_read_top_k: int

    def __post_init__(self) -> None:
        dimensions = (
            self.width,
            self.hidden_width,
            self.graph_layers,
            self.graph_heads,
            self.transition_rank,
            self.pointer_slots,
            self.role_slots,
            self.memory_read_top_k,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("software-pipeline profile dimensions must be integers")
        if any(value <= 0 for value in dimensions):
            raise ValueError("software-pipeline profile dimensions must be positive")
        if self.width % 2 or self.width % self.graph_heads:
            raise ValueError("profile width must be even and divisible by graph heads")
        if self.transition_rank >= self.width:
            raise ValueError("transition rank must be smaller than width")
        if self.memory_read_top_k > min(self.pointer_slots, self.role_slots):
            raise ValueError("memory top-k exceeds a lane capacity")


SOFTWARE_PIPELINE_PROFILES: Mapping[str, SoftwarePipelineRunProfile] = {
    "smoke": SoftwarePipelineRunProfile(
        name="smoke",
        width=32,
        hidden_width=64,
        graph_layers=1,
        graph_heads=4,
        transition_rank=8,
        pointer_slots=16,
        role_slots=16,
        memory_read_top_k=2,
    ),
    "resource_graph": SoftwarePipelineRunProfile(
        name="resource_graph",
        width=512,
        hidden_width=1024,
        graph_layers=5,
        graph_heads=8,
        transition_rank=64,
        pointer_slots=128,
        role_slots=128,
        memory_read_top_k=4,
    ),
}


@dataclass(frozen=True, slots=True)
class SoftwarePipelineExperimentConfig:
    profile: str
    seed: int
    train_mechanisms: int
    development_mechanisms: int
    final_mechanisms: int
    training_epochs: int
    supports_per_motif: int
    queries_per_mechanism: int
    maximum_steps: int
    learning_rate: float
    gradient_clip: float
    rollout_temperature: float

    def __post_init__(self) -> None:
        if self.profile not in SOFTWARE_PIPELINE_PROFILES:
            raise ValueError("software-pipeline profile is not registered")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("experiment seed must be a nonnegative integer")
        for value, maximum, label in (
            (self.train_mechanisms, 64, "train_mechanisms"),
            (self.development_mechanisms, 16, "development_mechanisms"),
            (self.final_mechanisms, 16, "final_mechanisms"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{label} must be between one and {maximum}")
        for value, label in (
            (self.training_epochs, "training_epochs"),
            (self.supports_per_motif, "supports_per_motif"),
            (self.queries_per_mechanism, "queries_per_mechanism"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.maximum_steps, bool)
            or not isinstance(self.maximum_steps, int)
            or not 1 <= self.maximum_steps <= _MAX_STEPS
        ):
            raise ValueError("maximum_steps must be one through four")
        for value, label in (
            (self.learning_rate, "learning_rate"),
            (self.gradient_clip, "gradient_clip"),
            (self.rollout_temperature, "rollout_temperature"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")


def default_software_pipeline_experiment_config(
    profile: str,
    *,
    seed: int = 131_071,
) -> SoftwarePipelineExperimentConfig:
    if profile == "smoke":
        return SoftwarePipelineExperimentConfig(
            profile=profile,
            seed=seed,
            train_mechanisms=8,
            development_mechanisms=4,
            final_mechanisms=4,
            training_epochs=1,
            supports_per_motif=2,
            queries_per_mechanism=1,
            maximum_steps=4,
            learning_rate=2.0e-3,
            gradient_clip=5.0,
            rollout_temperature=1.0,
        )
    if profile == "resource_graph":
        return SoftwarePipelineExperimentConfig(
            profile=profile,
            seed=seed,
            train_mechanisms=64,
            development_mechanisms=16,
            final_mechanisms=16,
            training_epochs=1,
            supports_per_motif=4,
            queries_per_mechanism=8,
            maximum_steps=4,
            learning_rate=5.0e-4,
            gradient_clip=5.0,
            rollout_temperature=1.0,
        )
    raise ValueError(f"unknown software-pipeline profile: {profile}")


@dataclass(frozen=True, slots=True)
class SoftwareReconstructionState:
    """Two separately ablatable bounded memories with one lineage."""

    pointer: GlyphAssociativeState
    role: GlyphAssociativeState
    context_trace_keys: torch.Tensor
    relation_trace_values: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.pointer, GlyphAssociativeState) or not isinstance(
            self.role, GlyphAssociativeState
        ):
            raise TypeError("software reconstruction lanes must be Glyph states")
        if self.pointer.batch_size != self.role.batch_size:
            raise ValueError("software reconstruction lanes must share batch size")
        if self.pointer.width != self.role.width:
            raise ValueError("software reconstruction lanes must share width")
        if self.pointer.keys.device != self.role.keys.device or (
            self.pointer.keys.dtype != self.role.keys.dtype
        ):
            raise ValueError("software reconstruction lanes must share device and dtype")
        for name, value in (
            ("context trace keys", self.context_trace_keys),
            ("relation trace values", self.relation_trace_values),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != self.role.keys.shape
                or value.device != self.role.keys.device
                or value.dtype != self.role.keys.dtype
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"{name} must match the role memory tensor")
        trace_slots = self.role.slot_count // 2
        allowed = self.role.occupied.clone()
        allowed[:, trace_slots:] = False
        for name, value in (
            ("context trace keys", self.context_trace_keys),
            ("relation trace values", self.relation_trace_values),
        ):
            invalid = value.masked_select(
                (~allowed).unsqueeze(-1).expand_as(value)
            )
            if bool((invalid != 0.0).any().item()):
                raise ValueError(f"{name} must align to occupied trace slots")

    @property
    def batch_size(self) -> int:
        return self.pointer.batch_size

    @property
    def width(self) -> int:
        return self.pointer.width


@dataclass(frozen=True, slots=True)
class SoftwareTaskEncoding:
    pointer_state_embeddings: torch.Tensor
    pointer_component_embeddings: torch.Tensor
    pointer_pair_ids: torch.Tensor
    pointer_successor_ids: torch.Tensor
    role_state_embeddings: torch.Tensor
    role_component_embeddings: torch.Tensor
    operator_embeddings: torch.Tensor
    local_pair_embeddings: torch.Tensor
    relative_effect_embeddings: torch.Tensor
    stop_relation_embeddings: torch.Tensor
    role_pair_keys: torch.Tensor
    relation_context_embeddings: torch.Tensor
    relation_component_embeddings: torch.Tensor
    origin_index: int
    goal_index: int


@dataclass(frozen=True, slots=True)
class SoftwareStepScores:
    logits: torch.Tensor
    action_logits: torch.Tensor
    stop_logit: torch.Tensor
    successor_state_logits: torch.Tensor
    pointer_contexts: torch.Tensor
    role_contexts: torch.Tensor
    outcome_contexts: torch.Tensor
    evidence_match_scores: torch.Tensor
    reasoning_node_codes: torch.Tensor
    current_state_belief: torch.Tensor


@dataclass(frozen=True, slots=True)
class Phase6DenseRoleRead:
    """Phase-6-only dense trace retrieval with an explicit null match."""

    contexts: torch.Tensor
    attention_weights: torch.Tensor
    null_weights: torch.Tensor
    evidence_probabilities: torch.Tensor
    evidence_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class PublicRelationFitRow:
    """One balanced public counterfactual row with two directional arms."""

    heldout_index: int
    transition_index: int
    positive_margin: torch.Tensor
    negative_margin: torch.Tensor
    loss: torch.Tensor

    def __post_init__(self) -> None:
        for value in (self.positive_margin, self.negative_margin, self.loss):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != ()
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).item())
            ):
                raise ValueError("relation-fit row values must be finite scalars")


@dataclass(frozen=True, slots=True)
class PublicRelationCreditRow:
    """One public trace row with training-only slot credit."""

    heldout_index: int
    transition_index: int
    positive_index: int
    negative_index: int
    positive_margin: torch.Tensor
    negative_margin: torch.Tensor
    instance_loss: torch.Tensor
    context_loss: torch.Tensor
    separation_loss: torch.Tensor
    joint_loss: torch.Tensor
    slot_losses: torch.Tensor
    slot_positive_margins: torch.Tensor
    slot_negative_margins: torch.Tensor
    responsibilities: torch.Tensor
    context_weights: torch.Tensor
    context_null_weight: torch.Tensor

    def __post_init__(self) -> None:
        if (
            isinstance(self.heldout_index, bool)
            or not isinstance(self.heldout_index, int)
            or self.heldout_index < 0
            or isinstance(self.transition_index, bool)
            or not isinstance(self.transition_index, int)
            or self.transition_index < 0
            or isinstance(self.positive_index, bool)
            or not isinstance(self.positive_index, int)
            or self.positive_index < 0
            or isinstance(self.negative_index, bool)
            or not isinstance(self.negative_index, int)
            or self.negative_index < 0
            or self.positive_index == self.negative_index
        ):
            raise ValueError("relation-credit row indices are invalid")
        scalars = (
            self.positive_margin,
            self.negative_margin,
            self.instance_loss,
            self.context_loss,
            self.separation_loss,
            self.joint_loss,
            self.context_null_weight,
        )
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != ()
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).item())
            for value in scalars
        ):
            raise ValueError("relation-credit row losses must be finite scalars")
        vectors = (
            self.slot_losses,
            self.slot_positive_margins,
            self.slot_negative_margins,
            self.responsibilities,
            self.context_weights,
        )
        if (
            any(
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
                for value in vectors
            )
            or len({value.shape for value in vectors}) != 1
            or self.responsibilities.numel() == 0
            or bool((self.context_weights < 0.0).any().item())
            or bool((self.context_null_weight < 0.0).item())
            or abs(
                float(
                    (
                        self.context_weights.sum() + self.context_null_weight
                    ).item()
                )
                - 1.0
            )
            > 1.0e-6
        ):
            raise ValueError("relation-credit slot tensors must be aligned vectors")


@dataclass(frozen=True, slots=True)
class _PublicRelationFold:
    heldout_index: int
    transition_index: int
    masked_task: PublicSoftwarePipelineTask
    before: State
    positive_action: GroundAction
    negative_action: GroundAction
    positive_state: SoftwareReconstructionState
    negative_state: SoftwareReconstructionState


@dataclass(frozen=True, slots=True)
class SoftwareNeuralRollout:
    pipeline: CommittedSoftwarePipeline
    step_logits: tuple[torch.Tensor, ...]
    selected_indices: tuple[int, ...]
    step_role_keys: tuple[torch.Tensor, ...]
    step_current_embeddings: tuple[torch.Tensor, ...]
    step_state_beliefs: tuple[torch.Tensor, ...]
    component_count: int
    task_binding: str
    incoming_state_digest: str

    def __post_init__(self) -> None:
        lengths = {
            len(self.step_logits),
            len(self.selected_indices),
            len(self.step_role_keys),
            len(self.step_current_embeddings),
            len(self.step_state_beliefs),
        }
        if len(lengths) != 1 or not self.step_logits:
            raise ValueError("rollout decision records must be aligned and nonempty")
        for logits, selected in zip(self.step_logits, self.selected_indices, strict=True):
            if logits.shape != (self.component_count + 1,):
                raise ValueError("rollout logits must cover components plus STOP")
            if not 0 <= selected <= self.component_count:
                raise ValueError("rollout selection is outside components plus STOP")


@dataclass(frozen=True, slots=True)
class SoftwareTraceAcquisition:
    state: SoftwareReconstructionState
    public_transitions: int
    pointer_writes: int
    role_writes: int


@dataclass(frozen=True, slots=True)
class SoftwareScalarFeedback:
    state: SoftwareReconstructionState
    accepted: bool
    scalar_observations: int
    write_slots: tuple[int, ...]
    delta_norm: float


class RenameInvariantRoleEncoder(nn.Module):
    """Learned equivariant graph encoder over name-free structural features.

    Python-side feature extraction records only counts, equality relations,
    argument arities, origin/goal flags, and graph incidence.  No identifier's
    spelling, digest, task metadata, or evaluator-owned value enters this lane.
    """

    _STATE_FEATURES = 24
    _COMPONENT_FEATURES = 24
    _TOPOLOGY_FEATURES = 1
    _MULTIPLEX_PAIR_FEATURES = 5
    _STATE_COMPONENT_FEATURES = 16
    _RELATIVE_EFFECT_FEATURES = 24
    _STOP_RELATION_FEATURES = 12

    def __init__(self, profile: SoftwarePipelineRunProfile) -> None:
        super().__init__()
        width = profile.width
        hidden_width = profile.hidden_width
        self.width = width
        self.state_projection = nn.Sequential(
            nn.LayerNorm(self._STATE_FEATURES),
            nn.Linear(self._STATE_FEATURES, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.component_projection = nn.Sequential(
            nn.LayerNorm(self._COMPONENT_FEATURES),
            nn.Linear(self._COMPONENT_FEATURES, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        topology_rank = min(hidden_width, max(16, 2 * profile.transition_rank))
        self.topology_input = nn.Sequential(
            nn.LayerNorm(self._TOPOLOGY_FEATURES),
            nn.Linear(self._TOPOLOGY_FEATURES, width),
            nn.SiLU(),
        )
        # A fixed relational depth is independent of motif labels and remains
        # small compared with the main controller.  Directed incoming/outgoing
        # messages distinguish equal-size, equal-degree incidence graphs that
        # no scalar aggregate can separate.
        self.topology_updates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(3 * width),
                nn.Linear(3 * width, topology_rank),
                nn.SiLU(),
                nn.Linear(topology_rank, width),
            )
            for _ in range(4)
        )
        self.topology_pool = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.multiplex_pair_input = nn.Sequential(
            nn.LayerNorm(self._MULTIPLEX_PAIR_FEATURES),
            nn.Linear(self._MULTIPLEX_PAIR_FEATURES, width),
            nn.SiLU(),
        )
        # Ordered node-pair states are updated by a learned, coupled i-k-j
        # aggregation.  This is permutation equivariant and strictly more
        # expressive than pooling node colors: it can retain which anonymous
        # predecessor edge corresponds to which candidate edge without a
        # canonical order, signature, transform label, or motif rule.
        self.multiplex_pair_left = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, topology_rank, bias=False),
                nn.SiLU(),
            )
            for _ in range(3)
        )
        self.multiplex_pair_right = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, topology_rank, bias=False),
                nn.SiLU(),
            )
            for _ in range(3)
        )
        self.multiplex_pair_updates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * width + topology_rank),
                nn.Linear(2 * width + topology_rank, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, width),
            )
            for _ in range(3)
        )
        self.multiplex_pair_pool = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.local_pair_projection = nn.Sequential(
            nn.LayerNorm(self._STATE_COMPONENT_FEATURES),
            nn.Linear(self._STATE_COMPONENT_FEATURES, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.relative_effect_projection = nn.Sequential(
            nn.LayerNorm(self._RELATIVE_EFFECT_FEATURES),
            nn.Linear(self._RELATIVE_EFFECT_FEATURES, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.stop_relation_projection = nn.Sequential(
            nn.LayerNorm(self._STOP_RELATION_FEATURES),
            nn.Linear(self._STOP_RELATION_FEATURES, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.state_relation_updates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(3 * width),
                nn.Linear(3 * width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, width),
            )
            for _ in range(profile.graph_layers)
        )
        self.component_relation_updates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(3 * width),
                nn.Linear(3 * width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, width),
            )
            for _ in range(profile.graph_layers)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=profile.graph_heads,
            dim_feedforward=hidden_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=profile.graph_layers,
            enable_nested_tensor=False,
        )
        self.type_codes = nn.Parameter(torch.empty(2, width))
        self.output_norm = nn.LayerNorm(width)
        nn.init.normal_(self.type_codes, mean=0.0, std=1.0 / math.sqrt(width))

    def forward(
        self,
        task: PublicSoftwarePipelineTask,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        components = _components_in_candidate_order(task)
        reference = self.type_codes
        state_features, state_relations = _state_role_features(task)
        component_features, component_relations = _component_role_features(
            components,
            task.grounded_candidates,
        )
        states = self.state_projection(
            torch.tensor(state_features, device=reference.device, dtype=reference.dtype)
        )
        actions = self.component_projection(
            torch.tensor(
                component_features,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        actions = actions + self._incidence_topology_embeddings(
            components,
            reference,
        )
        operator_embeddings = self._multiplex_relation_embeddings(
            components,
            reference,
        )
        actions = actions + operator_embeddings
        state_relation_tensor = torch.tensor(
            state_relations,
            device=reference.device,
            dtype=reference.dtype,
        )
        component_relation_tensor = torch.tensor(
            component_relations,
            device=reference.device,
            dtype=reference.dtype,
        )
        for state_update, component_update in zip(
            self.state_relation_updates,
            self.component_relation_updates,
            strict=True,
        ):
            states = states + state_update(
                torch.cat(
                    (
                        states,
                        _normalized_relation_pool(state_relation_tensor, states),
                        _normalized_relation_pool(
                            state_relation_tensor.transpose(0, 1), states
                        ),
                    ),
                    dim=-1,
                )
            )
            actions = actions + component_update(
                torch.cat(
                    (
                        actions,
                        _normalized_relation_pool(component_relation_tensor, actions),
                        _normalized_relation_pool(
                            component_relation_tensor.transpose(0, 1), actions
                        ),
                    ),
                    dim=-1,
                )
            )
        joined = torch.cat(
            (states + self.type_codes[0], actions + self.type_codes[1]), dim=0
        ).unsqueeze(0)
        encoded = self.output_norm(self.context(joined)).squeeze(0)
        return (
            encoded[: len(task.states)],
            encoded[len(task.states) :],
            operator_embeddings,
        )

    def _incidence_topology_embeddings(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        pooled = []
        for component in components:
            node_features, adjacency = _incidence_graph(component)
            nodes = self.topology_input(
                torch.tensor(
                    node_features,
                    device=reference.device,
                    dtype=reference.dtype,
                )
            )
            edges = torch.tensor(
                adjacency,
                device=reference.device,
                dtype=reference.dtype,
            )
            for update in self.topology_updates:
                incoming = edges.transpose(0, 1) @ nodes
                outgoing = edges @ nodes
                nodes = nodes + update(
                    torch.cat((nodes, incoming, outgoing), dim=-1)
                )
            pooled.append(
                self.topology_pool(
                    torch.cat((nodes.mean(dim=0), nodes.amax(dim=0)), dim=-1)
                )
            )
        return torch.stack(pooled)

    def _multiplex_relation_embeddings(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        for candidate in components:
            predecessors = tuple(
                component
                for component in components
                if component.output_type == candidate.input_type
            )
            relation_codes = []
            for predecessor in predecessors:
                _, predecessor_adjacency, candidate_adjacency = (
                    _shared_incidence_graphs(predecessor, candidate)
                )
                predecessor_edges = torch.tensor(
                    predecessor_adjacency,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                candidate_edges = torch.tensor(
                    candidate_adjacency,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                relation_codes.append(
                    self._ordered_pair_multiplex_embedding(
                        predecessor_edges,
                        candidate_edges,
                    )
                )
            if relation_codes:
                rows.append(torch.stack(relation_codes).mean(dim=0))
            else:
                rows.append(reference.new_zeros(self.width))
        return torch.stack(rows)

    def _ordered_pair_multiplex_embedding(
        self,
        predecessor_adjacency: torch.Tensor,
        candidate_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a two-edge-colored graph on ordered anonymous node pairs."""

        pairs = self._ordered_pair_multiplex_tensor(
            predecessor_adjacency,
            candidate_adjacency,
        )
        diagonal_pairs = pairs.diagonal(dim1=0, dim2=1).transpose(0, 1)
        return self.multiplex_pair_pool(
            torch.cat(
                (
                    pairs.mean(dim=(0, 1)),
                    pairs.amax(dim=(0, 1)),
                    diagonal_pairs.mean(dim=0),
                    diagonal_pairs.amax(dim=0),
                ),
                dim=-1,
            )
        )

    def _ordered_pair_multiplex_tensor(
        self,
        predecessor_adjacency: torch.Tensor,
        candidate_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final equivariant ordered-pair states before pooling."""

        if (
            predecessor_adjacency.ndim != 2
            or predecessor_adjacency.shape[0] != predecessor_adjacency.shape[1]
            or candidate_adjacency.shape != predecessor_adjacency.shape
            or predecessor_adjacency.device != candidate_adjacency.device
            or predecessor_adjacency.dtype != candidate_adjacency.dtype
            or not predecessor_adjacency.is_floating_point()
            or predecessor_adjacency.shape[0] <= 0
        ):
            raise ValueError("multiplex adjacency matrices must share one square float shape")
        node_count = predecessor_adjacency.shape[0]
        diagonal = torch.eye(
            node_count,
            device=predecessor_adjacency.device,
            dtype=predecessor_adjacency.dtype,
        )
        raw_pairs = torch.stack(
            (
                predecessor_adjacency,
                predecessor_adjacency.transpose(0, 1),
                candidate_adjacency,
                candidate_adjacency.transpose(0, 1),
                diagonal,
            ),
            dim=-1,
        )
        pairs = self.multiplex_pair_input(raw_pairs)
        scale = math.sqrt(node_count)
        for left_map, right_map, update in zip(
            self.multiplex_pair_left,
            self.multiplex_pair_right,
            self.multiplex_pair_updates,
            strict=True,
        ):
            left = left_map(pairs)
            right = right_map(pairs)
            composed = torch.einsum("ikh,kjh->ijh", left, right) / scale
            reverse = pairs.transpose(0, 1)
            pairs = pairs + update(torch.cat((pairs, reverse, composed), dim=-1))
        return pairs

    def multiplex_relation_tensors(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Expose full pair states without interpreting their graph relation."""

        rows: list[tuple[torch.Tensor, ...]] = []
        for candidate in components:
            tensors = []
            for predecessor in components:
                if predecessor.output_type != candidate.input_type:
                    continue
                _, predecessor_adjacency, candidate_adjacency = (
                    _shared_incidence_graphs(predecessor, candidate)
                )
                tensors.append(
                    self._ordered_pair_multiplex_tensor(
                        torch.tensor(
                            predecessor_adjacency,
                            device=reference.device,
                            dtype=reference.dtype,
                        ),
                        torch.tensor(
                            candidate_adjacency,
                            device=reference.device,
                            dtype=reference.dtype,
                        ),
                    )
                )
            rows.append(tuple(tensors))
        return tuple(rows)

    def multiplex_context_relation_tensors(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[tuple[tuple[int, torch.Tensor], ...], ...],
    ]:
        """Expose candidate-independent contexts and directed relation tensors.

        Each predecessor context uses only that predecessor's raw adjacency in
        both multiplex channels.  A directed relation instead uses the shared
        anonymous node set of the predecessor and candidate.  Component tuple
        positions associate tensors for pooling but are never encoded.
        """

        context_tensors = []
        for predecessor in components:
            _, adjacency = _incidence_graph(predecessor)
            edges = torch.tensor(
                adjacency,
                device=reference.device,
                dtype=reference.dtype,
            )
            context_tensors.append(
                self._ordered_pair_multiplex_tensor(
                    edges,
                    torch.zeros_like(edges),
                )
            )
        relation_rows: list[tuple[tuple[int, torch.Tensor], ...]] = []
        for candidate in components:
            alternatives = []
            for predecessor_index, predecessor in enumerate(components):
                if predecessor.output_type != candidate.input_type:
                    continue
                _, predecessor_adjacency, candidate_adjacency = (
                    _shared_incidence_graphs(predecessor, candidate)
                )
                alternatives.append(
                    (
                        predecessor_index,
                        self._ordered_pair_multiplex_tensor(
                            torch.tensor(
                                predecessor_adjacency,
                                device=reference.device,
                                dtype=reference.dtype,
                            ),
                            torch.tensor(
                                candidate_adjacency,
                                device=reference.device,
                                dtype=reference.dtype,
                            ),
                        ),
                    )
                )
            relation_rows.append(tuple(alternatives))
        return tuple(context_tensors), tuple(relation_rows)

    def local_relation_embeddings(
        self,
        task: PublicSoftwarePipelineTask,
        components: Sequence[PublicComponentContract],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode only public, package-local equality and effect relations."""

        reference = self.type_codes
        local_pairs = self.local_pair_projection(
            torch.tensor(
                _local_state_component_features(task.states, components),
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        relative_effects = self.relative_effect_projection(
            torch.tensor(
                _relative_effect_candidate_features(task.states, components),
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        stop_relations = self.stop_relation_projection(
            torch.tensor(
                _local_state_goal_features(task.states, task.required_output),
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        return local_pairs, relative_effects, stop_relations


class EvidenceOrderedPairEncoder(nn.Module):
    """Trainable evidence-only encoder over anonymous directed graph pairs.

    The module sees only five raw adjacency channels: predecessor edges and
    their reverse, candidate edges and their reverse, and the diagonal.  Four
    learned i-k-j residual rounds are permutation equivariant.  A controller
    owns independent instances for context and relation learning; a context
    call leaves both candidate channels exactly zero so it cannot depend on a
    candidate twin.
    """

    _RAW_FEATURES = 5
    _ROUNDS = 4

    def __init__(self, profile: SoftwarePipelineRunProfile) -> None:
        super().__init__()
        if not isinstance(profile, SoftwarePipelineRunProfile):
            raise TypeError("profile must be SoftwarePipelineRunProfile")
        self.width = profile.width
        topology_rank = min(
            profile.hidden_width,
            max(16, 2 * profile.transition_rank),
        )
        self.pair_input = nn.Sequential(
            nn.LayerNorm(self._RAW_FEATURES),
            nn.Linear(self._RAW_FEATURES, profile.width),
            nn.SiLU(),
        )
        self.pair_left = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(profile.width),
                nn.Linear(profile.width, topology_rank, bias=False),
                nn.SiLU(),
            )
            for _ in range(self._ROUNDS)
        )
        self.pair_right = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(profile.width),
                nn.Linear(profile.width, topology_rank, bias=False),
                nn.SiLU(),
            )
            for _ in range(self._ROUNDS)
        )
        self.pair_updates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * profile.width + topology_rank),
                nn.Linear(
                    2 * profile.width + topology_rank,
                    profile.hidden_width,
                ),
                nn.SiLU(),
                nn.Linear(profile.hidden_width, profile.width),
            )
            for _ in range(self._ROUNDS)
        )

    def forward(
        self,
        predecessor_adjacency: torch.Tensor,
        candidate_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        if (
            predecessor_adjacency.ndim != 2
            or predecessor_adjacency.shape[0] != predecessor_adjacency.shape[1]
            or candidate_adjacency.shape != predecessor_adjacency.shape
            or predecessor_adjacency.device != candidate_adjacency.device
            or predecessor_adjacency.dtype != candidate_adjacency.dtype
            or not predecessor_adjacency.is_floating_point()
            or predecessor_adjacency.shape[0] <= 0
            or not bool(torch.isfinite(predecessor_adjacency).all().item())
            or not bool(torch.isfinite(candidate_adjacency).all().item())
        ):
            raise ValueError(
                "evidence adjacency matrices must share one finite square float shape"
            )
        node_count = predecessor_adjacency.shape[0]
        diagonal = torch.eye(
            node_count,
            device=predecessor_adjacency.device,
            dtype=predecessor_adjacency.dtype,
        )
        pairs = self.pair_input(
            torch.stack(
                (
                    predecessor_adjacency,
                    predecessor_adjacency.transpose(0, 1),
                    candidate_adjacency,
                    candidate_adjacency.transpose(0, 1),
                    diagonal,
                ),
                dim=-1,
            )
        )
        scale = math.sqrt(node_count)
        for left_map, right_map, update in zip(
            self.pair_left,
            self.pair_right,
            self.pair_updates,
            strict=True,
        ):
            left = left_map(pairs)
            right = right_map(pairs)
            composed = torch.einsum("ikh,kjh->ijh", left, right) / scale
            pairs = pairs + update(
                torch.cat((pairs, pairs.transpose(0, 1), composed), dim=-1)
            )
        return pairs

    def context_tensors(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return candidate-independent predecessor tensors."""

        contexts = []
        for predecessor in components:
            _, adjacency = _incidence_graph(predecessor)
            predecessor_edges = torch.tensor(
                adjacency,
                device=reference.device,
                dtype=reference.dtype,
            )
            contexts.append(
                self(predecessor_edges, torch.zeros_like(predecessor_edges))
            )
        return tuple(contexts)

    def relation_tensors(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[tuple[tuple[int, torch.Tensor], ...], ...]:
        """Return directed predecessor-to-candidate relation tensors."""

        relation_rows: list[tuple[tuple[int, torch.Tensor], ...]] = []
        for candidate in components:
            alternatives = []
            for predecessor_index, predecessor in enumerate(components):
                if predecessor.output_type != candidate.input_type:
                    continue
                _, predecessor_adjacency, candidate_adjacency = (
                    _shared_incidence_graphs(predecessor, candidate)
                )
                alternatives.append(
                    (
                        predecessor_index,
                        self(
                            torch.tensor(
                                predecessor_adjacency,
                                device=reference.device,
                                dtype=reference.dtype,
                            ),
                            torch.tensor(
                                candidate_adjacency,
                                device=reference.device,
                                dtype=reference.dtype,
                            ),
                        ),
                    )
                )
            relation_rows.append(tuple(alternatives))
        return tuple(relation_rows)

    def context_relation_tensors(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[tuple[tuple[int, torch.Tensor], ...], ...],
    ]:
        """Return both tensor families for compatibility with focused probes."""

        return (
            self.context_tensors(components, reference),
            self.relation_tensors(components, reference),
        )


class RelationAxisSetReadout(nn.Module):
    """Learned invariant residual features over anonymous node incidence.

    A simultaneous node renaming permutes both graph axes.  Row and column
    summaries therefore move with their anonymous node.  Attention, mean, and
    maximum then pool that node set without discarding which incoming and
    outgoing cells shared an endpoint.
    """

    _AXIS_HEADS = 2
    _NODE_HEADS = 2

    def __init__(self, profile: SoftwarePipelineRunProfile) -> None:
        super().__init__()
        if not isinstance(profile, SoftwarePipelineRunProfile):
            raise TypeError("profile must be SoftwarePipelineRunProfile")
        width = profile.width
        self.width = width
        self.row_attention = nn.Linear(width, self._AXIS_HEADS, bias=False)
        self.column_attention = nn.Linear(width, self._AXIS_HEADS, bias=False)
        self.node_projection = nn.Sequential(
            nn.LayerNorm(5 * width),
            nn.Linear(5 * width, profile.hidden_width),
            nn.SiLU(),
            nn.Linear(profile.hidden_width, width),
        )
        self.node_pool_attention = nn.Linear(
            width,
            self._NODE_HEADS,
            bias=False,
        )

    def forward(self, pair_states: torch.Tensor) -> torch.Tensor:
        if (
            pair_states.ndim != 3
            or pair_states.shape[0] != pair_states.shape[1]
            or pair_states.shape[-1] != self.width
            or pair_states.shape[0] <= 0
            or not pair_states.is_floating_point()
            or not bool(torch.isfinite(pair_states).all().item())
        ):
            raise ValueError(
                "relation pair states must be finite [nodes, nodes, width]"
            )
        node_count = pair_states.shape[0]
        row_weights = torch.softmax(self.row_attention(pair_states), dim=1)
        row_summaries = torch.einsum(
            "ijh,ijw->ihw",
            row_weights,
            pair_states,
        )
        column_states = pair_states.transpose(0, 1)
        column_weights = torch.softmax(
            self.column_attention(column_states),
            dim=1,
        )
        column_summaries = torch.einsum(
            "ijh,ijw->ihw",
            column_weights,
            column_states,
        )
        diagonal_index = torch.arange(node_count, device=pair_states.device)
        diagonal = pair_states[diagonal_index, diagonal_index]
        node_tokens = self.node_projection(
            torch.cat(
                (
                    row_summaries.reshape(node_count, -1),
                    column_summaries.reshape(node_count, -1),
                    diagonal,
                ),
                dim=-1,
            )
        )
        node_weights = torch.softmax(
            self.node_pool_attention(node_tokens).transpose(0, 1),
            dim=-1,
        )
        attended_nodes = node_weights @ node_tokens
        return torch.cat(
            (
                attended_nodes.reshape(-1),
                node_tokens.mean(dim=0),
                node_tokens.amax(dim=0),
            ),
            dim=-1,
        )


class AnonymousRelationCell(nn.Module):
    """One independently plastic, identity-free relation procedure cell."""

    _POOL_HEADS = 2

    def __init__(self) -> None:
        super().__init__()
        self.width = _CLUSTER_CELL_WIDTH
        profile = SoftwarePipelineRunProfile(
            name="anonymous_relation_cell",
            width=_CLUSTER_CELL_WIDTH,
            hidden_width=_CLUSTER_CELL_HIDDEN_WIDTH,
            graph_layers=1,
            graph_heads=4,
            transition_rank=8,
            pointer_slots=2,
            role_slots=2,
            memory_read_top_k=1,
        )
        self.pair_encoder = EvidenceOrderedPairEncoder(profile)
        self.pool_attention = nn.Linear(
            self.width,
            self._POOL_HEADS,
            bias=False,
        )
        self.pool_projection = nn.Linear(4 * self.width, self.width)
        self.comparator = nn.Sequential(
            nn.Linear(3 * self.width, _CLUSTER_CELL_HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Linear(_CLUSTER_CELL_HIDDEN_WIDTH, 1, bias=False),
        )

    def pool(self, pair_states: torch.Tensor) -> torch.Tensor:
        if (
            pair_states.ndim != 3
            or pair_states.shape[0] != pair_states.shape[1]
            or pair_states.shape[-1] != self.width
            or pair_states.shape[0] <= 0
            or not pair_states.is_floating_point()
            or not bool(torch.isfinite(pair_states).all().item())
        ):
            raise ValueError("relation cell requires finite square pair states")
        cells = pair_states.reshape(-1, self.width)
        attention = torch.softmax(
            self.pool_attention(cells).transpose(0, 1),
            dim=-1,
        )
        attended = attention @ cells
        features = torch.cat(
            (
                attended.reshape(-1),
                cells.mean(dim=0),
                cells.amax(dim=0),
            ),
            dim=-1,
        )
        return F.normalize(
            self.pool_projection(features),
            dim=-1,
            eps=1.0e-8,
        )

    def relation_tensors(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[tuple[tuple[int, torch.Tensor], ...], ...]:
        return self.pair_encoder.relation_tensors(components, reference)

    def pair_logits(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            query_codes.ndim != 2
            or stored_codes.ndim != 2
            or query_codes.shape[1:] != stored_codes.shape[1:]
            or query_codes.shape[1] != self.width
            or query_codes.device != stored_codes.device
            or query_codes.dtype != stored_codes.dtype
            or not query_codes.is_floating_point()
            or not bool(torch.isfinite(query_codes).all().item())
            or not bool(torch.isfinite(stored_codes).all().item())
        ):
            raise ValueError("relation cell codes must be finite aligned matrices")
        query = query_codes[:, None, :]
        stored = stored_codes[None, :, :]
        features = torch.cat(
            (
                query * stored,
                (query - stored).abs(),
                (query + stored) * 0.5,
            ),
            dim=-1,
        )
        return torch.tanh(self.comparator(features).squeeze(-1))


class AnonymousAllActiveRelationComposer(nn.Module):
    """Fuse an anonymous cell set while retaining every cell in every read."""

    def __init__(
        self,
        *,
        cell_count: int = _CLUSTER_CELL_COUNT,
        cell_width: int = _CLUSTER_CELL_WIDTH,
        hidden_width: int = _CLUSTER_COMPOSER_HIDDEN_WIDTH,
        anchor_weight: float = _CLUSTER_COMPOSER_ANCHOR_WEIGHT,
    ) -> None:
        super().__init__()
        if (
            isinstance(cell_count, bool)
            or not isinstance(cell_count, int)
            or cell_count < 2
            or isinstance(cell_width, bool)
            or not isinstance(cell_width, int)
            or cell_width <= 0
            or isinstance(hidden_width, bool)
            or not isinstance(hidden_width, int)
            or hidden_width <= 0
            or not math.isfinite(anchor_weight)
            or not 0.0 < anchor_weight <= 1.0
        ):
            raise ValueError("relation composer dimensions or anchor are invalid")
        self.cell_count = cell_count
        self.cell_width = cell_width
        self.hidden_width = hidden_width
        self.anchor_weight = float(anchor_weight)
        feature_width = 2 * cell_width
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.residual_scorer = nn.Sequential(
            nn.Linear(2 * hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1, bias=False),
        )
        nn.init.zeros_(self.residual_scorer[-1].weight)

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
            raise ValueError("relation composer inputs must be finite and aligned")
        query = query_codes[:, None, :, :]
        stored = stored_codes[None, :, :, :]
        features = torch.cat(
            (query * stored, (query - stored).abs()),
            dim=-1,
        )
        local = self.local_encoder(features)
        shared = local.mean(dim=2, keepdim=True).expand_as(local)
        residual_logits = self.residual_scorer(
            torch.cat((local, shared), dim=-1)
        ).squeeze(-1)
        learned = torch.softmax(residual_logits, dim=-1)
        uniform = torch.full_like(learned, 1.0 / self.cell_count)
        weights = self.anchor_weight * uniform + (1.0 - self.anchor_weight) * learned
        fused = (weights * cell_logits).sum(dim=-1)
        return fused, weights, residual_logits, features


class SoftwarePipelineController(nn.Module):
    """Dynamic learned component controller with separated transfer lanes."""

    def __init__(self, profile: SoftwarePipelineRunProfile) -> None:
        super().__init__()
        if not isinstance(profile, SoftwarePipelineRunProfile):
            raise TypeError("profile must be SoftwarePipelineRunProfile")
        self.profile = profile
        width = profile.width
        hidden_width = profile.hidden_width
        self.role_encoder = RenameInvariantRoleEncoder(profile)
        self.pointer_features = FrozenHashTextEncoder(width)
        self.pointer_memory = GlyphAssociativeMemory(
            width,
            slots=profile.pointer_slots,
            read_top_k=profile.memory_read_top_k,
        )
        self.role_memory = GlyphAssociativeMemory(
            width,
            slots=profile.role_slots,
            read_top_k=profile.memory_read_top_k,
        )
        self.local_role_key_encoder = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.role_value_encoder = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.role_outcome_encoder = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.causal_transition = ConditionalReversibleTransition(
            width,
            rank=profile.transition_rank,
        )
        nn.init.normal_(
            self.causal_transition.first_up.weight,
            mean=0.0,
            std=1.0e-3 / math.sqrt(profile.transition_rank),
        )
        nn.init.normal_(
            self.causal_transition.second_up.weight,
            mean=0.0,
            std=1.0e-3 / math.sqrt(profile.transition_rank),
        )
        self.successor_query = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.pointer_to_role = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width, bias=False),
        )
        self.role_match_scale = nn.Parameter(torch.zeros(()))
        self.role_null_logit = nn.Parameter(torch.zeros(()))
        self.evidence_token = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.evidence_token, mean=0.0, std=1.0 / math.sqrt(width))
        self.evidence_action_head = nn.Sequential(
            nn.Linear(1, hidden_width),
            nn.Softplus(),
            nn.Linear(hidden_width, 1, bias=False),
        )
        # A constant-created positive gate consumes no RNG, preserving every
        # downstream initialization from the prior same-shaped head.  Its
        # small initial value keeps uncalibrated retrieval from dominating
        # the recurrent action path.
        self.evidence_action_log_gate = nn.Parameter(
            torch.tensor(math.log(math.expm1(_INITIAL_EVIDENCE_ACTION_GATE)))
        )
        reasoner_profile = GlyphMachineRunProfile(
            name=f"software_{profile.name}",
            width=width,
            hidden_width=hidden_width,
            hash_width=max(64, width // 2),
            graph_layers=profile.graph_layers,
            graph_heads=profile.graph_heads,
            transition_rank=profile.transition_rank,
            memory_slots=profile.role_slots,
            memory_heads=profile.graph_heads,
            memory_read_top_k=profile.memory_read_top_k,
        )
        self.backward_reasoner = GlyphBackwardProcedureReasoner(reasoner_profile)
        self.forward_action_head = nn.Sequential(
            nn.LayerNorm(5 * width),
            nn.Linear(5 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        self.stop_key_encoder = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        self.procedure_cell = nn.GRUCell(width, width)
        self.procedure_start = nn.Parameter(torch.zeros(width))

        # The evidence-only subsystem is deliberately constructed last.  It
        # therefore cannot perturb RNG-derived initialization of any audited
        # encoder, transition, reasoner, STOP, or ordinary action parameter.
        self.relation_pool_attention = nn.Linear(width, 2, bias=False)
        self.relation_pool_projection = nn.Linear(4 * width, width)
        self.relation_comparator = nn.Sequential(
            nn.Linear(3 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1, bias=False),
        )
        self.relation_context_pool_attention = nn.Linear(width, 2, bias=False)
        self.relation_context_pool_projection = nn.Linear(4 * width, width)
        self.relation_context_comparator = nn.Sequential(
            nn.Linear(3 * width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1, bias=False),
        )
        # Constructed after every shared path so fresh evidence geometry cannot
        # perturb the initialization of audited reasoning or action tensors.
        # The established relation tower remains first; context gets an
        # independent trunk so relevance credit cannot distort relation codes.
        self.evidence_pair_encoder = EvidenceOrderedPairEncoder(profile)
        self.evidence_context_encoder = copy.deepcopy(self.evidence_pair_encoder)
        # Appended only after every legacy evidence module has consumed its
        # original RNG sequence.  The zero projection makes this a function-
        # preserving learned expansion: step one opens the residual, and later
        # steps can train its anonymous incidence features.
        self.relation_incidence_readout = RelationAxisSetReadout(profile)
        self.relation_incidence_projection = nn.Linear(
            4 * width,
            width,
            bias=False,
        )
        nn.init.zeros_(self.relation_incidence_projection.weight)

    def initial_state(self, batch_size: int = 1) -> SoftwareReconstructionState:
        pointer = self.pointer_memory.initial_state(batch_size)
        role = self.role_memory.initial_state(batch_size)
        return SoftwareReconstructionState(
            pointer=pointer,
            role=role,
            context_trace_keys=torch.zeros_like(role.keys),
            relation_trace_values=torch.zeros_like(role.keys),
        )

    def _pool_relation_tensor(self, pair_states: torch.Tensor) -> torch.Tensor:
        """Fuse the proven global code with a learned incidence residual."""

        incidence_features = self.relation_incidence_readout(pair_states)
        cells = pair_states.reshape(-1, self.profile.width)
        attention = torch.softmax(
            self.relation_pool_attention(cells).transpose(0, 1),
            dim=-1,
        )
        attended = attention @ cells
        global_features = torch.cat(
            (
                attended.reshape(-1),
                cells.mean(dim=0),
                cells.amax(dim=0),
            ),
            dim=-1,
        )
        precode = self.relation_pool_projection(global_features)
        precode = precode + self.relation_incidence_projection(
            incidence_features
        )
        return F.normalize(
            precode,
            dim=-1,
            eps=1.0e-8,
        )

    def _pool_context_tensor(self, pair_states: torch.Tensor) -> torch.Tensor:
        """Pool a trainable candidate-independent predecessor tensor."""

        if (
            pair_states.ndim != 3
            or pair_states.shape[0] != pair_states.shape[1]
            or pair_states.shape[-1] != self.profile.width
            or not pair_states.is_floating_point()
            or not bool(torch.isfinite(pair_states).all().item())
        ):
            raise ValueError("context pair states must be finite [nodes, nodes, width]")
        cells = pair_states.reshape(-1, self.profile.width)
        attention = torch.softmax(
            self.relation_context_pool_attention(cells).transpose(0, 1),
            dim=-1,
        )
        attended = attention @ cells
        pooled = torch.cat(
            (
                attended.reshape(-1),
                cells.mean(dim=0),
                cells.amax(dim=0),
            ),
            dim=-1,
        )
        return F.normalize(
            self.relation_context_pool_projection(pooled),
            dim=-1,
            eps=1.0e-8,
        )

    def _context_pair_logits(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded symmetric context matches before slot normalization."""

        if (
            query_codes.ndim != 2
            or stored_codes.ndim != 2
            or query_codes.shape[1:] != stored_codes.shape[1:]
            or query_codes.shape[1] != self.profile.width
            or query_codes.device != stored_codes.device
            or query_codes.dtype != stored_codes.dtype
            or not query_codes.is_floating_point()
            or not bool(torch.isfinite(query_codes).all().item())
            or not bool(torch.isfinite(stored_codes).all().item())
        ):
            raise ValueError("context match codes must be finite aligned matrices")
        query = query_codes[:, None, :]
        stored = stored_codes[None, :, :]
        features = torch.cat(
            (
                query * stored,
                (query - stored).abs(),
                (query + stored) * 0.5,
            ),
            dim=-1,
        )
        return torch.tanh(self.relation_context_comparator(features).squeeze(-1))

    def _relation_pair_logits(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded symmetric directed-relation matches per slot."""

        if (
            query_codes.ndim != 2
            or stored_codes.ndim != 2
            or query_codes.shape[1:] != stored_codes.shape[1:]
            or query_codes.shape[1] != self.profile.width
            or query_codes.device != stored_codes.device
            or query_codes.dtype != stored_codes.dtype
            or not query_codes.is_floating_point()
            or not bool(torch.isfinite(query_codes).all().item())
            or not bool(torch.isfinite(stored_codes).all().item())
        ):
            raise ValueError("relation match codes must be finite aligned matrices")
        query = query_codes[:, None, :]
        stored = stored_codes[None, :, :]
        features = torch.cat(
            (
                query * stored,
                (query - stored).abs(),
                (query + stored) * 0.5,
            ),
            dim=-1,
        )
        return torch.tanh(self.relation_comparator(features).squeeze(-1))

    def _relation_evidence_read(
        self,
        query_context_codes: torch.Tensor,
        query_relation_codes: torch.Tensor,
        stored_contexts: torch.Tensor,
        stored_relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the one runtime retrieval computation to explicit real slots."""

        if (
            query_context_codes.shape != query_relation_codes.shape
            or stored_contexts.shape != stored_relations.shape
            or stored_contexts.ndim != 2
            or stored_contexts.shape[0] <= 0
            or query_context_codes.ndim != 2
            or query_context_codes.shape[1] != self.profile.width
            or stored_contexts.shape[1] != self.profile.width
            or query_context_codes.device != query_relation_codes.device
            or query_context_codes.device != stored_contexts.device
            or query_context_codes.device != stored_relations.device
            or query_context_codes.dtype != query_relation_codes.dtype
            or query_context_codes.dtype != stored_contexts.dtype
            or query_context_codes.dtype != stored_relations.dtype
            or not query_context_codes.is_floating_point()
            or not bool(torch.isfinite(query_context_codes).all().item())
            or not bool(torch.isfinite(query_relation_codes).all().item())
            or not bool(torch.isfinite(stored_contexts).all().item())
            or not bool(torch.isfinite(stored_relations).all().item())
        ):
            raise ValueError("relation evidence read requires aligned nonempty slots")
        context_logits = self._context_pair_logits(
            query_context_codes,
            stored_contexts,
        )
        null_column = context_logits.new_zeros(query_context_codes.shape[0], 1)
        all_context_weights = torch.softmax(
            torch.cat(
                (
                    context_logits / _RELATION_CONTEXT_AUX_TEMPERATURE,
                    null_column,
                ),
                dim=-1,
            ),
            dim=-1,
        )
        context_weights = all_context_weights[:, : stored_contexts.shape[0]]
        context_null_weights = all_context_weights[:, stored_contexts.shape[0]]
        relation_logits = self._relation_pair_logits(
            query_relation_codes,
            stored_relations,
        )
        scores = (context_weights * relation_logits).sum(dim=-1)
        return scores, context_weights, context_null_weights, relation_logits

    def _factorized_relation_embeddings(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one predecessor-context key and directed value per component."""

        context_tensors = self.evidence_context_encoder.context_tensors(
            components,
            reference,
        )
        relation_rows = self.evidence_pair_encoder.relation_tensors(
            components,
            reference,
        )
        predecessor_codes = tuple(
            self._pool_context_tensor(value) for value in context_tensors
        )
        context_codes = []
        relation_codes = []
        for alternatives in relation_rows:
            if not alternatives:
                context_codes.append(reference.new_zeros(self.profile.width))
                relation_codes.append(reference.new_zeros(self.profile.width))
                continue
            if len(alternatives) != 1:
                raise ValueError(
                    "factorized evidence requires one public predecessor per component"
                )
            context_codes.append(
                F.normalize(
                    torch.stack(
                        tuple(
                            predecessor_codes[predecessor_index]
                            for predecessor_index, _ in alternatives
                        )
                    ).mean(dim=0),
                    dim=-1,
                    eps=1.0e-8,
                )
            )
            relation_codes.append(
                F.normalize(
                    torch.stack(
                        tuple(
                            self._pool_relation_tensor(value)
                            for _, value in alternatives
                        )
                    ).mean(dim=0),
                    dim=-1,
                    eps=1.0e-8,
                )
            )
        return torch.stack(context_codes), torch.stack(relation_codes)

    def _relation_evidence_scores(
        self,
        query_context_codes: torch.Tensor,
        query_relation_codes: torch.Tensor,
        state: SoftwareReconstructionState,
    ) -> torch.Tensor:
        """Select public contexts before comparing directed relation values."""

        _validate_controller_state(self, state)
        for name, value, stored in (
            ("context", query_context_codes, state.context_trace_keys),
            ("relation", query_relation_codes, state.relation_trace_values),
        ):
            if (
                value.ndim != 2
                or value.shape[1] != self.profile.width
                or value.device != stored.device
                or value.dtype != stored.dtype
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(
                    f"{name} queries must be finite [components, width]"
                )
        if query_context_codes.shape != query_relation_codes.shape:
            raise ValueError("context and relation queries must align")
        trace_slots = self.role_memory.trace_slot_count
        occupied = state.role.occupied[0, :trace_slots]
        stored_contexts = state.context_trace_keys[0, :trace_slots]
        stored_relations = state.relation_trace_values[0, :trace_slots]
        occupied = (
            occupied
            & (stored_contexts.norm(dim=-1) > 1.0e-8)
            & (stored_relations.norm(dim=-1) > 1.0e-8)
        )
        query_present = (
            (query_context_codes.norm(dim=-1) > 1.0e-8)
            & (query_relation_codes.norm(dim=-1) > 1.0e-8)
        )
        if not bool(occupied.any().item()) or not bool(query_present.any().item()):
            return query_relation_codes.new_zeros(query_relation_codes.shape[0])
        context_keys = stored_contexts[occupied]
        relation_values = stored_relations[occupied]
        scores, _, _, _ = self._relation_evidence_read(
            query_context_codes,
            query_relation_codes,
            context_keys,
            relation_values,
        )
        return torch.where(query_present, scores, torch.zeros_like(scores))

    def encode_task(self, task: PublicSoftwarePipelineTask) -> SoftwareTaskEncoding:
        _validate_public_task(task)
        role_states, role_components, operator_embeddings = self.role_encoder(task)
        reference = role_states
        components = _components_in_candidate_order(task)
        local_pairs, relative_effects, stop_relations = (
            self.role_encoder.local_relation_embeddings(task, components)
        )
        pointer_states = self.pointer_features.encode_texts(
            [_pointer_state_text(value) for value in task.states],
            device=reference.device,
            dtype=reference.dtype,
        ) * math.sqrt(self.profile.width)
        pointer_components = self.pointer_features.encode_texts(
            [
                _pointer_component_text(component, candidate)
                for component, candidate in zip(
                    components,
                    task.grounded_candidates,
                    strict=True,
                )
            ],
            device=reference.device,
            dtype=reference.dtype,
        ) * math.sqrt(self.profile.width)
        pointer_pair_ids = torch.tensor(
            [
                _pointer_words(_pointer_pair_text(state, component, candidate))
                for state in task.states
                for component, candidate in zip(
                    components,
                    task.grounded_candidates,
                    strict=True,
                )
            ],
            device=reference.device,
            dtype=torch.long,
        ).reshape(len(task.states), len(task.grounded_candidates), _POINTER_WORDS)
        pointer_successor_ids = torch.tensor(
            [_pointer_words(_pointer_state_text(state)) for state in task.states],
            device=reference.device,
            dtype=torch.long,
        )
        role_pair_keys = self._factorized_role_keys(
            local_pairs,
            operator_embeddings,
        )
        relation_contexts, relation_components = (
            self._factorized_relation_embeddings(
                components,
                reference,
            )
        )
        origin_index = _state_index(task.states, task.origin)
        goal_index = _goal_state_index(task.states, task.required_output)
        return SoftwareTaskEncoding(
            pointer_state_embeddings=pointer_states,
            pointer_component_embeddings=pointer_components,
            pointer_pair_ids=pointer_pair_ids,
            pointer_successor_ids=pointer_successor_ids,
            role_state_embeddings=role_states,
            role_component_embeddings=role_components,
            operator_embeddings=operator_embeddings,
            local_pair_embeddings=local_pairs,
            relative_effect_embeddings=relative_effects,
            stop_relation_embeddings=stop_relations,
            role_pair_keys=role_pair_keys,
            relation_context_embeddings=relation_contexts,
            relation_component_embeddings=relation_components,
            origin_index=origin_index,
            goal_index=goal_index,
        )

    def _factorized_role_keys(
        self,
        local_pairs: torch.Tensor,
        operator_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve an operator anchor while learning a bounded local residual."""

        if (
            local_pairs.ndim != 3
            or local_pairs.shape[-1] != self.profile.width
            or operator_embeddings.shape
            != (local_pairs.shape[1], self.profile.width)
            or operator_embeddings.device != local_pairs.device
            or operator_embeddings.dtype != local_pairs.dtype
        ):
            raise ValueError("factorized role inputs have incompatible shapes")
        local = F.normalize(
            self.local_role_key_encoder(local_pairs),
            dim=-1,
            eps=1.0e-8,
        )
        centered_operator = F.layer_norm(
            operator_embeddings,
            (self.profile.width,),
        )
        operator = F.normalize(centered_operator, dim=-1, eps=1.0e-8)
        operator_rows = operator.unsqueeze(0).expand_as(local)
        anchored = F.normalize(
            _bounded_role_anchor(operator_rows, local),
            dim=-1,
            eps=1.0e-8,
        )
        has_operator = operator_embeddings.norm(dim=-1) > 1.0e-8
        return torch.where(has_operator.reshape(1, -1, 1), anchored, local)

    def _dense_role_trace_read(
        self,
        query_keys: torch.Tensor,
        state: GlyphAssociativeState,
    ) -> Phase6DenseRoleRead:
        """Read every occupied trace slot plus one learned null alternative."""

        if query_keys.ndim == 1:
            query_keys = query_keys.unsqueeze(0)
        if (
            query_keys.ndim != 2
            or query_keys.shape[0] <= 0
            or query_keys.shape[1] != self.profile.width
            or query_keys.device != state.keys.device
            or query_keys.dtype != state.keys.dtype
            or state.batch_size != 1
            or not bool(torch.isfinite(query_keys).all().item())
        ):
            raise ValueError("dense role queries must be finite [count, width]")
        trace_slots = self.role_memory.trace_slot_count
        occupied = state.occupied[0, :trace_slots]
        count = query_keys.shape[0]
        if not bool(occupied.any().item()):
            return Phase6DenseRoleRead(
                contexts=query_keys.new_zeros((count, self.profile.width)),
                attention_weights=query_keys.new_zeros((count, trace_slots)),
                null_weights=query_keys.new_ones((count,)),
                evidence_probabilities=query_keys.new_zeros((count,)),
                evidence_logits=query_keys.new_zeros((count,)),
            )
        normalized_queries = F.normalize(query_keys, dim=-1, eps=1.0e-8)
        normalized_keys = F.normalize(
            state.keys[0, :trace_slots], dim=-1, eps=1.0e-8
        )
        scale = math.sqrt(self.profile.width) * (
            0.5 + torch.sigmoid(self.role_match_scale)
        )
        slot_logits = normalized_queries @ normalized_keys.transpose(0, 1) * scale
        occupied_count = occupied.sum().to(dtype=slot_logits.dtype)
        slot_logits = slot_logits - occupied_count.log()
        slot_logits = slot_logits.masked_fill(~occupied.unsqueeze(0), -torch.inf)
        null_logits = self.role_null_logit.expand(count, 1)
        combined = torch.cat((slot_logits, null_logits), dim=-1)
        combined_weights = torch.softmax(combined, dim=-1)
        slot_weights = combined_weights[:, :trace_slots]
        null_weights = combined_weights[:, trace_slots]
        contexts = slot_weights @ state.values[0, :trace_slots]
        evidence_probabilities = 1.0 - null_weights
        evidence_logits = torch.logsumexp(slot_logits, dim=-1) - self.role_null_logit
        if not bool(torch.isfinite(contexts).all().item()) or not bool(
            torch.isfinite(evidence_logits).all().item()
        ):
            raise RuntimeError("dense role retrieval produced non-finite values")
        return Phase6DenseRoleRead(
            contexts=contexts,
            attention_weights=slot_weights,
            null_weights=null_weights,
            evidence_probabilities=evidence_probabilities,
            evidence_logits=evidence_logits,
        )

    def trace_role_value(
        self,
        local_pair: torch.Tensor,
        component: torch.Tensor,
        relative_effect: torch.Tensor,
    ) -> torch.Tensor:
        learned = self.role_value_encoder(
            torch.cat((local_pair, component, relative_effect), dim=-1)
        )
        return _bounded_role_anchor(relative_effect, learned)

    def _evidence_action_contribution(
        self,
        evidence_match_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Map stronger retrieved evidence to a strictly larger action term."""

        if not evidence_match_scores.is_floating_point() or not bool(
            torch.isfinite(evidence_match_scores).all().item()
        ):
            raise ValueError("evidence match scores must be finite floating values")
        input_layer = self.evidence_action_head[0]
        activation = self.evidence_action_head[1]
        output_layer = self.evidence_action_head[2]
        epsilon = torch.finfo(evidence_match_scores.dtype).eps
        input_weights = F.softplus(input_layer.weight) + epsilon
        input_weights = input_weights / input_weights.mean().clamp_min(epsilon)
        output_weights = F.softplus(output_layer.weight) + epsilon
        output_weights = output_weights / output_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(epsilon)
        activated = activation(
            F.linear(
                evidence_match_scores.unsqueeze(-1),
                input_weights,
                input_layer.bias,
            )
        )
        zero_activated = activation(input_layer.bias)
        centered = activated - zero_activated
        gate = F.softplus(self.evidence_action_log_gate) + epsilon
        return gate * 2.0 * F.linear(centered, output_weights).squeeze(-1)

    def _public_retrieval_contrast_losses(
        self,
        evidence_match_scores: torch.Tensor,
        target_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return public selected-action CE and one fixed hardest-negative margin."""

        if (
            evidence_match_scores.ndim != 1
            or evidence_match_scores.numel() < 2
            or not evidence_match_scores.is_floating_point()
            or not bool(torch.isfinite(evidence_match_scores).all().item())
        ):
            raise ValueError("retrieval contrast needs finite component scores")
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or not 0 <= target_index < evidence_match_scores.numel()
        ):
            raise ValueError("retrieval target index is invalid")
        target = torch.tensor(
            (target_index,),
            device=evidence_match_scores.device,
            dtype=torch.long,
        )
        classification = F.cross_entropy(
            evidence_match_scores.unsqueeze(0),
            target,
        )
        negative_mask = torch.ones_like(evidence_match_scores, dtype=torch.bool)
        negative_mask[target_index] = False
        hardest_negative = evidence_match_scores[negative_mask].max()
        margin = F.relu(
            evidence_match_scores.new_tensor(_PUBLIC_RETRIEVAL_MARGIN)
            - evidence_match_scores[target_index]
            + hardest_negative
        )
        return classification, margin

    def _public_role_memory_causal_hinge(
        self,
        task: PublicSoftwarePipelineTask,
        role_on_action_logits: torch.Tensor,
        role_off_action_logits: torch.Tensor,
        target_index: int,
    ) -> torch.Tensor | None:
        """Require role memory to improve the target within its public twin class."""

        components = _components_in_candidate_order(task)
        if (
            role_on_action_logits.shape != (len(components),)
            or role_off_action_logits.shape != role_on_action_logits.shape
            or not bool(torch.isfinite(role_on_action_logits).all().item())
            or not bool(torch.isfinite(role_off_action_logits).all().item())
        ):
            raise ValueError("causal role logits must cover finite declared actions")
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or not 0 <= target_index < len(components)
        ):
            raise ValueError("causal role target index is invalid")
        target_contract = components[target_index]
        target_effect = _public_effect_equivalence_key(target_contract)
        class_indices = tuple(
            index
            for index, component in enumerate(components)
            if _public_effect_equivalence_key(component) == target_effect
        )
        if len(class_indices) < 2:
            return None
        class_tensor = torch.tensor(
            class_indices,
            device=role_on_action_logits.device,
            dtype=torch.long,
        )
        local_target = class_indices.index(target_index)
        role_on_log_probability = F.log_softmax(
            role_on_action_logits.index_select(0, class_tensor),
            dim=0,
        )[local_target]
        role_off_log_probability = F.log_softmax(
            role_off_action_logits.detach().index_select(0, class_tensor),
            dim=0,
        )[local_target]
        improvement = role_on_log_probability - role_off_log_probability
        return F.relu(
            improvement.new_tensor(_PUBLIC_ROLE_CAUSAL_MARGIN_NATS)
            - improvement
        )

    def transition_lattice(
        self,
        encoding: SoftwareTaskEncoding,
        state: SoftwareReconstructionState,
        *,
        include_pointer_memory: bool = True,
        include_role_memory: bool = True,
    ) -> GlyphTransitionLattice:
        _validate_controller_state(self, state)
        if type(include_pointer_memory) is not bool or type(include_role_memory) is not bool:
            raise TypeError("memory ablation flags must be bool")
        state_count, component_count, width = encoding.role_pair_keys.shape
        flat_role_keys = encoding.role_pair_keys.reshape(-1, width)
        role_read = self._dense_role_trace_read(flat_role_keys, state.role)
        outcome_read = self.role_memory.read(flat_role_keys, state.role, lane="outcome")
        role_contexts = role_read.contexts.reshape(state_count, component_count, width)
        evidence_probabilities = role_read.evidence_probabilities.reshape(
            state_count, component_count
        )
        outcome_contexts = outcome_read.contexts[0].reshape(
            state_count, component_count, width
        )
        pointer_contexts = _read_exact_pointer_contexts(
            encoding.pointer_pair_ids.reshape(-1, _POINTER_WORDS),
            state.pointer,
            self.pointer_memory.trace_slot_count,
        ).reshape(state_count, component_count, width)
        if not include_pointer_memory:
            pointer_contexts = torch.zeros_like(pointer_contexts)
        if not include_role_memory:
            role_contexts = torch.zeros_like(role_contexts)
            outcome_contexts = torch.zeros_like(outcome_contexts)
            evidence_probabilities = torch.zeros_like(evidence_probabilities)

        role_states = encoding.role_state_embeddings
        role_components = encoding.operator_embeddings
        local_pair_grid = encoding.local_pair_embeddings
        component_grid = role_components[None, :, :].expand_as(role_contexts)
        pointer_roles = self.pointer_to_role(pointer_contexts.reshape(-1, width)).reshape_as(
            pointer_contexts
        )
        evidence_codes = evidence_probabilities.unsqueeze(-1) * self.evidence_token
        combined_trace = role_contexts + pointer_roles + evidence_codes
        raw_effects = self.causal_transition(
            local_pair_grid.reshape(-1, width),
            torch.cat(
                (
                    component_grid.reshape(-1, width),
                    combined_trace.reshape(-1, width),
                ),
                dim=-1,
            ),
        ).reshape(state_count, component_count, width)
        queried_effects = self.successor_query(raw_effects.reshape(-1, width)).reshape(
            state_count, component_count, width
        )
        causal_logits = torch.einsum(
            "saw,satw->sat",
            queried_effects,
            encoding.relative_effect_embeddings,
        ) / math.sqrt(width)
        role_recall_logits = torch.einsum(
            "saw,satw->sat",
            role_contexts,
            encoding.relative_effect_embeddings,
        ) / math.sqrt(width)
        pointer_recall_logits = (
            pointer_contexts.reshape(-1, width)
            @ encoding.pointer_state_embeddings.transpose(0, 1)
        ).reshape(state_count, component_count, state_count) / math.sqrt(width)
        successor_logits = causal_logits + role_recall_logits + pointer_recall_logits
        successor_probabilities = torch.softmax(successor_logits, dim=-1)
        predicted_successors = torch.einsum(
            "sat,tw->saw", successor_probabilities, role_states
        )
        return GlyphTransitionLattice(
            successor_state_logits=successor_logits,
            successor_probabilities=successor_probabilities,
            associative_recall_logits=role_recall_logits + pointer_recall_logits,
            predicted_successors=predicted_successors,
            raw_reversible_successors=raw_effects,
            trace_contexts=combined_trace,
            outcome_contexts=outcome_contexts,
        )

    def score_actions(
        self,
        task: PublicSoftwarePipelineTask,
        state: SoftwareReconstructionState,
        *,
        current_state_belief: torch.Tensor | None = None,
        steps_remaining: int | None = None,
        encoding: SoftwareTaskEncoding | None = None,
        include_pointer_memory: bool = True,
        include_role_memory: bool = True,
        include_backward_reasoning: bool = True,
        detach_evidence_action_input: bool = False,
        use_legacy_evidence: bool = False,
    ) -> SoftwareStepScores:
        _validate_controller_state(self, state)
        if type(detach_evidence_action_input) is not bool:
            raise TypeError("detach_evidence_action_input must be bool")
        if type(use_legacy_evidence) is not bool:
            raise TypeError("use_legacy_evidence must be bool")
        encoded = self.encode_task(task) if encoding is None else encoding
        state_count = len(task.states)
        if current_state_belief is None:
            current_state_belief = F.one_hot(
                torch.tensor(
                    encoded.origin_index,
                    device=encoded.role_state_embeddings.device,
                ),
                state_count,
            ).to(dtype=encoded.role_state_embeddings.dtype)
        _validate_state_belief(current_state_belief, state_count)
        if steps_remaining is None:
            steps_remaining = task.max_steps
        if (
            isinstance(steps_remaining, bool)
            or not isinstance(steps_remaining, int)
            or not 1 <= steps_remaining <= _MAX_STEPS
        ):
            raise ValueError("steps_remaining must be one through four")
        lattice = self.transition_lattice(
            encoded,
            state,
            include_pointer_memory=include_pointer_memory,
            include_role_memory=include_role_memory,
        )
        component_probabilities = torch.einsum(
            "s,sat->at", current_state_belief, lattice.successor_probabilities
        )
        dense_role_read = self._dense_role_trace_read(
            encoded.role_pair_keys.reshape(-1, self.profile.width),
            state.role,
        )
        if use_legacy_evidence:
            evidence_lattice = dense_role_read.evidence_logits.reshape(
                state_count, len(task.grounded_candidates)
            )
            evidence_match_scores = torch.einsum(
                "s,sa->a", current_state_belief, evidence_lattice
            )
        else:
            evidence_match_scores = self._relation_evidence_scores(
                encoded.relation_context_embeddings,
                encoded.relation_component_embeddings,
                state,
            )
        if not include_role_memory:
            evidence_match_scores = torch.zeros_like(evidence_match_scores)
        successor_logits = torch.log(
            component_probabilities.clamp_min(torch.finfo(component_probabilities.dtype).tiny)
        )
        if include_backward_reasoning:
            action_logits, node_codes = self.backward_reasoner(
                encoded.role_state_embeddings,
                encoded.goal_index,
                lattice,
                current_state_belief,
                steps_remaining=steps_remaining,
            )
        else:
            current = current_state_belief @ encoded.role_state_embeddings
            goal = encoded.role_state_embeddings[encoded.goal_index]
            successors = component_probabilities @ encoded.role_state_embeddings
            contexts = torch.einsum(
                "s,saw->aw", current_state_belief, lattice.trace_contexts
            )
            count = len(task.grounded_candidates)
            action_logits = self.forward_action_head(
                torch.cat(
                    (
                        current.expand(count, -1),
                        goal.expand(count, -1),
                        encoded.role_component_embeddings,
                        successors,
                        contexts,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            node_codes = torch.zeros_like(encoded.role_state_embeddings)
        evidence_action_input = (
            evidence_match_scores.detach()
            if detach_evidence_action_input
            else evidence_match_scores
        )
        action_logits = action_logits + self._evidence_action_contribution(
            evidence_action_input
        )
        stop_relation = current_state_belief @ encoded.stop_relation_embeddings
        stop_key = F.normalize(
            self.stop_key_encoder(stop_relation),
            dim=-1,
            eps=1.0e-8,
        )
        stop_context = self.role_memory.read(
            stop_key.reshape(1, -1), state.role, lane="outcome"
        ).contexts[0, 0]
        if not include_role_memory:
            stop_context = torch.zeros_like(stop_context)
        stop_features = torch.cat(
            (
                stop_relation,
                stop_context,
                stop_relation * stop_context,
                stop_context - stop_relation,
            ),
            dim=-1,
        )
        stop_logit = self.stop_head(stop_features).reshape(())
        logits = torch.cat((action_logits, stop_logit.unsqueeze(0)), dim=0)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("software-pipeline controller produced non-finite logits")
        pointer_contexts = _read_exact_pointer_contexts(
            encoded.pointer_pair_ids.reshape(-1, _POINTER_WORDS),
            state.pointer,
            self.pointer_memory.trace_slot_count,
        ).reshape(len(task.states), len(task.grounded_candidates), self.profile.width)
        role_contexts = self._dense_role_trace_read(
            encoded.role_pair_keys.reshape(-1, self.profile.width),
            state.role,
        ).contexts.reshape_as(pointer_contexts)
        outcome_contexts = self.role_memory.read(
            encoded.role_pair_keys.reshape(-1, self.profile.width),
            state.role,
            lane="outcome",
        ).contexts[0].reshape_as(pointer_contexts)
        if not include_pointer_memory:
            pointer_contexts = torch.zeros_like(pointer_contexts)
        if not include_role_memory:
            role_contexts = torch.zeros_like(role_contexts)
            outcome_contexts = torch.zeros_like(outcome_contexts)
        return SoftwareStepScores(
            logits=logits,
            action_logits=action_logits,
            stop_logit=stop_logit,
            successor_state_logits=successor_logits,
            pointer_contexts=pointer_contexts,
            role_contexts=role_contexts,
            outcome_contexts=outcome_contexts,
            evidence_match_scores=evidence_match_scores,
            reasoning_node_codes=node_codes,
            current_state_belief=current_state_belief,
        )

    def summarize_pipeline(
        self,
        task: PublicSoftwarePipelineTask,
        pipeline: CommittedSoftwarePipeline,
        *,
        encoding: SoftwareTaskEncoding | None = None,
    ) -> torch.Tensor:
        encoded = self.encode_task(task) if encoding is None else encoding
        code = self.procedure_start
        for action in pipeline.actions:
            index = _action_index(task.grounded_candidates, action)
            code = self.procedure_cell(encoded.role_component_embeddings[index], code)
        return code

    def public_trace_losses(
        self,
        task: PublicSoftwarePipelineTask,
        state: SoftwareReconstructionState,
    ) -> torch.Tensor:
        transitions = _public_transitions(task)
        if not transitions:
            raise ValueError("public trace loss requires visible transitions")
        encoded = self.encode_task(task)
        lattice = self.transition_lattice(encoded, state)
        losses = []
        for transition in transitions:
            before_index = _state_index(task.states, transition.before)
            after_index = _state_index(task.states, transition.after)
            component_index = _action_index(task.grounded_candidates, transition.action)
            logits = lattice.successor_state_logits[
                before_index, component_index
            ].unsqueeze(0)
            target = torch.tensor(
                (after_index,), device=logits.device, dtype=torch.long
            )
            classification = F.cross_entropy(logits, target)
            alignment = 1.0 - F.cosine_similarity(
                lattice.raw_reversible_successors[
                    before_index, component_index
                ].unsqueeze(0),
                encoded.relative_effect_embeddings[
                    before_index, component_index, after_index
                ].unsqueeze(0),
                dim=-1,
                eps=1.0e-8,
            ).mean()
            losses.append(classification + 0.25 * alignment)
        return torch.stack(losses)

    def public_heldout_production_losses(
        self,
        observed_task: PublicSoftwarePipelineTask,
        state: SoftwareReconstructionState,
        *,
        include_role_memory_causal_hinge: bool = False,
        detach_evidence_action_input: bool = False,
        use_legacy_evidence: bool = False,
    ) -> torch.Tensor:
        """Supervise a held-out public package through the production path.

        The trace supplies labels only.  Encoding, memory reads, transition
        prediction, action scoring, and STOP scoring all receive a task whose
        observations have been removed.  Exact pointers are disabled, so the
        only usable evidence came from sibling packages already in ``state``.
        """

        if type(include_role_memory_causal_hinge) is not bool:
            raise TypeError("include_role_memory_causal_hinge must be bool")
        if type(detach_evidence_action_input) is not bool:
            raise TypeError("detach_evidence_action_input must be bool")
        if type(use_legacy_evidence) is not bool:
            raise TypeError("use_legacy_evidence must be bool")
        transitions = _public_transitions(observed_task)
        if not transitions:
            raise ValueError("held-out production loss requires a public trace")
        masked = replace(observed_task, observations=())
        encoded = self.encode_task(masked)
        lattice = self.transition_lattice(
            encoded,
            state,
            include_pointer_memory=False,
        )
        losses = []
        state_count = len(masked.states)
        ordered_components = _components_in_candidate_order(masked)
        for position, transition in enumerate(transitions):
            before_index = _state_index(masked.states, transition.before)
            after_index = _state_index(masked.states, transition.after)
            component_index = _action_index(
                masked.grounded_candidates, transition.action
            )
            belief = F.one_hot(
                torch.tensor(
                    before_index,
                    device=encoded.role_state_embeddings.device,
                ),
                state_count,
            ).to(dtype=encoded.role_state_embeddings.dtype)
            scores = self.score_actions(
                masked,
                state,
                current_state_belief=belief,
                steps_remaining=len(transitions) - position,
                encoding=encoded,
                include_pointer_memory=False,
                detach_evidence_action_input=detach_evidence_action_input,
                use_legacy_evidence=use_legacy_evidence,
            )
            successor_target = torch.tensor(
                (after_index,), device=scores.logits.device, dtype=torch.long
            )
            successor_loss = F.cross_entropy(
                scores.successor_state_logits[component_index].unsqueeze(0),
                successor_target,
            )
            relative_alignment = 1.0 - F.cosine_similarity(
                lattice.raw_reversible_successors[
                    before_index, component_index
                ].unsqueeze(0),
                encoded.relative_effect_embeddings[
                    before_index, component_index, after_index
                ].unsqueeze(0),
                dim=-1,
                eps=1.0e-8,
            ).mean()
            losses.append(successor_loss + 0.25 * relative_alignment)
            action_target = torch.tensor(
                (component_index,), device=scores.logits.device, dtype=torch.long
            )
            losses.append(
                F.cross_entropy(scores.logits.unsqueeze(0), action_target)
            )
            retrieval_ce, retrieval_margin = (
                self._public_retrieval_contrast_losses(
                    scores.evidence_match_scores,
                    component_index,
                )
            )
            losses.append(retrieval_ce)
            losses.append(
                _PUBLIC_RETRIEVAL_MARGIN_WEIGHT * retrieval_margin
            )
            if include_role_memory_causal_hinge:
                target_effect = _public_effect_equivalence_key(
                    ordered_components[component_index]
                )
                effect_class_size = sum(
                    _public_effect_equivalence_key(component) == target_effect
                    for component in ordered_components
                )
            else:
                effect_class_size = 0
            if effect_class_size >= 2:
                role_off_scores = self.score_actions(
                    masked,
                    state,
                    current_state_belief=belief,
                    steps_remaining=len(transitions) - position,
                    encoding=encoded,
                    include_pointer_memory=False,
                    include_role_memory=False,
                    detach_evidence_action_input=detach_evidence_action_input,
                    use_legacy_evidence=use_legacy_evidence,
                )
                causal_hinge = self._public_role_memory_causal_hinge(
                    masked,
                    scores.action_logits,
                    role_off_scores.action_logits,
                    component_index,
                )
                if causal_hinge is None:
                    raise RuntimeError("public twin class lost its causal hinge")
                losses.append(
                    _PUBLIC_ROLE_CAUSAL_MARGIN_WEIGHT * causal_hinge
                )
        endpoint_index = _state_index(masked.states, transitions[-1].after)
        endpoint_belief = F.one_hot(
            torch.tensor(
                endpoint_index,
                device=encoded.role_state_embeddings.device,
            ),
            state_count,
        ).to(dtype=encoded.role_state_embeddings.dtype)
        endpoint_scores = self.score_actions(
            masked,
            state,
            current_state_belief=endpoint_belief,
            steps_remaining=1,
            encoding=encoded,
            include_pointer_memory=False,
            detach_evidence_action_input=detach_evidence_action_input,
            use_legacy_evidence=use_legacy_evidence,
        )
        stop_target = torch.tensor(
            (len(masked.grounded_candidates),),
            device=endpoint_scores.logits.device,
            dtype=torch.long,
        )
        losses.append(
            F.cross_entropy(endpoint_scores.logits.unsqueeze(0), stop_target)
        )
        return torch.stack(losses)

    def public_backward_reasoning_losses(
        self,
        task: PublicSoftwarePipelineTask,
        state: SoftwareReconstructionState,
        *,
        include_pointer_memory: bool = False,
    ) -> torch.Tensor:
        """Supervise the production reasoner from public suffixes.

        Exact package pointers are disabled by default so they cannot teach a
        shortcut that disappears on every fresh query package.  The same role
        memory and backward reasoner used at query time remain active.
        """

        if type(include_pointer_memory) is not bool:
            raise TypeError("include_pointer_memory must be bool")
        encoded = self.encode_task(task)
        lattice = self.transition_lattice(
            encoded,
            state,
            include_pointer_memory=include_pointer_memory,
        )
        prefix = []
        endpoints = []
        state_count = len(task.states)
        stop_index = len(task.grounded_candidates)
        for observation in task.observations:
            if not observation.transitions:
                continue
            goal_index = _state_index(task.states, observation.final_state)
            suffix_start = max(0, len(observation.transitions) - _MAX_STEPS)
            for start in range(suffix_start, len(observation.transitions)):
                transition = observation.transitions[start]
                before_index = _state_index(task.states, transition.before)
                if before_index == goal_index:
                    continue
                belief = F.one_hot(
                    torch.tensor(before_index, device=encoded.role_state_embeddings.device),
                    state_count,
                ).to(dtype=encoded.role_state_embeddings.dtype)
                scores = self.score_actions(
                    task,
                    state,
                    current_state_belief=belief,
                    steps_remaining=len(observation.transitions) - start,
                    encoding=encoded,
                    include_pointer_memory=include_pointer_memory,
                )
                target = torch.tensor(
                    (_action_index(task.grounded_candidates, transition.action),),
                    device=scores.logits.device,
                    dtype=torch.long,
                )
                prefix.append(F.cross_entropy(scores.logits.unsqueeze(0), target))
            endpoint_belief = F.one_hot(
                torch.tensor(goal_index, device=encoded.role_state_embeddings.device),
                state_count,
            ).to(dtype=encoded.role_state_embeddings.dtype)
            endpoint_scores = self.score_actions(
                task,
                state,
                current_state_belief=endpoint_belief,
                steps_remaining=1,
                encoding=encoded,
                include_pointer_memory=include_pointer_memory,
            )
            target = torch.tensor(
                (stop_index,), device=endpoint_scores.logits.device, dtype=torch.long
            )
            endpoints.append(
                F.cross_entropy(endpoint_scores.logits.unsqueeze(0), target)
            )
        groups = []
        if prefix:
            groups.append(torch.stack(prefix).mean())
        if endpoints:
            groups.append(torch.stack(endpoints).mean())
        if not groups:
            return encoded.role_state_embeddings.new_empty((0,))
        return torch.stack(groups)


class CapacityMatchedMonolithController(SoftwarePipelineController):
    """Unchanged monolithic relation arm marked only for paired block parity."""

    relation_pilot_only = True


class CapacityMatchedClusterController(SoftwarePipelineController):
    """Relation-only four-cell pilot with learned anonymous all-active fusion."""

    relation_pilot_only = True

    def __init__(
        self,
        profile: SoftwarePipelineRunProfile,
        *,
        cell_seed: int,
        composer_seed: int,
    ) -> None:
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in (cell_seed, composer_seed)
        ):
            raise ValueError("cluster seeds must be nonnegative integers")
        super().__init__(profile)
        for name in (
            "relation_pool_attention",
            "relation_pool_projection",
            "relation_comparator",
            "evidence_pair_encoder",
            "relation_incidence_readout",
            "relation_incidence_projection",
        ):
            delattr(self, name)
        cpu_rng_state = torch.get_rng_state()
        try:
            torch.default_generator.manual_seed(cell_seed)
            self.relation_cells = nn.ModuleList(
                AnonymousRelationCell() for _ in range(_CLUSTER_CELL_COUNT)
            )
            torch.default_generator.manual_seed(composer_seed)
            self.relation_composer = AnonymousAllActiveRelationComposer()
        finally:
            torch.set_rng_state(cpu_rng_state)
        self._relation_diagnostic_lesion: tuple[str, int | None] | None = None
        self._relation_diagnostic_records: list[dict[str, object]] | None = None

    @property
    def clustered_relation_width(self) -> int:
        return _CLUSTER_CELL_COUNT * _CLUSTER_CELL_WIDTH

    def _factorized_relation_embeddings(
        self,
        components: Sequence[PublicComponentContract],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_tensors = self.evidence_context_encoder.context_tensors(
            components,
            reference,
        )
        predecessor_codes = tuple(
            self._pool_context_tensor(value) for value in context_tensors
        )
        cell_rows = tuple(
            cell.relation_tensors(components, reference)
            for cell in self.relation_cells
        )
        context_codes = []
        relation_codes = []
        for candidate_index in range(len(components)):
            alternatives_by_cell = tuple(
                rows[candidate_index] for rows in cell_rows
            )
            predecessor_indices = tuple(
                predecessor_index
                for predecessor_index, _ in alternatives_by_cell[0]
            )
            if any(
                tuple(index for index, _ in alternatives) != predecessor_indices
                for alternatives in alternatives_by_cell[1:]
            ):
                raise RuntimeError("anonymous cells lost predecessor alignment")
            if not predecessor_indices:
                context_codes.append(reference.new_zeros(self.profile.width))
                relation_codes.append(
                    reference.new_zeros(self.clustered_relation_width)
                )
                continue
            if len(predecessor_indices) != 1:
                raise ValueError(
                    "clustered evidence requires one public predecessor per component"
                )
            context_codes.append(
                F.normalize(
                    torch.stack(
                        tuple(
                            predecessor_codes[index]
                            for index in predecessor_indices
                        )
                    ).mean(dim=0),
                    dim=-1,
                    eps=1.0e-8,
                )
            )
            per_cell_codes = tuple(
                F.normalize(
                    torch.stack(
                        tuple(
                            cell.pool(value)
                            for _, value in alternatives
                        )
                    ).mean(dim=0),
                    dim=-1,
                    eps=1.0e-8,
                )
                for cell, alternatives in zip(
                    self.relation_cells,
                    alternatives_by_cell,
                    strict=True,
                )
            )
            relation_codes.append(torch.cat(per_cell_codes, dim=-1))
        return torch.stack(context_codes), torch.stack(relation_codes)

    def _cluster_relation_logits(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            query_codes.ndim != 2
            or stored_codes.ndim != 2
            or query_codes.shape[1] != self.clustered_relation_width
            or stored_codes.shape[1] != self.clustered_relation_width
            or query_codes.device != stored_codes.device
            or query_codes.dtype != stored_codes.dtype
            or not query_codes.is_floating_point()
            or not bool(torch.isfinite(query_codes).all().item())
            or not bool(torch.isfinite(stored_codes).all().item())
        ):
            raise ValueError("cluster relation codes must be finite aligned matrices")
        query_cells = query_codes.reshape(
            query_codes.shape[0],
            _CLUSTER_CELL_COUNT,
            _CLUSTER_CELL_WIDTH,
        )
        stored_cells = stored_codes.reshape(
            stored_codes.shape[0],
            _CLUSTER_CELL_COUNT,
            _CLUSTER_CELL_WIDTH,
        )
        cell_logits = torch.stack(
            tuple(
                cell.pair_logits(
                    query_cells[:, index, :],
                    stored_cells[:, index, :],
                )
                for index, cell in enumerate(self.relation_cells)
            ),
            dim=-1,
        )
        fused, weights, _, _ = self.relation_composer(
            query_cells,
            stored_cells,
            cell_logits,
        )
        lesion = self._relation_diagnostic_lesion
        if lesion is not None:
            kind, index = lesion
            if kind == "uniform":
                weights = torch.full_like(weights, 1.0 / _CLUSTER_CELL_COUNT)
            elif kind == "single" and index is not None:
                weights = torch.zeros_like(weights)
                weights[..., index] = 1.0
            elif kind == "drop" and index is not None:
                weights = weights.clone()
                weights[..., index] = 0.0
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(weights.dtype).tiny
                )
            else:
                raise RuntimeError("cluster diagnostic lesion is invalid")
            fused = (weights * cell_logits).sum(dim=-1)
        if self._relation_diagnostic_records is not None:
            pairwise = tuple(
                float(
                    (cell_logits[..., left] - cell_logits[..., right])
                    .detach()
                    .abs()
                    .mean()
                    .item()
                )
                for left in range(_CLUSTER_CELL_COUNT)
                for right in range(left + 1, _CLUSTER_CELL_COUNT)
            )
            self._relation_diagnostic_records.append(
                {
                    "mean_weights": tuple(
                        float(value)
                        for value in weights.detach().mean(dim=(0, 1)).tolist()
                    ),
                    "mean_effective_cell_count": float(
                        weights.detach()
                        .square()
                        .sum(dim=-1)
                        .reciprocal()
                        .mean()
                        .item()
                    ),
                    "mean_pairwise_logit_differences": pairwise,
                }
            )
        return fused, weights, cell_logits

    def _relation_pair_logits(
        self,
        query_codes: torch.Tensor,
        stored_codes: torch.Tensor,
    ) -> torch.Tensor:
        return self._cluster_relation_logits(query_codes, stored_codes)[0]

    def _relation_evidence_read(
        self,
        query_context_codes: torch.Tensor,
        query_relation_codes: torch.Tensor,
        stored_contexts: torch.Tensor,
        stored_relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            query_context_codes.ndim != 2
            or stored_contexts.ndim != 2
            or query_context_codes.shape[1] != self.profile.width
            or stored_contexts.shape[1] != self.profile.width
            or query_context_codes.device != stored_contexts.device
            or query_context_codes.dtype != stored_contexts.dtype
            or not query_context_codes.is_floating_point()
            or not bool(torch.isfinite(query_context_codes).all().item())
            or not bool(torch.isfinite(stored_contexts).all().item())
            or stored_contexts.shape[0] <= 0
        ):
            raise ValueError("cluster context codes must be finite aligned matrices")
        context_logits = self._context_pair_logits(
            query_context_codes,
            stored_contexts,
        )
        null_column = context_logits.new_zeros(query_context_codes.shape[0], 1)
        all_context_weights = torch.softmax(
            torch.cat(
                (
                    context_logits / _RELATION_CONTEXT_AUX_TEMPERATURE,
                    null_column,
                ),
                dim=-1,
            ),
            dim=-1,
        )
        context_weights = all_context_weights[:, : stored_contexts.shape[0]]
        context_null_weights = all_context_weights[:, stored_contexts.shape[0]]
        relation_logits = self._relation_pair_logits(
            query_relation_codes,
            stored_relations,
        )
        scores = (context_weights * relation_logits).sum(dim=-1)
        return scores, context_weights, context_null_weights, relation_logits

    def set_relation_diagnostic_lesion(
        self,
        kind: str | None,
        index: int | None = None,
    ) -> None:
        if kind is None:
            if index is not None:
                raise ValueError("clearing a lesion cannot retain an index")
            self._relation_diagnostic_lesion = None
            return
        if kind == "uniform":
            if index is not None:
                raise ValueError("uniform lesion has no cell index")
        elif kind in ("single", "drop"):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < _CLUSTER_CELL_COUNT
            ):
                raise ValueError("cell lesion index is invalid")
        else:
            raise ValueError("unknown cluster diagnostic lesion")
        self._relation_diagnostic_lesion = (kind, index)

    def begin_relation_diagnostics(self) -> None:
        if self._relation_diagnostic_records is not None:
            raise RuntimeError("cluster diagnostics are already active")
        self._relation_diagnostic_records = []

    def end_relation_diagnostics(self) -> dict[str, object]:
        records = self._relation_diagnostic_records
        self._relation_diagnostic_records = None
        if records is None or not records:
            raise RuntimeError("cluster diagnostics have no recorded reads")
        weights = tuple(record["mean_weights"] for record in records)
        effective = tuple(
            float(record["mean_effective_cell_count"]) for record in records
        )
        pairwise = tuple(
            value
            for record in records
            for value in record["mean_pairwise_logit_differences"]
        )
        return {
            "relation_reads": len(records),
            "mean_cell_weights": tuple(
                sum(float(row[index]) for row in weights) / len(weights)
                for index in range(_CLUSTER_CELL_COUNT)
            ),
            "mean_effective_cell_count": sum(effective) / len(effective),
            "mean_pairwise_logit_difference": sum(pairwise) / len(pairwise),
            "all_cells_positive_weight": all(
                float(value) > 0.0 for row in weights for value in row
            ),
        }


def build_software_pipeline_controller(
    profile: str,
    *,
    device: torch.device | str = "cpu",
) -> SoftwarePipelineController:
    try:
        selected = SOFTWARE_PIPELINE_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown software-pipeline profile: {profile}") from error
    return SoftwarePipelineController(selected).to(device)


def acquire_public_pipeline_traces(
    controller: SoftwarePipelineController,
    task: PublicSoftwarePipelineTask,
    state: SoftwareReconstructionState,
) -> SoftwareTraceAcquisition:
    """Atomically retain exact local pointers and transferable role events."""

    _validate_controller_state(controller, state)
    transitions = _public_transitions(task)
    if not transitions:
        return SoftwareTraceAcquisition(state, 0, 0, 0)
    encoded = controller.encode_task(task)
    pointer_keys = []
    pointer_values = []
    pointer_pair_ids = []
    pointer_successor_ids = []
    role_keys = []
    role_values = []
    for transition in transitions:
        before_index = _state_index(task.states, transition.before)
        after_index = _state_index(task.states, transition.after)
        component_index = _action_index(task.grounded_candidates, transition.action)
        pointer_keys.append(
            F.normalize(
                encoded.pointer_state_embeddings[before_index]
                + encoded.pointer_component_embeddings[component_index],
                dim=-1,
                eps=1.0e-8,
            )
        )
        pointer_values.append(encoded.pointer_state_embeddings[after_index])
        pointer_pair_ids.append(encoded.pointer_pair_ids[before_index, component_index])
        pointer_successor_ids.append(encoded.pointer_successor_ids[after_index])
        role_keys.append(encoded.role_pair_keys[before_index, component_index])
        role_values.append(
            controller.trace_role_value(
                encoded.local_pair_embeddings[before_index, component_index],
                encoded.operator_embeddings[component_index],
                encoded.relative_effect_embeddings[
                    before_index, component_index, after_index
                ],
            )
        )
    pointer_write = controller.pointer_memory.write_events(
        torch.stack(pointer_keys),
        torch.stack(pointer_values),
        state.pointer,
        lane="trace",
        public_source_action_ids=torch.stack(pointer_pair_ids),
        public_successor_ids=torch.stack(pointer_successor_ids),
    )
    zero_ids = torch.zeros(
        (len(role_keys), _POINTER_WORDS),
        device=state.role.keys.device,
        dtype=torch.long,
    )
    role_write = controller.role_memory.write_events(
        torch.stack(role_keys),
        torch.stack(role_values),
        state.role,
        lane="trace",
        public_source_action_ids=zero_ids,
        public_successor_ids=zero_ids,
    )
    if not pointer_write.accepted or not role_write.accepted:
        return SoftwareTraceAcquisition(state, len(transitions), 0, 0)
    context_trace_keys = state.context_trace_keys.clone()
    relation_trace_values = state.relation_trace_values.clone()
    context_events = tuple(
        encoded.relation_context_embeddings[
            _action_index(task.grounded_candidates, transition.action)
        ]
        for transition in transitions
    )
    relation_events = tuple(
        encoded.relation_component_embeddings[
            _action_index(task.grounded_candidates, transition.action)
        ]
        for transition in transitions
    )
    if not (
        len(context_events)
        == len(relation_events)
        == len(role_write.write_slots)
    ):
        raise RuntimeError("factorized relation events lost role-slot alignment")
    for slot, context_code, relation_code in zip(
        role_write.write_slots,
        context_events,
        relation_events,
        strict=True,
    ):
        if not 0 <= slot < controller.role_memory.trace_slot_count:
            raise RuntimeError("factorized event escaped the role trace partition")
        context_trace_keys[0, slot] = context_code
        relation_trace_values[0, slot] = relation_code
    candidate = SoftwareReconstructionState(
        pointer=pointer_write.state,
        role=role_write.state,
        context_trace_keys=context_trace_keys,
        relation_trace_values=relation_trace_values,
    )
    try:
        finite = bool(torch.isfinite(controller.score_actions(task, candidate).logits).all().item())
    except (RuntimeError, ValueError):
        finite = False
    if not finite:
        return SoftwareTraceAcquisition(state, len(transitions), 0, 0)
    return SoftwareTraceAcquisition(
        candidate,
        len(transitions),
        len(pointer_write.write_slots),
        len(role_write.write_slots),
    )


def rollout_software_pipeline(
    controller: SoftwarePipelineController,
    task: PublicSoftwarePipelineTask,
    state: SoftwareReconstructionState,
    *,
    greedy: bool = True,
    temperature: float = 1.0,
    include_pointer_memory: bool = True,
    include_role_memory: bool = True,
    include_backward_reasoning: bool = True,
) -> SoftwareNeuralRollout:
    """Commit one bounded autoregressive component-or-STOP trajectory."""

    _validate_controller_state(controller, state)
    _validate_public_task(task)
    if type(greedy) is not bool:
        raise TypeError("greedy must be bool")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    encoded = controller.encode_task(task)
    current_belief = F.one_hot(
        torch.tensor(encoded.origin_index, device=encoded.role_state_embeddings.device),
        len(task.states),
    ).to(dtype=encoded.role_state_embeddings.dtype)
    actions: list[GroundAction] = []
    step_logits = []
    selected_indices = []
    step_role_keys = []
    step_currents = []
    step_beliefs = []
    stopped = False
    stop_index = len(task.grounded_candidates)
    for step in range(task.max_steps):
        scores = controller.score_actions(
            task,
            state,
            current_state_belief=current_belief,
            steps_remaining=task.max_steps - step,
            encoding=encoded,
            include_pointer_memory=include_pointer_memory,
            include_role_memory=include_role_memory,
            include_backward_reasoning=include_backward_reasoning,
        )
        decision_logits = scores.logits / temperature
        if greedy:
            selected = int(decision_logits.argmax(dim=-1).item())
        else:
            selected = int(
                torch.multinomial(torch.softmax(decision_logits, dim=-1), 1).item()
            )
        current = current_belief @ encoded.role_state_embeddings
        if selected == stop_index:
            stop_relation = current_belief @ encoded.stop_relation_embeddings
            role_key = F.normalize(
                controller.stop_key_encoder(stop_relation),
                dim=-1,
                eps=1.0e-8,
            )
        else:
            role_key = torch.einsum(
                "s,sw->w", current_belief, encoded.role_pair_keys[:, selected]
            )
        step_logits.append(decision_logits)
        selected_indices.append(selected)
        step_role_keys.append(role_key)
        step_currents.append(current)
        step_beliefs.append(current_belief)
        if selected == stop_index:
            stopped = True
            break
        actions.append(task.grounded_candidates[selected])
        current_belief = torch.softmax(scores.successor_state_logits[selected], dim=-1)
    pipeline = commit_software_pipeline(task, actions, stopped=stopped)
    return SoftwareNeuralRollout(
        pipeline=pipeline,
        step_logits=tuple(step_logits),
        selected_indices=tuple(selected_indices),
        step_role_keys=tuple(step_role_keys),
        step_current_embeddings=tuple(step_currents),
        step_state_beliefs=tuple(step_beliefs),
        component_count=len(task.grounded_candidates),
        task_binding=_public_task_binding(task),
        incoming_state_digest=software_reconstruction_state_digest(state),
    )


def scalar_pipeline_outcome_loss(
    rollout: SoftwareNeuralRollout,
    reward: float,
) -> torch.Tensor:
    advantage = _validate_reward(reward) - 0.5
    return -rollout.step_logits[0].new_tensor(advantage) * _trajectory_log_probability(
        rollout
    )


def centered_pipeline_preference_loss(
    rollouts: Sequence[SoftwareNeuralRollout],
    rewards: Sequence[float],
) -> torch.Tensor:
    if len(rollouts) != len(rewards) or len(rollouts) < 2:
        raise ValueError("pipeline preference requires aligned repeated attempts")
    numeric = tuple(_validate_reward(value) for value in rewards)
    mean_reward = sum(numeric) / len(numeric)
    reference = rollouts[0].step_logits[0]
    return torch.stack(
        [
            -reference.new_tensor(reward - mean_reward)
            * _trajectory_log_probability(rollout)
            for rollout, reward in zip(rollouts, numeric, strict=True)
        ]
    ).mean()


def apply_scalar_pipeline_feedback(
    controller: SoftwarePipelineController,
    task: PublicSoftwarePipelineTask,
    rollout: SoftwareNeuralRollout,
    reward: float,
    state: SoftwareReconstructionState,
    *,
    binding_state: SoftwareReconstructionState | None = None,
    minimum_effect: float = 0.0,
) -> SoftwareScalarFeedback:
    """Apply one terminal scalar to the role lane or restore exactly."""

    numeric = _validate_reward(reward)
    _validate_controller_state(controller, state)
    if rollout.task_binding != _public_task_binding(task):
        raise ValueError("rollout is bound to a different public task")
    bound = state if binding_state is None else binding_state
    _validate_controller_state(controller, bound)
    if rollout.incoming_state_digest != software_reconstruction_state_digest(bound):
        raise ValueError("rollout is stale for its declared competence state")
    if binding_state is not None and not _public_trace_lanes_equal(state, binding_state):
        raise ValueError("rebound scalar feedback changed retained public traces")
    if not math.isfinite(minimum_effect) or minimum_effect < 0.0:
        raise ValueError("minimum_effect must be finite and nonnegative")
    encoding = controller.encode_task(task)
    procedure = controller.summarize_pipeline(task, rollout.pipeline, encoding=encoding)
    values = []
    for current, key in zip(
        rollout.step_current_embeddings, rollout.step_role_keys, strict=True
    ):
        value = torch.tanh(
            controller.role_outcome_encoder(torch.cat((current, key, procedure), dim=-1))
        )
        values.append(value * (2.0 * numeric - 1.0))
    write = controller.role_memory.write_events(
        torch.stack(rollout.step_role_keys),
        torch.stack(values),
        state.role,
        lane="outcome",
        minimum_effect=minimum_effect,
    )
    candidate = SoftwareReconstructionState(
        pointer=state.pointer,
        role=write.state,
        context_trace_keys=state.context_trace_keys,
        relation_trace_values=state.relation_trace_values,
    )
    accepted = write.accepted
    if accepted:
        try:
            finite = bool(torch.isfinite(controller.score_actions(task, candidate).logits).all().item())
        except (RuntimeError, ValueError):
            finite = False
        if not finite:
            candidate = state
            accepted = False
    if not accepted:
        candidate = state
    return SoftwareScalarFeedback(
        state=candidate,
        accepted=accepted,
        scalar_observations=1,
        write_slots=write.write_slots,
        delta_norm=write.delta_norm if accepted else 0.0,
    )


def snapshot_software_reconstruction_state(
    state: SoftwareReconstructionState,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, SoftwareReconstructionState):
        raise TypeError("state must be SoftwareReconstructionState")
    result = {}
    for lane, lane_state in (("pointer", state.pointer), ("role", state.role)):
        for name, value in snapshot_glyph_state(lane_state).items():
            result[f"{lane}.{name}"] = value
    result["context_trace_keys"] = state.context_trace_keys.detach().clone()
    result["relation_trace_values"] = state.relation_trace_values.detach().clone()
    return result


def restore_software_reconstruction_state(
    snapshot: Mapping[str, torch.Tensor],
) -> SoftwareReconstructionState:
    glyph_names = set(snapshot_glyph_state(_empty_reference_state()).keys())
    expected = {
        f"{lane}.{name}" for lane in ("pointer", "role") for name in glyph_names
    } | {"context_trace_keys", "relation_trace_values"}
    if set(snapshot) != expected:
        raise ValueError("software reconstruction snapshot keys differ")
    return SoftwareReconstructionState(
        pointer=restore_glyph_state(
            {name: snapshot[f"pointer.{name}"] for name in glyph_names}
        ),
        role=restore_glyph_state(
            {name: snapshot[f"role.{name}"] for name in glyph_names}
        ),
        context_trace_keys=snapshot["context_trace_keys"].detach().clone(),
        relation_trace_values=snapshot["relation_trace_values"].detach().clone(),
    )


def software_reconstruction_state_digest(state: SoftwareReconstructionState) -> str:
    digest = hashlib.sha256(_STATE_DIGEST_DOMAIN)
    for name, value in sorted(snapshot_software_reconstruction_state(state).items()):
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
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def software_pipeline_model_digest(
    controller: SoftwarePipelineController,
) -> str:
    """Return a scalar-safe digest of all controller parameters and buffers."""

    if not isinstance(controller, SoftwarePipelineController):
        raise TypeError("controller must be SoftwarePipelineController")
    digest = hashlib.sha256(_MODEL_DIGEST_DOMAIN)
    for name, value in sorted(controller.state_dict().items()):
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
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return "sha256:" + digest.hexdigest()


def anonymous_conflict_mixer_digest(mixer: AnonymousConflictMixer) -> str:
    """Return a scalar-safe digest of the learned v12 update rule."""

    if not isinstance(mixer, AnonymousConflictMixer):
        raise TypeError("mixer must be AnonymousConflictMixer")
    digest = hashlib.sha256(_CONFLICT_MIXER_DIGEST_DOMAIN)
    digest.update(
        json.dumps(
            {
                "feature_count": mixer.feature_count,
                "hidden_width": mixer.hidden_width,
                "anchor_weight": mixer.anchor_weight,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, value in sorted(mixer.state_dict().items()):
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


def public_relation_conflict_system_digest(
    controller: SoftwarePipelineController,
    mixer: AnonymousConflictMixer,
    state: SoftwareReconstructionState,
) -> str:
    """Bind all learned and live competence state in one lineage identity."""

    if not isinstance(state, SoftwareReconstructionState):
        raise TypeError("state must be SoftwareReconstructionState")
    digest = hashlib.sha256(_CONFLICT_SYSTEM_DIGEST_DOMAIN)
    for value in (
        software_pipeline_model_digest(controller),
        anonymous_conflict_mixer_digest(mixer),
        software_reconstruction_state_digest(state),
    ):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def public_relation_conflict_parameter_report(
    controller: SoftwarePipelineController,
    mixer: AnonymousConflictMixer,
) -> dict[str, int | float | str]:
    """Report the complete learned system, including its update rule."""

    if not isinstance(controller, SoftwarePipelineController):
        raise TypeError("controller must be SoftwarePipelineController")
    if not isinstance(mixer, AnonymousConflictMixer):
        raise TypeError("mixer must be AnonymousConflictMixer")
    controller_total = sum(parameter.numel() for parameter in controller.parameters())
    controller_trainable = sum(
        parameter.numel()
        for parameter in controller.parameters()
        if parameter.requires_grad
    )
    mixer_total = sum(parameter.numel() for parameter in mixer.parameters())
    mixer_trainable = sum(
        parameter.numel() for parameter in mixer.parameters() if parameter.requires_grad
    )
    return {
        "protocol_id": _CONFLICT_PROTOCOL_ID,
        "profile": controller.profile.name,
        "controller_parameters": controller_total,
        "controller_trainable_parameters": controller_trainable,
        "mixer_parameters": mixer_total,
        "mixer_trainable_parameters": mixer_trainable,
        "complete_learned_system_parameters": controller_total + mixer_total,
        "mixer_feature_count": mixer.feature_count,
        "mixer_hidden_width": mixer.hidden_width,
        "mixer_anchor_weight": mixer.anchor_weight,
    }


def save_public_relation_conflict_checkpoint(
    path: str | Path,
    controller: SoftwarePipelineController,
    mixer: AnonymousConflictMixer,
    state: SoftwareReconstructionState,
) -> None:
    """Persist the controller, competence state, and learned update rule together."""

    if not isinstance(controller, SoftwarePipelineController):
        raise TypeError("controller must be SoftwarePipelineController")
    if not isinstance(mixer, AnonymousConflictMixer):
        raise TypeError("mixer must be AnonymousConflictMixer")
    if not isinstance(state, SoftwareReconstructionState):
        raise TypeError("state must be SoftwareReconstructionState")
    payload = {
        "version": _CONFLICT_CHECKPOINT_VERSION,
        "protocol_id": _CONFLICT_PROTOCOL_ID,
        "profile": asdict(controller.profile),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in controller.state_dict().items()
        },
        "controller_digest": software_pipeline_model_digest(controller),
        "competence_state": {
            name: value.detach().cpu().clone()
            for name, value in snapshot_software_reconstruction_state(state).items()
        },
        "competence_digest": software_reconstruction_state_digest(state),
        "mixer_config": {
            "feature_count": mixer.feature_count,
            "hidden_width": mixer.hidden_width,
            "anchor_weight": mixer.anchor_weight,
        },
        "mixer_state": {
            name: value.detach().cpu().clone()
            for name, value in mixer.state_dict().items()
        },
        "mixer_digest": anonymous_conflict_mixer_digest(mixer),
        "system_digest": public_relation_conflict_system_digest(
            controller,
            mixer,
            state,
        ),
        "parameter_report": public_relation_conflict_parameter_report(
            controller,
            mixer,
        ),
    }
    torch.save(payload, Path(path))


def load_public_relation_conflict_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[
    SoftwarePipelineController,
    AnonymousConflictMixer,
    SoftwareReconstructionState,
]:
    """Restore one complete v12 lineage and verify every persisted digest."""

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _CONFLICT_CHECKPOINT_VERSION
        or payload.get("protocol_id") != _CONFLICT_PROTOCOL_ID
    ):
        raise RuntimeError("conflict-reconcile checkpoint identity is invalid")
    profile = SoftwarePipelineRunProfile(**payload["profile"])
    if SOFTWARE_PIPELINE_PROFILES.get(profile.name) != profile:
        raise RuntimeError("conflict-reconcile checkpoint profile is not registered")
    mixer_config = payload.get("mixer_config")
    if not isinstance(mixer_config, dict) or set(mixer_config) != {
        "feature_count",
        "hidden_width",
        "anchor_weight",
    }:
        raise RuntimeError("conflict-reconcile mixer config is invalid")
    controller = SoftwarePipelineController(profile).to(device)
    controller.load_state_dict(payload["model_state"], strict=True)
    mixer = AnonymousConflictMixer(**mixer_config).to(device)
    mixer.load_state_dict(payload["mixer_state"], strict=True)
    state = restore_software_reconstruction_state(payload["competence_state"])
    if software_pipeline_model_digest(controller) != payload.get("controller_digest"):
        raise RuntimeError("conflict-reconcile controller changed")
    if anonymous_conflict_mixer_digest(mixer) != payload.get("mixer_digest"):
        raise RuntimeError("conflict-reconcile learned update rule changed")
    if software_reconstruction_state_digest(state) != payload.get("competence_digest"):
        raise RuntimeError("conflict-reconcile competence state changed")
    if public_relation_conflict_system_digest(
        controller,
        mixer,
        state,
    ) != payload.get("system_digest"):
        raise RuntimeError("conflict-reconcile combined lineage changed")
    if public_relation_conflict_parameter_report(
        controller,
        mixer,
    ) != payload.get("parameter_report"):
        raise RuntimeError("conflict-reconcile parameter report changed")
    controller.eval()
    mixer.eval()
    return controller, mixer, state


def save_software_pipeline_checkpoint(
    path: str | Path,
    controller: SoftwarePipelineController,
    state: SoftwareReconstructionState,
) -> None:
    payload = {
        "version": _CHECKPOINT_VERSION,
        "profile": asdict(controller.profile),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in controller.state_dict().items()
        },
        "competence_state": {
            name: value.detach().cpu().clone()
            for name, value in snapshot_software_reconstruction_state(state).items()
        },
        "competence_digest": software_reconstruction_state_digest(state),
    }
    torch.save(payload, Path(path))


def load_software_pipeline_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[SoftwarePipelineController, SoftwareReconstructionState]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != _CHECKPOINT_VERSION:
        raise RuntimeError("software-pipeline checkpoint identity is invalid")
    profile = SoftwarePipelineRunProfile(**payload["profile"])
    if SOFTWARE_PIPELINE_PROFILES.get(profile.name) != profile:
        raise RuntimeError("software-pipeline checkpoint profile is not registered")
    controller = SoftwarePipelineController(profile).to(device)
    controller.load_state_dict(payload["model_state"], strict=True)
    state = restore_software_reconstruction_state(payload["competence_state"])
    if software_reconstruction_state_digest(state) != payload.get("competence_digest"):
        raise RuntimeError("software-pipeline checkpoint competence state changed")
    controller.eval()
    return controller, state


def software_pipeline_parameter_report(
    controller: SoftwarePipelineController,
) -> dict[str, int | str]:
    total = sum(parameter.numel() for parameter in controller.parameters())
    trainable = sum(
        parameter.numel() for parameter in controller.parameters() if parameter.requires_grad
    )
    return {
        "protocol_id": _RELATION_PROTOCOL_ID,
        "profile": controller.profile.name,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "pointer_state_elements": controller.pointer_memory.state_numel(1),
        "role_state_elements": controller.role_memory.state_numel(1),
        "context_trace_state_elements": controller.profile.role_slots
        * controller.profile.width,
        "relation_trace_value_state_elements": controller.profile.role_slots
        * controller.profile.width,
        "factorized_relation_state_elements": 2
        * controller.profile.role_slots
        * controller.profile.width,
        "complete_pipeline_candidates": 0,
        "reasoner": "recurrent_backward_goal_messages",
    }


@dataclass(slots=True)
class _ScalarLedger:
    judge: Callable[[GeneratedSoftwarePipelineTask, CommittedSoftwarePipeline], float]
    calls: int = 0

    def __call__(
        self,
        pair: GeneratedSoftwarePipelineTask,
        pipeline: CommittedSoftwarePipeline,
    ) -> float:
        value = _validate_reward(self.judge(pair, pipeline))
        self.calls += 1
        return value


def _legacy_public_relation_fit_plan() -> dict[str, object]:
    """Return the retired v2 wrong-evidence schedule for audit-only helpers."""

    commitments = software_pipeline_mechanism_partition("train")
    fit = commitments[:_RELATION_FIT_COMMITMENTS]
    gate = commitments[
        _RELATION_GATE_COMMITMENT_OFFSET :
        _RELATION_GATE_COMMITMENT_OFFSET + _RELATION_GATE_COMMITMENTS
    ]
    if len(fit) != _RELATION_FIT_COMMITMENTS or len(gate) != _RELATION_GATE_COMMITMENTS:
        raise RuntimeError("train partition cannot satisfy the legacy relation plan")
    return {
        "protocol_id": _LEGACY_RELATION_PROTOCOL_ID,
        "partition": "train",
        "fit_commitments": fit,
        "fit_seed_pairs": _RELATION_FIT_SEED_PAIRS,
        "fit_rows": len(fit) * len(_RELATION_FIT_SEED_PAIRS) * 4,
        "fit_directional_arms": len(fit) * len(_RELATION_FIT_SEED_PAIRS) * 8,
        "gate_commitments": gate,
        "gate_seed_pairs": _RELATION_GATE_SEED_PAIRS,
        "gate_rows": len(gate) * len(_RELATION_GATE_SEED_PAIRS) * 4,
        "supports_per_motif": 2,
        "supports_per_fold": 3,
        "queries_per_stream": 1,
        "maximum_steps": 4,
    }


def _relation_credit_train_seed_pair(
    update_index: int,
    commitment_index: int,
) -> tuple[int, int]:
    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or not 0 <= update_index < sum(_RELATION_CREDIT_STAGE_UPDATES.values())
        or isinstance(commitment_index, bool)
        or not isinstance(commitment_index, int)
        or not 0 <= commitment_index < _RELATION_CREDIT_COMMITMENTS
    ):
        raise ValueError("relation-credit seed indices are outside the fixed schedule")
    offset = 100_000 * update_index + 1_000 * commitment_index
    return (
        _RELATION_CREDIT_TRAIN_TOPOLOGY_BASE + offset,
        _RELATION_CREDIT_TRAIN_SURFACE_BASE + offset,
    )


def _relation_credit_panel_seed_pairs(
    topology_base: int,
    surface_base: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (topology_base + 1_000 * index, surface_base + 1_000 * index)
        for index in range(_RELATION_CREDIT_COMMITMENTS)
    )


def public_relation_fit_plan() -> dict[str, object]:
    """Return the exact v11 public-credit schedule without executing it."""

    commitments = software_pipeline_mechanism_partition("train")[
        :_RELATION_CREDIT_COMMITMENTS
    ]
    if len(commitments) != _RELATION_CREDIT_COMMITMENTS:
        raise RuntimeError("train partition cannot satisfy the public-credit plan")
    stage_seed_batches: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
    update_offset = 0
    for stage, update_count in _RELATION_CREDIT_STAGE_UPDATES.items():
        stage_seed_batches[stage] = tuple(
            tuple(
                _relation_credit_train_seed_pair(update_index, commitment_index)
                for commitment_index in range(_RELATION_CREDIT_COMMITMENTS)
            )
            for update_index in range(update_offset, update_offset + update_count)
        )
        update_offset += update_count
    relation_panel = _relation_credit_panel_seed_pairs(
        _RELATION_CREDIT_PANEL_TOPOLOGY_BASE,
        _RELATION_CREDIT_PANEL_SURFACE_BASE,
    )
    final_panel = _relation_credit_panel_seed_pairs(
        _RELATION_CREDIT_FINAL_TOPOLOGY_BASE,
        _RELATION_CREDIT_FINAL_SURFACE_BASE,
    )
    all_training_pairs = tuple(
        pair
        for batches in stage_seed_batches.values()
        for batch in batches
        for pair in batch
    )
    if (
        len(set(commitments)) != len(commitments)
        or len(all_training_pairs)
        != sum(_RELATION_CREDIT_STAGE_UPDATES.values())
        * _RELATION_CREDIT_STREAMS_PER_UPDATE
        or len(set(all_training_pairs)) != len(all_training_pairs)
        or len(set(relation_panel)) != len(relation_panel)
        or len(set(final_panel)) != len(final_panel)
        or set(all_training_pairs) & set(relation_panel)
        or set(all_training_pairs) & set(final_panel)
        or set(relation_panel) & set(final_panel)
    ):
        raise RuntimeError("public-credit schedule is not unique and disjoint")
    return {
        "protocol_id": _RELATION_PROTOCOL_ID,
        "partition": "train",
        "initialization_seed": _RELATION_CREDIT_INITIALIZATION_SEED,
        "commitments": commitments,
        "stage_updates": dict(_RELATION_CREDIT_STAGE_UPDATES),
        "relation_terminal_gate_update": _RELATION_CREDIT_STAGE_UPDATES[
            "relation"
        ],
        "relation_intermediate_selection": "none",
        "relation_replay": False,
        "streams_per_update": _RELATION_CREDIT_STREAMS_PER_UPDATE,
        "rows_per_stream": 4,
        "stage_seed_batches": stage_seed_batches,
        "relation_context_panel_seed_pairs": relation_panel,
        "final_panel_seed_pairs": final_panel,
        "supports_per_motif": 2,
        "queries_per_stream": 1,
        "maximum_steps": 4,
        "optimizer": "AdamW",
        "optimizer_state_policy": "fresh_per_stage",
        "encoder_learning_rate": 3.0e-4,
        "head_learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "gradient_clip": 5.0,
        "stream_objective": {
            "relation": "anonymous_entropic_worst_stream",
            "context": "supported_valid_set_row_mean",
            "joint": "anonymous_entropic_worst_stream",
            "temperature": _RELATION_CREDIT_STREAM_TEMPERATURE,
            "mean_weight": _RELATION_CREDIT_STREAM_MEAN_WEIGHT,
            "robust_weight": _RELATION_CREDIT_STREAM_ROBUST_WEIGHT,
            "minimum_robust_gradient_weight_per_stream": 0.5
            / _RELATION_CREDIT_STREAMS_PER_UPDATE,
            "context_stream_weighting": "supported_row_count_fraction",
        },
        "row_objective": {
            "relation": "anonymous_entropic_worst_row_within_stream",
            "context": "supported_valid_set_rows_only",
            "joint": "anonymous_entropic_worst_row_within_stream",
            "temperature": _RELATION_CREDIT_ROW_TEMPERATURE,
            "mean_weight": _RELATION_CREDIT_ROW_MEAN_WEIGHT,
            "robust_weight": _RELATION_CREDIT_ROW_ROBUST_WEIGHT,
            "minimum_gradient_weight_per_row": (
                _RELATION_CREDIT_ROW_MEAN_WEIGHT / 4
            ),
        },
        "valid_witness_set": {
            "positive_margin_minimum": 0.05,
            "negative_margin_maximum": -0.05,
            "witness_minimum": _RELATION_FIT_MARGIN,
            "same_slot_conjunction": True,
            "relation_supported_if_nonempty": True,
            "context_denominator": "supported_rows_only",
            "context_mass": "sum_all_valid_real_slots",
            "context_top_one": (
                "strict_valid_max_gt_invalid_real_and_null_max"
            ),
            "empty_set_relation_supported": False,
            "empty_set_context_top_one": False,
            "context_training_loss": "negative_log_valid_set_mass",
            "context_training_rows": "supported_rows_only",
            "context_training_mask": "detached_relation_margins",
            "context_training_runtime_effect": "none",
        },
        "thresholds": {
            "relation_supported_rows": _RELATION_CREDIT_RELATION_CONFIDENT_ROWS,
            "relation_supported_streams": _RELATION_CREDIT_RELATION_CONFIDENT_STREAMS,
            "context_valid_set_top_one_fraction_supported": (
                _RELATION_CREDIT_CONTEXT_TOP_ONE
            ),
            "context_valid_set_mass_mean_supported": _RELATION_CREDIT_CONTEXT_MASS,
            "final_positive_margin_mean": _RELATION_FIT_MARGIN,
            "final_negative_margin_mean": -_RELATION_FIT_MARGIN,
            "final_separation_mean": 2.0 * _RELATION_FIT_MARGIN,
            "final_signed_rows": _RELATION_CREDIT_FINAL_SIGNED_ROWS,
            "final_signed_streams": _RELATION_CREDIT_FINAL_SIGNED_STREAMS,
            "permutation_max_delta": _RELATION_GATE_PERMUTATION_TOLERANCE,
        },
    }


def _conflict_train_seed_pair(
    update_index: int,
    commitment_index: int,
) -> tuple[int, int]:
    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or not 0 <= update_index < sum(_RELATION_CREDIT_STAGE_UPDATES.values())
        or isinstance(commitment_index, bool)
        or not isinstance(commitment_index, int)
        or not 0 <= commitment_index < _RELATION_CREDIT_COMMITMENTS
    ):
        raise ValueError("conflict-reconcile seed indices are outside the schedule")
    offset = 100_000 * update_index + 1_000 * commitment_index
    return (
        _CONFLICT_TRAIN_TOPOLOGY_BASE + offset,
        _CONFLICT_TRAIN_SURFACE_BASE + offset,
    )


def public_relation_conflict_fit_plan() -> dict[str, object]:
    """Return the fixed v12 learned conflict-reconciliation schedule."""

    base = public_relation_fit_plan()
    stage_seed_batches: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
    update_offset = 0
    for stage, update_count in _RELATION_CREDIT_STAGE_UPDATES.items():
        stage_seed_batches[stage] = tuple(
            tuple(
                _conflict_train_seed_pair(update_index, commitment_index)
                for commitment_index in range(_RELATION_CREDIT_COMMITMENTS)
            )
            for update_index in range(update_offset, update_offset + update_count)
        )
        update_offset += update_count
    relation_panel = _relation_credit_panel_seed_pairs(
        _CONFLICT_PANEL_TOPOLOGY_BASE,
        _CONFLICT_PANEL_SURFACE_BASE,
    )
    final_panel = _relation_credit_panel_seed_pairs(
        _CONFLICT_FINAL_TOPOLOGY_BASE,
        _CONFLICT_FINAL_SURFACE_BASE,
    )
    training_pairs = {
        pair
        for batches in stage_seed_batches.values()
        for batch in batches
        for pair in batch
    }
    if (
        len(training_pairs)
        != sum(_RELATION_CREDIT_STAGE_UPDATES.values())
        * _RELATION_CREDIT_STREAMS_PER_UPDATE
        or training_pairs & set(relation_panel)
        or training_pairs & set(final_panel)
        or set(relation_panel) & set(final_panel)
    ):
        raise RuntimeError("v12 conflict schedule is not unique and disjoint")
    return {
        **base,
        "protocol_id": _CONFLICT_PROTOCOL_ID,
        "initialization_seed": _CONFLICT_INITIALIZATION_SEED,
        "mixer_initialization_seed": _CONFLICT_MIXER_INITIALIZATION_SEED,
        "stage_seed_batches": stage_seed_batches,
        "relation_context_panel_seed_pairs": relation_panel,
        "final_panel_seed_pairs": final_panel,
        "update_rule": {
            "name": "anonymous_learned_blockwise_conflict_reconciliation",
            "relation_parameter_blocks": (
                "pair_encoder",
                "global_readout",
                "incidence_readout",
                "comparator",
            ),
            "joint_parameter_blocks": (
                "pair_encoder",
                "global_readout",
                "incidence_readout",
                "comparator",
                "context",
            ),
            "features": (
                "standardized_public_stream_loss",
                "standardized_log_gradient_norm",
                "existing_outer_weight",
                "mean_gradient_cosine",
                "minimum_gradient_cosine",
                "negative_gradient_fraction",
            ),
            "features_detached_from_controller": True,
            "stream_identity_input": False,
            "task_identity_input": False,
            "residual_initialization": "exact_zero",
            "first_relation_update_twin_atol": 2.0e-6,
            "first_relation_update_twin_rtol": 2.0e-5,
            "anchored_existing_weight": _CONFLICT_MIXER_ANCHOR_WEIGHT,
            "mixer_learning_rate": _CONFLICT_MIXER_LEARNING_RATE,
            "meta_objective": "symmetric_leave_one_stream_out_alignment",
            "withheld_folds_per_update": _RELATION_CREDIT_STREAMS_PER_UPDATE,
            "alignment_margin": _CONFLICT_ALIGNMENT_MARGIN,
            "alignment_temperature": _CONFLICT_ALIGNMENT_TEMPERATURE,
            "meta_mean_weight": _CONFLICT_META_MEAN_WEIGHT,
            "meta_robust_weight": _CONFLICT_META_ROBUST_WEIGHT,
            "meta_kl_weight": _CONFLICT_META_KL_WEIGHT,
            "deterministic_gradient_projection": False,
        },
        "cluster_pilot_rule": {
            "selection_basis": "mechanistic_integrity_not_v12_performance",
            "relation_or_final_gate_used_as_go_threshold": False,
            "required_runtime_observations": (
                "complete_relation_r80",
                "finite_recorded_geometry_and_meta_components",
                "frozen_controller_parameters_unchanged",
                "eight_symmetric_withheld_folds_per_update",
                "mixer_terminal_digest_differs_from_initial",
                "at_least_one_post_first_weight_vector_differs_from_existing",
                "at_least_one_post_first_applied_direction_differs_from_legacy",
            ),
            "required_pre_run_evidence": (
                "numerical_twin_equivalence_of_first_relation_update",
                "meta_objective_permutation_gradient_and_update_equivariance",
            ),
            "required_artifact": "reloadable_combined_controller_competence_mixer_checkpoint",
            "performance_interpretation": (
                "relation_and_final outcomes are recorded but cannot select whether "
                "the capacity-matched causal cluster pilot runs"
            ),
        },
    }


def _cluster_stream_binding_digest(
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
    digest = hashlib.sha256(_CLUSTER_DIGEST_DOMAIN)
    digest.update(json.dumps(payload, separators=(",", ":")).encode("ascii"))
    return "sha256:" + digest.hexdigest()


def capacity_matched_relation_cluster_fit_plan() -> dict[str, object]:
    """Return the fixed paired V13 relation-only causal comparison."""

    commitments = software_pipeline_mechanism_partition("train")[
        :_RELATION_CREDIT_COMMITMENTS
    ]
    if len(commitments) != _RELATION_CREDIT_COMMITMENTS:
        raise RuntimeError("train partition cannot satisfy the cluster pilot")
    replicates = []
    all_train_pairs: set[tuple[int, int]] = set()
    all_panel_pairs: set[tuple[int, int]] = set()
    for replicate, seeds in enumerate(_CLUSTER_REPLICATE_SEEDS):
        offset = _CLUSTER_REPLICATE_SEED_STRIDE * replicate
        train_batches = tuple(
            tuple(
                (
                    _CLUSTER_TRAIN_TOPOLOGY_BASE
                    + offset
                    + 100_000 * update
                    + 1_000 * stream,
                    _CLUSTER_TRAIN_SURFACE_BASE
                    + offset
                    + 100_000 * update
                    + 1_000 * stream,
                )
                for stream in range(_RELATION_CREDIT_STREAMS_PER_UPDATE)
            )
            for update in range(_RELATION_CREDIT_STAGE_UPDATES["relation"])
        )
        panel_a = _relation_credit_panel_seed_pairs(
            _CLUSTER_PANEL_A_TOPOLOGY_BASE + offset,
            _CLUSTER_PANEL_A_SURFACE_BASE + offset,
        )
        panel_a_rerender = _relation_credit_panel_seed_pairs(
            _CLUSTER_PANEL_A_TOPOLOGY_BASE + offset,
            _CLUSTER_PANEL_A_RERENDER_SURFACE_BASE + offset,
        )
        panel_b = _relation_credit_panel_seed_pairs(
            _CLUSTER_PANEL_B_TOPOLOGY_BASE + offset,
            _CLUSTER_PANEL_B_SURFACE_BASE + offset,
        )
        train_pairs = {pair for batch in train_batches for pair in batch}
        panel_pairs = set(panel_a) | set(panel_a_rerender) | set(panel_b)
        if (
            len(train_pairs)
            != _RELATION_CREDIT_STAGE_UPDATES["relation"]
            * _RELATION_CREDIT_STREAMS_PER_UPDATE
            or len(panel_pairs) != 3 * _RELATION_CREDIT_COMMITMENTS
            or train_pairs & panel_pairs
            or all_train_pairs & train_pairs
            or all_panel_pairs & panel_pairs
            or all_train_pairs & panel_pairs
            or all_panel_pairs & train_pairs
        ):
            raise RuntimeError("cluster replicate identities overlap")
        all_train_pairs.update(train_pairs)
        all_panel_pairs.update(panel_pairs)
        binding = _cluster_stream_binding_digest(commitments, train_batches)
        replicates.append(
            {
                "replicate": replicate,
                "shared_controller_seed": seeds[0],
                "cluster_cell_seed": seeds[1],
                "cluster_composer_seed": seeds[2],
                "common_mixer_seed": seeds[3],
                "arm_order": (
                    ("monolith", "cluster")
                    if replicate % 2 == 0
                    else ("cluster", "monolith")
                ),
                "train_seed_batches": train_batches,
                "panel_a_seed_pairs": panel_a,
                "panel_a_rerender_seed_pairs": panel_a_rerender,
                "panel_b_seed_pairs": panel_b,
                "monolith_stream_binding_digest": binding,
                "cluster_stream_binding_digest": binding,
            }
        )
    v12 = public_relation_conflict_fit_plan()
    v12_pairs = {
        pair
        for batches in v12["stage_seed_batches"].values()
        for batch in batches
        for pair in batch
    } | set(v12["relation_context_panel_seed_pairs"]) | set(
        v12["final_panel_seed_pairs"]
    )
    if (all_train_pairs | all_panel_pairs) & v12_pairs:
        raise RuntimeError("cluster pilot overlaps V12 identities")
    return {
        "protocol_id": _CLUSTER_PROTOCOL_ID,
        "partition": "train",
        "replicate_count": len(replicates),
        "replicates": tuple(replicates),
        "commitments": commitments,
        "stage": "relation",
        "updates_per_arm_per_replicate": _RELATION_CREDIT_STAGE_UPDATES[
            "relation"
        ],
        "streams_per_update": _RELATION_CREDIT_STREAMS_PER_UPDATE,
        "rows_per_stream": 4,
        "streams_per_arm_per_replicate": (
            _RELATION_CREDIT_STAGE_UPDATES["relation"]
            * _RELATION_CREDIT_STREAMS_PER_UPDATE
        ),
        "rows_per_arm_per_replicate": (
            _RELATION_CREDIT_STAGE_UPDATES["relation"]
            * _RELATION_CREDIT_STREAMS_PER_UPDATE
            * 4
        ),
        "optimizer_and_updater": "identical_anonymous_conflict_mixer_v12_interface",
        "monolith_parameter_blocks": (
            "pair_encoder",
            "global_readout",
            "incidence_readout",
            "incidence_projection",
            "comparator",
        ),
        "cluster_parameter_blocks": (
            "cell_0",
            "cell_1",
            "cell_2",
            "cell_3",
            "composer",
        ),
        "cell_count": _CLUSTER_CELL_COUNT,
        "cell_width": _CLUSTER_CELL_WIDTH,
        "composer": "learned_permutation_invariant_all_active_soft_fusion",
        "minimum_cell_weight": (
            _CLUSTER_COMPOSER_ANCHOR_WEIGHT / _CLUSTER_CELL_COUNT
        ),
        "context_or_joint_training": False,
        "early_stopping": False,
        "adaptive_rerun": False,
        "historical_v12_score_comparison": False,
        "v12_checkpoint_reuse": False,
        "stream_sharding": False,
        "fixed_cell_roles": False,
        "voting": False,
        "deterministic_solver": False,
        "support_rule": {
            "aggregate_supported_rows": "cluster_strictly_greater",
            "aggregate_qualifying_streams": "cluster_at_least_monolith",
            "mean_target_loss": "cluster_strictly_lower",
            "non_regressing_replicates": 2,
            "learned_fusion_dominates_every_lesion": True,
            "surface_discrete_exact": True,
            "surface_continuous_max_delta": 1.0e-6,
            "all_cell_and_composer_digests_change": True,
        },
    }


def _public_static_contract_fields(
    component: PublicComponentContract,
) -> tuple[object, ...]:
    return (
        component.input_type,
        component.output_type,
        component.error_type,
        component.state_reads,
        component.state_writes,
    )


def _same_contract_alternative_index(
    components: Sequence[PublicComponentContract],
    observed_index: int,
) -> int | None:
    if not 0 <= observed_index < len(components):
        raise ValueError("observed component index is outside the public candidates")
    fields = _public_static_contract_fields(components[observed_index])
    alternatives = tuple(
        index
        for index, component in enumerate(components)
        if index != observed_index and _public_static_contract_fields(component) == fields
    )
    if len(alternatives) > 1:
        raise RuntimeError("public trace has more than one same-contract alternative")
    return alternatives[0] if alternatives else None


def _relation_credit_task(
    controller: SoftwarePipelineController,
    task: PublicSoftwarePipelineTask,
) -> tuple[
    tuple[Transition, ...],
    tuple[int, ...],
    tuple[int | None, ...],
    torch.Tensor,
    torch.Tensor,
]:
    """Encode only public candidates and trace actions for credit assignment."""

    _validate_public_task(task)
    components = _components_in_candidate_order(task)
    context_codes, relation_codes = controller._factorized_relation_embeddings(
        components,
        controller.procedure_start,
    )
    transitions = _public_transitions(task)
    if not transitions:
        raise ValueError("relation credit requires a public observation")
    observed_indices = tuple(
        _action_index(task.grounded_candidates, transition.action)
        for transition in transitions
    )
    alternative_indices = tuple(
        _same_contract_alternative_index(components, index)
        for index in observed_indices
    )
    return (
        transitions,
        observed_indices,
        alternative_indices,
        context_codes,
        relation_codes,
    )


def _relation_instance_losses(
    positive_margins: torch.Tensor,
    negative_margins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-slot loss, normalized soft minimum, and detached credit."""

    if (
        positive_margins.ndim != 1
        or positive_margins.shape != negative_margins.shape
        or positive_margins.numel() == 0
        or positive_margins.device != negative_margins.device
        or positive_margins.dtype != negative_margins.dtype
        or not positive_margins.is_floating_point()
        or not bool(torch.isfinite(positive_margins).all().item())
        or not bool(torch.isfinite(negative_margins).all().item())
    ):
        raise ValueError("relation instance margins must be finite aligned vectors")
    margin = positive_margins.new_tensor(_RELATION_FIT_MARGIN)
    temperature = positive_margins.new_tensor(_RELATION_CREDIT_TEMPERATURE)
    losses = (
        temperature * F.softplus((margin - positive_margins) / temperature)
        + temperature * F.softplus((margin + negative_margins) / temperature)
        + _RELATION_FIT_BALANCE_WEIGHT
        * F.smooth_l1_loss(
            positive_margins,
            -negative_margins,
            reduction="none",
        )
    )
    instance_loss = -temperature * (
        torch.logsumexp(-losses / temperature, dim=0)
        - math.log(losses.numel())
    )
    responsibilities = torch.softmax(
        -losses.detach() / _RELATION_CONTEXT_TARGET_TEMPERATURE,
        dim=0,
    )
    return losses, instance_loss, responsibilities


def _relation_valid_set_metrics(
    slot_positive_margins: torch.Tensor,
    slot_negative_margins: torch.Tensor,
    context_weights: torch.Tensor,
    context_null_weight: torch.Tensor,
) -> dict[str, object]:
    """Measure an order-invariant set of sign-valid public witnesses."""

    if (
        slot_positive_margins.ndim != 1
        or slot_positive_margins.shape != slot_negative_margins.shape
        or slot_positive_margins.shape != context_weights.shape
        or slot_positive_margins.numel() == 0
        or slot_positive_margins.device != slot_negative_margins.device
        or slot_positive_margins.device != context_weights.device
        or slot_positive_margins.device != context_null_weight.device
        or slot_positive_margins.dtype != slot_negative_margins.dtype
        or slot_positive_margins.dtype != context_weights.dtype
        or slot_positive_margins.dtype != context_null_weight.dtype
        or not slot_positive_margins.is_floating_point()
        or context_null_weight.shape != ()
        or not bool(torch.isfinite(slot_positive_margins).all().item())
        or not bool(torch.isfinite(slot_negative_margins).all().item())
        or not bool(torch.isfinite(context_weights).all().item())
        or not bool(torch.isfinite(context_null_weight).item())
        or bool((context_weights < 0.0).any().item())
        or bool((context_null_weight < 0.0).item())
        or abs(
            float((context_weights.sum() + context_null_weight).item()) - 1.0
        )
        > 1.0e-6
    ):
        raise ValueError("valid-set metrics require aligned finite public slots")
    witnesses = slot_positive_margins - slot_negative_margins
    valid_mask = (
        (slot_positive_margins >= 0.05)
        & (slot_negative_margins <= -0.05)
        & (witnesses >= _RELATION_FIT_MARGIN)
    )
    valid_count = int(valid_mask.sum().item())
    null_mass = float(context_null_weight.item())
    if valid_count:
        set_mass = float(context_weights.masked_select(valid_mask).sum().item())
        valid_max = float(context_weights.masked_select(valid_mask).max().item())
        invalid_weights = context_weights.masked_select(~valid_mask)
        invalid_max = (
            float(invalid_weights.max().item())
            if invalid_weights.numel()
            else -math.inf
        )
        competing_max = max(invalid_max, null_mass)
        top_one = valid_max > competing_max
    else:
        set_mass = 0.0
        valid_max = -math.inf
        competing_max = max(float(context_weights.max().item()), null_mass)
        top_one = False
    return {
        "valid_mask": valid_mask,
        "valid_slot_count": valid_count,
        "relation_supported": valid_count > 0,
        "context_null_mass": null_mass,
        "context_valid_set_mass": set_mass,
        "context_valid_max": valid_max,
        "context_competing_max": competing_max,
        "context_valid_set_top_one": top_one,
    }


def _context_valid_set_training_term(
    row: PublicRelationCreditRow,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    """Return the current C-only detached valid-set likelihood and diagnostics."""

    if not isinstance(row, PublicRelationCreditRow):
        raise TypeError("valid-set context training requires a public credit row")
    metrics = _relation_valid_set_metrics(
        row.slot_positive_margins.detach(),
        row.slot_negative_margins.detach(),
        row.context_weights,
        row.context_null_weight,
    )
    valid_mask = metrics["valid_mask"]
    assert isinstance(valid_mask, torch.Tensor)
    supported = metrics["relation_supported"] is True
    if not supported:
        return None, {
            "supported": False,
            "valid_slot_count": 0,
        }
    valid_mass = row.context_weights.masked_select(valid_mask).sum()
    loss = -torch.log(
        valid_mass.clamp_min(torch.finfo(valid_mass.dtype).tiny)
    )
    responsibility_mass = row.responsibilities.masked_select(valid_mask).sum()
    responsibility_argmax = int(torch.argmax(row.responsibilities).item())
    real_mass = (row.context_weights.sum()).clamp_min(
        torch.finfo(row.context_weights.dtype).tiny
    )
    return loss, {
        "supported": True,
        "valid_slot_count": metrics["valid_slot_count"],
        "responsibility_valid_set_mass": float(responsibility_mass.item()),
        "responsibility_argmax_in_valid_set": bool(
            valid_mask[responsibility_argmax].item()
        ),
        "context_null_mass": metrics["context_null_mass"],
        "context_valid_set_mass": float(valid_mass.detach().item()),
        "context_valid_set_real_normalized_mass": float(
            (valid_mass.detach() / real_mass.detach()).item()
        ),
        "context_valid_set_top_one": metrics["context_valid_set_top_one"],
    }


def public_relation_credit_rows(
    controller: SoftwarePipelineController,
    stream: SoftwarePipelineStream,
    *,
    reverse_evidence_order: bool = False,
    reverse_public_presentation: bool = False,
) -> tuple[PublicRelationCreditRow, ...]:
    """Build public observed/alternative slot credit without a runtime control arm."""

    if not isinstance(stream, SoftwarePipelineStream):
        raise TypeError("relation credit requires a SoftwarePipelineStream")
    if type(reverse_evidence_order) is not bool or type(reverse_public_presentation) is not bool:
        raise TypeError("relation credit covariance flags must be bool")
    if stream.mechanism_partition != "train" or stream.control_arm != "correct":
        raise ValueError("relation credit accepts only original train streams")
    if len(stream.supports) != 4:
        raise ValueError("relation credit requires four public support packages")
    tasks = []
    for pair in stream.supports:
        task = pair.learner
        if reverse_public_presentation:
            task = replace(
                task,
                components=tuple(reversed(task.components)),
                grounded_candidates=tuple(reversed(task.grounded_candidates)),
                states=tuple(reversed(task.states)),
            )
        tasks.append(task)
    encoded_tasks = tuple(
        _relation_credit_task(controller, task) for task in tasks
    )
    rows = []
    for heldout_index, encoded_query in enumerate(encoded_tasks):
        (
            query_transitions,
            query_observed,
            query_alternatives,
            query_context_codes,
            query_relation_codes,
        ) = encoded_query
        discriminating = tuple(
            index
            for index, alternative in enumerate(query_alternatives)
            if alternative is not None
        )
        if len(discriminating) != 1:
            raise RuntimeError("each public support must expose one declared contrast")
        transition_index = discriminating[0]
        positive_index = query_observed[transition_index]
        negative_index = query_alternatives[transition_index]
        if negative_index is None:
            raise AssertionError("validated public contrast disappeared")
        if not torch.equal(
            query_context_codes[positive_index],
            query_context_codes[negative_index],
        ):
            raise RuntimeError("same-contract query alternatives changed context")
        pair_indices = torch.tensor(
            (positive_index, negative_index),
            device=query_relation_codes.device,
            dtype=torch.long,
        )
        query_relations = query_relation_codes.index_select(0, pair_indices)
        evidence_indices = [
            index for index in range(len(encoded_tasks)) if index != heldout_index
        ]
        if reverse_evidence_order:
            evidence_indices.reverse()
        stored_contexts = []
        positive_values = []
        negative_values = []
        changed_values = []
        for evidence_index in evidence_indices:
            (
                evidence_transitions,
                evidence_observed,
                evidence_alternatives,
                evidence_context_codes,
                evidence_relation_codes,
            ) = encoded_tasks[evidence_index]
            if not (
                len(evidence_transitions)
                == len(evidence_observed)
                == len(evidence_alternatives)
            ):
                raise RuntimeError("public evidence transitions lost alignment")
            for observed_index, alternative_index in zip(
                evidence_observed,
                evidence_alternatives,
                strict=True,
            ):
                stored_contexts.append(evidence_context_codes[observed_index])
                positive_values.append(evidence_relation_codes[observed_index])
                if alternative_index is None:
                    negative_values.append(evidence_relation_codes[observed_index])
                    changed_values.append(False)
                else:
                    negative_values.append(evidence_relation_codes[alternative_index])
                    changed_values.append(True)
        context_matrix = torch.stack(stored_contexts)
        positive_matrix = torch.stack(positive_values)
        negative_matrix = torch.stack(negative_values)
        positive_present = (
            (context_matrix.norm(dim=-1) > 1.0e-8)
            & (positive_matrix.norm(dim=-1) > 1.0e-8)
        )
        negative_present = (
            (context_matrix.norm(dim=-1) > 1.0e-8)
            & (negative_matrix.norm(dim=-1) > 1.0e-8)
        )
        if not torch.equal(positive_present, negative_present):
            raise RuntimeError("public alternatives changed occupied relation slots")
        if not bool(positive_present.any().item()):
            raise RuntimeError("public credit has no transferable relation slots")
        context_matrix = context_matrix[positive_present]
        positive_matrix = positive_matrix[positive_present]
        negative_matrix = negative_matrix[positive_present]
        changed_values = [
            changed
            for changed, present in zip(
                changed_values,
                positive_present.tolist(),
                strict=True,
            )
            if present
        ]
        positive_values = [
            value
            for value, present in zip(
                positive_values,
                positive_present.tolist(),
                strict=True,
            )
            if present
        ]
        negative_values = [
            value
            for value, present in zip(
                negative_values,
                positive_present.tolist(),
                strict=True,
            )
            if present
        ]
        (
            all_positive_scores,
            all_context_weights,
            all_context_null_weights,
            all_positive_logits,
        ) = controller._relation_evidence_read(
            query_context_codes,
            query_relation_codes,
            context_matrix,
            positive_matrix,
        )
        (
            all_negative_scores,
            negative_context_weights,
            negative_context_null_weights,
            all_negative_logits,
        ) = controller._relation_evidence_read(
            query_context_codes,
            query_relation_codes,
            context_matrix,
            negative_matrix,
        )
        if not torch.equal(all_context_weights, negative_context_weights):
            raise RuntimeError("relation alternatives changed public context weights")
        if not torch.equal(
            all_context_null_weights,
            negative_context_null_weights,
        ):
            raise RuntimeError("relation alternatives changed public null weights")
        positive_logits = all_positive_logits.index_select(0, pair_indices)
        negative_logits = all_negative_logits.index_select(0, pair_indices)
        slot_positive = positive_logits[0] - positive_logits[1]
        slot_negative = negative_logits[0] - negative_logits[1]
        slot_losses, instance_loss, responsibilities = _relation_instance_losses(
            slot_positive,
            slot_negative,
        )
        pair_context_weights = all_context_weights.index_select(0, pair_indices)
        if not torch.allclose(
            pair_context_weights[0],
            pair_context_weights[1],
            atol=_RELATION_GATE_PERMUTATION_TOLERANCE,
            rtol=0.0,
        ):
            raise RuntimeError("same-contract query alternatives changed context weights")
        context_weights = pair_context_weights[0]
        pair_context_null_weights = all_context_null_weights.index_select(
            0,
            pair_indices,
        )
        if not torch.allclose(
            pair_context_null_weights[0],
            pair_context_null_weights[1],
            atol=_RELATION_GATE_PERMUTATION_TOLERANCE,
            rtol=0.0,
        ):
            raise RuntimeError("same-contract query alternatives changed null weight")
        context_null_weight = pair_context_null_weights[0]
        context_loss = -(
            responsibilities
            * torch.log(context_weights.clamp_min(torch.finfo(context_weights.dtype).tiny))
        ).sum()
        positive_scores = all_positive_scores.index_select(0, pair_indices)
        negative_scores = all_negative_scores.index_select(0, pair_indices)
        positive_margin = positive_scores[0] - positive_scores[1]
        negative_margin = negative_scores[0] - negative_scores[1]
        aggregate_loss = _paired_relation_margin_loss(
            positive_margin,
            negative_margin,
        )
        separation_terms = [
            F.relu(
                query_relations.new_tensor(_RELATION_FIT_MARGIN)
                - (1.0 - F.cosine_similarity(
                    query_relations[0],
                    query_relations[1],
                    dim=0,
                ))
            )
        ]
        for changed, positive_value, negative_value in zip(
            changed_values,
            positive_values,
            negative_values,
            strict=True,
        ):
            if changed:
                separation_terms.append(
                    F.relu(
                        positive_value.new_tensor(_RELATION_FIT_MARGIN)
                        - (1.0 - F.cosine_similarity(
                            positive_value,
                            negative_value,
                            dim=0,
                        ))
                    )
                )
        separation_loss = torch.stack(separation_terms).mean()
        joint_loss = (
            aggregate_loss
            + _RELATION_CREDIT_CONTEXT_WEIGHT * context_loss
            + _RELATION_CREDIT_INSTANCE_WEIGHT * instance_loss
            + _RELATION_CREDIT_SEPARATION_WEIGHT * separation_loss
        )
        rows.append(
            PublicRelationCreditRow(
                heldout_index=heldout_index,
                transition_index=transition_index,
                positive_index=positive_index,
                negative_index=negative_index,
                positive_margin=positive_margin,
                negative_margin=negative_margin,
                instance_loss=instance_loss,
                context_loss=context_loss,
                separation_loss=separation_loss,
                joint_loss=joint_loss,
                slot_losses=slot_losses,
                slot_positive_margins=slot_positive,
                slot_negative_margins=slot_negative,
                responsibilities=responsibilities,
                context_weights=context_weights,
                context_null_weight=context_null_weight,
            )
        )
    if len(rows) != 4:
        raise RuntimeError("relation credit stream must yield four whole-trace rows")
    return tuple(rows)


def _acquire_public_task_set(
    controller: SoftwarePipelineController,
    tasks: Sequence[PublicSoftwarePipelineTask],
) -> SoftwareReconstructionState:
    state = controller.initial_state()
    for task in tasks:
        state = acquire_public_pipeline_traces(controller, task, state).state
    return state


def _paired_public_relation_folds(
    controller: SoftwarePipelineController,
    stream: SoftwarePipelineStream,
    *,
    reverse_evidence_order: bool = False,
    reverse_public_presentation: bool = False,
) -> tuple[_PublicRelationFold, ...]:
    """Build aligned whole-fold positive/negative public evidence states."""

    if not isinstance(stream, SoftwarePipelineStream):
        raise TypeError("relation fit requires a SoftwarePipelineStream")
    if type(reverse_evidence_order) is not bool or type(reverse_public_presentation) is not bool:
        raise TypeError("relation covariance flags must be bool")
    if stream.mechanism_partition != "train" or stream.control_arm != "correct":
        raise ValueError("relation fit accepts only original train streams")
    if len(stream.supports) != 4:
        raise ValueError("relation fit requires four public support packages")
    wrong = make_software_pipeline_control_stream(stream, "wrong_evidence")
    if len(wrong.supports) != len(stream.supports):
        raise RuntimeError("wrong-evidence control lost support alignment")
    folds = []
    for heldout_index, (positive_pair, negative_pair) in enumerate(
        zip(stream.supports, wrong.supports, strict=True)
    ):
        positive_task = positive_pair.learner
        negative_task = negative_pair.learner
        masked_positive = replace(positive_task, observations=())
        masked_negative = replace(negative_task, observations=())
        if masked_positive != masked_negative:
            raise RuntimeError("counterfactual heldout tasks differ outside observations")
        evidence_indices = [
            index for index in range(len(stream.supports)) if index != heldout_index
        ]
        if reverse_evidence_order:
            evidence_indices.reverse()
        positive_state = _acquire_public_task_set(
            controller,
            tuple(stream.supports[index].learner for index in evidence_indices),
        )
        negative_state = _acquire_public_task_set(
            controller,
            tuple(wrong.supports[index].learner for index in evidence_indices),
        )
        positive_transitions = _public_transitions(positive_task)
        negative_transitions = _public_transitions(negative_task)
        if len(positive_transitions) != len(negative_transitions):
            raise RuntimeError("counterfactual traces lost transition alignment")
        changed = 0
        for transition_index, (positive, negative) in enumerate(
            zip(positive_transitions, negative_transitions, strict=True)
        ):
            if positive.action == negative.action:
                continue
            changed += 1
            if (
                positive.before != negative.before
                or positive.after != negative.after
                or positive.applied != negative.applied
                or positive.outcome != negative.outcome
            ):
                raise RuntimeError("counterfactual transition changed public outcomes")
            positive_component = _components_in_candidate_order(positive_task)[
                _action_index(positive_task.grounded_candidates, positive.action)
            ]
            negative_component = _components_in_candidate_order(negative_task)[
                _action_index(negative_task.grounded_candidates, negative.action)
            ]
            if _public_static_contract_fields(
                positive_component
            ) != _public_static_contract_fields(negative_component):
                raise RuntimeError("counterfactual actions changed static contracts")
            masked = masked_positive
            if reverse_public_presentation:
                masked = replace(
                    masked,
                    components=tuple(reversed(masked.components)),
                    grounded_candidates=tuple(reversed(masked.grounded_candidates)),
                    states=tuple(reversed(masked.states)),
                )
            folds.append(
                _PublicRelationFold(
                    heldout_index=heldout_index,
                    transition_index=transition_index,
                    masked_task=masked,
                    before=positive.before,
                    positive_action=positive.action,
                    negative_action=negative.action,
                    positive_state=positive_state,
                    negative_state=negative_state,
                )
            )
        if changed != 1:
            raise RuntimeError("each heldout fold must expose one changed public action")
    return tuple(folds)


def public_paired_relation_fit_rows(
    controller: SoftwarePipelineController,
    stream: SoftwarePipelineStream,
    *,
    reverse_evidence_order: bool = False,
    reverse_public_presentation: bool = False,
) -> tuple[PublicRelationFitRow, ...]:
    """Return direct, undiluted relation losses for balanced public arms.

    ``positive_margin`` is original-evidence(original action minus swapped
    action). ``negative_margin`` uses that same action ordering under swapped
    evidence.  Thus successful acquisition makes the first positive and the
    second negative; the symmetric loss compares their opposite signs.
    """

    rows = []
    for fold in _paired_public_relation_folds(
        controller,
        stream,
        reverse_evidence_order=reverse_evidence_order,
        reverse_public_presentation=reverse_public_presentation,
    ):
        encoded = controller.encode_task(fold.masked_task)
        positive_scores = controller._relation_evidence_scores(
            encoded.relation_context_embeddings,
            encoded.relation_component_embeddings,
            fold.positive_state,
        )
        negative_scores = controller._relation_evidence_scores(
            encoded.relation_context_embeddings,
            encoded.relation_component_embeddings,
            fold.negative_state,
        )
        positive_index = _action_index(
            fold.masked_task.grounded_candidates,
            fold.positive_action,
        )
        negative_index = _action_index(
            fold.masked_task.grounded_candidates,
            fold.negative_action,
        )
        positive_margin = positive_scores[positive_index] - positive_scores[negative_index]
        negative_margin = negative_scores[positive_index] - negative_scores[negative_index]
        loss = _paired_relation_margin_loss(positive_margin, negative_margin)
        rows.append(
            PublicRelationFitRow(
                heldout_index=fold.heldout_index,
                transition_index=fold.transition_index,
                positive_margin=positive_margin,
                negative_margin=negative_margin,
                loss=loss,
            )
        )
    if len(rows) != 4:
        raise RuntimeError("relation fit stream must yield four whole-fold rows")
    return tuple(rows)


def _paired_relation_margin_loss(
    positive_margin: torch.Tensor,
    negative_margin: torch.Tensor,
) -> torch.Tensor:
    if (
        positive_margin.shape != ()
        or negative_margin.shape != ()
        or positive_margin.device != negative_margin.device
        or positive_margin.dtype != negative_margin.dtype
        or not positive_margin.is_floating_point()
        or not bool(torch.isfinite(positive_margin).item())
        or not bool(torch.isfinite(negative_margin).item())
    ):
        raise ValueError("paired relation margins must be finite matching scalars")
    margin = positive_margin.new_tensor(_RELATION_FIT_MARGIN)
    return (
        F.relu(margin - positive_margin)
        + F.relu(margin + negative_margin)
        + _RELATION_FIT_BALANCE_WEIGHT
        * F.smooth_l1_loss(positive_margin, -negative_margin)
    )


def _relation_matcher_parameter_names(
    controller: SoftwarePipelineController,
) -> tuple[str, ...]:
    prefixes = (
        "evidence_pair_encoder.",
        "evidence_context_encoder.",
        "relation_pool_attention.",
        "relation_pool_projection.",
        "relation_incidence_readout.",
        "relation_incidence_projection.",
        "relation_comparator.",
        "relation_context_pool_attention.",
        "relation_context_pool_projection.",
        "relation_context_comparator.",
    )
    return tuple(
        name
        for name, _ in controller.named_parameters()
        if name.startswith(prefixes)
    )


def _relation_credit_parameter_names(
    controller: SoftwarePipelineController,
    stage: str,
) -> tuple[str, ...]:
    if isinstance(controller, CapacityMatchedClusterController):
        relation_prefixes = (
            "relation_cells.",
            "relation_composer.",
        )
    else:
        relation_prefixes = (
            "evidence_pair_encoder.",
            "relation_pool_attention.",
            "relation_pool_projection.",
            "relation_incidence_readout.",
            "relation_incidence_projection.",
            "relation_comparator.",
        )
    context_prefixes = (
        "evidence_context_encoder.",
        "relation_context_pool_attention.",
        "relation_context_pool_projection.",
        "relation_context_comparator.",
    )
    if stage == "relation":
        prefixes = relation_prefixes
    elif stage == "context":
        prefixes = context_prefixes
    elif stage == "joint":
        prefixes = relation_prefixes + context_prefixes
    else:
        raise ValueError("relation-credit stage must be relation, context, or joint")
    return tuple(
        name
        for name, _ in controller.named_parameters()
        if name.startswith(prefixes)
    )


def _relation_encoder_parameter_name(name: str) -> bool:
    return name.startswith(
        ("evidence_pair_encoder.", "evidence_context_encoder.")
    ) or (name.startswith("relation_cells.") and ".pair_encoder." in name)


def _conflict_parameter_blocks(
    controller: SoftwarePipelineController,
    stage: str,
) -> dict[str, tuple[str, ...]]:
    """Partition every mutable v12 tensor into an anonymous update block."""

    selected = set(_relation_credit_parameter_names(controller, stage))
    if isinstance(controller, CapacityMatchedClusterController):
        if stage not in ("relation", "joint"):
            raise ValueError("cluster conflict reconciliation is relation-only")
        blocks = {
            **{
                f"cell_{index}": tuple(
                    name
                    for name, _ in controller.named_parameters()
                    if name in selected
                    and name.startswith(f"relation_cells.{index}.")
                )
                for index in range(_CLUSTER_CELL_COUNT)
            },
            "composer": tuple(
                name
                for name, _ in controller.named_parameters()
                if name in selected and name.startswith("relation_composer.")
            ),
        }
        if stage == "joint":
            blocks["context"] = tuple(
                name
                for name, _ in controller.named_parameters()
                if name in selected
                and name.startswith(
                    (
                        "evidence_context_encoder.",
                        "relation_context_pool_attention.",
                        "relation_context_pool_projection.",
                        "relation_context_comparator.",
                    )
                )
            )
        flattened = tuple(name for names in blocks.values() for name in names)
        if (
            any(not names for names in blocks.values())
            or len(flattened) != len(set(flattened))
            or set(flattened) != selected
        ):
            raise RuntimeError("cluster blocks do not exactly partition the stage")
        return blocks
    if isinstance(controller, CapacityMatchedMonolithController):
        prefix_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("pair_encoder", ("evidence_pair_encoder.",)),
            (
                "global_readout",
                ("relation_pool_attention.", "relation_pool_projection."),
            ),
            ("incidence_readout", ("relation_incidence_readout.",)),
            ("incidence_projection", ("relation_incidence_projection.",)),
            ("comparator", ("relation_comparator.",)),
        )
        if stage == "joint":
            prefix_groups += (
                (
                    "context",
                    (
                        "evidence_context_encoder.",
                        "relation_context_pool_attention.",
                        "relation_context_pool_projection.",
                        "relation_context_comparator.",
                    ),
                ),
            )
        elif stage != "relation":
            raise ValueError("paired monolith reconciliation is relation-only")
        blocks = {
            block: tuple(
                name
                for name, _ in controller.named_parameters()
                if name in selected and name.startswith(prefixes)
            )
            for block, prefixes in prefix_groups
        }
        flattened = tuple(name for names in blocks.values() for name in names)
        if (
            any(not names for names in blocks.values())
            or len(flattened) != len(set(flattened))
            or set(flattened) != selected
        ):
            raise RuntimeError("paired monolith blocks do not exactly partition")
        return blocks
    prefix_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("pair_encoder", ("evidence_pair_encoder.",)),
        (
            "global_readout",
            ("relation_pool_attention.", "relation_pool_projection."),
        ),
        (
            "incidence_readout",
            ("relation_incidence_readout.", "relation_incidence_projection."),
        ),
        ("comparator", ("relation_comparator.",)),
    )
    if stage == "joint":
        prefix_groups += (
            (
                "context",
                (
                    "evidence_context_encoder.",
                    "relation_context_pool_attention.",
                    "relation_context_pool_projection.",
                    "relation_context_comparator.",
                ),
            ),
        )
    elif stage != "relation":
        raise ValueError("conflict reconciliation applies only to relation or joint")
    blocks = {
        block: tuple(
            name
            for name, _ in controller.named_parameters()
            if name in selected and name.startswith(prefixes)
        )
        for block, prefixes in prefix_groups
    }
    flattened = tuple(name for names in blocks.values() for name in names)
    if (
        any(not names for names in blocks.values())
        or len(flattened) != len(set(flattened))
        or set(flattened) != selected
    ):
        raise RuntimeError("conflict parameter blocks do not exactly partition stage")
    return blocks


def _conflict_gradient_geometry(
    stream_gradients: Sequence[Sequence[torch.Tensor]],
    parameter_names: Sequence[str],
    blocks: Mapping[str, Sequence[str]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return block norms/cosines and flattened detached stream gradients."""

    if (
        not stream_gradients
        or not parameter_names
        or any(len(row) != len(parameter_names) for row in stream_gradients)
    ):
        raise ValueError("conflict geometry requires aligned stream gradients")
    name_to_index = {name: index for index, name in enumerate(parameter_names)}
    if len(name_to_index) != len(parameter_names):
        raise ValueError("conflict geometry parameter names must be unique")
    matrices = []
    norms = []
    grams = []
    for names in blocks.values():
        indices = tuple(name_to_index[name] for name in names)
        matrix = torch.stack(
            tuple(
                torch.cat(
                    tuple(row[index].detach().reshape(-1) for index in indices)
                )
                for row in stream_gradients
            )
        )
        block_norms = matrix.norm(dim=-1)
        normalized = matrix / block_norms.clamp_min(1.0e-12).unsqueeze(-1)
        gram = normalized @ normalized.transpose(0, 1)
        zero = block_norms <= 1.0e-12
        if bool(zero.any().item()):
            gram = gram.masked_fill(zero.unsqueeze(0) | zero.unsqueeze(1), 0.0)
        matrices.append(matrix)
        norms.append(block_norms)
        grams.append(gram.clamp(min=-1.0, max=1.0))
    return torch.stack(norms), torch.stack(grams), tuple(matrices)


def _conflict_direction_diagnostics(
    block_weights: torch.Tensor,
    gradient_norms: torch.Tensor,
    cosine_grams: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure first-order compatibility of one blockwise shared direction."""

    if (
        block_weights.shape != gradient_norms.shape
        or cosine_grams.shape
        != (
            gradient_norms.shape[0],
            gradient_norms.shape[1],
            gradient_norms.shape[1],
        )
        or not bool(torch.isfinite(block_weights).all().item())
        or not bool(torch.isfinite(gradient_norms).all().item())
        or not bool(torch.isfinite(cosine_grams).all().item())
    ):
        raise ValueError("direction diagnostics require aligned finite tensors")
    weighted_norms = block_weights * gradient_norms
    direction_squared = torch.einsum(
        "bi,bij,bj->b",
        weighted_norms,
        cosine_grams,
        weighted_norms,
    ).clamp_min(0.0)
    direction_norms = direction_squared.sqrt()
    numerators = torch.einsum(
        "bij,bj->bi",
        cosine_grams,
        weighted_norms,
    )
    alignments = torch.where(
        (gradient_norms > 1.0e-12)
        & (direction_norms.unsqueeze(-1) > 1.0e-12),
        numerators / direction_norms.clamp_min(1.0e-12).unsqueeze(-1),
        torch.zeros_like(numerators),
    ).clamp(min=-1.0, max=1.0)
    cancellation = direction_norms / weighted_norms.sum(dim=-1).clamp_min(1.0e-12)
    return {
        "stream_alignments": alignments,
        "direction_norms": direction_norms,
        "cancellation_ratios": cancellation,
        "negative_alignment_fractions": (alignments < 0.0).to(
            alignments.dtype
        ).mean(dim=-1),
    }


def _conflict_leave_one_out_meta_objective(
    mixer: AnonymousConflictMixer,
    stream_losses: torch.Tensor,
    base_weights: torch.Tensor,
    gradient_norms: torch.Tensor,
    cosine_grams: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teach the mixer from symmetric public withheld-stream consequences."""

    if stream_losses.numel() < 3:
        raise ValueError("conflict meta-objective requires at least three streams")
    alignments = []
    divergences = []
    stream_count = stream_losses.numel()
    all_indices = torch.arange(stream_count, device=stream_losses.device)
    for heldout in range(stream_count):
        keep = all_indices != heldout
        kept_losses = stream_losses[keep]
        kept_base = base_weights[keep]
        kept_base = kept_base / kept_base.sum()
        kept_norms = gradient_norms[:, keep]
        kept_grams = cosine_grams[:, keep][:, :, keep]
        proposed, _, _ = mixer(
            kept_losses,
            kept_base,
            kept_norms,
            kept_grams,
        )
        weighted_norms = proposed * kept_norms
        direction_squared = torch.einsum(
            "bi,bij,bj->b",
            weighted_norms,
            kept_grams,
            weighted_norms,
        ).clamp_min(0.0)
        stable_direction_norm = direction_squared.clamp_min(1.0e-24).sqrt()
        cross_cosines = cosine_grams[:, heldout, keep]
        numerator = (weighted_norms * cross_cosines).sum(dim=-1)
        heldout_norm = gradient_norms[:, heldout]
        alignment = torch.where(
            (heldout_norm > 1.0e-12) & (direction_squared > 1.0e-24),
            numerator / stable_direction_norm.clamp_min(1.0e-12),
            torch.zeros_like(numerator),
        ).clamp(min=-1.0, max=1.0)
        alignments.append(alignment)
        reference = kept_base.unsqueeze(0).expand_as(proposed)
        divergences.append(
            (
                proposed
                * (
                    torch.log(proposed.clamp_min(1.0e-12))
                    - torch.log(reference.clamp_min(1.0e-12))
                )
            ).sum(dim=-1)
        )
    alignment_matrix = torch.stack(alignments, dim=1)
    penalties = _CONFLICT_ALIGNMENT_TEMPERATURE * F.softplus(
        (
            alignment_matrix.new_tensor(_CONFLICT_ALIGNMENT_MARGIN)
            - alignment_matrix
        )
        / _CONFLICT_ALIGNMENT_TEMPERATURE
    )
    flat = penalties.mean()
    robust = _CONFLICT_ALIGNMENT_TEMPERATURE * (
        torch.logsumexp(
            penalties.reshape(-1) / _CONFLICT_ALIGNMENT_TEMPERATURE,
            dim=0,
        )
        - math.log(penalties.numel())
    )
    divergence = torch.stack(divergences).mean()
    objective = (
        _CONFLICT_META_MEAN_WEIGHT * flat
        + _CONFLICT_META_ROBUST_WEIGHT * robust
        + _CONFLICT_META_KL_WEIGHT * divergence
    )
    return objective, {
        "withheld_alignments": alignment_matrix,
        "alignment_penalties": penalties,
        "flat_penalty": flat,
        "robust_penalty": robust,
        "mean_kl_from_existing_weights": divergence,
    }


def _assign_conflict_block_gradients(
    parameters: Sequence[nn.Parameter],
    parameter_names: Sequence[str],
    stream_gradients: Sequence[Sequence[torch.Tensor]],
    blocks: Mapping[str, Sequence[str]],
    block_weights: torch.Tensor,
) -> None:
    """Assign one learned shared direction while retaining one model lineage."""

    if block_weights.shape != (len(blocks), len(stream_gradients)):
        raise ValueError("block weights do not match gradient partition")
    name_to_block = {
        name: block_index
        for block_index, names in enumerate(blocks.values())
        for name in names
    }
    if set(name_to_block) != set(parameter_names):
        raise RuntimeError("gradient assignment lost a mutable parameter")
    for parameter_index, (name, parameter) in enumerate(
        zip(parameter_names, parameters, strict=True)
    ):
        block_index = name_to_block[name]
        parameter.grad = torch.stack(
            tuple(
                block_weights[block_index, stream_index]
                * stream_gradients[stream_index][parameter_index]
                for stream_index in range(len(stream_gradients))
            )
        ).sum(dim=0)


def _conflict_direction_digest(
    parameters: Sequence[nn.Parameter],
    parameter_names: Sequence[str],
    stream_gradients: Sequence[Sequence[torch.Tensor]],
    blocks: Mapping[str, Sequence[str]],
    block_weights: torch.Tensor,
) -> str:
    """Digest the exact pre-clip shared direction implied by one weight matrix."""

    if (
        len(parameters) != len(parameter_names)
        or not stream_gradients
        or any(len(row) != len(parameters) for row in stream_gradients)
        or block_weights.shape != (len(blocks), len(stream_gradients))
    ):
        raise ValueError("conflict direction digest inputs do not align")
    name_to_block = {
        name: block_index
        for block_index, names in enumerate(blocks.values())
        for name in names
    }
    if set(name_to_block) != set(parameter_names):
        raise RuntimeError("conflict direction digest lost a mutable parameter")
    digest = hashlib.sha256(_CONFLICT_DIRECTION_DIGEST_DOMAIN)
    for parameter_index, name in enumerate(parameter_names):
        block_index = name_to_block[name]
        direction = torch.stack(
            tuple(
                block_weights[block_index, stream_index]
                * stream_gradients[stream_index][parameter_index]
                for stream_index in range(len(stream_gradients))
            )
        ).sum(dim=0)
        tensor = direction.detach().cpu().contiguous()
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


def _assigned_gradient_digest(
    parameters: Sequence[nn.Parameter],
    parameter_names: Sequence[str],
) -> str:
    """Digest the exact gradients that the controller optimizer will consume."""

    if len(parameters) != len(parameter_names):
        raise ValueError("assigned gradient digest inputs do not align")
    digest = hashlib.sha256(_CONFLICT_DIRECTION_DIGEST_DOMAIN)
    for name, parameter in zip(parameter_names, parameters, strict=True):
        if parameter.grad is None:
            raise RuntimeError(f"conflict update left gradient unassigned: {name}")
        tensor = parameter.grad.detach().cpu().contiguous()
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


def _conflict_weight_trace_digest(
    trace: Sequence[Sequence[Sequence[float]]],
) -> str:
    """Digest a finite post-first block-by-stream weight history."""

    normalized = tuple(
        tuple(tuple(float(value) for value in row) for row in update)
        for update in trace
    )
    if any(
        not math.isfinite(value)
        for update in normalized
        for row in update
        for value in row
    ):
        raise ValueError("conflict weight trace must be finite")
    digest = hashlib.sha256(_CONFLICT_WEIGHT_TRACE_DIGEST_DOMAIN)
    digest.update(
        json.dumps(normalized, separators=(",", ":")).encode("ascii")
    )
    return "sha256:" + digest.hexdigest()


def _anonymous_entropic_stream_objective(
    stream_losses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the normalized anonymous robust objective and its diagnostics."""

    if (
        stream_losses.ndim != 1
        or stream_losses.numel() == 0
        or not stream_losses.is_floating_point()
        or not bool(torch.isfinite(stream_losses).all().item())
    ):
        raise ValueError("anonymous stream losses must be a finite vector")
    temperature = stream_losses.new_tensor(_RELATION_CREDIT_STREAM_TEMPERATURE)
    flat_mean = stream_losses.mean()
    entropic = temperature * (
        torch.logsumexp(stream_losses / temperature, dim=0)
        - math.log(stream_losses.numel())
    )
    objective = (
        _RELATION_CREDIT_STREAM_MEAN_WEIGHT * flat_mean
        + _RELATION_CREDIT_STREAM_ROBUST_WEIGHT * entropic
    )
    gradient_weights = (
        _RELATION_CREDIT_STREAM_MEAN_WEIGHT / stream_losses.numel()
        + _RELATION_CREDIT_STREAM_ROBUST_WEIGHT
        * torch.softmax(stream_losses.detach() / temperature, dim=0)
    )
    effective_stream_count = gradient_weights.square().sum().reciprocal()
    return (
        objective,
        flat_mean,
        entropic,
        gradient_weights,
        effective_stream_count,
    )


def _anonymous_entropic_row_objective(
    row_losses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return smooth upper-tail risk over four anonymous rows in one stream."""

    if (
        row_losses.ndim != 1
        or row_losses.numel() != 4
        or not row_losses.is_floating_point()
        or not bool(torch.isfinite(row_losses).all().item())
    ):
        raise ValueError("anonymous row losses must be a finite four-vector")
    temperature = row_losses.new_tensor(_RELATION_CREDIT_ROW_TEMPERATURE)
    flat_mean = row_losses.mean()
    entropic = temperature * (
        torch.logsumexp(row_losses / temperature, dim=0)
        - math.log(row_losses.numel())
    )
    objective = (
        _RELATION_CREDIT_ROW_MEAN_WEIGHT * flat_mean
        + _RELATION_CREDIT_ROW_ROBUST_WEIGHT * entropic
    )
    gradient_weights = (
        _RELATION_CREDIT_ROW_MEAN_WEIGHT / row_losses.numel()
        + _RELATION_CREDIT_ROW_ROBUST_WEIGHT
        * torch.softmax(row_losses.detach() / temperature, dim=0)
    )
    effective_row_count = gradient_weights.square().sum().reciprocal()
    return (
        objective,
        flat_mean,
        entropic,
        gradient_weights,
        effective_row_count,
    )


def _relation_credit_stream_objective(
    stream_losses: torch.Tensor,
    *,
    stage: str,
    stream_row_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Apply robust R/J or supported-row-weighted C aggregation."""

    if stage not in ("relation", "context", "joint"):
        raise ValueError("relation-credit stage must be relation, context, or joint")
    (
        robust_objective,
        flat_mean,
        entropic,
        robust_weights,
        robust_effective_count,
    ) = _anonymous_entropic_stream_objective(stream_losses)
    robust = stage in ("relation", "joint")
    if robust:
        if stream_row_counts is not None:
            raise ValueError("robust stages do not accept context row counts")
        return (
            robust_objective,
            flat_mean,
            entropic,
            robust_weights,
            robust_effective_count,
            True,
        )
    if (
        stream_row_counts is None
        or stream_row_counts.shape != stream_losses.shape
        or stream_row_counts.device != stream_losses.device
        or stream_row_counts.dtype != stream_losses.dtype
        or not bool(torch.isfinite(stream_row_counts).all().item())
        or bool((stream_row_counts < 0.0).any().item())
        or not bool((stream_row_counts.sum() > 0.0).item())
    ):
        raise ValueError("context aggregation requires positive aligned row counts")
    context_weights = stream_row_counts / stream_row_counts.sum()
    context_objective = (context_weights * stream_losses).sum()
    return (
        context_objective,
        context_objective,
        entropic,
        context_weights,
        context_weights.square().sum().reciprocal(),
        False,
    )


def _fit_public_relation_credit_batches(
    controller: SoftwarePipelineController,
    stream_batches: Sequence[Sequence[SoftwarePipelineStream]],
    *,
    stage: str,
    encoder_learning_rate: float = 3.0e-4,
    head_learning_rate: float = 1.0e-3,
    gradient_clip: float = 5.0,
) -> dict[str, object]:
    """Fit one fixed stage of public relation credit over balanced batches."""

    if not stream_batches or any(not batch for batch in stream_batches):
        raise ValueError("relation-credit fit requires nonempty stream batches")
    for name, value in (
        ("encoder learning rate", encoder_learning_rate),
        ("head learning rate", head_learning_rate),
        ("gradient clip", gradient_clip),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    selected_names = set(_relation_credit_parameter_names(controller, stage))
    named = tuple(controller.named_parameters())
    if not selected_names:
        raise RuntimeError("relation-credit stage has no trainable parameters")
    previous_requires_grad = {
        name: parameter.requires_grad for name, parameter in named
    }
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in named
        if name not in selected_names
    }
    for name, parameter in named:
        parameter.requires_grad_(name in selected_names)
    encoder_parameters = tuple(
        parameter
        for name, parameter in named
        if name in selected_names and _relation_encoder_parameter_name(name)
    )
    head_parameters = tuple(
        parameter
        for name, parameter in named
        if name in selected_names and not _relation_encoder_parameter_name(name)
    )
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": encoder_learning_rate}
        )
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": head_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    parameters = encoder_parameters + head_parameters
    losses = []
    gradient_norms = []
    stream_loss_history = []
    stream_weight_history = []
    flat_mean_history = []
    entropic_history = []
    effective_stream_history = []
    row_loss_history = []
    row_weight_history = []
    row_flat_mean_history = []
    row_entropic_history = []
    effective_row_history = []
    context_supported_rows_per_stream_history = []
    context_supported_rows_history = []
    context_responsibility_mass_history = []
    context_responsibility_top_one_history = []
    context_null_mass_history = []
    context_valid_set_mass_history = []
    context_real_normalized_mass_history = []
    context_valid_set_top_one_history = []
    row_count = 0
    try:
        for batch in stream_batches:
            row_groups = tuple(
                public_relation_credit_rows(controller, stream) for stream in batch
            )
            rows = tuple(row for group in row_groups for row in group)
            context_row_counts = None
            context_diagnostics: tuple[dict[str, object], ...] = ()
            row_risk_diagnostics: tuple[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                ...,
            ] = ()
            if stage == "relation":
                per_stream_row_losses = tuple(
                    torch.stack(
                        tuple(
                            row.instance_loss
                            + _RELATION_CREDIT_SEPARATION_WEIGHT
                            * row.separation_loss
                            for row in group
                        )
                    )
                    for group in row_groups
                )
                row_risk_diagnostics = tuple(
                    _anonymous_entropic_row_objective(group_losses)
                    for group_losses in per_stream_row_losses
                )
                per_stream_losses = torch.stack(
                    tuple(values[0] for values in row_risk_diagnostics)
                )
            elif stage == "context":
                per_stream_terms = []
                supported_counts = []
                diagnostics = []
                for group in row_groups:
                    group_terms = []
                    for row in group:
                        term, diagnostic = _context_valid_set_training_term(row)
                        if term is not None:
                            group_terms.append(term)
                            diagnostics.append(diagnostic)
                    supported_counts.append(len(group_terms))
                    per_stream_terms.append(
                        torch.stack(tuple(group_terms)).mean()
                        if group_terms
                        else group[0].context_weights.sum() * 0.0
                    )
                per_stream_losses = torch.stack(
                    tuple(per_stream_terms)
                )
                context_row_counts = per_stream_losses.new_tensor(
                    supported_counts
                )
                context_diagnostics = tuple(diagnostics)
            else:
                per_stream_row_losses = tuple(
                    torch.stack(tuple(row.joint_loss for row in group))
                    for group in row_groups
                )
                row_risk_diagnostics = tuple(
                    _anonymous_entropic_row_objective(group_losses)
                    for group_losses in per_stream_row_losses
                )
                per_stream_losses = torch.stack(
                    tuple(values[0] for values in row_risk_diagnostics)
                )
            (
                objective,
                flat_mean,
                entropic,
                gradient_weights,
                effective_count,
                robust_applied,
            ) = _relation_credit_stream_objective(
                per_stream_losses,
                stage=stage,
                stream_row_counts=context_row_counts,
            )
            if not bool(torch.isfinite(objective).item()):
                raise RuntimeError("relation-credit objective is non-finite")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("relation-credit gradient is non-finite")
            optimizer.step()
            losses.append(float(objective.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
            stream_loss_history.append(
                tuple(float(value) for value in per_stream_losses.detach().tolist())
            )
            stream_weight_history.append(
                tuple(float(value) for value in gradient_weights.detach().tolist())
            )
            flat_mean_history.append(float(flat_mean.detach().item()))
            entropic_history.append(float(entropic.detach().item()))
            effective_stream_history.append(float(effective_count.detach().item()))
            if stage in ("relation", "joint"):
                row_loss_history.append(
                    tuple(
                        tuple(float(value) for value in group_losses.detach().tolist())
                        for group_losses in per_stream_row_losses
                    )
                )
                row_weight_history.append(
                    tuple(
                        tuple(float(value) for value in values[3].detach().tolist())
                        for values in row_risk_diagnostics
                    )
                )
                row_flat_mean_history.append(
                    tuple(float(values[1].detach().item()) for values in row_risk_diagnostics)
                )
                row_entropic_history.append(
                    tuple(float(values[2].detach().item()) for values in row_risk_diagnostics)
                )
                effective_row_history.append(
                    tuple(float(values[4].detach().item()) for values in row_risk_diagnostics)
                )
            if stage == "context":
                if not context_diagnostics:
                    raise RuntimeError("context update has no supported public rows")
                context_supported_rows_per_stream_history.append(
                    tuple(int(value) for value in context_row_counts.tolist())
                )
                context_supported_rows_history.append(len(context_diagnostics))
                context_responsibility_mass_history.append(
                    sum(
                        float(value["responsibility_valid_set_mass"])
                        for value in context_diagnostics
                    )
                    / len(context_diagnostics)
                )
                context_responsibility_top_one_history.append(
                    sum(
                        value["responsibility_argmax_in_valid_set"] is True
                        for value in context_diagnostics
                    )
                    / len(context_diagnostics)
                )
                context_null_mass_history.append(
                    sum(
                        float(value["context_null_mass"])
                        for value in context_diagnostics
                    )
                    / len(context_diagnostics)
                )
                context_valid_set_mass_history.append(
                    sum(
                        float(value["context_valid_set_mass"])
                        for value in context_diagnostics
                    )
                    / len(context_diagnostics)
                )
                context_real_normalized_mass_history.append(
                    sum(
                        float(value["context_valid_set_real_normalized_mass"])
                        for value in context_diagnostics
                    )
                    / len(context_diagnostics)
                )
                context_valid_set_top_one_history.append(
                    sum(
                        value["context_valid_set_top_one"] is True
                        for value in context_diagnostics
                    )
                    / len(context_diagnostics)
                )
            row_count += len(rows)
        after = dict(named)
        for name, before in frozen.items():
            if not torch.equal(before, after[name].detach()):
                raise RuntimeError(
                    f"relation-credit fit changed frozen parameter: {name}"
                )
    finally:
        for name, parameter in named:
            parameter.requires_grad_(previous_requires_grad[name])
    return {
        "stage": stage,
        "optimizer_steps": len(stream_batches),
        "streams": sum(len(batch) for batch in stream_batches),
        "rows": row_count,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": tuple(losses),
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "gradient_norms": tuple(gradient_norms),
        "stream_losses": tuple(stream_loss_history),
        "stream_gradient_weights": tuple(stream_weight_history),
        "flat_mean_losses": tuple(flat_mean_history),
        "entropic_terms": tuple(entropic_history),
        "effective_stream_counts": tuple(effective_stream_history),
        "row_losses": tuple(row_loss_history),
        "row_gradient_weights": tuple(row_weight_history),
        "row_flat_mean_losses": tuple(row_flat_mean_history),
        "row_entropic_terms": tuple(row_entropic_history),
        "effective_row_counts": tuple(effective_row_history),
        "context_supported_rows_per_stream": tuple(
            context_supported_rows_per_stream_history
        ),
        "context_supported_rows": tuple(context_supported_rows_history),
        "context_responsibility_valid_set_mass": tuple(
            context_responsibility_mass_history
        ),
        "context_responsibility_argmax_in_valid_fraction": tuple(
            context_responsibility_top_one_history
        ),
        "context_null_mass": tuple(context_null_mass_history),
        "context_valid_set_mass": tuple(context_valid_set_mass_history),
        "context_valid_set_real_normalized_mass": tuple(
            context_real_normalized_mass_history
        ),
        "context_valid_set_top_one_fraction": tuple(
            context_valid_set_top_one_history
        ),
        "robust_stream_objective_applied": robust_applied,
        "robust_row_objective_applied": stage in ("relation", "joint"),
        "trainable_parameter_names": tuple(sorted(selected_names)),
        "frozen_parameters_unchanged": True,
        "freshness_enforced_by_caller": False,
    }


def _fit_public_relation_conflict_batches(
    controller: SoftwarePipelineController,
    mixer: AnonymousConflictMixer,
    stream_batches: Sequence[Sequence[SoftwarePipelineStream]],
    *,
    stage: str,
    require_legacy_first_update: bool,
    encoder_learning_rate: float = 3.0e-4,
    head_learning_rate: float = 1.0e-3,
    mixer_learning_rate: float = _CONFLICT_MIXER_LEARNING_RATE,
    gradient_clip: float = 5.0,
) -> dict[str, object]:
    """Fit R/J with one learned blockwise, leave-one-stream-out updater."""

    if stage not in ("relation", "joint"):
        raise ValueError("conflict fit applies only to relation or joint")
    if not isinstance(require_legacy_first_update, bool):
        raise TypeError("require_legacy_first_update must be bool")
    if require_legacy_first_update and stage != "relation":
        raise ValueError("only the first relation stage may require a legacy start")
    if not stream_batches or any(
        len(batch) != _RELATION_CREDIT_STREAMS_PER_UPDATE
        for batch in stream_batches
    ):
        raise ValueError("conflict fit requires fixed nonempty eight-stream batches")
    for name, value in (
        ("encoder learning rate", encoder_learning_rate),
        ("head learning rate", head_learning_rate),
        ("mixer learning rate", mixer_learning_rate),
        ("gradient clip", gradient_clip),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    selected_names = set(_relation_credit_parameter_names(controller, stage))
    blocks = _conflict_parameter_blocks(controller, stage)
    named = tuple(controller.named_parameters())
    selected_named_in_model_order = tuple(
        (name, parameter)
        for name, parameter in named
        if name in selected_names
    )
    encoder_named = tuple(
        (name, parameter)
        for name, parameter in selected_named_in_model_order
        if _relation_encoder_parameter_name(name)
    )
    head_named = tuple(
        (name, parameter)
        for name, parameter in selected_named_in_model_order
        if not _relation_encoder_parameter_name(name)
    )
    # Preserve the legacy updater's parameter and global-clipping order so the
    # zero-start relation update can be compared without an ordering confound.
    selected_named = encoder_named + head_named
    selected_parameter_map = dict(selected_named)
    parameter_names = tuple(name for name, _ in selected_named)
    parameters = tuple(parameter for _, parameter in selected_named)
    if not parameters:
        raise RuntimeError("conflict fit has no controller parameters")
    previous_requires_grad = {
        name: parameter.requires_grad for name, parameter in named
    }
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in named
        if name not in selected_names
    }
    for name, parameter in named:
        parameter.requires_grad_(name in selected_names)
    encoder_parameters = tuple(parameter for _, parameter in encoder_named)
    head_parameters = tuple(parameter for _, parameter in head_named)
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": encoder_learning_rate}
        )
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": head_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    mixer_optimizer = torch.optim.AdamW(
        mixer.parameters(),
        lr=mixer_learning_rate,
        weight_decay=0.0,
    )
    mixer_before = {
        name: value.detach().clone() for name, value in mixer.state_dict().items()
    }
    mixer_initial_digest = anonymous_conflict_mixer_digest(mixer)
    if require_legacy_first_update and torch.count_nonzero(
        mixer.residual_scorer[-1].weight.detach()
    ).item() != 0:
        raise RuntimeError("v12 relation stage lost its global zero start")
    reference_losses = []
    controller_gradient_norms = []
    mixer_gradient_norms = []
    meta_losses = []
    meta_flat_penalties = []
    meta_robust_penalties = []
    meta_mean_kl_history = []
    stream_loss_history = []
    row_loss_history = []
    row_weight_history = []
    effective_row_history = []
    existing_weight_history = []
    applied_weight_history = []
    residual_logit_history = []
    withheld_alignment_history = []
    block_gradient_norm_history = []
    block_cosine_gram_history = []
    legacy_negative_alignment_history = []
    applied_negative_alignment_history = []
    legacy_cancellation_history = []
    applied_cancellation_history = []
    legacy_direction_norm_history = []
    applied_direction_norm_history = []
    legacy_direction_digests = []
    applied_direction_digests = []
    first_update_used_legacy_weights = False
    controller_step_mixer_unchanged = True
    mixer_step_controller_unchanged = True
    row_count = 0
    try:
        for update_index, batch in enumerate(stream_batches):
            row_groups = tuple(
                public_relation_credit_rows(controller, stream) for stream in batch
            )
            rows = tuple(row for group in row_groups for row in group)
            if any(len(group) != 4 for group in row_groups):
                raise RuntimeError("conflict update lost an anonymous row")
            if stage == "relation":
                per_stream_row_losses = tuple(
                    torch.stack(
                        tuple(
                            row.instance_loss
                            + _RELATION_CREDIT_SEPARATION_WEIGHT
                            * row.separation_loss
                            for row in group
                        )
                    )
                    for group in row_groups
                )
            else:
                per_stream_row_losses = tuple(
                    torch.stack(tuple(row.joint_loss for row in group))
                    for group in row_groups
                )
            row_risk = tuple(
                _anonymous_entropic_row_objective(losses)
                for losses in per_stream_row_losses
            )
            stream_loss_tensors = tuple(values[0] for values in row_risk)
            per_stream_losses = torch.stack(stream_loss_tensors)
            (
                reference_objective,
                _,
                _,
                existing_weights,
                _,
            ) = _anonymous_entropic_stream_objective(per_stream_losses)
            optimizer.zero_grad(set_to_none=True)
            stream_gradients = []
            for stream_loss in stream_loss_tensors:
                gradients = torch.autograd.grad(
                    stream_loss,
                    parameters,
                    retain_graph=(
                        require_legacy_first_update and update_index == 0
                    ),
                    create_graph=False,
                    allow_unused=True,
                )
                stream_gradients.append(
                    tuple(
                        torch.zeros_like(parameter) if gradient is None else gradient
                        for parameter, gradient in zip(
                            parameters,
                            gradients,
                            strict=True,
                        )
                    )
                )
            gradient_norms, cosine_grams, _ = _conflict_gradient_geometry(
                stream_gradients,
                parameter_names,
                blocks,
            )
            proposed_weights, residual_logits, _ = mixer(
                per_stream_losses,
                existing_weights,
                gradient_norms,
                cosine_grams,
            )
            if require_legacy_first_update and update_index == 0:
                applied_weights = existing_weights.detach().unsqueeze(0).expand(
                    len(blocks), -1
                )
                first_update_used_legacy_weights = True
            else:
                applied_weights = proposed_weights.detach()
            legacy_weights = existing_weights.detach().unsqueeze(0).expand(
                len(blocks), -1
            )
            legacy_diagnostics = _conflict_direction_diagnostics(
                legacy_weights,
                gradient_norms,
                cosine_grams,
            )
            applied_diagnostics = _conflict_direction_diagnostics(
                applied_weights,
                gradient_norms,
                cosine_grams,
            )
            legacy_direction_digests.append(
                _conflict_direction_digest(
                    parameters,
                    parameter_names,
                    stream_gradients,
                    blocks,
                    legacy_weights,
                )
            )
            meta_objective, meta_diagnostics = (
                _conflict_leave_one_out_meta_objective(
                    mixer,
                    per_stream_losses,
                    existing_weights,
                    gradient_norms,
                    cosine_grams,
                )
            )
            if not bool(torch.isfinite(meta_objective).item()):
                raise RuntimeError("conflict mixer meta-objective is non-finite")
            if require_legacy_first_update and update_index == 0:
                reference_objective.backward()
            else:
                _assign_conflict_block_gradients(
                    parameters,
                    parameter_names,
                    stream_gradients,
                    blocks,
                    applied_weights,
                )
            applied_direction_digests.append(
                _assigned_gradient_digest(parameters, parameter_names)
            )
            controller_gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                gradient_clip,
            )
            if not bool(torch.isfinite(controller_gradient_norm).item()):
                raise RuntimeError("conflict controller gradient is non-finite")
            mixer_before_controller_step = {
                name: value.detach().clone()
                for name, value in mixer.state_dict().items()
            }
            optimizer.step()
            if any(
                not torch.equal(value, mixer.state_dict()[name].detach())
                for name, value in mixer_before_controller_step.items()
            ):
                controller_step_mixer_unchanged = False
                raise RuntimeError("controller optimizer changed conflict mixer")
            controller_after_controller_step = {
                name: parameter.detach().clone()
                for name, parameter in selected_named
            }
            mixer_optimizer.zero_grad(set_to_none=True)
            meta_objective.backward()
            mixer_gradient_norm = torch.nn.utils.clip_grad_norm_(
                tuple(mixer.parameters()),
                gradient_clip,
            )
            if not bool(torch.isfinite(mixer_gradient_norm).item()):
                raise RuntimeError("conflict mixer gradient is non-finite")
            mixer_optimizer.step()
            if any(
                not torch.equal(value, selected_parameter_map[name].detach())
                for name, value in controller_after_controller_step.items()
            ):
                mixer_step_controller_unchanged = False
                raise RuntimeError("mixer optimizer changed controller")
            reference_losses.append(float(reference_objective.detach().item()))
            controller_gradient_norms.append(
                float(controller_gradient_norm.detach().item())
            )
            mixer_gradient_norms.append(float(mixer_gradient_norm.detach().item()))
            meta_losses.append(float(meta_objective.detach().item()))
            meta_flat_penalties.append(
                float(meta_diagnostics["flat_penalty"].detach().item())
            )
            meta_robust_penalties.append(
                float(meta_diagnostics["robust_penalty"].detach().item())
            )
            meta_mean_kl_history.append(
                float(
                    meta_diagnostics["mean_kl_from_existing_weights"]
                    .detach()
                    .item()
                )
            )
            stream_loss_history.append(
                tuple(float(value) for value in per_stream_losses.detach().tolist())
            )
            row_loss_history.append(
                tuple(
                    tuple(float(value) for value in losses.detach().tolist())
                    for losses in per_stream_row_losses
                )
            )
            row_weight_history.append(
                tuple(
                    tuple(float(value) for value in values[3].detach().tolist())
                    for values in row_risk
                )
            )
            effective_row_history.append(
                tuple(float(values[4].detach().item()) for values in row_risk)
            )
            existing_weight_history.append(
                tuple(float(value) for value in existing_weights.detach().tolist())
            )
            applied_weight_history.append(
                tuple(
                    tuple(float(value) for value in row)
                    for row in applied_weights.detach().tolist()
                )
            )
            residual_logit_history.append(
                tuple(
                    tuple(float(value) for value in row)
                    for row in residual_logits.detach().tolist()
                )
            )
            withheld_alignment_history.append(
                tuple(
                    tuple(float(value) for value in row)
                    for row in meta_diagnostics["withheld_alignments"]
                    .detach()
                    .tolist()
                )
            )
            block_gradient_norm_history.append(
                tuple(
                    tuple(float(value) for value in row)
                    for row in gradient_norms.detach().tolist()
                )
            )
            block_cosine_gram_history.append(
                tuple(
                    tuple(tuple(float(value) for value in row) for row in block)
                    for block in cosine_grams.detach().tolist()
                )
            )
            legacy_negative_alignment_history.append(
                tuple(
                    float(value)
                    for value in legacy_diagnostics[
                        "negative_alignment_fractions"
                    ]
                    .detach()
                    .tolist()
                )
            )
            applied_negative_alignment_history.append(
                tuple(
                    float(value)
                    for value in applied_diagnostics[
                        "negative_alignment_fractions"
                    ]
                    .detach()
                    .tolist()
                )
            )
            legacy_cancellation_history.append(
                tuple(
                    float(value)
                    for value in legacy_diagnostics["cancellation_ratios"]
                    .detach()
                    .tolist()
                )
            )
            applied_cancellation_history.append(
                tuple(
                    float(value)
                    for value in applied_diagnostics["cancellation_ratios"]
                    .detach()
                    .tolist()
                )
            )
            legacy_direction_norm_history.append(
                tuple(
                    float(value)
                    for value in legacy_diagnostics["direction_norms"]
                    .detach()
                    .tolist()
                )
            )
            applied_direction_norm_history.append(
                tuple(
                    float(value)
                    for value in applied_diagnostics["direction_norms"]
                    .detach()
                    .tolist()
                )
            )
            row_count += len(rows)
        after = dict(named)
        for name, before in frozen.items():
            if not torch.equal(before, after[name].detach()):
                raise RuntimeError(f"conflict fit changed frozen parameter: {name}")
    finally:
        for name, parameter in named:
            parameter.requires_grad_(previous_requires_grad[name])
    mixer_delta_squared = sum(
        float(
            (
                value.detach().cpu() - mixer_before[name].detach().cpu()
            ).double().square().sum().item()
        )
        for name, value in mixer.state_dict().items()
    )
    post_first_existing_trace = tuple(
        tuple(tuple(update) for _ in blocks)
        for update in existing_weight_history[1:]
    )
    post_first_applied_trace = tuple(applied_weight_history[1:])
    return {
        "stage": stage,
        "optimizer_steps": len(stream_batches),
        "streams": sum(len(batch) for batch in stream_batches),
        "rows": row_count,
        "first_reference_loss": reference_losses[0],
        "last_reference_loss": reference_losses[-1],
        "reference_losses": tuple(reference_losses),
        "meta_losses": tuple(meta_losses),
        "meta_flat_penalties": tuple(meta_flat_penalties),
        "meta_robust_penalties": tuple(meta_robust_penalties),
        "meta_mean_kl_from_existing_weights": tuple(meta_mean_kl_history),
        "mean_controller_gradient_norm": (
            sum(controller_gradient_norms) / len(controller_gradient_norms)
        ),
        "controller_gradient_norms": tuple(controller_gradient_norms),
        "mean_mixer_gradient_norm": (
            sum(mixer_gradient_norms) / len(mixer_gradient_norms)
        ),
        "mixer_gradient_norms": tuple(mixer_gradient_norms),
        "stream_losses": tuple(stream_loss_history),
        "row_losses": tuple(row_loss_history),
        "row_gradient_weights": tuple(row_weight_history),
        "effective_row_counts": tuple(effective_row_history),
        "existing_stream_weights": tuple(existing_weight_history),
        "applied_block_weights": tuple(applied_weight_history),
        "residual_logits": tuple(residual_logit_history),
        "withheld_alignments": tuple(withheld_alignment_history),
        "block_gradient_norms": tuple(block_gradient_norm_history),
        "block_cosine_grams": tuple(block_cosine_gram_history),
        "legacy_negative_alignment_fractions": tuple(
            legacy_negative_alignment_history
        ),
        "applied_negative_alignment_fractions": tuple(
            applied_negative_alignment_history
        ),
        "legacy_cancellation_ratios": tuple(legacy_cancellation_history),
        "applied_cancellation_ratios": tuple(applied_cancellation_history),
        "legacy_direction_norms": tuple(legacy_direction_norm_history),
        "applied_direction_norms": tuple(applied_direction_norm_history),
        "legacy_direction_digests": tuple(legacy_direction_digests),
        "applied_direction_digests": tuple(applied_direction_digests),
        "parameter_blocks": {
            name: tuple(values) for name, values in blocks.items()
        },
        "legacy_first_update_required": require_legacy_first_update,
        "first_update_used_legacy_weights": first_update_used_legacy_weights,
        "mixer_parameter_delta_l2": math.sqrt(mixer_delta_squared),
        "mixer_parameters_changed": mixer_delta_squared > 0.0,
        "mixer_initial_digest": mixer_initial_digest,
        "mixer_terminal_digest": anonymous_conflict_mixer_digest(mixer),
        "post_first_existing_weight_trace_digest": (
            _conflict_weight_trace_digest(post_first_existing_trace)
        ),
        "post_first_applied_weight_trace_digest": (
            _conflict_weight_trace_digest(post_first_applied_trace)
        ),
        "trainable_parameter_names": tuple(sorted(selected_names)),
        "frozen_parameters_unchanged": True,
        "controller_step_mixer_unchanged": controller_step_mixer_unchanged,
        "mixer_step_controller_unchanged": mixer_step_controller_unchanged,
        "mixer_optimizer_moments": "fresh_per_relation_or_joint_stage",
        "public_leave_one_out_folds_per_update": _RELATION_CREDIT_STREAMS_PER_UPDATE,
        "stream_identity_input": False,
        "task_identity_input": False,
        "deterministic_gradient_projection": False,
    }


def evaluate_public_relation_credit_panel(
    controller: SoftwarePipelineController,
    streams: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    """Measure raw witness, selector, and aggregate margins on public streams."""

    if not streams:
        raise ValueError("relation-credit panel requires public streams")
    was_training = controller.training
    controller.eval()
    try:
        with torch.no_grad():
            rows = tuple(
                row
                for stream in streams
                for row in public_relation_credit_rows(controller, stream)
            )
    finally:
        controller.train(was_training)
    target_indices = tuple(int(torch.argmin(row.slot_losses).item()) for row in rows)
    if any(
        index != int(torch.argmax(row.responsibilities).item())
        for row, index in zip(rows, target_indices, strict=True)
    ):
        raise RuntimeError("relation-credit target disagrees with training responsibility")
    target_positive = tuple(
        float(row.slot_positive_margins[index].item())
        for row, index in zip(rows, target_indices, strict=True)
    )
    target_negative = tuple(
        float(row.slot_negative_margins[index].item())
        for row, index in zip(rows, target_indices, strict=True)
    )
    target_witness = tuple(
        positive - negative
        for positive, negative in zip(target_positive, target_negative, strict=True)
    )
    target_loss = tuple(
        float(row.slot_losses[index].item())
        for row, index in zip(rows, target_indices, strict=True)
    )
    target_loss_gaps = tuple(
        float(
            torch.topk(row.slot_losses, k=2, largest=False).values.diff().item()
        )
        for row in rows
    )
    target_responsibility = tuple(
        float(row.responsibilities[index].item())
        for row, index in zip(rows, target_indices, strict=True)
    )
    raw_indices = tuple(
        int(torch.argmax(row.slot_positive_margins - row.slot_negative_margins).item())
        for row in rows
    )
    raw_positive = tuple(
        float(row.slot_positive_margins[index].item())
        for row, index in zip(rows, raw_indices, strict=True)
    )
    raw_negative = tuple(
        float(row.slot_negative_margins[index].item())
        for row, index in zip(rows, raw_indices, strict=True)
    )
    raw_witness = tuple(
        left - right for left, right in zip(raw_positive, raw_negative, strict=True)
    )
    context_mass = tuple(
        float(row.context_weights[index].item())
        for row, index in zip(rows, target_indices, strict=True)
    )
    context_top_one = tuple(
        int(torch.argmax(row.context_weights).item()) == index
        for row, index in zip(rows, target_indices, strict=True)
    )
    valid_set_metrics = tuple(
        _relation_valid_set_metrics(
            row.slot_positive_margins,
            row.slot_negative_margins,
            row.context_weights,
            row.context_null_weight,
        )
        for row in rows
    )
    valid_slot_counts = tuple(
        int(metrics["valid_slot_count"]) for metrics in valid_set_metrics
    )
    relation_supported = tuple(
        metrics["relation_supported"] is True for metrics in valid_set_metrics
    )
    context_valid_set_mass = tuple(
        float(metrics["context_valid_set_mass"]) for metrics in valid_set_metrics
    )
    context_valid_set_top_one = tuple(
        metrics["context_valid_set_top_one"] is True
        for metrics in valid_set_metrics
    )
    supported_count = sum(relation_supported)
    supported_context_mass_mean = (
        sum(
            mass
            for mass, supported in zip(
                context_valid_set_mass,
                relation_supported,
                strict=True,
            )
            if supported
        )
        / supported_count
        if supported_count
        else 0.0
    )
    supported_context_top_one_fraction = (
        sum(
            top_one
            for top_one, supported in zip(
                context_valid_set_top_one,
                relation_supported,
                strict=True,
            )
            if supported
        )
        / supported_count
        if supported_count
        else 0.0
    )
    positive = tuple(float(row.positive_margin.item()) for row in rows)
    negative = tuple(float(row.negative_margin.item()) for row in rows)
    signed = tuple(
        left >= _RELATION_FIT_MARGIN and right <= -_RELATION_FIT_MARGIN
        for left, right in zip(positive, negative, strict=True)
    )
    unique_confident = tuple(
        left >= 0.05
        and right <= -0.05
        and witness >= _RELATION_FIT_MARGIN
        and loss_gap >= 0.02
        for left, right, witness, loss_gap in zip(
            target_positive,
            target_negative,
            target_witness,
            target_loss_gaps,
            strict=True,
        )
    )
    stream_signed = tuple(
        sum(signed[index : index + 4])
        for index in range(0, len(signed), 4)
    )
    stream_unique_confident = tuple(
        sum(unique_confident[index : index + 4])
        for index in range(0, len(unique_confident), 4)
    )
    stream_supported = tuple(
        sum(relation_supported[index : index + 4])
        for index in range(0, len(relation_supported), 4)
    )
    row_reports = tuple(
        {
            "stream_index": row_index // 4,
            "heldout_index": row.heldout_index,
            "transition_index": row.transition_index,
            "target_slot": target_index,
            "target_positive_margin": target_left,
            "target_negative_margin": target_right,
            "target_witness": target_value,
            "target_loss": loss_value,
            "target_loss_gap": loss_gap,
            "target_responsibility": responsibility,
            "context_target_mass": mass,
            "context_top_one": top_one,
            "valid_slots": tuple(
                int(index)
                for index in metrics["valid_mask"].nonzero().flatten().tolist()
            ),
            "valid_slot_count": metrics["valid_slot_count"],
            "relation_supported": metrics["relation_supported"],
            "context_null_mass": metrics["context_null_mass"],
            "context_valid_set_mass": metrics["context_valid_set_mass"],
            "context_valid_set_top_one": metrics["context_valid_set_top_one"],
            "raw_slot": raw_index,
            "raw_positive_margin": raw_left,
            "raw_negative_margin": raw_right,
            "raw_witness": raw_value,
            "slot_positive_margins": tuple(
                float(value) for value in row.slot_positive_margins.tolist()
            ),
            "slot_negative_margins": tuple(
                float(value) for value in row.slot_negative_margins.tolist()
            ),
            "slot_losses": tuple(float(value) for value in row.slot_losses.tolist()),
            "responsibilities": tuple(
                float(value) for value in row.responsibilities.tolist()
            ),
            "context_weights": tuple(
                float(value) for value in row.context_weights.tolist()
            ),
            "positive_margin": aggregate_left,
            "negative_margin": aggregate_right,
            "unique_loss_selected_confident": is_unique_confident,
            "signed": is_signed,
        }
        for row_index, (
            row,
            target_index,
            target_left,
            target_right,
            target_value,
            loss_value,
            loss_gap,
            responsibility,
            mass,
            top_one,
            metrics,
            raw_index,
            raw_left,
            raw_right,
            raw_value,
            aggregate_left,
            aggregate_right,
            is_unique_confident,
            is_signed,
        ) in enumerate(
            zip(
                rows,
                target_indices,
                target_positive,
                target_negative,
                target_witness,
                target_loss,
                target_loss_gaps,
                target_responsibility,
                context_mass,
                context_top_one,
                valid_set_metrics,
                raw_indices,
                raw_positive,
                raw_negative,
                raw_witness,
                positive,
                negative,
                unique_confident,
                signed,
                strict=True,
            )
        )
    )
    return {
        "streams": len(streams),
        "rows": len(rows),
        "target_positive_mean": sum(target_positive) / len(target_positive),
        "target_negative_mean": sum(target_negative) / len(target_negative),
        "target_witness_mean": sum(target_witness) / len(target_witness),
        "target_loss_mean": sum(target_loss) / len(target_loss),
        "target_loss_gap_mean": sum(target_loss_gaps) / len(target_loss_gaps),
        "target_responsibility_mean": sum(target_responsibility)
        / len(target_responsibility),
        "raw_best_positive_mean": sum(raw_positive) / len(raw_positive),
        "raw_best_negative_mean": sum(raw_negative) / len(raw_negative),
        "raw_best_witness_mean": sum(raw_witness) / len(raw_witness),
        "unique_loss_selected_confident_rows": sum(unique_confident),
        "streams_with_three_unique_loss_selected_confident_rows": sum(
            value >= 3 for value in stream_unique_confident
        ),
        "relation_supported_rows": supported_count,
        "streams_with_three_supported_rows": sum(
            value >= 3 for value in stream_supported
        ),
        "supported_rows_per_stream": stream_supported,
        "valid_slot_count_histogram": tuple(
            valid_slot_counts.count(count) for count in range(4)
        ),
        "positive_margin_mean": sum(positive) / len(positive),
        "negative_margin_mean": sum(negative) / len(negative),
        "separation_mean": sum(
            left - right for left, right in zip(positive, negative, strict=True)
        )
        / len(positive),
        "signed_rows": sum(signed),
        "streams_with_three_signed_rows": sum(value >= 3 for value in stream_signed),
        "context_target_mass_mean": sum(context_mass) / len(context_mass),
        "context_top_one_fraction": sum(context_top_one) / len(context_top_one),
        "context_valid_set_mass_mean_supported": supported_context_mass_mean,
        "context_valid_set_mass_mean_all_rows": sum(context_valid_set_mass)
        / len(context_valid_set_mass),
        "context_valid_set_top_one_fraction_supported": (
            supported_context_top_one_fraction
        ),
        "row_reports": row_reports,
    }


def _paired_public_relation_action_loss(
    controller: SoftwarePipelineController,
    stream: SoftwarePipelineStream,
) -> tuple[torch.Tensor, int]:
    """Calibrate only evidence action tensors after the raw relation gate."""

    terms = []
    for fold in _paired_public_relation_folds(controller, stream):
        encoded = controller.encode_task(fold.masked_task)
        before_index = _state_index(fold.masked_task.states, fold.before)
        belief = F.one_hot(
            torch.tensor(before_index, device=encoded.role_state_embeddings.device),
            len(fold.masked_task.states),
        ).to(dtype=encoded.role_state_embeddings.dtype)
        base = controller.score_actions(
            fold.masked_task,
            fold.positive_state,
            current_state_belief=belief,
            encoding=encoded,
            include_pointer_memory=False,
            include_role_memory=False,
        ).action_logits.detach()
        positive_evidence = controller._relation_evidence_scores(
            encoded.relation_context_embeddings,
            encoded.relation_component_embeddings,
            fold.positive_state,
        )
        negative_evidence = controller._relation_evidence_scores(
            encoded.relation_context_embeddings,
            encoded.relation_component_embeddings,
            fold.negative_state,
        )
        positive = base + controller._evidence_action_contribution(
            positive_evidence
        )
        negative = base + controller._evidence_action_contribution(
            negative_evidence
        )
        positive_index = _action_index(
            fold.masked_task.grounded_candidates,
            fold.positive_action,
        )
        negative_index = _action_index(
            fold.masked_task.grounded_candidates,
            fold.negative_action,
        )
        pair_indices = torch.tensor(
            (positive_index, negative_index),
            device=positive.device,
            dtype=torch.long,
        )
        positive_pair = positive.index_select(0, pair_indices)
        negative_pair = negative.index_select(0, pair_indices)
        zero = torch.zeros((1,), device=positive.device, dtype=torch.long)
        one = torch.ones((1,), device=positive.device, dtype=torch.long)
        positive_delta = positive_pair[0] - positive_pair[1]
        negative_delta = negative_pair[0] - negative_pair[1]
        margin = positive.new_tensor(_PUBLIC_ROLE_CAUSAL_MARGIN_NATS)
        terms.append(
            0.5
            * (
                F.cross_entropy(positive_pair.unsqueeze(0), zero)
                + F.cross_entropy(negative_pair.unsqueeze(0), one)
            )
            + F.relu(margin - positive_delta)
            + F.relu(margin + negative_delta)
        )
    if len(terms) != 4:
        raise RuntimeError("relation action calibration lost whole-fold balance")
    return torch.stack(terms).mean(), 2 * len(terms)


def _fit_public_relation_matcher_streams(
    controller: SoftwarePipelineController,
    streams: Sequence[SoftwarePipelineStream],
    *,
    learning_rate: float,
    gradient_clip: float = 5.0,
) -> dict[str, object]:
    """Fit only the new relation matcher over an already bounded stream set."""

    if not streams:
        raise ValueError("relation fit needs at least one stream")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("relation fit learning rate must be finite and positive")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0.0:
        raise ValueError("relation fit gradient clip must be finite and positive")
    selected_names = set(_relation_matcher_parameter_names(controller))
    named = tuple(controller.named_parameters())
    if not selected_names:
        raise RuntimeError("relation matcher has no trainable parameters")
    previous_requires_grad = {
        name: parameter.requires_grad for name, parameter in named
    }
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in named
        if name not in selected_names
    }
    for name, parameter in named:
        parameter.requires_grad_(name in selected_names)
    parameters = tuple(
        parameter for name, parameter in named if name in selected_names
    )
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    report_rows = []
    gradient_norms = []
    try:
        for stream_index, stream in enumerate(streams):
            # Row construction creates all fast states after the preceding
            # optimizer update; no learned relation key survives an update.
            rows = public_paired_relation_fit_rows(controller, stream)
            optimizer.zero_grad(set_to_none=True)
            objective = torch.stack(tuple(row.loss for row in rows)).mean()
            if not bool(torch.isfinite(objective).item()):
                raise RuntimeError("relation fit objective is non-finite")
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("relation fit gradient is non-finite")
            optimizer.step()
            gradient_norms.append(float(gradient_norm.detach().item()))
            report_rows.extend(
                {
                    "stream_index": stream_index,
                    "heldout_index": row.heldout_index,
                    "transition_index": row.transition_index,
                    "positive_margin": float(row.positive_margin.detach().item()),
                    "negative_margin": float(row.negative_margin.detach().item()),
                    "loss": float(row.loss.detach().item()),
                }
                for row in rows
            )
        for name, before in frozen.items():
            if not torch.equal(before, dict(named)[name].detach()):
                raise RuntimeError(f"relation fit changed frozen parameter: {name}")
    finally:
        for name, parameter in named:
            parameter.requires_grad_(previous_requires_grad[name])
    return {
        "optimizer_steps": len(streams),
        "rows": tuple(report_rows),
        "row_count": len(report_rows),
        "directional_arms": 2 * len(report_rows),
        "trainable_parameter_names": tuple(sorted(selected_names)),
        "frozen_parameters_unchanged": True,
        "fresh_fast_state_after_every_update": True,
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
    }


def _calibrate_public_relation_action_streams(
    controller: SoftwarePipelineController,
    streams: Sequence[SoftwarePipelineStream],
    *,
    raw_gate_passed: bool,
    learning_rate: float,
    gradient_clip: float = 5.0,
) -> dict[str, object]:
    """Calibrate four legacy action tensors only after the raw public gate."""

    if raw_gate_passed is not True:
        raise RuntimeError("relation action calibration requires a passed raw gate")
    if not streams:
        raise ValueError("relation action calibration needs public streams")
    calibration_names = {
        "evidence_action_log_gate",
        "evidence_action_head.0.weight",
        "evidence_action_head.0.bias",
        "evidence_action_head.2.weight",
    }
    named = tuple(controller.named_parameters())
    if {name for name, _ in named if name in calibration_names} != calibration_names:
        raise RuntimeError("relation action calibration boundary changed")
    previous_requires_grad = {
        name: parameter.requires_grad for name, parameter in named
    }
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in named
        if name not in calibration_names
    }
    for name, parameter in named:
        parameter.requires_grad_(name in calibration_names)
    parameters = tuple(
        parameter for name, parameter in named if name in calibration_names
    )
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    losses = []
    gradient_norms = []
    directional_arms = 0
    probe = streams[0]
    probe_task = replace(probe.supports[0].learner, observations=())
    probe_state = _acquire_public_task_set(
        controller,
        tuple(pair.learner for pair in probe.supports[1:]),
    )
    with torch.no_grad():
        no_memory_before = controller.score_actions(
            probe_task,
            probe_state,
            include_pointer_memory=False,
            include_role_memory=False,
        ).action_logits.detach().clone()
    try:
        for stream in streams:
            optimizer.zero_grad(set_to_none=True)
            objective, arms = _paired_public_relation_action_loss(
                controller,
                stream,
            )
            if not bool(torch.isfinite(objective).item()):
                raise RuntimeError("relation action calibration is non-finite")
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("relation action gradient is non-finite")
            optimizer.step()
            losses.append(float(objective.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
            directional_arms += arms
        with torch.no_grad():
            no_memory_after = controller.score_actions(
                probe_task,
                probe_state,
                include_pointer_memory=False,
                include_role_memory=False,
            ).action_logits.detach().clone()
        if not torch.equal(no_memory_before, no_memory_after):
            raise RuntimeError("relation action calibration changed no-memory logits")
        for name, before in frozen.items():
            if not torch.equal(before, dict(named)[name].detach()):
                raise RuntimeError(
                    f"relation action calibration changed frozen parameter: {name}"
                )
    finally:
        for name, parameter in named:
            parameter.requires_grad_(previous_requires_grad[name])
    return {
        "optimizer_steps": len(losses),
        "directional_arms": directional_arms,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "trainable_parameter_names": tuple(sorted(calibration_names)),
        "relation_matcher_frozen": True,
        "all_other_parameters_unchanged": True,
        "no_memory_logits_exact": True,
    }


def _scheduled_relation_streams(
    commitments: Sequence[str],
    seed_pairs: Sequence[tuple[int, int]],
) -> tuple[SoftwarePipelineStream, ...]:
    return tuple(
        make_software_pipeline_stream(
            topology_seed,
            surface_seed=surface_seed,
            supports_per_motif=2,
            queries=1,
            maximum_steps=4,
            mechanism_commitment=commitment,
            mechanism_partition="train",
        )
        for commitment in commitments
        for topology_seed, surface_seed in seed_pairs
    )


def build_public_relation_credit_controller(
    *,
    device: torch.device | str = "cpu",
) -> SoftwarePipelineController:
    """Build the exact clean-start controller bound to the v11 fit plan."""

    cpu_rng_state = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(_RELATION_CREDIT_INITIALIZATION_SEED)
        controller = build_software_pipeline_controller("smoke")
    finally:
        torch.set_rng_state(cpu_rng_state)
    return controller.to(device)


def build_public_relation_conflict_system(
    *,
    device: torch.device | str = "cpu",
) -> tuple[SoftwarePipelineController, AnonymousConflictMixer]:
    """Build the fresh v12 controller and its separately learned update rule."""

    cpu_rng_state = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(_CONFLICT_INITIALIZATION_SEED)
        controller = build_software_pipeline_controller("smoke")
        torch.default_generator.manual_seed(_CONFLICT_MIXER_INITIALIZATION_SEED)
        mixer = AnonymousConflictMixer()
    finally:
        torch.set_rng_state(cpu_rng_state)
    return controller.to(device), mixer.to(device)


def build_capacity_matched_relation_cluster_pair(
    replicate: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[
    SoftwarePipelineController,
    CapacityMatchedClusterController,
    AnonymousConflictMixer,
    AnonymousConflictMixer,
]:
    """Build one causally paired monolith/cluster system without RNG coupling."""

    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or not 0 <= replicate < len(_CLUSTER_REPLICATE_SEEDS)
    ):
        raise ValueError("cluster replicate is outside the fixed plan")
    shared_seed, cell_seed, composer_seed, mixer_seed = (
        _CLUSTER_REPLICATE_SEEDS[replicate]
    )
    profile = SOFTWARE_PIPELINE_PROFILES["smoke"]
    cpu_rng_state = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(shared_seed)
        monolith = CapacityMatchedMonolithController(profile)
        torch.default_generator.manual_seed(shared_seed)
        cluster = CapacityMatchedClusterController(
            profile,
            cell_seed=cell_seed,
            composer_seed=composer_seed,
        )
        torch.default_generator.manual_seed(mixer_seed)
        monolith_mixer = AnonymousConflictMixer()
        torch.default_generator.manual_seed(mixer_seed)
        cluster_mixer = AnonymousConflictMixer()
    finally:
        torch.set_rng_state(cpu_rng_state)
    if anonymous_conflict_mixer_digest(
        monolith_mixer
    ) != anonymous_conflict_mixer_digest(cluster_mixer):
        raise RuntimeError("paired conflict mixers do not share initialization")
    if not _capacity_matched_shared_parameters_are_exact(monolith, cluster):
        raise RuntimeError("paired arms do not share exact non-relation parameters")
    return (
        monolith.to(device),
        cluster.to(device),
        monolith_mixer.to(device),
        cluster_mixer.to(device),
    )


def _capacity_matched_shared_parameters_are_exact(
    monolith: SoftwarePipelineController,
    cluster: CapacityMatchedClusterController,
) -> bool:
    if (
        not isinstance(monolith, SoftwarePipelineController)
        or isinstance(monolith, CapacityMatchedClusterController)
        or not isinstance(cluster, CapacityMatchedClusterController)
        or monolith.profile != cluster.profile
    ):
        return False
    monolith_state = monolith.state_dict()
    cluster_state = cluster.state_dict()
    cluster_only_prefixes = ("relation_cells.", "relation_composer.")
    monolith_relation_prefixes = (
        "evidence_pair_encoder.",
        "relation_pool_attention.",
        "relation_pool_projection.",
        "relation_incidence_readout.",
        "relation_incidence_projection.",
        "relation_comparator.",
    )
    shared_names = tuple(
        name
        for name in monolith_state
        if not name.startswith(monolith_relation_prefixes)
    )
    if any(name.startswith(cluster_only_prefixes) for name in shared_names):
        return False
    return (
        set(shared_names)
        == {
            name
            for name in cluster_state
            if not name.startswith(cluster_only_prefixes)
        }
        and all(
            torch.equal(monolith_state[name], cluster_state[name])
            for name in shared_names
        )
    )


def capacity_matched_relation_cluster_parameter_report(
    monolith: SoftwarePipelineController,
    cluster: CapacityMatchedClusterController,
    monolith_mixer: AnonymousConflictMixer,
    cluster_mixer: AnonymousConflictMixer,
) -> dict[str, object]:
    """Report complete learned capacity, excluding no inert parity padding."""

    monolith_total = sum(parameter.numel() for parameter in monolith.parameters())
    cluster_total = sum(parameter.numel() for parameter in cluster.parameters())
    monolith_mixer_total = sum(
        parameter.numel() for parameter in monolith_mixer.parameters()
    )
    cluster_mixer_total = sum(
        parameter.numel() for parameter in cluster_mixer.parameters()
    )
    monolith_complete = monolith_total + monolith_mixer_total
    cluster_complete = cluster_total + cluster_mixer_total
    complete_difference = cluster_complete - monolith_complete
    complete_fraction = abs(complete_difference) / monolith_complete
    monolith_active = sum(
        dict(monolith.named_parameters())[name].numel()
        for name in _relation_credit_parameter_names(monolith, "relation")
    ) + monolith_mixer_total
    cluster_active = sum(
        dict(cluster.named_parameters())[name].numel()
        for name in _relation_credit_parameter_names(cluster, "relation")
    ) + cluster_mixer_total
    active_difference = cluster_active - monolith_active
    active_fraction = abs(active_difference) / monolith_active
    cell_parameters = tuple(
        sum(parameter.numel() for parameter in cell.parameters())
        for cell in cluster.relation_cells
    )
    composer_parameters = sum(
        parameter.numel() for parameter in cluster.relation_composer.parameters()
    )
    return {
        "protocol_id": _CLUSTER_PROTOCOL_ID,
        "monolith_controller_parameters": monolith_total,
        "cluster_controller_parameters": cluster_total,
        "monolith_mixer_parameters": monolith_mixer_total,
        "cluster_mixer_parameters": cluster_mixer_total,
        "monolith_complete_parameters": monolith_complete,
        "cluster_complete_parameters": cluster_complete,
        "cluster_minus_monolith_complete_parameters": complete_difference,
        "absolute_complete_fractional_difference": complete_fraction,
        "monolith_active_trainable_parameters": monolith_active,
        "cluster_active_trainable_parameters": cluster_active,
        "cluster_minus_monolith_active_parameters": active_difference,
        "absolute_active_fractional_difference": active_fraction,
        "within_declared_tolerance": complete_fraction
        <= _CLUSTER_PARAMETER_TOLERANCE_FRACTION
        and active_fraction <= _CLUSTER_PARAMETER_TOLERANCE_FRACTION,
        "cell_parameters": cell_parameters,
        "composer_parameters": composer_parameters,
        "inert_padding_parameters": 0,
        "shared_non_relation_parameters_bit_exact": (
            _capacity_matched_shared_parameters_are_exact(monolith, cluster)
        ),
    }


def _learned_module_digest(module: nn.Module) -> str:
    if not isinstance(module, nn.Module):
        raise TypeError("learned module digest requires an nn.Module")
    digest = hashlib.sha256(_CLUSTER_DIGEST_DOMAIN)
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


def capacity_matched_relation_cluster_system_digest(
    monolith: SoftwarePipelineController,
    cluster: CapacityMatchedClusterController,
    monolith_mixer: AnonymousConflictMixer,
    cluster_mixer: AnonymousConflictMixer,
    replicate: int,
) -> str:
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or not 0 <= replicate < len(_CLUSTER_REPLICATE_SEEDS)
    ):
        raise ValueError("cluster system replicate is invalid")
    digest = hashlib.sha256(_CLUSTER_DIGEST_DOMAIN)
    values = (
        _CLUSTER_PROTOCOL_ID,
        str(replicate),
        software_pipeline_model_digest(monolith),
        software_pipeline_model_digest(cluster),
        anonymous_conflict_mixer_digest(monolith_mixer),
        anonymous_conflict_mixer_digest(cluster_mixer),
    )
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def capacity_matched_relation_cluster_plan_digest() -> str:
    digest = hashlib.sha256(_CLUSTER_DIGEST_DOMAIN)
    digest.update(
        json.dumps(
            capacity_matched_relation_cluster_fit_plan(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return "sha256:" + digest.hexdigest()


def save_capacity_matched_relation_cluster_checkpoint(
    path: str | Path,
    systems: Sequence[
        tuple[
            SoftwarePipelineController,
            CapacityMatchedClusterController,
            AnonymousConflictMixer,
            AnonymousConflictMixer,
        ]
    ],
) -> None:
    if len(systems) != len(_CLUSTER_REPLICATE_SEEDS):
        raise ValueError("cluster checkpoint requires every fixed replicate")
    plan = capacity_matched_relation_cluster_fit_plan()
    records = []
    for replicate, system in enumerate(systems):
        monolith, cluster, monolith_mixer, cluster_mixer = system
        parameter_report = capacity_matched_relation_cluster_parameter_report(
            monolith,
            cluster,
            monolith_mixer,
            cluster_mixer,
        )
        if (
            parameter_report["within_declared_tolerance"] is not True
            or parameter_report["shared_non_relation_parameters_bit_exact"]
            is not True
        ):
            raise RuntimeError("cluster checkpoint system violates paired capacity")
        records.append(
            {
                "replicate": replicate,
                "monolith_state": {
                    name: value.detach().cpu().clone()
                    for name, value in monolith.state_dict().items()
                },
                "cluster_state": {
                    name: value.detach().cpu().clone()
                    for name, value in cluster.state_dict().items()
                },
                "monolith_mixer_state": {
                    name: value.detach().cpu().clone()
                    for name, value in monolith_mixer.state_dict().items()
                },
                "cluster_mixer_state": {
                    name: value.detach().cpu().clone()
                    for name, value in cluster_mixer.state_dict().items()
                },
                "monolith_digest": software_pipeline_model_digest(monolith),
                "cluster_digest": software_pipeline_model_digest(cluster),
                "cell_digests": tuple(
                    _learned_module_digest(cell) for cell in cluster.relation_cells
                ),
                "composer_digest": _learned_module_digest(
                    cluster.relation_composer
                ),
                "monolith_mixer_digest": anonymous_conflict_mixer_digest(
                    monolith_mixer
                ),
                "cluster_mixer_digest": anonymous_conflict_mixer_digest(
                    cluster_mixer
                ),
                "system_digest": (
                    capacity_matched_relation_cluster_system_digest(
                        monolith,
                        cluster,
                        monolith_mixer,
                        cluster_mixer,
                        replicate,
                    )
                ),
                "parameter_report": parameter_report,
            }
        )
    torch.save(
        {
            "version": _CLUSTER_CHECKPOINT_VERSION,
            "protocol_id": _CLUSTER_PROTOCOL_ID,
            "plan": plan,
            "plan_digest": capacity_matched_relation_cluster_plan_digest(),
            "replicates": tuple(records),
        },
        Path(path),
    )


def load_capacity_matched_relation_cluster_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[
    tuple[
        SoftwarePipelineController,
        CapacityMatchedClusterController,
        AnonymousConflictMixer,
        AnonymousConflictMixer,
    ],
    ...,
]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    expected_plan = capacity_matched_relation_cluster_fit_plan()
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _CLUSTER_CHECKPOINT_VERSION
        or payload.get("protocol_id") != _CLUSTER_PROTOCOL_ID
        or payload.get("plan") != expected_plan
        or payload.get("plan_digest")
        != capacity_matched_relation_cluster_plan_digest()
    ):
        raise RuntimeError("cluster checkpoint identity or seed plan is invalid")
    records = payload.get("replicates")
    if not isinstance(records, (tuple, list)) or len(records) != len(
        _CLUSTER_REPLICATE_SEEDS
    ):
        raise RuntimeError("cluster checkpoint replicate set is invalid")
    restored = []
    for replicate, record in enumerate(records):
        if not isinstance(record, dict) or record.get("replicate") != replicate:
            raise RuntimeError("cluster checkpoint replicate identity changed")
        system = build_capacity_matched_relation_cluster_pair(
            replicate,
            device=device,
        )
        monolith, cluster, monolith_mixer, cluster_mixer = system
        monolith.load_state_dict(record["monolith_state"], strict=True)
        cluster.load_state_dict(record["cluster_state"], strict=True)
        monolith_mixer.load_state_dict(record["monolith_mixer_state"], strict=True)
        cluster_mixer.load_state_dict(record["cluster_mixer_state"], strict=True)
        observed = {
            "monolith_digest": software_pipeline_model_digest(monolith),
            "cluster_digest": software_pipeline_model_digest(cluster),
            "cell_digests": tuple(
                _learned_module_digest(cell) for cell in cluster.relation_cells
            ),
            "composer_digest": _learned_module_digest(
                cluster.relation_composer
            ),
            "monolith_mixer_digest": anonymous_conflict_mixer_digest(
                monolith_mixer
            ),
            "cluster_mixer_digest": anonymous_conflict_mixer_digest(
                cluster_mixer
            ),
            "system_digest": capacity_matched_relation_cluster_system_digest(
                monolith,
                cluster,
                monolith_mixer,
                cluster_mixer,
                replicate,
            ),
            "parameter_report": capacity_matched_relation_cluster_parameter_report(
                monolith,
                cluster,
                monolith_mixer,
                cluster_mixer,
            ),
        }
        if any(observed[key] != record.get(key) for key in observed):
            raise RuntimeError("cluster checkpoint learned lineage changed")
        monolith.eval()
        cluster.eval()
        monolith_mixer.eval()
        cluster_mixer.eval()
        restored.append(system)
    return tuple(restored)


def _public_relation_conflict_system_is_fresh(
    controller: SoftwarePipelineController,
    mixer: AnonymousConflictMixer,
) -> bool:
    if (
        not isinstance(controller, SoftwarePipelineController)
        or not isinstance(mixer, AnonymousConflictMixer)
        or controller.profile != SOFTWARE_PIPELINE_PROFILES["smoke"]
    ):
        return False
    reference_controller, reference_mixer = build_public_relation_conflict_system()
    expected_controller = reference_controller.state_dict()
    expected_mixer = reference_mixer.state_dict()
    actual_controller = controller.state_dict()
    actual_mixer = mixer.state_dict()
    return (
        expected_controller.keys() == actual_controller.keys()
        and expected_mixer.keys() == actual_mixer.keys()
        and all(
            value.shape == expected_controller[name].shape
            and value.dtype == expected_controller[name].dtype
            and torch.equal(value.detach().cpu(), expected_controller[name])
            for name, value in actual_controller.items()
        )
        and all(
            value.shape == expected_mixer[name].shape
            and value.dtype == expected_mixer[name].dtype
            and torch.equal(value.detach().cpu(), expected_mixer[name])
            for name, value in actual_mixer.items()
        )
    )


def _public_relation_credit_controller_is_fresh(
    controller: SoftwarePipelineController,
) -> bool:
    if (
        not isinstance(controller, SoftwarePipelineController)
        or controller.profile != SOFTWARE_PIPELINE_PROFILES["smoke"]
        or controller.pointer_memory.slots != controller.profile.pointer_slots
        or controller.role_memory.slots != controller.profile.role_slots
        or controller.pointer_memory.read_top_k != controller.profile.memory_read_top_k
        or controller.role_memory.read_top_k != controller.profile.memory_read_top_k
    ):
        return False
    reference = build_public_relation_credit_controller()
    expected = reference.state_dict()
    actual = controller.state_dict()
    return expected.keys() == actual.keys() and all(
        value.shape == expected[name].shape
        and value.dtype == expected[name].dtype
        and torch.equal(value.detach().cpu(), expected[name])
        for name, value in actual.items()
    )


def _relation_credit_stream_batches(
    commitments: Sequence[str],
    seed_batches: Sequence[Sequence[tuple[int, int]]],
) -> tuple[tuple[SoftwarePipelineStream, ...], ...]:
    if (
        len(commitments) != _RELATION_CREDIT_COMMITMENTS
        or len(set(commitments)) != len(commitments)
        or not seed_batches
    ):
        raise ValueError("relation-credit batches require the fixed commitments")
    batches = []
    seen_pairs: set[tuple[int, int]] = set()
    for seed_batch in seed_batches:
        if len(seed_batch) != _RELATION_CREDIT_STREAMS_PER_UPDATE:
            raise ValueError("relation-credit update lost a stream")
        if len(set(seed_batch)) != len(seed_batch) or seen_pairs & set(seed_batch):
            raise ValueError("relation-credit update reused a seed pair")
        seen_pairs.update(seed_batch)
        batch = tuple(
            make_software_pipeline_stream(
                topology_seed,
                surface_seed=surface_seed,
                supports_per_motif=2,
                queries=1,
                maximum_steps=4,
                mechanism_commitment=commitment,
                mechanism_partition="train",
            )
            for commitment, (topology_seed, surface_seed) in zip(
                commitments,
                seed_batch,
                strict=True,
            )
        )
        if tuple(stream.mechanism_commitment for stream in batch) != tuple(commitments):
            raise RuntimeError("relation-credit batch changed commitment coverage")
        batches.append(batch)
    return tuple(batches)


def _relation_credit_panel_streams(
    commitments: Sequence[str],
    seed_pairs: Sequence[tuple[int, int]],
) -> tuple[SoftwarePipelineStream, ...]:
    if (
        len(commitments) != _RELATION_CREDIT_COMMITMENTS
        or len(seed_pairs) != _RELATION_CREDIT_COMMITMENTS
        or len(set(commitments)) != len(commitments)
        or len(set(seed_pairs)) != len(seed_pairs)
    ):
        raise ValueError("relation-credit panel identity changed")
    streams = tuple(
        make_software_pipeline_stream(
            topology_seed,
            surface_seed=surface_seed,
            supports_per_motif=2,
            queries=1,
            maximum_steps=4,
            mechanism_commitment=commitment,
            mechanism_partition="train",
        )
        for commitment, (topology_seed, surface_seed) in zip(
            commitments,
            seed_pairs,
            strict=True,
        )
    )
    if tuple(stream.mechanism_commitment for stream in streams) != tuple(commitments):
        raise RuntimeError("relation-credit panel changed commitment coverage")
    return streams


def _evaluate_public_relation_credit_invariants(
    controller: SoftwarePipelineController,
    streams: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    """Check covariance and exact empty-memory behavior on one public panel."""

    if len(streams) != _RELATION_CREDIT_COMMITMENTS:
        raise ValueError("relation-credit invariants require the fixed panel")
    was_training = controller.training
    controller.eval()
    axis_deltas = {
        "evidence_order": 0.0,
        "public_presentation": 0.0,
        "combined": 0.0,
    }
    valid_set_covariant = True
    empty_memory_zero = True
    transformations = (
        ("evidence_order", True, False),
        ("public_presentation", False, True),
        ("combined", True, True),
    )
    try:
        with torch.no_grad():
            for stream in streams:
                ordinary = public_relation_credit_rows(controller, stream)
                for axis, reverse_evidence, reverse_presentation in transformations:
                    transformed = public_relation_credit_rows(
                        controller,
                        stream,
                        reverse_evidence_order=reverse_evidence,
                        reverse_public_presentation=reverse_presentation,
                    )
                    for left, right in zip(ordinary, transformed, strict=True):
                        for field in (
                            "positive_margin",
                            "negative_margin",
                            "instance_loss",
                            "context_loss",
                            "separation_loss",
                            "joint_loss",
                            "context_null_weight",
                        ):
                            axis_deltas[axis] = max(
                                axis_deltas[axis],
                                float(
                                    (
                                        getattr(left, field) - getattr(right, field)
                                    )
                                    .abs()
                                    .item()
                                ),
                            )
                        left_metrics = _relation_valid_set_metrics(
                            left.slot_positive_margins,
                            left.slot_negative_margins,
                            left.context_weights,
                            left.context_null_weight,
                        )
                        right_positive = right.slot_positive_margins
                        right_negative = right.slot_negative_margins
                        right_context = right.context_weights
                        if reverse_evidence:
                            right_positive = right_positive.flip(0)
                            right_negative = right_negative.flip(0)
                            right_context = right_context.flip(0)
                        right_metrics = _relation_valid_set_metrics(
                            right_positive,
                            right_negative,
                            right_context,
                            right.context_null_weight,
                        )
                        valid_set_covariant = valid_set_covariant and torch.equal(
                            left_metrics["valid_mask"],
                            right_metrics["valid_mask"],
                        )
                        valid_set_covariant = valid_set_covariant and all(
                            left_metrics[field] == right_metrics[field]
                            for field in (
                                "valid_slot_count",
                                "relation_supported",
                                "context_valid_set_top_one",
                            )
                        )
                        for field in (
                            "slot_losses",
                            "slot_positive_margins",
                            "slot_negative_margins",
                            "responsibilities",
                            "context_weights",
                        ):
                            right_value = getattr(right, field)
                            if reverse_evidence:
                                right_value = right_value.flip(0)
                            axis_deltas[axis] = max(
                                axis_deltas[axis],
                                float(
                                    (getattr(left, field) - right_value)
                                    .abs()
                                    .max()
                                    .item()
                                ),
                            )
                for pair in stream.supports:
                    task = replace(pair.learner, observations=())
                    encoding = controller.encode_task(task)
                    scores = controller._relation_evidence_scores(
                        encoding.relation_context_embeddings,
                        encoding.relation_component_embeddings,
                        controller.initial_state(),
                    )
                    empty_memory_zero = empty_memory_zero and torch.equal(
                        scores,
                        torch.zeros_like(scores),
                    )
    finally:
        controller.train(was_training)
    maximum_delta = max(axis_deltas.values())
    return {
        "permutation_max_delta": maximum_delta,
        "evidence_order_max_delta": axis_deltas["evidence_order"],
        "public_presentation_max_delta": axis_deltas["public_presentation"],
        "combined_max_delta": axis_deltas["combined"],
        "valid_set_covariant": valid_set_covariant,
        "permutation_covariant": (
            maximum_delta <= _RELATION_GATE_PERMUTATION_TOLERANCE
            and valid_set_covariant
        ),
        "empty_memory_zero_exact": empty_memory_zero,
    }


def _relation_credit_relation_gate(
    panel: Mapping[str, object],
    invariants: Mapping[str, object],
) -> dict[str, object]:
    supported_rows = panel.get("relation_supported_rows")
    supported_streams = panel.get("streams_with_three_supported_rows")
    per_stream = panel.get("supported_rows_per_stream")
    histogram = panel.get("valid_slot_count_histogram")
    per_stream_consistent = (
        isinstance(per_stream, (tuple, list))
        and len(per_stream) == 8
        and all(type(value) is int and 0 <= value <= 4 for value in per_stream)
        and type(supported_rows) is int
        and sum(per_stream) == supported_rows
        and type(supported_streams) is int
        and sum(value >= 3 for value in per_stream) == supported_streams
    )
    histogram_consistent = (
        isinstance(histogram, (tuple, list))
        and len(histogram) == 4
        and all(type(value) is int and value >= 0 for value in histogram)
        and sum(histogram) == 32
        and type(supported_rows) is int
        and 32 - histogram[0] == supported_rows
    )
    checks = {
        "fixed_stream_count": panel.get("streams") == 8,
        "fixed_row_count": panel.get("rows") == 32,
        "per_stream_counts_consistent": per_stream_consistent,
        "valid_slot_histogram_consistent": histogram_consistent,
        "supported_rows": type(supported_rows) is int
        and supported_rows >= _RELATION_CREDIT_RELATION_CONFIDENT_ROWS,
        "supported_streams": type(supported_streams) is int
        and supported_streams >= _RELATION_CREDIT_RELATION_CONFIDENT_STREAMS,
        "permutation_covariant": invariants.get("permutation_covariant") is True,
        "empty_memory_zero_exact": invariants.get("empty_memory_zero_exact") is True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _relation_credit_context_gate(
    panel: Mapping[str, object],
    invariants: Mapping[str, object],
) -> dict[str, object]:
    relation = _relation_credit_relation_gate(panel, invariants)
    checks = {
        "relation_boundary_retained": relation["passed"] is True,
        "context_valid_set_top_one_fraction_supported": panel.get(
            "context_valid_set_top_one_fraction_supported", -1.0
        )
        >= _RELATION_CREDIT_CONTEXT_TOP_ONE,
        "context_valid_set_mass_mean_supported": panel.get(
            "context_valid_set_mass_mean_supported", -1.0
        )
        >= _RELATION_CREDIT_CONTEXT_MASS,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relation_gate": relation,
    }


def _relation_credit_final_gate(
    panel: Mapping[str, object],
    invariants: Mapping[str, object],
    *,
    shared_parameters_bit_exact: bool,
) -> dict[str, object]:
    relation = _relation_credit_relation_gate(panel, invariants)
    checks = {
        "relation_boundary_retained": relation["passed"] is True,
        "positive_margin_mean": panel.get("positive_margin_mean", -math.inf)
        >= _RELATION_FIT_MARGIN,
        "negative_margin_mean": panel.get("negative_margin_mean", math.inf)
        <= -_RELATION_FIT_MARGIN,
        "separation_mean": panel.get("separation_mean", -math.inf)
        >= 2.0 * _RELATION_FIT_MARGIN,
        "signed_rows": panel.get("signed_rows", -1)
        >= _RELATION_CREDIT_FINAL_SIGNED_ROWS,
        "signed_streams": panel.get("streams_with_three_signed_rows", -1)
        >= _RELATION_CREDIT_FINAL_SIGNED_STREAMS,
        "context_valid_set_top_one_fraction_supported": panel.get(
            "context_valid_set_top_one_fraction_supported", -1.0
        )
        >= _RELATION_CREDIT_CONTEXT_TOP_ONE,
        "context_valid_set_mass_mean_supported": panel.get(
            "context_valid_set_mass_mean_supported", -1.0
        )
        >= _RELATION_CREDIT_CONTEXT_MASS,
        "permutation_covariant": invariants.get("permutation_covariant") is True,
        "empty_memory_zero_exact": invariants.get("empty_memory_zero_exact") is True,
        "shared_parameters_bit_exact": shared_parameters_bit_exact,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relation_gate": relation,
    }


def fit_public_relation_matcher(
    controller: SoftwarePipelineController,
) -> dict[str, object]:
    """Execute the one enforcing v11 R80/C25/J35 public-credit protocol."""

    plan = public_relation_fit_plan()
    if not _public_relation_credit_controller_is_fresh(controller):
        raise RuntimeError("public-credit v11 requires its exact clean initialization")
    commitments = plan["commitments"]
    seed_batches = plan["stage_seed_batches"]
    relation_panel_pairs = plan["relation_context_panel_seed_pairs"]
    final_panel_pairs = plan["final_panel_seed_pairs"]
    assert isinstance(commitments, tuple)
    assert isinstance(seed_batches, dict)
    assert isinstance(relation_panel_pairs, tuple)
    assert isinstance(final_panel_pairs, tuple)
    train_pairs = {
        pair
        for batches in seed_batches.values()
        for batch in batches
        for pair in batch
    }
    if (
        train_pairs & set(relation_panel_pairs)
        or train_pairs & set(final_panel_pairs)
        or set(relation_panel_pairs) & set(final_panel_pairs)
    ):
        raise RuntimeError("public-credit training and panel seeds overlap")

    mutable_names = set(_relation_credit_parameter_names(controller, "joint"))
    shared_before = {
        name: parameter.detach().clone()
        for name, parameter in controller.named_parameters()
        if name not in mutable_names
    }

    def shared_parameters_are_exact() -> bool:
        current = dict(controller.named_parameters())
        return all(
            torch.equal(before, current[name].detach())
            for name, before in shared_before.items()
        )

    reports: dict[str, object] = {}
    panels: dict[str, object] = {}
    gates: dict[str, object] = {}

    def require_stage_exposure(report: Mapping[str, object], stage: str) -> None:
        updates = _RELATION_CREDIT_STAGE_UPDATES[stage]
        streams_per_update = _RELATION_CREDIT_STREAMS_PER_UPDATE
        stream_losses = report.get("stream_losses")
        stream_weights = report.get("stream_gradient_weights")
        flat_means = report.get("flat_mean_losses")
        entropic_terms = report.get("entropic_terms")
        effective_counts = report.get("effective_stream_counts")
        row_losses = report.get("row_losses")
        row_weights = report.get("row_gradient_weights")
        row_flat_means = report.get("row_flat_mean_losses")
        row_entropic_terms = report.get("row_entropic_terms")
        effective_row_counts = report.get("effective_row_counts")
        objectives = report.get("losses")
        gradient_norms = report.get("gradient_norms")
        mean_gradient_norm = report.get("mean_gradient_norm")
        first_loss = report.get("first_loss")
        last_loss = report.get("last_loss")
        context_rows_per_stream = report.get(
            "context_supported_rows_per_stream"
        )
        context_supported_rows = report.get("context_supported_rows")
        context_diagnostic_fields = (
            "context_responsibility_valid_set_mass",
            "context_responsibility_argmax_in_valid_fraction",
            "context_null_mass",
            "context_valid_set_mass",
            "context_valid_set_real_normalized_mass",
            "context_valid_set_top_one_fraction",
        )

        def finite_history(value: object) -> bool:
            return (
                isinstance(value, (tuple, list))
                and len(value) == updates
                and all(
                    type(item) in (int, float) and math.isfinite(float(item))
                    for item in value
                )
            )

        def entropic_diagnostics_match(
            loss_values: object,
            flat_value: object,
            entropic_value: object,
            weight_values: object,
            effective_value: object,
            *,
            temperature: float,
            mean_weight: float,
            robust_weight: float,
        ) -> bool:
            if (
                not isinstance(loss_values, (tuple, list))
                or not isinstance(weight_values, (tuple, list))
                or len(loss_values) == 0
                or len(weight_values) != len(loss_values)
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    for value in (*loss_values, *weight_values)
                )
                or type(flat_value) not in (int, float)
                or type(entropic_value) not in (int, float)
                or type(effective_value) not in (int, float)
                or not all(
                    math.isfinite(float(value))
                    for value in (flat_value, entropic_value, effective_value)
                )
            ):
                return False
            losses = tuple(float(value) for value in loss_values)
            scaled = tuple(value / temperature for value in losses)
            maximum = max(scaled)
            exponentials = tuple(math.exp(value - maximum) for value in scaled)
            denominator = sum(exponentials)
            expected_flat = sum(losses) / len(losses)
            expected_entropic = temperature * (
                maximum + math.log(denominator) - math.log(len(losses))
            )
            expected_weights = tuple(
                mean_weight / len(losses)
                + robust_weight * exponential / denominator
                for exponential in exponentials
            )
            expected_effective = 1.0 / sum(
                value * value for value in expected_weights
            )
            return (
                abs(float(flat_value) - expected_flat) <= 1.0e-6
                and abs(float(entropic_value) - expected_entropic) <= 1.0e-6
                and all(
                    abs(float(actual) - expected) <= 1.0e-6
                    for actual, expected in zip(
                        weight_values,
                        expected_weights,
                        strict=True,
                    )
                )
                and abs(float(effective_value) - expected_effective) <= 1.0e-5
            )

        vector_histories_valid = (
            isinstance(stream_losses, (tuple, list))
            and isinstance(stream_weights, (tuple, list))
            and len(stream_losses) == updates
            and len(stream_weights) == updates
            and all(
                isinstance(loss_row, (tuple, list))
                and isinstance(weight_row, (tuple, list))
                and len(loss_row) == streams_per_update
                and len(weight_row) == streams_per_update
                and all(
                    type(value) in (int, float) and math.isfinite(float(value))
                    for value in (*loss_row, *weight_row)
                )
                and all(float(value) >= 0.0 for value in weight_row)
                and abs(sum(float(value) for value in weight_row) - 1.0)
                <= 1.0e-6
                for loss_row, weight_row in zip(
                    stream_losses,
                    stream_weights,
                    strict=True,
                )
            )
        )
        scalar_histories_valid = all(
            finite_history(value)
            for value in (
                flat_means,
                entropic_terms,
                effective_counts,
                objectives,
            )
        )
        outer_risk_history_valid = (
            stage == "context"
            or (
                vector_histories_valid
                and scalar_histories_valid
                and all(
                    entropic_diagnostics_match(
                        loss_row,
                        flat,
                        entropic,
                        weight_row,
                        effective,
                        temperature=_RELATION_CREDIT_STREAM_TEMPERATURE,
                        mean_weight=_RELATION_CREDIT_STREAM_MEAN_WEIGHT,
                        robust_weight=_RELATION_CREDIT_STREAM_ROBUST_WEIGHT,
                    )
                    for loss_row, flat, entropic, weight_row, effective in zip(
                        stream_losses,
                        flat_means,
                        entropic_terms,
                        stream_weights,
                        effective_counts,
                        strict=True,
                    )
                )
            )
        )
        gradient_history_valid = (
            finite_history(gradient_norms)
            and all(float(value) >= 0.0 for value in gradient_norms)
            and type(mean_gradient_norm) in (int, float)
            and math.isfinite(float(mean_gradient_norm))
            and float(mean_gradient_norm) >= 0.0
            and abs(
                float(mean_gradient_norm)
                - sum(float(value) for value in gradient_norms) / updates
            )
            <= 1.0e-6
        )
        objective_history_valid = scalar_histories_valid and all(
            abs(
                float(objective)
                - (
                    float(flat)
                    if stage == "context"
                    else _RELATION_CREDIT_STREAM_MEAN_WEIGHT * float(flat)
                    + _RELATION_CREDIT_STREAM_ROBUST_WEIGHT * float(entropic)
                )
            )
            <= 1.0e-6
            and 0.0 < float(effective) <= streams_per_update + 1.0e-6
            for objective, flat, entropic, effective in zip(
                objectives if isinstance(objectives, (tuple, list)) else (),
                flat_means if isinstance(flat_means, (tuple, list)) else (),
                entropic_terms
                if isinstance(entropic_terms, (tuple, list))
                else (),
                effective_counts
                if isinstance(effective_counts, (tuple, list))
                else (),
                strict=True,
            )
        )
        endpoint_history_valid = (
            finite_history(objectives)
            and type(first_loss) in (int, float)
            and type(last_loss) in (int, float)
            and math.isfinite(float(first_loss))
            and math.isfinite(float(last_loss))
            and abs(float(first_loss) - float(objectives[0])) <= 1.0e-6
            and abs(float(last_loss) - float(objectives[-1])) <= 1.0e-6
        )
        if stage in ("relation", "joint"):
            row_risk_history_valid = (
                isinstance(row_losses, (tuple, list))
                and isinstance(row_weights, (tuple, list))
                and isinstance(row_flat_means, (tuple, list))
                and isinstance(row_entropic_terms, (tuple, list))
                and isinstance(effective_row_counts, (tuple, list))
                and isinstance(stream_losses, (tuple, list))
                and len(stream_losses) == updates
                and len(row_losses) == updates
                and len(row_weights) == updates
                and len(row_flat_means) == updates
                and len(row_entropic_terms) == updates
                and len(effective_row_counts) == updates
                and all(
                    isinstance(loss_groups, (tuple, list))
                    and isinstance(weight_groups, (tuple, list))
                    and isinstance(flat_groups, (tuple, list))
                    and isinstance(entropic_groups, (tuple, list))
                    and isinstance(effective_groups, (tuple, list))
                    and isinstance(stream_loss_groups, (tuple, list))
                    and len(loss_groups) == streams_per_update
                    and len(weight_groups) == streams_per_update
                    and len(flat_groups) == streams_per_update
                    and len(entropic_groups) == streams_per_update
                    and len(effective_groups) == streams_per_update
                    and len(stream_loss_groups) == streams_per_update
                    and all(
                        isinstance(loss_row, (tuple, list))
                        and isinstance(weight_row, (tuple, list))
                        and len(loss_row) == 4
                        and len(weight_row) == 4
                        and all(
                            type(value) in (int, float)
                            and math.isfinite(float(value))
                            for value in (*loss_row, *weight_row)
                        )
                        and all(
                            float(value)
                            >= _RELATION_CREDIT_ROW_MEAN_WEIGHT / 4 - 1.0e-7
                            for value in weight_row
                        )
                        and abs(sum(float(value) for value in weight_row) - 1.0)
                        <= 1.0e-6
                        and type(flat) in (int, float)
                        and type(entropic) in (int, float)
                        and type(effective) in (int, float)
                        and type(stream_loss) in (int, float)
                        and all(
                            math.isfinite(float(value))
                            for value in (flat, entropic, effective, stream_loss)
                        )
                        and abs(
                            float(flat)
                            - sum(float(value) for value in loss_row) / 4
                        )
                        <= 1.0e-6
                        and abs(
                            float(stream_loss)
                            - (
                                _RELATION_CREDIT_ROW_MEAN_WEIGHT * float(flat)
                                + _RELATION_CREDIT_ROW_ROBUST_WEIGHT
                                * float(entropic)
                            )
                        )
                        <= 1.0e-6
                        and abs(
                            float(effective)
                            - 1.0
                            / sum(float(value) ** 2 for value in weight_row)
                        )
                        <= 1.0e-5
                        and 0.0 < float(effective) <= 4.0 + 1.0e-6
                        and entropic_diagnostics_match(
                            loss_row,
                            flat,
                            entropic,
                            weight_row,
                            effective,
                            temperature=_RELATION_CREDIT_ROW_TEMPERATURE,
                            mean_weight=_RELATION_CREDIT_ROW_MEAN_WEIGHT,
                            robust_weight=_RELATION_CREDIT_ROW_ROBUST_WEIGHT,
                        )
                        for loss_row, weight_row, flat, entropic, effective, stream_loss in zip(
                            loss_groups,
                            weight_groups,
                            flat_groups,
                            entropic_groups,
                            effective_groups,
                            stream_loss_groups,
                            strict=True,
                        )
                    )
                    for loss_groups, weight_groups, flat_groups, entropic_groups, effective_groups, stream_loss_groups in zip(
                        row_losses,
                        row_weights,
                        row_flat_means,
                        row_entropic_terms,
                        effective_row_counts,
                        stream_losses,
                        strict=True,
                    )
                )
            )
        else:
            row_risk_history_valid = (
                row_losses == ()
                and row_weights == ()
                and row_flat_means == ()
                and row_entropic_terms == ()
                and effective_row_counts == ()
            )
        if stage == "context":
            context_history_valid = (
                isinstance(context_rows_per_stream, (tuple, list))
                and len(context_rows_per_stream) == updates
                and finite_history(context_supported_rows)
                and all(type(total) is int for total in context_supported_rows)
                and all(
                    isinstance(counts, (tuple, list))
                    and len(counts) == streams_per_update
                    and all(type(value) is int and 0 <= value <= 4 for value in counts)
                    and sum(counts) == total
                    and total > 0
                    for counts, total in zip(
                        context_rows_per_stream,
                        context_supported_rows,
                        strict=True,
                    )
                )
                and all(
                    finite_history(report.get(field))
                    and all(
                        -1.0e-6 <= float(value) <= 1.0 + 1.0e-6
                        for value in report[field]
                    )
                    for field in context_diagnostic_fields
                )
            )
            context_objective_reconstruction_valid = (
                context_history_valid
                and vector_histories_valid
                and scalar_histories_valid
                and all(
                    all(
                        count > 0 or abs(float(stream_loss)) <= 1.0e-7
                        for stream_loss, count in zip(
                            loss_row,
                            counts,
                            strict=True,
                        )
                    )
                    and abs(
                        float(flat)
                        - sum(
                            float(stream_loss) * count / total
                            for stream_loss, count in zip(
                                loss_row,
                                counts,
                                strict=True,
                            )
                        )
                    )
                    <= 1.0e-6
                    and abs(
                        float(objective)
                        - sum(
                            float(stream_loss) * count / total
                            for stream_loss, count in zip(
                                loss_row,
                                counts,
                                strict=True,
                            )
                        )
                    )
                    <= 1.0e-6
                    for loss_row, counts, total, flat, objective in zip(
                        stream_losses,
                        context_rows_per_stream,
                        context_supported_rows,
                        flat_means,
                        objectives,
                        strict=True,
                    )
                )
            )
            weight_policy_valid = (
                vector_histories_valid
                and context_history_valid
                and all(
                    all(
                        abs(float(weight) - count / total) <= 1.0e-6
                        for weight, count in zip(
                            weight_row,
                            counts,
                            strict=True,
                        )
                    )
                    for weight_row, counts, total in zip(
                        stream_weights,
                        context_rows_per_stream,
                        context_supported_rows,
                        strict=True,
                    )
                )
            )
        else:
            context_objective_reconstruction_valid = True
            context_history_valid = (
                context_rows_per_stream == ()
                and context_supported_rows == ()
                and all(report.get(field) == () for field in context_diagnostic_fields)
            )
            weight_policy_valid = vector_histories_valid and all(
                all(
                    float(value)
                    >= _RELATION_CREDIT_STREAM_MEAN_WEIGHT / streams_per_update
                    - 1.0e-7
                    for value in weight_row
                )
                for weight_row in stream_weights
            )
        if (
            report.get("stage") != stage
            or report.get("optimizer_steps") != updates
            or report.get("streams")
            != updates * streams_per_update
            or report.get("rows")
            != updates * streams_per_update * 4
            or report.get("robust_stream_objective_applied")
            is not (stage in ("relation", "joint"))
            or report.get("robust_row_objective_applied")
            is not (stage in ("relation", "joint"))
            or not vector_histories_valid
            or not scalar_histories_valid
            or not outer_risk_history_valid
            or not gradient_history_valid
            or not objective_history_valid
            or not endpoint_history_valid
            or not row_risk_history_valid
            or not context_history_valid
            or not context_objective_reconstruction_valid
            or not weight_policy_valid
            or report.get("frozen_parameters_unchanged") is not True
        ):
            raise RuntimeError(f"{stage} stage exposure accounting changed")

    relation_batches = _relation_credit_stream_batches(
        commitments,
        seed_batches["relation"],
    )
    relation_fit = _fit_public_relation_credit_batches(
        controller,
        relation_batches,
        stage="relation",
    )
    require_stage_exposure(relation_fit, "relation")
    reports["relation"] = {
        **relation_fit,
        "freshness_enforced_by_orchestrator": True,
    }
    relation_panel_streams = _relation_credit_panel_streams(
        commitments,
        relation_panel_pairs,
    )
    relation_panel = evaluate_public_relation_credit_panel(
        controller,
        relation_panel_streams,
    )
    relation_invariants = _evaluate_public_relation_credit_invariants(
        controller,
        relation_panel_streams,
    )
    panels["after_relation"] = {
        **relation_panel,
        "invariants": relation_invariants,
    }
    relation_gate = _relation_credit_relation_gate(
        relation_panel,
        relation_invariants,
    )
    gates["relation"] = relation_gate
    if relation_gate["passed"] is not True:
        return {
            "protocol_id": _RELATION_PROTOCOL_ID,
            "status": "STOPPED_AFTER_RELATION_GATE",
            "passed": False,
            "plan": plan,
            "stage_reports": reports,
            "panels": panels,
            "gates": gates,
            "shared_parameters_bit_exact": shared_parameters_are_exact(),
            "development_or_final_access": False,
            "wrong_evidence_training_streams": 0,
            "scalar_judge_calls": 0,
        }

    context_batches = _relation_credit_stream_batches(
        commitments,
        seed_batches["context"],
    )
    context_fit = _fit_public_relation_credit_batches(
        controller,
        context_batches,
        stage="context",
    )
    require_stage_exposure(context_fit, "context")
    reports["context"] = {
        **context_fit,
        "freshness_enforced_by_orchestrator": True,
    }
    context_panel = evaluate_public_relation_credit_panel(
        controller,
        relation_panel_streams,
    )
    context_invariants = _evaluate_public_relation_credit_invariants(
        controller,
        relation_panel_streams,
    )
    panels["after_context"] = {
        **context_panel,
        "invariants": context_invariants,
    }
    context_gate = _relation_credit_context_gate(
        context_panel,
        context_invariants,
    )
    gates["context"] = context_gate
    if context_gate["passed"] is not True:
        return {
            "protocol_id": _RELATION_PROTOCOL_ID,
            "status": "STOPPED_AFTER_CONTEXT_GATE",
            "passed": False,
            "plan": plan,
            "stage_reports": reports,
            "panels": panels,
            "gates": gates,
            "shared_parameters_bit_exact": shared_parameters_are_exact(),
            "development_or_final_access": False,
            "wrong_evidence_training_streams": 0,
            "scalar_judge_calls": 0,
        }

    joint_batches = _relation_credit_stream_batches(
        commitments,
        seed_batches["joint"],
    )
    joint_fit = _fit_public_relation_credit_batches(
        controller,
        joint_batches,
        stage="joint",
    )
    require_stage_exposure(joint_fit, "joint")
    reports["joint"] = {
        **joint_fit,
        "freshness_enforced_by_orchestrator": True,
    }
    final_panel_streams = _relation_credit_panel_streams(
        commitments,
        final_panel_pairs,
    )
    final_panel = evaluate_public_relation_credit_panel(
        controller,
        final_panel_streams,
    )
    final_invariants = _evaluate_public_relation_credit_invariants(
        controller,
        final_panel_streams,
    )
    shared_exact = shared_parameters_are_exact()
    panels["final"] = {**final_panel, "invariants": final_invariants}
    final_gate = _relation_credit_final_gate(
        final_panel,
        final_invariants,
        shared_parameters_bit_exact=shared_exact,
    )
    gates["final"] = final_gate
    return {
        "protocol_id": _RELATION_PROTOCOL_ID,
        "status": "PASSED" if final_gate["passed"] is True else "FAILED_FINAL_GATE",
        "passed": final_gate["passed"] is True,
        "plan": plan,
        "stage_reports": reports,
        "panels": panels,
        "gates": gates,
        "shared_parameters_bit_exact": shared_exact,
        "development_or_final_access": False,
        "wrong_evidence_training_streams": 0,
        "scalar_judge_calls": 0,
    }


def _conflict_cluster_pilot_runtime_assessment(
    relation_report: Mapping[str, object],
) -> dict[str, object]:
    """Apply the fixed performance-independent runtime half of the pilot rule."""

    updates = _RELATION_CREDIT_STAGE_UPDATES["relation"]
    meta_fields = tuple(
        relation_report.get(field)
        for field in (
            "meta_losses",
            "meta_flat_penalties",
            "meta_robust_penalties",
            "meta_mean_kl_from_existing_weights",
        )
    )
    geometry = relation_report.get("block_cosine_grams")
    finite_meta = all(
        isinstance(history, (tuple, list))
        and len(history) == updates
        and all(math.isfinite(float(value)) for value in history)
        for history in meta_fields
    )
    finite_geometry = isinstance(geometry, (tuple, list)) and len(geometry) == updates and all(
        math.isfinite(float(value))
        for update in geometry
        for block in update
        for row in block
        for value in row
    )
    parameter_blocks = relation_report.get("parameter_blocks")
    trainable_names = relation_report.get("trainable_parameter_names")
    complete_partition = (
        isinstance(parameter_blocks, dict)
        and bool(parameter_blocks)
        and all(parameter_blocks.values())
        and isinstance(trainable_names, (tuple, list))
        and bool(trainable_names)
        and set(
            name for names in parameter_blocks.values() for name in names
        )
        == set(trainable_names)
    )
    legacy_directions = relation_report.get("legacy_direction_digests")
    applied_directions = relation_report.get("applied_direction_digests")
    direction_changed = (
        isinstance(legacy_directions, (tuple, list))
        and isinstance(applied_directions, (tuple, list))
        and len(legacy_directions) == updates
        and len(applied_directions) == updates
        and any(
            legacy != applied
            for legacy, applied in zip(
                legacy_directions[1:],
                applied_directions[1:],
                strict=True,
            )
        )
    )
    observations = {
        "complete_relation_r80": (
            relation_report.get("optimizer_steps") == updates
            and relation_report.get("streams")
            == updates * _RELATION_CREDIT_STREAMS_PER_UPDATE
            and relation_report.get("rows")
            == updates * _RELATION_CREDIT_STREAMS_PER_UPDATE * 4
        ),
        "finite_recorded_geometry_and_meta_components": finite_meta
        and finite_geometry,
        "frozen_controller_parameters_unchanged": relation_report.get(
            "frozen_parameters_unchanged"
        )
        is True,
        "eight_symmetric_withheld_folds_per_update": relation_report.get(
            "public_leave_one_out_folds_per_update"
        )
        == _RELATION_CREDIT_STREAMS_PER_UPDATE,
        "mixer_terminal_digest_differs_from_initial": relation_report.get(
            "mixer_initial_digest"
        )
        != relation_report.get("mixer_terminal_digest"),
        "at_least_one_post_first_weight_vector_differs_from_existing": (
            relation_report.get("post_first_existing_weight_trace_digest")
            != relation_report.get("post_first_applied_weight_trace_digest")
        ),
        "at_least_one_post_first_applied_direction_differs_from_legacy": (
            direction_changed
        ),
        "relation_parameter_partition_complete": complete_partition,
        "optimizer_ownership_separated": relation_report.get(
            "controller_step_mixer_unchanged"
        )
        is True
        and relation_report.get("mixer_step_controller_unchanged") is True,
    }
    return {
        "selection_basis": "mechanistic_integrity_not_v12_performance",
        "runtime_preconditions_passed": all(observations.values()),
        "runtime_observations": observations,
        "relation_or_final_gate_used_as_go_threshold": False,
        "external_requirements_remaining": (
            "passing_fixed_pre_run_unit_receipt",
            "reloadable_combined_controller_competence_mixer_checkpoint",
        ),
    }


def fit_public_relation_conflict_matcher(
    controller: SoftwarePipelineController,
    mixer: AnonymousConflictMixer,
) -> dict[str, object]:
    """Execute the one fixed v12 learned conflict-reconciliation protocol."""

    plan = public_relation_conflict_fit_plan()
    if not _public_relation_conflict_system_is_fresh(controller, mixer):
        raise RuntimeError("v12 conflict protocol requires its exact clean system")
    initial_controller_digest = software_pipeline_model_digest(controller)
    initial_mixer_digest = anonymous_conflict_mixer_digest(mixer)
    commitments = plan["commitments"]
    seed_batches = plan["stage_seed_batches"]
    relation_panel_pairs = plan["relation_context_panel_seed_pairs"]
    final_panel_pairs = plan["final_panel_seed_pairs"]
    assert isinstance(commitments, tuple)
    assert isinstance(seed_batches, dict)
    assert isinstance(relation_panel_pairs, tuple)
    assert isinstance(final_panel_pairs, tuple)
    train_pairs = {
        pair
        for batches in seed_batches.values()
        for batch in batches
        for pair in batch
    }
    if (
        train_pairs & set(relation_panel_pairs)
        or train_pairs & set(final_panel_pairs)
        or set(relation_panel_pairs) & set(final_panel_pairs)
    ):
        raise RuntimeError("v12 training and panel identities overlap")
    mutable_names = set(_relation_credit_parameter_names(controller, "joint"))
    shared_before = {
        name: parameter.detach().clone()
        for name, parameter in controller.named_parameters()
        if name not in mutable_names
    }

    def shared_parameters_are_exact() -> bool:
        current = dict(controller.named_parameters())
        return all(
            torch.equal(before, current[name].detach())
            for name, before in shared_before.items()
        )

    def terminal_lineage() -> dict[str, object]:
        return {
            "initial_controller_digest": initial_controller_digest,
            "terminal_controller_digest": software_pipeline_model_digest(controller),
            "initial_mixer_digest": initial_mixer_digest,
            "terminal_mixer_digest": anonymous_conflict_mixer_digest(mixer),
            "parameter_report": public_relation_conflict_parameter_report(
                controller,
                mixer,
            ),
            "combined_checkpoint_required": True,
        }

    def require_conflict_stage(
        report: Mapping[str, object],
        stage: str,
    ) -> None:
        updates = _RELATION_CREDIT_STAGE_UPDATES[stage]
        blocks = len(_conflict_parameter_blocks(controller, stage))
        require_legacy = stage == "relation"
        if (
            report.get("stage") != stage
            or report.get("optimizer_steps") != updates
            or report.get("streams")
            != updates * _RELATION_CREDIT_STREAMS_PER_UPDATE
            or report.get("rows")
            != updates * _RELATION_CREDIT_STREAMS_PER_UPDATE * 4
            or report.get("legacy_first_update_required") is not require_legacy
            or report.get("first_update_used_legacy_weights") is not require_legacy
            or report.get("mixer_parameters_changed") is not True
            or report.get("frozen_parameters_unchanged") is not True
            or report.get("controller_step_mixer_unchanged") is not True
            or report.get("mixer_step_controller_unchanged") is not True
            or report.get("public_leave_one_out_folds_per_update")
            != _RELATION_CREDIT_STREAMS_PER_UPDATE
            or report.get("stream_identity_input") is not False
            or report.get("task_identity_input") is not False
            or report.get("deterministic_gradient_projection") is not False
        ):
            raise RuntimeError(f"v12 {stage} exposure accounting changed")
        for field in (
            "reference_losses",
            "meta_losses",
            "meta_flat_penalties",
            "meta_robust_penalties",
            "meta_mean_kl_from_existing_weights",
            "existing_stream_weights",
            "applied_block_weights",
            "residual_logits",
            "withheld_alignments",
            "block_gradient_norms",
            "block_cosine_grams",
            "legacy_negative_alignment_fractions",
            "applied_negative_alignment_fractions",
            "legacy_cancellation_ratios",
            "applied_cancellation_ratios",
            "legacy_direction_norms",
            "applied_direction_norms",
            "legacy_direction_digests",
            "applied_direction_digests",
        ):
            history = report.get(field)
            if not isinstance(history, (tuple, list)) or len(history) != updates:
                raise RuntimeError(f"v12 {stage} lost {field}")
        for weights in report["applied_block_weights"]:
            if (
                len(weights) != blocks
                or any(len(row) != _RELATION_CREDIT_STREAMS_PER_UPDATE for row in weights)
                or any(
                    not math.isfinite(float(value)) or float(value) <= 0.0
                    for row in weights
                    for value in row
                )
                or any(abs(sum(float(value) for value in row) - 1.0) > 1.0e-5 for row in weights)
            ):
                raise RuntimeError(f"v12 {stage} produced invalid learned weights")

    reports: dict[str, object] = {}
    panels: dict[str, object] = {}
    gates: dict[str, object] = {}
    relation_batches = _relation_credit_stream_batches(
        commitments,
        seed_batches["relation"],
    )
    relation_fit = _fit_public_relation_conflict_batches(
        controller,
        mixer,
        relation_batches,
        stage="relation",
        require_legacy_first_update=True,
    )
    require_conflict_stage(relation_fit, "relation")
    reports["relation"] = {
        **relation_fit,
        "freshness_enforced_by_orchestrator": True,
    }
    relation_panel_streams = _relation_credit_panel_streams(
        commitments,
        relation_panel_pairs,
    )
    relation_panel = evaluate_public_relation_credit_panel(
        controller,
        relation_panel_streams,
    )
    relation_invariants = _evaluate_public_relation_credit_invariants(
        controller,
        relation_panel_streams,
    )
    panels["after_relation"] = {
        **relation_panel,
        "invariants": relation_invariants,
    }
    relation_gate = _relation_credit_relation_gate(
        relation_panel,
        relation_invariants,
    )
    gates["relation"] = relation_gate
    cluster_pilot_runtime = _conflict_cluster_pilot_runtime_assessment(
        relation_fit
    )
    if relation_gate["passed"] is not True:
        return {
            "protocol_id": _CONFLICT_PROTOCOL_ID,
            "status": "STOPPED_AFTER_RELATION_GATE",
            "passed": False,
            "plan": plan,
            "stage_reports": reports,
            "panels": panels,
            "gates": gates,
            "cluster_pilot_runtime_assessment": cluster_pilot_runtime,
            "mixer_parameter_count": sum(
                parameter.numel() for parameter in mixer.parameters()
            ),
            "shared_parameters_bit_exact": shared_parameters_are_exact(),
            "development_or_final_access": False,
            "wrong_evidence_training_streams": 0,
            "scalar_judge_calls": 0,
            **terminal_lineage(),
        }

    context_batches = _relation_credit_stream_batches(
        commitments,
        seed_batches["context"],
    )
    context_fit = _fit_public_relation_credit_batches(
        controller,
        context_batches,
        stage="context",
    )
    if (
        context_fit.get("optimizer_steps") != _RELATION_CREDIT_STAGE_UPDATES["context"]
        or context_fit.get("streams")
        != _RELATION_CREDIT_STAGE_UPDATES["context"] * _RELATION_CREDIT_STREAMS_PER_UPDATE
        or context_fit.get("rows")
        != _RELATION_CREDIT_STAGE_UPDATES["context"]
        * _RELATION_CREDIT_STREAMS_PER_UPDATE
        * 4
        or context_fit.get("frozen_parameters_unchanged") is not True
    ):
        raise RuntimeError("v12 context exposure accounting changed")
    reports["context"] = {
        **context_fit,
        "freshness_enforced_by_orchestrator": True,
    }
    context_panel = evaluate_public_relation_credit_panel(
        controller,
        relation_panel_streams,
    )
    context_invariants = _evaluate_public_relation_credit_invariants(
        controller,
        relation_panel_streams,
    )
    panels["after_context"] = {
        **context_panel,
        "invariants": context_invariants,
    }
    context_gate = _relation_credit_context_gate(
        context_panel,
        context_invariants,
    )
    gates["context"] = context_gate
    if context_gate["passed"] is not True:
        return {
            "protocol_id": _CONFLICT_PROTOCOL_ID,
            "status": "STOPPED_AFTER_CONTEXT_GATE",
            "passed": False,
            "plan": plan,
            "stage_reports": reports,
            "panels": panels,
            "gates": gates,
            "cluster_pilot_runtime_assessment": cluster_pilot_runtime,
            "mixer_parameter_count": sum(
                parameter.numel() for parameter in mixer.parameters()
            ),
            "shared_parameters_bit_exact": shared_parameters_are_exact(),
            "development_or_final_access": False,
            "wrong_evidence_training_streams": 0,
            "scalar_judge_calls": 0,
            **terminal_lineage(),
        }

    joint_batches = _relation_credit_stream_batches(
        commitments,
        seed_batches["joint"],
    )
    joint_fit = _fit_public_relation_conflict_batches(
        controller,
        mixer,
        joint_batches,
        stage="joint",
        require_legacy_first_update=False,
    )
    require_conflict_stage(joint_fit, "joint")
    reports["joint"] = {
        **joint_fit,
        "freshness_enforced_by_orchestrator": True,
    }
    final_panel_streams = _relation_credit_panel_streams(
        commitments,
        final_panel_pairs,
    )
    final_panel = evaluate_public_relation_credit_panel(
        controller,
        final_panel_streams,
    )
    final_invariants = _evaluate_public_relation_credit_invariants(
        controller,
        final_panel_streams,
    )
    shared_exact = shared_parameters_are_exact()
    panels["final"] = {**final_panel, "invariants": final_invariants}
    final_gate = _relation_credit_final_gate(
        final_panel,
        final_invariants,
        shared_parameters_bit_exact=shared_exact,
    )
    gates["final"] = final_gate
    return {
        "protocol_id": _CONFLICT_PROTOCOL_ID,
        "status": "PASSED" if final_gate["passed"] is True else "FAILED_FINAL_GATE",
        "passed": final_gate["passed"] is True,
        "plan": plan,
        "stage_reports": reports,
        "panels": panels,
        "gates": gates,
        "cluster_pilot_runtime_assessment": cluster_pilot_runtime,
        "mixer_parameter_count": sum(
            parameter.numel() for parameter in mixer.parameters()
        ),
        "shared_parameters_bit_exact": shared_exact,
        "development_or_final_access": False,
        "wrong_evidence_training_streams": 0,
        "scalar_judge_calls": 0,
        **terminal_lineage(),
    }


def _relation_panel_pair_summary(
    panel_a: Mapping[str, object],
    panel_b: Mapping[str, object],
) -> dict[str, float | int]:
    for panel in (panel_a, panel_b):
        if panel.get("rows") != 32 or panel.get("streams") != 8:
            raise RuntimeError("paired cluster panel shape changed")
    return {
        "supported_rows": int(panel_a["relation_supported_rows"])
        + int(panel_b["relation_supported_rows"]),
        "qualifying_streams": int(panel_a["streams_with_three_supported_rows"])
        + int(panel_b["streams_with_three_supported_rows"]),
        "target_loss_mean": (
            float(panel_a["target_loss_mean"])
            + float(panel_b["target_loss_mean"])
        )
        / 2.0,
        "target_witness_mean": (
            float(panel_a["target_witness_mean"])
            + float(panel_b["target_witness_mean"])
        )
        / 2.0,
    }


def _relation_surface_stability(
    original: Mapping[str, object],
    rerendered: Mapping[str, object],
) -> dict[str, object]:
    left_rows = original.get("row_reports")
    right_rows = rerendered.get("row_reports")
    if (
        not isinstance(left_rows, (tuple, list))
        or not isinstance(right_rows, (tuple, list))
        or len(left_rows) != len(right_rows)
        or not left_rows
    ):
        raise ValueError("surface stability requires aligned nonempty panels")
    discrete_fields = (
        "target_slot",
        "valid_slots",
        "valid_slot_count",
        "relation_supported",
        "context_valid_set_top_one",
        "raw_slot",
        "unique_loss_selected_confident",
        "signed",
    )
    scalar_fields = (
        "target_positive_margin",
        "target_negative_margin",
        "target_witness",
        "target_loss",
        "target_loss_gap",
        "target_responsibility",
        "context_target_mass",
        "context_null_mass",
        "context_valid_set_mass",
        "raw_positive_margin",
        "raw_negative_margin",
        "raw_witness",
        "positive_margin",
        "negative_margin",
    )
    vector_fields = (
        "slot_positive_margins",
        "slot_negative_margins",
        "slot_losses",
        "responsibilities",
        "context_weights",
    )
    discrete_exact = True
    maximum_delta = 0.0
    for left, right in zip(left_rows, right_rows, strict=True):
        discrete_exact = discrete_exact and all(
            left[field] == right[field] for field in discrete_fields
        )
        maximum_delta = max(
            maximum_delta,
            *(abs(float(left[field]) - float(right[field])) for field in scalar_fields),
        )
        for field in vector_fields:
            if len(left[field]) != len(right[field]):
                raise RuntimeError("surface rerender changed vector shape")
            maximum_delta = max(
                maximum_delta,
                *(
                    abs(float(a) - float(b))
                    for a, b in zip(left[field], right[field], strict=True)
                ),
            )
    return {
        "discrete_exact": discrete_exact,
        "continuous_max_delta": maximum_delta,
        "passed": discrete_exact and maximum_delta <= 1.0e-6,
    }


def _relation_summary_dominates(
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
) -> bool:
    no_worse = (
        int(candidate["supported_rows"]) >= int(reference["supported_rows"])
        and int(candidate["qualifying_streams"])
        >= int(reference["qualifying_streams"])
        and float(candidate["target_loss_mean"])
        <= float(reference["target_loss_mean"])
    )
    strictly_better = (
        int(candidate["supported_rows"]) > int(reference["supported_rows"])
        or int(candidate["qualifying_streams"])
        > int(reference["qualifying_streams"])
        or float(candidate["target_loss_mean"])
        < float(reference["target_loss_mean"])
    )
    return no_worse and strictly_better


def _evaluate_cluster_lesions(
    cluster: CapacityMatchedClusterController,
    panel_a_streams: Sequence[SoftwarePipelineStream],
    panel_b_streams: Sequence[SoftwarePipelineStream],
) -> dict[str, object]:
    if any(parameter.grad is not None for parameter in cluster.parameters()):
        for parameter in cluster.parameters():
            parameter.grad = None
    before = software_pipeline_model_digest(cluster)
    lesions: dict[str, object] = {}
    specifications = (("uniform", None),) + tuple(
        ("single", index) for index in range(_CLUSTER_CELL_COUNT)
    ) + tuple(("drop", index) for index in range(_CLUSTER_CELL_COUNT))
    try:
        for kind, index in specifications:
            cluster.set_relation_diagnostic_lesion(kind, index)
            panel_a = evaluate_public_relation_credit_panel(
                cluster,
                panel_a_streams,
            )
            panel_b = evaluate_public_relation_credit_panel(
                cluster,
                panel_b_streams,
            )
            label = kind if index is None else f"{kind}_{index}"
            lesions[label] = _relation_panel_pair_summary(panel_a, panel_b)
    finally:
        cluster.set_relation_diagnostic_lesion(None)
    if software_pipeline_model_digest(cluster) != before:
        raise RuntimeError("read-only cluster lesion changed learned parameters")
    lesions["composer_removed"] = dict(lesions["uniform"])
    return lesions


def fit_capacity_matched_relation_cluster_pilot(
    *,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """Run all fixed paired R80 comparisons, regardless of intermediate score."""

    plan = capacity_matched_relation_cluster_fit_plan()
    commitments = plan["commitments"]
    assert isinstance(commitments, tuple)
    systems = []
    replicate_reports = []
    started = time.perf_counter()
    for replicate_spec in plan["replicates"]:
        replicate = int(replicate_spec["replicate"])
        system = build_capacity_matched_relation_cluster_pair(
            replicate,
            device=device,
        )
        monolith, cluster, monolith_mixer, cluster_mixer = system
        systems.append(system)
        parameter_report = capacity_matched_relation_cluster_parameter_report(
            monolith,
            cluster,
            monolith_mixer,
            cluster_mixer,
        )
        if (
            parameter_report["within_declared_tolerance"] is not True
            or parameter_report["shared_non_relation_parameters_bit_exact"]
            is not True
            or parameter_report["inert_padding_parameters"] != 0
        ):
            raise RuntimeError("cluster pilot lost capacity or shared-state parity")
        initial = {
            "monolith": software_pipeline_model_digest(monolith),
            "cluster": software_pipeline_model_digest(cluster),
            "monolith_mixer": anonymous_conflict_mixer_digest(monolith_mixer),
            "cluster_mixer": anonymous_conflict_mixer_digest(cluster_mixer),
            "cells": tuple(
                _learned_module_digest(cell) for cell in cluster.relation_cells
            ),
            "composer": _learned_module_digest(cluster.relation_composer),
        }
        if initial["monolith_mixer"] != initial["cluster_mixer"]:
            raise RuntimeError("paired mixer initialization changed")
        train_batches = _relation_credit_stream_batches(
            commitments,
            replicate_spec["train_seed_batches"],
        )
        fits: dict[str, object] = {}
        for arm in replicate_spec["arm_order"]:
            if arm == "monolith":
                fits[arm] = _fit_public_relation_conflict_batches(
                    monolith,
                    monolith_mixer,
                    train_batches,
                    stage="relation",
                    require_legacy_first_update=True,
                )
            elif arm == "cluster":
                fits[arm] = _fit_public_relation_conflict_batches(
                    cluster,
                    cluster_mixer,
                    train_batches,
                    stage="relation",
                    require_legacy_first_update=True,
                )
            else:
                raise RuntimeError("cluster pilot arm identity changed")
        for arm, fit in fits.items():
            expected_blocks = (
                plan["monolith_parameter_blocks"]
                if arm == "monolith"
                else plan["cluster_parameter_blocks"]
            )
            if (
                fit.get("optimizer_steps") != 80
                or fit.get("streams") != 640
                or fit.get("rows") != 2_560
                or tuple(fit.get("parameter_blocks", ())) != expected_blocks
                or fit.get("public_leave_one_out_folds_per_update") != 8
                or fit.get("first_update_used_legacy_weights") is not True
                or fit.get("frozen_parameters_unchanged") is not True
            ):
                raise RuntimeError(f"cluster pilot {arm} exposure changed")
        panel_a_streams = _relation_credit_panel_streams(
            commitments,
            replicate_spec["panel_a_seed_pairs"],
        )
        panel_a_rerender_streams = _relation_credit_panel_streams(
            commitments,
            replicate_spec["panel_a_rerender_seed_pairs"],
        )
        panel_b_streams = _relation_credit_panel_streams(
            commitments,
            replicate_spec["panel_b_seed_pairs"],
        )
        monolith_panels = {
            "panel_a": evaluate_public_relation_credit_panel(
                monolith,
                panel_a_streams,
            ),
            "panel_a_rerender": evaluate_public_relation_credit_panel(
                monolith,
                panel_a_rerender_streams,
            ),
            "panel_b": evaluate_public_relation_credit_panel(
                monolith,
                panel_b_streams,
            ),
        }
        cluster.begin_relation_diagnostics()
        cluster_panel_a = evaluate_public_relation_credit_panel(
            cluster,
            panel_a_streams,
        )
        cluster_panel_b = evaluate_public_relation_credit_panel(
            cluster,
            panel_b_streams,
        )
        cluster_diagnostics = cluster.end_relation_diagnostics()
        cluster_panels = {
            "panel_a": cluster_panel_a,
            "panel_a_rerender": evaluate_public_relation_credit_panel(
                cluster,
                panel_a_rerender_streams,
            ),
            "panel_b": cluster_panel_b,
        }
        monolith_summary = _relation_panel_pair_summary(
            monolith_panels["panel_a"],
            monolith_panels["panel_b"],
        )
        cluster_summary = _relation_panel_pair_summary(
            cluster_panels["panel_a"],
            cluster_panels["panel_b"],
        )
        lesions = _evaluate_cluster_lesions(
            cluster,
            panel_a_streams,
            panel_b_streams,
        )
        terminal = {
            "monolith": software_pipeline_model_digest(monolith),
            "cluster": software_pipeline_model_digest(cluster),
            "monolith_mixer": anonymous_conflict_mixer_digest(monolith_mixer),
            "cluster_mixer": anonymous_conflict_mixer_digest(cluster_mixer),
            "cells": tuple(
                _learned_module_digest(cell) for cell in cluster.relation_cells
            ),
            "composer": _learned_module_digest(cluster.relation_composer),
            "system": capacity_matched_relation_cluster_system_digest(
                monolith,
                cluster,
                monolith_mixer,
                cluster_mixer,
                replicate,
            ),
        }
        cell_and_composer_changed = (
            all(
                before != after
                for before, after in zip(
                    initial["cells"],
                    terminal["cells"],
                    strict=True,
                )
            )
            and initial["composer"] != terminal["composer"]
        )
        block_activity = tuple(
            any(
                float(update[index][stream]) > 0.0
                for update in fits["cluster"]["block_gradient_norms"]
                for stream in range(_RELATION_CREDIT_STREAMS_PER_UPDATE)
            )
            for index in range(_CLUSTER_CELL_COUNT + 1)
        )
        replicate_reports.append(
            {
                "replicate": replicate,
                "arm_order": replicate_spec["arm_order"],
                "stream_binding_digest": replicate_spec[
                    "monolith_stream_binding_digest"
                ],
                "paired_stream_binding_exact": replicate_spec[
                    "monolith_stream_binding_digest"
                ]
                == replicate_spec["cluster_stream_binding_digest"],
                "parameter_report": parameter_report,
                "initial_digests": initial,
                "terminal_digests": terminal,
                "cell_and_composer_digests_changed": cell_and_composer_changed,
                "all_five_cluster_blocks_received_gradient": all(block_activity),
                "cluster_block_activity": block_activity,
                "shared_non_relation_parameters_bit_exact": (
                    _capacity_matched_shared_parameters_are_exact(
                        monolith,
                        cluster,
                    )
                ),
                "fits": fits,
                "panels": {
                    "monolith": monolith_panels,
                    "cluster": cluster_panels,
                },
                "summaries": {
                    "monolith": monolith_summary,
                    "cluster": cluster_summary,
                },
                "cluster_diagnostics": cluster_diagnostics,
                "surface_stability": {
                    "monolith": _relation_surface_stability(
                        monolith_panels["panel_a"],
                        monolith_panels["panel_a_rerender"],
                    ),
                    "cluster": _relation_surface_stability(
                        cluster_panels["panel_a"],
                        cluster_panels["panel_a_rerender"],
                    ),
                },
                "cluster_lesions": lesions,
                "cluster_learned_fusion_dominates_all_lesions": all(
                    _relation_summary_dominates(cluster_summary, lesion)
                    for lesion in lesions.values()
                ),
                "context_or_joint_training_performed": False,
                "normal_runtime_memory_used": False,
            }
        )
    if checkpoint_path is not None:
        save_capacity_matched_relation_cluster_checkpoint(
            checkpoint_path,
            systems,
        )
    monolith_supported = sum(
        int(report["summaries"]["monolith"]["supported_rows"])
        for report in replicate_reports
    )
    cluster_supported = sum(
        int(report["summaries"]["cluster"]["supported_rows"])
        for report in replicate_reports
    )
    monolith_streams = sum(
        int(report["summaries"]["monolith"]["qualifying_streams"])
        for report in replicate_reports
    )
    cluster_streams = sum(
        int(report["summaries"]["cluster"]["qualifying_streams"])
        for report in replicate_reports
    )
    monolith_loss = sum(
        float(report["summaries"]["monolith"]["target_loss_mean"])
        for report in replicate_reports
    ) / len(replicate_reports)
    cluster_loss = sum(
        float(report["summaries"]["cluster"]["target_loss_mean"])
        for report in replicate_reports
    ) / len(replicate_reports)
    nonregressing = sum(
        int(report["summaries"]["cluster"]["supported_rows"])
        >= int(report["summaries"]["monolith"]["supported_rows"])
        for report in replicate_reports
    )
    checks = {
        "aggregate_supported_rows_strictly_greater": cluster_supported
        > monolith_supported,
        "aggregate_qualifying_streams_at_least_monolith": cluster_streams
        >= monolith_streams,
        "mean_target_loss_strictly_lower": cluster_loss < monolith_loss,
        "supported_rows_nonregressing_in_two_replicates": nonregressing >= 2,
        "learned_fusion_dominates_every_lesion": all(
            report["cluster_learned_fusion_dominates_all_lesions"] is True
            for report in replicate_reports
        ),
        "cluster_surface_stable": all(
            report["surface_stability"]["cluster"]["passed"] is True
            for report in replicate_reports
        ),
        "all_cell_and_composer_digests_changed": all(
            report["cell_and_composer_digests_changed"] is True
            for report in replicate_reports
        ),
        "all_cluster_blocks_received_gradient": all(
            report["all_five_cluster_blocks_received_gradient"] is True
            for report in replicate_reports
        ),
        "paired_exposure_and_shared_state_exact": all(
            report["paired_stream_binding_exact"] is True
            and report["shared_non_relation_parameters_bit_exact"] is True
            for report in replicate_reports
        ),
    }
    supportive = all(checks.values())
    harmful = (
        cluster_supported <= monolith_supported
        and cluster_streams <= monolith_streams
        and cluster_loss >= monolith_loss
        and (
            cluster_supported < monolith_supported
            or cluster_streams < monolith_streams
            or cluster_loss > monolith_loss
        )
    )
    return {
        "protocol_id": _CLUSTER_PROTOCOL_ID,
        "status": (
            "CLUSTER_SUPPORTED"
            if supportive
            else "CLUSTER_HARMFUL"
            if harmful
            else "CLUSTER_INCONCLUSIVE"
        ),
        "cluster_supported": supportive,
        "plan": plan,
        "replicates": tuple(replicate_reports),
        "aggregate": {
            "monolith_supported_rows": monolith_supported,
            "cluster_supported_rows": cluster_supported,
            "monolith_qualifying_streams": monolith_streams,
            "cluster_qualifying_streams": cluster_streams,
            "monolith_target_loss_mean": monolith_loss,
            "cluster_target_loss_mean": cluster_loss,
            "supported_row_nonregressing_replicates": nonregressing,
        },
        "support_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_written": checkpoint_path is not None,
        "context_or_joint_training_performed": False,
        "historical_v12_performance_used_as_control": False,
        "development_or_final_access": False,
        "wrong_evidence_training_streams": 0,
        "scalar_judge_calls": 0,
        "deterministic_solver_used": False,
    }


def _evaluate_public_relation_pre_smoke_gate_v2_audit_only(
    controller: SoftwarePipelineController,
) -> dict[str, object]:
    """Evaluate the retired v2 wrong-evidence gate for historical audit only."""

    plan = _legacy_public_relation_fit_plan()
    commitments = plan["gate_commitments"]
    assert isinstance(commitments, tuple)
    streams = _scheduled_relation_streams(
        commitments,
        _RELATION_GATE_SEED_PAIRS,
    )
    base_rows = []
    permutation_delta = 0.0
    rerender_pairs = []
    controller.eval()
    with torch.no_grad():
        for stream_index, stream in enumerate(streams):
            rows = public_paired_relation_fit_rows(controller, stream)
            permuted = public_paired_relation_fit_rows(
                controller,
                stream,
                reverse_evidence_order=True,
                reverse_public_presentation=True,
            )
            topology_seed, surface_seed = _RELATION_GATE_SEED_PAIRS[
                stream_index % len(_RELATION_GATE_SEED_PAIRS)
            ]
            commitment = commitments[
                stream_index // len(_RELATION_GATE_SEED_PAIRS)
            ]
            rerendered_stream = make_software_pipeline_stream(
                topology_seed,
                surface_seed=surface_seed + 10_000_000,
                supports_per_motif=2,
                queries=1,
                maximum_steps=4,
                mechanism_commitment=commitment,
                mechanism_partition="train",
            )
            rerendered = public_paired_relation_fit_rows(
                controller,
                rerendered_stream,
            )
            for row, covariance, rerender in zip(
                rows,
                permuted,
                rerendered,
                strict=True,
            ):
                if (
                    row.heldout_index != covariance.heldout_index
                    or row.transition_index != covariance.transition_index
                    or row.heldout_index != rerender.heldout_index
                    or row.transition_index != rerender.transition_index
                ):
                    raise RuntimeError("relation gate row alignment changed")
                positive = float(row.positive_margin.item())
                negative = float(row.negative_margin.item())
                covariance_positive = float(covariance.positive_margin.item())
                covariance_negative = float(covariance.negative_margin.item())
                permutation_delta = max(
                    permutation_delta,
                    abs(positive - covariance_positive),
                    abs(negative - covariance_negative),
                )
                rerender_positive = float(rerender.positive_margin.item())
                rerender_negative = float(rerender.negative_margin.item())
                separation = positive - negative
                rerender_separation = rerender_positive - rerender_negative
                rerender_pairs.append(
                    (
                        separation,
                        rerender_separation,
                        positive,
                        negative,
                        rerender_positive,
                        rerender_negative,
                    )
                )
                base_rows.append(
                    {
                        "stream_index": stream_index,
                        "heldout_index": row.heldout_index,
                        "transition_index": row.transition_index,
                        "positive_margin": positive,
                        "negative_margin": negative,
                        "separation": separation,
                        "rerender_positive_margin": rerender_positive,
                        "rerender_negative_margin": rerender_negative,
                        "permutation_max_delta": max(
                            abs(positive - covariance_positive),
                            abs(negative - covariance_negative),
                        ),
                    }
                )
        empty_task = replace(streams[0].supports[0].learner, observations=())
        empty_encoding = controller.encode_task(empty_task)
        empty_scores = controller._relation_evidence_scores(
            empty_encoding.relation_context_embeddings,
            empty_encoding.relation_component_embeddings,
            controller.initial_state(),
        )
        nonempty_state = _acquire_public_task_set(
            controller,
            tuple(pair.learner for pair in streams[0].supports[1:]),
        )
        no_role_scores = controller.score_actions(
            empty_task,
            nonempty_state,
            encoding=empty_encoding,
            include_pointer_memory=False,
            include_role_memory=False,
        ).evidence_match_scores
    if len(base_rows) != _RELATION_GATE_ROWS:
        raise RuntimeError("relation gate exposure count changed")
    positive_mean = sum(row["positive_margin"] for row in base_rows) / len(base_rows)
    negative_mean = sum(row["negative_margin"] for row in base_rows) / len(base_rows)
    separation_mean = sum(row["separation"] for row in base_rows) / len(base_rows)
    both_signs = sum(
        row["positive_margin"] >= _RELATION_FIT_MARGIN
        and row["negative_margin"] <= -_RELATION_FIT_MARGIN
        for row in base_rows
    )
    epsilon = 1.0e-12
    rerender_retention = sum(
        min(abs(rerender) / max(abs(original), epsilon), 1.0)
        for original, rerender, *_ in rerender_pairs
    ) / len(rerender_pairs)
    rerender_sign_fraction = sum(
        (original_positive >= 0.0) == (rerender_positive >= 0.0)
        and (original_negative <= 0.0) == (rerender_negative <= 0.0)
        for (
            _,
            _,
            original_positive,
            original_negative,
            rerender_positive,
            rerender_negative,
        ) in rerender_pairs
    ) / len(rerender_pairs)
    empty_exact = bool(torch.equal(empty_scores, torch.zeros_like(empty_scores)))
    no_role_exact = bool(
        torch.equal(no_role_scores, torch.zeros_like(no_role_scores))
    )
    passed = (
        positive_mean >= _RELATION_FIT_MARGIN
        and negative_mean <= -_RELATION_FIT_MARGIN
        and both_signs >= _RELATION_GATE_SIGN_ROWS
        and separation_mean >= 2.0 * _RELATION_FIT_MARGIN
        and permutation_delta <= _RELATION_GATE_PERMUTATION_TOLERANCE
        and rerender_retention >= _RELATION_GATE_RERENDER_RETENTION
        and rerender_sign_fraction >= _RELATION_GATE_RERENDER_SIGN_FRACTION
        and empty_exact
        and no_role_exact
    )
    return {
        "passed": passed,
        "partition": "train",
        "rows": tuple(base_rows),
        "row_count": len(base_rows),
        "positive_margin_mean": positive_mean,
        "negative_margin_mean": negative_mean,
        "both_sign_rows": both_signs,
        "separation_mean": separation_mean,
        "permutation_max_delta": permutation_delta,
        "rerender_separation_retention": rerender_retention,
        "rerender_sign_fraction": rerender_sign_fraction,
        "empty_relation_score_exact_zero": empty_exact,
        "nonempty_state_no_role_score_exact_zero": no_role_exact,
        "thresholds": {
            "positive_margin_mean": _RELATION_FIT_MARGIN,
            "negative_margin_mean": -_RELATION_FIT_MARGIN,
            "both_sign_rows": _RELATION_GATE_SIGN_ROWS,
            "separation_mean": 2.0 * _RELATION_FIT_MARGIN,
            "permutation_max_delta": _RELATION_GATE_PERMUTATION_TOLERANCE,
            "rerender_separation_retention": _RELATION_GATE_RERENDER_RETENTION,
            "rerender_sign_fraction": _RELATION_GATE_RERENDER_SIGN_FRACTION,
        },
    }


def _fit_public_relation_matcher_v2_audit_only(
    controller: SoftwarePipelineController,
    *,
    learning_rate: float = 2.0e-3,
    gradient_clip: float = 5.0,
) -> dict[str, object]:
    """Execute the retired v2 wrong-evidence schedule for historical audit only."""

    plan = _legacy_public_relation_fit_plan()
    commitments = plan["fit_commitments"]
    assert isinstance(commitments, tuple)
    streams = _scheduled_relation_streams(commitments, _RELATION_FIT_SEED_PAIRS)
    probe_task = replace(streams[0].supports[0].learner, observations=())
    probe_state = _acquire_public_task_set(
        controller,
        tuple(pair.learner for pair in streams[0].supports[1:]),
    )
    probe_encoding = controller.encode_task(probe_task)
    with torch.no_grad():
        legacy_before = controller.score_actions(
            probe_task,
            probe_state,
            encoding=probe_encoding,
            include_pointer_memory=False,
            use_legacy_evidence=True,
        )
        no_memory_before = controller.score_actions(
            probe_task,
            probe_state,
            encoding=probe_encoding,
            include_pointer_memory=False,
            include_role_memory=False,
        )
    fit = _fit_public_relation_matcher_streams(
        controller,
        streams,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
    )
    if fit["row_count"] != plan["fit_rows"] or fit["directional_arms"] != plan[
        "fit_directional_arms"
    ]:
        raise RuntimeError("relation fit exposure accounting changed")
    with torch.no_grad():
        legacy_after = controller.score_actions(
            probe_task,
            probe_state,
            encoding=probe_encoding,
            include_pointer_memory=False,
            use_legacy_evidence=True,
        )
        no_memory_after = controller.score_actions(
            probe_task,
            probe_state,
            encoding=probe_encoding,
            include_pointer_memory=False,
            include_role_memory=False,
        )
    base_invariants = {
        "legacy_action_logits_bit_exact": torch.equal(
            legacy_before.action_logits,
            legacy_after.action_logits,
        ),
        "transition_successors_bit_exact": torch.equal(
            legacy_before.successor_state_logits,
            legacy_after.successor_state_logits,
        ),
        "stop_logit_bit_exact": torch.equal(
            legacy_before.stop_logit,
            legacy_after.stop_logit,
        ),
        "no_memory_logits_bit_exact": torch.equal(
            no_memory_before.logits,
            no_memory_after.logits,
        ),
    }
    gate = _evaluate_public_relation_pre_smoke_gate_v2_audit_only(controller)
    gate["base_invariants"] = base_invariants
    gate["passed"] = bool(gate["passed"]) and all(base_invariants.values())
    return {
        "plan": plan,
        "fit": fit,
        "gate": gate,
        "raw_gate_passed": gate["passed"],
        "development_or_final_access": False,
        "scalar_judge_calls": 0,
    }


def _calibrate_public_relation_actions_v2_audit_only(
    controller: SoftwarePipelineController,
    relation_fit_report: Mapping[str, object],
    *,
    learning_rate: float = 2.0e-3,
    gradient_clip: float = 5.0,
) -> dict[str, object]:
    """Run retired v2 wrong-evidence action calibration for audit only."""

    if not isinstance(relation_fit_report, Mapping) or (
        relation_fit_report.get("raw_gate_passed") is not True
    ):
        raise RuntimeError("paired action calibration requires the passed fit report")
    plan = _legacy_public_relation_fit_plan()
    reported_plan = relation_fit_report.get("plan")
    if reported_plan != plan:
        raise RuntimeError("paired action calibration fit identity changed")
    commitments = plan["fit_commitments"]
    assert isinstance(commitments, tuple)
    streams = _scheduled_relation_streams(commitments, _RELATION_FIT_SEED_PAIRS)
    report = _calibrate_public_relation_action_streams(
        controller,
        streams,
        raw_gate_passed=True,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
    )
    expected_arms = int(plan["fit_directional_arms"])
    if report["directional_arms"] != expected_arms:
        raise RuntimeError("paired action calibration exposure count changed")
    return {
        **report,
        "plan": plan,
        "development_or_final_access": False,
        "scalar_judge_calls": 0,
    }


def train_software_pipeline_controller(
    controller: SoftwarePipelineController,
    config: SoftwarePipelineExperimentConfig,
    *,
    judge: Callable[
        [GeneratedSoftwarePipelineTask, CommittedSoftwarePipeline], float
    ] = judge_software_pipeline_attempt,
) -> dict[str, object]:
    """Train one slow lineage from public traces and terminal outcomes."""

    if controller.profile.name != config.profile:
        raise ValueError("controller and experiment profiles differ")
    relation_parameter_names = set(_relation_matcher_parameter_names(controller))
    parameters = tuple(
        parameter
        for name, parameter in controller.named_parameters()
        if parameter.requires_grad and name not in relation_parameter_names
    )
    if not parameters:
        raise RuntimeError("software-pipeline training selected no parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    commitments = software_pipeline_mechanism_partition("train")[
        : config.train_mechanisms
    ]
    ledger = _ScalarLedger(judge)
    losses: list[float] = []
    gradient_norms: list[float] = []
    public_trace_terms = 0
    public_retrieval_terms = 0
    public_transfer_terms = 0
    public_reasoning_terms = 0
    accepted_trace_writes = 0
    accepted_scalar_writes = 0
    started = time.perf_counter()
    controller.train()
    for epoch in range(config.training_epochs):
        for mechanism_index, commitment in enumerate(commitments):
            stream = make_software_pipeline_stream(
                _experiment_seed(config.seed, "train", epoch, mechanism_index),
                supports_per_motif=config.supports_per_motif,
                queries=config.queries_per_mechanism,
                maximum_steps=config.maximum_steps,
                mechanism_commitment=commitment,
                mechanism_partition="train",
            )
            optimizer.zero_grad(set_to_none=True)
            objective_terms = []
            if len(stream.supports) < 2:
                raise RuntimeError("public leave-one-package-out training needs siblings")
            # Rotate every observed package through the held-out production
            # role.  Each fold starts with fresh fast state, writes only the
            # other public packages, and removes the held-out observations
            # before production scoring.  No motif identity or support order
            # is interpreted by the learner.
            for heldout_index, heldout in enumerate(stream.supports):
                state = controller.initial_state()
                for evidence_index, evidence in enumerate(stream.supports):
                    if evidence_index == heldout_index:
                        continue
                    acquisition = acquire_public_pipeline_traces(
                        controller, evidence.learner, state
                    )
                    state = acquisition.state
                    accepted_trace_writes += acquisition.role_writes
                heldout_losses = controller.public_heldout_production_losses(
                    heldout.learner,
                    state,
                    detach_evidence_action_input=True,
                    use_legacy_evidence=True,
                )
                objective_terms.append(heldout_losses.mean())
                transition_count = len(_public_transitions(heldout.learner))
                public_trace_terms += transition_count
                retrieval_count = 2 * transition_count
                public_retrieval_terms += retrieval_count
                public_reasoning_terms += (
                    int(heldout_losses.numel())
                    - transition_count
                    - retrieval_count
                )
                public_transfer_terms += int(heldout_losses.numel())
            if not objective_terms:
                raise RuntimeError("software-pipeline training produced no objective")
            loss = torch.stack(objective_terms).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("software-pipeline training loss is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, config.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("software-pipeline gradient norm is non-finite")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
    calibration = _calibrate_public_evidence_path(
        controller,
        config,
        commitments,
    )
    elapsed = time.perf_counter() - started
    expected_calls = 0
    if ledger.calls != expected_calls:
        raise RuntimeError("training violated one scalar per committed attempt")
    return {
        "epochs": config.training_epochs,
        "mechanisms": config.train_mechanisms,
        "optimizer_steps": len(losses),
        "public_trace_terms": public_trace_terms,
        "public_retrieval_terms": public_retrieval_terms,
        "public_causal_terms": calibration["public_causal_terms"],
        "public_transfer_terms": public_transfer_terms,
        "public_reasoning_terms": public_reasoning_terms,
        "accepted_trace_writes": accepted_trace_writes,
        "accepted_scalar_writes": accepted_scalar_writes,
        "scalar_judge_calls": ledger.calls,
        "expected_scalar_judge_calls": expected_calls,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "elapsed_seconds": elapsed,
        "one_persistent_slow_lineage": True,
        "fresh_fast_state_per_mechanism": True,
        "fresh_fast_state_per_fold": True,
        "main_evidence_action_input_detached": True,
        "main_legacy_evidence_path": True,
        "main_relation_matcher_excluded": True,
        "evidence_calibration": calibration,
        "complete_pipeline_candidates": 0,
    }


def _calibrate_public_evidence_path(
    controller: SoftwarePipelineController,
    config: SoftwarePipelineExperimentConfig,
    commitments: Sequence[str],
) -> dict[str, object]:
    """Run one bounded public-only pass over just the evidence action path.

    This phase cannot update task encoders, transition/reasoning machinery,
    STOP, or the ordinary action path.  A role-memory-OFF probe is checked
    byte-for-byte before and after calibration because the centered evidence
    contribution must remain exactly zero when no role evidence is exposed.
    """

    calibration_names = {
        "evidence_action_log_gate",
        "evidence_action_head.0.weight",
        "evidence_action_head.0.bias",
        "evidence_action_head.2.weight",
    }
    named_parameters = tuple(controller.named_parameters())
    actual_names = {
        name for name, _ in named_parameters if name in calibration_names
    }
    if actual_names != calibration_names:
        raise RuntimeError("evidence calibration parameter boundary changed")
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in named_parameters
    }
    frozen_snapshots = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
        if name not in calibration_names
    }
    for name, parameter in named_parameters:
        parameter.requires_grad_(name in calibration_names)
    calibration_parameters = tuple(
        parameter
        for name, parameter in named_parameters
        if name in calibration_names
    )
    optimizer = torch.optim.AdamW(
        calibration_parameters,
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    probe_stream = make_software_pipeline_stream(
        _experiment_seed(config.seed, "calibration-probe", 0, 0),
        supports_per_motif=config.supports_per_motif,
        queries=1,
        maximum_steps=config.maximum_steps,
        mechanism_commitment=commitments[0],
        mechanism_partition="train",
    )
    probe_heldout = probe_stream.supports[0]
    probe_state = _acquire_support_set(
        controller,
        probe_stream.supports[1:],
    )
    probe_task = replace(probe_heldout.learner, observations=())
    with torch.no_grad():
        no_memory_before = controller.score_actions(
            probe_task,
            probe_state,
            include_pointer_memory=False,
            include_role_memory=False,
        ).action_logits.detach().clone()
    losses: list[float] = []
    gradient_norms: list[float] = []
    causal_terms = 0
    try:
        for mechanism_index, commitment in enumerate(commitments):
            stream = make_software_pipeline_stream(
                _experiment_seed(
                    config.seed,
                    "evidence-calibration",
                    0,
                    mechanism_index,
                ),
                supports_per_motif=config.supports_per_motif,
                queries=config.queries_per_mechanism,
                maximum_steps=config.maximum_steps,
                mechanism_commitment=commitment,
                mechanism_partition="train",
            )
            optimizer.zero_grad(set_to_none=True)
            objective_terms = []
            for heldout_index, heldout in enumerate(stream.supports):
                state = controller.initial_state()
                for evidence_index, evidence in enumerate(stream.supports):
                    if evidence_index != heldout_index:
                        state = acquire_public_pipeline_traces(
                            controller,
                            evidence.learner,
                            state,
                        ).state
                objective_terms.append(
                    controller.public_heldout_production_losses(
                        heldout.learner,
                        state,
                        include_role_memory_causal_hinge=True,
                        use_legacy_evidence=True,
                    ).mean()
                )
                causal_terms += _public_role_causal_target_count(
                    heldout.learner
                )
            if not objective_terms:
                raise RuntimeError("evidence calibration produced no objective")
            loss = torch.stack(objective_terms).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("evidence calibration loss is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                calibration_parameters,
                config.gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("evidence calibration gradient is non-finite")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))
        with torch.no_grad():
            no_memory_after = controller.score_actions(
                probe_task,
                probe_state,
                include_pointer_memory=False,
                include_role_memory=False,
            ).action_logits.detach().clone()
        if not torch.equal(no_memory_before, no_memory_after):
            raise RuntimeError("evidence calibration changed no-memory logits")
        for name, before in frozen_snapshots.items():
            current = dict(named_parameters)[name].detach()
            if not torch.equal(before, current):
                raise RuntimeError(
                    f"evidence calibration changed frozen parameter: {name}"
                )
    finally:
        for name, parameter in named_parameters:
            parameter.requires_grad_(original_requires_grad[name])
    gate = F.softplus(controller.evidence_action_log_gate)
    return {
        "optimizer_steps": len(losses),
        "public_causal_terms": causal_terms,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "trainable_parameter_names": tuple(sorted(calibration_names)),
        "frozen_parameters_unchanged": True,
        "no_memory_logits_exact": True,
        "no_memory_max_delta": 0.0,
        "evidence_action_input_detached": False,
        "evidence_gate": float(gate.detach().item()),
    }


def build_software_pipeline_evaluation_arms(
    stream: SoftwarePipelineStream,
) -> dict[str, SoftwarePipelineStream]:
    if not isinstance(stream, SoftwarePipelineStream):
        raise TypeError("stream must be SoftwarePipelineStream")
    return {
        "correct": make_software_pipeline_control_stream(stream, "correct"),
        "no_evidence": make_software_pipeline_control_stream(
            stream, "no_evidence"
        ),
        "wrong_evidence": make_software_pipeline_control_stream(
            stream, "wrong_evidence"
        ),
        "shuffled_outcome": make_software_pipeline_control_stream(
            stream, "shuffled_outcome"
        ),
        "a_only": make_software_pipeline_control_stream(stream, "a_only"),
        "b_only": make_software_pipeline_control_stream(stream, "b_only"),
    }


def evaluate_software_pipeline_partition(
    controller: SoftwarePipelineController,
    config: SoftwarePipelineExperimentConfig,
    *,
    partition: str,
    mechanism_count: int,
    judge: Callable[
        [GeneratedSoftwarePipelineTask, CommittedSoftwarePipeline], float
    ] = judge_software_pipeline_attempt,
) -> dict[str, object]:
    """Evaluate fixed controls and report their exact evidence-work boundary."""

    if partition not in ("development", "final"):
        raise ValueError("evaluation partition must be development or final")
    if (
        isinstance(mechanism_count, bool)
        or not isinstance(mechanism_count, int)
        or not 1 <= mechanism_count <= 16
    ):
        raise ValueError("mechanism_count must be between one and sixteen")
    if any(parameter.requires_grad for parameter in controller.parameters()):
        raise RuntimeError("evaluation requires frozen slow weights")
    commitments = software_pipeline_mechanism_partition(partition)[:mechanism_count]
    ledger = _ScalarLedger(judge)
    rows = []
    started = time.perf_counter()
    controller.eval()
    with torch.no_grad():
        for mechanism_index, commitment in enumerate(commitments):
            seed = _experiment_seed(config.seed, partition, 0, mechanism_index)
            stream = make_software_pipeline_stream(
                seed,
                supports_per_motif=config.supports_per_motif,
                queries=config.queries_per_mechanism,
                maximum_steps=config.maximum_steps,
                mechanism_commitment=commitment,
                mechanism_partition=partition,
            )
            rerendered = make_software_pipeline_stream(
                seed,
                surface_seed=_experiment_seed(seed, "rerender", 0, 0),
                supports_per_motif=config.supports_per_motif,
                queries=config.queries_per_mechanism,
                maximum_steps=config.maximum_steps,
                mechanism_commitment=commitment,
                mechanism_partition=partition,
            )
            arms = build_software_pipeline_evaluation_arms(stream)
            row: dict[str, object] = {
                "mechanism_commitment": commitment,
                "support_acquisition_counts": {
                    name: {
                        "packages": len(arm_stream.supports),
                        "public_transitions": sum(
                            len(trace.transitions)
                            for pair in arm_stream.supports
                            for trace in pair.learner.observations
                        ),
                    }
                    for name, arm_stream in arms.items()
                },
            }
            for name, arm_stream in arms.items():
                state = _acquire_support_set(
                    controller, arm_stream.supports
                )
                row[name] = _judge_query_set(
                    controller,
                    arm_stream.queries,
                    state,
                    ledger,
                )
            correct_state = _acquire_support_set(controller, arms["correct"].supports)
            row["pointer_only"] = _judge_query_set(
                controller,
                arms["correct"].queries,
                correct_state,
                ledger,
                include_role_memory=False,
            )
            row["role_memory_removed"] = row["pointer_only"]
            row["backward_reasoning_removed"] = _judge_query_set(
                controller,
                arms["correct"].queries,
                correct_state,
                ledger,
                include_backward_reasoning=False,
            )
            row["episodic_retrieval"] = _judge_episodic_query_set(
                arms["correct"].supports,
                arms["correct"].queries,
                ledger,
            )
            swapped_state = _acquire_support_set(controller, rerendered.supports)
            row["state_swap"] = _judge_query_set(
                controller,
                arms["correct"].queries,
                swapped_state,
                ledger,
            )
            row["alpha_rerender"] = _judge_query_set(
                controller,
                rerendered.queries,
                swapped_state,
                ledger,
            )
            row["correct_over_no_evidence"] = row["correct"] - row["no_evidence"]
            row["correct_over_wrong_evidence"] = (
                row["correct"] - row["wrong_evidence"]
            )
            row["composition_over_best_single"] = row["correct"] - max(
                row["a_only"], row["b_only"]
            )
            row["role_memory_contribution"] = (
                row["correct"] - row["role_memory_removed"]
            )
            row["backward_reasoning_contribution"] = (
                row["correct"] - row["backward_reasoning_removed"]
            )
            rows.append(row)
    elapsed = time.perf_counter() - started
    # Role-memory removal is exactly the pointer-only arm; every other score,
    # including the nonparametric episodic control, executes one fresh commit.
    calls_per_query = 11
    expected_calls = mechanism_count * config.queries_per_mechanism * calls_per_query
    if ledger.calls != expected_calls:
        raise RuntimeError("evaluation violated exact scalar judge accounting")
    return {
        "partition": partition,
        "mechanisms": mechanism_count,
        "rows": rows,
        "summary": _evaluation_summary(rows),
        "scalar_judge_calls": ledger.calls,
        "expected_scalar_judge_calls": expected_calls,
        "arm_matching": {
            "same_slow_weights": True,
            "same_query_packages": True,
            "same_query_rollout_compute": True,
            "same_evidence_work": False,
            "evidence_work_note": (
                "motif-pure A/B controls intentionally acquire only their "
                "declared support family; exact package/transition counts are "
                "reported per mechanism"
            ),
        },
        "elapsed_seconds": elapsed,
        "complete_pipeline_candidates": 0,
    }


def _acquire_support_set(
    controller: SoftwarePipelineController,
    pairs: Sequence[GeneratedSoftwarePipelineTask],
) -> SoftwareReconstructionState:
    state = controller.initial_state()
    for pair in pairs:
        state = acquire_public_pipeline_traces(
            controller, pair.learner, state
        ).state
    return state


def _judge_episodic_query_set(
    supports: Sequence[GeneratedSoftwarePipelineTask],
    pairs: Sequence[GeneratedSoftwarePipelineTask],
    ledger: _ScalarLedger,
) -> float:
    """Evaluate one isolated exact-episode retrieval control per query.

    This nonparametric baseline is deliberately outside the controller and all
    learner state.  It remembers directed-WL keys of publicly observed support
    actions, follows only the public type chain, and falls back to an opaque
    deterministic tie when no exact episode key transfers.
    """

    rewards = []
    for pair in pairs:
        pipeline = _commit_episodic_retrieval(supports, pair.learner)
        rewards.append(ledger(pair, pipeline))
    return sum(rewards) / len(rewards)


def _commit_episodic_retrieval(
    supports: Sequence[GeneratedSoftwarePipelineTask],
    task: PublicSoftwarePipelineTask,
) -> CommittedSoftwarePipeline:
    observed_keys: set[tuple[object, ...]] = set()
    for pair in supports:
        public = pair.learner
        contracts = {
            candidate.schema.digest: component
            for candidate in public.grounded_candidates
            for component in public.components
            if component.schema == candidate.schema
        }
        for trace in public.observations:
            for transition in trace.transitions:
                observed_keys.add(
                    _episodic_topology_key(
                        contracts[transition.action.schema.digest]
                    )
                )

    contracts = {
        action: component
        for action in task.grounded_candidates
        for component in task.components
        if component.schema == action.schema
    }
    output_types = {component.output_type for component in contracts.values()}
    first_actions = sorted(
        (
            action
            for action, component in contracts.items()
            if component.input_type not in output_types
        ),
        key=lambda action: action.digest,
    )
    actions: list[GroundAction] = []
    for first_action in first_actions:
        first = contracts[first_action]
        alternatives = sorted(
            (
                action
                for action, component in contracts.items()
                if component.input_type == first.output_type
            ),
            key=lambda action: action.digest,
        )
        if not alternatives:
            continue
        exact = tuple(
            action
            for action in alternatives
            if _episodic_topology_key(contracts[action]) in observed_keys
        )
        actions.extend((first_action, (exact or tuple(alternatives))[0]))
    return commit_software_pipeline(
        task,
        actions,
        stopped=len(actions) < task.max_steps,
    )


def _episodic_topology_key(
    component: PublicComponentContract,
) -> tuple[object, ...]:
    edges = _relational_edges(component)
    nodes = sorted({value for edge in edges for value in edge})
    labels: dict[str, object] = {
        node: (
            sum(target == node for _, target in edges),
            sum(source == node for source, _ in edges),
        )
        for node in nodes
    }
    for _ in nodes:
        rows = {
            node: (
                labels[node],
                tuple(
                    sorted(
                        labels[source]
                        for source, target in edges
                        if target == node
                    )
                ),
                tuple(
                    sorted(
                        labels[target]
                        for source, target in edges
                        if source == node
                    )
                ),
            )
            for node in nodes
        }
        vocabulary = {
            value: index
            for index, value in enumerate(sorted(set(rows.values()), key=repr))
        }
        labels = {node: vocabulary[value] for node, value in rows.items()}
    return (
        tuple(sorted(labels.values())),
        tuple(sorted((labels[source], labels[target]) for source, target in edges)),
    )


def _judge_query_set(
    controller: SoftwarePipelineController,
    pairs: Sequence[GeneratedSoftwarePipelineTask],
    state: SoftwareReconstructionState,
    ledger: _ScalarLedger,
    *,
    include_role_memory: bool = True,
    include_backward_reasoning: bool = True,
) -> float:
    snapshot = snapshot_software_reconstruction_state(state)
    rewards = []
    for pair in pairs:
        query_state = restore_software_reconstruction_state(snapshot)
        rollout = rollout_software_pipeline(
            controller,
            pair.learner,
            query_state,
            include_role_memory=include_role_memory,
            include_backward_reasoning=include_backward_reasoning,
        )
        rewards.append(ledger(pair, rollout.pipeline))
    return sum(rewards) / len(rewards)


def _evaluation_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    numeric_names = tuple(
        name
        for name, value in rows[0].items()
        if name != "mechanism_commitment" and isinstance(value, (int, float))
    )
    return {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in numeric_names
    }


def _experiment_seed(seed: int, scope: str, first: int, second: int) -> int:
    digest = hashlib.sha256(
        f"{seed}|{scope}|{first}|{second}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _trajectory_log_probability(rollout: SoftwareNeuralRollout) -> torch.Tensor:
    return torch.stack(
        [
            F.log_softmax(logits, dim=-1)[index]
            for logits, index in zip(
                rollout.step_logits, rollout.selected_indices, strict=True
            )
        ]
    ).mean()


def _read_exact_pointer_contexts(
    query_ids: torch.Tensor,
    state: GlyphAssociativeState,
    trace_slots: int,
) -> torch.Tensor:
    if query_ids.ndim != 2 or query_ids.shape[1] != _POINTER_WORDS:
        raise ValueError("pointer query ids must be [count, words]")
    if query_ids.device != state.keys.device or query_ids.dtype != torch.long:
        raise ValueError("pointer query ids must match state device and use long")
    occupied = state.occupied[0, :trace_slots]
    stored = state.public_source_action_ids[0, :trace_slots]
    matches = (query_ids[:, None, :] == stored[None, :, :]).all(dim=-1)
    matches = matches & occupied.unsqueeze(0)
    counts = state.write_counts[0, :trace_slots].to(dtype=state.keys.dtype)
    weights = matches.to(dtype=state.keys.dtype) * counts.unsqueeze(0)
    totals = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(
        totals > 0.0,
        weights / totals.clamp_min(torch.finfo(weights.dtype).tiny),
        torch.zeros_like(weights),
    )
    return weights @ state.values[0, :trace_slots]


def _normalized_relation_pool(
    relations: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if relations.ndim != 2 or relations.shape != (values.shape[0], values.shape[0]):
        raise ValueError("relation matrix must match value rows")
    totals = relations.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (relations @ values) / totals


def _bounded_role_anchor(anchor: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if anchor.shape != residual.shape or not anchor.is_floating_point():
        raise ValueError("role anchor and residual must share one floating shape")
    epsilon = torch.finfo(anchor.dtype).eps
    unit = anchor / anchor.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    orthogonal = residual - (residual * unit).sum(dim=-1, keepdim=True) * unit
    norm = orthogonal.norm(dim=-1, keepdim=True)
    limit = norm.new_tensor(_ROLE_RESIDUAL_LIMIT)
    bounded = orthogonal * (
        limit * torch.tanh(norm / limit) / norm.clamp_min(epsilon)
    )
    return anchor + bounded


def _components_in_candidate_order(
    task: PublicSoftwarePipelineTask,
) -> tuple[PublicComponentContract, ...]:
    ordered = []
    for candidate in task.grounded_candidates:
        matches = tuple(
            component for component in task.components if component.schema == candidate.schema
        )
        if len(matches) != 1:
            raise ValueError("every grounded candidate must have one public component")
        ordered.append(matches[0])
    return tuple(ordered)


def _public_effect_equivalence_key(
    component: PublicComponentContract,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Return only the declared public transition effect, never identity."""

    return (
        component.input_type,
        component.output_type,
        component.state_reads,
        component.state_writes,
    )


def _public_role_causal_target_count(
    task: PublicSoftwarePipelineTask,
) -> int:
    """Count visible non-STOP targets that have a public effect twin."""

    components = _components_in_candidate_order(task)
    class_sizes: dict[
        tuple[str, str, tuple[str, ...], tuple[str, ...]], int
    ] = {}
    for component in components:
        key = _public_effect_equivalence_key(component)
        class_sizes[key] = class_sizes.get(key, 0) + 1
    return sum(
        class_sizes[
            _public_effect_equivalence_key(
                components[
                    _action_index(task.grounded_candidates, transition.action)
                ]
            )
        ]
        >= 2
        for transition in _public_transitions(task)
    )


def diagnose_public_lopo_evidence_margins(
    controller: SoftwarePipelineController,
    supports: Sequence[PublicSoftwarePipelineTask],
) -> dict[str, object]:
    """Measure raw leave-one-package-out evidence margins without learning.

    Every fast state is constructed afresh from public observations.  For
    each held-out transition, the report compares its target against every
    declared action and, when present, only its exact public-effect siblings.
    No package identity, semantic grouping, or evaluator-private value is
    read or returned.
    """

    if isinstance(supports, (str, bytes, bytearray)) or not isinstance(
        supports, Sequence
    ):
        raise TypeError("LOPO supports must be a finite task sequence")
    public_tasks = tuple(supports)
    if len(public_tasks) < 2:
        raise ValueError("LOPO evidence diagnosis needs at least two supports")
    if any(not isinstance(task, PublicSoftwarePipelineTask) for task in public_tasks):
        raise TypeError("LOPO evidence diagnosis accepts only public tasks")
    if any(not _public_transitions(task) for task in public_tasks):
        raise ValueError("every LOPO support must expose a public trace")

    rows: list[dict[str, object]] = []

    def acquire_tasks(
        tasks: Sequence[PublicSoftwarePipelineTask],
    ) -> SoftwareReconstructionState:
        state = controller.initial_state()
        for task in tasks:
            state = acquire_public_pipeline_traces(
                controller,
                task,
                state,
            ).state
        return state

    def raw_margins(
        task: PublicSoftwarePipelineTask,
        transition: Transition,
        state: SoftwareReconstructionState,
        *,
        position: int,
        transition_count: int,
    ) -> dict[str, float | int | None]:
        masked = replace(task, observations=())
        encoding = controller.encode_task(masked)
        before_index = _state_index(masked.states, transition.before)
        target_index = _action_index(
            masked.grounded_candidates,
            transition.action,
        )
        belief = F.one_hot(
            torch.tensor(
                before_index,
                device=encoding.role_state_embeddings.device,
            ),
            len(masked.states),
        ).to(dtype=encoding.role_state_embeddings.dtype)
        scores = controller.score_actions(
            masked,
            state,
            current_state_belief=belief,
            steps_remaining=transition_count - position,
            encoding=encoding,
            include_pointer_memory=False,
        )
        evidence = scores.evidence_match_scores
        negative_mask = torch.ones_like(evidence, dtype=torch.bool)
        negative_mask[target_index] = False
        target_vs_hardest = evidence[target_index] - evidence[negative_mask].max()
        components = _components_in_candidate_order(masked)
        target_effect = _public_effect_equivalence_key(components[target_index])
        sibling_indices = tuple(
            index
            for index, component in enumerate(components)
            if index != target_index
            and _public_effect_equivalence_key(component) == target_effect
        )
        target_vs_class = None
        if sibling_indices:
            sibling_tensor = torch.tensor(
                sibling_indices,
                device=evidence.device,
                dtype=torch.long,
            )
            target_vs_class = evidence[target_index] - evidence.index_select(
                0,
                sibling_tensor,
            ).max()
        return {
            "target_index": target_index,
            "effect_class_size": len(sibling_indices) + 1,
            "target_vs_hardest": float(target_vs_hardest.detach().item()),
            "target_vs_class_sibling": (
                None
                if target_vs_class is None
                else float(target_vs_class.detach().item())
            ),
        }

    with torch.no_grad():
        for heldout_index, heldout in enumerate(public_tasks):
            evidence_tasks = tuple(
                task
                for index, task in enumerate(public_tasks)
                if index != heldout_index
            )
            individual_states = tuple(
                acquire_tasks((task,))
                for task in evidence_tasks
            )
            all_state = acquire_tasks(evidence_tasks)
            transitions = _public_transitions(heldout)
            for position, transition in enumerate(transitions):
                individual = [
                    {
                        "evidence_support_index": evidence_index,
                        **raw_margins(
                            heldout,
                            transition,
                            state,
                            position=position,
                            transition_count=len(transitions),
                        ),
                    }
                    for evidence_index, state in enumerate(individual_states)
                ]
                combined = raw_margins(
                    heldout,
                    transition,
                    all_state,
                    position=position,
                    transition_count=len(transitions),
                )
                metric_rows = {}
                for name in (
                    "target_vs_hardest",
                    "target_vs_class_sibling",
                ):
                    singles = [
                        float(value[name])
                        for value in individual
                        if value[name] is not None
                    ]
                    all_value = combined[name]
                    if not singles or all_value is None:
                        metric_rows[name] = None
                        continue
                    best_single = max(singles)
                    metric_rows[name] = {
                        "best_single": best_single,
                        "all_minus_best_single": float(all_value) - best_single,
                        "all_vs_best_single_retention": (
                            float(all_value) / best_single
                            if best_single > 0.0
                            else None
                        ),
                    }
                rows.append(
                    {
                        "heldout_support_index": heldout_index,
                        "transition_index": position,
                        "effect_class_size": combined["effect_class_size"],
                        "individual": individual,
                        "all_others": combined,
                        "all_vs_best_single": metric_rows,
                    }
                )

    def summarize_metric(name: str) -> dict[str, float | int | None]:
        individual_values = [
            float(value[name])
            for row in rows
            for value in row["individual"]
            if value[name] is not None
        ]
        all_values = [
            float(row["all_others"][name])
            for row in rows
            if row["all_others"][name] is not None
        ]
        retention_rows = [
            row["all_vs_best_single"][name]
            for row in rows
            if row["all_vs_best_single"][name] is not None
        ]
        retention_values = [
            float(value["all_vs_best_single_retention"])
            for value in retention_rows
            if value["all_vs_best_single_retention"] is not None
        ]
        deltas = [
            float(value["all_minus_best_single"])
            for value in retention_rows
        ]

        def mean(values: Sequence[float]) -> float | None:
            return sum(values) / len(values) if values else None

        def positive_fraction(values: Sequence[float]) -> float | None:
            return (
                sum(value > 0.0 for value in values) / len(values)
                if values
                else None
            )

        return {
            "individual_count": len(individual_values),
            "individual_mean": mean(individual_values),
            "individual_positive_fraction": positive_fraction(individual_values),
            "all_count": len(all_values),
            "all_mean": mean(all_values),
            "all_positive_fraction": positive_fraction(all_values),
            "all_minus_best_single_mean": mean(deltas),
            "all_vs_best_single_retention_mean": mean(retention_values),
        }

    return {
        "support_packages": len(public_tasks),
        "heldout_transitions": len(rows),
        "public_observations_only": True,
        "fresh_fast_state_per_comparison": True,
        "rows": rows,
        "summary": {
            "target_vs_hardest": summarize_metric("target_vs_hardest"),
            "effect_equivalent_target_vs_sibling": summarize_metric(
                "target_vs_class_sibling"
            ),
        },
    }


def _state_role_features(
    task: PublicSoftwarePipelineTask,
) -> tuple[list[list[float]], list[list[float]]]:
    goal_index = _goal_state_index(task.states, task.required_output)
    origin_records = set(task.origin.records)
    goal_records = set(task.states[goal_index].records)
    state_sets = [set(state.records) for state in task.states]
    features = []
    relations = []
    for index, state in enumerate(task.states):
        records = state_sets[index]
        arities = [len(record.arguments) for record in state.records]
        arguments = [argument for record in state.records for argument in record.arguments]
        predicates = [record.predicate for record in state.records]
        shared_origin = len(records & origin_records)
        shared_goal = len(records & goal_records)
        row = [
            float(state == task.origin),
            float(index == goal_index),
            len(records) / 8.0,
            len(predicates) / 8.0,
            len(set(predicates)) / 8.0,
            len(arguments) / 16.0,
            len(set(arguments)) / 16.0,
            shared_origin / 8.0,
            shared_goal / 8.0,
            len(records - origin_records) / 8.0,
            len(origin_records - records) / 8.0,
            len(records - goal_records) / 8.0,
            len(goal_records - records) / 8.0,
            *[arities.count(arity) / 8.0 for arity in range(5)],
            sum(left == right for left in arguments for right in arguments) / 64.0,
            sum(bool(records & other) for other in state_sets) / 8.0,
            sum(len(records & other) for other in state_sets) / 32.0,
            float(task.required_output.exact),
            len(task.states) / 8.0,
            len(task.grounded_candidates) / 8.0,
        ]
        if len(row) != RenameInvariantRoleEncoder._STATE_FEATURES:
            raise RuntimeError("state role feature width changed")
        features.append(row)
        relation_row = []
        for other in state_sets:
            union = len(records | other)
            relation_row.append(1.0 if not union else len(records & other) / union)
        relations.append(relation_row)
    return features, relations


def _component_role_features(
    components: Sequence[PublicComponentContract],
    candidates: Sequence[GroundAction],
) -> tuple[list[list[float]], list[list[float]]]:
    relations = [
        [float(left.output_type == right.input_type) for right in components]
        for left in components
    ]
    features = []
    for index, (component, candidate) in enumerate(
        zip(components, candidates, strict=True)
    ):
        reads = set(component.state_reads)
        writes = set(component.state_writes)
        arities = [len(record.arguments) for record in component.incidence]
        predecessors = sum(other.output_type == component.input_type for other in components)
        successors = sum(component.output_type == other.input_type for other in components)
        peer_errors = sum(component.error_type == other.error_type for other in components)
        argument_equalities = sum(
            left == right for left in candidate.arguments for right in candidate.arguments
        )
        row = [
            len(component.schema.parameters) / 6.0,
            len(candidate.arguments) / 6.0,
            len(set(candidate.arguments)) / 6.0,
            float(component.input_type == component.output_type),
            float(component.input_type == component.error_type),
            float(component.output_type == component.error_type),
            len(reads) / 8.0,
            len(writes) / 8.0,
            len(reads & writes) / 8.0,
            len(component.incidence) / 8.0,
            sum(arities) / 16.0,
            len(set(record.predicate for record in component.incidence)) / 8.0,
            predecessors / 8.0,
            successors / 8.0,
            peer_errors / 8.0,
            sum(bool(reads & set(other.state_writes)) for other in components) / 8.0,
            sum(bool(writes & set(other.state_reads)) for other in components) / 8.0,
            argument_equalities / 36.0,
            *[arities.count(arity) / 8.0 for arity in range(5)],
            len(components) / 8.0,
        ]
        if len(row) != RenameInvariantRoleEncoder._COMPONENT_FEATURES:
            raise RuntimeError("component role feature width changed")
        features.append(row)
    return features, relations


def _state_arguments(state: State) -> set[str]:
    return {
        argument
        for record in state.records
        for argument in record.arguments
    }


def _local_state_component_features(
    states: Sequence[State],
    components: Sequence[PublicComponentContract],
) -> list[list[list[float]]]:
    """Describe applicability through within-package equality only."""

    rows = []
    for state in states:
        present = _state_arguments(state)
        state_rows = []
        for component in components:
            reads = set(component.state_reads)
            writes = set(component.state_writes)
            interface = reads | writes
            row = [
                len(reads) / 8.0,
                len(writes) / 8.0,
                len(present & reads) / 8.0,
                len(reads - present) / 8.0,
                len(present & writes) / 8.0,
                len(writes - present) / 8.0,
                len(reads & writes) / 8.0,
                float(reads <= present),
                float(writes <= present),
                float(bool(present & reads)),
                float(bool(present & writes)),
                len(interface & present) / 16.0,
                len(interface - present) / 16.0,
                float(not reads),
                float(not writes),
                float(reads == writes),
            ]
            if len(row) != RenameInvariantRoleEncoder._STATE_COMPONENT_FEATURES:
                raise RuntimeError("local state-component feature width changed")
            state_rows.append(row)
        rows.append(state_rows)
    return rows


def _relative_effect_candidate_features(
    states: Sequence[State],
    components: Sequence[PublicComponentContract],
) -> list[list[list[list[float]]]]:
    """Describe every public before/action/after candidate as a local delta."""

    arguments = [_state_arguments(state) for state in states]
    rows = []
    for before in arguments:
        action_rows = []
        for component in components:
            reads = set(component.state_reads)
            writes = set(component.state_writes)
            interface = reads | writes
            candidates = []
            for after in arguments:
                added = after - before
                removed = before - after
                retained = before & after
                union = before | after
                row = [
                    len(added) / 16.0,
                    len(removed) / 16.0,
                    len(retained) / 16.0,
                    len(before) / 16.0,
                    len(after) / 16.0,
                    (len(retained) / len(union)) if union else 1.0,
                    float(before == after),
                    float(before <= after),
                    float(after <= before),
                    len(added & writes) / 8.0,
                    len(writes - added) / 8.0,
                    len(removed & writes) / 8.0,
                    len(after & writes) / 8.0,
                    len(before & writes) / 8.0,
                    float(writes <= after),
                    float(writes <= before),
                    len(removed & reads) / 8.0,
                    len(after & reads) / 8.0,
                    len(before & reads) / 8.0,
                    float(reads <= before),
                    float(reads <= after),
                    len(added & reads) / 8.0,
                    len((added | removed) & interface) / 16.0,
                    len(interface) / 16.0,
                ]
                if len(row) != RenameInvariantRoleEncoder._RELATIVE_EFFECT_FEATURES:
                    raise RuntimeError("relative effect feature width changed")
                candidates.append(row)
            action_rows.append(candidates)
        rows.append(action_rows)
    return rows


def _local_state_goal_features(
    states: Sequence[State],
    goal: Goal,
) -> list[list[float]]:
    """Describe current-to-goal progress without a state identity embedding."""

    required = {
        argument
        for record in goal.required
        for argument in record.arguments
    }
    rows = []
    for state in states:
        present = _state_arguments(state)
        shared = present & required
        union = present | required
        row = [
            len(shared) / 16.0,
            len(present - required) / 16.0,
            len(required - present) / 16.0,
            len(union) / 16.0,
            (len(shared) / len(union)) if union else 1.0,
            float(present == required),
            float(present <= required),
            float(required <= present),
            len(present) / 16.0,
            len(required) / 16.0,
            float(goal.exact),
            (len(shared) / len(required)) if required else 1.0,
        ]
        if len(row) != RenameInvariantRoleEncoder._STOP_RELATION_FEATURES:
            raise RuntimeError("local state-goal feature width changed")
        rows.append(row)
    return rows


def _incidence_graph(
    component: PublicComponentContract,
) -> tuple[list[list[float]], list[list[float]]]:
    """Project anonymous incidence records to a directed relational graph.

    Identifier spelling is used only to preserve equality and adjacency inside
    this one public contract.  Node order is immaterial: all learned operations
    are equivariant and the final mean/max readout is invariant.
    """

    edges = _relational_edges(component)
    if not edges:
        raise ValueError("component incidence requires relational edges")
    node_names = sorted({value for edge in edges for value in edge})
    return [[1.0] for _ in node_names], _adjacency_for_nodes(edges, node_names)


def _shared_incidence_graphs(
    predecessor: PublicComponentContract,
    candidate: PublicComponentContract,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Align two public graphs only through anonymous within-package equality."""

    predecessor_edges = _relational_edges(predecessor)
    candidate_edges = _relational_edges(candidate)
    node_names = sorted(
        {
            value
            for edge in (*predecessor_edges, *candidate_edges)
            for value in edge
        }
    )
    if not node_names:
        raise ValueError("component pair requires relational incidence")
    features = [[1.0] for _ in node_names]
    return (
        features,
        _adjacency_for_nodes(predecessor_edges, node_names),
        _adjacency_for_nodes(candidate_edges, node_names),
    )


def _relational_edges(
    component: PublicComponentContract,
) -> list[tuple[str, str]]:
    return [
        (record.arguments[1], record.arguments[2])
        for record in component.incidence
        if record.predicate.endswith(".relates")
    ]


def _adjacency_for_nodes(
    edges: Sequence[tuple[str, str]],
    node_names: Sequence[str],
) -> list[list[float]]:
    node_indices = {value: index for index, value in enumerate(node_names)}
    adjacency = [[0.0 for _ in node_names] for _ in node_names]
    for source, target in edges:
        adjacency[node_indices[source]][node_indices[target]] = 1.0
    return adjacency


def _pointer_state_text(state: State) -> str:
    return json.dumps(
        {
            "namespace": state.namespace,
            "records": [
                [record.predicate, list(record.arguments)] for record in state.records
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _pointer_component_text(
    component: PublicComponentContract,
    candidate: GroundAction,
) -> str:
    return json.dumps(
        {
            "schema": component.schema.name,
            "arguments": list(candidate.arguments),
            "input_type": component.input_type,
            "output_type": component.output_type,
            "error_type": component.error_type,
            "state_reads": list(component.state_reads),
            "state_writes": list(component.state_writes),
            "incidence": [
                [record.predicate, list(record.arguments)]
                for record in component.incidence
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _pointer_pair_text(
    state: State,
    component: PublicComponentContract,
    candidate: GroundAction,
) -> str:
    return "state\x00" + _pointer_state_text(state) + "\x00component\x00" + _pointer_component_text(
        component, candidate
    )


def _pointer_words(text: str) -> tuple[int, ...]:
    raw = hashlib.sha256(_POINTER_ID_DOMAIN + text.encode("utf-8")).digest()
    maximum = (1 << 63) - 1
    return tuple(
        int.from_bytes(raw[offset : offset + 8], "big") & maximum
        for offset in range(0, 8 * _POINTER_WORDS, 8)
    )


def _public_task_binding(task: PublicSoftwarePipelineTask) -> str:
    material = json.dumps(
        task.to_canonical(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(_TASK_BINDING_DOMAIN + material).hexdigest()


def _public_transitions(task: PublicSoftwarePipelineTask) -> tuple[Transition, ...]:
    result = []
    seen = set()
    for observation in task.observations:
        for transition in observation.transitions:
            if transition not in seen:
                result.append(transition)
                seen.add(transition)
    return tuple(result)


def _state_index(states: Sequence[State], target: State) -> int:
    matches = [index for index, state in enumerate(states) if state == target]
    if len(matches) != 1:
        raise ValueError("state must appear exactly once in the public state set")
    return matches[0]


def _goal_state_index(states: Sequence[State], goal: Goal) -> int:
    matches = []
    required = set(goal.required)
    forbidden = set(goal.forbidden)
    for index, state in enumerate(states):
        records = set(state.records)
        satisfied = records == required if goal.exact else (
            required <= records and not (forbidden & records)
        )
        if satisfied:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError("required output must identify one declared public state")
    return matches[0]


def _action_index(actions: Sequence[GroundAction], target: GroundAction) -> int:
    matches = [index for index, action in enumerate(actions) if action == target]
    if len(matches) != 1:
        raise ValueError("action must appear exactly once among grounded candidates")
    return matches[0]


def _validate_public_task(task: PublicSoftwarePipelineTask) -> None:
    if not isinstance(task, PublicSoftwarePipelineTask):
        raise TypeError("task must be PublicSoftwarePipelineTask")
    if not 1 <= task.max_steps <= _MAX_STEPS:
        raise ValueError("public task max_steps must be one through four")
    _components_in_candidate_order(task)
    _state_index(task.states, task.origin)
    _goal_state_index(task.states, task.required_output)


def _validate_controller_state(
    controller: SoftwarePipelineController,
    state: SoftwareReconstructionState,
) -> None:
    if not isinstance(controller, SoftwarePipelineController):
        raise TypeError("controller must be SoftwarePipelineController")
    if not isinstance(state, SoftwareReconstructionState):
        raise TypeError("state must be SoftwareReconstructionState")
    controller.pointer_memory._validate_state(state.pointer)
    controller.role_memory._validate_state(state.role)
    for name, value in (
        ("context trace state", state.context_trace_keys),
        ("relation trace values", state.relation_trace_values),
    ):
        if (
            value.shape != state.role.keys.shape
            or value.device != state.role.keys.device
            or value.dtype != state.role.keys.dtype
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(f"{name} does not match the role lane")
    trace_slots = controller.role_memory.trace_slot_count
    allowed = state.role.occupied.clone()
    allowed[:, trace_slots:] = False
    for name, value in (
        ("context trace state", state.context_trace_keys),
        ("relation trace values", state.relation_trace_values),
    ):
        invalid = value.masked_select(
            (~allowed).unsqueeze(-1).expand_as(value)
        )
        if bool((invalid != 0.0).any().item()):
            raise ValueError(f"{name} is outside occupied trace slots")


def _validate_state_belief(belief: torch.Tensor, state_count: int) -> None:
    if (
        belief.shape != (state_count,)
        or not bool(torch.isfinite(belief).all().item())
        or bool((belief < 0.0).any().item())
        or not bool(
            torch.isclose(
                belief.sum(), belief.new_tensor(1.0), atol=1.0e-5, rtol=1.0e-5
            ).item()
        )
    ):
        raise ValueError("current state belief must be finite probabilities")


def _validate_reward(reward: float) -> float:
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError("reward must be numeric")
    numeric = float(reward)
    if numeric not in (0.0, 1.0):
        raise ValueError("software-pipeline reward must be terminal 0.0 or 1.0")
    return numeric


def _public_trace_lanes_equal(
    left: SoftwareReconstructionState,
    right: SoftwareReconstructionState,
) -> bool:
    for left_lane, right_lane, slots in (
        (left.pointer, right.pointer, left.pointer.slot_count // 2),
        (left.role, right.role, left.role.slot_count // 2),
    ):
        for left_value, right_value in (
            (left_lane.keys[:, :slots], right_lane.keys[:, :slots]),
            (left_lane.values[:, :slots], right_lane.values[:, :slots]),
            (left_lane.occupied[:, :slots], right_lane.occupied[:, :slots]),
            (left_lane.write_counts[:, :slots], right_lane.write_counts[:, :slots]),
            (left_lane.trace_cursor, right_lane.trace_cursor),
        ):
            if not torch.equal(left_value, right_value):
                return False
    trace_slots = left.role.slot_count // 2
    for left_value, right_value in (
        (
            left.context_trace_keys[:, :trace_slots],
            right.context_trace_keys[:, :trace_slots],
        ),
        (
            left.relation_trace_values[:, :trace_slots],
            right.relation_trace_values[:, :trace_slots],
        ),
    ):
        if not torch.equal(left_value, right_value):
            return False
    return True


def _empty_reference_state() -> GlyphAssociativeState:
    return GlyphAssociativeMemory(2, slots=2, read_top_k=1).initial_state()


__all__ = [
    "AnonymousAllActiveRelationComposer",
    "AnonymousConflictMixer",
    "AnonymousRelationCell",
    "CapacityMatchedClusterController",
    "CapacityMatchedMonolithController",
    "SOFTWARE_PIPELINE_PROFILES",
    "SoftwareNeuralRollout",
    "SoftwarePipelineController",
    "SoftwarePipelineExperimentConfig",
    "SoftwarePipelineRunProfile",
    "PublicRelationCreditRow",
    "SoftwareReconstructionState",
    "SoftwareScalarFeedback",
    "SoftwareStepScores",
    "SoftwareTaskEncoding",
    "SoftwareTraceAcquisition",
    "acquire_public_pipeline_traces",
    "anonymous_conflict_mixer_digest",
    "apply_scalar_pipeline_feedback",
    "build_public_relation_credit_controller",
    "build_public_relation_conflict_system",
    "build_capacity_matched_relation_cluster_pair",
    "build_software_pipeline_controller",
    "build_software_pipeline_evaluation_arms",
    "centered_pipeline_preference_loss",
    "default_software_pipeline_experiment_config",
    "evaluate_software_pipeline_partition",
    "evaluate_public_relation_credit_panel",
    "fit_public_relation_matcher",
    "fit_public_relation_conflict_matcher",
    "fit_capacity_matched_relation_cluster_pilot",
    "capacity_matched_relation_cluster_fit_plan",
    "capacity_matched_relation_cluster_parameter_report",
    "capacity_matched_relation_cluster_plan_digest",
    "capacity_matched_relation_cluster_system_digest",
    "load_capacity_matched_relation_cluster_checkpoint",
    "load_public_relation_conflict_checkpoint",
    "load_software_pipeline_checkpoint",
    "rollout_software_pipeline",
    "public_relation_credit_rows",
    "public_relation_fit_plan",
    "public_relation_conflict_fit_plan",
    "public_relation_conflict_parameter_report",
    "public_relation_conflict_system_digest",
    "save_public_relation_conflict_checkpoint",
    "save_capacity_matched_relation_cluster_checkpoint",
    "save_software_pipeline_checkpoint",
    "scalar_pipeline_outcome_loss",
    "snapshot_software_reconstruction_state",
    "software_pipeline_parameter_report",
    "software_pipeline_model_digest",
    "software_reconstruction_state_digest",
    "train_software_pipeline_controller",
]
